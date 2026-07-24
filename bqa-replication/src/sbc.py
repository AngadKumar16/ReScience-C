"""Simulation-based calibration (SBC) for the BQA grid-inference procedure.

SBC checks that the inference procedure (src.bqa) recovers correctly
calibrated posteriors when the true generative process matches the model's
own likelihood exactly. This requires drawing synthetic data from
`src.simulate_q.simulate_from_q_model` (the model's own Eq. 8 gamma
likelihood) -- NOT from `src.simulate.simulate_responses` (the Gaussian
figure-reproduction simulator). See the TWO-SIMULATOR RULE docstrings in
both of those modules. Using the wrong simulator here would measure model
misspecification (Gaussian vs. gamma), not inference correctness, and would
invalidate every result this module produces.

Priors sampled here are the SETTLED priors from Bhumbra & Beato (2013):
  - p: arcsine/Jeffreys prior, 0.04 <= p <= 0.96 (uniform in
    arcsin(sqrt(p)); see src.grid).
  - log(v): uniform, 0.05 <= v <= 1.
  - n: UNIFORM over integers (2013 paper; do NOT use the later 2018/2019
    Jeffreys' prior on n).

SBC-ONLY ADDED CONVENTION (not from the paper -- read before using):
  The paper's model conditions on the condition means mu_k as FIXED KNOWN
  INPUTS computed from real recordings; mu_k (and hence the quantal size q,
  since mu_k = n*p_k*q) is never given a prior in the paper, because the
  paper never needs to *generate* mu_k -- it only ever observes it. SBC,
  however, requires a fully specified generative process: to simulate data
  we must pick a ground-truth q (equivalently lambda, given gamma = 1/v^2)
  and thus mu_k = n*p_k*q per condition. There is no paper-derived prior
  for this, so `sample_q_prior` below introduces a clearly-labeled,
  log-uniform SBC-only prior over q in a physiologically plausible pA
  range. This is a scaffolding necessity for calibration testing, not a
  claim about the paper's model or a reproduction of its numeric results.
  Baseline noise variance eps^2 is likewise fixed to the paper's confirmed
  Fig. 1B / simulation constant (25 pA SD, see NOTES.md) for the same
  reason -- SBC needs a concrete value, and the paper does not model eps^2
  as inferred either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats

from src.bqa import ConditionData, change_of_variables_and_marginalise
from src.grid import (
    P_MAX,
    P_MIN,
    V_MAX,
    V_MIN,
    arcsin_sqrt_to_p,
    build_grid,
    p_to_arcsin_sqrt,
)
from src.simulate_q import simulate_from_q_model

# SBC-only conventions (see module docstring): not paper-derived constants.
SBC_Q_MIN = 10.0  # pA
SBC_Q_MAX = 1000.0  # pA
SBC_BASELINE_SD = 25.0  # pA (paper's confirmed Fig. 1B / simulation constant)
SBC_EPS2 = SBC_BASELINE_SD**2


@dataclass
class SBCDraw:
    """One SBC iteration's ground truth and recovered rank(s).

    Attributes
    ----------
    theta_true : dict
        Ground-truth parameter draw (p1, p2, v, n, ...).
    ranks : dict
        Rank of theta_true within the posterior samples/grid mass, per
        parameter. For continuous parameters this is a plain rank; for the
        discrete parameter n it MUST be the jittered rank (see
        `jittered_rank_discrete`) to avoid artifactual histogram structure.
    """

    theta_true: dict
    ranks: dict


def sample_prior(
    n_candidates: Sequence[int],
    rng: np.random.Generator,
) -> dict:
    """Draw one parameter set from the settled BQA priors.

    p1, p2 ~ arcsine/Jeffreys on [0.04, 0.96] (uniform in arcsin(sqrt(p))).
    v ~ uniform on log(v) in [log(0.05), log(1)].
    n ~ uniform over the supplied candidate integers.

    Parameters
    ----------
    n_candidates : Sequence[int]
        Candidate integer values for n, drawn uniformly (2013 paper prior).
    rng : np.random.Generator
        Random number generator (caller-owned, for reproducibility).

    Returns
    -------
    dict
        Keys: "p1", "p2", "v", "n", "q" (see `sample_q_prior`).
    """
    theta_p_min = p_to_arcsin_sqrt(np.array(P_MIN))
    theta_p_max = p_to_arcsin_sqrt(np.array(P_MAX))
    theta1, theta2 = rng.uniform(theta_p_min, theta_p_max, size=2)
    p1 = float(arcsin_sqrt_to_p(theta1))
    p2 = float(arcsin_sqrt_to_p(theta2))

    log_v = rng.uniform(np.log(V_MIN), np.log(V_MAX))
    v = float(np.exp(log_v))

    n_candidates = np.asarray(n_candidates, dtype=int)
    n = int(rng.choice(n_candidates))

    q = sample_q_prior(rng)

    return {"p1": p1, "p2": p2, "v": v, "n": n, "q": q}


def sample_q_prior(rng: np.random.Generator) -> float:
    """Draw a ground-truth quantal size q for SBC (SBC-only convention).

    Log-uniform over [SBC_Q_MIN, SBC_Q_MAX] pA. See module docstring
    "SBC-ONLY ADDED CONVENTION" for why this exists and why it is not a
    paper-derived prior.
    """
    log_q = rng.uniform(np.log(SBC_Q_MIN), np.log(SBC_Q_MAX))
    return float(np.exp(log_q))


def jittered_rank_discrete(
    true_value: int,
    posterior_samples: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Compute a randomized (jittered) rank for a discrete parameter.

    Plain ranks on a discrete parameter (counting posterior samples strictly
    less than the true value) produce artifactual, non-uniform-looking rank
    histograms even under perfect calibration: ties at the true (discrete)
    value get bucketed inconsistently, and the histogram develops spurious
    structure (e.g. sawtooth or edge effects) purely from discreteness, not
    from miscalibration. The standard fix (Sailynoja et al. 2022 and
    related SBC-for-discrete-parameters literature) is to break ties
    randomly: for each true value, uniformly jitter its rank within the
    block of posterior samples equal to it, rather than assigning it a
    fixed position among ties.

    Parameters
    ----------
    true_value : int
        Ground-truth value of the discrete parameter (e.g. n) for this draw.
    posterior_samples : np.ndarray
        Posterior samples (or grid-weighted draws) of the discrete
        parameter from the inference procedure.
    rng : np.random.Generator
        Random number generator for tie-breaking jitter.

    Returns
    -------
    float
        Jittered rank in [0, len(posterior_samples)].
    """
    n_less = int(np.sum(posterior_samples < true_value))
    n_equal = int(np.sum(posterior_samples == true_value))
    jitter = rng.uniform(0.0, n_equal) if n_equal > 0 else 0.0
    return n_less + jitter


def weighted_cdf_pit(true_value: float, axis_values: np.ndarray, weights: np.ndarray) -> float:
    """Probability integral transform of `true_value` under a discretized marginal.

    Standard SBC uses ranks among L posterior samples, which reduce (under
    calibration) to Uniform(0, L); dividing by L gives a PIT value uniform
    on (0, 1). Since BQA is grid/brute-force rather than sample-based, we
    compute the PIT directly from the (normalised) weighted marginal
    posterior: the cumulative mass strictly below `true_value`, plus half
    the mass in `true_value`'s own bin (mid-P correction, avoiding the
    discretization bias of using pure left- or right-cumulative mass).

    Parameters
    ----------
    true_value : float
        Ground-truth value to locate within the marginal.
    axis_values : np.ndarray
        Sorted ascending axis values the marginal is defined over.
    weights : np.ndarray
        Normalised (or unnormalised) posterior mass at each axis value.

    Returns
    -------
    float
        PIT value in [0, 1].
    """
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    weights = weights / total

    idx = np.searchsorted(axis_values, true_value)
    idx = np.clip(idx, 0, len(axis_values) - 1)
    mass_below = weights[:idx].sum()
    mass_at = weights[idx]
    return float(mass_below + 0.5 * mass_at)


def jittered_rank_from_pmf(
    true_value: int,
    candidate_values: np.ndarray,
    pmf: np.ndarray,
    rng: np.random.Generator,
) -> float:
    """Randomized PIT for a discrete parameter given its (weighted) pmf.

    Same rationale as `jittered_rank_discrete` (avoid artifactual histogram
    structure from discreteness), adapted to a weighted pmf over candidate
    values rather than a raw sample array: mass strictly below true_value,
    plus a uniform jitter within the mass exactly at true_value.

    Parameters
    ----------
    true_value : int
        Ground-truth discrete value (e.g. n) for this draw.
    candidate_values : np.ndarray
        Candidate values the pmf is defined over (e.g. n_candidates).
    pmf : np.ndarray
        Normalised probability mass at each candidate value.
    rng : np.random.Generator
        Random number generator for tie-breaking jitter.

    Returns
    -------
    float
        Jittered PIT value in [0, 1].
    """
    candidate_values = np.asarray(candidate_values)
    pmf = np.asarray(pmf, dtype=float)
    pmf = pmf / pmf.sum()

    match = candidate_values == true_value
    if not np.any(match):
        raise ValueError(f"true_value={true_value} not among candidate_values")

    mass_below = pmf[candidate_values < true_value].sum()
    mass_at = pmf[match].sum()
    return float(mass_below + rng.uniform(0.0, mass_at))


def run_sbc(
    n_iterations: int,
    n_candidates: Sequence[int],
    n_obs_per_condition: int,
    rng: np.random.Generator,
    simulate_from_q_model_fn: Callable = simulate_from_q_model,
    inference_fn: Callable | None = None,
) -> list[SBCDraw]:
    """Run the SBC loop: draw theta*, simulate, infer, rank.

    Loop body per iteration:
      1. theta* = (p1, p2, v, n, q) ~ sample_prior(...) + sample_q_prior(...)
         (q via the SBC-only convention documented in the module docstring).
      2. Derive gamma* = 1/v*^2, and per condition k in {1, 2}: mu_k =
         n*p_k*q*, lambda_k = q*/gamma* (Eq. A3 collapses to this since q,
         gamma are shared and mu_k is itself derived from them here, rather
         than the reverse as in real data).
      3. data_k ~ simulate_from_q_model_fn(n*, p_k*, gamma*, lambda*, eps2,
         n_obs_per_condition, rng) per condition -- the model's-own-
         likelihood simulator (src.simulate_q), never
         src.simulate.simulate_responses (see TWO-SIMULATOR RULE).
      4. posterior = inference_fn(conditions, ...) -- defaults to
         `src.bqa.change_of_variables_and_marginalise`.
      5. rank theta* within the posterior marginals: `weighted_cdf_pit` for
         continuous parameters (q, gamma/v, r), `jittered_rank_from_pmf`
         for the discrete parameter n.

    Parameters
    ----------
    n_iterations : int
        Number of SBC iterations to run.
    n_candidates : Sequence[int]
        Candidate integer values for n (uniform prior support, and the grid
        search range passed to inference).
    n_obs_per_condition : int
        Number of observations to simulate per condition.
    rng : np.random.Generator
        Random number generator (caller-owned, for reproducibility).
    simulate_from_q_model_fn : Callable, optional
        Model's-own-likelihood simulator; defaults to
        `src.simulate_q.simulate_from_q_model`. Overridable for testing.
    inference_fn : Callable, optional
        BQA inference routine taking (conditions, grid, n_candidates) and
        returning the dict produced by
        `src.bqa.change_of_variables_and_marginalise`; defaults to that
        function (with a grid built internally each iteration). Overridable
        for testing.

    Returns
    -------
    list[SBCDraw]
        One entry per SBC iteration. `theta_true` includes "q" in addition
        to "p1", "p2", "v", "n".
    """
    n_candidates = list(n_candidates)
    draws: list[SBCDraw] = []

    for _ in range(n_iterations):
        theta_true = sample_prior(n_candidates, rng)
        n_true, p1_true, p2_true, v_true, q_true = (
            theta_true["n"],
            theta_true["p1"],
            theta_true["p2"],
            theta_true["v"],
            theta_true["q"],
        )
        gamma_true = 1.0 / v_true**2
        lambda_true = q_true / gamma_true

        conditions = []
        for p_true in (p1_true, p2_true):
            mu_true = n_true * p_true * q_true
            amps = simulate_from_q_model_fn(
                n=n_true,
                p=p_true,
                gamma=gamma_true,
                lam=lambda_true,
                eps2=SBC_EPS2,
                n_obs=n_obs_per_condition,
                rng=rng,
            )
            conditions.append(ConditionData(mu=mu_true, eps2=SBC_EPS2, amplitudes=amps))

        if inference_fn is None:
            grid = build_grid(n_values=np.array(n_candidates))
            posterior = change_of_variables_and_marginalise(conditions, grid, n_candidates)
        else:
            posterior = inference_fn(conditions, n_candidates)

        r_true = n_true * q_true
        # gamma_values from bqa is aligned to v_values (ascending v => descending
        # gamma); weighted_cdf_pit assumes an ascending axis, so sort first.
        gamma_order = np.argsort(posterior["gamma_values"])
        ranks = {
            "q": weighted_cdf_pit(q_true, posterior["q_axis"], posterior["q_marginal"]),
            "r": weighted_cdf_pit(r_true, posterior["r_axis"], posterior["r_marginal"]),
            "gamma": weighted_cdf_pit(
                gamma_true,
                posterior["gamma_values"][gamma_order],
                posterior["gamma_marginal"][gamma_order],
            ),
            "n": jittered_rank_from_pmf(
                n_true, posterior["n_candidates"], posterior["n_marginal"], rng
            ),
        }

        draws.append(SBCDraw(theta_true=theta_true, ranks=ranks))

    return draws


def run_sbc_reproducible(
    n_iterations: int,
    n_candidates: Sequence[int],
    n_obs_per_condition: int,
    base_seed: int = 1,
) -> list[SBCDraw]:
    """Run SBC with an independent RNG per iteration.

    Each iteration i uses `np.random.default_rng(base_seed, i)` (a distinct
    spawned stream), so any single iteration reproduces on its own regardless
    of how many came before it. This makes the SBC run both deterministic and
    resumable: a partially completed run can be extended by computing only the
    missing iteration indices, and the result does not depend on iteration
    order. Preferred over threading one shared RNG through `run_sbc` when the
    run is long enough to need checkpointing.
    """
    draws: list[SBCDraw] = []
    for i in range(n_iterations):
        rng = np.random.default_rng([base_seed, i])
        draws.extend(run_sbc(1, n_candidates, n_obs_per_condition, rng))
    return draws


def sbc_uniformity_test(draws: list[SBCDraw]) -> dict:
    """Test whether SBC rank statistics are uniformly distributed.

    Per-parameter Kolmogorov-Smirnov test of the rank statistics against
    Uniform(0, 1), following the ECDF-based calibration check used by
    Talts et al. (2018) / Sailynoja et al. (2022). KS (rather than
    chi-squared binning) is used uniformly across all parameters, including
    the jittered PIT for the discrete parameter n, since after jittering
    (`jittered_rank_from_pmf`) its values are continuous-valued and
    chi-squared binning would reintroduce the artifactual structure the
    jitter was designed to avoid.

    `run_sbc`'s ranks are already probability-integral-transform (PIT)
    values in [0, 1] (via `weighted_cdf_pit` / `jittered_rank_from_pmf`,
    since BQA is grid-based rather than sample-based), so no additional
    normalization is applied here.

    This does not correct for multiple comparisons across parameters; treat
    per-parameter p-values as diagnostic, not a single pass/fail gate.

    Parameters
    ----------
    draws : list[SBCDraw]
        Output of `run_sbc`.

    Returns
    -------
    dict
        Keyed by parameter name; each value has "ks_statistic", "p_value",
        "calibrated" (p_value > 0.05), and "n_draws".
    """
    if not draws:
        raise ValueError("draws must be non-empty")

    param_names = draws[0].ranks.keys()
    results = {}
    for name in param_names:
        ranks = np.clip(np.array([d.ranks[name] for d in draws], dtype=float), 0.0, 1.0)

        ks_result = stats.kstest(ranks, "uniform")
        results[name] = {
            "ks_statistic": float(ks_result.statistic),
            "p_value": float(ks_result.pvalue),
            "calibrated": bool(ks_result.pvalue > 0.05),
            "n_draws": len(draws),
        }

    return results
