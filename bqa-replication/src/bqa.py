"""Bayesian Quantal Analysis (BQA): per-condition likelihoods, combination
across conditions, change of variables to (q, gamma, r), and marginalisation
to posteriors over (p1, p2, v, n).

Reference (paper equations, transcribed from the published PDF; see
NOTES.md for provenance):

  Condition means mu_k and baseline-noise variance eps^2 are FIXED KNOWN
  inputs (METHODS, "Quantal probabilities").

  Eq. A2: prior for p_k is the arcsine (Jeffreys') distribution,
          f(p_k) = 1 / (pi*sqrt(p_k*(1-p_k))), bounded 0.04 <= p_k <= 0.96.
          This is uniform in arcsin(sqrt(p_k)) -- see src/grid.py.
  Eq. A3: lambda_k = mu_k / (n * p_k * gamma)   [from mu=npq (Eq. 2) and
          q=gamma*lambda (Eq. 7): q = mu_k/(n p_k) => lambda=q/gamma].
  Eq. A4: since arcsin(sqrt(p_k)), log_e(v), and n are all assigned
          UNIFORM independent priors, the joint prior in these transformed
          coordinates is flat, so the posterior in transformed coordinates
          is directly proportional to the likelihood (no extra prior
          weighting/Jacobian needed once the grid itself is built in
          arcsin(sqrt(p))/log(v) coordinates -- see src/grid.py).
  Eq. A5/A6: per-condition log-likelihood L_k = sum_x log Q(x|n, p_k,
          gamma, lambda_k, eps^2), Q from Eq. 8 (src/q_model.py).
  Eq. A7: change of variables to ideal parameters psi=(q, gamma, r):
          q = mu_k/(n*p_k), gamma = 1/v^2, r = mu_k/p_k = n*q. Since r=n*q
          identically for a fixed n, only q needs resampling onto a common
          axis across conditions (mu_k differs); gamma is already a
          monotonic reparametrisation of the shared v-axis and needs no
          resampling. r depends additionally on the candidate n, so a
          shared r-axis is used to marginalise across candidate n values
          (paper: "computationally optimal if their respective values are
          sampled intermixed directly" / log-spaced, resolution 128).
  Eq. A8: f(q, gamma, r | X, M) = prod_k f_k(q, gamma, r | x_k, mu_k)
          -- combine conditions by elementwise product of their (resampled)
          posterior mass grids, for a fixed candidate n.
  Eq. A9/A10: joint posterior normalised over the full (q, gamma, r) grid.
  Eq. A4 (n): n is also assigned a uniform prior; marginalising over
          candidate n values is implemented as a SUM (sum rule, Eq. 12) of
          each candidate n's (q, gamma) posterior mass, scattered onto the
          shared r-axis at r = n*q (this reproduces the "discrete
          discontinuities" / comb-like structure described for Fig. 3).
  Eq. A11: n_hat = r_hat/q_hat, lambda_hat = q_hat/gamma_hat,
          v_hat = sqrt(1/gamma_hat), using medians of the final marginal
          posteriors for q, gamma, r.

Heterogeneous-release extension (Eqs. A12-A13, beta model over p across
sites) is implemented separately in `heterogeneous_release_pmf` below; see
its docstring for the 2^n enumeration caveat.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy import stats

from src.grid import Grid


@dataclass
class ConditionData:
    """Fixed, known inputs for one experimental condition.

    Attributes
    ----------
    mu : float
        Condition mean response amplitude (fixed known input, not sampled).
    eps2 : float
        Baseline-noise variance for this condition (fixed known input).
    amplitudes : np.ndarray
        Observed response amplitudes for this condition.
    """

    mu: float
    eps2: float
    amplitudes: np.ndarray


def per_condition_log_likelihood(
    condition: ConditionData,
    grid: Grid,
    n: int,
) -> np.ndarray:
    """Evaluate the per-condition log-likelihood surface over the (p, v) grid.

    For each (p, v) grid point, sums log Q(x_i | n, p, gamma, lam, eps2)
    over observations x_i in `condition.amplitudes`, where gamma = 1/v^2
    and lambda = mu / (n * p * gamma) (Eq. A3).

    Parameters
    ----------
    condition : ConditionData
        Fixed known inputs and observed amplitudes for this condition.
    grid : Grid
        Grid axes produced by `grid.build_grid`.
    n : int
        Candidate number of release sites for this evaluation.

    Returns
    -------
    np.ndarray, shape (P_RESOLUTION, V_RESOLUTION)
        Log-likelihood evaluated at each (p, v) grid point.

    Note
    ----
    This is a vectorized reimplementation of Eq. 8 (mirroring
    `src.q_model.q_function` term-for-term) rather than a per-cell call
    into that function, since the latter is not vectorized over (p, v) and
    a Python-level loop over a 128x128 grid is prohibitively slow when
    called repeatedly (SBC, identifiability sweep). Any change to Eq. 8's
    form in `q_model.q_function` must be mirrored here.
    """
    p = grid.p_values  # (n_p,)
    v = grid.v_values  # (n_v,)
    gamma = 1.0 / v**2  # (n_v,)
    n_p, n_v = p.size, v.size

    with np.errstate(divide="ignore", invalid="ignore"):
        lam = condition.mu / (n * p[:, None] * gamma[None, :])  # (n_p, n_v)
    valid = np.isfinite(lam) & (lam > 0)

    x = condition.amplitudes[:, None, None]  # (n_obs, 1, 1)
    baseline_sd = np.sqrt(condition.eps2)

    b0 = stats.binom.pmf(0, n, p).reshape(1, n_p, 1)
    total = b0 * stats.norm.pdf(x, loc=0.0, scale=baseline_sd)  # (n_obs, n_p, 1)

    for i in range(1, n + 1):
        b_i = stats.binom.pmf(i, n, p).reshape(1, n_p, 1)
        a_i = (i * gamma).reshape(1, 1, n_v)
        scale_i = np.where(valid, lam, 1.0).reshape(1, n_p, n_v)
        term = stats.gamma.pdf(x, a=a_i, scale=scale_i)  # (n_obs, n_p, n_v)
        total = total + b_i * term  # broadcasts to (n_obs, n_p, n_v)

    total = np.clip(total, 1e-300, None)
    log_l = np.sum(np.log(total), axis=0)  # (n_p, n_v)
    log_l = np.where(valid, log_l, -np.inf)
    return log_l


def _normalize_log_grid(log_grid: np.ndarray) -> np.ndarray:
    """Convert a log-likelihood/log-posterior grid to a normalised mass grid."""
    finite = np.isfinite(log_grid)
    if not np.any(finite):
        raise ValueError("log_grid has no finite entries to normalize")
    m = np.max(log_grid[finite])
    weights = np.where(finite, np.exp(log_grid - m), 0.0)
    total = weights.sum()
    if total <= 0:
        raise ValueError("normalized weights sum to zero")
    return weights / total


def build_q_axis(
    conditions: list[ConditionData],
    n_candidates: list[int],
    p_min: float,
    p_max: float,
    resolution: int = 128,
) -> np.ndarray:
    """Build a shared log-spaced q-axis spanning all conditions and candidate n.

    q = mu_k / (n * p_k) (Eq. A7); bounds are derived from the extremes of
    mu_k, n, and the p-grid bounds so every condition's/candidate-n's
    q-values fall within range.
    """
    mus = np.array([c.mu for c in conditions], dtype=float)
    n_arr = np.array(n_candidates, dtype=float)
    q_min = np.min(mus) / (np.max(n_arr) * p_max)
    q_max = np.max(mus) / (np.min(n_arr) * p_min)
    return np.geomspace(q_min, q_max, resolution)


def build_r_axis(q_axis: np.ndarray, n_candidates: list[int]) -> np.ndarray:
    """Build a shared log-spaced r-axis spanning r = n*q across candidate n."""
    n_arr = np.array(n_candidates, dtype=float)
    r_min = q_axis.min() * n_arr.min()
    r_max = q_axis.max() * n_arr.max()
    return np.geomspace(r_min, r_max, len(q_axis))


def _nearest_axis_index(values: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Index of the nearest axis bin (in linear space) for each value."""
    idx = np.searchsorted(axis, values)
    idx = np.clip(idx, 1, len(axis) - 1)
    left = idx - 1
    choose_left = np.abs(axis[left] - values) < np.abs(axis[idx] - values)
    return np.where(choose_left, left, idx)


def _remap_p_axis_to_q(
    mass_pv: np.ndarray,
    p_values: np.ndarray,
    mu: float,
    n: int,
    q_axis: np.ndarray,
) -> np.ndarray:
    """Mass-conserving remap of a (P_RES, V_RES) mass grid's p-axis onto q_axis.

    q = mu / (n * p) (Eq. A7) is monotonic in p, so each p-row's mass is
    scattered (nearest-bin, mass-conserving) onto the corresponding q-bin;
    the v-axis (gamma) dimension is untouched since q does not depend on v.
    """
    q_values = mu / (n * p_values)
    target_idx = _nearest_axis_index(q_values, q_axis)
    out = np.zeros((len(q_axis), mass_pv.shape[1]))
    np.add.at(out, target_idx, mass_pv)
    return out


def combine_conditions(
    conditions: list[ConditionData],
    grid: Grid,
    n: int,
    q_axis: np.ndarray,
) -> np.ndarray:
    """Combine per-condition posteriors across K conditions for fixed n (Eq. A8).

    For each condition: compute per_condition_log_likelihood, normalise to a
    posterior mass grid over (p, v) (valid since priors are flat in these
    transformed coordinates, Eq. A4), remap the p-axis to the shared q-axis
    (Eq. A7), then combine conditions by elementwise product (Eq. A8).

    Parameters
    ----------
    conditions : list[ConditionData]
        One entry per experimental condition (K conditions).
    grid : Grid
        Grid axes produced by `grid.build_grid`.
    n : int
        Candidate number of release sites.
    q_axis : np.ndarray
        Shared log-spaced q-axis (see `build_q_axis`).

    Returns
    -------
    np.ndarray, shape (len(q_axis), V_RESOLUTION)
        Combined (unnormalised) posterior mass over (q, gamma) for this n.
    """
    combined = np.ones((len(q_axis), grid.v_values.size))
    for condition in conditions:
        log_l = per_condition_log_likelihood(condition, grid, n)
        mass_pv = _normalize_log_grid(log_l)
        mass_qv = _remap_p_axis_to_q(mass_pv, grid.p_values, condition.mu, n, q_axis)
        combined = combined * mass_qv
    return combined


def change_of_variables_and_marginalise(
    conditions: list[ConditionData],
    grid: Grid,
    n_candidates: list[int],
    q_axis: np.ndarray | None = None,
    r_axis: np.ndarray | None = None,
) -> dict:
    """Apply the Eqs. A7-A11 change of variables and marginalise over n.

    For each candidate n: combine conditions on the shared q-axis (Eq. A8),
    then scatter the resulting (q, gamma) mass onto the shared r-axis at
    r = n*q (mass-conserving), accumulating across n (sum rule, Eq. 12,
    since n also carries a uniform prior -- Eq. A4). The final joint grid
    is normalised (Eq. A10) and its marginals give posteriors for q, gamma
    (via the v-axis), and r; medians of these give the parameter estimates
    (Eq. A11).

    Parameters
    ----------
    conditions : list[ConditionData]
        One entry per experimental condition.
    grid : Grid
        Grid axes produced by `grid.build_grid`.
    n_candidates : list[int]
        Candidate integer values for n (uniform prior support).
    q_axis, r_axis : np.ndarray, optional
        Shared axes; built via `build_q_axis`/`build_r_axis` if omitted.

    Returns
    -------
    dict
        Keys: "q_axis", "r_axis", "v_values", "joint_mass" (shape
        (len(q_axis), V_RESOLUTION, len(r_axis))), "q_marginal",
        "gamma_marginal", "gamma_values", "r_marginal", "n_candidates",
        "n_marginal" (unnormalised-then-normalised mass per candidate n,
        summed before scattering onto the shared r-axis -- the closest
        this pipeline has to a genuine marginal posterior over n, since
        the final n_hat is instead a derived point estimate per Eq. A11),
        "q_hat", "r_hat", "gamma_hat", "lambda_hat", "v_hat", "n_hat".
    """
    p_min, p_max = grid.p_values.min(), grid.p_values.max()
    if q_axis is None:
        q_axis = build_q_axis(conditions, n_candidates, p_min, p_max)
    if r_axis is None:
        r_axis = build_r_axis(q_axis, n_candidates)

    n_q = len(q_axis)
    n_v = grid.v_values.size
    n_r = len(r_axis)
    joint_mass = np.zeros((n_q, n_v, n_r))
    n_mass_raw = np.zeros(len(n_candidates))  # unnormalised mass per candidate n

    for ni, n in enumerate(n_candidates):
        mass_qv = combine_conditions(conditions, grid, n, q_axis)
        n_mass_raw[ni] = mass_qv.sum()
        r_values = n * q_axis
        r_idx = _nearest_axis_index(r_values, r_axis)
        joint_mass[np.arange(n_q), :, r_idx] += mass_qv

    total = joint_mass.sum()
    if total <= 0:
        raise ValueError("joint posterior mass is zero everywhere")
    joint_mass = joint_mass / total
    n_marginal = n_mass_raw / n_mass_raw.sum()

    q_marginal = joint_mass.sum(axis=(1, 2))
    gamma_values = 1.0 / grid.v_values**2
    gamma_marginal = joint_mass.sum(axis=(0, 2))
    r_marginal = joint_mass.sum(axis=(0, 1))

    q_hat = _weighted_median(q_axis, q_marginal)
    r_hat = _weighted_median(r_axis, r_marginal)
    gamma_order = np.argsort(gamma_values)
    gamma_hat = _weighted_median(gamma_values[gamma_order], gamma_marginal[gamma_order])

    lambda_hat = q_hat / gamma_hat
    v_hat = np.sqrt(1.0 / gamma_hat)
    n_hat = estimate_n(r_hat, q_hat)

    return {
        "q_axis": q_axis,
        "r_axis": r_axis,
        "v_values": grid.v_values,
        "joint_mass": joint_mass,
        "q_marginal": q_marginal,
        "gamma_marginal": gamma_marginal,
        "gamma_values": gamma_values,
        "r_marginal": r_marginal,
        "n_candidates": np.asarray(n_candidates),
        "n_marginal": n_marginal,
        "q_hat": q_hat,
        "r_hat": r_hat,
        "gamma_hat": gamma_hat,
        "lambda_hat": lambda_hat,
        "v_hat": v_hat,
        "n_hat": n_hat,
    }


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Median of a discrete weighted distribution (values assumed sorted ascending)."""
    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]
    idx = np.searchsorted(cdf, 0.5)
    idx = np.clip(idx, 0, len(values) - 1)
    return float(values[idx])


def estimate_n(median_r: float, median_q: float) -> float:
    """Estimate n as median(r) / median(q), per Bhumbra & Beato (2013), Eq. A11.

    Parameters
    ----------
    median_r : float
        Posterior median of r (derived quantity).
    median_q : float
        Posterior median of q (derived quantity).

    Returns
    -------
    float
        Estimated number of release sites, n = median(r) / median(q).
    """
    if median_q == 0:
        raise ValueError("median_q must be nonzero to estimate n")
    return median_r / median_q


def heterogeneous_release_pmf(
    n: int,
    mean_p: float,
    alpha: float,
) -> np.ndarray:
    """Numerically compute the heterogeneous-release success-count pmf (Eqs. A12-A13).

    Central synapses may exhibit heterogeneous release probabilities across
    sites (paper's "Modeling heterogeneous release probabilities" section).
    Site release probabilities are modeled as n i.i.d. draws from a Beta
    distribution Beta(alpha, beta_k) parameterised by homogeneity alpha and
    condition mean release probability mean_p (Eq. A12-A13):

      P(p|alpha, beta) = p^(alpha-1) * (1-p)^(beta-1) / B(alpha, beta)
      beta_k = alpha * (1 - mean_p) / mean_p                        (Eq. A13)

    There is no closed-form analytical expression for the resulting
    generalized binomial pmf B'(i|n, mean_p, alpha) (paper: "To our
    knowledge, there is no analytical expression for the analogous
    probability mass function"). The paper's numerical procedure evaluates
    every permutation of release/failure across the n sites -- 2^n release-
    state combinations -- and accumulates their probabilities via the
    product rule into the pmf for i total successes.

    WARNING: this is the hardest, most compute-intensive step in the paper
    (explicitly flagged as such in NOTES.md). Cost is O(2^n * n); it is
    only practical for small n (single digits). It does NOT sample p per
    site from the Beta distribution directly per permutation -- doing so
    would require re-deriving per-site probabilities analytically. Instead,
    following the paper's stated procedure, we assign per-site release
    probabilities by evenly sampling the Beta CDF across the n sites
    ("the beta cumulative density function is sampled evenly to assign
    Bernoulli probabilities of release to each of n release sites
    according to a beta distribution of mean mean_p and uniformity alpha").

    Parameters
    ----------
    n : int
        Number of release sites. Kept small (<= ~15) for tractability.
    mean_p : float
        Mean release probability across sites for this condition.
    alpha : float
        Beta-distribution homogeneity shaping parameter (larger = more
        homogeneous release across sites).

    Returns
    -------
    np.ndarray, shape (n + 1,)
        B'(i|n, mean_p, alpha) for i = 0, ..., n.
    """
    if n > 20:
        raise ValueError(
            f"n={n} makes the 2^n enumeration in heterogeneous_release_pmf "
            "intractable; keep n small (paper's own caveat)."
        )
    if not (0.0 < mean_p < 1.0):
        raise ValueError(f"mean_p must be in (0, 1), got {mean_p}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    beta_param = alpha * (1.0 - mean_p) / mean_p

    # Sample the beta CDF evenly across the n sites (paper's stated method).
    quantiles = (np.arange(n) + 0.5) / n
    site_p = stats.beta.ppf(quantiles, alpha, beta_param)

    pmf = np.zeros(n + 1)
    for outcomes in product([0, 1], repeat=n):
        prob = 1.0
        i = 0
        for site_idx, released in enumerate(outcomes):
            if released:
                prob *= site_p[site_idx]
                i += 1
            else:
                prob *= 1.0 - site_p[site_idx]
        pmf[i] += prob

    return pmf
