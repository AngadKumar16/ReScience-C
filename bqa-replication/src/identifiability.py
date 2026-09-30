"""Identifiability map: parameter sweep over DeltaP (the original paper never
performed this analysis).

Sweeps P1 fixed at 0.1, DeltaP from 0.05 to 0.80 in steps of 0.01 (so
P2 = P1 + DeltaP), and for each condition pair, runs BQA inference and
records identifiability metrics (posterior credible-interval widths for q
and n) to characterize where the model can and cannot resolve the true
parameters.

Uses `src.simulate_q.simulate_from_q_model` (the model's own Eq. 8
likelihood, NOT `src.simulate.simulate_responses`) to generate condition
data, so that identifiability reflects the inference procedure's own
resolving power rather than being confounded with Gaussian-vs-gamma model
misspecification (see TWO-SIMULATOR RULE in src/simulate.py and
src/simulate_q.py).
"""

from __future__ import annotations

import numpy as np

from src.bqa import ConditionData, change_of_variables_and_marginalise
from src.grid import build_grid
from src.simulate_q import simulate_from_q_model

# Confirmed sweep constants (from the paper text).
P1_FIXED = 0.1
DELTA_P_MIN = 0.05
DELTA_P_MAX = 0.80
DELTA_P_STEP = 0.01
Q_QUANTAL = 100.0  # pA
N_SITES = 6
CV_INTRA = 0.3
BASELINE_SD = 25.0  # pA
N_OBS_PER_CONDITION = 60

# Candidate n range for the grid search (kept narrow around N_SITES so the
# sweep runs in reasonable time; widen if resolving power away from the
# true n is itself of interest).
N_CANDIDATES = [4, 5, 6, 7, 8, 9]


def delta_p_sweep_values() -> np.ndarray:
    """Return the swept DeltaP values: 0.05 to 0.80 inclusive, step 0.01."""
    n_steps = round((DELTA_P_MAX - DELTA_P_MIN) / DELTA_P_STEP) + 1
    return np.round(np.linspace(DELTA_P_MIN, DELTA_P_MAX, n_steps), 2)


def _credible_interval_width(
    axis_values: np.ndarray, marginal: np.ndarray, level: float = 0.95
) -> float:
    """Width of the central `level` credible interval of a weighted marginal."""
    weights = marginal / marginal.sum()
    cdf = np.cumsum(weights)
    lo_idx = np.searchsorted(cdf, (1 - level) / 2)
    hi_idx = np.searchsorted(cdf, 1 - (1 - level) / 2)
    lo_idx = np.clip(lo_idx, 0, len(axis_values) - 1)
    hi_idx = np.clip(hi_idx, 0, len(axis_values) - 1)
    return float(axis_values[hi_idx] - axis_values[lo_idx])


def compute_identifiability_metrics(
    p1: float,
    p2: float,
    n: int,
    v: float,
    rng: np.random.Generator,
    n_obs: int = N_OBS_PER_CONDITION,
    q: float = Q_QUANTAL,
    eps2: float = BASELINE_SD**2,
    n_candidates: list[int] = N_CANDIDATES,
) -> dict:
    """Simulate one (p1, p2) condition pair and compute BQA identifiability metrics.

    Simulates matched-model data (`simulate_from_q_model`) at the given true
    parameters, runs `change_of_variables_and_marginalise`, and reports
    posterior point estimates plus 95% credible-interval widths for q and n
    as identifiability metrics: a wide interval at a given DeltaP means the
    parameter is poorly resolved there.

    Parameters
    ----------
    p1, p2 : float
        True release probabilities for the two conditions.
    n : int
        True number of release sites.
    v : float
        True quantal coefficient of variation.
    rng : np.random.Generator
        Random number generator (caller-owned, for reproducibility).
    n_obs : int
        Observations per condition.
    q : float
        True quantal size (pA).
    eps2 : float
        Baseline noise variance.
    n_candidates : list[int]
        Candidate n values for the grid search.

    Returns
    -------
    dict
        "q_hat", "n_hat", "v_hat", "q_ci95_width", "n_ci95_width".
    """
    gamma = 1.0 / v**2
    lam = q / gamma

    conditions = []
    for p_true in (p1, p2):
        mu_true = n * p_true * q
        amps = simulate_from_q_model(n, p_true, gamma, lam, eps2, n_obs, rng)
        conditions.append(ConditionData(mu=mu_true, eps2=eps2, amplitudes=amps))

    grid = build_grid(n_values=np.array(n_candidates))
    posterior = change_of_variables_and_marginalise(conditions, grid, n_candidates)

    return {
        "q_hat": posterior["q_hat"],
        "n_hat": posterior["n_hat"],
        "v_hat": posterior["v_hat"],
        "q_ci95_width": _credible_interval_width(
            posterior["q_axis"], posterior["q_marginal"]
        ),
        "n_ci95_width": _credible_interval_width(
            np.asarray(n_candidates, dtype=float), posterior["n_marginal"]
        ),
    }


def run_identifiability_sweep(
    rng: np.random.Generator | None = None,
    n_obs: int = N_OBS_PER_CONDITION,
    per_point_base_seed: int | None = 1000,
) -> list[dict]:
    """Run the full DeltaP identifiability sweep (P1=0.1 fixed).

    For each DeltaP in `delta_p_sweep_values()`, sets P2 = P1_FIXED + DeltaP,
    simulates condition data at the paper's confirmed constants (q=100 pA,
    n=6 sites, v=CV_intra=0.3, baseline SD=25 pA, 60 obs/condition), runs BQA inference, and records identifiability metrics.

    This is a novel analysis (the original paper never performed it); no
    paper-derived numeric results are used as expected/reference values
    here, only the paper's confirmed simulation constants as inputs.

    Note: this runs one BQA grid-inference call per DeltaP value (76 by
    default), each over a 128x128 grid x 6 candidate n values x 2
    conditions; expect this to take on the order of minutes.

    Parameters
    ----------
    rng : np.random.Generator, optional
        Random number generator; a fixed default seed is used if omitted
        (for reproducibility of the sweep across runs).
    n_obs : int
        Observations per condition.

    Returns
    -------
    list[dict]
        One metrics dict per swept DeltaP value, each also containing
        "delta_p", "p1", and "p2".
    """
    if rng is None:
        rng = np.random.default_rng(0)

    v_true = CV_INTRA
    results = []
    for i, delta_p in enumerate(delta_p_sweep_values()):
        p1 = P1_FIXED
        p2 = P1_FIXED + delta_p
        # Per-point independent RNG makes each sweep point reproducible on its
        # own and order-independent (so the sweep can be checkpointed/resumed);
        # pass per_point_base_seed=None to fall back to the shared rng.
        point_rng = (
            np.random.default_rng(per_point_base_seed + i)
            if per_point_base_seed is not None
            else rng
        )
        metrics = compute_identifiability_metrics(
            p1, p2, N_SITES, v_true, point_rng, n_obs=n_obs
        )
        metrics["delta_p"] = float(delta_p)
        metrics["p1"] = p1
        metrics["p2"] = p2
        results.append(metrics)

    return results
