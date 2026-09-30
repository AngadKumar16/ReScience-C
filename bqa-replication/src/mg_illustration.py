"""Model-mismatch-Gaussian ("MG") detectability illustration.

NOTE(author) on the "MG" naming: the source paper's text does not use the
literal acronym "MG"; the glossary and body text were checked directly and no such term appears. This module instead
implements the closest concretely-documented phenomenon in the paper that
"MG detectability" could plausibly refer to -- the paper's own explicit
claim (METHODS/Results/Discussion) that data simulated under a *Gaussian*
intrasite-noise model (`src.simulate.simulate_responses`) are nonetheless
accurately fit by BQA's *gamma* quantal likelihood (Eq. 8,
`src.q_model.q_function`), i.e. that BQA is robust to Gaussian-vs-gamma
Model mismatch/Gamma-vs-Gaussian ("MG") misspecification:

  "Despite the mismatch between the simulation and model probability
  density function, the quantal size and variability were estimated
  accurately by BQA." (Discussion, "Estimation of quantal variability")

This is a deliberate, explicit use of `simulate_responses` (Gaussian) fed
into the gamma-likelihood BQA inference pipeline -- the ONE place in this
repository where that combination is intentional rather than a
TWO-SIMULATOR RULE violation, precisely because the point here is to
measure robustness to that mismatch, not inference calibration (contrast
with src/sbc.py, which must never do this -- see TWO-SIMULATOR RULE in
src/simulate.py and src/simulate_q.py).

If the human author confirms a different, paper-specific meaning of "MG"
(e.g. from a source outside the transcribed sections), this module should
be revisited; the interpretation above is documented rather than silently
assumed.
"""

from __future__ import annotations

import numpy as np

from src.bqa import ConditionData, change_of_variables_and_marginalise
from src.grid import build_grid
from src.simulate import simulate_responses


def demonstrate_mg_detectability(
    n_sites: int = 6,
    q: float = 100.0,
    cv_intra: float = 0.3,
    baseline_sd: float = 25.0,
    n_obs: int = 60,
    p1: float = 0.1,
    p2: float = 0.3,
    n_candidates: list[int] | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Demonstrate BQA's robustness to Gaussian-vs-gamma model mismatch.

    Simulates two-condition data under the Gaussian intrasite-noise model
    (`simulate_responses`) at the paper's confirmed constants, then runs it
    through the gamma-likelihood BQA inference pipeline (`src.bqa`), which
    is a genuine mismatch between generative and inferential models. Reports
    how closely the (mismatched) BQA estimates track the true simulation
    parameters, illustrating the paper's own robustness claim (see module
    docstring for the exact quoted passage this reproduces the spirit of).

    Parameters
    ----------
    n_sites : int
        True number of release sites.
    q : float
        True mean quantal size (pA).
    cv_intra : float
        True intrasite coefficient of variation (Gaussian simulator).
    baseline_sd : float
        True baseline noise SD (pA).
    n_obs : int
        Observations per condition.
    p1, p2 : float
        True release probabilities for the two conditions.
    n_candidates : list[int], optional
        Candidate n values for the BQA grid search; defaults to a small
        window around `n_sites`.
    rng : np.random.Generator, optional
        Random number generator; a fixed default seed is used if omitted.

    Returns
    -------
    dict
        "q_true", "q_hat", "n_true", "n_hat", "v_hat" (BQA's estimate of
        the gamma-model CV, not directly comparable to cv_intra but
        reported for reference), plus the raw BQA posterior dict under key
        "posterior".
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if n_candidates is None:
        n_candidates = [n_sites - 2, n_sites - 1, n_sites, n_sites + 1, n_sites + 2]

    eps2 = baseline_sd**2
    conditions = []
    for p_true in (p1, p2):
        sim = simulate_responses(n_sites, p_true, q, cv_intra, baseline_sd, n_obs, rng)
        mu_true = n_sites * p_true * q
        conditions.append(ConditionData(mu=mu_true, eps2=eps2, amplitudes=sim.amplitudes))

    grid = build_grid(n_values=np.array(n_candidates))
    posterior = change_of_variables_and_marginalise(conditions, grid, n_candidates)

    return {
        "q_true": q,
        "q_hat": posterior["q_hat"],
        "n_true": n_sites,
        "n_hat": posterior["n_hat"],
        "v_hat": posterior["v_hat"],
        "posterior": posterior,
    }
