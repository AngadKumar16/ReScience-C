"""Draws from the BQA model's own gamma likelihood (Eq. 8).

TWO-SIMULATOR RULE (do not conflate):
  `simulate_from_q_model` here draws synthetic data directly from the
  inference model's own likelihood (`src/q_model.py::q_function`, Eq. 8,
  gamma-distributed quantal amplitudes). It exists ONLY to feed simulation-
  based calibration (SBC, see `src/sbc.py`).

  It must NEVER be substituted with `src/simulate.py::simulate_responses`
  (the Gaussian intrasite-noise simulator used for paper-figure
  reproduction). SBC is only valid if data are generated from the model's
  own likelihood: using `simulate_responses` here would measure
  Gaussian-vs-gamma model misspecification, not inference error. Any SBC
  result computed with the wrong simulator is invalid and must be
  discarded.

Sampling procedure (mixture form of Eq. 8): for each observation, draw the
number of Bernoulli successes i ~ Binomial(n, p). If i == 0, draw the
amplitude from N(0, eps2) (pure baseline noise, no quantal contribution).
If i >= 1, draw the amplitude from Gamma(shape=i*gamma, scale=lam) -- the
convolution of i i.i.d. Gamma(gamma, lam) quantal events -- with baseline
noise neglected on this branch, exactly mirroring the paper's stated Eq. 8
simplification (see src/q_model.py docstring).
"""

from __future__ import annotations

import numpy as np


def simulate_from_q_model(
    n: int,
    p: float,
    gamma: float,
    lam: float,
    eps2: float,
    n_obs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw synthetic response amplitudes from the model's own Eq. 8 likelihood.

    SBC is only valid if data are generated from the model's own likelihood.
    Using `src/simulate.py::simulate_responses` here would measure
    Gaussian-vs-gamma model misspecification, not inference error.

    Parameters
    ----------
    n : int
        Number of release sites.
    p : float
        Release probability per site.
    gamma : float
        Gamma-distribution shape parameter for a single quantal event.
    lam : float
        Gamma-distribution scale parameter for a single quantal event
        (q = gamma * lam).
    eps2 : float
        Baseline noise variance (fixed, known input).
    n_obs : int
        Number of observations to draw.
    rng : np.random.Generator
        Random number generator (caller-owned, for reproducibility).

    Returns
    -------
    np.ndarray, shape (n_obs,)
        Simulated response amplitudes drawn from the model's own likelihood.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")
    if gamma <= 0 or lam <= 0 or eps2 <= 0:
        raise ValueError("gamma, lam, and eps2 must all be strictly positive")

    i_counts = rng.binomial(n, p, size=n_obs)
    amplitudes = np.empty(n_obs, dtype=float)

    zero_mask = i_counts == 0
    n_zero = int(zero_mask.sum())
    if n_zero > 0:
        amplitudes[zero_mask] = rng.normal(loc=0.0, scale=np.sqrt(eps2), size=n_zero)

    for i in np.unique(i_counts[~zero_mask]):
        mask = i_counts == i
        amplitudes[mask] = rng.gamma(shape=i * gamma, scale=lam, size=int(mask.sum()))

    return amplitudes
