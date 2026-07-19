"""Gaussian intrasite-noise simulator for reproducing Bhumbra & Beato (2013) figures.

TWO-SIMULATOR RULE (do not conflate):
  This module (`simulate_responses`) draws synaptic response amplitudes using
  a Gaussian model of intrasite (per-quantum) variability. It is used ONLY to
  reproduce the paper's figures (e.g. Fig 1A-D synthetic traces), and as the
  ground-truth generator for any figure/demo that is meant to look like real
  recorded data.

  It must NEVER be used to generate data for simulation-based calibration
  (SBC). SBC requires draws from the model's OWN likelihood (Eq. 8, gamma
  form) — see `src/simulate_q.py::simulate_from_q_model`. Using this Gaussian
  simulator for SBC would measure Gaussian-vs-gamma model misspecification,
  not inference correctness of the BQA grid procedure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimulatedResponses:
    """Container for a batch of simulated synaptic response amplitudes.

    Attributes
    ----------
    amplitudes : np.ndarray, shape (n_obs,)
        Simulated response amplitudes (e.g. pA).
    k_released : np.ndarray, shape (n_obs,)
        Number of quanta released per observation, k ~ Binomial(n_sites, p).
    """

    amplitudes: np.ndarray
    k_released: np.ndarray


def simulate_responses(
    n_sites: int,
    p: float,
    q: float,
    cv_intra: float,
    sigma_baseline: float,
    n_obs: int,
    rng: np.random.Generator,
) -> SimulatedResponses:
    """Simulate synaptic response amplitudes under Gaussian intrasite noise.

    Model: for each observation, draw the number of released quanta
    k ~ Binomial(n_sites, p). The amplitude is the sum of k independent
    quantal contributions, each drawn from N(q, (cv_intra * q)^2), plus
    additive baseline recording noise N(0, sigma_baseline^2).

    This is the simple multi-quantal Gaussian model used by Bhumbra & Beato
    (2013) to generate synthetic figures (Fig 1). It is NOT the model's
    inferential likelihood (Eq. 8, gamma-distributed quantal amplitudes) —
    see `src/simulate_q.py` for the SBC-appropriate simulator, and the
    module docstring above for why the two must not be conflated.

    Parameters
    ----------
    n_sites : int
        Number of release sites, n.
    p : float
        Release probability per site, 0 <= p <= 1.
    q : float
        Mean quantal amplitude.
    cv_intra : float
        Coefficient of variation of intrasite (per-quantum) amplitude noise.
    sigma_baseline : float
        Standard deviation of additive baseline recording noise.
    n_obs : int
        Number of observations (trials) to simulate.
    rng : np.random.Generator
        Random number generator (caller-owned, for reproducibility).

    Returns
    -------
    SimulatedResponses
        Amplitudes and per-trial release counts.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")
    if n_sites < 0:
        raise ValueError(f"n_sites must be >= 0, got {n_sites}")
    if n_obs < 0:
        raise ValueError(f"n_obs must be >= 0, got {n_obs}")

    k_released = rng.binomial(n_sites, p, size=n_obs)

    amplitudes = np.empty(n_obs, dtype=float)
    quantal_sd = cv_intra * q
    for i, k in enumerate(k_released):
        if k > 0:
            quanta = rng.normal(loc=q, scale=quantal_sd, size=k)
            amplitudes[i] = quanta.sum()
        else:
            amplitudes[i] = 0.0

    amplitudes += rng.normal(loc=0.0, scale=sigma_baseline, size=n_obs)

    return SimulatedResponses(amplitudes=amplitudes, k_released=k_released)
