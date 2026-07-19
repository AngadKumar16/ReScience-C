"""Eq. 8 quantal likelihood (Q-function), Bhumbra & Beato (2013).

TWO-SIMULATOR RULE (do not conflate):
  `q_function` here is the model's own likelihood, used both for BQA
  inference (src/bqa.py) and as the target that `src/simulate_q.py::
  simulate_from_q_model` draws from for SBC. It must NOT be confused
  with the Gaussian synthetic-figure generator in `src/simulate.py::
  simulate_responses`, which exists only to reproduce paper figures and is
  not a model of the inference procedure's own generative process.

Reference (paper equations, transcribed from the published PDF):
  Eq. 1: B(i|n, p) = n!/(i!(n-i)!) p^i (1-p)^(n-i)          Binomial pmf.
  Eq. 5: N(x|0, eps^2) = 1/sqrt(2*pi*eps^2) * exp(-x^2/2eps^2)  bias-free
         Gaussian for additive baseline noise.
  Eq. 6: G(x|gamma, lambda) = 1/(lambda^gamma * Gamma(gamma)) *
         x^(gamma-1) * exp(-x/lambda), x, gamma, lambda > 0     Gamma pdf,
         shape gamma, SCALE lambda (not rate -- confirmed by Eq. 7 below
         and Fig. 1C: gamma=11.1, lambda=9 pA gives q = gamma*lambda =
         99.9 ~ 100 pA, which only holds under the scale convention).
  Eq. 7: q = gamma * lambda                                   mean quantal
         size as the gamma distribution's first moment.
  Eq. 8: Q(x|n, p, gamma, lambda, eps^2)
           = B(0|n, p) * N(x|0, eps^2)
             + sum_{i=1}^{n} B(i|n, p) * G(x|i*gamma, lambda)
         The convolution of i i.i.d. Gamma(gamma, lambda) quantal
         amplitudes is Gamma(i*gamma, lambda) (same scale, shapes add).
         Background noise is neglected on the i>=1 terms since its
         additive effect becomes small relative to multiplicative quantal
         variability as i increases (paper's stated simplification).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def q_function(
    x: np.ndarray,
    n: int,
    p: float,
    gamma: float,
    lam: float,
    eps2: float,
) -> np.ndarray:
    """Evaluate the combined quantal likelihood Q(x) per Eq. 8.

    Reproducing Fig 1 panel D (see src/figure1.py) with this function is the
    project's feasibility gate.

    Parameters
    ----------
    x : np.ndarray
        Response amplitude values at which to evaluate the likelihood.
    n : int
        Number of release sites.
    p : float
        Release probability per site, 0 <= p <= 1.
    gamma : float
        Gamma-distribution shape parameter for a single quantal event.
    lam : float
        Gamma-distribution scale parameter for a single quantal event
        (q = gamma * lam; CV = 1 / sqrt(gamma), Eq. 6-7).
    eps2 : float
        Baseline noise variance (fixed, known input, not inferred).

    Returns
    -------
    np.ndarray
        Q(x), the likelihood density evaluated at each x, same shape as x.
    """
    x = np.asarray(x, dtype=float)
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")
    if gamma <= 0 or lam <= 0 or eps2 <= 0:
        raise ValueError("gamma, lam, and eps2 must all be strictly positive")

    baseline_sd = np.sqrt(eps2)
    b0 = stats.binom.pmf(0, n, p)
    total = b0 * stats.norm.pdf(x, loc=0.0, scale=baseline_sd)

    for i in range(1, n + 1):
        b_i = stats.binom.pmf(i, n, p)
        total = total + b_i * stats.gamma.pdf(x, a=i * gamma, scale=lam)

    return total
