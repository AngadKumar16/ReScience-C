"""Guard tests that keep the two Eq. 8 implementations in lockstep and pin
the feasibility gate to a numeric criterion.

`src.bqa.per_condition_log_likelihood` is a vectorized reimplementation of
Eq. 8 for speed; `src.q_model.q_function` is the reference. They must agree
cell for cell, otherwise a change to one silently diverges from the other.
"""

from __future__ import annotations

import numpy as np

from src.bqa import (
    ConditionData,
    change_of_variables_and_marginalise,
    per_condition_log_likelihood,
)
from src.figure1 import GAMMA_LAMBDA, GAMMA_SHAPE, Q_QUANTAL, gate_peak_check, panel_c
from src.grid import build_grid
from src.q_model import q_function
from src.simulate_q import simulate_from_q_model


def test_bqa_likelihood_matches_q_model_reference():
    """The vectorized per-cell likelihood must equal the reference q_function."""
    rng = np.random.default_rng(0)
    n = 5
    mu = 180.0
    eps2 = 25.0**2
    amplitudes = rng.gamma(shape=11.1 * 2, scale=9.0, size=12)

    grid = build_grid(np.array([n]), p_resolution=16, v_resolution=16)
    cond = ConditionData(mu=mu, eps2=eps2, amplitudes=amplitudes)

    vectorized = per_condition_log_likelihood(cond, grid, n)

    # Reference: loop q_function over every (p, v) grid cell.
    reference = np.full_like(vectorized, -np.inf)
    for i, p in enumerate(grid.p_values):
        for j, v in enumerate(grid.v_values):
            gamma = 1.0 / v**2
            lam = mu / (n * p * gamma)
            if not np.isfinite(lam) or lam <= 0:
                continue
            q = q_function(amplitudes, n=n, p=p, gamma=gamma, lam=lam, eps2=eps2)
            reference[i, j] = np.sum(np.log(np.clip(q, 1e-300, None)))

    finite = np.isfinite(vectorized) & np.isfinite(reference)
    assert finite.any()
    assert np.allclose(vectorized[finite], reference[finite], rtol=1e-9, atol=1e-6)


def test_feasibility_gate_peaks_near_multiples_of_q():
    """Fig 1D modes must sit near integer multiples of the quantal size."""
    result = gate_peak_check()
    assert result["passed"]
    assert result["interior_peaks_pa"].size >= 1
    assert result["max_rel_error"] <= 0.15


def test_both_resampling_modes_conserve_mass_and_agree_closely():
    """Nearest-bin and linear transport both conserve mass and give close estimates.

    Linear interpolation is offered as an alternative transport; it splits
    mass between bracketing bins rather than dumping it in one. Both must
    conserve total posterior mass exactly, and on model-matched data they
    should land on similar point estimates (the SBC study shows linear does
    not, however, fix the interval calibration; see the paper).
    """
    rng = np.random.default_rng(0)
    n, q, v = 6, 100.0, 0.3
    gamma, eps2 = 1.0 / v**2, 25.0**2
    lam = q / gamma
    conds = []
    for p in (0.2, 0.5):
        amps = simulate_from_q_model(n, p, gamma, lam, eps2, 60, rng)
        conds.append(ConditionData(mu=n * p * q, eps2=eps2, amplitudes=amps))
    n_candidates = [4, 5, 6, 7, 8]
    grid = build_grid(np.array(n_candidates))

    est = {}
    for mode in ("nearest", "linear"):
        post = change_of_variables_and_marginalise(
            conds, grid, n_candidates, interpolation=mode
        )
        assert np.isclose(post["joint_mass"].sum(), 1.0, atol=1e-9)
        est[mode] = post["q_hat"]
    # Same ballpark for the point estimate (transport choice is second order).
    assert abs(est["nearest"] - est["linear"]) < 0.5 * q


def _q_posterior_sd(post):
    a, w = post["q_axis"], post["q_marginal"] / post["q_marginal"].sum()
    m = (a * w).sum()
    return float(np.sqrt(((a - m) ** 2 * w).sum()))


def test_product_combination_oversharpens_with_more_conditions():
    """Diagnose the SBC miscalibration: the product combination applies the
    prior once per condition, so the posterior narrows as conditions are
    added even when the extra conditions carry the same q. The log-likelihood
    combination applies the prior once and does not over-sharpen nearly as
    much. This locks the cause identified in the paper.
    """
    rng = np.random.default_rng(3)
    n, q, v = 6, 100.0, 0.3
    gamma, eps2, lam = 1.0 / v**2, 25.0**2, q / (1.0 / v**2)
    n_candidates = [4, 5, 6, 7, 8]
    grid = build_grid(np.array(n_candidates))

    def sd_for(probs, combine):
        conds = [
            ConditionData(
                mu=n * p * q,
                eps2=eps2,
                amplitudes=simulate_from_q_model(n, p, gamma, lam, eps2, 60, rng),
            )
            for p in probs
        ]
        post = change_of_variables_and_marginalise(
            conds, grid, n_candidates, combine=combine
        )
        assert np.isclose(post["joint_mass"].sum(), 1.0, atol=1e-9)
        return _q_posterior_sd(post)

    sd_1 = sd_for([0.3], "product")
    sd_3 = sd_for([0.2, 0.4, 0.6], "product")
    # Product: three conditions report a much tighter posterior than one.
    assert sd_3 < 0.75 * sd_1
    # loglik still conserves mass and does not collapse as hard.
    sd_3_loglik = sd_for([0.2, 0.4, 0.6], "loglik")
    assert sd_3_loglik > sd_3


def test_panel_c_uses_lambda_as_scale_not_rate():
    """Fig. 1C must be the gamma whose mean IS the quantal size.

    Eq. 7 sets q = gamma * lambda, so lambda is the scale. Passing it to
    scipy as a rate (scale=1/lambda) puts the mean at 1.23 pA instead of
    100 pA, which renders the published panel empty on its 0-300 pA axis.
    This regression test pins the parameterisation to the one
    `src.q_model.q_function` uses.
    """
    x, pdf = panel_c()
    mass = np.trapezoid(pdf, x)
    mean = np.trapezoid(x * pdf, x)
    assert np.isclose(mass, 1.0, atol=1e-3), "panel C truncated: density does not integrate to 1"
    assert np.isclose(mean, Q_QUANTAL, rtol=1e-3)
    assert np.isclose(GAMMA_SHAPE * GAMMA_LAMBDA, Q_QUANTAL)
    # The panel must actually be visible on its own axis.
    assert x[np.argmax(pdf)] > 50.0


def test_panel_c_matches_q_model_single_quantum_term():
    """Panel C and the i = 1 gamma term of Eq. 8 must be the same density."""
    from scipy import stats

    x, pdf = panel_c()
    reference = stats.gamma.pdf(x, a=GAMMA_SHAPE, scale=GAMMA_LAMBDA)
    assert np.allclose(pdf, reference)
