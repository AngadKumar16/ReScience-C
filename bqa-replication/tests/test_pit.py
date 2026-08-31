"""Regression tests for the probability-integral transform used by SBC.

`weighted_cdf_pit` carried a half-cell offset that inflated every rank and
produced the one-sided quantal-size miscalibration reported in earlier drafts
(see src/sbc.py and src/pit_diagnostics.py). These tests pin the corrected
behaviour so the defect cannot return silently.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from src.sbc import weighted_cdf_pit, weighted_cdf_pit_legacy


def _lognormal_mass(axis: np.ndarray, s: float, scale: float) -> np.ndarray:
    """Probability MASS at each axis point (density times bin width)."""
    z = (np.log(axis) - math.log(scale)) / s
    dens = np.exp(-0.5 * z * z) / (axis * s * math.sqrt(2.0 * math.pi))
    mass = dens * np.gradient(axis)
    return mass / mass.sum()


def test_median_of_symmetric_mass_is_one_half():
    """The PIT of the median of a symmetric marginal must be 0.5."""
    axis = np.geomspace(10.0, 1000.0, 129)
    mass = _lognormal_mass(axis, s=0.4, scale=100.0)
    assert weighted_cdf_pit(100.0, axis, mass) == pytest.approx(0.5, abs=5e-3)


def test_uniform_ranks_on_a_known_distribution():
    """Truth drawn from the same distribution the marginal describes.

    Under a correct PIT the ranks must be Uniform(0, 1). This is the test the
    defective implementation fails: it returns a mean PIT near 0.51 and about
    5.9% of ranks above 0.95 against 4.7% below 0.05.
    """
    rng = np.random.default_rng(0)
    axis = np.geomspace(10.0, 1000.0, 128)
    mass = _lognormal_mass(axis, s=0.4, scale=100.0)
    samples = np.exp(rng.normal(math.log(100.0), 0.4, size=4000))

    ranks = np.array([weighted_cdf_pit(x, axis, mass) for x in samples])
    assert stats.kstest(ranks, "uniform").pvalue > 0.01
    assert ranks.mean() == pytest.approx(0.5, abs=0.01)


def test_legacy_pit_is_biased_upward():
    """The old transform is never below the new one and is biased above it.

    This is the signature that showed up as 'the quantal-size marginal's upper
    tail is systematically too short': a strictly non-negative rank inflation
    bounded by one cell's mass.
    """
    rng = np.random.default_rng(1)
    axis = np.geomspace(10.0, 1000.0, 128)
    mass = _lognormal_mass(axis, s=0.4, scale=100.0)
    samples = np.exp(rng.normal(math.log(100.0), 0.4, size=1000))

    new = np.array([weighted_cdf_pit(x, axis, mass) for x in samples])
    old = np.array([weighted_cdf_pit_legacy(x, axis, mass) for x in samples])
    assert np.all(old >= new - 1e-12)
    assert old.mean() > new.mean()


def test_pit_is_invariant_under_a_decreasing_reparametrisation():
    """PIT must be reparametrisation-invariant, which is what SBC relies on.

    With mu and n fixed, q = mu / (n p) is strictly decreasing in p, so the q
    rank must equal one minus the p rank. Checked here on a synthetic marginal
    so it tests the transform alone, independently of any BQA machinery.
    """
    mu, n = 600.0, 6.0
    p_axis = np.linspace(0.04, 0.96, 256)
    p_mass = np.exp(-0.5 * ((p_axis - 0.4) / 0.08) ** 2)
    p_mass /= p_mass.sum()

    # Same distribution expressed on the corresponding q axis (mass moves with
    # the points, so no Jacobian is involved).
    q_of_p = mu / (n * p_axis)
    order = np.argsort(q_of_p)
    q_axis, q_mass = q_of_p[order], p_mass[order]

    for p_true in (0.2, 0.35, 0.4, 0.55, 0.7):
        q_true = mu / (n * p_true)
        assert weighted_cdf_pit(q_true, q_axis, q_mass) == pytest.approx(
            1.0 - weighted_cdf_pit(p_true, p_axis, p_mass), abs=1e-3
        )


def test_rejects_degenerate_weights():
    axis = np.geomspace(1.0, 10.0, 16)
    with pytest.raises(ValueError):
        weighted_cdf_pit(5.0, axis, np.zeros_like(axis))


def test_q_marginal_is_invariant_to_r_axis_transport():
    """Nearest-bin vs linear transport cannot move the quantal-size marginal.

    In `change_of_variables_and_marginalise` the r-axis assignment only
    decides WHICH r bin each (q, v) cell's mass lands in; the quantal-size
    marginal is `joint_mass.sum(axis=(1, 2))`, which sums that axis away. So
    the two transport modes must give bit-identical q marginals, and the
    nearest-bin transport cannot be a cause of any q rank failure. This is
    asserted here rather than tested empirically with a 200-draw SBC arm.
    """
    from src.bqa import ConditionData, change_of_variables_and_marginalise
    from src.grid import build_grid
    from src.simulate_q import simulate_from_q_model

    rng = np.random.default_rng(7)
    n, p, q, v = 6, 0.35, 100.0, 0.30
    gamma = 1.0 / v**2
    amps = simulate_from_q_model(n, p, gamma, q / gamma, 25.0**2, 60, rng)
    conds = [ConditionData(mu=n * p * q, eps2=25.0**2, amplitudes=amps)]
    candidates = [4, 5, 6, 7, 8]
    grid = build_grid(np.array(candidates), p_resolution=64, v_resolution=64)

    kwargs = dict(combine="joint", q_prior="induced")
    near = change_of_variables_and_marginalise(
        conds, grid, candidates, interpolation="nearest", **kwargs
    )
    lin = change_of_variables_and_marginalise(
        conds, grid, candidates, interpolation="linear", **kwargs
    )
    assert np.allclose(near["q_marginal"], lin["q_marginal"], rtol=0, atol=1e-15)
    assert np.allclose(near["q_axis"], lin["q_axis"])
    # The r marginal, by contrast, IS transport-dependent.
    assert not np.allclose(near["r_marginal"], lin["r_marginal"])
