"""Tests for the MPFA baseline and the BQA-vs-MPFA comparison."""

from __future__ import annotations

import numpy as np

from src.mpfa import compare_bqa_vs_mpfa, fit_mpfa


def test_fit_mpfa_recovers_parabola_exactly():
    """On noiseless parabolic data the fit must recover q and N exactly."""
    q, n = 100.0, 6.0
    means = np.array([50.0, 120.0, 200.0, 300.0, 400.0])
    variances = q * means - means**2 / n  # CV = 0 case
    res = fit_mpfa(means, variances, cv_known=0.0)
    assert np.isclose(res.q_hat_apparent, q, rtol=1e-6)
    assert np.isclose(res.n_hat, n, rtol=1e-6)


def test_fit_mpfa_needs_three_conditions():
    try:
        fit_mpfa(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_bqa_beats_mpfa_on_quantal_size():
    """BQA should estimate q more accurately than uncorrected MPFA.

    MPFA's initial slope is inflated by the quantal variability (factor
    1 + CV^2); BQA estimates that variability jointly, so its q estimate
    should land closer to the truth on the same data.
    """
    res = compare_bqa_vs_mpfa(rng=np.random.default_rng(0))
    q_true = res["q_true"]
    bqa_err = abs(res["bqa_q_hat"] - q_true)
    mpfa_err = abs(res["mpfa_q_apparent"] - q_true)
    assert bqa_err < mpfa_err
    # Both methods should still be in the right ballpark for n.
    assert 4.0 <= res["bqa_n_hat"] <= 8.0
    assert 3.0 <= res["mpfa_n_hat"] <= 9.0
