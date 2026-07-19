import numpy as np
from scipy import stats

from src.bqa import (
    ConditionData,
    change_of_variables_and_marginalise,
    estimate_n,
    heterogeneous_release_pmf,
)
from src.grid import build_grid
from src.simulate_q import simulate_from_q_model


def test_estimate_n_basic():
    assert estimate_n(median_r=600.0, median_q=100.0) == 6.0


def test_estimate_n_rejects_zero_q():
    try:
        estimate_n(median_r=600.0, median_q=0.0)
        assert False, "expected ValueError for median_q == 0"
    except ValueError:
        pass


def test_bqa_recovers_approximate_parameters_on_matched_data():
    """Model-matched data (simulate_from_q_model) should be recovered within
    coarse tolerance by the BQA grid-inference pipeline. This is a sanity
    check on inference machinery, not a reproduction of the paper's
    published numeric results (none are hardcoded here)."""
    rng = np.random.default_rng(42)
    n_true, gamma_true, lam_true, eps2 = 6, 11.1, 9.0, 625.0
    q_true = gamma_true * lam_true

    conditions = []
    for p_true in (0.1, 0.3):
        amps = simulate_from_q_model(n_true, p_true, gamma_true, lam_true, eps2, 60, rng)
        mu = n_true * p_true * q_true
        conditions.append(ConditionData(mu=mu, eps2=eps2, amplitudes=amps))

    grid = build_grid(n_values=np.array([n_true]))
    result = change_of_variables_and_marginalise(
        conditions, grid, n_candidates=[4, 5, 6, 7, 8]
    )

    assert abs(result["n_hat"] - n_true) < 2.0
    assert 0.5 * q_true < result["q_hat"] < 1.5 * q_true
    assert np.isclose(result["joint_mass"].sum(), 1.0)


def test_heterogeneous_release_pmf_sums_to_one():
    pmf = heterogeneous_release_pmf(n=6, mean_p=0.35, alpha=5.0)
    assert pmf.shape == (7,)
    assert np.isclose(pmf.sum(), 1.0)


def test_heterogeneous_release_pmf_converges_to_binomial_at_high_alpha():
    """Very large alpha means near-homogeneous release probabilities across
    sites (see Eq. A13: beta_k = alpha*(1-mean_p)/mean_p grows with alpha,
    concentrating the beta distribution tightly around mean_p), so the
    heterogeneous pmf should approach the plain binomial pmf."""
    n, mean_p = 6, 0.35
    pmf_hetero = heterogeneous_release_pmf(n=n, mean_p=mean_p, alpha=1e6)
    pmf_binom = stats.binom.pmf(np.arange(n + 1), n, mean_p)
    np.testing.assert_allclose(pmf_hetero, pmf_binom, atol=1e-3)


def test_heterogeneous_release_pmf_rejects_large_n():
    try:
        heterogeneous_release_pmf(n=25, mean_p=0.35, alpha=5.0)
        assert False, "expected ValueError for intractably large n"
    except ValueError:
        pass
