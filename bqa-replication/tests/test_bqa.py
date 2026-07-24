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


def test_joint_grid_normalizes_to_one():
    """The joint-grid posterior must conserve mass (normalise to 1)."""
    rng = np.random.default_rng(1)
    n_true, gamma_true, lam_true, eps2 = 6, 11.1, 9.0, 625.0
    q_true = gamma_true * lam_true
    conds = [
        ConditionData(
            mu=n_true * p * q_true,
            eps2=eps2,
            amplitudes=simulate_from_q_model(n_true, p, gamma_true, lam_true, eps2, 40, rng),
        )
        for p in (0.2, 0.4, 0.6)
    ]
    grid = build_grid(np.array([4, 5, 6, 7, 8]))
    for q_prior in ("geometric", "reference"):
        post = change_of_variables_and_marginalise(
            conds, grid, [4, 5, 6, 7, 8], combine="joint", q_prior=q_prior
        )
        assert np.isclose(post["joint_mass"].sum(), 1.0, atol=1e-9)


def test_joint_matches_product_single_condition_point_estimate():
    """On single-condition data the joint and product point estimates agree.

    A single condition is already calibrated (no cross-condition
    multiplication), so the joint grid -- which places the same induced
    arcsine prior on q -- must recover essentially the same q_hat.
    """
    rng = np.random.default_rng(3)
    n_true, gamma_true, lam_true, eps2 = 6, 11.1, 9.0, 625.0
    q_true = gamma_true * lam_true
    grid = build_grid(np.array([4, 5, 6, 7, 8]))

    def cond():
        return [
            ConditionData(
                mu=n_true * 0.3 * q_true,
                eps2=eps2,
                amplitudes=simulate_from_q_model(
                    n_true, 0.3, gamma_true, lam_true, eps2, 60,
                    np.random.default_rng(7),
                ),
            )
        ]

    q_prod = change_of_variables_and_marginalise(
        cond(), grid, [4, 5, 6, 7, 8], combine="product"
    )["q_hat"]
    q_joint = change_of_variables_and_marginalise(
        cond(), grid, [4, 5, 6, 7, 8], combine="joint"
    )["q_hat"]
    # Same q-axis + same implied arcsine prior => same median bin.
    assert abs(q_prod - q_joint) < 0.05 * q_true


def test_joint_does_not_oversharpen_like_product():
    """Adding genuine same-q conditions must not collapse the joint posterior
    the way the product combination does (the SBC miscalibration cause)."""
    rng = np.random.default_rng(3)
    n, q, v = 6, 100.0, 0.3
    gamma, eps2, lam = 1.0 / v**2, 25.0**2, q / (1.0 / v**2)
    grid = build_grid(np.array([4, 5, 6, 7, 8]))

    def q_sd(probs, combine):
        conds = [
            ConditionData(
                mu=n * p * q,
                eps2=eps2,
                amplitudes=simulate_from_q_model(n, p, gamma, lam, eps2, 60, rng),
            )
            for p in probs
        ]
        post = change_of_variables_and_marginalise(
            conds, grid, [4, 5, 6, 7, 8], combine=combine
        )
        a, w = post["q_axis"], post["q_marginal"] / post["q_marginal"].sum()
        m = (a * w).sum()
        return float(np.sqrt(((a - m) ** 2 * w).sum()))

    sd1_prod = q_sd([0.3], "product")
    sd3_prod = q_sd([0.2, 0.4, 0.6], "product")
    sd3_joint = q_sd([0.2, 0.4, 0.6], "joint")
    # Product over-sharpens hard; joint retains substantially more width.
    assert sd3_prod < 0.6 * sd1_prod
    assert sd3_joint > sd3_prod


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
