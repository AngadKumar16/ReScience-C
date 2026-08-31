"""Does conditioning on a data-derived mu explain the interval defect?

Why this exists
---------------
`src/sbc_fixed_n_check.py` localises the quantal-size interval miscalibration
to the per-condition posterior, upstream of every derived-parameter step, and
leaves four candidate causes open. This module tests the one the discussion
names first: the legitimacy of treating mu as exactly known.

In the calibration harness mu is the TRUE mean n*p*q, supplied from outside.
In real use (and in the paper's own procedure) mu is the EMPIRICAL mean of the
same amplitudes that then enter the likelihood. If using the data twice is what
produces the too-narrow posterior, then switching the harness from the true
mean to the sample mean should change the coverage. If the failure is
unchanged, mu conditioning is exonerated and the remaining candidates are the
Eq. 8 likelihood implementation, the PIT code, and the q-axis grid.

Configuration is the tightest one available (K = 1, n fixed to the truth,
`combine="joint"`, `q_prior="induced"`), so nothing between the per-condition
grid and the q marginal can contribute. The two arms share seeds draw for
draw: same theta, same amplitudes, only the mu handed to the inference
differs.

Run:
    python -m src.mu_conditioning_check
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

from src.bqa import ConditionData, change_of_variables_and_marginalise
from src.grid import build_grid
from src.sbc import SBC_EPS2, sample_prior, weighted_cdf_pit
from src.simulate_q import simulate_from_q_model

N_CANDIDATES = [4, 5, 6, 7, 8]  # prior support for the n DRAW only
N_OBS = 40
N_ITERATIONS = 200
BASE_SEED = 1  # same base seed as sbc_fixed_n_check: the true-mu arm reproduces it
RES = int(os.environ.get("BQA_SBC_RES", "128"))

ARMS = ("true_mu", "sample_mu")


def _one_iteration(i: int) -> list[dict]:
    """One SBC draw, inferred twice: once with the true mu, once with the sample mean."""
    rng = np.random.default_rng([BASE_SEED, i])
    theta = sample_prior(N_CANDIDATES, rng)
    n_true, p1, v_true, q_true = theta["n"], theta["p1"], theta["v"], theta["q"]
    gamma_true = 1.0 / v_true**2

    amps = simulate_from_q_model(
        n_true, p1, gamma_true, q_true / gamma_true, SBC_EPS2, N_OBS, rng
    )

    mu_by_arm = {
        "true_mu": float(n_true * p1 * q_true),
        "sample_mu": float(np.mean(amps)),
    }

    candidates = [n_true]  # n fixed to the truth: no marginalisation over n
    grid = build_grid(np.array(candidates), p_resolution=RES, v_resolution=RES)

    rows = []
    for arm in ARMS:
        mu = mu_by_arm[arm]
        if mu <= 0:
            # A sample mean can go non-positive at very low signal; the grid
            # inversion q = mu/(n p) is undefined there. Record and skip.
            rows.append(
                {
                    "arm": arm,
                    "iteration": i,
                    "q_rank": np.nan,
                    "q_true": q_true,
                    "q_hat": np.nan,
                    "mu_used": mu,
                    "mu_true": mu_by_arm["true_mu"],
                    "p1_true": p1,
                    "n_true": n_true,
                }
            )
            continue
        conds = [ConditionData(mu=mu, eps2=SBC_EPS2, amplitudes=amps)]
        post = change_of_variables_and_marginalise(
            conds, grid, candidates, combine="joint", q_prior="induced"
        )
        rows.append(
            {
                "arm": arm,
                "iteration": i,
                "q_rank": weighted_cdf_pit(q_true, post["q_axis"], post["q_marginal"]),
                "q_true": q_true,
                "q_hat": post["q_hat"],
                "mu_used": mu,
                "mu_true": mu_by_arm["true_mu"],
                "p1_true": p1,
                "n_true": n_true,
            }
        )
    return rows


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for arm in ARMS:
        sub = df[(df["arm"] == arm) & df["q_rank"].notna()]
        ranks = sub["q_rank"].to_numpy()
        ratios = (sub["q_hat"] / sub["q_true"]).to_numpy()
        out.append(
            {
                "arm": arm,
                "n_draws": len(ranks),
                "resolution": RES,
                "q_ks_p": stats.kstest(ranks, "uniform").pvalue,
                "mean_pit": float(ranks.mean()),
                "frac_above_p95": float((ranks > 0.95).mean()),
                "frac_below_p05": float((ranks < 0.05).mean()),
                "coverage_90": float(((ranks > 0.05) & (ranks < 0.95)).mean()),
                "median_qhat_over_qtrue": float(np.median(ratios)),
            }
        )
    return pd.DataFrame(out)


def run(n_iterations: int = N_ITERATIONS, procs: int = 4) -> pd.DataFrame:
    with ProcessPoolExecutor(max_workers=procs) as ex:
        nested = list(ex.map(_one_iteration, range(n_iterations)))
    df = pd.DataFrame([row for rows in nested for row in rows])
    summary = summarise(df)
    os.makedirs("results/calibration", exist_ok=True)
    df.to_csv("results/calibration/sbc_mu_conditioning_ranks.csv", index=False)
    summary.to_csv("results/calibration/sbc_mu_conditioning_summary.csv", index=False)
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run()
