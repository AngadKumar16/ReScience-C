"""K = 1 with n fixed to the truth: the tightest possible SBC configuration.

Why this exists
---------------
`src/sbc_self_consistency.py` shows the q rank statistic fails at K = 1 even
with the exactly-correct `induced` prior, which rules out the condition-
combination rule and the shared-q prior as causes. This module removes the one
remaining piece of machinery between the per-condition grid and the q marginal:
marginalisation over the candidate n.

The argument is that the probability-integral transform is invariant under
monotone reparametrisation. At fixed mu and fixed n, the change of variables

    q = mu / (n p)

is strictly monotone in p, so the q rank MUST equal the p rank exactly. There
is no room for the transport-onto-a-shared-log-q-axis step to introduce
anything. Marginalising over several candidate n is what breaks the
equivalence, because it mixes a different monotone map per candidate.

So:
- If q is uniform here but fails once n is marginalised, the defect is the
  scatter-onto-shared-q-axis / sum-over-n step.
- If q fails HERE, the defect is upstream of every derived-parameter
  operation: the per-condition posterior over p on the (arcsin sqrt p, log v)
  grid is itself miscalibrated, which is part of the core reproduction rather
  than any of the beyond-the-paper bookkeeping.

Observed (200 draws, RES = 128): KS p = 1.2e-8, mean PIT 0.641, 81% coverage of
a nominal 90% interval, 17.0% of truths above the posterior's 95th percentile
against 2.0% below. The second case holds.

Run:
    python -m src.sbc_fixed_n_check
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
BASE_SEED = 1
RES = int(os.environ.get("BQA_SBC_RES", "128"))


def _one_iteration(i: int) -> dict:
    rng = np.random.default_rng([BASE_SEED, i])
    theta = sample_prior(N_CANDIDATES, rng)
    n_true, p1, v_true, q_true = theta["n"], theta["p1"], theta["v"], theta["q"]
    gamma_true = 1.0 / v_true**2

    amps = simulate_from_q_model(
        n_true, p1, gamma_true, q_true / gamma_true, SBC_EPS2, N_OBS, rng
    )
    conds = [ConditionData(mu=n_true * p1 * q_true, eps2=SBC_EPS2, amplitudes=amps)]

    candidates = [n_true]  # n fixed to the truth: no marginalisation over n
    grid = build_grid(np.array(candidates), p_resolution=RES, v_resolution=RES)
    post = change_of_variables_and_marginalise(
        conds, grid, candidates, combine="joint", q_prior="induced"
    )
    return {
        "iteration": i,
        "q_rank": weighted_cdf_pit(q_true, post["q_axis"], post["q_marginal"]),
        "q_true": q_true,
        "q_hat": post["q_hat"],
        "p1_true": p1,
        "n_true": n_true,
    }


def run(n_iterations: int = N_ITERATIONS, procs: int = 4) -> pd.DataFrame:
    with ProcessPoolExecutor(max_workers=procs) as ex:
        rows = list(ex.map(_one_iteration, range(n_iterations)))
    df = pd.DataFrame(rows)
    ranks = df["q_rank"].to_numpy()
    ratios = (df["q_hat"] / df["q_true"]).to_numpy()
    summary = pd.DataFrame(
        [
            {
                "n_draws": len(ranks),
                "resolution": RES,
                "q_ks_p": stats.kstest(ranks, "uniform").pvalue,
                "mean_pit": ranks.mean(),
                "frac_above_p95": (ranks > 0.95).mean(),
                "frac_below_p05": (ranks < 0.05).mean(),
                "coverage_90": ((ranks > 0.05) & (ranks < 0.95)).mean(),
                "median_qhat_over_qtrue": float(np.median(ratios)),
            }
        ]
    )
    os.makedirs("results/calibration", exist_ok=True)
    df.to_csv("results/calibration/sbc_fixed_n_ranks.csv", index=False)
    summary.to_csv("results/calibration/sbc_fixed_n_summary.csv", index=False)
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    run()
