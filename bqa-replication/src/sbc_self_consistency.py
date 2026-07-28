"""Self-consistency check for the SBC harness itself (K = 1, exact prior).

Why this exists
---------------
Every SBC run in this replication reports the quantal-size rank statistic as
non-uniform. Before that can be read as a statement about BQA, the harness has
to be shown to be self-consistent, because SBC ranks are non-uniform whenever
the generator's prior and the inference's prior disagree -- regardless of
whether the inference code is correct.

For a SINGLE condition (K = 1) the correct posterior under this harness's own
generative model is available in closed form, so there is a configuration in
which no prior mismatch is possible. The harness draws

    p ~ arcsine on [P_MIN, P_MAX],  q ~ log-uniform,  v ~ log-uniform,
    n ~ uniform,   then sets  mu = n p q

and passes (mu, amplitudes) to the inference. Treating mu as exactly known
imposes the constraint p = mu / (n q), so changing variables p -> mu at fixed
(q, n) contributes a Jacobian (n q)^-1, and the exact posterior density in the
log-q coordinate the grid uses is

    log w(q, n) = -log n - log q - 0.5 log( p(q) (1 - p(q)) ) + log L(y | .)

which is exactly what `bqa._joint_log_q_prior(..., q_prior="induced")` returns
at K = 1, evaluated on a single joint (q, v) grid with the likelihood applied
once. So `combine="joint", q_prior="induced", K=1` is a configuration where
generator and inference provably share the same prior.

Reading the result
------------------
- If the q rank statistic is UNIFORM here, the harness is validated and the
  multi-condition q failures are attributable to the combination rule and/or
  the shared-q prior choice, as the paper argues.
- If the q rank statistic FAILS here, no prior choice and no combination rule
  can be the cause, and the defect is localized to the grid/transport/PIT
  machinery (or to conditioning on mu at all). The paper's headline claim must
  then be stated as a property of this implementation, not of BQA.

Run (checkpointed, resumable):
    python -m src.sbc_self_consistency batch [q_prior] [budget_s]
    python -m src.sbc_self_consistency finalize
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.bqa import ConditionData, change_of_variables_and_marginalise
from src.grid import build_grid
from src.sbc import (
    SBC_EPS2,
    jittered_rank_from_pmf,
    sample_prior,
    sbc_uniformity_test,
    SBCDraw,
    weighted_cdf_pit,
)
from src.simulate_q import simulate_from_q_model

CAL_DIR = "results/calibration"
CKPT_DIR = os.path.join(CAL_DIR, "_ckpt")
N_CANDIDATES = [4, 5, 6, 7, 8]
N_OBS = 40
N_ITERATIONS = 200
BASE_SEED = 1
RES = int(os.environ.get("BQA_SBC_RES", "128"))
Q_PRIORS = ("induced", "geometric", "flat")


def _one_iteration(args):
    """One K = 1 SBC iteration. Single condition, so no combination step runs."""
    i, q_prior = args
    rng = np.random.default_rng([BASE_SEED, i])
    theta = sample_prior(N_CANDIDATES, rng)
    n_true, p1, v_true, q_true = theta["n"], theta["p1"], theta["v"], theta["q"]
    gamma_true = 1.0 / v_true**2
    lam_true = q_true / gamma_true

    amps = simulate_from_q_model(
        n_true, p1, gamma_true, lam_true, SBC_EPS2, N_OBS, rng
    )
    conds = [ConditionData(mu=n_true * p1 * q_true, eps2=SBC_EPS2, amplitudes=amps)]

    grid = build_grid(np.array(N_CANDIDATES), p_resolution=RES, v_resolution=RES)
    post = change_of_variables_and_marginalise(
        conds, grid, N_CANDIDATES, combine="joint", q_prior=q_prior
    )

    gamma_order = np.argsort(post["gamma_values"])
    return {
        "iteration": i,
        "q_rank": weighted_cdf_pit(q_true, post["q_axis"], post["q_marginal"]),
        "r_rank": weighted_cdf_pit(
            n_true * q_true, post["r_axis"], post["r_marginal"]
        ),
        "gamma_rank": weighted_cdf_pit(
            gamma_true,
            post["gamma_values"][gamma_order],
            post["gamma_marginal"][gamma_order],
        ),
        "n_rank": jittered_rank_from_pmf(
            n_true, post["n_candidates"], post["n_marginal"], rng
        ),
        "q_true": q_true,
        "q_hat": post["q_hat"],
    }


def _tag(q_prior):
    return f"selfconsist_K1_{q_prior}" + (f"_res{RES}" if RES != 128 else "")


def _ckpt_path(q_prior):
    return os.path.join(CKPT_DIR, f"{_tag(q_prior)}.jsonl")


def _load_done(q_prior):
    path = _ckpt_path(q_prior)
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done[r["iteration"]] = r
    return done


def batch(q_prior="induced", budget_s=38.0, procs=4):
    """Run as many missing iterations as fit in the time budget, then checkpoint."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    done = _load_done(q_prior)
    remaining = [i for i in range(N_ITERATIONS) if i not in done]
    if not remaining:
        print(f"[{_tag(q_prior)}] all {N_ITERATIONS} done")
        return
    t0 = time.time()
    path = _ckpt_path(q_prior)
    with ProcessPoolExecutor(max_workers=procs) as ex, open(path, "a") as f:
        for res in ex.map(_one_iteration, [(i, q_prior) for i in remaining]):
            f.write(json.dumps(res) + "\n")
            f.flush()
            if time.time() - t0 > budget_s:
                break
    left = N_ITERATIONS - len(_load_done(q_prior))
    print(f"[{_tag(q_prior)}] {left} iterations remaining")


def finalize(q_priors=Q_PRIORS):
    """Write the self-consistency table from whatever checkpoints exist."""
    rows = []
    for q_prior in q_priors:
        done = _load_done(q_prior)
        if not done:
            continue
        draws = [
            SBCDraw(
                theta_true={"q": r["q_true"]},
                ranks={
                    "q": r["q_rank"],
                    "r": r["r_rank"],
                    "gamma": r["gamma_rank"],
                    "n": r["n_rank"],
                },
            )
            for r in done.values()
        ]
        report = sbc_uniformity_test(draws)
        ratios = [r["q_hat"] / r["q_true"] for r in done.values()]
        rows.append(
            {
                "q_prior": q_prior,
                "n_conditions": 1,
                "n_draws": len(draws),
                "q_ks_p": report["q"]["p_value"],
                "r_ks_p": report["r"]["p_value"],
                "gamma_ks_p": report["gamma"]["p_value"],
                "n_ks_p": report["n"]["p_value"],
                "median_qhat_over_qtrue": float(np.median(ratios)),
            }
        )
    df = pd.DataFrame(rows)
    os.makedirs(CAL_DIR, exist_ok=True)
    out = os.path.join(CAL_DIR, "sbc_self_consistency_k1.csv")
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"wrote {out}")
    return df


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "batch"
    if cmd == "finalize":
        finalize()
    else:
        qp = sys.argv[2] if len(sys.argv) > 2 else "induced"
        budget = float(sys.argv[3]) if len(sys.argv) > 3 else 38.0
        batch(qp, budget)
