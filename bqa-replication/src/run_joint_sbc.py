"""Full 200-iteration SBC for the joint-grid inference path (§4.1), resumable.

Mirrors ``src.sbc.run_sbc_reproducible`` (independent RNG per iteration, so the
run is deterministic and resumable) but routes inference through
``change_of_variables_and_marginalise(..., combine="joint")`` and records
q_hat/q_true per draw so unbiasedness can be checked alongside calibration.

Each invocation processes a wall-clock BATCH of the still-missing iterations
(farmed across processes) and appends them to a JSONL checkpoint on disk, then
exits reporting how many remain. Call repeatedly until ``remaining=0``; the run
is deterministic regardless of how the iterations are chunked. Finalise with
``--finalize`` to write the report/results CSVs. Nothing hardcodes paper
numbers. Usage:

    python -m src.run_joint_sbc [combine] [q_prior] [batch_seconds]
    python -m src.run_joint_sbc [combine] [q_prior] finalize
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy import stats

from src.bqa import ConditionData, change_of_variables_and_marginalise
from src.grid import build_grid
from src.sbc import (
    SBC_EPS2,
    jittered_rank_from_pmf,
    sample_prior,
    weighted_cdf_pit,
)
from src.simulate_q import simulate_from_q_model

CAL_DIR = "results/calibration"
CKPT_DIR = os.path.join(CAL_DIR, "_ckpt")
N_CANDIDATES = [4, 5, 6, 7, 8]
N_OBS = 40
N_ITERATIONS = 200
BASE_SEED = 1
RES = int(os.environ.get("BQA_SBC_RES", "128"))  # grid resolution knob


def _one_iteration(args):
    i, combine, q_prior = args
    rng = np.random.default_rng([BASE_SEED, i])
    theta = sample_prior(N_CANDIDATES, rng)
    n_true, p1, p2, v_true, q_true = (
        theta["n"], theta["p1"], theta["p2"], theta["v"], theta["q"]
    )
    gamma_true = 1.0 / v_true**2
    lam_true = q_true / gamma_true

    conds = []
    for p in (p1, p2):
        amps = simulate_from_q_model(
            n_true, p, gamma_true, lam_true, SBC_EPS2, N_OBS, rng
        )
        conds.append(ConditionData(mu=n_true * p * q_true, eps2=SBC_EPS2, amplitudes=amps))

    grid = build_grid(np.array(N_CANDIDATES), p_resolution=RES, v_resolution=RES)
    post = change_of_variables_and_marginalise(
        conds, grid, N_CANDIDATES, combine=combine, q_prior=q_prior
    )

    r_true = n_true * q_true
    gamma_order = np.argsort(post["gamma_values"])
    return {
        "iteration": i,
        "q_rank": weighted_cdf_pit(q_true, post["q_axis"], post["q_marginal"]),
        "r_rank": weighted_cdf_pit(r_true, post["r_axis"], post["r_marginal"]),
        "gamma_rank": weighted_cdf_pit(
            gamma_true, post["gamma_values"][gamma_order],
            post["gamma_marginal"][gamma_order]),
        "n_rank": jittered_rank_from_pmf(
            n_true, post["n_candidates"], post["n_marginal"], rng),
        "q_true": q_true, "q_hat": post["q_hat"],
    }


def _ckpt_path(tag):
    return os.path.join(CKPT_DIR, f"sbc_{tag}.jsonl")


def _load_done(tag):
    path = _ckpt_path(tag)
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done[r["iteration"]] = r
    return done


def batch(combine="joint", q_prior="geometric", budget_s=38.0, procs=4):
    tag = f"{combine}_{q_prior}" + (f"_res{RES}" if RES != 128 else "")
    os.makedirs(CKPT_DIR, exist_ok=True)
    done = _load_done(tag)
    remaining = [i for i in range(N_ITERATIONS) if i not in done]
    if not remaining:
        print(f"[{tag}] all {N_ITERATIONS} done")
        return 0
    t0 = time.time()
    path = _ckpt_path(tag)
    processed = 0
    with Pool(procs) as pool, open(path, "a") as f:
        for start in range(0, len(remaining), procs):
            if time.time() - t0 > budget_s:
                break
            chunk = remaining[start:start + procs]
            for r in pool.map(_one_iteration, [(i, combine, q_prior) for i in chunk]):
                f.write(json.dumps(r) + "\n")
                processed += 1
            f.flush()
    left = N_ITERATIONS - len(_load_done(tag))
    print(f"[{tag}] processed {processed} this batch, remaining {left}")
    return left


def finalize(combine="joint", q_prior="geometric"):
    tag = f"{combine}_{q_prior}" + (f"_res{RES}" if RES != 128 else "")
    done = _load_done(tag)
    rows = [done[i] for i in sorted(done)]
    if len(rows) < N_ITERATIONS:
        print(f"[{tag}] only {len(rows)}/{N_ITERATIONS} done; run more batches")
        return None
    report = {}
    for name in ("q", "r", "gamma", "n"):
        ranks = np.clip([r[f"{name}_rank"] for r in rows], 0.0, 1.0)
        ks = stats.kstest(ranks, "uniform")
        report[name] = {"ks_statistic": float(ks.statistic),
                        "p_value": float(ks.pvalue),
                        "calibrated": bool(ks.pvalue > 0.05),
                        "n_draws": len(rows)}
    median_ratio = float(np.median([r["q_hat"] / r["q_true"] for r in rows]))
    print(f"=== joint SBC ({tag}, {len(rows)} iters) ===")
    for name, s in report.items():
        print(f"  {name}: KS p={s['p_value']:.4f} "
              f"({'calibrated' if s['calibrated'] else 'NOT'})")
    print(f"  median q_hat/q_true = {median_ratio:.3f}")

    rp = os.path.join(CAL_DIR, f"sbc_report_{tag}.csv")
    with open(rp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["parameter", "ks_statistic", "p_value",
                                          "calibrated", "n_draws"])
        w.writeheader()
        for p, s in report.items():
            w.writerow({"parameter": p, **s})
    rr = os.path.join(CAL_DIR, f"sbc_results_{tag}.csv")
    with open(rr, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iteration", "q_rank", "r_rank",
                                          "gamma_rank", "n_rank", "q_true", "q_hat"])
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {rp}\n  wrote {rr}")
    return report, median_ratio


if __name__ == "__main__":
    combine = sys.argv[1] if len(sys.argv) > 1 else "joint"
    q_prior = sys.argv[2] if len(sys.argv) > 2 else "geometric"
    arg3 = sys.argv[3] if len(sys.argv) > 3 else "38"
    if arg3 == "finalize":
        finalize(combine, q_prior)
    else:
        batch(combine, q_prior, budget_s=float(arg3))
