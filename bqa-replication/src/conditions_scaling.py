"""Controlled conditions-scaling SBC experiment (diagnostic for §4.1), resumable.

Isolates the combine-then-multiply miscalibration by holding the release
probabilities of the conditions FIXED and well-separated, drawing the shared
(q, v, n) from the priors, and asking how the calibration of the quantal-size
posterior degrades as conditions are ADDED. Under a correct combination adding
a genuine same-q condition should tighten the posterior only as much as the
extra data warrant; the product combination instead re-applies the prior once
per condition and over-sharpens, and that over-sharpening should grow with the
number of conditions (and vanish for a single condition).

For each combine rule and each condition count K it reports:
  - ``ks_p``: KS p-value of the q rank statistics against Uniform(0, 1)
    (``> 0.05`` = calibrated intervals).
  - ``overconf``: ratio of the realised error spread to the stated posterior
    width, ``std(q_hat - q_true) / mean(posterior sd of q)``; 1.0 is honest,
    larger is over-confident.

Resumable: each (K, mode, iteration) result is appended to a JSONL checkpoint
on disk, so the run can be chunked across many short invocations. This is a
diagnostic harness added by the replication, not from the paper; it hardcodes
no paper numbers. Usage:

    python -m src.conditions_scaling batch [modes] [budget_s]
    python -m src.conditions_scaling finalize
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
from src.sbc import SBC_EPS2, sample_q_prior, weighted_cdf_pit
from src.simulate_q import simulate_from_q_model

CAL_DIR = "results/calibration"
CKPT = os.path.join(CAL_DIR, "_ckpt", "conditions_scaling.jsonl")
P_SETS = {1: [0.5], 2: [0.3, 0.6], 3: [0.2, 0.4, 0.6]}
N_CANDIDATES = [4, 5, 6, 7, 8]
N_OBS = 40
N_ITERS = 120
ALL_MODES = ["product", "loglik", "density", "joint"]


def _posterior_sd(axis, marginal):
    w = marginal / marginal.sum()
    m = (axis * w).sum()
    return float(np.sqrt(((axis - m) ** 2 * w).sum()))


def _one(args):
    k, mode, it, base_seed = args
    p_set = P_SETS[k]
    grid = build_grid(np.array(N_CANDIDATES))
    # Seed depends only on (k, it), NOT the mode, so every combine rule sees
    # the SAME simulated datasets and the comparison is within-draw fair.
    rng = np.random.default_rng([base_seed, k, it])
    n_true = int(rng.choice(N_CANDIDATES))
    v_true = float(np.exp(rng.uniform(np.log(0.05), np.log(1.0))))
    q_true = sample_q_prior(rng)
    gamma_true = 1.0 / v_true**2
    lam_true = q_true / gamma_true
    conds = [
        ConditionData(
            mu=n_true * p * q_true, eps2=SBC_EPS2,
            amplitudes=simulate_from_q_model(
                n_true, p, gamma_true, lam_true, SBC_EPS2, N_OBS, rng),
        )
        for p in p_set
    ]
    post = change_of_variables_and_marginalise(conds, grid, N_CANDIDATES, combine=mode)
    return {
        "k": k, "mode": mode, "it": it,
        "rank": weighted_cdf_pit(q_true, post["q_axis"], post["q_marginal"]),
        "err": post["q_hat"] - q_true,
        "psd": _posterior_sd(post["q_axis"], post["q_marginal"]),
    }


def _load_done():
    done = set()
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["k"], r["mode"], r["it"]))
    return done


def batch(modes=("joint",), budget_s=38.0, base_seed=101, procs=4):
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    done = _load_done()
    jobs = [
        (k, m, it, base_seed)
        for k in sorted(P_SETS) for m in modes for it in range(N_ITERS)
        if (k, m, it) not in done
    ]
    if not jobs:
        print(f"[scaling] modes={list(modes)} all done")
        return 0
    t0 = time.time()
    processed = 0
    with Pool(procs) as pool, open(CKPT, "a") as f:
        for s in range(0, len(jobs), procs):
            if time.time() - t0 > budget_s:
                break
            for r in pool.map(_one, jobs[s:s + procs]):
                f.write(json.dumps(r) + "\n")
                processed += 1
            f.flush()
    print(f"[scaling] processed {processed}, remaining {len(jobs) - processed}")
    return len(jobs) - processed


def finalize():
    rows_ckpt = [json.loads(l) for l in open(CKPT) if l.strip()]
    modes_present = sorted({r["mode"] for r in rows_ckpt}, key=ALL_MODES.index)
    out = []
    for k in sorted(P_SETS):
        row = {"n_conditions": k}
        for m in modes_present:
            sub = [r for r in rows_ckpt if r["k"] == k and r["mode"] == m]
            if len(sub) < N_ITERS:
                print(f"  incomplete: k={k} mode={m} ({len(sub)}/{N_ITERS})")
            ranks = np.clip([r["rank"] for r in sub], 0, 1)
            ks = stats.kstest(ranks, "uniform").pvalue
            mean_psd = float(np.mean([r["psd"] for r in sub]))
            overconf = float(np.std([r["err"] for r in sub]) / mean_psd)
            row[f"{m}_ks_p"] = round(float(ks), 4)
            row[f"{m}_overconf"] = round(overconf, 2)
            row[f"{m}_meansd"] = round(mean_psd, 1)
        out.append(row)
        print(row)
    path = os.path.join(CAL_DIR, "sbc_conditions_scaling_joint.csv")
    fns = ["n_conditions"] + [f"{m}_{s}" for m in modes_present
                              for s in ("ks_p", "overconf", "meansd")]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "batch"
    if cmd == "finalize":
        finalize()
    else:
        modes = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("joint",)
        budget = float(sys.argv[3]) if len(sys.argv) > 3 else 38.0
        batch(modes=modes, budget_s=budget)
