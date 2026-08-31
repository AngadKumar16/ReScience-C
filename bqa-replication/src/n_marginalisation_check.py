"""Where does the residual quantal-size miscalibration live: the sum over n?

Why this exists
---------------
After the `weighted_cdf_pit` fix (see PIT_DEFECT.md) the localisation flipped.
The per-condition posterior over release probability is calibrated: at K = 1
with n fixed to the truth the quantal-size rank statistic passes
(`src/sbc_fixed_n_check.py`, KS p = 0.107). The same configuration with n
marginalised still fails (`src/sbc_self_consistency.py`, KS p = 0.0038). That
is exactly the branch `sbc_fixed_n_check` pre-registered:

    "If q is uniform here but fails once n is marginalised, the defect is the
    scatter-onto-shared-q-axis / sum-over-n step."

This module turns that binary into a dose-response. It holds everything else
fixed (K = 1, `combine="joint"`, `q_prior="induced"`, identical draws and
identical amplitudes) and varies only how many candidate n the inference
marginalises over: just the truth, the truth +/- 1, the truth +/- 2. If the
sum over n is responsible, the failure should grow with the width of the
candidate set. Two further arms anchor and control the sweep:

  - `fullset`  -- the fixed candidate set [4..8] the rest of the calibration
    harness uses, so the sweep connects to the failing K = 1 run in
    `src/sbc_self_consistency.py` rather than living in its own world.
  - `res256`  -- the widest truth-centred set at double grid resolution. The
    fixed-n configuration has 35% of draws whose posterior is narrower than
    one grid cell, so discretisation is a live confound; if the failure is
    discretisation it must ease off here.

The r-axis transport (`interpolation`) is deliberately NOT swept. In the joint
path the quantal-size marginal is ``joint_mass.sum(axis=(1, 2))``, which sums
over the r axis, so nearest-bin and linear transport give bit-identical q
marginals by construction. `tests/test_pit.py` asserts this rather than
spending a 200-draw arm on it.

Run:
    python -m src.n_marginalisation_check
"""

from __future__ import annotations

import json
import os
import sys
import time
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
BASE_SEED = 1  # shared with sbc_fixed_n_check: the width-0 arm reproduces it

RESULTS_DIR = "results/calibration"
CKPT = "results/calibration/_ckpt/n_marginalisation.jsonl"

# (arm name, half-width of the truth-centred candidate set or None for the
#  fixed [4..8] set, resolution)
ARMS = (
    ("width0", 0, 128),
    ("width1", 1, 128),
    ("width2", 2, 128),
    ("fullset", None, 128),
    ("width2_res256", 2, 256),
)


def _candidates_for(half_width: int | None, n_true: int) -> list[int]:
    if half_width is None:
        return list(N_CANDIDATES)
    return [c for c in range(n_true - half_width, n_true + half_width + 1) if c >= 1]


def _one_iteration(i: int) -> list[dict]:
    rng = np.random.default_rng([BASE_SEED, i])
    theta = sample_prior(N_CANDIDATES, rng)
    n_true, p1, v_true, q_true = theta["n"], theta["p1"], theta["v"], theta["q"]
    gamma_true = 1.0 / v_true**2

    amps = simulate_from_q_model(
        n_true, p1, gamma_true, q_true / gamma_true, SBC_EPS2, N_OBS, rng
    )
    conds = [ConditionData(mu=float(n_true * p1 * q_true), eps2=SBC_EPS2, amplitudes=amps)]

    rows = []
    for arm, half_width, res in ARMS:
        candidates = _candidates_for(half_width, n_true)
        grid = build_grid(np.array(candidates), p_resolution=res, v_resolution=res)
        post = change_of_variables_and_marginalise(
            conds, grid, candidates, combine="joint", q_prior="induced"
        )
        rows.append(
            {
                "arm": arm,
                "iteration": i,
                "n_candidates": len(candidates),
                "resolution": res,
                "q_rank": weighted_cdf_pit(q_true, post["q_axis"], post["q_marginal"]),
                "q_true": q_true,
                "q_hat": post["q_hat"],
                "p1_true": p1,
                "n_true": n_true,
            }
        )
    return rows


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for arm, half_width, res in ARMS:
        sub = df[df["arm"] == arm]
        if sub.empty:
            continue
        ranks = sub["q_rank"].to_numpy()
        ratios = (sub["q_hat"] / sub["q_true"]).to_numpy()
        out.append(
            {
                "arm": arm,
                "candidate_half_width": -1 if half_width is None else half_width,
                "resolution": res,
                "n_draws": len(ranks),
                "q_ks_p": stats.kstest(ranks, "uniform").pvalue,
                "mean_pit": float(ranks.mean()),
                "frac_above_p95": float((ranks > 0.95).mean()),
                "frac_below_p05": float((ranks < 0.05).mean()),
                "coverage_90": float(((ranks > 0.05) & (ranks < 0.95)).mean()),
                "median_qhat_over_qtrue": float(np.median(ratios)),
            }
        )
    return pd.DataFrame(out)


def _load_done() -> set[int]:
    if not os.path.exists(CKPT):
        return set()
    done = set()
    with open(CKPT) as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["iteration"])
    return done


def _load_checkpoint() -> pd.DataFrame:
    rows = []
    with open(CKPT) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.extend(json.loads(line)["rows"])
    return pd.DataFrame(rows)


def batch(budget_s: float = 150.0, procs: int = 4) -> int:
    """Run as many un-checkpointed iterations as fit in `budget_s`. Resumable.

    Each call appends whole iterations (all arms for one draw) to a JSONL
    checkpoint, so an interrupted run loses at most one batch. Returns the
    number of iterations still outstanding.
    """
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    done = _load_done()
    todo = [i for i in range(N_ITERATIONS) if i not in done]
    if not todo:
        print(f"[n-marg] all {N_ITERATIONS} iterations done")
        return 0
    t0 = time.time()
    processed = 0
    with ProcessPoolExecutor(max_workers=procs) as ex, open(CKPT, "a") as f:
        for s in range(0, len(todo), procs):
            if time.time() - t0 > budget_s:
                break
            chunk = todo[s : s + procs]
            for i, rows in zip(chunk, ex.map(_one_iteration, chunk)):
                f.write(json.dumps({"iteration": i, "rows": rows}) + "\n")
                processed += 1
            f.flush()
    remaining = len(todo) - processed
    print(f"[n-marg] processed {processed}, remaining {remaining}")
    return remaining


def plot(df: pd.DataFrame, summary: pd.DataFrame, path: str) -> str:
    """Rank histograms across the sweep, plus the width-response of the sweep."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = ["width0", "width1", "width2"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, arm in zip(axes[:3], shown):
        ranks = df[df["arm"] == arm]["q_rank"].to_numpy()
        n_cand = int(df[df["arm"] == arm]["n_candidates"].iloc[0])
        ax.hist(ranks, bins=14, range=(0, 1), edgecolor="white")
        ax.axhline(len(ranks) / 14.0, color="red", ls="--", lw=1)
        ks = float(summary.loc[summary["arm"] == arm, "q_ks_p"].iloc[0])
        ax.set_title(f"{n_cand} candidate n  (KS p = {ks:.3g})")
        ax.set_xlabel("PIT rank")
    axes[0].set_ylabel("count")

    sweep = summary[summary["arm"].isin(shown)]
    counts = [
        int(df[df["arm"] == a]["n_candidates"].iloc[0]) for a in sweep["arm"]
    ]
    sds = [df[df["arm"] == a]["q_rank"].std() for a in sweep["arm"]]
    ax = axes[3]
    ax.plot(counts, sds, "o-", label="rank SD")
    ax.axhline(1.0 / np.sqrt(12.0), color="red", ls="--", lw=1, label="uniform (0.289)")
    ax.set_xticks(counts)
    ax.set_xlabel("number of candidate n")
    ax.set_ylabel("SD of PIT rank")
    ax.set_title("ranks concentrate as n is marginalised")
    ax.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def finalize() -> pd.DataFrame:
    df = _load_checkpoint().sort_values(["arm", "iteration"])
    summary = summarise(df)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "sbc_n_marginalisation_ranks.csv"), index=False)
    summary.to_csv(
        os.path.join(RESULTS_DIR, "sbc_n_marginalisation_summary.csv"), index=False
    )
    print(summary.to_string(index=False))
    print("  wrote " + plot(df, summary, "results/figures/n_marginalisation_sweep.png"))
    return summary


def run(budget_s: float = 1e9, procs: int = 4) -> pd.DataFrame:
    while batch(budget_s=budget_s, procs=procs):
        pass
    return finalize()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "finalize":
        finalize()
    elif mode == "batch":
        batch(budget_s=float(sys.argv[2]) if len(sys.argv) > 2 else 150.0)
    else:
        run()
