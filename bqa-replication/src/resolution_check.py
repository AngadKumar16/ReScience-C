"""Grid-resolution bias check.

The inference remaps posterior mass between axes (p -> shared q-axis, and
per-n r = n*q -> shared r-axis) with nearest-bin, mass-conserving histogram
assignment rather than smooth interpolation. That is exactly mass-conserving
but blurs the posterior, and the blur can bias the median-based point
estimates. This module quantifies that bias directly: it reruns the same
low signal-to-noise scenario at several grid resolutions and records how the
q and n estimates move as the grid is refined. If the estimates converge
toward the truth as resolution grows, the offset seen at the paper's stated
128 resolution is a resampling artifact rather than a property of the data.

Run: python -m src.resolution_check
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.bqa import ConditionData, change_of_variables_and_marginalise
from src.grid import build_grid
from src.simulate import simulate_responses

# Same scenario as the pathological stress test, so the numbers are directly
# comparable to results/identifiability/pathological_mg_results.csv.
N_SITES = 12
Q_TRUE = 30.0
CV_INTRA = 0.3
BASELINE_SD = 25.0
N_OBS = 100
P1, P2 = 0.15, 0.45
# Candidate set must be wide enough that posterior mass is not clipped at the
# edges. The narrower {8,...,16} used previously truncates: several seeds place
# n_hat within one bin of the lower edge, so their mass is clipped and the
# ensemble mean is pulled artificially toward the truth. Matches
# src.low_snr_seeds.N_CANDIDATES so the two are directly comparable.
N_CANDIDATES = list(range(4, 21))
RESOLUTIONS = [64, 128, 256]
# A single seed is not enough: on the narrow grid, seed 42 alone returned
# n_hat = 12.3, which would have been reported as an accurate recovery. Average
# over the same ensemble src.low_snr_seeds uses.
SEEDS = tuple(range(12))


def run_resolution_bias(
    resolutions: list[int] = RESOLUTIONS,
    seeds: tuple[int, ...] = SEEDS,
    output_path: str = "results/identifiability/resolution_bias.csv",
    summary_path: str = "results/identifiability/resolution_bias_summary.csv",
) -> pd.DataFrame:
    """Rerun the low-SNR scenario at several grid resolutions and tabulate bias.

    For each (resolution, seed) pair a fresh, identically seeded simulator draw
    is used, so within a seed only the grid resolution changes between rows (the
    data are held fixed; just the inference grid is refined). Averaging over
    seeds is what makes the comparison across resolutions readable: a single
    seed's error is dominated by that draw's noise, not by the grid.
    """
    eps2 = BASELINE_SD**2
    rows = []
    done = set()
    if os.path.exists(output_path):
        prev = pd.read_csv(output_path)
        if "seed" in prev.columns:
            rows = prev.to_dict("records")
            done = set((int(r["resolution"]), int(r["seed"])) for r in rows)
    for res in resolutions:
        for seed in seeds:
            if (res, seed) in done:
                continue
            rng = np.random.default_rng(seed)  # identical data every resolution
            conditions = []
            for p_true in (P1, P2):
                sim = simulate_responses(
                    N_SITES, p_true, Q_TRUE, CV_INTRA, BASELINE_SD, N_OBS, rng
                )
                mu_true = N_SITES * p_true * Q_TRUE
                conditions.append(
                    ConditionData(mu=mu_true, eps2=eps2, amplitudes=sim.amplitudes)
                )

            grid = build_grid(np.array(N_CANDIDATES), p_resolution=res, v_resolution=res)
            post = change_of_variables_and_marginalise(conditions, grid, N_CANDIDATES)
            rows.append(
                {
                    "resolution": res,
                    "seed": seed,
                    "q_true": Q_TRUE,
                    "q_hat": post["q_hat"],
                    "q_abs_error": abs(post["q_hat"] - Q_TRUE),
                    "n_true": N_SITES,
                    "n_hat": post["n_hat"],
                    "n_abs_error": abs(post["n_hat"] - N_SITES),
                    "v_hat": post["v_hat"],
                }
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pd.DataFrame(rows).sort_values(["resolution", "seed"]).to_csv(
                output_path, index=False
            )

    df = pd.DataFrame(rows).sort_values(["resolution", "seed"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)

    summary = (
        df.groupby("resolution")
        .agg(
            n_seeds=("seed", "size"),
            q_hat_mean=("q_hat", "mean"),
            q_hat_sd=("q_hat", "std"),
            q_abs_error_mean=("q_abs_error", "mean"),
            q_abs_error_sd=("q_abs_error", "std"),
            n_hat_mean=("n_hat", "mean"),
            n_hat_sd=("n_hat", "std"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)
    return df, summary


if __name__ == "__main__":
    df, summary = run_resolution_bias()
    print(summary.to_string(index=False))
