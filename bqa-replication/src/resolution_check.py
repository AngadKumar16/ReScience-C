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
N_CANDIDATES = [8, 9, 10, 11, 12, 13, 14, 15, 16]
RESOLUTIONS = [64, 128, 256]
SEED = 42


def run_resolution_bias(
    resolutions: list[int] = RESOLUTIONS,
    output_path: str = "results/identifiability/resolution_bias.csv",
) -> pd.DataFrame:
    """Rerun the low-SNR scenario at several grid resolutions and tabulate bias.

    A fresh, identically seeded simulator draw is used for every resolution
    so that only the grid resolution changes between rows (the data are held
    fixed; just the inference grid is refined).
    """
    eps2 = BASELINE_SD**2
    rows = []
    done = set()
    if os.path.exists(output_path):
        prev = pd.read_csv(output_path)
        rows = prev.to_dict("records")
        done = set(int(r["resolution"]) for r in rows)
    for res in resolutions:
        if res in done:
            continue
        rng = np.random.default_rng(SEED)  # identical data every resolution
        conditions = []
        for p_true in (P1, P2):
            sim = simulate_responses(N_SITES, p_true, Q_TRUE, CV_INTRA, BASELINE_SD, N_OBS, rng)
            mu_true = N_SITES * p_true * Q_TRUE
            conditions.append(ConditionData(mu=mu_true, eps2=eps2, amplitudes=sim.amplitudes))

        grid = build_grid(np.array(N_CANDIDATES), p_resolution=res, v_resolution=res)
        post = change_of_variables_and_marginalise(conditions, grid, N_CANDIDATES)
        rows.append(
            {
                "resolution": res,
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
        pd.DataFrame(rows).sort_values("resolution").to_csv(output_path, index=False)

    df = pd.DataFrame(rows).sort_values("resolution").reset_index(drop=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = run_resolution_bias()
    print(df.to_string(index=False))
