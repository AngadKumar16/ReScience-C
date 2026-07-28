"""Seed ensemble for the low signal-to-noise stress test, with a matched-generator control.

The single-seed low-SNR result (`src/pathological_mg.py`, seed 42) is not
representative: seed 42 happens to give the best site-count recovery of the
seeds tried. This module reruns the same scenario over an ensemble of
independent seeds, and crucially runs it under TWO generators:

  "gaussian" -- `src.simulate.simulate_responses`, the Gaussian intrasite-noise
      simulator that `pathological_mg.py` uses. Feeding this to the gamma
      likelihood in `src.bqa` is a model mismatch, so any bias it shows is
      low-SNR effect *plus* Gaussian-vs-gamma misspecification, confounded.

  "gamma" -- `src.simulate_q.simulate_from_q_model`, the model's own Eq. 8
      likelihood. This is the matched control. It isolates the pure low-SNR
      effect, because generative and inferential models now agree.

Comparing the two arms is what separates "weak signal" from "wrong model" as
the cause of the quantal-size overestimate. The resolution sweep in
`src/resolution_check.py` only rules out grid coarseness; it cannot rule out
misspecification, because it holds the generator fixed.

The candidate site-count grid is widened to n in {4,...,20}. The original
{8,...,16} truncates: several seeds return n_hat within one bin of the lower
edge, so their posterior mass is clipped and the ensemble mean is biased
upward.

Results are checkpointed per (generator, seed) row, so the run is resumable.

Run: python -m src.low_snr_seeds
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.bqa import ConditionData, change_of_variables_and_marginalise
from src.grid import build_grid
from src.simulate import simulate_responses
from src.simulate_q import simulate_from_q_model

# Same physical scenario as src/pathological_mg.py, so the numbers are
# directly comparable to results/identifiability/low_snr_results.csv.
N_SITES = 12
Q_TRUE = 30.0
CV_INTRA = 0.3
BASELINE_SD = 25.0
N_OBS = 100
P1, P2 = 0.15, 0.45

# Widened from {8..16}: the narrow grid truncates the low tail.
N_CANDIDATES = list(range(4, 21))

GENERATORS = ("gaussian", "gamma")
SEEDS = tuple(range(12))

OUTPUT_PATH = "results/identifiability/low_snr_seed_ensemble.csv"


def _draw(generator: str, p: float, rng: np.random.Generator) -> np.ndarray:
    """Draw one condition's amplitudes from the requested generator."""
    if generator == "gaussian":
        return simulate_responses(N_SITES, p, Q_TRUE, CV_INTRA, BASELINE_SD, N_OBS, rng).amplitudes
    if generator == "gamma":
        gamma_shape = 1.0 / CV_INTRA**2  # 11.11, matching the paper's gamma = 11.1
        lam = Q_TRUE / gamma_shape  # so that q = gamma * lambda
        return simulate_from_q_model(
            N_SITES, p, gamma_shape, lam, BASELINE_SD**2, N_OBS, rng
        )
    raise ValueError(f"unknown generator {generator!r}")


def run_one(generator: str, seed: int) -> dict:
    """Run a single low-SNR inference for one generator and seed."""
    rng = np.random.default_rng(seed)
    eps2 = BASELINE_SD**2
    conditions = []
    for p_true in (P1, P2):
        amplitudes = _draw(generator, p_true, rng)
        conditions.append(
            ConditionData(mu=N_SITES * p_true * Q_TRUE, eps2=eps2, amplitudes=amplitudes)
        )
    grid = build_grid(np.array(N_CANDIDATES))
    post = change_of_variables_and_marginalise(conditions, grid, N_CANDIDATES)
    return {
        "generator": generator,
        "seed": seed,
        "q_true": Q_TRUE,
        "q_hat": post["q_hat"],
        "n_true": N_SITES,
        "n_hat": post["n_hat"],
        "v_hat": post["v_hat"],
    }


def run_ensemble(
    generators: tuple[str, ...] = GENERATORS,
    seeds: tuple[int, ...] = SEEDS,
    output_path: str = OUTPUT_PATH,
    max_new: int | None = None,
) -> pd.DataFrame:
    """Run every (generator, seed) cell, resuming from any existing checkpoint.

    Each completed row is written immediately, so an interrupted run can be
    restarted and will only compute the cells it is missing. `max_new` caps how
    many new cells this invocation computes, for running under a time limit.
    """
    rows: list[dict] = []
    done: set[tuple[str, int]] = set()
    if os.path.exists(output_path):
        prev = pd.read_csv(output_path)
        rows = prev.to_dict("records")
        done = {(str(r["generator"]), int(r["seed"])) for r in rows}

    computed = 0
    for generator in generators:
        for seed in seeds:
            if (generator, seed) in done:
                continue
            if max_new is not None and computed >= max_new:
                break
            rows.append(run_one(generator, seed))
            computed += 1
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pd.DataFrame(rows).sort_values(["generator", "seed"]).to_csv(
                output_path, index=False
            )

    return pd.DataFrame(rows).sort_values(["generator", "seed"]).reset_index(drop=True)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-generator mean and SD of the point estimates, with percent bias."""
    out = []
    for generator, sub in df.groupby("generator"):
        out.append(
            {
                "generator": generator,
                "n_seeds": len(sub),
                "q_mean": sub["q_hat"].mean(),
                "q_sd": sub["q_hat"].std(ddof=1),
                "q_bias_pct": 100 * (sub["q_hat"].mean() / Q_TRUE - 1),
                "n_mean": sub["n_hat"].mean(),
                "n_sd": sub["n_hat"].std(ddof=1),
                "n_bias_pct": 100 * (sub["n_hat"].mean() / N_SITES - 1),
            }
        )
    return pd.DataFrame(out)


if __name__ == "__main__":
    df = run_ensemble()
    print(df.to_string(index=False))
    print()
    print(summarise(df).to_string(index=False))
