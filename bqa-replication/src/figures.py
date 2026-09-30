"""Top-level figure/analysis pipeline for the replication.

Each stage uses its own independently seeded RNG (derived from `base_seed`),
so stages are reproducible on their own and do not depend on the order in
which earlier stages happened to consume a shared RNG. Every stage also
writes a machine-readable CSV next to its figure, so the numeric results in
the paper are pinned to files rather than only to figure pixels.

Stages: Fig 1 feasibility gate (now a numeric peak check), SBC calibration,
DeltaP identifiability sweep, Gaussian-vs-gamma robustness illustration,
BQA-vs-MPFA comparison, grid-resolution bias check, and the low-SNR
("pathological") stress test.

The identifiability sweep and the SBC run dominate runtime; expect the full
pipeline to take a few minutes.
"""

from __future__ import annotations

import csv
import os

import numpy as np

from src import figure1, identifiability, mg_illustration, mpfa, plots, sbc
from src.pathological_mg import run_pathological_mg_analysis
from src.resolution_check import run_resolution_bias

CSV_DIR = "results/identifiability"
CAL_DIR = "results/calibration"

# SBC iteration count. 20 is a smoke run; the default here is powered enough
# to make the per-parameter uniformity tests meaningful.
SBC_ITERATIONS = 200
SBC_N_CANDIDATES = [4, 5, 6, 7, 8]
SBC_N_OBS = 40


def _write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})
    return path


def run_all(base_seed: int = 0, sbc_iterations: int = SBC_ITERATIONS) -> None:
    """Run every figure/analysis stage and write figures + CSVs."""
    figures_dir = plots.ensure_figures_dir()
    print(f"Writing figures to {figures_dir}/ and CSVs to {CSV_DIR}/, {CAL_DIR}/")

    # --- Fig 1 feasibility gate (deterministic; numeric peak check) ---------
    print("=== Fig 1A-D feasibility gate ===")
    gate = figure1.gate_peak_check()
    print(f"  gate passed: {gate['passed']} "
          f"(interior peaks {np.round(gate['interior_peaks_pa'], 1)} pA, "
          f"max rel. error {gate['max_rel_error']:.3f})")
    _write_csv(
        os.path.join(CSV_DIR, "figure1_gate.csv"),
        ["peak_pa", "nearest_multiple_of_q", "rel_error"],
        [
            {"peak_pa": float(pk), "nearest_multiple_of_q": float(m),
             "rel_error": abs(pk - m * figure1.Q_QUANTAL) / figure1.Q_QUANTAL}
            for pk, m in zip(gate["interior_peaks_pa"], gate["nearest_multiple"])
        ],
    )
    print(f"  wrote {plots.save_figure1(figures_dir)}")

    # --- SBC calibration ----------------------------------------------------
    print(f"=== SBC calibration ({sbc_iterations} iterations) ===")
    draws = sbc.run_sbc_reproducible(
        n_iterations=sbc_iterations,
        n_candidates=SBC_N_CANDIDATES,
        n_obs_per_condition=SBC_N_OBS,
        base_seed=base_seed + 1,
    )
    report = sbc.sbc_uniformity_test(draws)
    for param, s in report.items():
        print(f"  {param}: KS p={s['p_value']:.3f} "
              f"({'calibrated' if s['calibrated'] else 'NOT calibrated'}, "
              f"n={s['n_draws']})")
    _write_csv(
        os.path.join(CAL_DIR, "sbc_report.csv"),
        ["parameter", "ks_statistic", "p_value", "calibrated", "n_draws"],
        [{"parameter": p, **s} for p, s in report.items()],
    )
    _write_csv(
        os.path.join(CAL_DIR, "sbc_results.csv"),
        ["iteration", "q_rank", "r_rank", "gamma_rank", "n_rank"],
        [{"iteration": i, "q_rank": d.ranks["q"], "r_rank": d.ranks["r"],
          "gamma_rank": d.ranks["gamma"], "n_rank": d.ranks["n"]}
         for i, d in enumerate(draws)],
    )
    print(f"  wrote {plots.save_sbc_rank_histograms(draws, figures_dir)}")

    # --- Joint-grid vs product SBC comparison (Section 4.1) -----------------
    print(f"=== Joint-grid vs product SBC ({sbc_iterations} iterations) ===")
    from src import run_joint_sbc
    comp, comp_csv = run_joint_sbc.run_comparison(
        n_iterations=sbc_iterations,
        methods=(("product", "geometric"), ("joint", "geometric")),
    )
    for name, (report, mr, _rows) in comp.items():
        print(f"  {name}: " + ", ".join(
            f"{p} p={s['p_value']:.3f}" for p, s in report.items())
            + f"  (median q_hat/q_true={mr:.3f})")
    print(f"  wrote {comp_csv}")
    rows_by_method = {name: rows for name, (_r, _m, rows) in comp.items()}
    print(f"  wrote {plots.save_joint_vs_product_histograms(rows_by_method, figures_dir)}")

    # --- Identifiability sweep ---------------------------------------------
    print("=== Identifiability map (DeltaP sweep, 76 points) ===")
    id_results = identifiability.run_identifiability_sweep()
    _write_csv(
        os.path.join(CSV_DIR, "identifiability_results.csv"),
        ["delta_p", "p1", "p2", "q_hat", "n_hat", "v_hat",
         "q_ci95_width", "n_ci95_width"],
        id_results,
    )
    widest = max(id_results, key=lambda r: r["q_ci95_width"])
    narrowest = min(id_results, key=lambda r: r["q_ci95_width"])
    print(f"  q 95% CI width: {narrowest['q_ci95_width']:.1f} pA (min) "
          f"to {widest['q_ci95_width']:.1f} pA (max)")
    print(f"  wrote {plots.save_identifiability_map(id_results, figures_dir)}")

    # --- Gaussian-vs-gamma robustness illustration -------------------------
    print("=== Gaussian-vs-gamma mismatch robustness ===")
    mg_result = mg_illustration.demonstrate_mg_detectability(
        rng=np.random.default_rng(base_seed + 3)
    )
    print(f"  q_true={mg_result['q_true']:.1f}, q_hat={mg_result['q_hat']:.1f}; "
          f"n_true={mg_result['n_true']}, n_hat={mg_result['n_hat']:.2f}")
    _write_csv(
        os.path.join(CSV_DIR, "mg_result.csv"),
        ["q_true", "q_hat", "n_true", "n_hat", "v_hat"],
        [{k: mg_result[k] for k in ["q_true", "q_hat", "n_true", "n_hat", "v_hat"]}],
    )
    print(f"  wrote {plots.save_mg_posterior(mg_result, figures_dir)}")

    # --- BQA vs MPFA --------------------------------------------------------
    print("=== BQA vs MPFA on identical data ===")
    cmp = mpfa.compare_bqa_vs_mpfa(rng=np.random.default_rng(base_seed + 4))
    print(f"  truth q={cmp['q_true']}, n={cmp['n_true']}")
    print(f"  MPFA q_apparent={cmp['mpfa_q_apparent']:.1f}, "
          f"q_cv_corrected={cmp['mpfa_q_corrected']:.1f}, n={cmp['mpfa_n_hat']:.2f}")
    print(f"  BQA  q={cmp['bqa_q_hat']:.1f}, n={cmp['bqa_n_hat']:.2f}, "
          f"v={cmp['bqa_v_hat']:.2f}")
    print(f"  wrote {mpfa.save_comparison_csv(cmp)}")
    print(f"  wrote {plots.save_mpfa_comparison(cmp, figures_dir)}")

    # --- Grid-resolution bias check ----------------------------------------
    print("=== Grid-resolution bias check ===")
    _, bias = run_resolution_bias()
    for _, row in bias.iterrows():
        print(f"  res={int(row['resolution'])}: q_hat={row['q_hat_mean']:.1f} "
              f"(mean abs err {row['q_abs_error_mean']:.2f}), "
              f"n_hat={row['n_hat_mean']:.2f} ({int(row['n_seeds'])} seeds)")

    # --- Low-SNR stress test ------------------------------------------------
    print("=== Low signal-to-noise stress test ===")
    run_pathological_mg_analysis()


if __name__ == "__main__":
    run_all()
