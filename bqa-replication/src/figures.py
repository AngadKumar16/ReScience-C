"""Top-level figure-generation entry points for the reproduction pipeline.

Dispatches to the individual figure/analysis modules. All stages are now
implemented (Eq. 8, Appendix A1-A13, and downstream SBC/identifiability/MG
analyses); this module runs them end to end. `run_identifiability_sweep`
(76 grid-inference calls) is the slow stage -- expect the full pipeline to
take a few minutes.
"""

from __future__ import annotations

import numpy as np

from src import figure1, identifiability, mg_illustration, sbc


def run_all(rng_seed: int = 0) -> None:
    """Run every figure/analysis stage."""
    rng = np.random.default_rng(rng_seed)

    print("=== Fig 1A-D feasibility gate ===")
    _, gate_passed = figure1.plot_figure1()
    print("PASSED" if gate_passed else "NOT PASSED")

    print("=== SBC calibration (smoke run, 20 iterations) ===")
    draws = sbc.run_sbc(
        n_iterations=20,
        n_candidates=[4, 5, 6, 7, 8],
        n_obs_per_condition=40,
        rng=rng,
    )
    report = sbc.sbc_uniformity_test(draws)
    for param, stats_ in report.items():
        print(
            f"  {param}: KS p={stats_['p_value']:.3f} "
            f"({'calibrated' if stats_['calibrated'] else 'NOT calibrated'} "
            f"at alpha=0.05, n={stats_['n_draws']} draws)"
        )
    print(
        "  (20 iterations is a smoke run, not a well-powered calibration "
        "check -- increase n_iterations for a real SBC report.)"
    )

    print("=== Identifiability map (DeltaP sweep, 76 points) ===")
    id_results = identifiability.run_identifiability_sweep(rng=rng)
    print(f"  computed {len(id_results)} DeltaP points")
    widest = max(id_results, key=lambda r: r["q_ci95_width"])
    narrowest = min(id_results, key=lambda r: r["q_ci95_width"])
    print(
        f"  widest q 95% CI at DeltaP={widest['delta_p']:.2f} "
        f"(width={widest['q_ci95_width']:.1f} pA)"
    )
    print(
        f"  narrowest q 95% CI at DeltaP={narrowest['delta_p']:.2f} "
        f"(width={narrowest['q_ci95_width']:.1f} pA)"
    )

    print("=== MG (Gaussian-vs-gamma mismatch) detectability illustration ===")
    mg_result = mg_illustration.demonstrate_mg_detectability(rng=rng)
    print(
        f"  q_true={mg_result['q_true']:.1f} pA, q_hat={mg_result['q_hat']:.1f} pA"
    )
    print(f"  n_true={mg_result['n_true']}, n_hat={mg_result['n_hat']:.2f}")


if __name__ == "__main__":
    run_all()
