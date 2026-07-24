"""Figure-writing helpers for the reproduction pipeline.

Every function renders a matplotlib figure and writes it to disk, so the
pipeline produces graph files on the filesystem rather than only computing
results in memory. Output directory defaults to ``results/figures/``.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend: write files, never block
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from src import figure1

FIGURES_DIR = "results/figures"


def _require_mpl() -> None:
    if plt is None:  # pragma: no cover
        raise ImportError("matplotlib is required to write figures")


def ensure_figures_dir(figures_dir: str = FIGURES_DIR) -> str:
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def save_figure1(figures_dir: str = FIGURES_DIR) -> str:
    """Render Fig 1A-D and write it as a PNG. Returns the path written."""
    _require_mpl()
    ensure_figures_dir(figures_dir)
    path = os.path.join(figures_dir, "figure1_feasibility_gate.png")
    fig, _gate_passed = figure1.plot_figure1(save_path=path)
    plt.close(fig)
    return path


def save_sbc_rank_histograms(draws, figures_dir: str = FIGURES_DIR) -> str:
    """Rank/PIT histogram per parameter -- uniform bars => calibrated."""
    _require_mpl()
    ensure_figures_dir(figures_dir)
    params = ["q", "r", "gamma", "n"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    n_bins = max(5, int(np.sqrt(max(len(draws), 1))))
    for ax, param in zip(axes.ravel(), params):
        ranks = [d.ranks[param] for d in draws]
        ax.hist(ranks, bins=n_bins, range=(0.0, 1.0), edgecolor="black")
        ax.axhline(len(ranks) / n_bins, color="red", ls="--", lw=1)
        ax.set_title(f"SBC PIT: {param}")
        ax.set_xlabel("PIT rank")
        ax.set_ylabel("count")
    fig.suptitle(f"SBC rank calibration (n={len(draws)} draws)")
    fig.tight_layout()
    path = os.path.join(figures_dir, "sbc_rank_histograms.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def save_joint_vs_product_histograms(rows_by_method, figures_dir: str = FIGURES_DIR) -> str:
    """Overlay SBC PIT histograms for the joint grid vs the product method.

    `rows_by_method` maps a method label to its list of per-iteration result
    dicts (keys ``q_rank``/``r_rank``/``gamma_rank``/``n_rank``), as produced by
    `src.run_joint_sbc.run_comparison`. Flat bars => calibrated; the joint grid
    should flatten gamma and n relative to product while q and r stay skewed.
    """
    _require_mpl()
    ensure_figures_dir(figures_dir)
    params = ["q", "r", "gamma", "n"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    colors = {"product": "tab:orange", "joint": "tab:blue"}
    for ax, param in zip(axes.ravel(), params):
        for label, rows in rows_by_method.items():
            ranks = [r[f"{param}_rank"] for r in rows]
            n_bins = max(5, int(np.sqrt(max(len(ranks), 1))))
            key = "joint" if "joint" in label else ("product" if "product" in label else label)
            ax.hist(ranks, bins=n_bins, range=(0.0, 1.0), histtype="step",
                    lw=2, label=label, color=colors.get(key))
            ax.axhline(len(ranks) / n_bins, color="gray", ls="--", lw=1)
        ax.set_title(f"SBC PIT: {param}")
        ax.set_xlabel("PIT rank")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    fig.suptitle("SBC rank calibration: joint grid vs product combination")
    fig.tight_layout()
    path = os.path.join(figures_dir, "sbc_joint_vs_product.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def save_identifiability_map(id_results: Sequence[dict], figures_dir: str = FIGURES_DIR) -> str:
    """q 95% CI width vs DeltaP -- the identifiability sweep."""
    _require_mpl()
    ensure_figures_dir(figures_dir)
    ordered = sorted(id_results, key=lambda r: r["delta_p"])
    delta_p = [r["delta_p"] for r in ordered]
    ci_width = [r["q_ci95_width"] for r in ordered]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(delta_p, ci_width, marker="o", ms=3)
    ax.set_xlabel(r"$\Delta P$")
    ax.set_ylabel("q 95% CI width (pA)")
    ax.set_title("Identifiability map: q credible-interval width vs DeltaP")
    fig.tight_layout()
    path = os.path.join(figures_dir, "identifiability_map.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def save_mpfa_comparison(mpfa_result: dict, figures_dir: str = FIGURES_DIR) -> str:
    """Variance-mean parabola with BQA vs MPFA parameter estimates annotated."""
    _require_mpl()
    ensure_figures_dir(figures_dir)
    mpfa = mpfa_result["mpfa"]
    means = mpfa.means
    variances = mpfa.variances

    order = np.argsort(means)
    grid_i = np.linspace(0, means.max() * 1.05, 200)
    fit_curve = mpfa.slope * grid_i - grid_i**2 * (1.0 / mpfa.n_hat)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(means[order], variances[order], color="black", zorder=3, label="conditions")
    ax.plot(grid_i, fit_curve, color="tab:blue", label="MPFA parabola fit")
    ax.set_xlabel("mean evoked response (pA)")
    ax.set_ylabel("evoked variance (pA$^2$)")
    txt = (
        f"truth: q={mpfa_result['q_true']:.0f}, n={mpfa_result['n_true']}\n"
        f"MPFA: q_app={mpfa_result['mpfa_q_apparent']:.1f}, "
        f"q_corr={mpfa_result['mpfa_q_corrected']:.1f}, n={mpfa_result['mpfa_n_hat']:.2f}\n"
        f"BQA: q={mpfa_result['bqa_q_hat']:.1f}, n={mpfa_result['bqa_n_hat']:.2f}"
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
    ax.set_title("BQA vs MPFA on identical data")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = os.path.join(figures_dir, "mpfa_vs_bqa.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def save_mg_posterior(mg_result: dict, figures_dir: str = FIGURES_DIR) -> str:
    """MG mismatch: q posterior marginal with true vs estimated q marked."""
    _require_mpl()
    ensure_figures_dir(figures_dir)
    posterior = mg_result["posterior"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(posterior["q_axis"], posterior["q_marginal"], label="q posterior")
    ax.axvline(mg_result["q_true"], color="green", ls="--", label=f"q_true={mg_result['q_true']:.1f}")
    ax.axvline(mg_result["q_hat"], color="red", ls=":", label=f"q_hat={mg_result['q_hat']:.1f}")
    ax.set_xlabel("q (pA)")
    ax.set_ylabel("posterior density (marginal weight)")
    ax.set_title("MG Gaussian-vs-gamma mismatch: q recovery")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(figures_dir, "mg_detectability.png")
    fig.savefig(path)
    plt.close(fig)
    return path
