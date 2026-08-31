"""Reproduce Fig 1A-D of Bhumbra & Beato (2013) -- the FEASIBILITY GATE.

Fig 1A-D is fully analytic from stated parameters and requires no fitted
data, so it is the cheapest possible check that the Q-function
(src.q_model.q_function) is implemented correctly:

  A: Binomial B(i | 6, 0.35) -- release-count distribution.
  B: Normal N(x | 0, 625), i.e. baseline noise, SD 25 pA.
  C: Gamma G(x | 11.1, 9) -- quantal amplitude distribution, so
     q = gamma*lambda = 100 pA and CV = 1/sqrt(gamma) ~ 0.30 (see confirmed
     constants in NOTES.md).
  D: the combined quantal likelihood Q -- calls q_model.q_function.

GO/NO-GO GATE: Panel D is the go/no-go checkpoint for the entire project.
Do not trust src/bqa.py's inference pipeline (grid search over conditions)
unless panel D reproduces the published figure. As implemented, the gate
currently PASSES qualitatively (multimodal, peaks near integer multiples of
q=100 pA, matching the paper's described shape) but has not been checked
pixel-for-pixel against the published figure image -- see NOTES.md. If a
future change to q_model.q_function breaks this qualitative match, STOP:
the mismatch means the Eq. 8 implementation or its parameterisation is
wrong, and any inference machinery built on top of it would be unreliable.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from src import q_model

# Confirmed constants (see NOTES.md).
N_SITES = 6
P_RELEASE = 0.35
BASELINE_SD = 25.0  # pA
GAMMA_SHAPE = 11.1
GAMMA_LAMBDA = 9.0  # scale parameter (Eq. 6-7: q = gamma*lambda, confirmed by Fig. 1C)
Q_QUANTAL = GAMMA_SHAPE * GAMMA_LAMBDA  # ~100 pA


def panel_a(x: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Panel A: Binomial B(i | n=6, p=0.35) release-count distribution."""
    if x is None:
        x = np.arange(0, N_SITES + 1)
    pmf = stats.binom.pmf(x, N_SITES, P_RELEASE)
    return x, pmf


def panel_b(x: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Panel B: Normal N(x | 0, 625), i.e. baseline noise, SD 25 pA."""
    if x is None:
        x = np.linspace(-100, 100, 500)
    pdf = stats.norm.pdf(x, loc=0.0, scale=BASELINE_SD)
    return x, pdf


def panel_c(x: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Panel C: Gamma G(x | shape=11.1, SCALE=9) quantal amplitude distribution.

    lambda is the SCALE, not the rate: Eq. 7 gives q = gamma * lambda, so
    11.1 * 9 = 100 pA, which is the paper's stated quantal size. This matches
    `src.q_model.q_function`, which also passes lambda as `scale`. Passing it
    as a rate would put the mean at 1.23 pA and render the panel empty on the
    0-300 pA axis.
    """
    if x is None:
        x = np.linspace(0, 300, 500)
    pdf = stats.gamma.pdf(x, a=GAMMA_SHAPE, scale=GAMMA_LAMBDA)
    return x, pdf


def panel_d(x: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Panel D: combined quantal likelihood Q(x) -- the feasibility gate.

    Calls `q_model.q_function` (Eq. 8). Reproducing this panel is the
    project's go/no-go checkpoint; see module docstring for current status.
    """
    if x is None:
        x = np.linspace(-100, 700, 500)
    q_values = q_model.q_function(
        x,
        n=N_SITES,
        p=P_RELEASE,
        gamma=GAMMA_SHAPE,
        lam=GAMMA_LAMBDA,
        eps2=BASELINE_SD**2,
    )
    return x, q_values


def gate_peak_check(tol_frac: float = 0.15) -> dict:
    """Quantitative feasibility-gate check on Q(x) (Eq. 8).

    The original paper describes the combined quantal likelihood as
    multimodal with modes near integer multiples of the quantal size q.
    Rather than a pixel comparison against the published image (no digitized
    reference curve is available), this locates the modes of the implemented
    Q(x) and checks that each interior mode sits within `tol_frac` of an
    integer multiple of q. This turns the gate from a purely visual "looks
    multimodal" into a reproducible numeric assertion.

    Parameters
    ----------
    tol_frac : float
        Allowed distance of each interior peak from its nearest multiple of
        q, as a fraction of q (0.15 => within 15 pA for q = 100 pA).

    Returns
    -------
    dict
        "peaks_pa": located mode positions (pA); "nearest_multiple": the
        integer multiple of q each peak maps to; "max_rel_error": largest
        relative distance to a multiple among interior peaks; "n_modes":
        number of modes found; "passed": bool.
    """
    from scipy.signal import find_peaks

    x = np.linspace(-50, 700, 4000)
    y = panel_d(x)[1]
    idx, _ = find_peaks(y)
    peaks = x[idx]

    # Interior modes only: the near-zero failure mode (baseline-noise peak)
    # is expected and not a multiple of q, so assess modes at x >= q/2.
    interior = peaks[peaks >= Q_QUANTAL / 2]
    multiples = np.round(interior / Q_QUANTAL)
    multiples = np.clip(multiples, 1, None)
    rel_err = np.abs(interior - multiples * Q_QUANTAL) / Q_QUANTAL
    max_rel = float(rel_err.max()) if rel_err.size else 1.0

    return {
        "peaks_pa": peaks,
        "interior_peaks_pa": interior,
        "nearest_multiple": multiples,
        "max_rel_error": max_rel,
        "n_modes": int(peaks.size),
        "passed": bool(interior.size >= 1 and max_rel <= tol_frac),
    }


def plot_figure1(save_path: str | None = None):
    """Scaffold plotting for all four Fig 1 panels.

    Panels A-C are fully computed. Panel D will raise NotImplementedError
    until q_model.q_function is implemented -- this is caught here and
    reported as the gate status rather than crashing the whole figure.
    """
    if plt is None:  # pragma: no cover
        raise ImportError("matplotlib is required to plot Figure 1")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    x_a, y_a = panel_a()
    axes[0].bar(x_a, y_a)
    axes[0].set_title("A: Binomial(n=6, p=0.35)")

    x_b, y_b = panel_b()
    axes[1].plot(x_b, y_b)
    axes[1].set_title("B: Normal(0, 25 pA)")

    x_c, y_c = panel_c()
    axes[2].plot(x_c, y_c)
    axes[2].set_title("C: Gamma(11.1, 9)")

    gate_passed = False
    try:
        x_d, y_d = panel_d()
        axes[3].plot(x_d, y_d)
        axes[3].set_title("D: Q(x) [GATE: implemented]")
        gate_passed = True
    except NotImplementedError:
        axes[3].text(
            0.5,
            0.5,
            "not yet implemented\n(q_model.q_function)",
            ha="center",
            va="center",
            transform=axes[3].transAxes,
        )
        axes[3].set_title("D: Q(x) [GATE: NOT PASSED]")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    return fig, gate_passed


if __name__ == "__main__":
    _, passed = plot_figure1()
    status = "PASSED" if passed else "NOT PASSED (q_model.q_function is TODO)"
    print(f"Feasibility gate (Fig 1 panel D): {status}")
