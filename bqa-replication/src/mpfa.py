"""Multiple-probability fluctuation analysis (MPFA) and a head-to-head with BQA.

The original paper (Bhumbra & Beato 2013) positions BQA against MPFA
(Silver 2003) as the established alternative. That comparison was out of
scope for the first pass of this replication because Silver (2003) was not
transcribed. This module adds a standard, textbook MPFA implemented from the
general variance-mean relationship (NOT transcribed from Silver 2003's
specific equations) so that BQA can be scored against a concrete baseline on
identical simulated data.

MPFA in brief. Across conditions that differ only in release probability P,
the mean evoked response is I = N P q and, for a binomial model with quantal
size q, per-quantum coefficient of variation CV, and N release sites, the
variance of the evoked response is

    sigma^2(I) = q (1 + CV^2) I  -  I^2 / N .

So a plot of variance against mean is a parabola through the origin. Its
initial slope is q(1 + CV^2) (an *apparent* quantal size inflated by the
quantal variability) and its curvature gives N. This is the well-known
limitation MPFA carries and BQA avoids: without an independent estimate of
CV, MPFA's slope overestimates q, whereas BQA estimates the quantal
variability jointly.

`compare_bqa_vs_mpfa` simulates several release-probability conditions from
the Gaussian intrasite model (`src.simulate.simulate_responses`), runs both
estimators on the same data, and returns their q and N estimates alongside
the truth. Using the Gaussian simulator here is deliberate (it is the
generative model both methods are being stress-tested against, mirroring the
robustness illustration in src/mg_illustration.py), not an SBC-style
calibration run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.bqa import ConditionData, change_of_variables_and_marginalise
from src.grid import build_grid
from src.simulate import simulate_responses


@dataclass
class MPFAResult:
    """MPFA parabola fit and the parameter estimates it implies."""

    means: np.ndarray
    variances: np.ndarray
    slope: float  # q * (1 + CV^2): apparent quantal size
    q_hat_apparent: float
    q_hat_cv_corrected: float
    n_hat: float


def fit_mpfa(
    means: np.ndarray,
    variances: np.ndarray,
    cv_known: float | None = None,
) -> MPFAResult:
    """Fit the variance-mean parabola sigma^2 = a*I - b*I^2 through the origin.

    Parameters
    ----------
    means, variances : np.ndarray
        Per-condition mean and (baseline-subtracted) variance of the evoked
        response. At least three conditions are needed to constrain the
        parabola.
    cv_known : float, optional
        If given, the apparent quantal size (the initial slope) is corrected
        by dividing by (1 + cv_known^2) to recover the true q. This is the
        information BQA infers on its own; MPFA needs it supplied.

    Returns
    -------
    MPFAResult
    """
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)
    if means.size < 3:
        raise ValueError("MPFA needs at least 3 release-probability conditions")

    # Least-squares fit of sigma^2 = a*I + c*I^2 (no intercept). The physical
    # parabola is sigma^2 = a*I - b*I^2, so b = -c and N = 1/b.
    design = np.column_stack([means, means**2])
    coeffs, *_ = np.linalg.lstsq(design, variances, rcond=None)
    a, c = float(coeffs[0]), float(coeffs[1])
    b = -c
    n_hat = 1.0 / b if b > 0 else float("inf")

    q_apparent = a
    q_corrected = a / (1.0 + cv_known**2) if cv_known is not None else float("nan")

    return MPFAResult(
        means=means,
        variances=variances,
        slope=a,
        q_hat_apparent=q_apparent,
        q_hat_cv_corrected=q_corrected,
        n_hat=n_hat,
    )


def compare_bqa_vs_mpfa(
    n_sites: int = 6,
    q: float = 100.0,
    cv_intra: float = 0.3,
    baseline_sd: float = 25.0,
    n_obs: int = 60,
    release_probs: tuple[float, ...] = (0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9),
    rng: np.random.Generator | None = None,
) -> dict:
    """Run BQA and MPFA on identical multi-condition data and compare.

    Simulates one condition per entry of `release_probs` under the Gaussian
    intrasite model, then:
      - MPFA: computes per-condition mean and baseline-subtracted variance,
        fits the variance-mean parabola, reports q (apparent and
        CV-corrected) and N.
      - BQA: feeds all conditions into the grid inference and reports its
        median q and n estimates.

    Returns a dict of both methods' estimates and the ground truth.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    eps2 = baseline_sd**2

    means, variances, conditions = [], [], []
    for p_true in release_probs:
        sim = simulate_responses(n_sites, p_true, q, cv_intra, baseline_sd, n_obs, rng)
        amps = sim.amplitudes
        means.append(float(np.mean(amps)))
        # Subtract the known baseline (recording-noise) variance so the fit
        # sees the evoked variance only.
        variances.append(float(np.var(amps, ddof=1) - eps2))
        mu_true = n_sites * p_true * q
        conditions.append(ConditionData(mu=mu_true, eps2=eps2, amplitudes=amps))

    mpfa = fit_mpfa(np.array(means), np.array(variances), cv_known=cv_intra)

    n_candidates = [n_sites - 2, n_sites - 1, n_sites, n_sites + 1, n_sites + 2]
    grid = build_grid(np.array(n_candidates))
    post = change_of_variables_and_marginalise(conditions, grid, n_candidates)

    return {
        "q_true": q,
        "n_true": n_sites,
        "cv_true": cv_intra,
        "mpfa": mpfa,
        "mpfa_q_apparent": mpfa.q_hat_apparent,
        "mpfa_q_corrected": mpfa.q_hat_cv_corrected,
        "mpfa_n_hat": mpfa.n_hat,
        "bqa_q_hat": post["q_hat"],
        "bqa_n_hat": post["n_hat"],
        "bqa_v_hat": post["v_hat"],
        "posterior": post,
    }


def save_comparison_csv(
    result: dict, output_path: str = "results/identifiability/mpfa_vs_bqa.csv"
) -> str:
    """Write the BQA-vs-MPFA estimates to a CSV and return the path."""
    rows = [
        {"parameter": "q", "truth": result["q_true"], "MPFA_apparent": result["mpfa_q_apparent"], "MPFA_cv_corrected": result["mpfa_q_corrected"], "BQA": result["bqa_q_hat"]},
        {"parameter": "n", "truth": result["n_true"], "MPFA_apparent": result["mpfa_n_hat"], "MPFA_cv_corrected": result["mpfa_n_hat"], "BQA": result["bqa_n_hat"]},
    ]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    res = compare_bqa_vs_mpfa()
    print(f"truth: q={res['q_true']}, n={res['n_true']}, cv={res['cv_true']}")
    print(f"MPFA:  q_apparent={res['mpfa_q_apparent']:.1f}, "
          f"q_cv_corrected={res['mpfa_q_corrected']:.1f}, n={res['mpfa_n_hat']:.2f}")
    print(f"BQA:   q={res['bqa_q_hat']:.1f}, n={res['bqa_n_hat']:.2f}, v={res['bqa_v_hat']:.2f}")
    print("wrote", save_comparison_csv(res))
