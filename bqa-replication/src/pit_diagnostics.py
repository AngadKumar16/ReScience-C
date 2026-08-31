"""Localising the K = 1, fixed-n quantal-size SBC failure.

Background
----------
`src/sbc_fixed_n_check.py` reports that the q rank statistic fails even at
K = 1 with n fixed to the truth and the exactly-correct ``induced`` prior
(KS p = 1.2e-8, mean PIT 0.641, 17.0% of truths above the posterior's 95th
percentile against 2.0% below). The failure is one-sided: the marginal's
upper tail is systematically too short.

Three explanations were live. This module tests all three in one run.

H1 -- mass-versus-density on the log-spaced q axis.
    If ``q_marginal`` held density values while `weighted_cdf_pit` treated
    them as probability mass, the widening bins of a geomspace axis would be
    under-weighted to the right, tilting the posterior left and pushing
    ranks up. `test_pit_mass_vs_density` isolates this with no BQA involved.

    Note this is NOT the test circulated earlier, which built its input as
    ``marg = dens / dens.sum()`` -- that bakes the suspected bug into the
    test input, so it fails whether or not `weighted_cdf_pit` is correct and
    can therefore not distinguish the two. Here the mass and density inputs
    are constructed and passed separately.

H2 -- the mid-P correction in `weighted_cdf_pit` is off by half a cell.
    The function computes ``mass_below + 0.5 * mass_at`` with
    ``idx = np.searchsorted(axis_values, true_value)``. searchsorted returns
    the first index whose axis value is >= the true value, so ``weights[:idx]``
    already contains the whole cell BELOW the true value and ``weights[idx]``
    is the cell ABOVE it. The intended half-cell credit is therefore applied
    to the wrong cell, biasing the PIT upward by up to one cell's mass and on
    average by about half of it. The bias is strictly non-negative, which is
    the one-sidedness the paper reports. `pit_interpolated` is the corrected
    version: it linearly interpolates the cumulative mass at the true value
    instead of snapping to a cell boundary.

H3 -- grid discretisation. If the defect shrinks as the axes are refined it
    is quadrature error, not a bug in the bookkeeping.

Per draw the module records five ranks, all on the identical simulated data:

    rank_current   the shipped route (joint grid, induced prior, PIT as-is)
    rank_interp    the same posterior, corrected interpolated PIT (H2)
    rank_p_direct  computed on the (arcsin sqrt p, log v) grid with no
                   transport and no log q axis, mapped through the strictly
                   decreasing q = mu/(n p); at fixed mu and n the PIT is
                   invariant under this map, so this must agree with
                   rank_current up to quadrature (H3 / localisation)
    rank_res256    the shipped route at resolution 512 (H3)
    sd_log_q       posterior SD of log q, which is what a shift-type
                   explanation has to be measured against

Run:
    python -m src.pit_diagnostics
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

from src.bqa import (
    ConditionData,
    _normalize_log_grid,
    change_of_variables_and_marginalise,
    per_condition_log_likelihood,
)
from src.grid import build_grid
from src.sbc import SBC_EPS2, sample_prior, weighted_cdf_pit
from src.simulate_q import simulate_from_q_model

N_CANDIDATES = [4, 5, 6, 7, 8]
N_OBS = 40
N_ITERATIONS = int(os.environ.get("BQA_PIT_ITERS", "200"))
BASE_SEED = 1
RES = 128
RES_FINE = 256


# --------------------------------------------------------------------------
# H2: corrected PIT
# --------------------------------------------------------------------------
def pit_interpolated(true_value: float, axis_values: np.ndarray,
                     weights: np.ndarray) -> float:
    """PIT by linear interpolation of the cumulative mass at `true_value`.

    `weighted_cdf_pit` snaps to a cell boundary and credits half of the cell
    ABOVE the true value (see module docstring, H2). Here the discrete
    marginal is read as mass located at the axis points, the cumulative mass
    is defined at cell midpoints, and the value at `true_value` is obtained
    by linear interpolation in log space (the q axis is geomspace). The
    result is exact for a piecewise-uniform reconstruction and carries no
    systematic offset.
    """
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    w = w / total

    x = np.log(np.asarray(axis_values, dtype=float))
    t = math.log(float(true_value))

    # Cumulative mass evaluated at each axis point, mid-cell convention.
    cum = np.cumsum(w) - 0.5 * w
    if t <= x[0]:
        return float(np.clip(cum[0] * (1.0 if t == x[0] else 0.0), 0.0, 1.0))
    if t >= x[-1]:
        return 1.0
    return float(np.interp(t, x, cum))


# --------------------------------------------------------------------------
# H1: isolate the PIT, no BQA involved
# --------------------------------------------------------------------------
def test_pit_mass_vs_density(n_samples: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Feed `weighted_cdf_pit` a known analytic distribution on a log axis.

    Two inputs, same distribution:
      - "mass": density * bin width, normalised -- what a correct grid produces
      - "density": density normalised by its own sum -- the suspected bug

    A correct PIT returns uniform ranks for "mass". If it also returns uniform
    ranks for "density" the function is silently insensitive to the
    distinction; if "density" fails while "mass" passes, the mechanism is real
    but only matters if the pipeline actually passes densities.
    """
    rng = np.random.default_rng(seed)
    q_axis = np.geomspace(10.0, 1000.0, 128)
    s, scale = 0.4, 100.0
    dens = stats_lognorm_pdf(q_axis, s, scale)
    width = np.gradient(q_axis)

    mass = dens * width
    mass = mass / mass.sum()
    density = dens / dens.sum()

    samples = np.exp(rng.normal(math.log(scale), s, size=n_samples))
    rows = []
    for label, w in (("mass", mass), ("density", density)):
        ranks = np.array([weighted_cdf_pit(x, q_axis, w) for x in samples])
        ranks_i = np.array([pit_interpolated(x, q_axis, w) for x in samples])
        rows.append({
            "input": label,
            "pit": "weighted_cdf_pit",
            "ks_p": stats.kstest(ranks, "uniform").pvalue,
            "mean_pit": ranks.mean(),
            "frac_above_p95": float((ranks > 0.95).mean()),
            "frac_below_p05": float((ranks < 0.05).mean()),
        })
        rows.append({
            "input": label,
            "pit": "pit_interpolated",
            "ks_p": stats.kstest(ranks_i, "uniform").pvalue,
            "mean_pit": ranks_i.mean(),
            "frac_above_p95": float((ranks_i > 0.95).mean()),
            "frac_below_p05": float((ranks_i < 0.05).mean()),
        })
    return pd.DataFrame(rows)


def stats_lognorm_pdf(x: np.ndarray, s: float, scale: float) -> np.ndarray:
    """Lognormal density, written out so the module needs no scipy.stats.lognorm."""
    x = np.asarray(x, dtype=float)
    z = (np.log(x) - math.log(scale)) / s
    return np.exp(-0.5 * z * z) / (x * s * math.sqrt(2.0 * math.pi))


# --------------------------------------------------------------------------
# H2 / H3: the five ranks per draw
# --------------------------------------------------------------------------
def _posterior_sd_log(axis: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    lx = np.log(axis)
    m = float(np.sum(w * lx))
    return float(math.sqrt(max(np.sum(w * (lx - m) ** 2), 0.0)))


def _one_iteration(i: int) -> dict:
    rng = np.random.default_rng([BASE_SEED, i])
    theta = sample_prior(N_CANDIDATES, rng)
    n_true, p1, v_true, q_true = theta["n"], theta["p1"], theta["v"], theta["q"]
    gamma_true = 1.0 / v_true**2

    amps = simulate_from_q_model(
        n_true, p1, gamma_true, q_true / gamma_true, SBC_EPS2, N_OBS, rng
    )
    mu = n_true * p1 * q_true
    conds = [ConditionData(mu=mu, eps2=SBC_EPS2, amplitudes=amps)]
    candidates = [n_true]  # n fixed to the truth

    grid = build_grid(np.array(candidates), p_resolution=RES, v_resolution=RES)
    post = change_of_variables_and_marginalise(
        conds, grid, candidates, combine="joint", q_prior="induced"
    )
    q_axis, q_marg = post["q_axis"], post["q_marginal"]

    # Route B: the p grid itself. No transport, no log q axis. The p grid is
    # uniform in arcsin(sqrt p) and the arcsine prior is flat there, so the
    # normalised mass grid is already the posterior mass per cell.
    log_l = per_condition_log_likelihood(conds[0], grid, n_true)
    mass_pv = _normalize_log_grid(log_l)
    p_marg = mass_pv.sum(axis=1)
    rank_p = weighted_cdf_pit(p1, grid.p_values, p_marg)
    rank_p_interp_raw = _pit_interp_linear(p1, grid.p_values, p_marg)
    # q = mu/(n p) is strictly DECREASING in p, so the q rank is 1 - p rank.
    rank_p_direct = 1.0 - rank_p
    rank_p_direct_interp = 1.0 - rank_p_interp_raw

    # Route C: same shipped route, 4x resolution.
    grid_f = build_grid(np.array(candidates), p_resolution=RES_FINE,
                        v_resolution=RES_FINE)
    post_f = change_of_variables_and_marginalise(
        conds, grid_f, candidates, combine="joint", q_prior="induced"
    )

    return {
        "iteration": i,
        "q_true": q_true,
        "p1_true": p1,
        "n_true": n_true,
        "rank_current": weighted_cdf_pit(q_true, q_axis, q_marg),
        "rank_interp": pit_interpolated(q_true, q_axis, q_marg),
        "rank_p_direct": rank_p_direct,
        "rank_p_direct_interp": rank_p_direct_interp,
        "rank_res256": weighted_cdf_pit(q_true, post_f["q_axis"],
                                        post_f["q_marginal"]),
        "rank_res256_interp": pit_interpolated(q_true, post_f["q_axis"],
                                               post_f["q_marginal"]),
        "sd_log_q": _posterior_sd_log(q_axis, q_marg),
        "cells_per_sd": _posterior_sd_log(q_axis, q_marg)
        / float(np.diff(np.log(q_axis))[0]),
    }


def _pit_interp_linear(true_value: float, axis_values: np.ndarray,
                       weights: np.ndarray) -> float:
    """Interpolated PIT on a linear (non-log) axis -- used for the p grid."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    x = np.asarray(axis_values, dtype=float)
    cum = np.cumsum(w) - 0.5 * w
    if true_value <= x[0]:
        return 0.0
    if true_value >= x[-1]:
        return 1.0
    return float(np.interp(true_value, x, cum))


def _summarise(ranks: np.ndarray, label: str) -> dict:
    return {
        "route": label,
        "ks_p": stats.kstest(ranks, "uniform").pvalue,
        "mean_pit": float(ranks.mean()),
        "frac_above_p95": float((ranks > 0.95).mean()),
        "frac_below_p05": float((ranks < 0.05).mean()),
        "coverage_90": float(((ranks > 0.05) & (ranks < 0.95)).mean()),
    }


CHUNK_DIR = "results/calibration/_pitchunks"


def run_chunk(start: int, end: int, procs: int = 4) -> str:
    """Run draws [start, end) and write them to their own CSV.

    Exists so the suite can be run in slices under a wall-clock cap and
    reassembled by `collect_chunks`; the per-draw results are independent
    and seeded by iteration index, so slicing changes nothing.
    """
    os.makedirs(CHUNK_DIR, exist_ok=True)
    with ProcessPoolExecutor(max_workers=procs) as ex:
        rows = list(ex.map(_one_iteration, range(start, end)))
    path = f"{CHUNK_DIR}/chunk_{start:04d}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def collect_chunks() -> pd.DataFrame:
    """Reassemble every chunk CSV into one frame, ordered by iteration."""
    import glob

    files = sorted(glob.glob(f"{CHUNK_DIR}/chunk_*.csv"))
    if not files:
        raise FileNotFoundError(f"no chunks in {CHUNK_DIR}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.sort_values("iteration").reset_index(drop=True)


ROUTES = [
    ("rank_current", "shipped (joint, res 128, weighted_cdf_pit)"),
    ("rank_interp", "same posterior, interpolated PIT"),
    ("rank_p_direct", "p grid direct, weighted_cdf_pit"),
    ("rank_p_direct_interp", "p grid direct, interpolated PIT"),
    ("rank_res256", "shipped route at res 256"),
    ("rank_res256_interp", "res 256, interpolated PIT"),
]


def summarise_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([_summarise(df[c].to_numpy(), label)
                         for c, label in ROUTES])


def run(n_iterations: int = N_ITERATIONS, procs: int = 4) -> pd.DataFrame:
    with ProcessPoolExecutor(max_workers=procs) as ex:
        rows = list(ex.map(_one_iteration, range(n_iterations)))
    df = pd.DataFrame(rows)

    routes = [
        ("rank_current", "shipped (joint, res 128, weighted_cdf_pit)"),
        ("rank_interp", "same posterior, interpolated PIT"),
        ("rank_p_direct", "p grid direct, weighted_cdf_pit"),
        ("rank_p_direct_interp", "p grid direct, interpolated PIT"),
        ("rank_res256", "shipped route at res 256"),
        ("rank_res256_interp", "res 256, interpolated PIT"),
    ]
    summary = pd.DataFrame([_summarise(df[c].to_numpy(), label)
                            for c, label in routes])

    os.makedirs("results/calibration", exist_ok=True)
    df.to_csv("results/calibration/pit_diagnostics_ranks.csv", index=False)
    summary.to_csv("results/calibration/pit_diagnostics_summary.csv", index=False)

    print("\n=== H1: PIT in isolation, mass versus density input ===")
    print(test_pit_mass_vs_density().to_string(index=False))
    print("\n=== H2/H3: rank by route, K = 1, n fixed ===")
    print(summary.to_string(index=False))
    print("\n=== posterior width ===")
    print(f"mean sd(log q) = {df.sd_log_q.mean():.4f}   "
          f"median cells per sd = {df.cells_per_sd.median():.2f}")
    print("\nmax |rank_current - rank_p_direct| = "
          f"{(df.rank_current - df.rank_p_direct).abs().max():.4f}")
    print("mean  rank_current - rank_p_direct  = "
          f"{(df.rank_current - df.rank_p_direct).mean():.4f}")
    return summary


if __name__ == "__main__":
    run()
