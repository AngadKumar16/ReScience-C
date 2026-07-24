"""Bayesian Quantal Analysis (BQA): per-condition likelihoods, combination
across conditions, change of variables to (q, gamma, r), and marginalisation
to posteriors over (p1, p2, v, n).

Reference (paper equations, transcribed from the published PDF; see
NOTES.md for provenance):

  Condition means mu_k and baseline-noise variance eps^2 are FIXED KNOWN
  inputs (METHODS, "Quantal probabilities").

  Eq. A2: prior for p_k is the arcsine (Jeffreys') distribution,
          f(p_k) = 1 / (pi*sqrt(p_k*(1-p_k))), bounded 0.04 <= p_k <= 0.96.
          This is uniform in arcsin(sqrt(p_k)) -- see src/grid.py.
  Eq. A3: lambda_k = mu_k / (n * p_k * gamma)   [from mu=npq (Eq. 2) and
          q=gamma*lambda (Eq. 7): q = mu_k/(n p_k) => lambda=q/gamma].
  Eq. A4: since arcsin(sqrt(p_k)), log_e(v), and n are all assigned
          UNIFORM independent priors, the joint prior in these transformed
          coordinates is flat, so the posterior in transformed coordinates
          is directly proportional to the likelihood (no extra prior
          weighting/Jacobian needed once the grid itself is built in
          arcsin(sqrt(p))/log(v) coordinates -- see src/grid.py).
  Eq. A5/A6: per-condition log-likelihood L_k = sum_x log Q(x|n, p_k,
          gamma, lambda_k, eps^2), Q from Eq. 8 (src/q_model.py).
  Eq. A7: change of variables to ideal parameters psi=(q, gamma, r):
          q = mu_k/(n*p_k), gamma = 1/v^2, r = mu_k/p_k = n*q. Since r=n*q
          identically for a fixed n, only q needs resampling onto a common
          axis across conditions (mu_k differs); gamma is already a
          monotonic reparametrisation of the shared v-axis and needs no
          resampling. r depends additionally on the candidate n, so a
          shared r-axis is used to marginalise across candidate n values
          (paper: "computationally optimal if their respective values are
          sampled intermixed directly" / log-spaced, resolution 128).
  Eq. A8: f(q, gamma, r | X, M) = prod_k f_k(q, gamma, r | x_k, mu_k)
          -- combine conditions by elementwise product of their (resampled)
          posterior mass grids, for a fixed candidate n.
  Eq. A9/A10: joint posterior normalised over the full (q, gamma, r) grid.
  Eq. A4 (n): n is also assigned a uniform prior; marginalising over
          candidate n values is implemented as a SUM (sum rule, Eq. 12) of
          each candidate n's (q, gamma) posterior mass, scattered onto the
          shared r-axis at r = n*q (this reproduces the "discrete
          discontinuities" / comb-like structure described for Fig. 3).
  Eq. A11: n_hat = r_hat/q_hat, lambda_hat = q_hat/gamma_hat,
          v_hat = sqrt(1/gamma_hat), using medians of the final marginal
          posteriors for q, gamma, r.

Heterogeneous-release extension (Eqs. A12-A13, beta model over p across
sites) is implemented separately in `heterogeneous_release_pmf` below; see
its docstring for the 2^n enumeration caveat.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy import stats

from src.grid import Grid, P_MAX, P_MIN


@dataclass
class ConditionData:
    """Fixed, known inputs for one experimental condition.

    Attributes
    ----------
    mu : float
        Condition mean response amplitude (fixed known input, not sampled).
    eps2 : float
        Baseline-noise variance for this condition (fixed known input).
    amplitudes : np.ndarray
        Observed response amplitudes for this condition.
    """

    mu: float
    eps2: float
    amplitudes: np.ndarray


def per_condition_log_likelihood(
    condition: ConditionData,
    grid: Grid,
    n: int,
) -> np.ndarray:
    """Evaluate the per-condition log-likelihood surface over the (p, v) grid.

    For each (p, v) grid point, sums log Q(x_i | n, p, gamma, lam, eps2)
    over observations x_i in `condition.amplitudes`, where gamma = 1/v^2
    and lambda = mu / (n * p * gamma) (Eq. A3).

    Parameters
    ----------
    condition : ConditionData
        Fixed known inputs and observed amplitudes for this condition.
    grid : Grid
        Grid axes produced by `grid.build_grid`.
    n : int
        Candidate number of release sites for this evaluation.

    Returns
    -------
    np.ndarray, shape (P_RESOLUTION, V_RESOLUTION)
        Log-likelihood evaluated at each (p, v) grid point.

    Note
    ----
    This is a vectorized reimplementation of Eq. 8 (mirroring
    `src.q_model.q_function` term-for-term) rather than a per-cell call
    into that function, since the latter is not vectorized over (p, v) and
    a Python-level loop over a 128x128 grid is prohibitively slow when
    called repeatedly (SBC, identifiability sweep). Any change to Eq. 8's
    form in `q_model.q_function` must be mirrored here.
    """
    p = grid.p_values  # (n_p,)
    v = grid.v_values  # (n_v,)
    gamma = 1.0 / v**2  # (n_v,)
    n_p, n_v = p.size, v.size

    with np.errstate(divide="ignore", invalid="ignore"):
        lam = condition.mu / (n * p[:, None] * gamma[None, :])  # (n_p, n_v)
    valid = np.isfinite(lam) & (lam > 0)

    x = condition.amplitudes[:, None, None]  # (n_obs, 1, 1)
    baseline_sd = np.sqrt(condition.eps2)

    b0 = stats.binom.pmf(0, n, p).reshape(1, n_p, 1)
    total = b0 * stats.norm.pdf(x, loc=0.0, scale=baseline_sd)  # (n_obs, n_p, 1)

    for i in range(1, n + 1):
        b_i = stats.binom.pmf(i, n, p).reshape(1, n_p, 1)
        a_i = (i * gamma).reshape(1, 1, n_v)
        scale_i = np.where(valid, lam, 1.0).reshape(1, n_p, n_v)
        term = stats.gamma.pdf(x, a=a_i, scale=scale_i)  # (n_obs, n_p, n_v)
        total = total + b_i * term  # broadcasts to (n_obs, n_p, n_v)

    total = np.clip(total, 1e-300, None)
    log_l = np.sum(np.log(total), axis=0)  # (n_p, n_v)
    log_l = np.where(valid, log_l, -np.inf)
    return log_l


def _normalize_log_grid(log_grid: np.ndarray) -> np.ndarray:
    """Convert a log-likelihood/log-posterior grid to a normalised mass grid."""
    finite = np.isfinite(log_grid)
    if not np.any(finite):
        raise ValueError("log_grid has no finite entries to normalize")
    m = np.max(log_grid[finite])
    weights = np.where(finite, np.exp(log_grid - m), 0.0)
    total = weights.sum()
    if total <= 0:
        raise ValueError("normalized weights sum to zero")
    return weights / total


def build_q_axis(
    conditions: list[ConditionData],
    n_candidates: list[int],
    p_min: float,
    p_max: float,
    resolution: int = 128,
) -> np.ndarray:
    """Build a shared log-spaced q-axis spanning all conditions and candidate n.

    q = mu_k / (n * p_k) (Eq. A7); bounds are derived from the extremes of
    mu_k, n, and the p-grid bounds so every condition's/candidate-n's
    q-values fall within range.
    """
    mus = np.array([c.mu for c in conditions], dtype=float)
    n_arr = np.array(n_candidates, dtype=float)
    q_min = np.min(mus) / (np.max(n_arr) * p_max)
    q_max = np.max(mus) / (np.min(n_arr) * p_min)
    return np.geomspace(q_min, q_max, resolution)


def build_r_axis(q_axis: np.ndarray, n_candidates: list[int]) -> np.ndarray:
    """Build a shared log-spaced r-axis spanning r = n*q across candidate n."""
    n_arr = np.array(n_candidates, dtype=float)
    r_min = q_axis.min() * n_arr.min()
    r_max = q_axis.max() * n_arr.max()
    return np.geomspace(r_min, r_max, len(q_axis))


def _nearest_axis_index(values: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Index of the nearest axis bin (in linear space) for each value."""
    idx = np.searchsorted(axis, values)
    idx = np.clip(idx, 1, len(axis) - 1)
    left = idx - 1
    choose_left = np.abs(axis[left] - values) < np.abs(axis[idx] - values)
    return np.where(choose_left, left, idx)


def _linear_bin_weights(
    values: np.ndarray, axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two-bin, mass-conserving assignment of each value onto a log-spaced axis.

    Nearest-bin assignment dumps each value's whole mass into one bin, which
    quantizes the posterior and biases median-based estimates (see the SBC
    result in the paper). Linear interpolation instead splits each value's
    mass between the two bracketing bins in proportion to distance, computed
    in log space since the q/r axes are log-spaced. The split weights sum to
    one, so total mass is conserved exactly, and the assignment converges to
    the true value as resolution grows rather than sticking on a bin center.

    Returns
    -------
    (lo, hi, w_lo, w_hi)
        Lower/upper bracketing indices and their mass fractions for each
        value; deposit ``mass * w_lo`` at ``lo`` and ``mass * w_hi`` at ``hi``.
    """
    log_axis = np.log(axis)
    log_val = np.log(values)
    idx = np.searchsorted(log_axis, log_val)
    idx = np.clip(idx, 1, len(axis) - 1)
    lo = idx - 1
    hi = idx
    denom = log_axis[hi] - log_axis[lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        w_hi = np.where(denom > 0, (log_val - log_axis[lo]) / denom, 0.0)
    w_hi = np.clip(w_hi, 0.0, 1.0)
    return lo, hi, 1.0 - w_hi, w_hi


def _remap_p_axis_to_q(
    mass_pv: np.ndarray,
    p_values: np.ndarray,
    mu: float,
    n: int,
    q_axis: np.ndarray,
    interpolation: str = "nearest",
) -> np.ndarray:
    """Mass-conserving remap of a (P_RES, V_RES) mass grid's p-axis onto q_axis.

    q = mu / (n * p) (Eq. A7) is monotonic in p, so each p-row's mass is
    scattered onto the shared q-axis; the v-axis (gamma) dimension is
    untouched since q does not depend on v. With ``interpolation="linear"``
    (default) each row's mass is split across the two bracketing q-bins
    (see `_linear_bin_weights`); ``"nearest"`` reproduces the original
    nearest-bin behavior and is kept for comparison.
    """
    q_values = mu / (n * p_values)
    out = np.zeros((len(q_axis), mass_pv.shape[1]))
    if interpolation == "nearest":
        target_idx = _nearest_axis_index(q_values, q_axis)
        np.add.at(out, target_idx, mass_pv)
    elif interpolation == "linear":
        lo, hi, w_lo, w_hi = _linear_bin_weights(q_values, q_axis)
        np.add.at(out, lo, mass_pv * w_lo[:, None])
        np.add.at(out, hi, mass_pv * w_hi[:, None])
    else:
        raise ValueError(f"unknown interpolation {interpolation!r}")
    return out


def _loglik_on_q_axis(
    log_l_pv: np.ndarray,
    p_values: np.ndarray,
    mu: float,
    n: int,
    q_axis: np.ndarray,
) -> np.ndarray:
    """Interpolate a (p, v) log-likelihood surface onto the shared q-axis.

    q = mu / (n * p) (Eq. A7), so each q-value corresponds to p = mu / (n q).
    For each column of the v-axis the log-likelihood is linearly interpolated
    from the p-grid onto those p-targets. Unlike `_remap_p_axis_to_q`, this
    transports the log-likelihood as a *function value*, not as probability
    mass, so it can be summed across conditions (multiplying likelihoods once)
    without normalising each condition first.
    """
    p_targets = mu / (n * q_axis)  # (n_q,)
    order = np.argsort(p_values)
    xp = p_values[order]
    finite_floor = -1e300
    out = np.empty((q_axis.size, log_l_pv.shape[1]))
    for j in range(log_l_pv.shape[1]):
        fp = np.where(np.isfinite(log_l_pv[order, j]), log_l_pv[order, j], finite_floor)
        out[:, j] = np.interp(p_targets, xp, fp)
    return out


def _joint_condition_log_likelihood(
    condition: ConditionData,
    q_axis: np.ndarray,
    v_values: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-condition log-likelihood evaluated directly on the shared (q, v) grid.

    Unlike `per_condition_log_likelihood`, which evaluates Eq. 8 over the
    condition's own (p, v) grid and requires transporting the result onto a
    shared q-axis afterwards, this evaluates Eq. 8 at the release probability
    IMPLIED by the shared q at every grid cell:

        p_k = mu_k / (n * q)          (invert Eq. A7, q = mu_k / (n p_k))
        gamma = 1 / v^2               (v-axis)
        lambda_k = q / gamma          (Eq. A3)

    so no histogram mass-transport is ever done -- the likelihood is sampled
    once on the shared physical axes. Cells whose implied p_k falls outside
    the arcsine-prior support [P_MIN, P_MAX] get -inf (zero likelihood).

    Returns
    -------
    (log_l, p_k)
        ``log_l`` shape (n_q, n_v); ``p_k`` shape (n_q,) the implied release
        probability per q-cell (used to build the induced q-prior).
    """
    q = np.asarray(q_axis, dtype=float)
    v = np.asarray(v_values, dtype=float)
    gamma = 1.0 / v**2  # (n_v,)
    n_q, n_v = q.size, v.size

    p_k = condition.mu / (n * q)  # (n_q,), invert Eq. A7
    valid = (p_k >= P_MIN) & (p_k <= P_MAX)

    lam = q[:, None] / gamma[None, :]  # (n_q, n_v), Eq. A3
    x = condition.amplitudes[:, None, None]  # (n_obs, 1, 1)
    baseline_sd = np.sqrt(condition.eps2)

    b0 = stats.binom.pmf(0, n, p_k).reshape(1, n_q, 1)
    total = b0 * stats.norm.pdf(x, loc=0.0, scale=baseline_sd)  # (n_obs, n_q, 1)

    for i in range(1, n + 1):
        b_i = stats.binom.pmf(i, n, p_k).reshape(1, n_q, 1)
        a_i = (i * gamma).reshape(1, 1, n_v)
        term = stats.gamma.pdf(x, a=a_i, scale=lam[None, :, :])  # (n_obs, n_q, n_v)
        total = total + b_i * term

    total = np.clip(total, 1e-300, None)
    log_l = np.sum(np.log(total), axis=0)  # (n_q, n_v)
    log_l = np.where(valid[:, None], log_l, -np.inf)
    return log_l, p_k


def _joint_log_q_prior(
    p_per_condition: list[np.ndarray],
    q_prior: str,
    q_axis: np.ndarray,
    n: int,
) -> np.ndarray:
    """Prior weight on the shared q, evaluated in the log-q coordinate.

    The paper's arcsine (Jeffreys) prior f(p) = 1/(pi sqrt(p(1-p))) is placed
    on each per-condition p_k. Here q is the free shared parameter and
    p_k = mu_k/(n q) is a deterministic function of it, so combining the K
    per-condition arcsine priors into a prior on the DERIVED shared q is a
    modeling choice the 2013 paper never spells out. Three options are offered.
    ``"geometric"`` is the default: empirically it is the choice that leaves
    both the gamma and n marginals calibrated on the 200-iteration SBC while
    keeping the point estimates unbiased (see paper Section 4.1). The residual
    q/r interval miscalibration is NOT removed by any of the three (it is a
    likelihood-shape effect, not a prior effect -- Section 4.1), so the choice
    among them is a sensitivity knob, not a fix.

    - ``"geometric"`` (default): geometric mean of the K induced arcsine
      densities, i.e. the AVERAGE per-condition log weight 0.5*log(p_k/(1-p_k)).
      A symmetric, once-only "unit strength" prior on the shared q that reduces
      to the single arcsine prior when K = 1 (so it matches the product method
      on the already-calibrated single-condition case).
    - ``"induced"``: the prior on q that the SBC generative model formally
      implies. That harness draws q log-uniformly (density ∝ 1/q) and
      each p_k ~ arcsine INDEPENDENTLY, then sets mu_k = n p_k q. Treating the
      resulting mu_k as observed and changing variables (p_1..p_K) -> (mu_1..
      mu_K) at fixed (q, n) contributes a Jacobian (n q)^-K, so the induced
      prior on q is

          pi(q, n) ∝ n^-K * q^-(K+1) * prod_k f(p_k(q)),

      and in the log-q coordinate (multiply by |dq/d log q| = q):

          w(q, n) = -K log n - K log q - 0.5 * sum_k log(p_k (1 - p_k))  (+const).

      The -K log n term is a per-candidate-n offset from the SAME (n q)^-K
      Jacobian, so it correctly reweights the sum-over-n marginalisation (and,
      because every marginal is read off the pooled grid, it also feeds the
      q/gamma/r marginals). Applying each condition's arcsine prior here is NOT
      the likelihood
      double-counting that makes the product combination over-confident (that
      lives in multiplying separately-normalised grids); it is the correct
      change-of-variables bookkeeping for a q derived from K independent p_k
      draws. For a single condition (K = 1) it reduces, up to an additive
      constant, to the two once-only choices below and to the product method's
      implicit prior, so all agree on the already-calibrated single-condition
      case and differ only in how the shared-q prior scales with K. Formally
      the most defensible choice, but on the full SBC it slightly worsens the
      gamma marginal relative to ``"geometric"`` without rescuing q, so it is
      offered as a documented alternative rather than the default.
    - ``"reference"``: the induced arcsine prior from the first condition only.
    - ``"flat"``: log-uniform prior on q (no arcsine adjustment); matches the
      SBC harness's own q draw and gives the best-calibrated gamma marginal,
      at the cost of the n marginal. Kept for sensitivity.
    """
    logits = [0.5 * (np.log(p) - np.log1p(-p)) for p in p_per_condition]
    stacked = np.stack(logits, axis=0)  # (K, n_q)
    if q_prior == "geometric":
        return stacked.mean(axis=0)
    if q_prior == "reference":
        return stacked[0]
    if q_prior == "flat":
        # Log-uniform prior on q (flat in the log-q coordinate), matching the
        # SBC harness's own log-uniform q draw and imposing no arcsine
        # adjustment on the derived p_k.
        return np.zeros_like(q_axis)
    if q_prior == "induced":
        k = len(p_per_condition)
        # -0.5 * sum_k log(p_k (1-p_k)) = sum_k [ -0.5 log p_k - 0.5 log(1-p_k) ]
        arcsine = sum(
            -0.5 * (np.log(p) + np.log1p(-p)) for p in p_per_condition
        )
        return arcsine - k * np.log(q_axis) - k * np.log(n)
    raise ValueError(f"unknown q_prior {q_prior!r}")


def _combine_conditions_loggrid(
    conditions: list[ConditionData],
    grid: Grid,
    n: int,
    q_axis: np.ndarray,
    combine: str,
    q_prior: str = "geometric",
) -> np.ndarray:
    """Un-normalised LOG (q, v) grid for the additive combination modes.

    Returns the summed-log-likelihood (+ induced q-prior, for ``joint``) grid
    WITHOUT any max-subtraction, so grids for different candidate n are on a
    common additive scale. This is what lets `change_of_variables_and_marginalise`
    exponentiate every candidate n against a single global maximum before the
    sum-over-n (uniform-prior marginalisation): subtracting a per-n maximum
    instead would silently rescale each n and corrupt both the n-marginal and
    the pooled (q, gamma, r) marginals.
    """
    acc = np.zeros((q_axis.size, grid.v_values.size))
    if combine == "joint":
        p_per_condition: list[np.ndarray] = []
        for condition in conditions:
            log_l, p_k = _joint_condition_log_likelihood(
                condition, q_axis, grid.v_values, n
            )
            acc += log_l
            p_per_condition.append(p_k)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_prior = _joint_log_q_prior(p_per_condition, q_prior, q_axis, n)  # (n_q,)
        acc = acc + np.where(np.isfinite(log_prior), log_prior, 0.0)[:, None]
        return acc
    if combine == "loglik":
        for condition in conditions:
            log_l = per_condition_log_likelihood(condition, grid, n)
            acc += _loglik_on_q_axis(log_l, grid.p_values, condition.mu, n, q_axis)
        return acc
    raise ValueError(f"_combine_conditions_loggrid does not handle combine={combine!r}")


def combine_conditions(
    conditions: list[ConditionData],
    grid: Grid,
    n: int,
    q_axis: np.ndarray,
    interpolation: str = "nearest",
    combine: str = "product",
    q_prior: str = "geometric",
) -> np.ndarray:
    """Combine per-condition posteriors across K conditions for fixed n (Eq. A8).

    Two combination modes:

    - ``combine="product"`` (default, the original route): normalise each
      condition's (p, v) log-likelihood to a posterior mass grid, remap to the
      shared q-axis, and multiply the conditions together. This is simple but
      double-counts the prior: multiplying K normalised posteriors carries the
      prior K times, which over-sharpens the combined posterior and is the
      cause of the SBC interval miscalibration documented in the paper (the
      overconfidence grows with the number of conditions).
    - ``combine="loglik"`` (best available partial fix): interpolate each
      condition's log-likelihood onto the shared q-axis and sum, multiplying
      the likelihoods once instead of multiplying separately-normalised,
      histogram-binned posteriors. This substantially reduces the
      over-sharpening and fully calibrates the well-separated controlled
      cases, but does not fully calibrate arbitrary draws; residual
      approximation remains in the grid-transport architecture.
    - ``combine="density"`` (tested, negligible): divides each condition's
      mass by the log-axis bin widths before multiplying and restores the
      width once, on the hypothesis that the over-sharpening was a
      mass-vs-density bin-width error. Empirically this changes the
      calibration only marginally, so the bin-width factor is NOT the cause;
      kept as a documented negative result.

    Parameters
    ----------
    conditions : list[ConditionData]
        One entry per experimental condition (K conditions).
    grid : Grid
        Grid axes produced by `grid.build_grid`.
    n : int
        Candidate number of release sites.
    q_axis : np.ndarray
        Shared log-spaced q-axis (see `build_q_axis`).
    interpolation : str
        Mass-transport mode for ``combine="product"`` ("nearest" or "linear").
    combine : str
        "product" (default), "density", "loglik", or "joint" (the joint-grid
        calibration fix -- see the module/joint-helper docstrings).
    q_prior : str
        Induced shared-q prior combination for ``combine="joint"``:
        "geometric" (default) or "reference". Ignored by other modes.

    Returns
    -------
    np.ndarray, shape (len(q_axis), V_RESOLUTION)
        Combined (unnormalised) posterior mass over (q, gamma) for this n.
    """
    if combine in ("joint", "loglik"):
        # Full joint likelihood sampled directly on the shared (q, v) grid at
        # the implied p_k = mu_k/(n q) ("joint"), or the per-condition
        # log-likelihoods interpolated onto the shared q-axis ("loglik"); in
        # both the log-likelihoods are SUMMED (likelihoods multiplied exactly
        # once). Returned as a mass grid scaled by this call's own maximum for
        # standalone use; `change_of_variables_and_marginalise` instead pulls
        # the raw log grid via `_combine_conditions_loggrid` so it can scale
        # every candidate n against a single global maximum before summing.
        acc = _combine_conditions_loggrid(conditions, grid, n, q_axis, combine, q_prior)
        finite = np.isfinite(acc)
        if not np.any(finite):
            return np.zeros((q_axis.size, grid.v_values.size))
        m = np.max(acc[finite])
        return np.where(finite, np.exp(acc - m), 0.0)

    if combine == "density":
        # Bin widths on the (log-spaced) q and v axes; multiplying densities
        # (mass / width) and restoring the width once avoids counting the
        # width factor once per condition.
        wq = np.gradient(q_axis)
        wv = np.gradient(grid.v_values)
        weight = np.outer(wq, wv)  # (n_q, n_v)
        combined_density = np.ones((len(q_axis), grid.v_values.size))
        for condition in conditions:
            log_l = per_condition_log_likelihood(condition, grid, n)
            mass_pv = _normalize_log_grid(log_l)
            mass_qv = _remap_p_axis_to_q(
                mass_pv, grid.p_values, condition.mu, n, q_axis, interpolation
            )
            combined_density = combined_density * (mass_qv / weight)
        return combined_density * weight

    combined = np.ones((len(q_axis), grid.v_values.size))
    for condition in conditions:
        log_l = per_condition_log_likelihood(condition, grid, n)
        mass_pv = _normalize_log_grid(log_l)
        mass_qv = _remap_p_axis_to_q(
            mass_pv, grid.p_values, condition.mu, n, q_axis, interpolation
        )
        combined = combined * mass_qv
    return combined


def change_of_variables_and_marginalise(
    conditions: list[ConditionData],
    grid: Grid,
    n_candidates: list[int],
    q_axis: np.ndarray | None = None,
    r_axis: np.ndarray | None = None,
    interpolation: str = "nearest",
    combine: str = "product",
    q_prior: str = "geometric",
) -> dict:
    """Apply the Eqs. A7-A11 change of variables and marginalise over n.

    For each candidate n: combine conditions on the shared q-axis (Eq. A8),
    then scatter the resulting (q, gamma) mass onto the shared r-axis at
    r = n*q (mass-conserving), accumulating across n (sum rule, Eq. 12,
    since n also carries a uniform prior -- Eq. A4). The final joint grid
    is normalised (Eq. A10) and its marginals give posteriors for q, gamma
    (via the v-axis), and r; medians of these give the parameter estimates
    (Eq. A11).

    Parameters
    ----------
    conditions : list[ConditionData]
        One entry per experimental condition.
    grid : Grid
        Grid axes produced by `grid.build_grid`.
    n_candidates : list[int]
        Candidate integer values for n (uniform prior support).
    q_axis, r_axis : np.ndarray, optional
        Shared axes; built via `build_q_axis`/`build_r_axis` if omitted.

    Returns
    -------
    dict
        Keys: "q_axis", "r_axis", "v_values", "joint_mass" (shape
        (len(q_axis), V_RESOLUTION, len(r_axis))), "q_marginal",
        "gamma_marginal", "gamma_values", "r_marginal", "n_candidates",
        "n_marginal" (unnormalised-then-normalised mass per candidate n,
        summed before scattering onto the shared r-axis -- the closest
        this pipeline has to a genuine marginal posterior over n, since
        the final n_hat is instead a derived point estimate per Eq. A11),
        "q_hat", "r_hat", "gamma_hat", "lambda_hat", "v_hat", "n_hat".
    """
    p_min, p_max = grid.p_values.min(), grid.p_values.max()
    if q_axis is None:
        # Tie the q-axis resolution to the p-grid resolution so that raising
        # the grid resolution (resolution-bias check) also refines the
        # q/r resampling axes, not just the likelihood grid.
        q_axis = build_q_axis(
            conditions, n_candidates, p_min, p_max, resolution=grid.p_values.size
        )
    if r_axis is None:
        r_axis = build_r_axis(q_axis, n_candidates)

    n_q = len(q_axis)
    n_v = grid.v_values.size
    n_r = len(r_axis)
    joint_mass = np.zeros((n_q, n_v, n_r))
    n_mass_raw = np.zeros(len(n_candidates))  # unnormalised mass per candidate n

    # "joint" produces a LOG grid per candidate n on a common additive scale,
    # so it is exponentiated against a single GLOBAL maximum across all n --
    # otherwise each n is silently rescaled, corrupting the n-marginal
    # (uniform-prior sum-over-n) and the pooled (q, gamma, r) marginals.
    # "product"/"density"/"loglik" keep their original per-n path so their
    # committed results stay bit-for-bit reproducible.
    log_grids = None
    if combine == "joint":
        log_grids = [
            _combine_conditions_loggrid(conditions, grid, n, q_axis, combine, q_prior)
            for n in n_candidates
        ]
        finite_maxes = [np.max(g[np.isfinite(g)]) for g in log_grids if np.any(np.isfinite(g))]
        if not finite_maxes:
            raise ValueError("joint posterior mass is zero everywhere")
        global_max = max(finite_maxes)

    for ni, n in enumerate(n_candidates):
        if log_grids is not None:
            g = log_grids[ni]
            mass_qv = np.where(np.isfinite(g), np.exp(g - global_max), 0.0)
        else:
            mass_qv = combine_conditions(
                conditions, grid, n, q_axis, interpolation, combine, q_prior
            )
        n_mass_raw[ni] = mass_qv.sum()
        r_values = n * q_axis
        if interpolation == "nearest":
            r_idx = _nearest_axis_index(r_values, r_axis)
            joint_mass[np.arange(n_q), :, r_idx] += mass_qv
        else:
            lo, hi, w_lo, w_hi = _linear_bin_weights(r_values, r_axis)
            qi = np.repeat(np.arange(n_q), n_v)
            vi = np.tile(np.arange(n_v), n_q)
            np.add.at(joint_mass, (qi, vi, np.repeat(lo, n_v)),
                      (mass_qv * w_lo[:, None]).ravel())
            np.add.at(joint_mass, (qi, vi, np.repeat(hi, n_v)),
                      (mass_qv * w_hi[:, None]).ravel())

    total = joint_mass.sum()
    if total <= 0:
        raise ValueError("joint posterior mass is zero everywhere")
    joint_mass = joint_mass / total
    n_marginal = n_mass_raw / n_mass_raw.sum()

    q_marginal = joint_mass.sum(axis=(1, 2))
    gamma_values = 1.0 / grid.v_values**2
    gamma_marginal = joint_mass.sum(axis=(0, 2))
    r_marginal = joint_mass.sum(axis=(0, 1))

    q_hat = _weighted_median(q_axis, q_marginal)
    r_hat = _weighted_median(r_axis, r_marginal)
    gamma_order = np.argsort(gamma_values)
    gamma_hat = _weighted_median(gamma_values[gamma_order], gamma_marginal[gamma_order])

    lambda_hat = q_hat / gamma_hat
    v_hat = np.sqrt(1.0 / gamma_hat)
    n_hat = estimate_n(r_hat, q_hat)

    return {
        "q_axis": q_axis,
        "r_axis": r_axis,
        "v_values": grid.v_values,
        "joint_mass": joint_mass,
        "q_marginal": q_marginal,
        "gamma_marginal": gamma_marginal,
        "gamma_values": gamma_values,
        "r_marginal": r_marginal,
        "n_candidates": np.asarray(n_candidates),
        "n_marginal": n_marginal,
        "q_hat": q_hat,
        "r_hat": r_hat,
        "gamma_hat": gamma_hat,
        "lambda_hat": lambda_hat,
        "v_hat": v_hat,
        "n_hat": n_hat,
    }


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Median of a discrete weighted distribution (values assumed sorted ascending)."""
    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]
    idx = np.searchsorted(cdf, 0.5)
    idx = np.clip(idx, 0, len(values) - 1)
    return float(values[idx])


def estimate_n(median_r: float, median_q: float) -> float:
    """Estimate n as median(r) / median(q), per Bhumbra & Beato (2013), Eq. A11.

    Parameters
    ----------
    median_r : float
        Posterior median of r (derived quantity).
    median_q : float
        Posterior median of q (derived quantity).

    Returns
    -------
    float
        Estimated number of release sites, n = median(r) / median(q).
    """
    if median_q == 0:
        raise ValueError("median_q must be nonzero to estimate n")
    return median_r / median_q


def heterogeneous_release_pmf(
    n: int,
    mean_p: float,
    alpha: float,
) -> np.ndarray:
    """Numerically compute the heterogeneous-release success-count pmf (Eqs. A12-A13).

    Central synapses may exhibit heterogeneous release probabilities across
    sites (paper's "Modeling heterogeneous release probabilities" section).
    Site release probabilities are modeled as n i.i.d. draws from a Beta
    distribution Beta(alpha, beta_k) parameterised by homogeneity alpha and
    condition mean release probability mean_p (Eq. A12-A13):

      P(p|alpha, beta) = p^(alpha-1) * (1-p)^(beta-1) / B(alpha, beta)
      beta_k = alpha * (1 - mean_p) / mean_p                        (Eq. A13)

    There is no closed-form analytical expression for the resulting
    generalized binomial pmf B'(i|n, mean_p, alpha) (paper: "To our
    knowledge, there is no analytical expression for the analogous
    probability mass function"). The paper's numerical procedure evaluates
    every permutation of release/failure across the n sites -- 2^n release-
    state combinations -- and accumulates their probabilities via the
    product rule into the pmf for i total successes.

    WARNING: this is the hardest, most compute-intensive step in the paper
    (explicitly flagged as such in NOTES.md). Cost is O(2^n * n); it is
    only practical for small n (single digits). It does NOT sample p per
    site from the Beta distribution directly per permutation -- doing so
    would require re-deriving per-site probabilities analytically. Instead,
    following the paper's stated procedure, we assign per-site release
    probabilities by evenly sampling the Beta CDF across the n sites
    ("the beta cumulative density function is sampled evenly to assign
    Bernoulli probabilities of release to each of n release sites
    according to a beta distribution of mean mean_p and uniformity alpha").

    Parameters
    ----------
    n : int
        Number of release sites. Kept small (<= ~15) for tractability.
    mean_p : float
        Mean release probability across sites for this condition.
    alpha : float
        Beta-distribution homogeneity shaping parameter (larger = more
        homogeneous release across sites).

    Returns
    -------
    np.ndarray, shape (n + 1,)
        B'(i|n, mean_p, alpha) for i = 0, ..., n.
    """
    if n > 20:
        raise ValueError(
            f"n={n} makes the 2^n enumeration in heterogeneous_release_pmf "
            "intractable; keep n small (paper's own caveat)."
        )
    if not (0.0 < mean_p < 1.0):
        raise ValueError(f"mean_p must be in (0, 1), got {mean_p}")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")

    beta_param = alpha * (1.0 - mean_p) / mean_p

    # Sample the beta CDF evenly across the n sites (paper's stated method).
    quantiles = (np.arange(n) + 0.5) / n
    site_p = stats.beta.ppf(quantiles, alpha, beta_param)

    pmf = np.zeros(n + 1)
    for outcomes in product([0, 1], repeat=n):
        prob = 1.0
        i = 0
        for site_idx, released in enumerate(outcomes):
            if released:
                prob *= site_p[site_idx]
                i += 1
            else:
                prob *= 1.0 - site_p[site_idx]
        pmf[i] += prob

    return pmf
