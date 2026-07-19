"""Grid construction for BQA brute-force inference over (p1, p2, v, n).

Bhumbra & Beato (2013) infer parameters by brute-force grid evaluation, not
MCMC. This module builds the fixed grid axes and their forward/inverse
transforms + Jacobians:

  - p-axis: parameterised as arcsin(sqrt(p)), resolution 128, with
    0.04 <= p <= 0.96 (arcsine/Jeffreys prior on p is flat in this
    transformed coordinate).
  - v-axis: parameterised as log_e(v), resolution 128, with
    0.05 <= v <= 1 (uniform prior on log(v)).
  - n-axis: enumerated integers (release-site counts), UNIFORM prior
    (the 2013 paper; later 2018/2019 work switches to Jeffreys' -- do not
    use that here).

This is generic grid bookkeeping (not the paper's inferential method) and is
implemented fully.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

P_MIN = 0.04
P_MAX = 0.96
V_MIN = 0.05
V_MAX = 1.0
P_RESOLUTION = 128
V_RESOLUTION = 128


@dataclass
class Grid:
    """Grid axes over transformed (p, v) coordinates and enumerated n.

    Attributes
    ----------
    p_transformed : np.ndarray, shape (P_RESOLUTION,)
        arcsin(sqrt(p)) axis values, evenly spaced.
    p_values : np.ndarray, shape (P_RESOLUTION,)
        Corresponding p values (inverse-transformed), in [P_MIN, P_MAX].
    v_transformed : np.ndarray, shape (V_RESOLUTION,)
        log_e(v) axis values, evenly spaced.
    v_values : np.ndarray, shape (V_RESOLUTION,)
        Corresponding v values (inverse-transformed), in [V_MIN, V_MAX].
    n_values : np.ndarray
        Enumerated integer release-site counts.
    """

    p_transformed: np.ndarray
    p_values: np.ndarray
    v_transformed: np.ndarray
    v_values: np.ndarray
    n_values: np.ndarray


def p_to_arcsin_sqrt(p: np.ndarray) -> np.ndarray:
    """Forward transform: p -> arcsin(sqrt(p))."""
    return np.arcsin(np.sqrt(p))


def arcsin_sqrt_to_p(theta: np.ndarray) -> np.ndarray:
    """Inverse transform: arcsin(sqrt(p)) -> p."""
    return np.sin(theta) ** 2


def p_jacobian(p: np.ndarray) -> np.ndarray:
    """d(theta)/d(p) where theta = arcsin(sqrt(p)); i.e. |dtheta/dp|.

    Used to convert a density in p to a density in the transformed
    coordinate (or vice versa via its reciprocal).
    """
    p = np.asarray(p, dtype=float)
    return 1.0 / (2.0 * np.sqrt(p * (1.0 - p)))


def v_to_log(v: np.ndarray) -> np.ndarray:
    """Forward transform: v -> log_e(v)."""
    return np.log(v)


def log_to_v(log_v: np.ndarray) -> np.ndarray:
    """Inverse transform: log_e(v) -> v."""
    return np.exp(log_v)


def v_jacobian(v: np.ndarray) -> np.ndarray:
    """d(log v)/d(v) = 1/v; i.e. |d(log v)/dv|."""
    return 1.0 / np.asarray(v, dtype=float)


def build_grid(n_values: np.ndarray) -> Grid:
    """Construct the full BQA grid: p axis, v axis, and enumerated n.

    Parameters
    ----------
    n_values : np.ndarray
        Integer release-site counts to enumerate (the n-axis is a caller-
        supplied enumeration, not a fixed resolution, since candidate n
        ranges depend on the dataset/experiment design).

    Returns
    -------
    Grid
        Named structure holding both transformed and natural-scale axis
        values for p and v, plus the enumerated n values.
    """
    n_values = np.asarray(n_values, dtype=int)
    if n_values.ndim != 1 or n_values.size == 0:
        raise ValueError("n_values must be a non-empty 1-D array of integers")
    if np.any(n_values < 1):
        raise ValueError("n_values must all be >= 1")

    theta_min = p_to_arcsin_sqrt(np.array(P_MIN))
    theta_max = p_to_arcsin_sqrt(np.array(P_MAX))
    p_transformed = np.linspace(theta_min, theta_max, P_RESOLUTION)
    p_values = arcsin_sqrt_to_p(p_transformed)

    log_v_min = v_to_log(np.array(V_MIN))
    log_v_max = v_to_log(np.array(V_MAX))
    v_transformed = np.linspace(log_v_min, log_v_max, V_RESOLUTION)
    v_values = log_to_v(v_transformed)

    return Grid(
        p_transformed=p_transformed,
        p_values=p_values,
        v_transformed=v_transformed,
        v_values=v_values,
        n_values=n_values,
    )
