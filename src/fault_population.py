"""Fault-population models and their normalizers for BJSI.

The joint plane-selection likelihood of :mod:`bjsi` marginalizes the nodal-plane
ambiguity of each focal mechanism with a two-component mixture.  Historically the
mixture weights were a logistic function of the instability contrast,

    pi_2 = sigmoid(beta * (I_2 - I_1)),

and the mixture was used without a normalizer.  That term is not a probability
density in the observed mechanism because the weight depends on the observation
through ``(I_1, I_2)``.  Its total mass over mechanisms varies with ``(R, mu)``,
which rewards large friction independently of the data.

This module provides the generative alternative.  A fault normal is drawn from
the sphere with density proportional to a population weight ``g(I(n))`` and the
slip is drawn from the directional likelihood about the resolved shear traction.
Marginalizing the unknown plane label gives, for one event,

    p(x | theta) = [g(I_1) L_1 + g(I_2) L_2] / Z(theta),
    Z(theta)     = E_n[ g(I(n)) ],

where the expectation is over normals uniform on the sphere.  The conditional
plane probability is ``g(I_2) / (g(I_1) + g(I_2))``, so ``g = exp(beta I)``
reproduces the historical weights exactly while supplying the missing ``Z``.

``Z`` does not depend on the stress orientation because the uniform measure is
rotation invariant, so it is a function of ``(R, mu)`` and of the parameters of
``g`` only.  It is tabulated once by quasi-Monte Carlo and read by multilinear
interpolation inside the model graph.  Tables are cached on disk.

Families
--------
``exp``
    ``g(I) = exp(beta * I)``.  The generative model whose conditional plane
    probabilities equal the historical logistic weights.
``ramp``
    ``g(I; I_min) = softplus(k (I - I_min)) / (k (1 - I_min))``, a smooth form of
    the Mohr-Coulomb acceptance rule ``max(0, I - I_min) / (1 - I_min)``.
    ``I_min`` may be fixed or inferred jointly with the stress.

Both families may be mixed with a component that does not depend on the stress,

    g_mix(n) = w g(I(n)) + (1 - w) h(n),

where ``h`` is either the uniform density (``mix_uniform``) or a mixture of
axial Watson clusters representing an inherited fault fabric (``fabric_K``).
Each component of ``h`` is normalized against the uniform measure, so the
normalizer of the mixture is ``w Z + (1 - w)``.  The weight ``w`` is the fraction
of the orientation distribution explained by stress selection and is inferred.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytensor.tensor as pt

__all__ = [
    "resolve_population_spec",
    "instability_numpy",
    "instability_pt",
    "log_population_weight_pt",
    "log_watson_mixture_pt",
    "normalizer_table",
    "interp_normalizer_pt",
    "cache_dir",
]

LEGACY = "legacy"
FAMILIES = (LEGACY, "exp", "ramp")

_DEFAULT_R_POINTS = 101
_DEFAULT_MU_POINTS = 81
_DEFAULT_IMIN_POINTS = 34
_DEFAULT_IMIN_MAX = 0.99
_DEFAULT_POWER = 16
_DEFAULT_SEED = 20260916
_DEFAULT_RAMP_K = 100.0
# Bump when the definition of any population weight changes, so that cached
# normalizer tables built with an earlier definition are not reused.
_TABLE_VERSION = 2
_LOG_SOFTPLUS_LINEAR_BELOW = -30.0


# ----------------------------------------------------------------------------
# Specification
# ----------------------------------------------------------------------------
def resolve_population_spec(
    fault_population: Any,
    *,
    selection_beta: float,
    friction_range: Tuple[float, float],
) -> Dict[str, Any]:
    """Return a fully populated specification dictionary.

    ``fault_population`` may be ``None`` or ``"legacy"`` for the historical
    unnormalized mixture, a family name, or a dictionary with the keys below.

    family : {'legacy', 'exp', 'ramp'}
    beta : float, for the 'exp' family; defaults to the model's selection beta
    imin : float or 'infer', for the 'ramp' family
    imin_bounds : (low, high) prior support when ``imin='infer'``
    k : float, sharpness of the ramp softplus
    normalize : bool, add the ``-N log Z`` potential
    mix_uniform : bool, add a uniform component with inferred weight
    fabric_K : int, number of Watson fabric clusters, 0 to disable
    fabric_kappa_sigma : float, half-normal prior scale for the concentrations
    table : dict, overrides for the normalizer grid and sampling
    """
    if fault_population is None:
        spec: Dict[str, Any] = {"family": LEGACY}
    elif isinstance(fault_population, str):
        spec = {"family": fault_population.strip().lower()}
    elif isinstance(fault_population, dict):
        spec = {str(k).strip().lower(): v for k, v in fault_population.items()}
        spec["family"] = str(spec.get("family", "ramp")).strip().lower()
    else:
        raise TypeError("fault_population must be None, a family name, or a dict")

    family = spec["family"]
    if family in {"none", "off"}:
        family = spec["family"] = LEGACY
    if family not in FAMILIES:
        raise ValueError(f"fault_population family must be one of {FAMILIES}, got {family!r}")

    if family == LEGACY:
        return {"family": LEGACY, "normalize": False, "mix_uniform": False, "fabric_K": 0}

    out: Dict[str, Any] = {
        "family": family,
        "normalize": bool(spec.get("normalize", True)),
        "mix_uniform": bool(spec.get("mix_uniform", False)),
        "fabric_K": int(spec.get("fabric_k", spec.get("fabric_K", 0))),
        "fabric_kappa_sigma": float(spec.get("fabric_kappa_sigma", 40.0)),
        "table": dict(spec.get("table", {})),
    }
    if out["fabric_K"] < 0:
        raise ValueError("fabric_K must be >= 0")
    if out["mix_uniform"] and out["fabric_K"] > 0:
        raise ValueError("Use either mix_uniform or fabric_K, not both")

    if family == "exp":
        out["beta"] = float(spec.get("beta", selection_beta))
        if not np.isfinite(out["beta"]) or out["beta"] < 0.0:
            raise ValueError("beta must be finite and >= 0")
    else:
        imin = spec.get("imin", "infer")
        out["k"] = float(spec.get("k", _DEFAULT_RAMP_K))
        if out["k"] <= 0.0:
            raise ValueError("k must be > 0")
        if isinstance(imin, str):
            if imin.strip().lower() != "infer":
                raise ValueError("imin must be a float or 'infer'")
            out["imin"] = "infer"
            lo, hi = spec.get("imin_bounds", (0.0, _DEFAULT_IMIN_MAX))
            out["imin_bounds"] = (float(lo), float(hi))
            if not 0.0 <= out["imin_bounds"][0] < out["imin_bounds"][1] < 1.0:
                raise ValueError("imin_bounds must satisfy 0 <= low < high < 1")
        else:
            out["imin"] = float(imin)
            if not 0.0 <= out["imin"] < 1.0:
                raise ValueError("imin must lie in [0, 1)")

    table = out["table"]
    table.setdefault("R_points", _DEFAULT_R_POINTS)
    table.setdefault("mu_points", _DEFAULT_MU_POINTS)
    table.setdefault("imin_points", _DEFAULT_IMIN_POINTS)
    table.setdefault("power", _DEFAULT_POWER)
    table.setdefault("seed", _DEFAULT_SEED)
    table.setdefault("mu_range", (float(friction_range[0]), float(friction_range[1])))
    return out


# ----------------------------------------------------------------------------
# NumPy side: instability, population weights, normalizer tables
# ----------------------------------------------------------------------------
def sobol_unit_normals(power: int, seed: int) -> np.ndarray:
    """Return ``2**power`` unit vectors, area-uniform on the sphere."""
    from scipy.stats import qmc

    u = qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(int(power))
    z = 2.0 * u[:, 0] - 1.0
    phi = 2.0 * np.pi * u[:, 1]
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((r * np.cos(phi), r * np.sin(phi), z))


def instability_numpy(n: np.ndarray, R: float, mu: float) -> np.ndarray:
    """Normalized Mohr-Coulomb instability of normals ``n`` in the principal frame.

    The reduced stress tensor is ``diag(-1, 2R - 1, 1)`` with compression
    negative, matching :func:`bjsi._build_joint_model`.  Because the uniform
    measure on normals is rotation invariant, evaluating in the principal frame
    is equivalent to evaluating in any rotated frame.
    """
    eig = np.array([-1.0, 2.0 * float(R) - 1.0, 1.0])
    t = n * eig
    sigma_n = np.einsum("ij,ij->i", n, t)
    tau = np.linalg.norm(t - sigma_n[:, None] * n, axis=1)
    return (tau + mu * (1.0 + sigma_n)) / (np.sqrt(1.0 + mu * mu) + mu)


def _log_softplus_numpy(x: np.ndarray) -> np.ndarray:
    """``log(log(1 + exp(x)))``, linear in ``x`` where the exponential underflows."""
    safe = np.clip(x, _LOG_SOFTPLUS_LINEAR_BELOW, None)
    return np.where(x < _LOG_SOFTPLUS_LINEAR_BELOW, x, np.log(np.logaddexp(0.0, safe)))


def _log_population_weight_numpy(I: np.ndarray, spec: Dict[str, Any], imin) -> np.ndarray:
    if spec["family"] == "exp":
        return spec["beta"] * I
    k = spec["k"]
    return _log_softplus_numpy(k * (I - imin)) - np.log(k * (1.0 - imin))


def cache_dir() -> Path:
    """Directory holding cached normalizer tables (``BJSI_CACHE_DIR`` overrides)."""
    root = os.environ.get("BJSI_CACHE_DIR")
    path = Path(root) if root else Path.home() / ".cache" / "bjsi"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _table_key(spec: Dict[str, Any]) -> str:
    table = spec["table"]
    payload = {
        "version": _TABLE_VERSION,
        "family": spec["family"],
        "R_points": table["R_points"],
        "mu_points": table["mu_points"],
        "mu_range": [round(float(v), 8) for v in table["mu_range"]],
        "power": table["power"],
        "seed": table["seed"],
    }
    if spec["family"] == "exp":
        payload["beta"] = round(float(spec["beta"]), 8)
    else:
        payload["k"] = round(float(spec["k"]), 8)
        if spec["imin"] == "infer":
            payload["imin"] = "infer"
            payload["imin_points"] = table["imin_points"]
            payload["imin_bounds"] = [round(float(v), 8) for v in spec["imin_bounds"]]
        else:
            payload["imin"] = round(float(spec["imin"]), 8)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return f"znorm_{spec['family']}_{digest}"


def build_normalizer_table(spec: Dict[str, Any], verbose: bool = True) -> Dict[str, np.ndarray]:
    """Compute ``log Z`` on a grid by quasi-Monte Carlo over the sphere.

    Returns a dictionary with the grids and ``logZ``.  The array is indexed
    ``[mu, R]`` for a two-dimensional table and ``[mu, R, imin]`` when ``I_min``
    is inferred.
    """
    table = spec["table"]
    R_grid = np.linspace(0.0, 1.0, int(table["R_points"]))
    mu_grid = np.linspace(float(table["mu_range"][0]), float(table["mu_range"][1]),
                          int(table["mu_points"]))
    normals = sobol_unit_normals(table["power"], table["seed"])
    infer_imin = spec["family"] == "ramp" and spec["imin"] == "infer"
    if infer_imin:
        lo, hi = spec["imin_bounds"]
        imin_grid = np.linspace(float(lo), float(hi), int(table["imin_points"]))
        logZ = np.empty((mu_grid.size, R_grid.size, imin_grid.size))
    else:
        imin_grid = None
        logZ = np.empty((mu_grid.size, R_grid.size))

    started = time.time()
    if verbose:
        cells = logZ.size
        print(f"[bjsi] building fault-population normalizer table "
              f"({spec['family']}, {cells} cells, 2**{table['power']} samples)", flush=True)
    for i, mu in enumerate(mu_grid):
        for j, R in enumerate(R_grid):
            I = instability_numpy(normals, R, mu)
            if infer_imin:
                logg = _log_population_weight_numpy(I[:, None], spec, imin_grid[None, :])
                logZ[i, j] = np.log(np.mean(np.exp(logg), axis=0))
            else:
                imin = None if spec["family"] == "exp" else spec["imin"]
                logZ[i, j] = np.log(np.mean(np.exp(_log_population_weight_numpy(I, spec, imin))))
    if verbose:
        print(f"[bjsi] normalizer table built in {time.time() - started:.0f} s", flush=True)

    out = {"R": R_grid, "mu": mu_grid, "logZ": logZ}
    if imin_grid is not None:
        out["imin"] = imin_grid
    return out


def normalizer_table(spec: Dict[str, Any], *, rebuild: bool = False,
                     verbose: bool = True) -> Dict[str, np.ndarray]:
    """Return the normalizer table for ``spec``, using the on-disk cache."""
    path = cache_dir() / f"{_table_key(spec)}.npz"
    if path.exists() and not rebuild:
        with np.load(path) as data:
            return {k: data[k] for k in data.files}
    out = build_normalizer_table(spec, verbose=verbose)
    try:
        np.savez_compressed(path, **out)
    except OSError as exc:  # pragma: no cover - cache is an optimization only
        warnings.warn(f"Could not cache normalizer table at {path}: {exc}", RuntimeWarning)
    return out


# ----------------------------------------------------------------------------
# PyTensor side
# ----------------------------------------------------------------------------
def instability_pt(Sigma, n, mu):
    """Normalized instability of normals ``n`` under ``Sigma``, as a graph.

    Identical in value to the expression used by the historical plane prior of
    :mod:`bjsi`, written so that it can be shared by the population weight.
    """
    sig1 = -1.0
    denom = pt.sqrt(1.0 + mu ** 2)
    tau_c = 1.0 / denom
    sig_c = mu / denom
    t = pt.dot(Sigma, n.T).T
    sigma_n = pt.sum(t * n, axis=-1)
    ts = t - sigma_n[:, None] * n
    tau = pt.sqrt(pt.sum(ts ** 2, axis=-1) + 1e-12)
    return (tau - mu * (sig1 - sigma_n)) / (tau_c - mu * (sig1 - sig_c))


def log_population_weight_pt(I, spec: Dict[str, Any], imin=None):
    """Log population weight ``log g(I)`` as a graph.

    For the ramp family this is ``log softplus(k (I - I_min)) - log(k (1 - I_min))``.
    Far below the threshold the softplus underflows, so the linear branch
    ``k (I - I_min)`` is used there; it is the limit of the same expression and
    keeps the gradient finite.
    """
    if spec["family"] == "exp":
        return spec["beta"] * I
    k = float(spec["k"])
    imin_val = spec["imin"] if imin is None else imin
    x = k * (I - imin_val)
    safe = pt.log(pt.softplus(pt.clip(x, _LOG_SOFTPLUS_LINEAR_BELOW, np.inf)))
    return pt.where(x < _LOG_SOFTPLUS_LINEAR_BELOW, x, safe) - pt.log(k * (1.0 - imin_val))


def log_watson_mixture_pt(n, axes, kappa, log_pi, n_quad: int = 48):
    """Log density of an axial Watson mixture on the sphere, uniform reference.

    Each component is ``exp(kappa (a . n)^2) / M(1/2, 3/2, kappa)`` with
    ``M(1/2, 3/2, kappa) = int_0^1 exp(kappa t^2) dt`` evaluated by Gauss-Legendre
    quadrature, so that its expectation under the uniform measure is one.  The
    density is invariant to the sign of ``n`` and of each axis.
    """
    nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    log_M = pt.log(pt.sum(pt.as_tensor_variable(weights)[None, :]
                          * pt.exp(kappa[:, None] * pt.as_tensor_variable(nodes ** 2)[None, :]),
                          axis=1))
    cos2 = pt.dot(n, axes.T) ** 2
    return pt.logsumexp(log_pi[None, :] + kappa[None, :] * cos2 - log_M[None, :], axis=1)


def _axis_index(grid: np.ndarray, value):
    lo = float(grid[0])
    step = float(grid[1] - grid[0])
    u = pt.clip((value - lo) / step, 0.0, float(grid.size) - 1.0 - 1e-9)
    idx = pt.floor(u).astype("int64")
    return idx, u - idx


def interp_normalizer_pt(table: Dict[str, np.ndarray], R, mu, imin=None):
    """Multilinear interpolation of ``log Z`` at ``(R, mu)`` and optionally ``imin``."""
    values = pt.as_tensor_variable(table["logZ"])
    i, fy = _axis_index(table["mu"], mu)
    j, fx = _axis_index(table["R"], R)
    if "imin" not in table:
        return ((1 - fy) * (1 - fx) * values[i, j]
                + (1 - fy) * fx * values[i, j + 1]
                + fy * (1 - fx) * values[i + 1, j]
                + fy * fx * values[i + 1, j + 1])
    if imin is None:
        raise ValueError("The table has an I_min axis but no I_min value was given")
    l, fz = _axis_index(table["imin"], imin)

    def corner(di, dj, dl):
        return values[i + di, j + dj, l + dl]

    return ((1 - fy) * (1 - fx) * (1 - fz) * corner(0, 0, 0)
            + (1 - fy) * fx * (1 - fz) * corner(0, 1, 0)
            + fy * (1 - fx) * (1 - fz) * corner(1, 0, 0)
            + fy * fx * (1 - fz) * corner(1, 1, 0)
            + (1 - fy) * (1 - fx) * fz * corner(0, 0, 1)
            + (1 - fy) * fx * fz * corner(0, 1, 1)
            + fy * (1 - fx) * fz * corner(1, 0, 1)
            + fy * fx * fz * corner(1, 1, 1))
