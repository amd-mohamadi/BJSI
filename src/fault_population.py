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

where ``h`` is either the uniform density (``mix_uniform``) or a mixture of axial
clusters representing an inherited fault fabric (``fabric_K``).  A fabric
component is either Watson, which is symmetric about one axis, or Bingham, which
is not and can represent the girdle produced by a fixed strike with variable dip
(``fabric_family``).
Each component of ``h`` is normalized against the uniform measure, so the
normalizer of the mixture is ``w Z + (1 - w)``.  The weight ``w`` is the fraction
of the orientation distribution explained by stress selection and is inferred.

In that additive form the stress and the fabric compete for the same density,
and a concentrated fabric wins: on the Cushing catalog ``w`` falls to 0.07 and
the plane selection is the fabric's alone.  ``fabric_mode='product'`` instead
lets faults exist according to the fabric and reactivate according to the
stress,

    g_prod(n) = g(I(n)) h(n),    Z(theta) = E_n[ g(I(n)) h(n) ],

so both terms act on every fault.  ``Z`` then depends on the fabric axes
relative to the stress frame and is not tabulated; it is a quasi-Monte Carlo
average over ``2**product_qmc_power`` fixed Sobol normals inside the graph.

Related families
----------------
With ``K > 1`` a spare component is free to adopt any sub-population of
normals, including one that the nodal-plane ambiguity manufactures: at Cushing
23 mechanisms whose second planes lie in the main girdle get a component of
their own on their steep first planes.  ``fabric_min_overlap`` requires every
other component to overlap the dominant one (largest weight) by at least that
much, where the overlap of two components is the Bhattacharyya coefficient

    O_kl = E_n[ sqrt(B_k(n) B_l(n)) ],

one for identical components and near zero for components with disjoint
support, evaluated on the same Sobol normals as the product normalizer.  A
secondary component can then only be a rotation, splay or sharper core of the
main family, not an unrelated set.  The constraint is a soft penalty
``-fabric_overlap_strength * max(0, O_min - O)**2`` per component, tempered
with the data by the SMC sampler; the log evidence it reports therefore
includes the prior mass of the admissible region, whose logarithm
:func:`overlap_prior_log_norm` estimates so that it can be removed.
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
import pymc as pm
import pytensor.tensor as pt

__all__ = [
    "DEFAULT_FAULT_POPULATION",
    "resolve_population_spec",
    "log_bingham_mixture_pt",
    "log_bingham_components_pt",
    "log_watson_components_pt",
    "component_overlap_pt",
    "overlap_penalty_pt",
    "overlap_prior_log_norm",
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
_DEFAULT_PRODUCT_POWER = 10   # Sobol normals for the in-graph product normalizer
_DEFAULT_SEED = 20260916
_DEFAULT_RAMP_K = 100.0

# Default population: the stress-selected ramp with inferred I_min and one
# Bingham fabric component (K = 1).  Extra components must overlap the main
# one by at least 0.2 (Bhattacharyya) so that K > 1 stays a perturbation of
# one fault family.  ``fault_population=None`` still selects the legacy model.
DEFAULT_FAULT_POPULATION: Dict[str, Any] = {
    "family": "ramp",
    "imin": "infer",
    "fabric_K": 1,
    "fabric_family": "bingham",
    "fabric_pi_alpha": 1.0,
    "fabric_min_overlap": 0.2,
    "fabric_overlap_strength": 2000.0,
}
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
    unnormalized mixture, ``"default"`` for :data:`DEFAULT_FAULT_POPULATION`,
    a family name, or a dictionary with the keys below.
    Missing keys take the values of :data:`DEFAULT_FAULT_POPULATION`: a ramp
    with inferred ``imin``, one Bingham fabric component, ``fabric_pi_alpha``
    1 and a minimum overlap of 0.2 enforced with strength 2000.

    family : {'legacy', 'exp', 'ramp'}
    beta : float, for the 'exp' family; defaults to the model's selection beta
    imin : float or 'infer', for the 'ramp' family
    imin_bounds : (low, high) prior support when ``imin='infer'``
    k : float, sharpness of the ramp softplus
    normalize : bool, add the ``-N log Z`` potential
    mix_uniform : bool, add a uniform component with inferred weight
    fabric_K : int, number of fabric clusters (default 1 for 'ramp', 0 for
        'exp'), 0 to disable
    fabric_family : {'watson', 'bingham'}, shape of one fabric component
        (default 'bingham')
    fabric_kappa_sigma : float, half-normal prior scale for the concentrations
    fabric_pi_alpha : float, Dirichlet concentration on the fabric proportions;
        values below one favour switching components off
    fabric_min_overlap : float in [0, 1), minimum Bhattacharyya overlap of
        every extra component with the dominant one (default 0.2), 0 to disable
    fabric_overlap_strength : float, stiffness of the overlap penalty
        (default 2000)
    table : dict, overrides for the normalizer grid and sampling
    """
    if fault_population is None:
        spec: Dict[str, Any] = {"family": LEGACY}
    elif isinstance(fault_population, str) and fault_population.strip().lower() == "default":
        spec = dict(DEFAULT_FAULT_POPULATION)
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

    _D = DEFAULT_FAULT_POPULATION
    out: Dict[str, Any] = {
        "family": family,
        "normalize": bool(spec.get("normalize", True)),
        "mix_uniform": bool(spec.get("mix_uniform", False)),
        # The default fabric belongs to the ramp model; 'exp' and the uniform
        # mixture stay purely stress-selected.
        "fabric_K": int(spec.get("fabric_k", spec.get(
            "fabric_K", _D["fabric_K"] if family == "ramp" and not spec.get("mix_uniform", False) else 0))),
        "fabric_family": str(spec.get("fabric_family", _D["fabric_family"])).strip().lower(),
        "fabric_kappa_sigma": float(spec.get("fabric_kappa_sigma", 40.0)),
        "fabric_pi_alpha": float(spec.get("fabric_pi_alpha", _D["fabric_pi_alpha"])),
        "fabric_mode": str(spec.get("fabric_mode", "additive")).strip().lower(),
        "product_qmc_power": int(spec.get("product_qmc_power", _DEFAULT_PRODUCT_POWER)),
        "fabric_min_overlap": float(spec.get("fabric_min_overlap", _D["fabric_min_overlap"])),
        "fabric_overlap_strength": float(spec.get("fabric_overlap_strength", _D["fabric_overlap_strength"])),
        "table": dict(spec.get("table", {})),
    }
    if out["fabric_mode"] not in {"additive", "product"}:
        raise ValueError("fabric_mode must be 'additive' or 'product'")
    if out["fabric_mode"] == "product" and out["fabric_K"] < 1:
        raise ValueError("fabric_mode='product' requires fabric_K >= 1")
    if not 0.0 <= out["fabric_min_overlap"] < 1.0:
        raise ValueError("fabric_min_overlap must lie in [0, 1)")
    if out["fabric_overlap_strength"] < 0.0:
        raise ValueError("fabric_overlap_strength must be >= 0")
    if out["fabric_K"] < 0:
        raise ValueError("fabric_K must be >= 0")
    if out["fabric_family"] not in {"watson", "bingham"}:
        raise ValueError("fabric_family must be 'watson' or 'bingham'")
    if out["fabric_pi_alpha"] <= 0.0:
        raise ValueError("fabric_pi_alpha must be > 0")
    if out["mix_uniform"] and out["fabric_K"] > 0:
        raise ValueError("Use either mix_uniform or fabric_K, not both")
    if out["fabric_mode"] == "product" and not out["normalize"]:
        raise ValueError("fabric_mode='product' is only defined with normalize=True")

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


def gnomonic_rotation(name: str, shape, initval=None):
    """Uniform (Haar) random rotations in a three-parameter chart.

    A unit quaternion ``q = (1, v) / sqrt(1 + |v|^2)`` with ``v`` in R^3 covers
    every rotation once (``q`` and ``-q`` are the same rotation), and the Haar
    measure pulled back to ``v`` is the trivariate Cauchy density
    ``p(v) = (1 + |v|^2)^-2 / pi^2``.  Compared with normalizing a Gaussian
    4-vector this has no free radial direction, so when the data pin the
    rotation to a fraction of a degree the posterior is a small blob in ``v``
    rather than a needle along a ray, and NUTS steps at the natural scale
    instead of at the needle's width.  Rotations by 180 degrees from the
    identity sit at infinity, but every stress or fabric frame has symmetric
    copies (sign of each axis) of which one lies within ``|v| <= sqrt(3)``.

    Returns ``(quaternions, v)`` with ``quaternions`` of shape ``shape + (4,)``.
    ``shape`` is ``()`` or ``(K,)``.
    """
    v = pm.Normal(f"{name}_gnomonic", mu=0.0, sigma=1.0, shape=tuple(shape) + (3,),
                  initval=initval)
    r2 = pt.sum(v ** 2, axis=-1)
    # Replace the Gaussian density by the Cauchy pull-back of the Haar measure.
    pm.Potential(f"{name}_haar", pt.sum(0.5 * r2 - 2.0 * pt.log1p(r2)))
    scale = 1.0 / pt.sqrt(1.0 + r2)
    q = pt.concatenate([scale[..., None], v * scale[..., None]], axis=-1)
    return q, v


def gnomonic_initval(quaternions: np.ndarray) -> np.ndarray:
    """Chart coordinates of unit quaternions, using the symmetric copy nearest the identity."""
    q = np.asarray(quaternions, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    # Right-multiplying by the unit quaternions 1, i, j, k permutes the components
    # (with signs), so moving the largest component to the front is a symmetric copy.
    out = np.empty(q.shape[:-1] + (3,))
    for idx in np.ndindex(q.shape[:-1]):
        qi = q[idx]
        k = int(np.argmax(np.abs(qi)))
        if k != 0:
            table = {1: (1, 0, 3, 2), 2: (2, 3, 0, 1), 3: (3, 2, 1, 0)}[k]
            signs = {1: (-1, 1, 1, -1), 2: (-1, -1, 1, 1), 3: (-1, 1, -1, 1)}[k]
            qi = np.array([sg * qi[j] for sg, j in zip(signs, table)])
        if qi[0] < 0:
            qi = -qi
        out[idx] = qi[1:] / qi[0]
    return out


def quat_to_rotation_pt(q):
    """Rotation matrix of a unit quaternion ``(w, x, y, z)``, as a graph.

    The columns are the three orthonormal axes.  This repeats the convention of
    the stress orientation in :func:`bjsi._build_joint_model` so that the fabric
    axes are expressed in the same frame as the fault normals.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    return pt.stacklists([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _log_bingham_normalizer_pt(kappa1, kappa2, n_quad: int = 64):
    """``log E_n[exp(-k1 (a1.n)^2 - k2 (a2.n)^2)]`` for normals uniform on the sphere.

    Integrating the azimuth around the third axis leaves a one-dimensional
    integral with a modified Bessel factor,

        c = 1/2 int_-1^1 exp(-(1-z^2)(k1+k2)/2) I0((1-z^2)(k1-k2)/2) dz,

    which is evaluated by Gauss-Legendre quadrature.  ``log I0`` uses the
    exponentially scaled Bessel function so that large concentrations are safe.
    """
    nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))
    z2 = pt.as_tensor_variable(nodes ** 2)
    log_w = pt.as_tensor_variable(np.log(weights / 2.0))
    one_minus = 1.0 - z2[None, :]
    half_sum = 0.5 * (kappa1 + kappa2)[:, None]
    half_diff = pt.abs(0.5 * (kappa1 - kappa2))[:, None] * one_minus
    log_i0 = half_diff + pt.log(pt.ive(0.0, half_diff))
    return pt.logsumexp(log_w[None, :] - one_minus * half_sum + log_i0, axis=1)


def log_bingham_mixture_pt(n, axes, kappas, log_pi, n_quad: int = 64):
    """Log density of an axial Bingham mixture on the sphere, uniform reference.

    ``axes`` has shape ``(K, 3, 3)`` with the three orthonormal axes of each
    component in its columns, and ``kappas`` has shape ``(K, 2)`` holding the
    non-negative concentrations against the first two axes.  Component ``k`` is

        B_k(n) = exp(-k1 (a1.n)^2 - k2 (a2.n)^2) / c(k1, k2),

    so that its expectation under the uniform measure is one.  Equal
    concentrations give a Watson component about the third axis, a large ``k1``
    with ``k2`` near zero gives a girdle in the plane of the second and third
    axes, and the density is invariant to the sign of ``n`` and of each axis.
    """
    cos2 = pt.tensordot(n, axes, axes=[[1], [1]]) ** 2    # (N, K, 3), axis j in the columns
    quad = -(kappas[None, :, 0] * cos2[:, :, 0] + kappas[None, :, 1] * cos2[:, :, 1])
    log_c = _log_bingham_normalizer_pt(kappas[:, 0], kappas[:, 1], n_quad)
    return pt.logsumexp(log_pi[None, :] + quad - log_c[None, :], axis=1)


def log_bingham_components_pt(n, axes, kappas, n_quad: int = 64):
    """Per-component log density ``(N, K)`` of :func:`log_bingham_mixture_pt`, without weights."""
    cos2 = pt.tensordot(n, axes, axes=[[1], [1]]) ** 2
    quad = -(kappas[None, :, 0] * cos2[:, :, 0] + kappas[None, :, 1] * cos2[:, :, 1])
    return quad - _log_bingham_normalizer_pt(kappas[:, 0], kappas[:, 1], n_quad)[None, :]


def log_watson_components_pt(n, axes, kappa, n_quad: int = 48):
    """Per-component log density ``(N, K)`` of :func:`log_watson_mixture_pt`, without weights."""
    nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    log_M = pt.log(pt.sum(pt.as_tensor_variable(weights)[None, :]
                          * pt.exp(kappa[:, None] * pt.as_tensor_variable(nodes ** 2)[None, :]),
                          axis=1))
    cos2 = pt.dot(n, axes.T) ** 2
    return kappa[None, :] * cos2 - log_M[None, :]


def component_overlap_pt(log_components, n_points: int):
    """Bhattacharyya overlap ``(K, K)`` of components from their ``(M, K)`` log densities on uniform normals."""
    half = 0.5 * log_components
    m = pt.max(half, axis=0, keepdims=True)
    e = pt.exp(half - m)
    return pt.dot(e.T, e) / float(n_points) * pt.exp(m.T + m)


def overlap_penalty_pt(overlap, pi, min_overlap: float, strength: float):
    """Soft penalty keeping every component within ``min_overlap`` of the dominant one.

    The dominant component is the one with the largest weight, so the term is
    symmetric under relabelling.  Returns ``(penalty, overlap_with_main)``.
    """
    main = pt.argmax(pi)
    o_main = overlap[:, main]
    shortfall = pt.maximum(0.0, float(min_overlap) - o_main)
    return -float(strength) * pt.sum(shortfall ** 2), o_main


def _bingham_components_numpy(n, axes, kappas):
    from scipy.special import ive
    nodes, w = np.polynomial.legendre.leggauss(64)
    om = 1.0 - nodes ** 2
    out = np.empty((n.shape[0], axes.shape[0]))
    for k in range(axes.shape[0]):
        k1, k2 = kappas[k]
        hd = abs(0.5 * (k1 - k2)) * om
        log_c = np.log(np.sum(w / 2.0 * np.exp(-om * 0.5 * (k1 + k2) + hd) * ive(0, hd)))
        c = (n @ axes[k]) ** 2
        out[:, k] = -(k1 * c[:, 0] + k2 * c[:, 1]) - log_c
    return out


def _watson_components_numpy(n, axes, kappa):
    nodes, w = np.polynomial.legendre.leggauss(48)
    nodes = 0.5 * (nodes + 1.0)
    w = 0.5 * w
    log_M = np.log(np.sum(w[None, :] * np.exp(kappa[:, None] * nodes[None, :] ** 2), axis=1))
    return kappa[None, :] * (n @ axes.T) ** 2 - log_M[None, :]


def overlap_prior_log_norm(spec: Dict[str, Any], n_samples: int = 4000, seed: int = 0) -> float:
    """``log E_prior[exp(penalty)]``: the log prior mass of the admissible fabric region.

    The overlap penalty is applied as a potential, so the evidence a sampler
    reports is that of the unnormalized penalized prior; subtracting this
    value gives the evidence under the properly normalized prior, which is
    what a comparison with an unconstrained model needs.  Zero when the
    constraint is off or ``K == 1``.
    """
    K = int(spec["fabric_K"])
    o_min, gamma = float(spec["fabric_min_overlap"]), float(spec["fabric_overlap_strength"])
    if K < 2 or o_min <= 0.0 or gamma <= 0.0:
        return 0.0
    rng = np.random.default_rng(seed)
    n = sobol_unit_normals(spec["product_qmc_power"], spec["table"]["seed"])
    sigma = float(spec["fabric_kappa_sigma"])
    log_terms = np.empty(n_samples)
    for i in range(n_samples):
        q = rng.normal(size=(K, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        rots = np.stack([_quat_to_rotation_numpy(qk) for qk in q])
        pi = rng.dirichlet(float(spec["fabric_pi_alpha"]) * np.ones(K))
        if spec["fabric_family"] == "bingham":
            kap = np.abs(rng.normal(scale=sigma, size=(K, 2)))
            lc = _bingham_components_numpy(n, rots, kap)
        else:
            kap = np.abs(rng.normal(scale=sigma, size=K))
            lc = _watson_components_numpy(n, rots[:, :, 2], kap)
        e = np.exp(0.5 * (lc - lc.max(axis=0, keepdims=True)))
        overlap = (e.T @ e) / n.shape[0] * np.exp(0.5 * (lc.max(axis=0)[:, None] + lc.max(axis=0)[None, :]))
        o_main = overlap[:, int(np.argmax(pi))]
        log_terms[i] = -gamma * np.sum(np.maximum(0.0, o_min - o_main) ** 2)
    m = log_terms.max()
    return float(m + np.log(np.mean(np.exp(log_terms - m))))


def _quat_to_rotation_numpy(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


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
