"""Basin diagnostics and per-basin evidence for multimodal BJSI posteriors.

Fault-population models are multimodal on real catalogs: chains settle in
distinct combinations of stress orientation, shape ratio and plane assignment,
and the usual convergence statistics then describe the disagreement between
basins rather than a sampling problem.  This module provides the two tools the
population models need after sampling.

``basin_report``
    Groups chains by the posterior they actually explored and reports the
    parameters of each group, so that a large ``r_hat`` can be read as "three
    basins" instead of "not converged".

``bridge_evidence``
    Estimates the log evidence of one chain by bridge sampling.  A chain
    confined to one basin, together with a proposal fitted to that chain, gives
    the evidence of that basin, which is what a comparison between modes needs.
    Chains in the same basin must agree; that agreement is the check that the
    estimate is usable.

Symmetries of the model, the sign of the quaternion, rotations by 180 degrees
about the principal axes, the sign of each fabric axis and the labelling of the
fabric components, multiply the evidence of every basin of a given model by the
same factor, so they cancel in comparisons within a model.  They do not cancel
when models with different fabric sizes are compared, where the factor is
``2**K K!`` for ``K`` fabric components.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytensor

__all__ = ["basin_report", "bridge_evidence", "chain_summary"]


def _stress_directions(tensors: np.ndarray) -> tuple:
    """Return sigma1 azimuth and the maximum horizontal compression azimuth.

    Model coordinates are north, west, up, and compression is negative, matching
    :func:`bjsi._build_joint_model`.
    """
    _, vectors = np.linalg.eigh(tensors)
    s1 = vectors[..., :, 0]
    s1_az = (-np.degrees(np.arctan2(s1[..., 1], s1[..., 0]))) % 180.0
    _, horizontal = np.linalg.eigh(tensors[..., :2, :2])
    h = horizontal[..., :, 0]
    shmax = (-np.degrees(np.arctan2(h[..., 1], h[..., 0]))) % 180.0
    return s1_az, shmax


def chain_summary(idata, extra_vars: Sequence[str] = ()) -> List[Dict[str, Any]]:
    """Per-chain medians of the quantities that distinguish basins."""
    post = idata.posterior
    _, shmax = _stress_directions(post["Sigma"].values)
    rows = []
    for chain in range(post.sizes["chain"]):
        row: Dict[str, Any] = {
            "chain": int(chain),
            "shmax": float(np.median(shmax[chain])),
            "R": float(np.median(post["R"].values[chain])),
        }
        for name in ("mu", "I_min", "w_population", *extra_vars):
            if name in post:
                row[name] = float(np.median(post[name].values[chain]))
        if "p_plane2_post" in post:
            row["plane2_fraction"] = float(np.mean(post["p_plane2_post"].values[chain] > 0.5))
        rows.append(row)
    return rows


def basin_report(idata, *, shmax_tol: float = 5.0, R_tol: float = 0.1,
                 plane_tol: float = 0.1) -> Dict[str, Any]:
    """Group chains into basins and describe each one.

    Two chains share a basin when their median SHmax differs by less than
    ``shmax_tol`` degrees, their median ``R`` by less than ``R_tol``, and the
    fraction of events assigned to plane two by less than ``plane_tol``.  SHmax
    is compared as an axis, so 179 and 1 degree are one degree apart.
    """
    rows = chain_summary(idata)
    basins: List[Dict[str, Any]] = []
    for row in rows:
        for basin in basins:
            d_shmax = abs(row["shmax"] - basin["shmax"])
            d_shmax = min(d_shmax, 180.0 - d_shmax)
            same = d_shmax < shmax_tol and abs(row["R"] - basin["R"]) < R_tol
            if same and "plane2_fraction" in row and "plane2_fraction" in basin:
                same = abs(row["plane2_fraction"] - basin["plane2_fraction"]) < plane_tol
            if same:
                basin["chains"].append(row["chain"])
                break
        else:
            basin = dict(row)
            basin["chains"] = [row.pop("chain")]
            basins.append(basin)
    for index, basin in enumerate(basins):
        basin["label"] = f"basin_{index}"
        basin.pop("chain", None)
    return {"n_basins": len(basins), "basins": basins, "per_chain": rows,
            "multimodal": len(basins) > 1}


def _unconstrained_draws(model, idata, chain: int) -> tuple:
    """Stack the draws of one chain in the sampler's unconstrained coordinates."""
    import pytensor.tensor as pt

    post = idata.posterior
    columns, shapes = [], []
    for rv, value_var in zip(model.free_RVs, model.value_vars):
        values = post[rv.name].values[chain]
        transform = model.rvs_to_transforms[rv]
        if transform is not None:
            values = transform.forward(pt.as_tensor_variable(values), *rv.owner.inputs).eval()
        values = np.asarray(values)
        columns.append(values.reshape(values.shape[0], -1))
        shapes.append(values.shape[1:])
    return np.concatenate(columns, axis=1), shapes


def _unpack(theta: np.ndarray, model, shapes) -> Dict[str, Any]:
    point, start = {}, 0
    for value_var, shape in zip(model.value_vars, shapes):
        size = int(np.prod(shape)) if shape else 1
        chunk = theta[start:start + size]
        point[value_var.name] = chunk.reshape(shape) if shape else float(chunk[0])
        start += size
    return point


def _bridge_fixed_point(log_l_post: np.ndarray, log_l_prop: np.ndarray,
                        iterations: int = 5000, tol: float = 1e-10) -> float:
    """Meng and Wong optimal bridge estimator, iterated in log space."""
    from scipy.special import logsumexp

    n1, n2 = log_l_post.size, log_l_prop.size
    log_s1, log_s2 = np.log(n1 / (n1 + n2)), np.log(n2 / (n1 + n2))
    log_r = float(np.median(log_l_post))
    for _ in range(iterations):
        numerator = logsumexp(log_l_prop - np.logaddexp(log_s1 + log_l_prop, log_s2 + log_r)) - np.log(n2)
        denominator = logsumexp(-np.logaddexp(log_s1 + log_l_post, log_s2 + log_r)) - np.log(n1)
        new = numerator - denominator
        if abs(new - log_r) < tol:
            return float(new)
        log_r = float(new)
    return log_r


def bridge_evidence(model, idata, chain: int, *, seed: int = 0,
                    n_proposal: Optional[int] = None) -> Dict[str, float]:
    """Log evidence of the basin explored by ``chain``, by bridge sampling.

    Half of the draws fit a multivariate normal proposal and the other half
    enter the estimate, so the proposal is independent of the draws it is
    compared against.  The returned ``bootstrap_se`` is a resampling spread, not
    a bound on the bias that a chain straddling two basins would introduce; use
    :func:`basin_report` first and compare chains of the same basin.
    """
    from scipy.stats import multivariate_normal

    logp = model.compile_logp()
    draws, shapes = _unconstrained_draws(model, idata, chain)
    half = draws.shape[0] // 2
    if half < 20:
        raise ValueError("Need at least 40 draws in the chain for a bridge estimate")
    fit, estimate = draws[:half], draws[half:]
    mean = fit.mean(axis=0)
    covariance = np.cov(fit.T) + 1e-8 * np.eye(draws.shape[1])
    proposal = multivariate_normal(mean, covariance, allow_singular=True)
    size = int(n_proposal or estimate.shape[0])
    rng = np.random.default_rng(seed)
    proposal_draws = np.asarray(proposal.rvs(size=size, random_state=rng)).reshape(size, -1)

    def log_target(rows):
        out = np.empty(rows.shape[0])
        for i, row in enumerate(rows):
            try:
                out[i] = float(logp(_unpack(row, model, shapes)))
            except Exception:
                out[i] = -np.inf
        return np.where(np.isfinite(out), out, -1e300)

    log_l_post = log_target(estimate) - proposal.logpdf(estimate)
    log_l_prop = log_target(proposal_draws) - proposal.logpdf(proposal_draws)
    log_evidence = _bridge_fixed_point(log_l_post, log_l_prop)

    boots = []
    for _ in range(20):
        i = rng.integers(0, log_l_post.size, log_l_post.size)
        j = rng.integers(0, log_l_prop.size, log_l_prop.size)
        boots.append(_bridge_fixed_point(log_l_post[i], log_l_prop[j]))
    return {
        "chain": int(chain),
        "log_evidence": float(log_evidence),
        "bootstrap_se": float(np.std(boots)),
        "dimension": int(draws.shape[1]),
        "finite_proposal_fraction": float(np.mean(log_l_prop > -1e299)),
    }
