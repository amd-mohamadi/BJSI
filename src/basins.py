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

``pool_runs`` / ``pooled_results``
    Combine independent SMC runs by their evidence.  Each run targets the
    whole posterior, so runs that disagree have each missed part of it; the
    mixture of runs weighted by their evidence estimates is the pooled
    posterior, and an equal-weight pool would give a run stuck in a mode of
    negligible mass the same say as the dominant one.

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

__all__ = ["basin_report", "bridge_evidence", "chain_summary", "pool_runs", "pooled_results",
           "subset_results"]


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


def basin_report(idata, *, shmax_tol: float = 5.0, R_tol: float = 0.05,
                 plane_tol: float = 0.05) -> Dict[str, Any]:
    """Group chains into basins and describe each one.

    Two chains share a basin when their median SHmax differs by less than
    ``shmax_tol`` degrees, their median ``R`` by less than ``R_tol``, and the
    fraction of events they assign to different planes is below ``plane_tol``.
    SHmax is compared as an axis, so 179 and 1 degree are one degree apart.
    Chains are compared with the first chain of a basin, not with each other,
    so a basin cannot grow by chaining small differences.  Basins are listed
    with the most chains first and labelled ``basin_1``, ``basin_2``, ...
    """
    rows = chain_summary(idata)
    post = idata.posterior
    plane_maps = None
    if "p_plane2_post" in post:
        plane_maps = [post["p_plane2_post"].values[c].mean(0) > 0.5 for c in range(post.sizes["chain"])]
    basins: List[Dict[str, Any]] = []
    for row in rows:
        for basin in basins:
            d_shmax = abs(row["shmax"] - basin["shmax"])
            d_shmax = min(d_shmax, 180.0 - d_shmax)
            same = d_shmax < shmax_tol and abs(row["R"] - basin["R"]) < R_tol
            if same and plane_maps is not None:
                same = np.mean(plane_maps[row["chain"]] != plane_maps[basin["chains"][0]]) < plane_tol
            if same:
                basin["chains"].append(row["chain"])
                break
        else:
            basin = dict(row)
            basin["chains"] = [basin.pop("chain")]
            basins.append(basin)
    # Dominant basin first: most chains, then earliest chain.
    basins.sort(key=lambda b: (-len(b["chains"]), b["chains"][0]))
    for index, basin in enumerate(basins):
        basin["label"] = f"basin_{index + 1}"
    return {"n_basins": len(basins), "basins": basins, "per_chain": rows,
            "multimodal": len(basins) > 1}


def subset_results(results: Dict[str, Any], chains: Sequence[int],
                   hdi_prob: Optional[float] = None) -> Dict[str, Any]:
    """Re-summarize a sampler result over a subset of its chains.

    The sampler pools all chains; when they sit in different basins the pooled
    element-wise median tensor and the pooled HDIs describe no basin.  This
    returns a copy of ``results`` with every posterior-derived entry (stress
    tensor, principal axes, R, friction, HDIs, plane probabilities and
    selection map, per-draw principal axes) recomputed from ``chains`` only,
    ready for the same output tools as the full result.  Entries that do not
    depend on the posterior draws, such as the population spec and the basin
    report, are kept.
    """
    chains = [int(c) for c in chains]
    # Chains index the sampler's runs, which an evidence-pooled result keeps apart.
    idata = results["idata_runs"] if results.get("idata_runs") is not None else results["idata"]
    n_chains = idata.posterior.sizes["chain"]
    if not chains or any(c < 0 or c >= n_chains for c in chains):
        raise ValueError(f"chains must be a non-empty subset of range({n_chains}), got {chains}")
    out = _resummarize(results, idata.sel(chain=chains), hdi_prob)
    out["convergence"] = dict(results.get("convergence") or {},
                              n_chains=len(chains), chains=chains,
                              n_samples=len(chains) * idata.posterior.sizes["draw"])
    return out


def _resummarize(results: Dict[str, Any], idata, hdi_prob: Optional[float]) -> Dict[str, Any]:
    """Copy of ``results`` with every posterior-derived entry taken from ``idata``."""
    try:
        from .bjsi import summarize_posterior
    except ImportError:
        from bjsi import summarize_posterior
    if hdi_prob is None:
        hdi_prob = float((results.get("hdi") or {}).get("prob", 0.9))
    fixed_mu = results.get("mu") if results.get("mu_samples") is None else None
    out = dict(results)
    out.update(summarize_posterior(idata, hdi_prob=hdi_prob))
    if fixed_mu is not None:
        out["mu"] = fixed_mu
    if results.get("fixed_plane_indices") is not None:
        out["plane_selection_map"] = results["plane_selection_map"]
        out["plane_probabilities"] = results["plane_probabilities"]
    return out


def pool_runs(idata, log_evidence: Optional[Sequence[float]] = None, *, seed: int = 0):
    """Pool independent SMC runs, one per chain, weighted by their evidence.

    Run ``c`` gets weight ``Z_c / sum(Z)``, the particles within a run being
    equally weighted.  The pool is drawn by systematic resampling of
    ``round(draws / sum(w**2))`` particles, so a dominant run is returned
    whole, each particle once, and ``C`` runs of equal evidence are returned
    whole as well.  The result has a single chain holding the pooled draws,
    with ``sample_stats`` resampled alongside, and records the run weights in
    ``attrs``.  ``log_evidence`` defaults to ``idata.attrs["log_evidence"]``.
    """
    import arviz as az
    import xarray as xr

    if log_evidence is None:
        log_evidence = idata.attrs.get("log_evidence")
    if log_evidence is None:
        raise ValueError("pool_runs needs the log evidence of each run")
    log_z = np.asarray(log_evidence, float)
    n_chains, n_draws = idata.posterior.sizes["chain"], idata.posterior.sizes["draw"]
    if log_z.shape != (n_chains,):
        raise ValueError(f"log_evidence has {log_z.size} entries for {n_chains} runs")
    weights = np.exp(log_z - log_z.max())
    weights /= weights.sum()
    n_out = max(1, int(round(n_draws / np.sum(weights ** 2))))
    # Systematic resampling over (run, particle), each particle weighing w_run / draws.
    cumulative = np.cumsum(np.repeat(weights / n_draws, n_draws))
    cumulative[-1] = 1.0
    u = (np.random.default_rng(seed).random() + np.arange(n_out)) / n_out
    flat = np.searchsorted(cumulative, u, side="right")
    chain_idx = xr.DataArray(flat // n_draws, dims="pooled")
    draw_idx = xr.DataArray(flat % n_draws, dims="pooled")

    def resample(ds):
        out = ds.isel(chain=chain_idx, draw=draw_idx).drop_vars(["chain", "draw"], errors="ignore")
        out = out.rename({"pooled": "draw"}).expand_dims(chain=[0])
        return out.assign_coords(draw=np.arange(n_out)).transpose("chain", "draw", ...)

    groups = {}
    for group in idata.groups():
        ds = getattr(idata, group)
        groups[group] = resample(ds) if {"chain", "draw"} <= set(ds.dims) else ds
    pooled = az.InferenceData(**groups)
    pooled.attrs.update(idata.attrs)
    pooled.attrs.update({"pooling": "evidence", "log_evidence": log_z.tolist(),
                         "run_weights": weights.tolist(), "n_runs": int(n_chains)})
    return pooled


def pooled_results(results: Dict[str, Any], hdi_prob: Optional[float] = None,
                   seed: int = 0) -> Dict[str, Any]:
    """Re-summarize a multi-run SMC result from its evidence-weighted pool.

    ``results["idata"]`` becomes the pool of :func:`pool_runs` and the runs are
    kept as ``results["idata_runs"]`` for the basin tools.  A result that is
    already pooled, has one run, or has no evidence is returned unchanged.
    """
    if results.get("idata_runs") is not None:
        return results
    idata = results["idata"]
    log_z = (results.get("smc") or {}).get("log_evidence") or idata.attrs.get("log_evidence")
    if log_z is None or idata.posterior.sizes["chain"] < 2:
        return results
    pooled = pool_runs(idata, log_z, seed=seed)
    out = _resummarize(results, pooled, hdi_prob)
    out["idata_runs"] = idata
    out["convergence"] = dict(results.get("convergence") or {}, pooling="evidence",
                              run_weights=pooled.attrs["run_weights"],
                              n_samples=pooled.posterior.sizes["draw"])
    return out


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
