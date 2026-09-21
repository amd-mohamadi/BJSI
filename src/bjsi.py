"""
Bayesian stress inversion with joint plane selection (SMC + NUTS).

This module implements a Bayesian approach to stress tensor inversion from
earthquake focal mechanisms using Sequential Monte Carlo sampling and a NUTS
variant. Unlike traditional iterative methods, this approach jointly samples
stress parameters and nodal plane selection in a single probabilistic model.

Key features:
- Joint inference of stress orientation, shape ratio, and plane selection
- Optional friction coefficient inference (sampled in NUTS; in SMC either sampled or post hoc)
- Selectable Gaussian or von Mises-Fisher slip-direction likelihood with optional tau^2 weighting
- Instability-based plane-selection prior (Mohr-Coulomb)
- Handles multimodal posteriors; no bootstrap required (full posterior from single run)

Author: ILSI Development Team
Date: 2025-10-24
"""

import warnings

import numpy as np
import arviz as az
import pymc as pm
import pytensor.tensor as pt
import pytensor.gradient as ptg
from typing import Optional, Tuple, Dict, Any

try:
    from .mechanism_uncertainty import prepare_mechanism_errors, validate_fixed_planes, latent_mechanism_vectors
    from . import fault_population as _fp
except ImportError:
    from mechanism_uncertainty import prepare_mechanism_errors, validate_fixed_planes, latent_mechanism_vectors
    import fault_population as _fp

# Optional import of deterministic routines
try:
    from . import ilsi as _det
except Exception:
    try:
        import ilsi as _det  # type: ignore
    except Exception:
        _det = None  # Only needed when iterative_plane_selection=True

# Import helper functions from utils_stress
try:
    from .utils_stress import stress_tensor_eigendecomposition
except ImportError:
    try:
        from utils_stress import stress_tensor_eigendecomposition
    except ImportError:
        # Fallback if utils_stress not available
        def stress_tensor_eigendecomposition(sigma):
            """Fallback eigendecomposition (should not be used in production)."""
            eigenvalues, eigenvectors = np.linalg.eigh(sigma)
            idx = np.argsort(eigenvalues)  # ascending order
            return eigenvalues[idx], eigenvectors[:, idx]


def normal_slip_vectors_batch(strike, dip, rake, direction="inward"):
    """
    Convert arrays of strike/dip/rake to normal and slip unit vectors.

    Parameters
    ----------
    strike : array_like
        Strike angles in degrees (N,)
    dip : array_like
        Dip angles in degrees (N,)
    rake : array_like
        Rake angles in degrees (N,)
    direction : str, default "inward"
        Normal vector direction: "inward" (footwall) or "outward" (hanging wall)

    Returns
    -------
    n : ndarray
        Unit normal vectors (N, 3)
    s : ndarray
        Unit slip vectors (N, 3)
    """
    strike = np.asarray(strike, dtype=np.float64)
    dip = np.asarray(dip, dtype=np.float64)
    rake = np.asarray(rake, dtype=np.float64)

    d2r = np.pi / 180.0
    s_rad = strike * d2r
    d_rad = dip * d2r
    r_rad = rake * d2r

    # Normal vector (footwall pointing into hanging wall)
    n = np.stack([
        -np.sin(d_rad) * np.sin(s_rad),
        -np.sin(d_rad) * np.cos(s_rad),
        np.cos(d_rad)
    ], axis=1)

    if direction == "outward":
        n *= -1.0
    elif direction != "inward":
        raise ValueError('direction should be either "inward" or "outward"')

    # Slip vector
    slip = np.stack([
        np.cos(r_rad) * np.cos(s_rad) + np.sin(r_rad) * np.cos(d_rad) * np.sin(s_rad),
        -np.cos(r_rad) * np.sin(s_rad) + np.sin(r_rad) * np.cos(d_rad) * np.cos(s_rad),
        np.sin(r_rad) * np.sin(d_rad)
    ], axis=1)

    return n, slip


def _canonicalize_slip_likelihood(slip_likelihood: Optional[str]) -> str:
    """Normalize the user-facing slip likelihood selector."""
    likelihood = str(slip_likelihood or "gaussian").strip().lower().replace("-", "_")
    aliases = {
        "gaussian": "gaussian",
        "normal": "gaussian",
        "vmf": "von_mises_fisher",
        "vonmisesfisher": "von_mises_fisher",
        "von_misesfisher": "von_mises_fisher",
        "von_mises_fisher": "von_mises_fisher",
    }
    try:
        return aliases[likelihood]
    except KeyError as exc:
        raise ValueError(
            "slip_likelihood must be one of {'gaussian', 'von_mises_fisher', 'vmf'}"
        ) from exc


def _resolve_slip_likelihood_params(
    slip_likelihood: Optional[str],
    slip_misfit_sigma: float,
    slip_vmf_kappa: Optional[float],
) -> Tuple[str, float, Optional[float]]:
    """Validate and resolve slip-likelihood hyperparameters."""
    likelihood_name = _canonicalize_slip_likelihood(slip_likelihood)

    sigma = float(slip_misfit_sigma)
    if not np.isfinite(sigma):
        raise ValueError("slip_misfit_sigma must be finite")

    kappa_val = None
    if slip_vmf_kappa is not None:
        kappa_val = float(slip_vmf_kappa)
        if not np.isfinite(kappa_val) or kappa_val <= 0.0:
            raise ValueError("slip_vmf_kappa must be finite and > 0")

    if likelihood_name == "gaussian" or kappa_val is None:
        if sigma <= 0.0:
            raise ValueError("slip_misfit_sigma must be > 0")

    if likelihood_name == "von_mises_fisher":
        if kappa_val is None:
            kappa_val = 1.0 / (sigma**2)
        if not np.isfinite(kappa_val) or kappa_val <= 0.0:
            raise ValueError("Resolved vMF concentration must be finite and > 0")

    return likelihood_name, sigma, kappa_val


def _slip_direction_logp(
    s_obs,
    s_pred,
    *,
    family: str,
    sigma: float,
    vmf_kappa: Optional[float],
    weight=1.0,
):
    """Return per-event slip-direction log-likelihoods on unit vectors."""
    safe_weight = pt.clip(pt.as_tensor_variable(weight), 1e-12, 1e12)

    obs_norm = pt.sqrt(pt.sum(pt.square(s_obs), axis=-1, keepdims=True))
    pred_norm = pt.sqrt(pt.sum(pt.square(s_pred), axis=-1, keepdims=True))
    s_obs_unit = s_obs / (obs_norm + 1e-12)
    s_pred_unit = s_pred / (pred_norm + 1e-12)

    if family == "gaussian":
        diff = s_obs_unit - s_pred_unit
        sigma2 = float(sigma) ** 2
        # Precision-weighted Gaussian: the weight scales the precision, so the
        # normalizer must carry it too (sigma2 / weight), otherwise the term is
        # not a density in s_obs for every stress state and the sampler is
        # rewarded for changing the stress to change the weights.
        return -0.5 * safe_weight * pt.sum(diff**2, axis=-1) / sigma2 - 1.5 * pt.log(
            2 * np.pi * (sigma2 / safe_weight)
        )

    if family != "von_mises_fisher":
        raise ValueError(f"Unsupported slip likelihood family: {family}")
    if vmf_kappa is None:
        raise ValueError("vmf_kappa must be provided for the von_mises_fisher likelihood")

    # The weight scales the concentration; the normalizer must use the same
    # effective concentration (see Gaussian branch above).
    #
    # The slip direction of a double couple is not free on the sphere: it lies
    # on the great circle orthogonal to the fault normal. The normalizer is
    # therefore the circular one, 1 / (2*pi*I0(kappa_eff)), not the spherical
    # C3(kappa_eff) used before. At a fixed concentration the two differ by a
    # stress-independent constant and the posterior is unchanged, but with
    # shear-traction weighting kappa_eff varies with stress and the constant
    # does not cancel. log I0 is evaluated in the exponentially scaled form so
    # that large concentrations do not overflow.
    kappa_eff = safe_weight * float(vmf_kappa)
    dot = pt.clip(pt.sum(s_obs_unit * s_pred_unit, axis=-1), -1.0, 1.0)
    log_i0 = kappa_eff + pt.log(pt.ive(0.0, kappa_eff))
    return kappa_eff * dot - np.log(2.0 * np.pi) - log_i0


# Import stress_tensor_eigendecomposition from utils_stress
# (not defined here - use the one from utils_stress module)


def _build_joint_model(
    *,
    n1, n2, s1, s2,
    q_prior_mu_arr, q_prior_sigma,
    R_prior_mu_val, R_prior_sigma,
    infer_friction: bool,
    infer_friction_method: Optional[str],
    friction_fixed: Optional[float],
    friction_prior_params: Tuple[float, float],
    friction_range: Tuple[float, float],
    iterative_plane_selection: bool,
    iterative_info: Dict[str, Any],
    event_weights: Optional[np.ndarray],
    weighted_likelihood: bool,
    likelihood_weight_mode: str,
    tau_weight_exponent: float,
    normalize_tau_weights: bool,
    tau_weight_clip: Optional[Tuple[float, float]],
    slip_likelihood_name: str,
    slip_misfit_sigma: float,
    slip_vmf_kappa_val: Optional[float],
    instability_beta: float,
    selection_beta: Optional[float],
    signed_instability: bool,
    clustering_prior_strength: float,
    plane2_prior_probs_arr: Optional[np.ndarray],
    plane_prior_strength: float,
    enforce_constant_shear: bool,
    shear_weight: float,
    shear_sigma: float,
    shear_center: str,
    shear_target: Optional[float],
    shear_target_sigma: float,
    mechanism_angles=None,
    mechanism_errors=None,
    fixed_plane_indices=None,
    fault_population=None,
) -> Tuple["pm.Model", Optional[float]]:
    """Build the joint PyMC model shared by the SMC and NUTS entry points.

    Returns ``(model, mu_const)`` where ``mu_const`` is the Python float used
    for fixed-friction posterior reporting (``None`` when μ is sampled).
    """

    def quat_to_rotation_matrix(q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        R11 = 1 - 2 * (y * y + z * z)
        R22 = 1 - 2 * (x * x + z * z)
        R33 = 1 - 2 * (x * x + y * y)
        R12 = 2 * (x * y - z * w); R21 = 2 * (x * y + z * w)
        R13 = 2 * (x * z + y * w); R31 = 2 * (x * z - y * w)
        R23 = 2 * (y * z - x * w); R32 = 2 * (y * z + x * w)
        return pt.stacklists([[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]])

    def stress_tensor_from_R_and_shape(Rmat, Rratio):
        sig1 = -1.0
        sig2 = 2.0 * Rratio - 1.0
        sig3 = 1.0
        diag_vals = pt.stack([sig1, sig2, sig3])
        RD = Rmat * diag_vals
        return pt.dot(RD, Rmat.T)

    def unit_vector(x, eps=1e-12):
        norm = pt.sqrt(pt.sum(pt.square(x), axis=-1, keepdims=True))
        return x / (norm + eps)

    def shear_magnitude(Sigma, n):
        t = pt.dot(Sigma, n.T).T
        tn = pt.sum(t * n, axis=-1, keepdims=True) * n
        ts = t - tn
        return pt.sqrt(pt.sum(pt.square(ts), axis=-1) + 1e-12)

    def shear_traction_direction(Sigma, n):
        t = pt.dot(Sigma, n.T).T
        tn = pt.sum(t * n, axis=-1, keepdims=True) * n
        ts = t - tn
        return unit_vector(ts)

    def _apply_weight_clip(w, clip):
        if clip is None:
            return w
        lo, hi = float(clip[0]), float(clip[1])
        if lo <= 0.0 or hi <= 0.0 or hi < lo:
            raise ValueError("tau_weight_clip must be (low>0, high>0, high>=low)")
        return pt.clip(w, lo, hi)

    def _tau_weights(tau1, tau2):
        if not weighted_likelihood:
            return pt.ones_like(tau1), pt.ones_like(tau2)

        mode = str(likelihood_weight_mode or "event").lower()
        p = float(tau_weight_exponent)
        if p < 0.0:
            raise ValueError("tau_weight_exponent must be >= 0")
        if mode in {"plane", "plane_specific"}:
            # EXPERIMENTAL. Because the along-slip shear component is identical
            # on both nodal planes, cos(misfit_k) = c / tau_k, so a per-plane
            # weight tau_k**p multiplies the plane likelihood by tau_k**(p-1):
            # p < 1 favours the lower-shear plane, p == 1 makes the likelihood
            # blind to the plane, p > 1 favours the higher-shear plane. This
            # couples the weighting to plane selection and can bias R.
            warnings.warn(
                "likelihood_weight_mode='plane' is experimental: per-plane shear "
                "weights couple to nodal-plane selection and can bias R. "
                "Use likelihood_weight_mode='event'.",
                UserWarning,
                stacklevel=3,
            )

        if normalize_tau_weights:
            tau_scale = pt.mean(0.5 * (tau1 + tau2)) + 1e-12
        else:
            tau_scale = 1.0

        if mode in {"plane", "plane_specific"}:
            w1 = (tau1 / tau_scale) ** p
            w2 = (tau2 / tau_scale) ** p
        elif mode in {"event", "common", "symmetric"}:
            tau_evt = 0.5 * (tau1 + tau2)
            w_evt = (tau_evt / tau_scale) ** p
            w1 = w_evt
            w2 = w_evt
        elif mode in {"none", "off"}:
            w1 = pt.ones_like(tau1)
            w2 = pt.ones_like(tau2)
        else:
            raise ValueError("likelihood_weight_mode must be one of {'plane','event','none'}")

        w1 = _apply_weight_clip(w1, tau_weight_clip)
        w2 = _apply_weight_clip(w2, tau_weight_clip)
        return w1, w2

    mu_const: Optional[float] = None

    n_events_total = int(np.shape(n1)[0])
    pop_spec = _fp.resolve_population_spec(
        fault_population,
        selection_beta=float(instability_beta if selection_beta is None else selection_beta),
        friction_range=friction_range,
    )
    if pop_spec["family"] != _fp.LEGACY:
        if iterative_plane_selection or fixed_plane_indices is not None:
            raise ValueError(
                "A fault population requires the joint two-plane mixture; it cannot be "
                "combined with iterative preselection or externally fixed plane labels"
            )
        if pop_spec["normalize"] and pop_spec.get("fabric_mode") != "product":
            pop_spec["_table"] = _fp.normalizer_table(pop_spec)

    with pm.Model() as model:
        if mechanism_angles is not None:
            n1, s1, n2, s2 = latent_mechanism_vectors(
                mechanism_angles, mechanism_errors, n1, s1, n2, s2,
            )
        # --- Stress orientation as unit quaternion ---
        if pop_spec["family"] != _fp.LEGACY and q_prior_mu_arr is None:
            # Population models pin the orientation to a fraction of a degree;
            # a normalized Gaussian 4-vector then has a free radial direction
            # 20-100 times wider than the rotational ones and NUTS steps at the
            # narrow scale (tree depth at its cap).  The gnomonic chart has the
            # same Haar prior with no radial direction.  The legacy path keeps
            # its parameterization so that its results are reproducible.
            q, _ = _fp.gnomonic_rotation("q", ())
            pm.Deterministic("q_unit", q)
        else:
            if q_prior_mu_arr is not None:
                q_raw = pm.Normal("q_raw", mu=q_prior_mu_arr, sigma=q_prior_sigma, shape=4)
            else:
                q_raw = pm.Normal("q_raw", mu=0.0, sigma=1.0, shape=4)
            q_norm = pt.sqrt(pt.sum(pt.square(q_raw))) + 1e-8
            q = q_raw / q_norm
        Rmat = pm.Deterministic("R_matrix", quat_to_rotation_matrix(q))

        # --- Shape ratio R ∈ (0,1) ---
        if R_prior_mu_val is not None:
            Rratio = pm.TruncatedNormal(
                "R", mu=R_prior_mu_val, sigma=R_prior_sigma, lower=0.001, upper=0.999
            )
        else:
            Rratio = pm.Beta("R", alpha=2.0, beta=2.0)

        Sigma = pm.Deterministic("Sigma", stress_tensor_from_R_and_shape(Rmat, Rratio))

        # --- Friction coefficient (μ) ---
        if iterative_plane_selection:
            mu_const = float(
                iterative_info.get("mu", 0.5 * (friction_range[0] + friction_range[1]))
            )
            mu = pm.Deterministic("mu", pt.as_tensor_variable(mu_const))
        else:
            method = str(infer_friction_method or "sample").lower()
            if infer_friction and method == "sample":
                mu_raw = pm.Beta(
                    "mu_raw",
                    alpha=float(friction_prior_params[0]),
                    beta=float(friction_prior_params[1]),
                )
                mu = pm.Deterministic(
                    "mu",
                    float(friction_range[0])
                    + (float(friction_range[1]) - float(friction_range[0])) * mu_raw,
                )
            else:
                mu_const = (
                    float(friction_fixed)
                    if friction_fixed is not None
                    else 0.5 * (friction_range[0] + friction_range[1])
                )
                mu = pm.Deterministic("mu", pt.as_tensor_variable(mu_const))

        # --- Optional (stress-independent) event weights ---
        if event_weights is not None:
            w_event = pm.Data("event_weights", event_weights)
            w_event = pt.clip(w_event, 1e-12, 1e12)
        else:
            w_event = 1.0

        # --- Likelihood: either fixed planes (preselection) or plane mixture ---
        if iterative_plane_selection:
            if mechanism_errors is not None and np.any(mechanism_errors > 0):
                mask = iterative_info["plane_map"][:, None].astype(bool)
                n_selected = pt.where(mask, n2, n1)
                s_observed = pt.where(mask, s2, s1)
            else:
                n_selected = pm.Data("n_selected", iterative_info["n_selected"])
                s_observed = pm.Data("s_observed", iterative_info["s_selected"])
            pm.Deterministic(
                "p_plane2_post",
                pt.as_tensor_variable(iterative_info["plane_map"].astype(float)),
            )

            s_predicted = shear_traction_direction(Sigma, n_selected)
            tau_mag = shear_magnitude(Sigma, n_selected)
            if weighted_likelihood:
                w_tau = _tau_weights(tau_mag, tau_mag)[0]
                loglike_per_event = _slip_direction_logp(
                    s_observed, s_predicted,
                    family=slip_likelihood_name,
                    sigma=slip_misfit_sigma,
                    vmf_kappa=slip_vmf_kappa_val,
                    weight=w_event * w_tau,
                )
            else:
                loglike_per_event = _slip_direction_logp(
                    s_observed, s_predicted,
                    family=slip_likelihood_name,
                    sigma=slip_misfit_sigma,
                    vmf_kappa=slip_vmf_kappa_val,
                    weight=w_event,
                )
            pm.Potential("likelihood", pt.sum(loglike_per_event))
        elif fixed_plane_indices is not None:
            mask = fixed_plane_indices[:, None].astype(bool)
            n_selected = pt.where(mask, n2, n1)
            s_observed = pt.where(mask, s2, s1)
            pm.Deterministic("p_plane2_post", pt.as_tensor_variable(fixed_plane_indices.astype(float)))
            s_predicted = shear_traction_direction(Sigma, n_selected)
            tau_mag = shear_magnitude(Sigma, n_selected)
            # Preserve event-mode weighting from the joint likelihood: the
            # same two-plane mean shear, even when the fault label is fixed.
            w1_tau, w2_tau = _tau_weights(shear_magnitude(Sigma, n1), shear_magnitude(Sigma, n2))
            weight = w_event * pt.where(fixed_plane_indices.astype(bool), w2_tau, w1_tau)
            loglike_per_event = _slip_direction_logp(
                s_observed, s_predicted, family=slip_likelihood_name,
                sigma=slip_misfit_sigma, vmf_kappa=slip_vmf_kappa_val, weight=weight,
            )
            pm.Potential("likelihood", pt.sum(loglike_per_event))
        else:
            s_pred1 = shear_traction_direction(Sigma, n1)
            s_pred2 = shear_traction_direction(Sigma, n2)

            def instability_parameter_log(Sigma, n, mu, R, s_pred=None, s_obs=None):
                sig1 = -1.0
                denom = pt.sqrt(1.0 + mu ** 2)
                tau_c = 1.0 / denom
                sig_c = mu / denom

                t = pt.dot(Sigma, n.T).T
                sigma_n = pt.sum(t * n, axis=-1)

                tn = sigma_n[:, None] * n
                ts = t - tn
                tau_local = pt.sqrt(pt.sum(ts ** 2, axis=-1) + 1e-12)

                numerator = tau_local - mu * (sig1 - sigma_n)
                denominator_I = tau_c - mu * (sig1 - sig_c)
                I_val = numerator / denominator_I

                if signed_instability and s_pred is not None and s_obs is not None:
                    dot_product = pt.sum(s_pred * s_obs, axis=-1)
                    I_val = I_val * pt.sign(dot_product)

                return I_val

            tau1 = shear_magnitude(Sigma, n1)
            tau2 = shear_magnitude(Sigma, n2)
            inst1 = instability_parameter_log(Sigma, n1, mu, Rratio, s_pred=s_pred1, s_obs=s1)
            inst2 = instability_parameter_log(Sigma, n2, mu, Rratio, s_pred=s_pred2, s_obs=s2)

            w1_tau, w2_tau = _tau_weights(tau1, tau2)
            w1 = w_event * w1_tau
            w2 = w_event * w2_tau

            ll1 = _slip_direction_logp(
                s1, s_pred1,
                family=slip_likelihood_name,
                sigma=slip_misfit_sigma,
                vmf_kappa=slip_vmf_kappa_val,
                weight=w1,
            )
            ll2 = _slip_direction_logp(
                s2, s_pred2,
                family=slip_likelihood_name,
                sigma=slip_misfit_sigma,
                vmf_kappa=slip_vmf_kappa_val,
                weight=w2,
            )

            eps = 1e-9
            if pop_spec["family"] != _fp.LEGACY:
                # --- Generative fault-population mixture (normalized) ---
                # The plane weights follow from an explicit population density
                # g(I) on the sphere of fault normals, so the per-event term is
                # a probability density in the mechanism once divided by
                # Z = E_n[g(I(n))]. See fault_population.py.
                if signed_instability:
                    raise ValueError(
                        "signed_instability is not supported with a fault population: "
                        "the population density is defined on fault normals alone"
                    )
                if clustering_prior_strength > 0.0:
                    raise ValueError(
                        "clustering_prior_strength is superseded by the fabric component; "
                        "use fault_population={'family': ..., 'fabric_K': K}"
                    )
                if plane2_prior_probs_arr is not None and plane_prior_strength > 0.0:
                    raise ValueError(
                        "External plane priors are not yet supported with a fault population"
                    )

                if pop_spec["family"] == "ramp" and pop_spec["imin"] == "infer":
                    imin_lo, imin_hi = pop_spec["imin_bounds"]
                    imin_var = pm.Uniform("I_min", lower=float(imin_lo), upper=float(imin_hi))
                else:
                    imin_var = None

                log_g1 = _fp.log_population_weight_pt(inst1, pop_spec, imin_var)
                log_g2 = _fp.log_population_weight_pt(inst2, pop_spec, imin_var)

                mix_weight = None
                product_fabric = pop_spec["fabric_K"] > 0 and pop_spec["fabric_mode"] == "product"
                if pop_spec["mix_uniform"] or pop_spec["fabric_K"] > 0:
                    if not product_fabric:
                        mix_weight = pm.Uniform("w_population", lower=0.0, upper=1.0)
                    if pop_spec["fabric_K"] > 0:
                        K = int(pop_spec["fabric_K"])
                        rng_init = np.random.default_rng(0)
                        if K > 1:
                            fabric_pi = pm.Dirichlet(
                                "fabric_pi", a=float(pop_spec["fabric_pi_alpha"]) * np.ones(K),
                            )
                            log_pi = pt.log(fabric_pi)
                        else:
                            log_pi = pt.zeros(1)
                        if pop_spec["fabric_family"] == "bingham":
                            # One component is a rotation, giving three orthonormal
                            # axes, and two concentrations against the first two of
                            # them. Equal concentrations reproduce a Watson
                            # component; one large and one near zero give a girdle.
                            quats, _ = _fp.gnomonic_rotation(
                                "fabric_quat", (K,),
                                initval=_fp.gnomonic_initval(rng_init.normal(size=(K, 4))),
                            )
                            axes = pt.stack([_fp.quat_to_rotation_pt(quats[k]) for k in range(K)])
                            pm.Deterministic("fabric_axes", axes)
                            fabric_kappa = pm.HalfNormal(
                                "fabric_kappa", sigma=float(pop_spec["fabric_kappa_sigma"]),
                                shape=(K, 2),
                            )
                            log_h1 = _fp.log_bingham_mixture_pt(n1, axes, fabric_kappa, log_pi)
                            log_h2 = _fp.log_bingham_mixture_pt(n2, axes, fabric_kappa, log_pi)
                        else:
                            # A Watson axis is the third column of a uniformly
                            # random rotation (its marginal is uniform on the
                            # sphere); the rotation about the axis is a flat,
                            # harmless direction.  Same chart as the Bingham case.
                            quats, _ = _fp.gnomonic_rotation(
                                "fabric_quat", (K,),
                                initval=_fp.gnomonic_initval(rng_init.normal(size=(K, 4))),
                            )
                            axes = pt.stack([_fp.quat_to_rotation_pt(quats[k])[:, 2] for k in range(K)])
                            pm.Deterministic("fabric_axes", axes)
                            fabric_kappa = pm.HalfNormal(
                                "fabric_kappa", sigma=float(pop_spec["fabric_kappa_sigma"]), shape=K,
                            )
                            log_h1 = _fp.log_watson_mixture_pt(n1, axes, fabric_kappa, log_pi)
                            log_h2 = _fp.log_watson_mixture_pt(n2, axes, fabric_kappa, log_pi)
                    else:
                        log_h1 = pt.zeros_like(log_g1)
                        log_h2 = pt.zeros_like(log_g2)
                    if product_fabric:
                        # Faults exist per the fabric and reactivate per the
                        # stress: g(I(n)) h(n), normalized below by quasi-Monte
                        # Carlo because Z couples the fabric axes to the frame.
                        log_g1 = log_g1 + log_h1
                        log_g2 = log_g2 + log_h2
                    else:
                        log_w = pt.log(pt.clip(mix_weight, eps, 1.0 - eps))
                        log_1mw = pt.log1p(-pt.clip(mix_weight, eps, 1.0 - eps))
                        log_g1 = pt.logaddexp(log_w + log_g1, log_1mw + log_h1)
                        log_g2 = pt.logaddexp(log_w + log_g2, log_1mw + log_h2)

                if pop_spec["fabric_K"] > 1 and pop_spec["fabric_min_overlap"] > 0.0:
                    # Every component must overlap the dominant one: secondary
                    # families are rotations or splays of the main set, not
                    # unrelated sets.  See fault_population.py.
                    n_ov = _fp.sobol_unit_normals(pop_spec["product_qmc_power"], pop_spec["table"]["seed"])
                    n_ov_t = pt.as_tensor_variable(n_ov)
                    if pop_spec["fabric_family"] == "bingham":
                        log_comp = _fp.log_bingham_components_pt(n_ov_t, axes, fabric_kappa)
                    else:
                        log_comp = _fp.log_watson_components_pt(n_ov_t, axes, fabric_kappa)
                    overlap = _fp.component_overlap_pt(log_comp, n_ov.shape[0])
                    penalty, o_main = _fp.overlap_penalty_pt(
                        overlap, fabric_pi, pop_spec["fabric_min_overlap"],
                        pop_spec["fabric_overlap_strength"],
                    )
                    pm.Deterministic("fabric_overlap", o_main)
                    pm.Potential("fabric_overlap_penalty", penalty)

                # Prior plane probability implied by the population.
                log_gsum = pt.logaddexp(log_g1, log_g2)
                p2 = pm.Deterministic("p_plane2", pt.clip(pt.exp(log_g2 - log_gsum), eps, 1.0 - eps))

                logw1 = log_g1 + ll1
                logw2 = log_g2 + ll2
                logmix = pt.logaddexp(logw1, logw2)
                pm.Potential("likelihood", pt.sum(logmix))

                if pop_spec["normalize"] and product_fabric:
                    n_qmc = _fp.sobol_unit_normals(
                        pop_spec["product_qmc_power"], pop_spec["table"]["seed"]
                    )
                    n_qmc_t = pt.as_tensor_variable(n_qmc)
                    inst_q = instability_parameter_log(Sigma, n_qmc_t, mu, Rratio)
                    log_gq = _fp.log_population_weight_pt(inst_q, pop_spec, imin_var)
                    if pop_spec["fabric_family"] == "bingham":
                        log_hq = _fp.log_bingham_mixture_pt(n_qmc_t, axes, fabric_kappa, log_pi)
                    else:
                        log_hq = _fp.log_watson_mixture_pt(n_qmc_t, axes, fabric_kappa, log_pi)
                    log_Z = pt.logsumexp(log_gq + log_hq) - np.log(float(n_qmc.shape[0]))
                    pm.Deterministic("log_Z_population", log_Z)
                    pm.Potential("population_normalizer", -float(n_events_total) * log_Z)
                elif pop_spec["normalize"]:
                    log_Z = _fp.interp_normalizer_pt(
                        pop_spec["_table"], Rratio, mu, imin_var
                    )
                    if mix_weight is not None:
                        log_Z = pt.log(mix_weight * pt.exp(log_Z) + (1.0 - mix_weight))
                    pm.Deterministic("log_Z_population", log_Z)
                    pm.Potential("population_normalizer", -float(n_events_total) * log_Z)

                p_plane2_post = pm.Deterministic("p_plane2_post", pt.exp(logw2 - logmix))
            else:
                inst_delta = inst2 - inst1
                beta_val = float(instability_beta if selection_beta is None else selection_beta)
                beta = pt.as_tensor_variable(beta_val)
                logit_local = beta * inst_delta

                # Certainty-weighted clustering prior
                if clustering_prior_strength > 0.0:
                    p_tentative = pm.math.sigmoid(logit_local)
                    certainty = pt.square(2.0 * p_tentative - 1.0)

                    M1 = n1[:, :, None] * n1[:, None, :]
                    M2 = n2[:, :, None] * n2[:, None, :]

                    sum_certainty = pt.sum(certainty) + 1e-12
                    T_conf = pt.sum(
                        certainty[:, None, None] * (
                            (1.0 - p_tentative)[:, None, None] * M1
                            + p_tentative[:, None, None] * M2
                        ),
                        axis=0,
                    ) / sum_certainty

                    E1 = pt.sum(n1 * pt.dot(T_conf, n1.T).T, axis=-1)
                    E2 = pt.sum(n2 * pt.dot(T_conf, n2.T).T, axis=-1)

                    logit_local = logit_local + float(clustering_prior_strength) * (E2 - E1)

                if plane2_prior_probs_arr is not None and plane_prior_strength > 0.0:
                    prior_p2 = pt.as_tensor_variable(plane2_prior_probs_arr)
                    prior_p2 = pt.clip(prior_p2, eps, 1.0 - eps)
                    prior_logit = pt.log(prior_p2) - pt.log1p(-prior_p2)
                    p2 = pm.math.sigmoid(logit_local + float(plane_prior_strength) * prior_logit)
                else:
                    p2 = pm.math.sigmoid(logit_local)
                p2 = pt.clip(p2, eps, 1.0 - eps)
                p2 = pm.Deterministic("p_plane2", p2)

                logw1 = pt.log1p(-p2) + ll1
                logw2 = pt.log(p2) + ll2
                logmix = pt.logaddexp(logw1, logw2)
                pm.Potential("likelihood", pt.sum(logmix))

                p_plane2_post = pm.Deterministic("p_plane2_post", pt.exp(logw2 - logmix))

            tau_mag = (1.0 - p_plane2_post) * tau1 + p_plane2_post * tau2

        # --- Optional: constant-shear constraint ---
        # Encourages equal shear-traction magnitudes |τ| across events. Under a
        # uniform stress tensor, |τ| generally varies with fault orientation.
        if enforce_constant_shear:
            sw = float(shear_weight)
            if sw < 0.0:
                raise ValueError("shear_weight must be >= 0")
            if sw > 0.0:
                ss = float(shear_sigma)
                if ss <= 0.0:
                    raise ValueError("shear_sigma must be > 0 when enforce_constant_shear=True")

                sc = (shear_center or "mean").lower()
                if sc not in {"mean", "learned", "fixed"}:
                    raise ValueError("shear_center must be one of {'mean','learned','fixed'}")

                if sc == "mean":
                    tau0 = pm.Deterministic("tau0", pt.mean(tau_mag))
                elif sc == "fixed":
                    if shear_target is None:
                        raise ValueError("shear_center='fixed' requires shear_target to be set")
                    tau0 = pm.Deterministic("tau0", pt.as_tensor_variable(float(shear_target)))
                else:  # "learned"
                    tau0 = pm.Normal("tau0", mu=0.5, sigma=float(shear_target_sigma))

                resid = (tau_mag - tau0) / ss
                pen = 0.5 * pt.sum(resid * resid)
                pm.Potential("shear_const_penalty", -sw * pen)

    return model, mu_const



def summarize_posterior(idata, hdi_prob: float = 0.9) -> Dict[str, Any]:
    """Posterior summaries of a NUTS ``InferenceData`` used by the output tools.

    Every chain in ``idata`` is pooled.  Population models are multimodal on
    real catalogs, and a pooled element-wise median then mixes basins, so the
    per-basin outputs are produced by calling this on ``idata.sel(chain=...)``
    (see ``basins.subset_results``).  The keys mirror the sampler result:
    ``stress_tensor`` (element-wise median, normalized to unit largest
    eigenvalue), ``principal_*``, ``R_*``, ``posterior_principal_*``,
    ``hdi``/``hdi_90``, ``plane_probabilities`` and ``plane_selection_map``
    (mean plane-2 probability over draws, thresholded at one half), and the
    friction and ``tau0`` statistics when those were sampled.
    """
    R_samples = idata.posterior["R"].stack(s=("chain", "draw")).values
    Sigma_samples_raw = idata.posterior["Sigma"].stack(s=("chain", "draw")).values  # (3, 3, S)
    mu_samples = None
    if "mu" in idata.posterior:
        mu_samples = idata.posterior["mu"].stack(s=("chain", "draw")).values
    tau0_samples = None
    if "tau0" in idata.posterior:
        tau0_samples = idata.posterior["tau0"].stack(s=("chain", "draw")).values

    def _hdi_1d(arr):
        arr = np.asarray(arr).reshape(-1)
        if arr.size == 0:
            return None
        try:
            h = az.hdi(arr, hdi_prob=hdi_prob)
            h = np.asarray(h).reshape(-1)
            if h.size >= 2:
                return (float(h[0]), float(h[1]))
        except Exception:
            pass
        return None

    R_median = float(np.median(R_samples))
    R_mean = float(np.mean(R_samples))
    R_std = float(np.std(R_samples))
    R_CI95 = (float(np.quantile(R_samples, 0.025)), float(np.quantile(R_samples, 0.975)))

    Sigma_samples = np.empty_like(Sigma_samples_raw)
    for i in range(Sigma_samples.shape[2]):
        Sigma_i = Sigma_samples_raw[:, :, i]
        norm_i = np.max(np.abs(np.linalg.eigvalsh(Sigma_i)))
        Sigma_samples[:, :, i] = Sigma_i / (norm_i if norm_i > 0 else 1.0)

    sigma_stack = np.moveaxis(Sigma_samples, 2, 0)  # (S, 3, 3) for HDI
    Sigma_hdi = None
    try:
        Sigma_hdi = np.full((3, 3, 2), np.nan, dtype=float)
        for i in range(3):
            for j in range(3):
                h_ij = _hdi_1d(sigma_stack[:, i, j])
                if h_ij is not None:
                    Sigma_hdi[i, j, 0] = h_ij[0]
                    Sigma_hdi[i, j, 1] = h_ij[1]
    except Exception:
        Sigma_hdi = None

    R_hdi = _hdi_1d(R_samples)
    mu_hdi = _hdi_1d(mu_samples) if mu_samples is not None else None
    tau0_hdi = _hdi_1d(tau0_samples) if tau0_samples is not None else None

    Sigma_median = np.median(Sigma_samples, axis=2)
    norm = np.max(np.abs(np.linalg.eigvalsh(Sigma_median)))
    Sigma_median = Sigma_median / (norm if norm > 0 else 1.0)

    ps_median, pd_median = stress_tensor_eigendecomposition(Sigma_median)

    n_boot = Sigma_samples.shape[2]
    boot_ps = np.zeros((n_boot, 3), dtype=np.float64)
    boot_pd = np.zeros((n_boot, 3, 3), dtype=np.float64)
    for i in range(n_boot):
        ps_i, pd_i = stress_tensor_eigendecomposition(Sigma_samples[:, :, i])
        boot_ps[i, :] = ps_i
        boot_pd[i, :, :] = pd_i

    results: Dict[str, Any] = {
        "stress_tensor": Sigma_median,
        "principal_stresses": ps_median,
        "principal_directions": pd_median,
        "R_median": R_median,
        "R_mean": R_mean,
        "R_std": R_std,
        "R_CI95": R_CI95,
        "idata": idata,
        "posterior_principal_stresses": boot_ps,
        "posterior_principal_directions": boot_pd,
        "R_posterior": R_samples,
        "mu_samples": mu_samples,
        "tau0_samples": tau0_samples,
    }
    hdi_info = {
        "prob": float(hdi_prob),
        "R": R_hdi,
        "mu": mu_hdi,
        "tau0": tau0_hdi,
        "Sigma": Sigma_hdi,
        "stress_tensor": Sigma_hdi,
    }
    results["hdi"] = hdi_info
    if abs(float(hdi_prob) - 0.9) < 1e-6:
        results["hdi_90"] = hdi_info

    if "p_plane2_post" in idata.posterior:
        p2_s = idata.posterior["p_plane2_post"].stack(s=("chain", "draw")).values  # (N, S)
        plane_2_prob = np.mean(p2_s, axis=1)
        plane_1_prob = 1.0 - plane_2_prob
        results["plane_probabilities"] = np.stack([plane_1_prob, plane_2_prob], axis=1)
        results["plane_selection_map"] = (plane_2_prob > 0.5).astype(int)

    if mu_samples is not None and mu_samples.size > 0:
        results["friction_median"] = float(np.median(mu_samples))
        results["friction_mean"] = float(np.mean(mu_samples))
        results["friction_coefficient"] = float(np.mean(mu_samples))
        results["mu"] = float(np.median(mu_samples))
        results["friction_std"] = float(np.std(mu_samples))
        results["friction_CI95"] = (
            float(np.quantile(mu_samples, 0.025)),
            float(np.quantile(mu_samples, 0.975)),
        )
    if tau0_samples is not None and tau0_samples.size > 0:
        results["tau0_median"] = float(np.median(tau0_samples))
        results["tau0_mean"] = float(np.mean(tau0_samples))
        results["tau0_std"] = float(np.std(tau0_samples))
        results["tau0_CI95"] = (
            float(np.quantile(tau0_samples, 0.025)),
            float(np.quantile(tau0_samples, 0.975)),
        )
    return results


def _population_results(idata, fault_population) -> Optional[Dict[str, Any]]:
    """Summarize the fault-population parameters and the basin structure.

    Returns ``None`` for the historical unnormalized mixture. Population models
    are multimodal on real catalogs, so the basin report is part of the result
    rather than an optional diagnostic: a large r_hat there usually means the
    chains explored different basins and must be summarized separately.
    """
    if fault_population is None or str(fault_population).strip().lower() in {"legacy", "none", "off"}:
        return None
    # The resolved specification records every default that applied.
    spec = _fp.resolve_population_spec(fault_population, selection_beta=1.0, friction_range=(0.2, 1.0))
    out: Dict[str, Any] = {"spec": {k: v for k, v in spec.items() if not k.startswith("_")}}
    post = getattr(idata, "posterior", None)
    if post is None:
        return out
    for name in ("I_min", "w_population", "log_Z_population"):
        if name in post:
            values = post[name].values
            out[name] = {
                "median": float(np.median(values)),
                "equal_tail_90": [float(np.quantile(values, 0.05)),
                                  float(np.quantile(values, 0.95))],
            }
    if "fabric_overlap" in post:
        o = post["fabric_overlap"].values
        out["fabric_overlap"] = {
            "min_overlap": spec["fabric_min_overlap"],
            "strength": spec["fabric_overlap_strength"],
            "median": np.median(o, axis=(0, 1)).tolist(),
            "fraction_below": float(np.mean(np.any(o < spec["fabric_min_overlap"], axis=-1))),
            "prior_log_norm": _fp.overlap_prior_log_norm(spec),
        }
    if "fabric_kappa" in post:
        out["fabric_kappa_median"] = np.median(post["fabric_kappa"].values, axis=(0, 1)).tolist()
        if "fabric_pi" in post:
            out["fabric_pi_median"] = np.median(post["fabric_pi"].values, axis=(0, 1)).tolist()
    try:
        from . import basins as _basins
    except ImportError:
        try:
            import basins as _basins
        except ImportError:
            return out
    try:
        out["basins"] = _basins.basin_report(idata)
    except Exception as exc:  # pragma: no cover - diagnostics must not break a run
        warnings.warn(f"Basin report failed: {exc}", RuntimeWarning)
    return out


def Bayesian_joint_plane_selection_SMC(
    strikes_1: np.ndarray,
    dips_1: np.ndarray,
    rakes_1: np.ndarray,
    strikes_2: np.ndarray,
    dips_2: np.ndarray,
    rakes_2: np.ndarray,
    *,
    infer_friction: bool = False,
    infer_friction_method: str = "posthoc",
    friction_prior_params: Tuple[float, float] = (2.0, 2.0),
    friction_range: Tuple[float, float] = (0.2, 0.9),
    draws: int = 2500,
    chains: int = 2,
    cores: int = 2,
    threshold: float = 0.25,
    correlation_threshold: float = 0.01,
    kernel: str = "IMH",
    random_seed: Optional[int | list[int]] = None,
    progressbar: bool = True,
    return_plane_probabilities: bool = True,
    hdi_prob: float = 0.9,
    # Plane selection mode: 
    # True -> Iterative pre-selection (Michael + instability) to fix nodal planes + mu
    # False -> Bayesian joint selection (soft plane prior + slip likelihood)
    iterative_plane_selection: bool = False,
    iterative_kwargs: Optional[Dict[str, Any]] = None,
    # Plane-selection prior sharpness (sigmoid beta) of the legacy mixture and the
    # 'exp' population family; the ramp family does not use it. If None, falls
    # back to instability_beta for backward compatibility.
    selection_beta: Optional[float] = 10.0,
    # Scale for the Gaussian slip likelihood (small-angle approximation).
    slip_misfit_sigma: float = 0.35,
    # Slip-direction likelihood family.
    slip_likelihood: str = "gaussian",
    # Concentration for the vMF slip likelihood; defaults to 1 / slip_misfit_sigma**2.
    slip_vmf_kappa: Optional[float] = None,
    # Back-compat name (historically used for instability-based plane selection).
    # Controls the sharpness of the slip-tendency plane-selection prior when
    # selection_beta is None.
    instability_beta: float = 10.0,
    signed_instability: bool = False,
    friction_fixed: Optional[float] = None,
    # Optional: encourage constant shear magnitudes across events
    enforce_constant_shear: bool = False,
    shear_sigma: float = 0.2,
    shear_target: Optional[float] = None,
    shear_target_sigma: float = 0.3,
    # Controls how the constant‑shear constraint is centered/weighted
    # "mean": shrink variance around the current sample mean τ̄ (scale‑invariant, recommended)
    # "learned": infer a global τ0 ~ Normal(0.5, shear_target_sigma)
    # "fixed": fix τ0 to shear_target (must be provided)
    shear_center: str = "mean",
    shear_weight: float = 0.1,
    weighted_likelihood: bool = False,
    # "event": both nodal planes of an event share one weight from their mean
    # shear traction (recommended). "plane": per-plane weights (EXPERIMENTAL,
    # couples the weight to nodal-plane selection; see _tau_weights).
    likelihood_weight_mode: str = "event",
    # Exponent p of the per-event shear weights tau**p. p = 0 gives unit weights,
    # which the fault-population normalizer assumes.
    tau_weight_exponent: float = 0.0,
    normalize_tau_weights: bool = False,
    tau_weight_clip: Optional[Tuple[float, float]] = None,
    event_weights: Optional[np.ndarray] = None,
    event_weight_power: float = 1.0,
    q_prior_mu: Optional[np.ndarray] = None,
    q_prior_sigma: float = 1.0,
    R_prior_mu: Optional[float] = None,
    R_prior_sigma: float = 2.0,
    plane2_prior_probs: Optional[np.ndarray] = None,
    plane_prior_strength: float = 0.0,
    clustering_prior_strength: float = 0.0,
    strike_sigma_deg=0.0,
    dip_sigma_deg=0.0,
    rake_sigma_deg=0.0,
    fixed_plane_indices: Optional[np.ndarray] = None,
    # "default" is fault_population.DEFAULT_FAULT_POPULATION (ramp, inferred
    # I_min, one Bingham fabric component, overlap >= 0.2); None is the
    # historical unnormalized mixture.
    fault_population="default",
) -> Dict[str, Any]:
    """
    Joint Bayesian inference of stress orientation, shape ratio, and nodal plane selection using SMC.

    Measurement uncertainty: strike_sigma_deg, dip_sigma_deg, rake_sigma_deg
    are independent one-sigma local errors in degrees for the FIRST supplied
    nodal plane. Each accepts a scalar or an (N,) array. All three default to
    zero (exact-angle model: the mechanisms enter as fixed data). Nonzero
    values add three latent mechanism parameters per event, sharing one
    latent mechanism between both nodal planes. The directional likelihood
    still describes residual model scatter; these errors additionally
    describe SDR measurement uncertainty. Posterior canonical plane-1 angles are stored in
    idata.posterior["mechanism_angles_deg"] when any error is nonzero.

    fixed_plane_indices optionally fixes all fault labels externally (0/1 for
    input plane 1/2). This bypasses instability and clustering for selection,
    without fixing the uncertain geometry. It cannot be combined with iterative
    preselection. Friction is not identifiable from the fixed-plane likelihood.

    fault_population selects the model used for nodal-plane selection.

    ``"default"`` selects ``fault_population.DEFAULT_FAULT_POPULATION``: the
    ramp density with inferred ``I_min`` and one Bingham fabric component
    (``fabric_K=1``), whose extra components (``fabric_K > 1``) must overlap
    the dominant one by at least 0.2. ``None`` keeps the historical behaviour:
    the mixture weights are ``sigmoid(beta * (I2 - I1))`` and the mixture is
    used without a normalizer. That term is not a probability density in the
    observed mechanism, and its total mass grows with mu, which biases the
    inferred friction upward.

    Any other value replaces the logistic weight by an explicit fault-population
    density g(I) on the sphere of fault normals and adds the matching normalizer
    ``-N log Z(R, mu)``, so that the per-event term is a density. Pass a family
    name or a dictionary, for example

        fault_population="exp"
            g = exp(beta I), the generative model whose plane probabilities are
            identical to the historical weights, now normalized.
        fault_population={"family": "ramp", "imin": "infer", "fabric_K": 0}
            g = softplus(k (I - I_min)) / (k (1 - I_min)) with I_min inferred,
            the purely stress-controlled model.
        fault_population={"family": "ramp", "imin": "infer", "fabric_K": 2}
            K Bingham clusters for an inherited fabric, with the stress-selected
            fraction w inferred and reported as ``w_population``; components
            beyond the dominant one are perturbations of it (``fabric_min_overlap``).

    The normalizer is tabulated once per configuration and cached on disk; the
    first call with a new configuration spends a few minutes building it. See
    fault_population.py for the full set of options, and basins.py for the
    basin diagnostics and per-basin evidence these models require, since they
    are multimodal on real catalogs.

    This function implements the Sequential Monte Carlo approach described in SMC.md, which jointly
    infers stress parameters and selects the fault plane for each focal mechanism in a single
    probabilistic model. Unlike iterative plane selection methods, this approach:

    - Marginalizes plane selection with a two-plane mixture and a soft instability prior
    - Naturally handles multimodal posteriors (e.g., 180-degree stress orientation symmetries)
    - Propagates plane-selection uncertainty into stress parameter estimates
    - Optionally estimates friction coefficient mu post hoc from the inferred stress
    - Uses a selectable Gaussian or von Mises-Fisher slip-direction misfit

    By default plane selection uses a soft instability prior derived from the
    Mohr-Coulomb criterion:

        I = (tau - mu * (sig1 - sigma_n)) / (tau_c - mu * (sig1 - sig_c))

        p(z_i=1 | stress) = sigmoid(beta * (I2 - I1))

    where `beta` controls how hard the prior is (typical 2-12).

    Alternatively, set iterative_plane_selection=True to first run the deterministic
    plane-selection loop using the Michael (1984) constant-shear method with the
    instability criterion to select the fault planes and friction coefficient.
    Then SMC/NUTS is run on the selected planes (z fixed), sampling only stress
    parameters. This replicates the classic iterative selection before Bayesian stress
    inference.

    The model samples stress orientation (as a quaternion) and shape ratio R while
    marginalizing per-event plane choice. SMC (Sequential Monte Carlo) is well-suited
    for this problem because it handles multimodal distributions effectively.

    Parameters
    ----------
    strikes_1, dips_1, rakes_1 : np.ndarray
        Strike, dip, rake angles (degrees) for nodal plane 1 (auxiliary plane).
        Shape: (N,) where N = number of focal mechanisms.
    strikes_2, dips_2, rakes_2 : np.ndarray
        Strike, dip, rake angles (degrees) for nodal plane 2 (fault plane).
        Shape: (N,).
    infer_friction : bool, default False
        If True, infer friction coefficient μ either:
        - in-model ("sample"): sample μ jointly with stress (like the NUTS implementation), or
        - post hoc ("posthoc"): estimate μ after sampling by maximizing the mean Mohr-Coulomb
          instability over a grid in `friction_range` under the posterior-median stress.
        If False, μ is treated as fixed (set by friction_fixed or the midpoint of
        friction_range) and returned as such.
    infer_friction_method : str, default "posthoc"
        How to handle μ when infer_friction=True:
        - "sample": sample μ in the PyMC model (similar to NUTS).
        - "posthoc": estimate μ after sampling by instability maximization (historical SMC behavior).
    friction_prior_params : tuple of float, default (3.0, 3.0)
        (alpha, beta) parameters for Beta prior on friction, mapped to friction_range.
        Beta(3,3) is fairly neutral, centered around 0.5 → μ ≈ 0.6 with friction_range=(0.2,1.0).
    friction_range : tuple of float, default (0.2, 1.0)
        (min, max) range for friction coefficient μ. Beta prior is scaled to this range.
    draws : int, default 2500
        Number of SMC particles (samples) per chain. More draws = better posterior approximation.
    chains : int, default 2
        Number of independent SMC chains. Usually chains=1 with many draws for SMC.
    cores : int, default 2
        Number of CPU cores for parallel tempering stages within SMC.
    threshold : float, default 0.25
        ESS (Effective Sample Size) threshold for resampling. Lower = fewer tempering stages.
        Range: 0.2-0.4 typical. Higher = more accurate but slower.
    correlation_threshold : float, default 0.01
        Correlation threshold used by PyMC's SMC mutation kernel to adapt the number of
        Metropolis(-Hastings) rejuvenation steps. Lower values generally trigger more
        mutation steps (more mixing, slower); higher values trigger fewer steps (faster,
        higher risk of particle impoverishment / spiky marginals).
        Typical range: 0.005–0.05.
    kernel : str, default "IMH"
        SMC kernel: "IMH" (Independent Metropolis-Hastings) or "MH" (Metropolis-Hastings).
        IMH generally works better for SMC.
    random_seed : int or list[int], optional
        Random seed(s) for reproducibility (PyMC accepts one seed per chain).
    progressbar : bool, default True
        Show progress bar during sampling.
    return_plane_probabilities : bool, default True
        If True, compute posterior probabilities p(z_i=1|data) for each event.
    n_bootstrap_samples : int, default 1000
        Number of posterior samples to use for bootstrap principal stress/direction arrays.
        If larger than available posterior samples, uses all available samples.
    iterative_plane_selection : bool, default False
        If True, run ilsi.inversion_one_set_instability with variable_shear=False (Michael),
        friction_coefficient=None (grid search), to pick planes and μ. Then run the Bayesian 
        inversion on those fixed planes (no latent z). Plane probabilities are one-hot and 
        the selected μ is returned in results["friction_coefficient_preselected"].
        If False (default), use the joint Bayesian selection with an instability-driven prior.
    slip_likelihood : {"gaussian", "von_mises_fisher"}, default "gaussian"
        Slip-direction likelihood family. The Gaussian option uses the historical
        small-angle proxy on the unit slip vectors, while `von_mises_fisher` uses a
        true directional likelihood on the sphere.
    slip_misfit_sigma : float, default 0.35
        Scale parameter for the Gaussian slip likelihood. When
        `slip_vmf_kappa` is not provided, the vMF path uses the small-angle mapping
        `kappa ~= 1 / slip_misfit_sigma**2`.
    slip_vmf_kappa : float, optional
        Explicit concentration parameter for the vMF slip likelihood. Must be > 0.
    instability_beta : float, default 6.0
        Temperature for the instability prior (higher → closer to hard argmax). Typical 2–12.
    signed_instability : bool, default False
        If True, multiply I by sign(shear·slip) as in Beaucé (2022).
    friction_fixed : float, optional
        If iterative_plane_selection=False and infer_friction=False, use this fixed friction.
        Defaults to the midpoint of friction_range when None.
    preselect_kwargs : dict, optional
        Extra kwargs forwarded to ilsi.inversion_one_set_instability, e.g.,
        {"n_averaging": 1, "n_random_selections": 20, "n_stress_iter": 10,
         "friction_min": 0.2, "friction_max": 0.8, "friction_step": 0.05,
         "signed_instability": False, "Tarantola_kwargs": {…}}.
    enforce_constant_shear : bool, default False
        If True, adds a Gaussian penalty that encourages predicted shear magnitudes τ_i
        to be nearly constant across events.
        By default (shear_center="mean"), this shrinks the variance Var(τ) by centering at
        the sample mean τ̄ for each posterior draw, which is scale‑invariant and avoids
        pulling the stress toward a specific τ level.
        Alternative modes:
        - shear_center="learned": infer a global τ0 ~ Normal(0.5, shear_target_sigma)
        - shear_center="fixed": fix τ0 to shear_target (must be provided)
    shear_sigma : float, default 0.2
        Standard deviation of the constant-shear penalty (smaller → stronger enforcement).
    shear_target : float, optional
        If provided, fixes τ0 to this value.
    shear_target_sigma : float, default 0.3
        Prior std for τ0 when it is inferred (no shear_target provided).
    shear_center : {"mean","learned","fixed"}, default "mean"
        Centering mode for the constant‑shear constraint; see enforce_constant_shear.
    shear_weight : float, default 1.0
        Extra weight applied to the constant‑shear penalty. Reduce (e.g., 0.2–0.5) if it
        overpowers slip‑fit likelihood; increase if too weak.

    Returns
    -------
    results : dict
        Dictionary with keys:
        - "stress_tensor": (3,3) ndarray, median posterior stress tensor (normalized, deviatoric)
        - "principal_stresses": (3,) ndarray, eigenvalues (σ1, σ2, σ3)
        - "principal_directions": (3,3) ndarray, eigenvectors as columns
        - "R_median": float, median shape ratio
        - "R_mean": float, mean shape ratio
        - "R_std": float, std dev of shape ratio
        - "R_CI95": tuple, (2.5%, 97.5%) quantiles of R
        - "idata": arviz InferenceData object with full posterior
        - "posterior_principal_stresses": (B, 3) ndarray, posterior samples of principal stresses
        - "posterior_principal_directions": (B, 3, 3) ndarray, posterior samples of principal directions
        - "R_posterior": (B,) ndarray, posterior samples of the shape ratio R
        - "plane_probabilities": (N, 2) ndarray, [(p_plane1, p_plane2), ...] if return_plane_probabilities=True
        - "plane_selection_map": (N,) int ndarray, MAP estimate of plane selection (0=plane1, 1=plane2)
        - "friction_coefficient": float, median friction (if infer_friction=True)
        - "friction_mean": float, mean friction (if infer_friction=True)
        - "friction_std": float, std dev of friction (if infer_friction=True)
        - "friction_CI95": tuple, friction credible interval (if infer_friction=True)
        - "convergence": dict with SMC diagnostics

    Notes
    -----
    - This function requires PyMC with SMC support (pm.sample_smc)
    - The model uses a quaternion parameterization for stress orientation to avoid gimbal lock
    - Uses a selectable Gaussian or vMF slip-direction likelihood, optionally weighted by tau^2
    - SMC is robust to multimodal posteriors; plane choice is marginalized via a two-plane mixture
    - Typical runtime: ~30s-2min for N=20-50 events with default parameters
    - **Recommended for N < 50 events due to particle degeneracy in high-dimensional posteriors**

    **Important for Jupyter notebooks:**
    To control CPU usage, set thread limits BEFORE importing NumPy/PyMC:

    >>> import os
    >>> os.environ["OMP_NUM_THREADS"] = "2"
    >>> os.environ["MKL_NUM_THREADS"] = "2"
    >>> os.environ["OPENBLAS_NUM_THREADS"] = "2"
    >>> # Then import and use the function
    >>> from stress_mc.src.ilsi_smc import Bayesian_joint_plane_selection_SMC

    References
    ----------
    - Vavryčuk (2014): "Iterative joint inversion for stress and fault orientations"
    - SMC.md documentation
    - Beaucé et al. (2022): "An Iterative Linear Method with Variable Shear Stress Magnitudes"

    Examples
    --------
    >>> # Basic usage: Wallace-Bott only (slip direction fit, no friction)
    >>> result = Bayesian_joint_plane_selection_SMC(
    ...     strikes_1, dips_1, rakes_1,
    ...     strikes_2, dips_2, rakes_2,
    ...     draws=1000, cores=4
    ... )
    >>> print(f"Shape ratio R = {result['R_median']:.2f} ± {result['R_std']:.2f}")

    >>> # Infer friction coefficient
    >>> result = Bayesian_joint_plane_selection_SMC(
    ...     strikes_1, dips_1, rakes_1,
    ...     strikes_2, dips_2, rakes_2,
    ...     infer_friction=True,
    ...     friction_prior_params=(3.0, 3.0),
    ...     friction_range=(0.2, 0.9),
    ...     draws=1500
    ... )
    >>> print(f"Friction μ = {result['friction_coefficient']:.2f}")

    >>> # RECOMMENDED WORKFLOW: SMC stress inference + deterministic plane selection
    >>> # Step 1: Use SMC to infer stress tensor
    >>> result = Bayesian_joint_plane_selection_SMC(
    ...     strikes_1, dips_1, rakes_1,
    ...     strikes_2, dips_2, rakes_2,
    ...     infer_friction=True,
    ...     draws=1500
    ... )
    >>>
    >>> # Step 2: Use instability-based plane selection on posterior stress
    >>> import ilsi
    >>> I, fp_strikes, fp_dips, fp_rakes = ilsi.compute_instability_parameter(
    ...     result["principal_directions"],
    ...     result["R_median"],
    ...     result["friction_coefficient"],
    ...     strikes_1, dips_1, rakes_1,
    ...     strikes_2, dips_2, rakes_2,
    ...     return_fault_planes=True  # Selects most unstable plane
    ... )
    >>>
    >>> # Now fp_strikes, fp_dips, fp_rakes contain properly selected planes
    >>> # that incorporate BOTH Wallace-Bott (from SMC) AND Mohr-Coulomb (from instability)
    """
    strikes_1 = np.asarray(strikes_1, dtype=np.float64)
    dips_1 = np.asarray(dips_1, dtype=np.float64)
    rakes_1 = np.asarray(rakes_1, dtype=np.float64)
    strikes_2 = np.asarray(strikes_2, dtype=np.float64)
    dips_2 = np.asarray(dips_2, dtype=np.float64)
    rakes_2 = np.asarray(rakes_2, dtype=np.float64)

    N = len(strikes_1)
    if not (len(dips_1) == len(rakes_1) == len(strikes_2) == len(dips_2) == len(rakes_2) == N):
        raise ValueError("All input arrays must have the same length")

    mechanism_angles, mechanism_errors = prepare_mechanism_errors(
        strikes_1, dips_1, rakes_1, strike_sigma_deg, dip_sigma_deg, rake_sigma_deg,
    )
    fixed_plane_indices = validate_fixed_planes(fixed_plane_indices, N, iterative_plane_selection)

    # Convert focal mechanisms to normal and slip unit vectors
    n1, s1 = normal_slip_vectors_batch(strikes_1, dips_1, rakes_1, direction="inward")
    n2, s2 = normal_slip_vectors_batch(strikes_2, dips_2, rakes_2, direction="inward")

    if event_weights is not None:
        event_weights = np.asarray(event_weights, dtype=np.float64).reshape(-1)
        if event_weights.size != N:
            raise ValueError(f"event_weights must have shape (N,), got {event_weights.shape} for N={N}")
        if not np.all(np.isfinite(event_weights)):
            raise ValueError("event_weights must be finite")
        if float(event_weight_power) != 1.0:
            event_weights = event_weights ** float(event_weight_power)

    q_prior_mu_arr = None
    if q_prior_mu is not None:
        q_prior_mu_arr = np.asarray(q_prior_mu, dtype=np.float64).reshape(-1)
        if q_prior_mu_arr.size != 4:
            raise ValueError("q_prior_mu must have 4 elements (quaternion components)")
        if not np.all(np.isfinite(q_prior_mu_arr)):
            raise ValueError("q_prior_mu must be finite")
        qn = float(np.linalg.norm(q_prior_mu_arr))
        if qn <= 0.0:
            raise ValueError("q_prior_mu norm must be > 0")
        q_prior_mu_arr = q_prior_mu_arr / qn
        if q_prior_mu_arr[0] < 0:
            q_prior_mu_arr = -q_prior_mu_arr

    q_prior_sigma = float(q_prior_sigma)
    if q_prior_sigma <= 0.0:
        raise ValueError("q_prior_sigma must be > 0")

    R_prior_mu_val = None
    if R_prior_mu is not None:
        R_prior_mu_val = float(R_prior_mu)
        if not (0.0 < R_prior_mu_val < 1.0):
            raise ValueError("R_prior_mu must be in (0, 1)")

    R_prior_sigma = float(R_prior_sigma)
    if R_prior_sigma <= 0.0:
        raise ValueError("R_prior_sigma must be > 0")

    plane2_prior_probs_arr = None
    if plane2_prior_probs is not None:
        plane2_prior_probs_arr = np.asarray(plane2_prior_probs, dtype=np.float64).reshape(-1)
        if plane2_prior_probs_arr.size != N:
            raise ValueError(
                f"plane2_prior_probs must have shape (N,), got {plane2_prior_probs_arr.shape} for N={N}"
            )
        if not np.all(np.isfinite(plane2_prior_probs_arr)):
            raise ValueError("plane2_prior_probs must be finite")
        plane2_prior_probs_arr = np.clip(plane2_prior_probs_arr, 1e-6, 1.0 - 1e-6)

    plane_prior_strength = float(plane_prior_strength)
    if plane_prior_strength < 0.0:
        raise ValueError("plane_prior_strength must be >= 0")
        
    clustering_prior_strength = float(clustering_prior_strength)
    if clustering_prior_strength < 0.0:
        raise ValueError("clustering_prior_strength must be >= 0")

    slip_likelihood_name, slip_misfit_sigma, slip_vmf_kappa_val = (
        _resolve_slip_likelihood_params(
            slip_likelihood,
            slip_misfit_sigma,
            slip_vmf_kappa,
        )
    )

    # --------------------------------------------------------
    # Optional iterative preselection (Michael + instability)
    # --------------------------------------------------------
    iterative_info: Dict[str, Any] = {}
    if iterative_plane_selection:
        if _det is None:
            raise RuntimeError("ilsi module not available; cannot perform iterative preselection")
        opts = {} if iterative_kwargs is None else dict(iterative_kwargs)
        # Force Michael method and unknown friction
        opts.setdefault("variable_shear", False)
        opts.setdefault("friction_coefficient", None)
        # Reasonable defaults for grid search if not provided
        opts.setdefault("friction_min", friction_range[0])
        opts.setdefault("friction_max", friction_range[1])
        opts.setdefault("friction_step", 0.05)
        # Run one-set inversion with instability to get stress + μ
        out = _det.inversion_one_set_instability(
            strikes_1, dips_1, rakes_1, **opts
        )
        mu_sel = float(out.get("friction_coefficient", 0.5 * (friction_range[0] + friction_range[1])))
        ps = out["principal_stresses"]; pd = out["principal_directions"]
        R_sel = float(_det.utils_stress.R_(ps))
        # Select planes deterministically from final stress + μ
        Ivals, fp_strikes, fp_dips, fp_rakes = _det.compute_instability_parameter(
            pd, R_sel, mu_sel,
            strikes_1, dips_1, rakes_1,
            strikes_2, dips_2, rakes_2,
            return_fault_planes=True,
            signed_instability=bool(opts.get("signed_instability", False)),
        )
        # Build selected normals/slips
        n_sel, s_sel = normal_slip_vectors_batch(fp_strikes, fp_dips, fp_rakes, direction="inward")
        # Plane map from I comparison
        plane2_mask = (Ivals[:, 1] > Ivals[:, 0])
        plane_map = plane2_mask.astype(int)
        plane_probs = np.column_stack([1 - plane2_mask, plane2_mask]).astype(float)
        iterative_info = {
            "mu": mu_sel,
            "R_from_preselection": R_sel,
            "fault_planes": (fp_strikes, fp_dips, fp_rakes),
            "plane_probabilities": plane_probs,
            "plane_map": plane_map,
            "n_selected": n_sel,
            "s_selected": s_sel,
        }

    model, _mu_const_unused = _build_joint_model(
        n1=n1, n2=n2, s1=s1, s2=s2,
        mechanism_angles=mechanism_angles, mechanism_errors=mechanism_errors,
        fixed_plane_indices=fixed_plane_indices,
        fault_population=fault_population,
        q_prior_mu_arr=q_prior_mu_arr, q_prior_sigma=q_prior_sigma,
        R_prior_mu_val=R_prior_mu_val, R_prior_sigma=R_prior_sigma,
        infer_friction=infer_friction,
        infer_friction_method=infer_friction_method,
        friction_fixed=friction_fixed,
        friction_prior_params=friction_prior_params,
        friction_range=friction_range,
        iterative_plane_selection=iterative_plane_selection,
        iterative_info=iterative_info,
        event_weights=event_weights,
        weighted_likelihood=weighted_likelihood,
        likelihood_weight_mode=likelihood_weight_mode,
        tau_weight_exponent=tau_weight_exponent,
        normalize_tau_weights=normalize_tau_weights,
        tau_weight_clip=tau_weight_clip,
        slip_likelihood_name=slip_likelihood_name,
        slip_misfit_sigma=slip_misfit_sigma,
        slip_vmf_kappa_val=slip_vmf_kappa_val,
        instability_beta=instability_beta,
        selection_beta=selection_beta,
        signed_instability=signed_instability,
        clustering_prior_strength=clustering_prior_strength,
        plane2_prior_probs_arr=plane2_prior_probs_arr,
        plane_prior_strength=plane_prior_strength,
        enforce_constant_shear=enforce_constant_shear,
        shear_weight=shear_weight,
        shear_sigma=shear_sigma,
        shear_center=shear_center,
        shear_target=shear_target,
        shear_target_sigma=shear_target_sigma,
    )

    with model:
        # --- SMC Sampling ---
        kernel_cls = pm.smc.kernels.IMH if kernel.upper() == "IMH" else pm.smc.kernels.MH
        idata = pm.sample_smc(
            draws=draws,
            kernel=kernel_cls,
            threshold=threshold,
            correlation_threshold=correlation_threshold,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            progressbar=progressbar,
            compute_convergence_checks=False,
            return_inferencedata=True,
        )

    # --- Post-processing ---
    # Extract posterior samples
    R_samples = idata.posterior["R"].stack(s=("chain", "draw")).values
    # z is absent when planes are fixed by preselection or when planes are marginalized
    z_samples = None
    if "z" in idata.posterior:
        z_samples = idata.posterior["z"].stack(s=("chain", "draw")).values  # (N, S)
    Sigma_samples = idata.posterior["Sigma"].stack(s=("chain", "draw")).values  # (3, 3, S)

    # Compute posterior statistics
    R_median = float(np.median(R_samples))
    R_mean = float(np.mean(R_samples))
    R_std = float(np.std(R_samples))
    R_CI95 = (float(np.quantile(R_samples, 0.025)), float(np.quantile(R_samples, 0.975)))

    # Median stress tensor
    Sigma_median = np.median(Sigma_samples, axis=2)  # (3, 3)
    norm = np.max(np.abs(np.linalg.eigvalsh(Sigma_median)))
    Sigma_median = Sigma_median / (norm if norm > 0 else 1.0)

    # Eigendecomposition of median stress tensor
    ps_median, pd_median = stress_tensor_eigendecomposition(Sigma_median)

    # --- HDI Calculation Helper ---
    def _hdi_1d(arr):
        arr = np.asarray(arr).reshape(-1)
        if arr.size == 0:
            return None
        try:
            h = az.hdi(arr, hdi_prob=hdi_prob)
            h = np.asarray(h).reshape(-1)
            if h.size >= 2:
                return (float(h[0]), float(h[1]))
        except Exception:
            pass
        return None

    # Extract other samples for HDI
    mu_samples = None
    if "mu" in idata.posterior:
        mu_samples = idata.posterior["mu"].stack(s=("chain", "draw")).values
    tau0_samples = None
    if "tau0" in idata.posterior:
        tau0_samples = idata.posterior["tau0"].stack(s=("chain", "draw")).values

    # Normalize Sigma samples for HDI and bootstrap
    Sigma_samples_norm = np.empty_like(Sigma_samples)
    for i in range(Sigma_samples.shape[2]):
        Sigma_i = Sigma_samples[:, :, i]
        norm_i = np.max(np.abs(np.linalg.eigvalsh(Sigma_i)))
        Sigma_samples_norm[:, :, i] = Sigma_i / (norm_i if norm_i > 0 else 1.0)
    
    # Compute HDI for Sigma components
    sigma_stack = np.moveaxis(Sigma_samples_norm, 2, 0)
    Sigma_hdi = None
    try:
        Sigma_hdi = np.full((3, 3, 2), np.nan, dtype=float)
        for i in range(3):
            for j in range(3):
                h_ij = _hdi_1d(sigma_stack[:, i, j])
                if h_ij is not None:
                    Sigma_hdi[i, j, 0] = h_ij[0]
                    Sigma_hdi[i, j, 1] = h_ij[1]
    except Exception:
        Sigma_hdi = None

    R_hdi = _hdi_1d(R_samples)
    mu_hdi = _hdi_1d(mu_samples) if mu_samples is not None else None
    tau0_hdi = _hdi_1d(tau0_samples) if tau0_samples is not None else None

    # Generate bootstrap samples from ALL posterior samples
    n_boot = Sigma_samples.shape[2]
    boot_ps = np.zeros((n_boot, 3), dtype=np.float64)
    boot_pd = np.zeros((n_boot, 3, 3), dtype=np.float64)

    for i in range(n_boot):
        # use normalized samples
        Sigma_i = Sigma_samples_norm[:, :, i]
        ps_i, pd_i = stress_tensor_eigendecomposition(Sigma_i)
        boot_ps[i, :] = ps_i
        boot_pd[i, :, :] = pd_i

    # Determine mu_const for compatibility if mu is not sampled
    mu_const = float(friction_fixed) if friction_fixed is not None else 0.5 * (
        float(friction_range[0]) + float(friction_range[1])
    )

    results = {
        "stress_tensor": Sigma_median,
        "principal_stresses": ps_median,
        "principal_directions": pd_median,
        "R_median": R_median,
        "R_mean": R_mean,
        "R_std": R_std,
        "R_CI95": R_CI95,
        "idata": idata,
        "posterior_principal_stresses": boot_ps,
        "posterior_principal_directions": boot_pd,
        "R_posterior": R_samples,
        "mu_samples": mu_samples,
        "tau0_samples": tau0_samples,
        "hdi": {
            "prob": float(hdi_prob),
            "R": R_hdi,
            "Sigma": Sigma_hdi,
            "mu": mu_hdi,
            "tau0": tau0_hdi,
        },
        # For compatibility with post-processing scripts expecting explicit 'mu'
        "mu": np.median(mu_samples) if mu_samples is not None else mu_const,
    }
    if abs(float(hdi_prob) - 0.9) < 1e-6:
        results["hdi_90"] = {  # Explicit key for clarity
            "prob": float(hdi_prob),
            "R": R_hdi,
            "Sigma": Sigma_hdi,
            "mu": mu_hdi,
            "tau0": tau0_hdi,
        }

    results["mechanism_sigma_deg"] = mechanism_errors.copy()
    results["fixed_plane_indices"] = None if fixed_plane_indices is None else fixed_plane_indices.copy()
    results["fault_population"] = _population_results(idata, fault_population)

    # Plane selection probabilities and MAP estimates
    if return_plane_probabilities:
        if iterative_plane_selection:
            results["plane_probabilities"] = iterative_info.get("plane_probabilities")
            results["plane_selection_map"] = iterative_info.get("plane_map")
        else:
            # Posterior probability that plane 2 is selected for each event
            if "p_plane2_post" in idata.posterior:
                p2_s = idata.posterior["p_plane2_post"].stack(s=("chain", "draw")).values  # (N, S)
                plane_2_prob = np.mean(p2_s, axis=1)
            elif z_samples is not None:
                plane_2_prob = np.mean(z_samples, axis=1)  # (N,)
            else:
                raise RuntimeError("Internal error: neither p_plane2_post nor z present in posterior")
            plane_1_prob = 1.0 - plane_2_prob  # (N,)
            plane_probabilities = np.stack([plane_1_prob, plane_2_prob], axis=1)
            plane_selection_map = (plane_2_prob > 0.5).astype(int)
            results["plane_probabilities"] = plane_probabilities
            results["plane_selection_map"] = plane_selection_map

    # Friction statistics
    # - iterative_plane_selection: deterministic μ returned by preselection
    # - infer_friction + infer_friction_method="sample": μ sampled in-model (like NUTS)
    # - infer_friction + infer_friction_method="posthoc": estimate μ by instability maximization
    # - otherwise: return the fixed μ used in the model
    if iterative_plane_selection:
        results["friction_coefficient"] = None
    elif infer_friction and str(infer_friction_method or "posthoc").lower() == "posthoc":
        mu_min, mu_max = float(friction_range[0]), float(friction_range[1])
        if _det is None:
            mu_hat = None
        else:
            mu_grid = np.linspace(mu_min, mu_max, 76)  # step≈0.01 for default range
            scores = np.empty(mu_grid.size, dtype=float)
            for i, mu_try in enumerate(mu_grid):
                I_try = _det.compute_instability_parameter(
                    pd_median,
                    float(R_median),
                    float(mu_try),
                    strikes_1,
                    dips_1,
                    rakes_1,
                    strikes_2,
                    dips_2,
                    rakes_2,
                    return_fault_planes=False,
                    signed_instability=bool(signed_instability),
                )
                scores[i] = float(np.mean(np.max(I_try, axis=1)))
            mu_hat = float(mu_grid[int(np.argmax(scores))])

        results["friction_coefficient"] = mu_hat
        results["mu"] = mu_hat
        if mu_hat is not None:
            results["friction_median"] = float(mu_hat)
            results["friction_mean"] = float(mu_hat)
            results["friction_std"] = 0.0
            results["friction_CI95"] = (float(mu_hat), float(mu_hat))
            # Align HDI output with the returned μ estimate (μ is not sampled here)
            results["hdi"]["mu"] = (float(mu_hat), float(mu_hat))
            if "hdi_90" in results:
                results["hdi_90"]["mu"] = (float(mu_hat), float(mu_hat))
    else:
        # μ either sampled in-model ("sample") or fixed.
        if mu_samples is not None and mu_samples.size > 0:
            mu_med = float(np.median(mu_samples))
            mu_mean = float(np.mean(mu_samples))
            mu_std = float(np.std(mu_samples))
            mu_ci = (
                float(np.quantile(mu_samples, 0.025)),
                float(np.quantile(mu_samples, 0.975)),
            )
            results["friction_coefficient"] = mu_mean
            results["mu"] = mu_med
            results["friction_median"] = mu_med
            results["friction_mean"] = mu_mean
            results["friction_std"] = mu_std
            results["friction_CI95"] = mu_ci
        else:
            mu_fixed = float(friction_fixed) if friction_fixed is not None else 0.5 * (
                float(friction_range[0]) + float(friction_range[1])
            )
            results["friction_coefficient"] = mu_fixed
            results["mu"] = mu_fixed
            results["friction_median"] = mu_fixed
            results["friction_mean"] = mu_fixed
            results["friction_std"] = 0.0
            results["friction_CI95"] = (mu_fixed, mu_fixed)

    # Convergence diagnostics
    results["convergence"] = {
        "n_samples": draws * chains,
        "n_chains": chains,
        "smc_kernel": kernel,
        "smc_threshold": float(threshold),
        "smc_correlation_threshold": float(correlation_threshold),
        "smc_random_seed": random_seed,
    }

    # Optional shear target summary
    if tau0_samples is not None and tau0_samples.size > 0:
        results["tau0_median"] = float(np.median(tau0_samples))
        results["tau0_mean"] = float(np.mean(tau0_samples))
        results["tau0_std"] = float(np.std(tau0_samples))
        results["tau0_CI95"] = (
            float(np.quantile(tau0_samples, 0.025)),
            float(np.quantile(tau0_samples, 0.975)),
        )

    if iterative_plane_selection:
        # Ensure friction in outputs matches the preselection
        mu_sel = iterative_info.get("mu")
        if mu_sel is not None:
            results["friction_coefficient"] = float(mu_sel)
            results["friction_median"] = float(mu_sel)
            results["friction_mean"] = float(mu_sel)
            results["friction_std"] = 0.0
            results["friction_CI95"] = (float(mu_sel), float(mu_sel))
        results["friction_coefficient_preselected"] = mu_sel
        results["preselected_planes"] = iterative_info.get("fault_planes")
        if "R_from_preselection" in iterative_info:
            results["R_preselected"] = iterative_info["R_from_preselection"]
        results["preselection"] = "Michael+instability"

    return results


def Bayesian_joint_plane_selection_NUTS(
    strikes_1: np.ndarray,
    dips_1: np.ndarray,
    rakes_1: np.ndarray,
    strikes_2: np.ndarray,
    dips_2: np.ndarray,
    rakes_2: np.ndarray,
    *,
    infer_friction: bool = False,
    infer_friction_method: str = "sample",
    friction_prior_params: Tuple[float, float] = (3.0, 3.0),
    friction_range: Tuple[float, float] = (0.2, 1.0),
    draws: int = 2500,
    tune: int = 1000,
    chains: int = 2,
    cores: int = 2,
    target_accept: float = 0.6,
    # "smc": BlackJAX adaptive tempered SMC (bjsi_smc.sample_smc); draws are
    # then particles and chains independent runs, with cores of them
    # concurrent.  "nutpie", "numpyro" or "pymc" select a NUTS backend.
    nuts_sampler: Optional[str] = "smc",
    nuts_sampler_kwargs: Optional[Dict[str, Any]] = None,
    random_seed: Optional[int] = None,
    progressbar: bool = True,
    return_plane_probabilities: bool = True,
    hdi_prob: float = 0.9,
    # Plane selection mode: 
    # True -> Iterative pre-selection (Michael + instability) to fix nodal planes + mu
    # False -> Bayesian joint selection (soft plane prior + slip likelihood)
    iterative_plane_selection: bool = False,
    iterative_kwargs: Optional[Dict[str, Any]] = None,
    # Plane-selection prior sharpness (sigmoid beta) of the legacy mixture and the
    # 'exp' population family; the ramp family does not use it. If None, falls
    # back to instability_beta for backward compatibility.
    selection_beta: Optional[float] = 10.0,
    # Scale for the Gaussian slip likelihood (small-angle approximation).
    slip_misfit_sigma: float = 0.35,
    # Slip-direction likelihood family.
    slip_likelihood: str = "gaussian",
    # Concentration for the vMF slip likelihood; defaults to 1 / slip_misfit_sigma**2.
    slip_vmf_kappa: Optional[float] = None,
    # Back-compat name (historically used for instability-based plane selection).
    # Now only controls the sharpness of the slip-tendency plane-selection prior when
    # selection_beta is None.
    instability_beta: float = 20.0,
    signed_instability: bool = False,
    friction_fixed: Optional[float] = None,
    # Optional: encourage constant shear magnitudes across events (softly)
    enforce_constant_shear: bool = False,
    shear_sigma: float = 0.2,
    shear_target: Optional[float] = None,
    shear_target_sigma: float = 0.3,
    shear_center: str = "mean",
    shear_weight: float = 0.1,
    weighted_likelihood: bool = False,
    # "event": both nodal planes of an event share one weight from their mean
    # shear traction (recommended). "plane": per-plane weights (EXPERIMENTAL,
    # couples the weight to nodal-plane selection; see _tau_weights).
    likelihood_weight_mode: str = "event",
    # Exponent p of the per-event shear weights tau**p. p = 0 gives unit weights,
    # which the fault-population normalizer assumes.
    tau_weight_exponent: float = 0.0,
    normalize_tau_weights: bool = False,
    tau_weight_clip: Optional[Tuple[float, float]] = None,
    event_weights: Optional[np.ndarray] = None,
    event_weight_power: float = 1.0,
    q_prior_mu: Optional[np.ndarray] = None,
    q_prior_sigma: float = 1.0,
    R_prior_mu: Optional[float] = None,
    R_prior_sigma: float = 2.0,
    plane2_prior_probs: Optional[np.ndarray] = None,
    plane_prior_strength: float = 0.0,
    clustering_prior_strength: float = 0.0,
    strike_sigma_deg=0.0,
    dip_sigma_deg=0.0,
    rake_sigma_deg=0.0,
    fixed_plane_indices: Optional[np.ndarray] = None,
    # "default" is fault_population.DEFAULT_FAULT_POPULATION (ramp, inferred
    # I_min, one Bingham fabric component, overlap >= 0.2); None is the
    # historical unnormalized mixture.
    fault_population="default",
    initvals=None,
) -> Dict[str, Any]:
    """
    Joint Bayesian inference of stress orientation, shape ratio, and nodal plane selection using NUTS.

    Measurement uncertainty: strike_sigma_deg, dip_sigma_deg, rake_sigma_deg
    are independent one-sigma local errors in degrees for the FIRST supplied
    nodal plane. Each accepts a scalar or an (N,) array. All three default to
    zero (exact-angle model: the mechanisms enter as fixed data). Nonzero
    values add three latent mechanism parameters per event, sharing one
    latent mechanism between both nodal planes. The directional likelihood
    still describes residual model scatter; these errors additionally
    describe SDR measurement uncertainty. Posterior canonical plane-1 angles are stored in
    idata.posterior["mechanism_angles_deg"] when any error is nonzero.

    fixed_plane_indices optionally fixes all fault labels externally (0/1 for
    input plane 1/2). This bypasses instability and clustering for selection,
    without fixing the uncertain geometry. It cannot be combined with iterative
    preselection. Friction is not identifiable from the fixed-plane likelihood.

    fault_population selects the model used for nodal-plane selection.

    ``"default"`` selects ``fault_population.DEFAULT_FAULT_POPULATION``: the
    ramp density with inferred ``I_min`` and one Bingham fabric component
    (``fabric_K=1``), whose extra components (``fabric_K > 1``) must overlap
    the dominant one by at least 0.2. ``None`` keeps the historical behaviour:
    the mixture weights are ``sigmoid(beta * (I2 - I1))`` and the mixture is
    used without a normalizer. That term is not a probability density in the
    observed mechanism, and its total mass grows with mu, which biases the
    inferred friction upward.

    Any other value replaces the logistic weight by an explicit fault-population
    density g(I) on the sphere of fault normals and adds the matching normalizer
    ``-N log Z(R, mu)``, so that the per-event term is a density. Pass a family
    name or a dictionary, for example

        fault_population="exp"
            g = exp(beta I), the generative model whose plane probabilities are
            identical to the historical weights, now normalized.
        fault_population={"family": "ramp", "imin": "infer", "fabric_K": 0}
            g = softplus(k (I - I_min)) / (k (1 - I_min)) with I_min inferred,
            the purely stress-controlled model.
        fault_population={"family": "ramp", "imin": "infer", "fabric_K": 2}
            K Bingham clusters for an inherited fabric, with the stress-selected
            fraction w inferred and reported as ``w_population``; components
            beyond the dominant one are perturbations of it (``fabric_min_overlap``).

    The normalizer is tabulated once per configuration and cached on disk; the
    first call with a new configuration spends a few minutes building it. See
    fault_population.py for the full set of options, and basins.py for the
    basin diagnostics and per-basin evidence these models require, since they
    are multimodal on real catalogs.

    NUTS cannot sample discrete plane indicators z_i directly. This implementation uses a
    differentiable marginalization: each event likelihood is a 2-component mixture over the two
    nodal planes,

        p(data_i | stress) = p1_i * L1_i + p2_i * L2_i

    where L1_i and L2_i are slip-direction likelihoods for plane 1 and plane 2 respectively
    (Gaussian or von Mises-Fisher, optionally weighted by tau^2), and (p1_i, p2_i) is a
    soft instability-based prior derived from the Mohr-Coulomb criterion (depends on mu).

    The returned plane probabilities correspond to the posterior responsibilities of the mixture,
    averaged over posterior draws: E[p(z_i=1 | params, data_i)].

    HDI summaries at level `hdi_prob` (default 0.9) are included for R, μ, τ0 (when inferred),
    and each stress tensor component, replacing the previous n_bootstrap_samples argument.

    To run on CPU with the JAX/NumPyro backend, pass `nuts_sampler="numpyro"` and
    `nuts_sampler_kwargs={"chain_method": "vectorized"}` (or "parallel"/"sequential").
    When using NumPyro, PyMC ignores `cores`; keep it at 1 to avoid confusion.
    """
    strikes_1 = np.asarray(strikes_1, dtype=np.float64)
    dips_1 = np.asarray(dips_1, dtype=np.float64)
    rakes_1 = np.asarray(rakes_1, dtype=np.float64)
    strikes_2 = np.asarray(strikes_2, dtype=np.float64)
    dips_2 = np.asarray(dips_2, dtype=np.float64)
    rakes_2 = np.asarray(rakes_2, dtype=np.float64)

    N = len(strikes_1)
    if not (
        len(dips_1) == len(rakes_1) == len(strikes_2) == len(dips_2) == len(rakes_2) == N
    ):
        raise ValueError("All input arrays must have the same length")

    mechanism_angles, mechanism_errors = prepare_mechanism_errors(
        strikes_1, dips_1, rakes_1, strike_sigma_deg, dip_sigma_deg, rake_sigma_deg,
    )
    fixed_plane_indices = validate_fixed_planes(fixed_plane_indices, N, iterative_plane_selection)

    # Convert focal mechanisms to normal and slip unit vectors
    n1, s1 = normal_slip_vectors_batch(strikes_1, dips_1, rakes_1, direction="inward")
    n2, s2 = normal_slip_vectors_batch(strikes_2, dips_2, rakes_2, direction="inward")

    if event_weights is not None:
        event_weights = np.asarray(event_weights, dtype=np.float64).reshape(-1)
        if event_weights.size != N:
            raise ValueError(f"event_weights must have shape (N,), got {event_weights.shape} for N={N}")
        if not np.all(np.isfinite(event_weights)):
            raise ValueError("event_weights must be finite")
        if float(event_weight_power) != 1.0:
            event_weights = event_weights ** float(event_weight_power)

    q_prior_mu_arr = None
    if q_prior_mu is not None:
        q_prior_mu_arr = np.asarray(q_prior_mu, dtype=np.float64).reshape(-1)
        if q_prior_mu_arr.size != 4:
            raise ValueError("q_prior_mu must have 4 elements (quaternion components)")
        if not np.all(np.isfinite(q_prior_mu_arr)):
            raise ValueError("q_prior_mu must be finite")
        qn = float(np.linalg.norm(q_prior_mu_arr))
        if qn <= 0.0:
            raise ValueError("q_prior_mu norm must be > 0")
        q_prior_mu_arr = q_prior_mu_arr / qn
        if q_prior_mu_arr[0] < 0:
            q_prior_mu_arr = -q_prior_mu_arr

    q_prior_sigma = float(q_prior_sigma)
    if q_prior_sigma <= 0.0:
        raise ValueError("q_prior_sigma must be > 0")

    R_prior_mu_val = None
    if R_prior_mu is not None:
        R_prior_mu_val = float(R_prior_mu)
        if not (0.0 < R_prior_mu_val < 1.0):
            raise ValueError("R_prior_mu must be in (0, 1)")

    R_prior_sigma = float(R_prior_sigma)
    if R_prior_sigma <= 0.0:
        raise ValueError("R_prior_sigma must be > 0")

    plane2_prior_probs_arr = None
    if plane2_prior_probs is not None:
        plane2_prior_probs_arr = np.asarray(plane2_prior_probs, dtype=np.float64).reshape(-1)
        if plane2_prior_probs_arr.size != N:
            raise ValueError(
                f"plane2_prior_probs must have shape (N,), got {plane2_prior_probs_arr.shape} for N={N}"
            )
        if not np.all(np.isfinite(plane2_prior_probs_arr)):
            raise ValueError("plane2_prior_probs must be finite")
        plane2_prior_probs_arr = np.clip(plane2_prior_probs_arr, 1e-6, 1.0 - 1e-6)

    plane_prior_strength = float(plane_prior_strength)
    if plane_prior_strength < 0.0:
        raise ValueError("plane_prior_strength must be >= 0")

    clustering_prior_strength = float(clustering_prior_strength)
    if clustering_prior_strength < 0.0:
        raise ValueError("clustering_prior_strength must be >= 0")

    slip_likelihood_name, slip_misfit_sigma, slip_vmf_kappa_val = (
        _resolve_slip_likelihood_params(
            slip_likelihood,
            slip_misfit_sigma,
            slip_vmf_kappa,
        )
    )

    # --------------------------------------------------------
    # Optional iterative preselection (Michael + instability)
    # --------------------------------------------------------
    iterative_info: Dict[str, Any] = {}
    if iterative_plane_selection:
        if _det is None:
            raise RuntimeError("ilsi module not available; cannot perform iterative preselection")
        opts = {} if iterative_kwargs is None else dict(iterative_kwargs)
        # Default preselection options
        opts.setdefault("variable_shear", False)
        opts.setdefault("friction_coefficient", None)
        opts.setdefault("friction_min", friction_range[0])
        opts.setdefault("friction_max", friction_range[1])
        opts.setdefault("friction_step", 0.05)
        # Robustness defaults
        opts.setdefault("n_stress_iter", 10)
        opts.setdefault("n_random_selections", 20)
        opts.setdefault("n_averaging", 3)
        out = _det.inversion_one_set_instability(strikes_1, dips_1, rakes_1, **opts)
        mu_sel = float(out.get("friction_coefficient", 0.5 * (friction_range[0] + friction_range[1])))
        ps = out["principal_stresses"]
        pd = out["principal_directions"]
        R_sel = float(_det.utils_stress.R_(ps))
        Ivals, fp_strikes, fp_dips, fp_rakes = _det.compute_instability_parameter(
            pd,
            R_sel,
            mu_sel,
            strikes_1,
            dips_1,
            rakes_1,
            strikes_2,
            dips_2,
            rakes_2,
            return_fault_planes=True,
            signed_instability=bool(opts.get("signed_instability", False)),
        )
        n_sel, s_sel = normal_slip_vectors_batch(fp_strikes, fp_dips, fp_rakes, direction="inward")
        plane2_mask = Ivals[:, 1] > Ivals[:, 0]
        plane_map = plane2_mask.astype(int)
        plane_probs = np.column_stack([1 - plane2_mask, plane2_mask]).astype(float)
        iterative_info = {
            "mu": mu_sel,
            "R_from_preselection": R_sel,
            "fault_planes": (fp_strikes, fp_dips, fp_rakes),
            "plane_probabilities": plane_probs,
            "plane_map": plane_map,
            "n_selected": n_sel,
            "s_selected": s_sel,
        }

    model, mu_const = _build_joint_model(
        n1=n1, n2=n2, s1=s1, s2=s2,
        mechanism_angles=mechanism_angles, mechanism_errors=mechanism_errors,
        fixed_plane_indices=fixed_plane_indices,
        fault_population=fault_population,
        q_prior_mu_arr=q_prior_mu_arr, q_prior_sigma=q_prior_sigma,
        R_prior_mu_val=R_prior_mu_val, R_prior_sigma=R_prior_sigma,
        infer_friction=infer_friction,
        infer_friction_method=infer_friction_method,
        friction_fixed=friction_fixed,
        friction_prior_params=friction_prior_params,
        friction_range=friction_range,
        iterative_plane_selection=iterative_plane_selection,
        iterative_info=iterative_info,
        event_weights=event_weights,
        weighted_likelihood=weighted_likelihood,
        likelihood_weight_mode=likelihood_weight_mode,
        tau_weight_exponent=tau_weight_exponent,
        normalize_tau_weights=normalize_tau_weights,
        tau_weight_clip=tau_weight_clip,
        slip_likelihood_name=slip_likelihood_name,
        slip_misfit_sigma=slip_misfit_sigma,
        slip_vmf_kappa_val=slip_vmf_kappa_val,
        instability_beta=instability_beta,
        selection_beta=selection_beta,
        signed_instability=signed_instability,
        clustering_prior_strength=clustering_prior_strength,
        plane2_prior_probs_arr=plane2_prior_probs_arr,
        plane_prior_strength=plane_prior_strength,
        enforce_constant_shear=enforce_constant_shear,
        shear_weight=shear_weight,
        shear_sigma=shear_sigma,
        shear_center=shear_center,
        shear_target=shear_target,
        shear_target_sigma=shear_target_sigma,
    )

    sampler_kwargs: Dict[str, Any] = {
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "cores": cores,
        "target_accept": target_accept,
        "random_seed": random_seed,
        "progressbar": progressbar,
        "return_inferencedata": True,
    }

    # Optional: choose an explicit NUTS backend (e.g., nutpie)
    nk = {} if nuts_sampler_kwargs is None else dict(nuts_sampler_kwargs)
    sampler_name = (nuts_sampler or "").lower()
    if initvals is not None:
        if sampler_name not in {"", "pymc"}:
            raise ValueError("Explicit initvals require the PyMC NUTS backend (nuts_sampler='pymc')")
        sampler_kwargs["initvals"] = initvals
        sampler_kwargs["init"] = "adapt_diag"
    smc_info = None
    with model:
        if sampler_name in {"smc", "blackjax_smc"}:
            # Adaptive tempered SMC on the same model graph; ``draws`` is the
            # number of particles and ``chains`` the number of independent runs.
            try:
                from . import bjsi_smc as _smc
            except ImportError:
                import bjsi_smc as _smc
            idata, smc_info = _smc.sample_smc(
                model, draws=draws, chains=chains, cores=cores, random_seed=random_seed,
                progressbar=progressbar, **nk,
            )
        elif sampler_name in {"nutpie", "nuts-nutpie"}:
            idata = pm.sample(nuts_sampler="nutpie", nuts_sampler_kwargs=nk, **sampler_kwargs)
        elif sampler_name in {"numpyro", "jax"}:
            # NumPyro backend runs on CPU here; chain_method can be "vectorized"/"parallel"/"sequential"
            sampler_kwargs["cores"] = 1
            idata = pm.sample(nuts_sampler="numpyro", nuts_sampler_kwargs=nk, **sampler_kwargs)
        else:
            # Default PyMC NUTS
            idata = pm.sample(**sampler_kwargs)

    # --- Post-processing (mirror SMC outputs) ---
    results: Dict[str, Any] = summarize_posterior(idata, hdi_prob=hdi_prob)
    R_median = results["R_median"]
    pd_median = results["principal_directions"]
    mu_samples = results["mu_samples"]
    tau0_samples = results["tau0_samples"]

    results["mechanism_sigma_deg"] = mechanism_errors.copy()
    results["fixed_plane_indices"] = None if fixed_plane_indices is None else fixed_plane_indices.copy()
    results["fault_population"] = _population_results(idata, fault_population)

    if return_plane_probabilities:
        if iterative_plane_selection:
            results["plane_probabilities"] = iterative_info.get("plane_probabilities")
            results["plane_selection_map"] = iterative_info.get("plane_map")
        elif "plane_probabilities" not in results:
            raise RuntimeError("Internal error: p_plane2_post missing from posterior")

    if iterative_plane_selection:
        results["friction_coefficient"] = None
    elif infer_friction and str(infer_friction_method or "sample").lower() == "posthoc":
        mu_min, mu_max = float(friction_range[0]), float(friction_range[1])
        if _det is None:
            mu_hat = None
        else:
            mu_grid = np.linspace(mu_min, mu_max, 76)
            scores = np.empty(mu_grid.size, dtype=float)
            for i, mu_try in enumerate(mu_grid):
                I_try = _det.compute_instability_parameter(
                    pd_median,
                    float(R_median),
                    float(mu_try),
                    strikes_1,
                    dips_1,
                    rakes_1,
                    strikes_2,
                    dips_2,
                    rakes_2,
                    return_fault_planes=False,
                    signed_instability=bool(signed_instability),
                )
                scores[i] = float(np.mean(np.max(I_try, axis=1)))
            mu_hat = float(mu_grid[int(np.argmax(scores))])

        results["friction_coefficient"] = mu_hat
        results["mu"] = mu_hat
        if mu_hat is not None:
            results["friction_median"] = float(mu_hat)
            results["friction_mean"] = float(mu_hat)
            results["friction_std"] = 0.0
            results["friction_CI95"] = (float(mu_hat), float(mu_hat))
            results["hdi"]["mu"] = (float(mu_hat), float(mu_hat))
            if "hdi_90" in results:
                results["hdi_90"]["mu"] = (float(mu_hat), float(mu_hat))
    elif mu_samples is None or mu_samples.size == 0:
        mu_fixed = (
            float(mu_const)
            if mu_const is not None
            else 0.5 * (float(friction_range[0]) + float(friction_range[1]))
        )
        results["friction_coefficient"] = mu_fixed
        results["mu"] = mu_fixed
        results["friction_median"] = mu_fixed
        results["friction_mean"] = mu_fixed
        results["friction_std"] = 0.0
        results["friction_CI95"] = (mu_fixed, mu_fixed)

    results["convergence"] = {
        "n_samples": draws * chains,
        "n_chains": chains,
        "sampler": "NUTS",
        "nuts_sampler": nuts_sampler or "pymc",
        "target_accept": target_accept,
        "tune": tune,
    }
    if smc_info is not None:
        results["convergence"].update({
            "sampler": "SMC", "nuts_sampler": "blackjax_smc", "target_accept": None, "tune": None,
            "kernel": smc_info["kernel"], "log_evidence": smc_info["log_evidence"],
            "n_iterations": smc_info["n_iterations"],
        })
        results["smc"] = smc_info
        if results.get("fault_population") is not None:
            results["fault_population"]["log_evidence"] = smc_info["log_evidence"]

    if iterative_plane_selection:
        mu_sel = iterative_info.get("mu")
        if mu_sel is not None:
            results["friction_coefficient"] = float(mu_sel)
            results["friction_median"] = float(mu_sel)
            results["friction_mean"] = float(mu_sel)
            results["friction_std"] = 0.0
            results["friction_CI95"] = (float(mu_sel), float(mu_sel))
        results["friction_coefficient_preselected"] = mu_sel
        results["preselected_planes"] = iterative_info.get("fault_planes")
        if "R_from_preselection" in iterative_info:
            results["R_preselected"] = iterative_info["R_from_preselection"]
        results["preselection"] = "Michael+instability"

    return results
