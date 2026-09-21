"""
BlackJAX adaptive-tempered SMC port of `Bayesian_joint_plane_selection_SMC`.

This module mirrors the public surface of `bjsi.Bayesian_joint_plane_selection_SMC`
but replaces the PyMC SMC loop with BlackJAX's `adaptive_tempered_smc` driving
an HMC mutation kernel. Use this when the PyMC SMC IMH/MH mutation produces
spiky / multimodal marginals due to particle degeneracy — the gradient-based
HMC mutation mixes the joint (quaternion, R) target much better, yielding
smoother marginals at comparable cost.

When to prefer this over `bjsi.Bayesian_joint_plane_selection_SMC`:

* You see spiky / staircase marginals on R or the quaternion components in
  the PyMC SMC output, or particle ESS collapses to a small number of
  unique values after resampling.
* N (number of focal mechanisms) > 30 — the joint posterior dimension grows
  and IMH proposals become inefficient.
* You need accurate posterior credible intervals on R or the principal
  directions, not just point estimates.

Scope: only the `iterative_plane_selection=False` branch is ported. The
`iterative_plane_selection=True` branch raises `NotImplementedError`; use
`bjsi.Bayesian_joint_plane_selection_SMC` for that path.
"""

from __future__ import annotations

import math
import sys as _sys
import time as _time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import arviz as az


# ---------------------------------------------------------------------------
# Imports for stress_tensor_eigendecomposition (mirror bjsi.py:36-48)
# ---------------------------------------------------------------------------
try:
    from .utils_stress import stress_tensor_eigendecomposition
except ImportError:
    try:
        from utils_stress import stress_tensor_eigendecomposition
    except ImportError:
        def stress_tensor_eigendecomposition(sigma):
            eigenvalues, eigenvectors = np.linalg.eigh(sigma)
            idx = np.argsort(eigenvalues)
            return eigenvalues[idx], eigenvectors[:, idx]

try:
    from . import ilsi as _det
except Exception:
    try:
        import ilsi as _det  # type: ignore
    except Exception:
        _det = None


# ---------------------------------------------------------------------------
# JAX / BlackJAX import guard
# ---------------------------------------------------------------------------
def _import_jax_blackjax():
    try:
        import jax
        import jax.numpy as jnp
        import blackjax
    except ImportError as exc:
        raise ImportError(
            "bjsi_blackjax requires `jax` and `blackjax` to be installed. "
            "Install with: pip install blackjax jax"
        ) from exc
    return jax, jnp, blackjax


# ---------------------------------------------------------------------------
# numpy helpers (no JAX dependency)
# ---------------------------------------------------------------------------
def normal_slip_vectors_batch(strike, dip, rake, direction="inward"):
    strike = np.asarray(strike, dtype=np.float64)
    dip = np.asarray(dip, dtype=np.float64)
    rake = np.asarray(rake, dtype=np.float64)

    d2r = np.pi / 180.0
    s_rad = strike * d2r
    d_rad = dip * d2r
    r_rad = rake * d2r

    n = np.stack([
        -np.sin(d_rad) * np.sin(s_rad),
        -np.sin(d_rad) * np.cos(s_rad),
        np.cos(d_rad),
    ], axis=1)
    if direction == "outward":
        n *= -1.0
    elif direction != "inward":
        raise ValueError('direction must be "inward" or "outward"')

    slip = np.stack([
        np.cos(r_rad) * np.cos(s_rad) + np.sin(r_rad) * np.cos(d_rad) * np.sin(s_rad),
        -np.cos(r_rad) * np.sin(s_rad) + np.sin(r_rad) * np.cos(d_rad) * np.cos(s_rad),
        np.sin(r_rad) * np.sin(d_rad),
    ], axis=1)
    return n, slip


def _canonicalize_slip_likelihood(slip_likelihood: Optional[str]) -> str:
    name = str(slip_likelihood or "gaussian").strip().lower().replace("-", "_")
    aliases = {
        "gaussian": "gaussian",
        "normal": "gaussian",
        "vmf": "von_mises_fisher",
        "vonmisesfisher": "von_mises_fisher",
        "von_misesfisher": "von_mises_fisher",
        "von_mises_fisher": "von_mises_fisher",
    }
    if name not in aliases:
        raise ValueError(
            "slip_likelihood must be one of {'gaussian', 'von_mises_fisher', 'vmf'}"
        )
    return aliases[name]


def _resolve_slip_likelihood_params(
    slip_likelihood: Optional[str],
    slip_misfit_sigma: float,
    slip_vmf_kappa: Optional[float],
) -> Tuple[str, float, Optional[float]]:
    name = _canonicalize_slip_likelihood(slip_likelihood)
    sigma = float(slip_misfit_sigma)
    if not np.isfinite(sigma):
        raise ValueError("slip_misfit_sigma must be finite")
    kappa_val: Optional[float] = None
    if slip_vmf_kappa is not None:
        kappa_val = float(slip_vmf_kappa)
        if not np.isfinite(kappa_val) or kappa_val <= 0.0:
            raise ValueError("slip_vmf_kappa must be finite and > 0")
    if name == "gaussian" or kappa_val is None:
        if sigma <= 0.0:
            raise ValueError("slip_misfit_sigma must be > 0")
    if name == "von_mises_fisher":
        if kappa_val is None:
            kappa_val = 1.0 / (sigma**2)
        if not np.isfinite(kappa_val) or kappa_val <= 0.0:
            raise ValueError("Resolved vMF concentration must be finite and > 0")
    return name, sigma, kappa_val


# ---------------------------------------------------------------------------
# JAX helpers (built lazily so import works without jax)
# ---------------------------------------------------------------------------
def _build_jax_helpers(jnp):
    """Return a namespace of jit-friendly helpers parameterized by jnp."""

    def quat_to_rotation_matrix(q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        R11 = 1 - 2 * (y * y + z * z)
        R22 = 1 - 2 * (x * x + z * z)
        R33 = 1 - 2 * (x * x + y * y)
        R12 = 2 * (x * y - z * w); R21 = 2 * (x * y + z * w)
        R13 = 2 * (x * z + y * w); R31 = 2 * (x * z - y * w)
        R23 = 2 * (y * z - x * w); R32 = 2 * (y * z + x * w)
        return jnp.stack([
            jnp.stack([R11, R12, R13]),
            jnp.stack([R21, R22, R23]),
            jnp.stack([R31, R32, R33]),
        ])

    def stress_tensor_from_R_and_shape(Rmat, Rratio):
        sig1 = -1.0
        sig2 = 2.0 * Rratio - 1.0
        sig3 = 1.0
        diag_vals = jnp.stack([sig1, sig2, sig3])
        RD = Rmat * diag_vals  # column-scale trick (broadcasts over rows)
        return RD @ Rmat.T

    def shear_traction_and_magnitude(Sigma, n):
        # n: (N,3); returns (t, ts, tau, sigma_n)
        t = (Sigma @ n.T).T
        sigma_n = jnp.sum(t * n, axis=-1)
        tn = sigma_n[:, None] * n
        ts = t - tn
        tau = jnp.sqrt(jnp.sum(ts**2, axis=-1) + 1e-12)
        return t, ts, tau, sigma_n

    def shear_traction_direction(ts, tau):
        return ts / (tau[:, None] + 1e-12)

    def slip_direction_logp(s_obs, s_pred, family, sigma, vmf_kappa, weight):
        safe_w = jnp.clip(weight, 1e-12, 1e12)
        obs_norm = jnp.sqrt(jnp.sum(s_obs**2, axis=-1, keepdims=True))
        pred_norm = jnp.sqrt(jnp.sum(s_pred**2, axis=-1, keepdims=True))
        s_obs_u = s_obs / (obs_norm + 1e-12)
        s_pred_u = s_pred / (pred_norm + 1e-12)
        if family == "gaussian":
            diff = s_obs_u - s_pred_u
            sigma2 = float(sigma) ** 2
            return (
                -0.5 * safe_w * jnp.sum(diff**2, axis=-1) / sigma2
                - 1.5 * math.log(2 * math.pi * sigma2)
            )
        if family == "von_mises_fisher":
            kappa_base = float(vmf_kappa)
            kappa_eff = safe_w * kappa_base
            dot = jnp.clip(jnp.sum(s_obs_u * s_pred_u, axis=-1), -1.0, 1.0)
            log_sinh_kappa = (
                kappa_base + math.log(-math.expm1(-2.0 * kappa_base)) - math.log(2.0)
            )
            log_c3 = math.log(kappa_base) - math.log(4.0 * math.pi) - log_sinh_kappa
            return log_c3 + kappa_eff * dot
        raise ValueError(f"Unsupported slip likelihood family: {family}")

    def tau_weights(tau1, tau2, weighted, mode, exponent, normalize, clip):
        if not weighted:
            return jnp.ones_like(tau1), jnp.ones_like(tau2)
        p = float(exponent)
        if normalize:
            tau_scale = jnp.mean(0.5 * (tau1 + tau2)) + 1e-12
        else:
            tau_scale = 1.0
        m = str(mode or "plane").lower()
        if m in {"plane", "plane_specific"}:
            w1 = (tau1 / tau_scale) ** p
            w2 = (tau2 / tau_scale) ** p
        elif m in {"event", "common", "symmetric"}:
            tau_e = 0.5 * (tau1 + tau2)
            w1 = w2 = (tau_e / tau_scale) ** p
        elif m in {"none", "off"}:
            w1 = jnp.ones_like(tau1)
            w2 = jnp.ones_like(tau2)
        else:
            raise ValueError("likelihood_weight_mode must be plane | event | none")
        if clip is not None:
            lo, hi = float(clip[0]), float(clip[1])
            w1 = jnp.clip(w1, lo, hi)
            w2 = jnp.clip(w2, lo, hi)
        return w1, w2

    def instability_log(Sigma, n, mu, ts, tau, sigma_n, s_pred=None, s_obs=None,
                        signed=False):
        sig1 = -1.0
        denom = jnp.sqrt(1.0 + mu * mu)
        tau_c = 1.0 / denom
        sig_c = mu / denom
        numerator = tau - mu * (sig1 - sigma_n)
        denominator_I = tau_c - mu * (sig1 - sig_c)
        I_val = numerator / denominator_I
        if signed and (s_pred is not None) and (s_obs is not None):
            dot_p = jnp.sum(s_pred * s_obs, axis=-1)
            I_val = I_val * jnp.sign(dot_p)
        return I_val

    return {
        "quat_to_rotation_matrix": quat_to_rotation_matrix,
        "stress_tensor_from_R_and_shape": stress_tensor_from_R_and_shape,
        "shear_traction_and_magnitude": shear_traction_and_magnitude,
        "shear_traction_direction": shear_traction_direction,
        "slip_direction_logp": slip_direction_logp,
        "tau_weights": tau_weights,
        "instability_log": instability_log,
    }


# ---------------------------------------------------------------------------
# Parameter-space layout
# ---------------------------------------------------------------------------
class _ThetaLayout:
    """Tracks which slots in the flat unconstrained vector hold which parameter."""

    def __init__(self, sample_mu: bool, sample_tau0: bool):
        self.sample_mu = sample_mu
        self.sample_tau0 = sample_tau0
        # q_raw [0..3], R_logit [4], optional mu_logit, optional tau0
        offset = 5
        self.mu_idx = None
        self.tau0_idx = None
        if sample_mu:
            self.mu_idx = offset
            offset += 1
        if sample_tau0:
            self.tau0_idx = offset
            offset += 1
        self.dim = offset


def _softplus(jnp, x):
    # Numerically stable log(1+exp(x))
    return jnp.logaddexp(x, 0.0)


# ---------------------------------------------------------------------------
# Build separate log_prior and log_likelihood functions for adaptive_tempered_smc
# ---------------------------------------------------------------------------
def _build_logp_fns(
    *,
    jnp,
    helpers,
    n1, n2, s1, s2,
    layout: _ThetaLayout,
    q_prior_mu_arr,
    q_prior_sigma: float,
    R_prior_mu_val: Optional[float],
    R_prior_sigma: float,
    mu_const: float,
    friction_prior_params: Tuple[float, float],
    friction_range: Tuple[float, float],
    event_weights_arr,
    weighted_likelihood: bool,
    likelihood_weight_mode: str,
    tau_weight_exponent: float,
    normalize_tau_weights: bool,
    tau_weight_clip,
    slip_likelihood_name: str,
    slip_misfit_sigma: float,
    slip_vmf_kappa_val: Optional[float],
    instability_beta: float,
    selection_beta: Optional[float],
    signed_instability: bool,
    clustering_prior_strength: float,
    plane2_prior_probs_arr,
    plane_prior_strength: float,
    enforce_constant_shear: bool,
    shear_weight: float,
    shear_sigma: float,
    shear_center: str,
    shear_target: Optional[float],
    shear_target_sigma: float,
):
    n1_j = jnp.asarray(n1)
    n2_j = jnp.asarray(n2)
    s1_j = jnp.asarray(s1)
    s2_j = jnp.asarray(s2)
    q_prior_mu_j = (
        jnp.asarray(q_prior_mu_arr) if q_prior_mu_arr is not None else jnp.zeros(4)
    )
    q_prior_sigma_f = float(q_prior_sigma if q_prior_mu_arr is not None else 1.0)
    plane2_prior_logit = None
    if plane2_prior_probs_arr is not None and plane_prior_strength > 0.0:
        eps = 1e-9
        clipped = np.clip(plane2_prior_probs_arr, eps, 1.0 - eps)
        plane2_prior_logit = jnp.asarray(np.log(clipped) - np.log1p(-clipped))

    event_w_j = (
        jnp.asarray(event_weights_arr) if event_weights_arr is not None else 1.0
    )

    beta_val = float(instability_beta if selection_beta is None else selection_beta)

    fr_lo = float(friction_range[0])
    fr_hi = float(friction_range[1])
    a_mu = float(friction_prior_params[0])
    b_mu = float(friction_prior_params[1])
    log_const_beta_mu = (
        math.lgamma(a_mu + b_mu) - math.lgamma(a_mu) - math.lgamma(b_mu)
    )

    R_use_truncnorm = R_prior_mu_val is not None
    if R_use_truncnorm:
        R_mu_v = float(R_prior_mu_val)
        R_sig_v = float(R_prior_sigma)
    # Beta(2,2) constants: log(6) = log Γ(4) - 2 log Γ(2) = log 6
    log_const_beta_R = math.log(6.0)

    use_clustering = float(clustering_prior_strength) > 0.0
    cluster_strength = float(clustering_prior_strength)
    plane_prior_strength_f = float(plane_prior_strength)

    qf = helpers
    sample_mu = layout.sample_mu
    sample_tau0 = layout.sample_tau0
    mu_idx = layout.mu_idx
    tau0_idx = layout.tau0_idx
    mu_const_f = float(mu_const)

    def to_constrained(theta):
        q_raw = theta[0:4]
        q_norm = jnp.sqrt(jnp.sum(q_raw**2)) + 1e-8
        q_unit = q_raw / q_norm
        R_logit = theta[4]
        R = jax_sigmoid(jnp, R_logit)
        if sample_mu:
            mu_logit = theta[mu_idx]
            mu_raw = jax_sigmoid(jnp, mu_logit)
            mu = fr_lo + (fr_hi - fr_lo) * mu_raw
        else:
            mu = jnp.asarray(mu_const_f)
        tau0 = jnp.asarray(0.0)
        if sample_tau0:
            tau0 = theta[tau0_idx]
        return q_unit, R, mu, tau0

    def log_prior_fn(theta):
        q_raw = theta[0:4]
        # Normal(q_prior_mu, q_prior_sigma) on each component
        diff_q = q_raw - q_prior_mu_j
        lp_q = -0.5 * jnp.sum((diff_q / q_prior_sigma_f) ** 2) - 4 * 0.5 * math.log(
            2 * math.pi * q_prior_sigma_f**2
        )

        R_logit = theta[4]
        # log p(R_logit) = log p(R) + log |dR/dR_logit|
        # log Beta(R; α, β) + log(R(1-R))
        # Using softplus identities: log R = -softplus(-R_logit), log(1-R) = -softplus(R_logit)
        log_R = -_softplus(jnp, -R_logit)
        log_1mR = -_softplus(jnp, R_logit)
        if R_use_truncnorm:
            R = jax_sigmoid(jnp, R_logit)
            # TruncatedNormal(R_mu, R_sig) on (0.001, 0.999) — drop normalization
            # constant for the truncation (irrelevant up to a constant for SMC).
            lp_R_constrained = -0.5 * ((R - R_mu_v) / R_sig_v) ** 2 - 0.5 * math.log(
                2 * math.pi * R_sig_v**2
            )
            lp_R = lp_R_constrained + log_R + log_1mR
        else:
            # Beta(2,2): log(6) + log R + log(1-R), plus Jacobian log R + log(1-R)
            lp_R = log_const_beta_R + 2.0 * (log_R + log_1mR)

        lp = lp_q + lp_R

        if sample_mu:
            mu_logit = theta[mu_idx]
            log_mu = -_softplus(jnp, -mu_logit)
            log_1mmu = -_softplus(jnp, mu_logit)
            # Beta(α, β) on raw, plus Jacobian
            lp_mu = (
                log_const_beta_mu
                + (a_mu - 1.0) * log_mu
                + (b_mu - 1.0) * log_1mmu
                + log_mu
                + log_1mmu
            )
            lp = lp + lp_mu

        if sample_tau0:
            tau0 = theta[tau0_idx]
            lp_tau0 = -0.5 * ((tau0 - 0.5) / float(shear_target_sigma)) ** 2 - 0.5 * math.log(
                2 * math.pi * float(shear_target_sigma) ** 2
            )
            lp = lp + lp_tau0

        return lp

    def log_likelihood_fn(theta):
        q_unit, R, mu, tau0 = to_constrained(theta)
        Rmat = qf["quat_to_rotation_matrix"](q_unit)
        Sigma = qf["stress_tensor_from_R_and_shape"](Rmat, R)

        _, ts1, tau1, sn1 = qf["shear_traction_and_magnitude"](Sigma, n1_j)
        _, ts2, tau2, sn2 = qf["shear_traction_and_magnitude"](Sigma, n2_j)
        s_pred1 = qf["shear_traction_direction"](ts1, tau1)
        s_pred2 = qf["shear_traction_direction"](ts2, tau2)

        inst1 = qf["instability_log"](
            Sigma, n1_j, mu, ts1, tau1, sn1, s_pred=s_pred1, s_obs=s1_j,
            signed=signed_instability,
        )
        inst2 = qf["instability_log"](
            Sigma, n2_j, mu, ts2, tau2, sn2, s_pred=s_pred2, s_obs=s2_j,
            signed=signed_instability,
        )

        w1_tau, w2_tau = qf["tau_weights"](
            tau1, tau2, weighted_likelihood, likelihood_weight_mode,
            tau_weight_exponent, normalize_tau_weights, tau_weight_clip,
        )
        w1 = event_w_j * w1_tau
        w2 = event_w_j * w2_tau

        ll1 = qf["slip_direction_logp"](
            s1_j, s_pred1, slip_likelihood_name, slip_misfit_sigma,
            slip_vmf_kappa_val, w1,
        )
        ll2 = qf["slip_direction_logp"](
            s2_j, s_pred2, slip_likelihood_name, slip_misfit_sigma,
            slip_vmf_kappa_val, w2,
        )

        inst_delta = inst2 - inst1
        logit_local = beta_val * inst_delta

        if use_clustering:
            p_tent = jax_sigmoid(jnp, logit_local)
            certainty = (2.0 * p_tent - 1.0) ** 2
            M1 = n1_j[:, :, None] * n1_j[:, None, :]
            M2 = n2_j[:, :, None] * n2_j[:, None, :]
            sum_c = jnp.sum(certainty) + 1e-12
            T_conf = jnp.sum(
                certainty[:, None, None] * (
                    (1.0 - p_tent)[:, None, None] * M1
                    + p_tent[:, None, None] * M2
                ),
                axis=0,
            ) / sum_c
            E1 = jnp.sum(n1_j * (T_conf @ n1_j.T).T, axis=-1)
            E2 = jnp.sum(n2_j * (T_conf @ n2_j.T).T, axis=-1)
            logit_local = logit_local + cluster_strength * (E2 - E1)

        if plane2_prior_logit is not None:
            logit_full = logit_local + plane_prior_strength_f * plane2_prior_logit
        else:
            logit_full = logit_local

        eps = 1e-9
        # numerically stable log(p2) and log(1-p2)
        log_p2 = -_softplus(jnp, -logit_full)
        log_1mp2 = -_softplus(jnp, logit_full)
        # clip via logit truncation: log(1-eps) ≈ -eps
        log_p2 = jnp.clip(log_p2, math.log(eps), math.log(1.0 - eps))
        log_1mp2 = jnp.clip(log_1mp2, math.log(eps), math.log(1.0 - eps))

        logw1 = log_1mp2 + ll1
        logw2 = log_p2 + ll2
        logmix = jnp.logaddexp(logw1, logw2)
        ll = jnp.sum(logmix)

        if enforce_constant_shear and float(shear_weight) > 0.0:
            sw = float(shear_weight)
            ss = float(shear_sigma)
            sc = (shear_center or "mean").lower()
            # posterior responsibility (for tau_mag aggregate, mirroring bjsi.py:489)
            p2_post = jnp.exp(logw2 - logmix)
            tau_mag = (1.0 - p2_post) * tau1 + p2_post * tau2
            if sc == "mean":
                tau0_use = jnp.mean(tau_mag)
            elif sc == "fixed":
                tau0_use = float(shear_target if shear_target is not None else 0.5)
            else:  # learned
                tau0_use = tau0
            resid = (tau_mag - tau0_use) / ss
            ll = ll - sw * 0.5 * jnp.sum(resid * resid)

        return ll

    return log_prior_fn, log_likelihood_fn, to_constrained


def jax_sigmoid(jnp, x):
    return 1.0 / (1.0 + jnp.exp(-x))


# ---------------------------------------------------------------------------
# Initialize particles from the prior
# ---------------------------------------------------------------------------
def _sample_initial_particles(jax, jnp, key, n_particles, layout, *,
                              q_prior_mu_arr, q_prior_sigma,
                              R_prior_mu_val, R_prior_sigma,
                              friction_prior_params, friction_range,
                              shear_target_sigma):
    """Draw `n_particles` from the prior in the unconstrained parameterization."""
    particles = jnp.zeros((n_particles, layout.dim))

    # q_raw ~ Normal(mu, sigma)
    key, sub = jax.random.split(key)
    q_mu = (
        jnp.asarray(q_prior_mu_arr) if q_prior_mu_arr is not None else jnp.zeros(4)
    )
    q_sig = float(q_prior_sigma)
    q_samples = q_mu[None, :] + q_sig * jax.random.normal(sub, (n_particles, 4))
    particles = particles.at[:, 0:4].set(q_samples)

    # R_logit chosen so that sigmoid(R_logit) ~ Beta(2,2) (or TruncNormal)
    key, sub = jax.random.split(key)
    if R_prior_mu_val is None:
        # Beta(2,2): sample R, transform to logit
        R_samp = jax.random.beta(sub, 2.0, 2.0, (n_particles,))
    else:
        # TruncNormal on (0.001, 0.999): sample via inverse-cdf rejection: just clip
        R_samp = jnp.clip(
            float(R_prior_mu_val) + float(R_prior_sigma) * jax.random.normal(sub, (n_particles,)),
            1e-3, 1.0 - 1e-3,
        )
    R_logit = jnp.log(R_samp / (1.0 - R_samp))
    particles = particles.at[:, 4].set(R_logit)

    if layout.sample_mu:
        key, sub = jax.random.split(key)
        a_mu = float(friction_prior_params[0])
        b_mu = float(friction_prior_params[1])
        mu_raw = jax.random.beta(sub, a_mu, b_mu, (n_particles,))
        mu_logit = jnp.log(mu_raw / (1.0 - mu_raw))
        particles = particles.at[:, layout.mu_idx].set(mu_logit)

    if layout.sample_tau0:
        key, sub = jax.random.split(key)
        tau0_samp = 0.5 + float(shear_target_sigma) * jax.random.normal(
            sub, (n_particles,)
        )
        particles = particles.at[:, layout.tau0_idx].set(tau0_samp)

    return particles, key


# ---------------------------------------------------------------------------
# SMC driver
# ---------------------------------------------------------------------------
def _build_smc_algorithm(jax, jnp, blackjax, *,
                         log_prior_fn, log_likelihood_fn,
                         n_particles: int, dim: int,
                         target_ess: float, num_mcmc_steps: int,
                         hmc_step_size: float, hmc_num_integration_steps: int,
                         cov_preconditioner: bool, waste_free: bool):
    """Construct the SMC SamplingAlgorithm.

    When `cov_preconditioner=True`, wraps adaptive_tempered_smc in
    `inner_kernel_tuning` so the inverse mass matrix is recomputed from the
    particle covariance after every SMC step (geometry-aware HMC mass).
    When `waste_free=True`, switches the update strategy to waste-free SMC,
    which keeps every intermediate MCMC state instead of just the last one
    (Dau & Chopin 2022) — turns N particles × p mutation steps into N×p
    samples without extra HMC cost.

    Returns (algorithm, wrapped: bool). When wrapped, state has shape
    StateWithParameterOverride(sampler_state=TemperedSMCState, parameter_override=dict);
    otherwise it's a plain TemperedSMCState. Use _extract_inner_state(state)
    to get the inner SMC state uniformly.
    """
    inv_mass_init = jnp.ones((1, dim))
    step_size_init = jnp.asarray([float(hmc_step_size)])
    n_int_init = jnp.asarray([int(hmc_num_integration_steps)])
    initial_params = dict(
        step_size=step_size_init,
        inverse_mass_matrix=inv_mass_init,
        num_integration_steps=n_int_init,
    )

    extra_kwargs: Dict[str, Any] = dict(target_ess=float(target_ess))

    if waste_free:
        if int(n_particles) % int(num_mcmc_steps) != 0:
            raise ValueError(
                f"waste_free SMC requires draws ({n_particles}) to be divisible "
                f"by num_mcmc_steps ({num_mcmc_steps}); pick num_mcmc_steps that "
                f"divides draws."
            )
        from blackjax.smc import waste_free as _wf
        extra_kwargs["update_strategy"] = _wf.waste_free_smc(
            n_particles=int(n_particles), p=int(num_mcmc_steps)
        )
        # waste_free.update_waste_free raises if num_mcmc_steps is not None,
        # so we must NOT forward our num_mcmc_steps to the SMC algo.
        smc_num_mcmc_steps = None
    else:
        smc_num_mcmc_steps = int(num_mcmc_steps)

    if cov_preconditioner:
        from blackjax.smc import inner_kernel_tuning as _ikt
        from blackjax.smc.tuning.from_particles import (
            particles_covariance_matrix as _pcm,
        )

        # Closures over step_size_init / n_int_init keep step size and L
        # fixed; only the inverse mass matrix is re-derived from particles.
        # This intentionally avoids dual-averaging the step size, which would
        # need access to the previous override (not exposed by the hook).
        def mcmc_parameter_update_fn(key, state, info):
            cov = _pcm(state.particles)
            diag_var = jnp.diag(cov)
            diag_var = jnp.maximum(diag_var, 1e-8)
            inv_mass_new = diag_var[None, :]
            return dict(
                step_size=step_size_init,
                inverse_mass_matrix=inv_mass_new,
                num_integration_steps=n_int_init,
            )

        algorithm = _ikt.as_top_level_api(
            blackjax.adaptive_tempered_smc,
            logprior_fn=log_prior_fn,
            loglikelihood_fn=log_likelihood_fn,
            mcmc_step_fn=blackjax.hmc.build_kernel(),
            mcmc_init_fn=blackjax.hmc.init,
            resampling_fn=blackjax.smc.resampling.systematic,
            mcmc_parameter_update_fn=mcmc_parameter_update_fn,
            initial_parameter_value=initial_params,
            num_mcmc_steps=smc_num_mcmc_steps,
            **extra_kwargs,
        )
        wrapped = True
    else:
        algorithm = blackjax.adaptive_tempered_smc(
            logprior_fn=log_prior_fn,
            loglikelihood_fn=log_likelihood_fn,
            mcmc_step_fn=blackjax.hmc.build_kernel(),
            mcmc_init_fn=blackjax.hmc.init,
            mcmc_parameters=initial_params,
            resampling_fn=blackjax.smc.resampling.systematic,
            num_mcmc_steps=smc_num_mcmc_steps,
            **extra_kwargs,
        )
        wrapped = False

    return algorithm, wrapped


def _extract_inner_state(state):
    """Return the inner TemperedSMCState whether or not inner_kernel_tuning wraps it."""
    return state.sampler_state if hasattr(state, "sampler_state") else state


def _run_adaptive_smc(jax, jnp, blackjax, *,
                      log_prior_fn, log_likelihood_fn, init_particles, key,
                      target_ess: float, num_mcmc_steps: int,
                      hmc_step_size: float, hmc_num_integration_steps: int,
                      cov_preconditioner: bool = True,
                      waste_free: bool = True,
                      adapt_hmc_step_size: bool = True,
                      hmc_target_accept: float = 0.75,
                      hmc_step_adaptation_rate: float = 0.5,
                      hmc_step_min: float = 1e-3,
                      hmc_step_max: float = 0.5,
                      max_iter: int = 200,
                      progressbar: bool = False, chain_id: int = 0,
                      n_chains: int = 1, log_every: int = 1):
    """Run BlackJAX adaptive_tempered_smc with HMC mutation. Returns (final_particles, info_history)."""
    n_particles, dim = init_particles.shape

    smc, _wrapped = _build_smc_algorithm(
        jax, jnp, blackjax,
        log_prior_fn=log_prior_fn, log_likelihood_fn=log_likelihood_fn,
        n_particles=int(n_particles), dim=int(dim),
        target_ess=float(target_ess), num_mcmc_steps=int(num_mcmc_steps),
        hmc_step_size=float(hmc_step_size),
        hmc_num_integration_steps=int(hmc_num_integration_steps),
        cov_preconditioner=bool(cov_preconditioner),
        waste_free=bool(waste_free),
    )

    # JIT once; each smc.step call would otherwise retrace and recompile on
    # every Python iteration, which dominates wall-clock for small per-step work.
    jit_step = jax.jit(smc.step)

    state = smc.init(init_particles)

    def _ess_from_weights(w):
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            return float("nan")
        wn = w / s
        denom = float(np.sum(wn * wn))
        return 1.0 / denom if denom > 0 else float("nan")

    t0 = _time.perf_counter()
    extras = []
    if cov_preconditioner:
        extras.append("cov-precond")
    if waste_free:
        extras.append(f"waste-free(p={int(num_mcmc_steps)})")
    if adapt_hmc_step_size:
        extras.append(f"adapt-eps(target={hmc_target_accept:.2f})")
    extras_str = (" | " + ",".join(extras)) if extras else ""
    if progressbar:
        print(
            f"[bjsi_blackjax] chain {chain_id + 1}/{n_chains} | "
            f"start | particles={n_particles} dim={dim} "
            f"target_ess={target_ess:.3g} hmc(eps={hmc_step_size:g}, "
            f"L={hmc_num_integration_steps}) mcmc_steps={num_mcmc_steps}"
            f"{extras_str}",
            flush=True,
        )

    inner = _extract_inner_state(state)
    lambdas = [float(inner.tempering_param)]
    n_iter = 0
    compile_time = None
    cur_step = float(hmc_step_size)
    last_accept = float("nan")
    # Step-size adaptation requires the wrapped state form so we can mutate
    # `state.parameter_override["step_size"]` between iterations.
    can_adapt_step = bool(adapt_hmc_step_size) and hasattr(state, "parameter_override")
    while float(_extract_inner_state(state).tempering_param) < 1.0 and n_iter < max_iter:
        key, sub = jax.random.split(key)
        step_t0 = _time.perf_counter()
        state, info = jit_step(sub, state)
        inner = _extract_inner_state(state)
        # Block until the JIT-compiled step actually finishes so timings are real.
        inner.particles.block_until_ready()
        step_dt = _time.perf_counter() - step_t0
        if n_iter == 0:
            compile_time = step_dt
        lam = float(inner.tempering_param)
        lambdas.append(lam)
        n_iter += 1

        if can_adapt_step:
            try:
                last_accept = float(jnp.mean(info.update_info.acceptance_rate))
            except Exception:
                last_accept = float("nan")
            if np.isfinite(last_accept):
                # Damped multiplicative update on log step. alpha < 1 prevents
                # the early-iter accept≈1 transient from blowing step up.
                cur_step = float(np.clip(
                    cur_step * float(np.exp(
                        hmc_step_adaptation_rate * (last_accept - hmc_target_accept)
                    )),
                    hmc_step_min, hmc_step_max,
                ))
                existing_step = state.parameter_override["step_size"]
                new_override = {
                    **state.parameter_override,
                    "step_size": jnp.asarray(
                        [cur_step], dtype=existing_step.dtype
                    ).reshape(existing_step.shape),
                }
                state = state._replace(parameter_override=new_override)

        if progressbar and (n_iter % max(1, int(log_every)) == 0 or lam >= 1.0):
            ess = _ess_from_weights(inner.weights)
            elapsed = _time.perf_counter() - t0
            tag = " [compile]" if n_iter == 1 else ""
            adapt_str = (
                f" | accept={last_accept:.2f} eps={cur_step:.4f}"
                if can_adapt_step else ""
            )
            print(
                f"[bjsi_blackjax] chain {chain_id + 1}/{n_chains} | "
                f"iter {n_iter:3d} | lambda={lam:.4f} | "
                f"ess={ess:7.1f}/{n_particles}{adapt_str} | "
                f"step={step_dt:5.2f}s{tag} | elapsed={elapsed:6.1f}s",
                flush=True,
            )

    inner = _extract_inner_state(state)
    final_lambda = float(inner.tempering_param)
    if progressbar:
        elapsed = _time.perf_counter() - t0
        status = "converged" if final_lambda >= 1.0 else "max_iter"
        steady_n = max(1, n_iter - 1)
        steady_total = elapsed - (compile_time or 0.0)
        avg_step = steady_total / steady_n if steady_n > 0 else float("nan")
        print(
            f"[bjsi_blackjax] chain {chain_id + 1}/{n_chains} | done ({status}) | "
            f"iters={n_iter} final_lambda={final_lambda:.4f} | "
            f"compile={(compile_time or 0.0):.1f}s avg_step={avg_step:.2f}s "
            f"total={elapsed:.1f}s",
            flush=True,
        )

    return np.asarray(inner.particles), {
        "n_iterations": n_iter,
        "lambda_schedule": lambdas,
        "final_lambda": final_lambda,
        "compile_time_s": float(compile_time) if compile_time is not None else None,
        "total_time_s": float(_time.perf_counter() - t0),
    }


def _run_chains_parallel(jax, jnp, blackjax, *,
                         log_prior_fn, log_likelihood_fn,
                         init_particles_per_chain,
                         keys_per_chain,
                         target_ess: float, num_mcmc_steps: int,
                         hmc_step_size: float,
                         hmc_num_integration_steps: int,
                         cov_preconditioner: bool = True,
                         waste_free: bool = True,
                         adapt_hmc_step_size: bool = True,
                         hmc_target_accept: float = 0.75,
                         hmc_step_adaptation_rate: float = 0.5,
                         hmc_step_min: float = 1e-3,
                         hmc_step_max: float = 0.5,
                         max_iter: int = 200,
                         progressbar: bool = False, log_every: int = 1):
    """Run all SMC chains in lock-step via jax.pmap.

    `init_particles_per_chain` shape: (n_chains, n_particles, dim).
    `keys_per_chain` shape: (n_chains, 2)  (stacked PRNGKeys).
    Returns (particles_arr, list_of_per_chain_info_dicts).
    """
    n_chains = int(init_particles_per_chain.shape[0])
    n_particles = int(init_particles_per_chain.shape[1])
    dim = int(init_particles_per_chain.shape[2])

    n_devices = jax.local_device_count()
    if n_devices < n_chains:
        raise RuntimeError(
            f"parallel_chains=True needs jax.local_device_count() >= chains; "
            f"got devices={n_devices}, chains={n_chains}. On CPU, set "
            f"XLA_FLAGS='--xla_force_host_platform_device_count={n_chains}' "
            f"BEFORE the first jax import in your script."
        )

    smc, _wrapped = _build_smc_algorithm(
        jax, jnp, blackjax,
        log_prior_fn=log_prior_fn, log_likelihood_fn=log_likelihood_fn,
        n_particles=n_particles, dim=dim,
        target_ess=float(target_ess), num_mcmc_steps=int(num_mcmc_steps),
        hmc_step_size=float(hmc_step_size),
        hmc_num_integration_steps=int(hmc_num_integration_steps),
        cov_preconditioner=bool(cov_preconditioner),
        waste_free=bool(waste_free),
    )

    p_init = jax.pmap(smc.init)
    p_step = jax.pmap(smc.step)

    states = p_init(init_particles_per_chain)

    def _ess_from_weights(w):
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            return float("nan")
        wn = w / s
        denom = float(np.sum(wn * wn))
        return 1.0 / denom if denom > 0 else float("nan")

    t0 = _time.perf_counter()
    extras = []
    if cov_preconditioner:
        extras.append("cov-precond")
    if waste_free:
        extras.append(f"waste-free(p={int(num_mcmc_steps)})")
    if adapt_hmc_step_size:
        extras.append(f"adapt-eps(target={hmc_target_accept:.2f})")
    extras_str = (" | " + ",".join(extras)) if extras else ""
    if progressbar:
        print(
            f"[bjsi_blackjax] parallel SMC | chains={n_chains} (pmap, devices="
            f"{n_devices}) | particles={n_particles} dim={dim} "
            f"target_ess={target_ess:.3g} hmc(eps={hmc_step_size:g}, "
            f"L={hmc_num_integration_steps}) mcmc_steps={num_mcmc_steps}"
            f"{extras_str}",
            flush=True,
        )

    keys = keys_per_chain
    n_iter = 0
    compile_time = None
    lams_init = np.asarray(_extract_inner_state(states).tempering_param)
    lambdas_per_chain = [[float(lams_init[c])] for c in range(n_chains)]

    use_ansi = bool(progressbar) and _sys.stdout.isatty()
    block_lines = n_chains + 1  # header + one line per chain
    printed_block = False
    cur_step_per_chain = np.full((n_chains,), float(hmc_step_size), dtype=np.float64)
    last_accept_per_chain = np.full((n_chains,), float("nan"), dtype=np.float64)
    can_adapt_step = bool(adapt_hmc_step_size) and hasattr(states, "parameter_override")

    def _emit_iter_block(n_iter_, step_dt_, elapsed_, lams_, ess_vals_):
        nonlocal printed_block
        tag = " [compile]" if n_iter_ == 1 else ""
        header = (
            f"[bjsi_blackjax] iter {n_iter_:3d} | step={step_dt_:5.2f}s{tag} "
            f"| elapsed={elapsed_:6.1f}s"
        )
        if can_adapt_step:
            per_chain = [
                f"  chain {c + 1}/{n_chains} | lambda={lams_[c]:.4f} | "
                f"ess={ess_vals_[c]:7.1f}/{n_particles} | "
                f"accept={last_accept_per_chain[c]:.2f} eps={cur_step_per_chain[c]:.4f}"
                for c in range(n_chains)
            ]
        else:
            per_chain = [
                f"  chain {c + 1}/{n_chains} | lambda={lams_[c]:.4f} | "
                f"ess={ess_vals_[c]:7.1f}/{n_particles}"
                for c in range(n_chains)
            ]
        if use_ansi:
            if printed_block:
                # Move cursor up to the start of the previous block.
                _sys.stdout.write(f"\033[{block_lines}F")
            for line in (header, *per_chain):
                _sys.stdout.write(f"\033[K{line}\n")  # clear line, then write
            _sys.stdout.flush()
            printed_block = True
        else:
            print(header, flush=False)
            for line in per_chain:
                print(line, flush=False)
            _sys.stdout.flush()

    while True:
        inner = _extract_inner_state(states)
        lams_now = np.asarray(inner.tempering_param)
        if float(np.min(lams_now)) >= 1.0 or n_iter >= max_iter:
            break

        keys_split = jax.vmap(lambda k: jax.random.split(k, 2))(keys)
        keys = keys_split[:, 0]
        sub_keys = keys_split[:, 1]

        step_t0 = _time.perf_counter()
        states, info = p_step(sub_keys, states)
        inner = _extract_inner_state(states)
        inner.particles.block_until_ready()
        step_dt = _time.perf_counter() - step_t0
        if n_iter == 0:
            compile_time = step_dt

        n_iter += 1
        lams_now = np.asarray(inner.tempering_param)
        for c in range(n_chains):
            lambdas_per_chain[c].append(float(lams_now[c]))

        if can_adapt_step:
            try:
                # info.update_info.acceptance_rate shape (n_chains, num_resampled, p-1)
                ar = np.asarray(info.update_info.acceptance_rate, dtype=np.float64)
                # Reduce all axes except chain.
                last_accept_per_chain = ar.reshape(n_chains, -1).mean(axis=1)
            except Exception:
                last_accept_per_chain = np.full((n_chains,), float("nan"))
            mask = np.isfinite(last_accept_per_chain)
            if mask.any():
                cur_step_per_chain[mask] = np.clip(
                    cur_step_per_chain[mask] * np.exp(
                        hmc_step_adaptation_rate
                        * (last_accept_per_chain[mask] - hmc_target_accept)
                    ),
                    hmc_step_min, hmc_step_max,
                )
                existing_step = states.parameter_override["step_size"]
                new_step_arr = jnp.asarray(
                    cur_step_per_chain.reshape(existing_step.shape),
                    dtype=existing_step.dtype,
                )
                new_override = {
                    **states.parameter_override,
                    "step_size": new_step_arr,
                }
                states = states._replace(parameter_override=new_override)

        if progressbar and (n_iter % max(1, int(log_every)) == 0
                            or float(np.min(lams_now)) >= 1.0):
            elapsed = _time.perf_counter() - t0
            ess_vals = [_ess_from_weights(inner.weights[c])
                        for c in range(n_chains)]
            _emit_iter_block(n_iter, step_dt, elapsed, lams_now, ess_vals)

    inner = _extract_inner_state(states)
    if progressbar:
        elapsed = _time.perf_counter() - t0
        steady_n = max(1, n_iter - 1)
        steady_total = elapsed - (compile_time or 0.0)
        avg_step = steady_total / steady_n if steady_n > 0 else float("nan")
        lams_now = np.asarray(inner.tempering_param)
        status = ("converged" if float(np.min(lams_now)) >= 1.0
                  else "max_iter")
        lam_str = " ".join(f"{l:.4f}" for l in lams_now)
        print(
            f"[bjsi_blackjax] parallel SMC done ({status}) | iters={n_iter} | "
            f"final_lambda=[{lam_str}] | compile={(compile_time or 0.0):.1f}s "
            f"avg_step={avg_step:.2f}s total={elapsed:.1f}s",
            flush=True,
        )

    particles_arr = np.asarray(inner.particles)
    lams_now = np.asarray(inner.tempering_param)
    convergence = []
    total_t = float(_time.perf_counter() - t0)
    for c in range(n_chains):
        convergence.append({
            "n_iterations": n_iter,
            "lambda_schedule": lambdas_per_chain[c],
            "final_lambda": float(lams_now[c]),
            "compile_time_s": (float(compile_time)
                                if compile_time is not None else None),
            "total_time_s": total_t,
        })
    return particles_arr, convergence


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------
def _hdi_1d(arr, hdi_prob):
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


def _build_inference_data(*, R, Sigma, R_matrix, mu_arr, p_plane2, p_plane2_post,
                          chains, draws, tau0_arr=None, mu_raw_arr=None,
                          q_raw_arr=None):
    """Wrap the per-particle constrained values into an arviz InferenceData
    with the variables the contract test requires."""
    import xarray as xr

    coords = {"chain": np.arange(chains), "draw": np.arange(draws)}

    def reshape_chain_draw(arr, extra_dims=()):
        return np.asarray(arr).reshape((chains, draws, *extra_dims))

    data_vars = {
        "R": (("chain", "draw"), reshape_chain_draw(R)),
        "Sigma": (
            ("chain", "draw", "Sigma_dim_0", "Sigma_dim_1"),
            reshape_chain_draw(Sigma, (3, 3)),
        ),
        "R_matrix": (
            ("chain", "draw", "R_matrix_dim_0", "R_matrix_dim_1"),
            reshape_chain_draw(R_matrix, (3, 3)),
        ),
        "mu": (("chain", "draw"), reshape_chain_draw(mu_arr)),
        "p_plane2": (
            ("chain", "draw", "p_plane2_dim_0"),
            reshape_chain_draw(p_plane2, (p_plane2.shape[-1],)),
        ),
        "p_plane2_post": (
            ("chain", "draw", "p_plane2_post_dim_0"),
            reshape_chain_draw(p_plane2_post, (p_plane2_post.shape[-1],)),
        ),
    }
    if tau0_arr is not None:
        data_vars["tau0"] = (("chain", "draw"), reshape_chain_draw(tau0_arr))
    if mu_raw_arr is not None:
        data_vars["mu_raw"] = (("chain", "draw"), reshape_chain_draw(mu_raw_arr))
    if q_raw_arr is not None:
        data_vars["q_raw"] = (
            ("chain", "draw", "q_raw_dim_0"),
            reshape_chain_draw(q_raw_arr, (4,)),
        )

    posterior = xr.Dataset(
        {k: xr.DataArray(v[1], dims=v[0]) for k, v in data_vars.items()},
        coords=coords,
    )
    return az.InferenceData(posterior=posterior)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
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
    draws: int = 1000,
    chains: int = 1,
    cores: int = 4,
    threshold: float = 0.25,
    correlation_threshold: float = 0.01,
    kernel: str = "IMH",  # accepted for compat, ignored
    random_seed: Optional[int | list[int]] = None,
    progressbar: bool = True,  # accepted for compat, ignored
    return_plane_probabilities: bool = True,
    hdi_prob: float = 0.9,
    iterative_plane_selection: bool = False,
    iterative_kwargs: Optional[Dict[str, Any]] = None,
    selection_beta: Optional[float] = None,
    slip_misfit_sigma: float = 0.35,
    slip_likelihood: str = "gaussian",
    slip_vmf_kappa: Optional[float] = None,
    instability_beta: float = 10.0,
    signed_instability: bool = False,
    friction_fixed: Optional[float] = None,
    enforce_constant_shear: bool = False,
    shear_sigma: float = 0.2,
    shear_target: Optional[float] = None,
    shear_target_sigma: float = 0.3,
    shear_center: str = "mean",
    shear_weight: float = 0.1,
    weighted_likelihood: bool = False,
    likelihood_weight_mode: str = "plane",
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
    # BlackJAX-specific tuning (drop-in defaults are reasonable):
    hmc_step_size: float = 0.05,
    hmc_num_integration_steps: int = 8,
    num_mcmc_steps: int = 30,
    smc_max_iterations: int = 200,
    parallel_chains: bool = False,
    # SMC mutation tuning (defaults match canonical BlackJAX recipe for
    # continuous targets — see Dau & Chopin 2022 for waste-free):
    cov_preconditioner: bool = True,
    waste_free: bool = True,
    adapt_hmc_step_size: bool = True,
    hmc_target_accept: float = 0.8,
    hmc_step_adaptation_rate: float = 0.3,
    hmc_step_min: float = 1e-3,
    hmc_step_max: float = 0.5,
) -> Dict[str, Any]:
    """BlackJAX adaptive tempered SMC port; see module docstring."""
    if iterative_plane_selection:
        raise NotImplementedError(
            "iterative_plane_selection=True is not supported by the BlackJAX port. "
            "Use bjsi.Bayesian_joint_plane_selection_SMC for that path."
        )

    jax, jnp, blackjax = _import_jax_blackjax()
    helpers = _build_jax_helpers(jnp)

    # ------ input validation (mirror bjsi.py:816-893 minus iterative branch) ------
    strikes_1 = np.asarray(strikes_1, dtype=np.float64)
    dips_1 = np.asarray(dips_1, dtype=np.float64)
    rakes_1 = np.asarray(rakes_1, dtype=np.float64)
    strikes_2 = np.asarray(strikes_2, dtype=np.float64)
    dips_2 = np.asarray(dips_2, dtype=np.float64)
    rakes_2 = np.asarray(rakes_2, dtype=np.float64)

    N = len(strikes_1)
    if not (len(dips_1) == len(rakes_1) == len(strikes_2) == len(dips_2)
            == len(rakes_2) == N):
        raise ValueError("All input arrays must have the same length")

    n1, s1 = normal_slip_vectors_batch(strikes_1, dips_1, rakes_1, direction="inward")
    n2, s2 = normal_slip_vectors_batch(strikes_2, dips_2, rakes_2, direction="inward")

    if event_weights is not None:
        event_weights = np.asarray(event_weights, dtype=np.float64).reshape(-1)
        if event_weights.size != N:
            raise ValueError(f"event_weights must have shape (N,), got {event_weights.shape}")
        if not np.all(np.isfinite(event_weights)):
            raise ValueError("event_weights must be finite")
        if float(event_weight_power) != 1.0:
            event_weights = event_weights ** float(event_weight_power)

    q_prior_mu_arr = None
    if q_prior_mu is not None:
        q_prior_mu_arr = np.asarray(q_prior_mu, dtype=np.float64).reshape(-1)
        if q_prior_mu_arr.size != 4:
            raise ValueError("q_prior_mu must have 4 elements")
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
            raise ValueError("plane2_prior_probs must have shape (N,)")
        plane2_prior_probs_arr = np.clip(plane2_prior_probs_arr, 1e-6, 1.0 - 1e-6)

    plane_prior_strength = float(plane_prior_strength)
    clustering_prior_strength = float(clustering_prior_strength)

    slip_likelihood_name, slip_misfit_sigma, slip_vmf_kappa_val = (
        _resolve_slip_likelihood_params(slip_likelihood, slip_misfit_sigma, slip_vmf_kappa)
    )

    # μ handling
    method = str(infer_friction_method or "sample").lower()
    sample_mu = bool(infer_friction and method == "sample")
    mu_const = (
        float(friction_fixed)
        if friction_fixed is not None
        else 0.5 * (float(friction_range[0]) + float(friction_range[1]))
    )

    # tau0 handling
    sample_tau0 = bool(
        enforce_constant_shear
        and float(shear_weight) > 0.0
        and (shear_center or "mean").lower() == "learned"
    )

    layout = _ThetaLayout(sample_mu=sample_mu, sample_tau0=sample_tau0)

    log_prior_fn, log_likelihood_fn, to_constrained = _build_logp_fns(
        jnp=jnp, helpers=helpers,
        n1=n1, n2=n2, s1=s1, s2=s2,
        layout=layout,
        q_prior_mu_arr=q_prior_mu_arr, q_prior_sigma=q_prior_sigma,
        R_prior_mu_val=R_prior_mu_val, R_prior_sigma=R_prior_sigma,
        mu_const=mu_const,
        friction_prior_params=friction_prior_params,
        friction_range=friction_range,
        event_weights_arr=event_weights,
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

    # ------ Run SMC for each chain ------
    if random_seed is None:
        seeds = [np.random.randint(0, 2**31 - 1) for _ in range(int(chains))]
    elif isinstance(random_seed, (list, tuple)):
        seeds = list(random_seed)
        if len(seeds) < chains:
            seeds = seeds + [int(seeds[-1]) + i + 1 for i in range(chains - len(seeds))]
    else:
        base = int(random_seed)
        seeds = [base + i for i in range(int(chains))]

    overall_t0 = _time.perf_counter()
    if progressbar:
        mode_str = "pmap (parallel)" if (parallel_chains and int(chains) > 1) else "sequential"
        print(
            f"[bjsi_blackjax] starting SMC: chains={int(chains)} draws={int(draws)} "
            f"events={N} | mode={mode_str}",
            flush=True,
        )

    if parallel_chains and int(chains) > 1:
        # Sample initial particles per chain in Python, then drive all chains
        # together via pmap.
        init_per_chain = []
        keys_after_init = []
        for c in range(int(chains)):
            key = jax.random.PRNGKey(int(seeds[c]))
            init_particles, key = _sample_initial_particles(
                jax, jnp, key, int(draws), layout,
                q_prior_mu_arr=q_prior_mu_arr, q_prior_sigma=q_prior_sigma,
                R_prior_mu_val=R_prior_mu_val, R_prior_sigma=R_prior_sigma,
                friction_prior_params=friction_prior_params,
                friction_range=friction_range,
                shear_target_sigma=shear_target_sigma,
            )
            init_per_chain.append(init_particles)
            keys_after_init.append(key)
        init_per_chain_arr = jnp.stack(init_per_chain, axis=0)
        keys_arr = jnp.stack(keys_after_init, axis=0)

        particles_arr, convergence_per_chain = _run_chains_parallel(
            jax, jnp, blackjax,
            log_prior_fn=log_prior_fn,
            log_likelihood_fn=log_likelihood_fn,
            init_particles_per_chain=init_per_chain_arr,
            keys_per_chain=keys_arr,
            target_ess=float(threshold),
            num_mcmc_steps=int(num_mcmc_steps),
            hmc_step_size=float(hmc_step_size),
            hmc_num_integration_steps=int(hmc_num_integration_steps),
            cov_preconditioner=bool(cov_preconditioner),
            waste_free=bool(waste_free),
            adapt_hmc_step_size=bool(adapt_hmc_step_size),
            hmc_target_accept=float(hmc_target_accept),
            hmc_step_adaptation_rate=float(hmc_step_adaptation_rate),
            hmc_step_min=float(hmc_step_min),
            hmc_step_max=float(hmc_step_max),
            max_iter=int(smc_max_iterations),
            progressbar=bool(progressbar),
        )
    else:
        all_particles = []
        convergence_per_chain = []
        for c in range(int(chains)):
            key = jax.random.PRNGKey(int(seeds[c]))
            init_particles, key = _sample_initial_particles(
                jax, jnp, key, int(draws), layout,
                q_prior_mu_arr=q_prior_mu_arr, q_prior_sigma=q_prior_sigma,
                R_prior_mu_val=R_prior_mu_val, R_prior_sigma=R_prior_sigma,
                friction_prior_params=friction_prior_params,
                friction_range=friction_range,
                shear_target_sigma=shear_target_sigma,
            )
            particles, info = _run_adaptive_smc(
                jax, jnp, blackjax,
                log_prior_fn=log_prior_fn,
                log_likelihood_fn=log_likelihood_fn,
                init_particles=init_particles,
                key=key,
                target_ess=float(threshold),
                num_mcmc_steps=int(num_mcmc_steps),
                hmc_step_size=float(hmc_step_size),
                hmc_num_integration_steps=int(hmc_num_integration_steps),
                cov_preconditioner=bool(cov_preconditioner),
                waste_free=bool(waste_free),
                adapt_hmc_step_size=bool(adapt_hmc_step_size),
                hmc_target_accept=float(hmc_target_accept),
                hmc_step_adaptation_rate=float(hmc_step_adaptation_rate),
                hmc_step_min=float(hmc_step_min),
                hmc_step_max=float(hmc_step_max),
                max_iter=int(smc_max_iterations),
                progressbar=bool(progressbar),
                chain_id=c,
                n_chains=int(chains),
            )
            all_particles.append(particles)
            convergence_per_chain.append(info)
        particles_arr = np.stack(all_particles, axis=0)

    if progressbar:
        print(
            f"[bjsi_blackjax] all chains done | total elapsed="
            f"{_time.perf_counter() - overall_t0:.1f}s",
            flush=True,
        )

    # ------ Convert particles → constrained, compute derived quantities ------
    flat = particles_arr.reshape(-1, particles_arr.shape[-1])

    # Vectorized constrained transforms in numpy
    q_raw_all = flat[:, 0:4]
    q_norm_all = np.linalg.norm(q_raw_all, axis=-1, keepdims=True) + 1e-8
    q_unit_all = q_raw_all / q_norm_all
    R_logit_all = flat[:, 4]
    R_all = 1.0 / (1.0 + np.exp(-R_logit_all))

    if sample_mu:
        mu_logit_all = flat[:, layout.mu_idx]
        mu_raw_all = 1.0 / (1.0 + np.exp(-mu_logit_all))
        mu_all = float(friction_range[0]) + (
            float(friction_range[1]) - float(friction_range[0])
        ) * mu_raw_all
    else:
        mu_raw_all = None
        mu_all = np.full(flat.shape[0], float(mu_const))

    if sample_tau0:
        tau0_all = flat[:, layout.tau0_idx]
    else:
        tau0_all = None

    # Compute Sigma, p_plane2, p_plane2_post per-particle in numpy (vectorized)
    R_matrix_all, Sigma_all, p2_all, p2_post_all = _compute_derived_numpy(
        q_unit_all, R_all, mu_all,
        n1, n2, s1, s2,
        slip_likelihood_name=slip_likelihood_name,
        slip_misfit_sigma=slip_misfit_sigma,
        slip_vmf_kappa_val=slip_vmf_kappa_val,
        instability_beta=float(
            instability_beta if selection_beta is None else selection_beta
        ),
        signed_instability=bool(signed_instability),
        clustering_prior_strength=float(clustering_prior_strength),
        plane2_prior_probs_arr=plane2_prior_probs_arr,
        plane_prior_strength=float(plane_prior_strength),
        weighted_likelihood=bool(weighted_likelihood),
        likelihood_weight_mode=likelihood_weight_mode,
        tau_weight_exponent=float(tau_weight_exponent),
        normalize_tau_weights=bool(normalize_tau_weights),
        tau_weight_clip=tau_weight_clip,
        event_weights=event_weights,
    )

    idata = _build_inference_data(
        R=R_all, Sigma=Sigma_all, R_matrix=R_matrix_all, mu_arr=mu_all,
        p_plane2=p2_all, p_plane2_post=p2_post_all,
        chains=int(chains), draws=int(draws),
        tau0_arr=tau0_all, mu_raw_arr=mu_raw_all, q_raw_arr=q_raw_all,
    )

    # ------ Post-processing (mirror bjsi.py:1000-1234) ------
    R_samples = R_all  # flat (chains*draws,)
    Sigma_samples = np.transpose(Sigma_all, (1, 2, 0))  # → (3,3,S)

    R_median = float(np.median(R_samples))
    R_mean = float(np.mean(R_samples))
    R_std = float(np.std(R_samples))
    R_CI95 = (
        float(np.quantile(R_samples, 0.025)),
        float(np.quantile(R_samples, 0.975)),
    )

    Sigma_median = np.median(Sigma_samples, axis=2)
    norm = np.max(np.abs(np.linalg.eigvalsh(Sigma_median)))
    Sigma_median = Sigma_median / (norm if norm > 0 else 1.0)
    ps_median, pd_median = stress_tensor_eigendecomposition(Sigma_median)

    mu_samples = mu_all if (sample_mu or True) else None  # always present
    tau0_samples = tau0_all

    Sigma_samples_norm = np.empty_like(Sigma_samples)
    for i in range(Sigma_samples.shape[2]):
        Sigma_i = Sigma_samples[:, :, i]
        norm_i = np.max(np.abs(np.linalg.eigvalsh(Sigma_i)))
        Sigma_samples_norm[:, :, i] = Sigma_i / (norm_i if norm_i > 0 else 1.0)

    # HDI for Sigma
    sigma_stack = np.moveaxis(Sigma_samples_norm, 2, 0)
    Sigma_hdi = np.full((3, 3, 2), np.nan, dtype=float)
    try:
        for i in range(3):
            for j in range(3):
                h_ij = _hdi_1d(sigma_stack[:, i, j], hdi_prob)
                if h_ij is not None:
                    Sigma_hdi[i, j, 0] = h_ij[0]
                    Sigma_hdi[i, j, 1] = h_ij[1]
    except Exception:
        Sigma_hdi = None

    R_hdi = _hdi_1d(R_samples, hdi_prob)
    mu_hdi = _hdi_1d(mu_samples, hdi_prob) if mu_samples is not None else None
    tau0_hdi = _hdi_1d(tau0_samples, hdi_prob) if tau0_samples is not None else None

    n_boot = Sigma_samples.shape[2]
    boot_ps = np.zeros((n_boot, 3), dtype=np.float64)
    boot_pd = np.zeros((n_boot, 3, 3), dtype=np.float64)
    for i in range(n_boot):
        ps_i, pd_i = stress_tensor_eigendecomposition(Sigma_samples_norm[:, :, i])
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
        "boot_principal_stresses": boot_ps,
        "boot_principal_directions": boot_pd,
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
        "mu": float(np.median(mu_samples)) if mu_samples is not None else float(mu_const),
    }
    if abs(float(hdi_prob) - 0.9) < 1e-6:
        results["hdi_90"] = {
            "prob": float(hdi_prob),
            "R": R_hdi,
            "Sigma": Sigma_hdi,
            "mu": mu_hdi,
            "tau0": tau0_hdi,
        }

    # Plane probabilities and MAP
    if return_plane_probabilities:
        # p2_post_all: (S, N) → mean over draws for marginal posterior probability
        plane_2_prob = np.mean(p2_post_all, axis=0)
        plane_1_prob = 1.0 - plane_2_prob
        results["plane_probabilities"] = np.stack([plane_1_prob, plane_2_prob], axis=1)
        results["plane_selection_map"] = (plane_2_prob > 0.5).astype(int)

    # Friction
    if infer_friction and method == "posthoc":
        if _det is None:
            mu_hat = None
        else:
            mu_min, mu_max = float(friction_range[0]), float(friction_range[1])
            mu_grid = np.linspace(mu_min, mu_max, 76)
            scores = np.empty(mu_grid.size, dtype=float)
            for i, mu_try in enumerate(mu_grid):
                I_try = _det.compute_instability_parameter(
                    pd_median, float(R_median), float(mu_try),
                    strikes_1, dips_1, rakes_1,
                    strikes_2, dips_2, rakes_2,
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
    elif sample_mu:
        mu_med = float(np.median(mu_all))
        mu_mean = float(np.mean(mu_all))
        mu_std = float(np.std(mu_all))
        mu_ci = (
            float(np.quantile(mu_all, 0.025)),
            float(np.quantile(mu_all, 0.975)),
        )
        results["friction_coefficient"] = mu_mean
        results["mu"] = mu_med
        results["friction_median"] = mu_med
        results["friction_mean"] = mu_mean
        results["friction_std"] = mu_std
        results["friction_CI95"] = mu_ci
    else:
        results["friction_coefficient"] = float(mu_const)
        results["mu"] = float(mu_const)
        results["friction_median"] = float(mu_const)
        results["friction_mean"] = float(mu_const)
        results["friction_std"] = 0.0
        results["friction_CI95"] = (float(mu_const), float(mu_const))

    if tau0_samples is not None and tau0_samples.size > 0:
        results["tau0_median"] = float(np.median(tau0_samples))
        results["tau0_mean"] = float(np.mean(tau0_samples))
        results["tau0_std"] = float(np.std(tau0_samples))
        results["tau0_CI95"] = (
            float(np.quantile(tau0_samples, 0.025)),
            float(np.quantile(tau0_samples, 0.975)),
        )

    results["convergence"] = {
        "n_samples": int(draws) * int(chains),
        "n_chains": int(chains),
        "smc_kernel": "blackjax_hmc",
        "smc_threshold": float(threshold),
        "smc_correlation_threshold": float(correlation_threshold),
        "smc_random_seed": random_seed,
        "per_chain": convergence_per_chain,
    }

    return results


# ---------------------------------------------------------------------------
# Numpy reproductions of the per-event derived quantities
# (used post-sampling to populate idata + plane probabilities)
# ---------------------------------------------------------------------------
def _compute_derived_numpy(
    q_unit_all, R_all, mu_all, n1, n2, s1, s2, *,
    slip_likelihood_name, slip_misfit_sigma, slip_vmf_kappa_val,
    instability_beta, signed_instability,
    clustering_prior_strength, plane2_prior_probs_arr, plane_prior_strength,
    weighted_likelihood, likelihood_weight_mode, tau_weight_exponent,
    normalize_tau_weights, tau_weight_clip, event_weights,
):
    """For each posterior draw, recompute Sigma, p_plane2 and p_plane2_post.

    Returns
    -------
    R_matrix_all : (S, 3, 3)
    Sigma_all   : (S, 3, 3)
    p2_all      : (S, N)
    p2_post_all : (S, N)
    """
    S = q_unit_all.shape[0]
    N = n1.shape[0]

    R_matrix_all = np.empty((S, 3, 3), dtype=np.float64)
    Sigma_all = np.empty((S, 3, 3), dtype=np.float64)
    p2_all = np.empty((S, N), dtype=np.float64)
    p2_post_all = np.empty((S, N), dtype=np.float64)

    eps = 1e-9
    for k in range(S):
        q = q_unit_all[k]
        Rmat = _quat_to_rotation_matrix_np(q)
        R_matrix_all[k] = Rmat
        sig2 = 2.0 * R_all[k] - 1.0
        diag_vals = np.array([-1.0, sig2, 1.0])
        Sigma = (Rmat * diag_vals) @ Rmat.T
        Sigma_all[k] = Sigma

        # shear traction & magnitudes
        t1 = (Sigma @ n1.T).T
        sn1 = np.sum(t1 * n1, axis=-1)
        ts1 = t1 - sn1[:, None] * n1
        tau1 = np.sqrt(np.sum(ts1**2, axis=-1) + 1e-12)
        s_pred1 = ts1 / (tau1[:, None] + 1e-12)

        t2 = (Sigma @ n2.T).T
        sn2 = np.sum(t2 * n2, axis=-1)
        ts2 = t2 - sn2[:, None] * n2
        tau2 = np.sqrt(np.sum(ts2**2, axis=-1) + 1e-12)
        s_pred2 = ts2 / (tau2[:, None] + 1e-12)

        # instability
        mu = float(mu_all[k])
        denom = math.sqrt(1.0 + mu * mu)
        tau_c = 1.0 / denom
        sig_c = mu / denom
        sig1_c = -1.0
        I1 = (tau1 - mu * (sig1_c - sn1)) / (tau_c - mu * (sig1_c - sig_c))
        I2 = (tau2 - mu * (sig1_c - sn2)) / (tau_c - mu * (sig1_c - sig_c))
        if signed_instability:
            I1 = I1 * np.sign(np.sum(s_pred1 * s1, axis=-1))
            I2 = I2 * np.sign(np.sum(s_pred2 * s2, axis=-1))

        logit_local = float(instability_beta) * (I2 - I1)

        if clustering_prior_strength > 0.0:
            p_tent = 1.0 / (1.0 + np.exp(-logit_local))
            certainty = (2.0 * p_tent - 1.0) ** 2
            M1 = n1[:, :, None] * n1[:, None, :]
            M2 = n2[:, :, None] * n2[:, None, :]
            sum_c = np.sum(certainty) + 1e-12
            T_conf = np.sum(
                certainty[:, None, None] * (
                    (1.0 - p_tent)[:, None, None] * M1
                    + p_tent[:, None, None] * M2
                ),
                axis=0,
            ) / sum_c
            E1 = np.sum(n1 * (T_conf @ n1.T).T, axis=-1)
            E2 = np.sum(n2 * (T_conf @ n2.T).T, axis=-1)
            logit_local = logit_local + clustering_prior_strength * (E2 - E1)

        if plane2_prior_probs_arr is not None and plane_prior_strength > 0.0:
            prior_p2 = np.clip(plane2_prior_probs_arr, eps, 1.0 - eps)
            prior_logit = np.log(prior_p2) - np.log1p(-prior_p2)
            logit_full = logit_local + plane_prior_strength * prior_logit
        else:
            logit_full = logit_local

        p2 = 1.0 / (1.0 + np.exp(-logit_full))
        p2 = np.clip(p2, eps, 1.0 - eps)
        p2_all[k] = p2

        # event weights
        w_event = event_weights if event_weights is not None else 1.0
        w1_tau, w2_tau = _tau_weights_np(
            tau1, tau2, weighted_likelihood, likelihood_weight_mode,
            tau_weight_exponent, normalize_tau_weights, tau_weight_clip,
        )
        w1 = w_event * w1_tau
        w2 = w_event * w2_tau

        ll1 = _slip_logp_np(s1, s_pred1, slip_likelihood_name,
                            slip_misfit_sigma, slip_vmf_kappa_val, w1)
        ll2 = _slip_logp_np(s2, s_pred2, slip_likelihood_name,
                            slip_misfit_sigma, slip_vmf_kappa_val, w2)

        log_p2 = np.log(p2)
        log_1mp2 = np.log1p(-p2)
        logw1 = log_1mp2 + ll1
        logw2 = log_p2 + ll2
        logmix = np.logaddexp(logw1, logw2)
        p2_post_all[k] = np.exp(logw2 - logmix)

    return R_matrix_all, Sigma_all, p2_all, p2_post_all


def _quat_to_rotation_matrix_np(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _slip_logp_np(s_obs, s_pred, family, sigma, vmf_kappa, weight):
    safe_w = np.clip(weight, 1e-12, 1e12)
    obs_n = np.linalg.norm(s_obs, axis=-1, keepdims=True)
    pred_n = np.linalg.norm(s_pred, axis=-1, keepdims=True)
    s_obs_u = s_obs / (obs_n + 1e-12)
    s_pred_u = s_pred / (pred_n + 1e-12)
    if family == "gaussian":
        diff = s_obs_u - s_pred_u
        sigma2 = float(sigma) ** 2
        return (
            -0.5 * safe_w * np.sum(diff**2, axis=-1) / sigma2
            - 1.5 * math.log(2 * math.pi * sigma2)
        )
    if family == "von_mises_fisher":
        kappa_base = float(vmf_kappa)
        kappa_eff = safe_w * kappa_base
        dot = np.clip(np.sum(s_obs_u * s_pred_u, axis=-1), -1.0, 1.0)
        log_sinh_kappa = (
            kappa_base + math.log(-math.expm1(-2.0 * kappa_base)) - math.log(2.0)
        )
        log_c3 = math.log(kappa_base) - math.log(4.0 * math.pi) - log_sinh_kappa
        return log_c3 + kappa_eff * dot
    raise ValueError(f"Unsupported slip likelihood: {family}")


def _tau_weights_np(tau1, tau2, weighted, mode, exponent, normalize, clip):
    if not weighted:
        return np.ones_like(tau1), np.ones_like(tau2)
    p = float(exponent)
    tau_scale = np.mean(0.5 * (tau1 + tau2)) + 1e-12 if normalize else 1.0
    m = str(mode or "plane").lower()
    if m in {"plane", "plane_specific"}:
        w1 = (tau1 / tau_scale) ** p
        w2 = (tau2 / tau_scale) ** p
    elif m in {"event", "common", "symmetric"}:
        tau_e = 0.5 * (tau1 + tau2)
        w1 = w2 = (tau_e / tau_scale) ** p
    elif m in {"none", "off"}:
        w1 = np.ones_like(tau1)
        w2 = np.ones_like(tau2)
    else:
        raise ValueError("likelihood_weight_mode must be plane | event | none")
    if clip is not None:
        lo, hi = float(clip[0]), float(clip[1])
        w1 = np.clip(w1, lo, hi)
        w2 = np.clip(w2, lo, hi)
    return w1, w2
