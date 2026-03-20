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

import numpy as np
import arviz as az
import pymc as pm
import pytensor.tensor as pt
import pytensor.gradient as ptg
from typing import Optional, Tuple, Dict, Any

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
        return -0.5 * safe_weight * pt.sum(diff**2, axis=-1) / sigma2 - 1.5 * pt.log(
            2 * np.pi * (sigma2 / safe_weight)
        )

    if family != "von_mises_fisher":
        raise ValueError(f"Unsupported slip likelihood family: {family}")
    if vmf_kappa is None:
        raise ValueError("vmf_kappa must be provided for the von_mises_fisher likelihood")

    kappa_eff = safe_weight * float(vmf_kappa)
    dot = pt.clip(pt.sum(s_obs_unit * s_pred_unit, axis=-1), -1.0, 1.0)
    log_sinh_kappa = kappa_eff + pt.log(-pt.expm1(-2.0 * kappa_eff)) - np.log(2.0)
    log_c3 = pt.log(kappa_eff) - np.log(4.0 * np.pi) - log_sinh_kappa
    return log_c3 + kappa_eff * dot


# Import stress_tensor_eigendecomposition from utils_stress
# (not defined here - use the one from utils_stress module)


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
    friction_prior_params: Tuple[float, float] = (3.0, 3.0),
    friction_range: Tuple[float, float] = (0.2, 1.0),
    draws: int = 1000,
    chains: int = 1,
    cores: int = 4,
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
    # Plane-selection prior sharpness (sigmoid beta). If None, falls back to instability_beta
    # for backward compatibility.
    selection_beta: Optional[float] = None,
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
    likelihood_weight_mode: str = "plane",
    tau_weight_exponent: float = 2.0,
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
) -> Dict[str, Any]:
    """
    Joint Bayesian inference of stress orientation, shape ratio, and nodal plane selection using SMC.

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
    draws : int, default 1000
        Number of SMC particles (samples) per chain. More draws = better posterior approximation.
    chains : int, default 1
        Number of independent SMC chains. Usually chains=1 with many draws for SMC.
    cores : int, default 4
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
        - "boot_principal_stresses": (B, 3) ndarray, bootstrap samples of principal stresses
        - "boot_principal_directions": (B, 3, 3) ndarray, bootstrap samples of principal directions
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

    # Helper functions for PyMC model
    def quat_to_rotation_matrix(q):
        """Convert unit quaternion (w,x,y,z) to 3x3 rotation matrix.
        
        NOTE: We use pt.stacklists instead of pt.stack([[...]]) to ensure
        the resulting matrix is contiguous, avoiding NumbaPerformanceWarning
        when the matrix or its transpose is used in pt.dot operations.
        """
        w, x, y, z = q[0], q[1], q[2], q[3]
        R11 = 1 - 2*(y*y + z*z)
        R22 = 1 - 2*(x*x + z*z)
        R33 = 1 - 2*(x*x + y*y)
        R12 = 2*(x*y - z*w); R21 = 2*(x*y + z*w)
        R13 = 2*(x*z + y*w); R31 = 2*(x*z - y*w)
        R23 = 2*(y*z - x*w); R32 = 2*(y*z + x*w)
        # Use stacklists for contiguous memory layout
        return pt.stacklists([[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]])

    def stress_tensor_from_R_and_shape(Rmat, Rratio):
        """Build deviatoric stress tensor with Beaucé/Vavryčuk convention.

        Uses reduced-stress parametrization consistent with ilsi.compute_instability_parameter:
        principal stresses (tension positive):
            sigma1 = -1, sigma2 = 2R - 1, sigma3 = +1

        Sigma = Rmat @ diag([sigma1, sigma2, sigma3]) @ Rmat.T
        
        NOTE: We avoid pt.diag(pt.stack(...)) which creates non-contiguous arrays
        and triggers NumbaPerformanceWarning in nutpie. Instead, we scale columns
        directly: Rmat @ D = Rmat * diag_vals (broadcasted column-wise).
        """
        sig1 = -1.0
        sig2 = 2.0 * Rratio - 1.0
        sig3 = 1.0
        # Scale columns of Rmat by principal stresses (equivalent to Rmat @ D)
        # This avoids creating a non-contiguous diagonal matrix
        diag_vals = pt.stack([sig1, sig2, sig3])  # (3,)
        RD = Rmat * diag_vals  # Broadcasting: (3,3) * (3,) scales each column
        return pt.dot(RD, Rmat.T)

    def unit_vector(x, eps=1e-12):
        """Normalize vector to unit length."""
        norm = pt.sqrt(pt.sum(pt.square(x), axis=-1, keepdims=True))
        return x / (norm + eps)

    def shear_magnitude(Sigma, n):
        """
        Compute scalar shear stress magnitude |tau| on plane with normal n.
        Sigma: (3, 3) stress tensor
        n: (N, 3) normal vectors for N events
        """
        t = pt.dot(Sigma, n.T).T  # (N, 3)
        tn = pt.sum(t * n, axis=-1, keepdims=True) * n  # Normal component (N, 3)
        ts = t - tn  # Shear component (N, 3)
        return pt.sqrt(pt.sum(pt.square(ts), axis=-1) + 1e-12)

    def shear_traction_direction(Sigma, n):
        """
        Compute predicted shear traction direction on plane with normal n.
        Sigma: (3, 3) stress tensor
        n: (N, 3) normal vectors for N events
        traction = Sigma @ n^T → (3, N) then transpose to (N, 3)
        shear = traction - (traction·n)n  [remove normal component]
        return unit(shear)
        """
        # Sigma @ n.T gives (3, N), transpose to (N, 3)
        t = pt.dot(Sigma, n.T).T  # (N, 3)
        tn = pt.sum(t * n, axis=-1, keepdims=True) * n  # Normal component (N, 3)
        ts = t - tn  # Shear component (N, 3)
        return unit_vector(ts)

    def _apply_weight_clip(w, clip):
        if clip is None:
            return w
        lo, hi = float(clip[0]), float(clip[1])
        if lo <= 0.0 or hi <= 0.0 or hi < lo:
            raise ValueError("tau_weight_clip must be (low>0, high>0, high>=low)")
        return pt.clip(w, lo, hi)

    def _tau_weights(tau1, tau2):
        """
        Compute (w1, w2) weights for the two-plane mixture likelihood.
        Returned arrays have shape (N,) and are passed to the slip-direction likelihood.
        """
        if not weighted_likelihood:
            return pt.ones_like(tau1), pt.ones_like(tau2)

        mode = str(likelihood_weight_mode or "plane").lower()
        p = float(tau_weight_exponent)
        if p < 0.0:
            raise ValueError("tau_weight_exponent must be >= 0")

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

    # Build PyMC model
    with pm.Model() as model:
        # --- Stress orientation as unit quaternion ---
        if q_prior_mu_arr is not None:
            q_raw = pm.Normal("q_raw", mu=q_prior_mu_arr, sigma=q_prior_sigma, shape=4)
        else:
            q_raw = pm.Normal("q_raw", mu=0.0, sigma=1.0, shape=4)
        # Avoid division by zero in test-value and graph build
        q_norm = pt.sqrt(pt.sum(pt.square(q_raw))) + 1e-8
        q = q_raw / q_norm  # Normalize to unit quaternion

        Rmat = pm.Deterministic("R_matrix", quat_to_rotation_matrix(q))

        # --- Shape ratio R ∈ (0,1) ---
        if R_prior_mu_val is not None:
            # Damped prior towards background R
            Rratio = pm.TruncatedNormal(
                "R", mu=R_prior_mu_val, sigma=R_prior_sigma, lower=0.001, upper=0.999
            )
        else:
            Rratio = pm.Beta("R", alpha=2.0, beta=2.0)

        # --- Build stress tensor ---
        Sigma = pm.Deterministic("Sigma", stress_tensor_from_R_and_shape(Rmat, Rratio))

        # --- Friction coefficient (μ) ---
        # Default SMC behavior keeps μ fixed (to avoid the “convenient μ” feedback loop).
        # For experiments, set infer_friction=True and infer_friction_method="sample" to
        # jointly sample μ with stress (similar to the NUTS implementation).
        if iterative_plane_selection:
            mu_const = float(
                iterative_info.get("mu", 0.5 * (friction_range[0] + friction_range[1]))
            )
            mu = pm.Deterministic("mu", pt.as_tensor_variable(mu_const))
        else:
            method = str(infer_friction_method or "posthoc").lower()
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

        # Optional external (stress-independent) event weights
        if event_weights is not None:
            w_event = pm.Data("event_weights", event_weights)
            w_event = pt.clip(w_event, 1e-12, 1e12)
        else:
            w_event = 1.0

        # --- Likelihood: either fixed planes (preselection) or mixture over both planes ---
        if iterative_plane_selection:
            # Fixed planes from preselection
            n_selected = pm.Data("n_selected", iterative_info["n_selected"])
            s_observed = pm.Data("s_observed", iterative_info["s_selected"])
            # Placeholders for post-processing
            p_plane2_post = pm.Deterministic(
                "p_plane2_post",
                pt.as_tensor_variable(iterative_info["plane_map"].astype(float)),
            )

            # Likelihood on the selected plane
            s_predicted = shear_traction_direction(Sigma, n_selected)  # (N, 3)
            tau_mag = shear_magnitude(Sigma, n_selected)  # (N,)
            if weighted_likelihood:
                w_tau = _tau_weights(tau_mag, tau_mag)[0]
                loglike_per_event = _slip_direction_logp(
                    s_observed,
                    s_predicted,
                    family=slip_likelihood_name,
                    sigma=slip_misfit_sigma,
                    vmf_kappa=slip_vmf_kappa_val,
                    weight=w_event * w_tau,
                )
            else:
                loglike_per_event = _slip_direction_logp(
                    s_observed,
                    s_predicted,
                    family=slip_likelihood_name,
                    sigma=slip_misfit_sigma,
                    vmf_kappa=slip_vmf_kappa_val,
                    weight=w_event,
                )
            pm.Potential("likelihood", pt.sum(loglike_per_event))
        else:
            # Marginalize nodal-plane choice (no discrete z) for SMC stability.
            # This avoids particle degeneracy from per-event discrete indicators.
            s_pred1 = shear_traction_direction(Sigma, n1)
            s_pred2 = shear_traction_direction(Sigma, n2)

            # Instability-based plane selection (depends on mu).
            def instability_parameter_log(Sigma, n, mu, R, s_pred=None, s_obs=None):
                sig1 = -1.0

                denom = pt.sqrt(1.0 + mu**2)
                tau_c = 1.0 / denom
                sig_c = mu / denom

                t = pt.dot(Sigma, n.T).T
                sigma_n = pt.sum(t * n, axis=-1)

                tn = sigma_n[:, None] * n
                ts = t - tn
                tau_mag = pt.sqrt(pt.sum(ts**2, axis=-1) + 1e-12)

                numerator = tau_mag - mu * (sig1 - sigma_n)
                denominator_I = tau_c - mu * (sig1 - sig_c)
                I_val = numerator / denominator_I
                
                if signed_instability and s_pred is not None and s_obs is not None:
                    # Multiply by the sign of the dot product between predicted shear and observed slip
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
                s1,
                s_pred1,
                family=slip_likelihood_name,
                sigma=slip_misfit_sigma,
                vmf_kappa=slip_vmf_kappa_val,
                weight=w1,
            )
            ll2 = _slip_direction_logp(
                s2,
                s_pred2,
                family=slip_likelihood_name,
                sigma=slip_misfit_sigma,
                vmf_kappa=slip_vmf_kappa_val,
                weight=w2,
            )

            inst_delta = inst2 - inst1
            beta_val = float(instability_beta if selection_beta is None else selection_beta)
            beta = pt.as_tensor_variable(beta_val)
            eps = 1e-9
            logit_local = beta * inst_delta
            
            # Apply Certainty-Weighted Clustering Prior
            if clustering_prior_strength > 0.0:
                p_tentative = pm.math.sigmoid(logit_local)
                certainty = pt.square(2.0 * p_tentative - 1.0)
                
                M1 = n1[:, :, None] * n1[:, None, :] 
                M2 = n2[:, :, None] * n2[:, None, :]
                
                sum_certainty = pt.sum(certainty) + 1e-12
                # T_conf is shape (3, 3) representing the average confident orientation
                T_conf = pt.sum(certainty[:, None, None] * ((1.0 - p_tentative)[:, None, None] * M1 + p_tentative[:, None, None] * M2), axis=0) / sum_certainty
                
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

            # Posterior responsibility for plane 2 per event (differentiable)
            p_plane2_post = pm.Deterministic("p_plane2_post", pt.exp(logw2 - logmix))
            tau_mag = (1.0 - p_plane2_post) * tau1 + p_plane2_post * tau2

            # --- Optional: constant‑shear constraint ---
            # Note: this encourages equal shear-traction magnitudes |τ| across events.
            # Use with care: under a uniform stress tensor, |τ| generally varies with fault orientation.
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
                        tau0 = pm.Deterministic("tau0", pt.mean(tau_mag))  # τ̄ per draw
                    elif sc == "fixed":
                        if shear_target is None:
                            raise ValueError("shear_center='fixed' requires shear_target to be set")
                        tau0 = pm.Deterministic("tau0", pt.as_tensor_variable(float(shear_target)))
                    else:  # sc == "learned"
                        tau0 = pm.Normal("tau0", mu=0.5, sigma=float(shear_target_sigma))

                    resid = (tau_mag - tau0) / ss
                    pen = 0.5 * pt.sum(resid * resid)
                    # Use a Potential carrying the full Normal log‑likelihood up to a constant
                    pm.Potential("shear_const_penalty", -sw * pen)

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
        "boot_principal_stresses": boot_ps,
        "boot_principal_directions": boot_pd,
        "R_bootstrap": R_samples,
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
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    cores: int = 4,
    target_accept: float = 0.9,
    nuts_sampler: Optional[str] = None,
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
    # Plane-selection prior sharpness (sigmoid beta). If None, falls back to instability_beta
    # for backward compatibility.
    selection_beta: Optional[float] = None,
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
    likelihood_weight_mode: str = "plane",
    tau_weight_exponent: float = 2.0,
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
) -> Dict[str, Any]:
    """
    Joint Bayesian inference of stress orientation, shape ratio, and nodal plane selection using NUTS.

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

    mu_const: Optional[float] = None

    # Helper functions (duplicated from SMC implementation to keep this self-contained)
    def quat_to_rotation_matrix(q):
        # Use pt.stacklists for contiguous memory layout (avoids NumbaPerformanceWarning)
        w, x, y, z = q[0], q[1], q[2], q[3]
        R11 = 1 - 2 * (y * y + z * z)
        R22 = 1 - 2 * (x * x + z * z)
        R33 = 1 - 2 * (x * x + y * y)
        R12 = 2 * (x * y - z * w)
        R21 = 2 * (x * y + z * w)
        R13 = 2 * (x * z + y * w)
        R31 = 2 * (x * z - y * w)
        R23 = 2 * (y * z - x * w)
        R32 = 2 * (y * z + x * w)
        return pt.stacklists([[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]])

    def stress_tensor_from_R_and_shape(Rmat, Rratio):
        # Scale columns of Rmat by principal stresses (equivalent to Rmat @ diag)
        # Avoids non-contiguous diagonal matrix from pt.stack that triggers
        # NumbaPerformanceWarning in nutpie.
        sig1 = -1.0
        sig2 = 2.0 * Rratio - 1.0
        sig3 = 1.0
        diag_vals = pt.stack([sig1, sig2, sig3])
        RD = Rmat * diag_vals  # (3,3) * (3,) - column-wise scaling
        return pt.dot(RD, Rmat.T)

    def unit_vector(x, eps=1e-12):
        norm = pt.sqrt(pt.sum(pt.square(x), axis=-1, keepdims=True))
        return x / (norm + eps)

    def shear_traction_direction(Sigma, n):
        t = pt.dot(Sigma, n.T).T  # (N, 3)
        tn = pt.sum(t * n, axis=-1, keepdims=True) * n
        ts = t - tn
        return unit_vector(ts)

    def shear_magnitude(Sigma, n):
        t_full = pt.dot(Sigma, n.T).T
        tn_comp = pt.sum(t_full * n, axis=-1, keepdims=True) * n
        ts_comp = t_full - tn_comp
        return pt.sqrt(pt.sum(pt.square(ts_comp), axis=-1) + 1e-12)

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

        mode = str(likelihood_weight_mode or "plane").lower()
        p = float(tau_weight_exponent)
        if p < 0.0:
            raise ValueError("tau_weight_exponent must be >= 0")

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

    # Build PyMC model
    with pm.Model() as model:
        # --- Stress orientation as unit quaternion ---
        if q_prior_mu_arr is not None:
            q_raw = pm.Normal("q_raw", mu=q_prior_mu_arr, sigma=q_prior_sigma, shape=4)
        else:
            q_raw = pm.Normal("q_raw", mu=0.0, sigma=1.0, shape=4)
        q_norm = pt.sqrt(pt.sum(pt.square(q_raw))) + 1e-8
        q = q_raw / q_norm
        Rmat = pm.Deterministic("R_matrix", quat_to_rotation_matrix(q))

        # --- Shape ratio R ∈ (0,1) ---
        if R_prior_mu_val is not None:
            # Damped prior towards background R
            Rratio = pm.TruncatedNormal(
                "R", mu=R_prior_mu_val, sigma=R_prior_sigma, lower=0.001, upper=0.999
            )
        else:
            Rratio = pm.Beta("R", alpha=2.0, beta=2.0)
        Sigma = pm.Deterministic("Sigma", stress_tensor_from_R_and_shape(Rmat, Rratio))

        # --- Optional: Friction coefficient (fixed in iterative mode) ---
        method = str(infer_friction_method or "sample").lower()
        if infer_friction and not iterative_plane_selection and method == "sample":
            mu_raw = pm.Beta("mu_raw", alpha=friction_prior_params[0], beta=friction_prior_params[1])
            mu = pm.Deterministic(
                "mu", friction_range[0] + (friction_range[1] - friction_range[0]) * mu_raw
            )
        else:
            if iterative_plane_selection:
                mu_const = float(iterative_info.get("mu", 0.5 * (friction_range[0] + friction_range[1])))
            else:
                mu_const = (
                    float(friction_fixed)
                    if friction_fixed is not None
                    else 0.5 * (friction_range[0] + friction_range[1])
                )
            mu = pm.Deterministic("mu", pt.as_tensor_variable(mu_const))

        if event_weights is not None:
            w_event = pm.Data("event_weights", event_weights)
            w_event = pt.clip(w_event, 1e-12, 1e12)
        else:
            w_event = 1.0
        # --- Likelihood: either fixed planes (preselection) or mixture over both planes ---
        p_plane2_post = None
        if iterative_plane_selection:
            n_selected = pm.Data("n_selected", iterative_info["n_selected"])
            s_observed = pm.Data("s_observed", iterative_info["s_selected"])
            s_pred = shear_traction_direction(Sigma, n_selected)
            tau_mag = shear_magnitude(Sigma, n_selected)
            if weighted_likelihood:
                w_tau = _tau_weights(tau_mag, tau_mag)[0]
                ll = _slip_direction_logp(
                    s_observed,
                    s_pred,
                    family=slip_likelihood_name,
                    sigma=slip_misfit_sigma,
                    vmf_kappa=slip_vmf_kappa_val,
                    weight=w_event * w_tau,
                )
            else:
                ll = _slip_direction_logp(
                    s_observed,
                    s_pred,
                    family=slip_likelihood_name,
                    sigma=slip_misfit_sigma,
                    vmf_kappa=slip_vmf_kappa_val,
                    weight=w_event,
                )
            pm.Potential("likelihood", pt.sum(ll))
            # Fix scoper for post-processing
            p_plane2_post = pm.Deterministic("p_plane2_post", pt.as_tensor_variable(iterative_info["plane_map"].astype(float)))
        else:
            s_pred1 = shear_traction_direction(Sigma, n1)
            s_pred2 = shear_traction_direction(Sigma, n2)

            eps = 1e-9
            # Instability-based plane selection (depends on mu).
            def instability_parameter_log(Sigma, n, mu, R, s_pred=None, s_obs=None):
                sig1 = -1.0
                # sig2 = 2.0 * R - 1.0  # Not used explicitly in formula but defines the tensor
                # sig3 = 1.0
                
                denom = pt.sqrt(1.0 + mu**2)
                tau_c = 1.0 / denom
                sig_c = mu / denom
                
                t = pt.dot(Sigma, n.T).T
                sigma_n = pt.sum(t * n, axis=-1)
                
                tn = sigma_n[:, None] * n
                ts = t - tn
                tau_mag = pt.sqrt(pt.sum(ts**2, axis=-1) + 1e-12)

                numerator = tau_mag - mu * (sig1 - sigma_n)
                denominator_I = tau_c - mu * (sig1 - sig_c)
                I_val = numerator / denominator_I
                
                if signed_instability and s_pred is not None and s_obs is not None:
                    # Multiply by the sign of the dot product between predicted shear and observed slip
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
                s1,
                s_pred1,
                family=slip_likelihood_name,
                sigma=slip_misfit_sigma,
                vmf_kappa=slip_vmf_kappa_val,
                weight=w1,
            )
            ll2 = _slip_direction_logp(
                s2,
                s_pred2,
                family=slip_likelihood_name,
                sigma=slip_misfit_sigma,
                vmf_kappa=slip_vmf_kappa_val,
                weight=w2,
            )

            inst_delta = (inst2 - inst1)
            beta_val = float(instability_beta if selection_beta is None else selection_beta)
            beta = pt.as_tensor_variable(beta_val)
            logit_local = beta * inst_delta
            
            # Apply Certainty-Weighted Clustering Prior
            if clustering_prior_strength > 0.0:
                p_tentative = pm.math.sigmoid(logit_local)
                certainty = pt.square(2.0 * p_tentative - 1.0)
                
                M1 = n1[:, :, None] * n1[:, None, :] 
                M2 = n2[:, :, None] * n2[:, None, :]
                
                sum_certainty = pt.sum(certainty) + 1e-12
                # T_conf is shape (3, 3) representing the average confident orientation
                T_conf = pt.sum(certainty[:, None, None] * ((1.0 - p_tentative)[:, None, None] * M1 + p_tentative[:, None, None] * M2), axis=0) / sum_certainty
                
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

            # Posterior responsibility for plane 2 per event (differentiable)
            p_plane2_post = pm.Deterministic("p_plane2_post", pt.exp(logw2 - logmix))

            # Shear magnitudes on each plane; blend by posterior responsibility
            tau1 = shear_magnitude(Sigma, n1)
            tau2 = shear_magnitude(Sigma, n2)
            tau_mag = (1.0 - p_plane2_post) * tau1 + p_plane2_post * tau2
        # --- Optional: constant‑shear constraint ---
        # Note: this encourages equal shear-traction magnitudes |τ| across events.
        # Use with care: under a uniform stress tensor, |τ| generally varies with fault orientation.
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
                else:
                    tau0 = pm.Normal("tau0", mu=0.5, sigma=float(shear_target_sigma))

                resid = (tau_mag - tau0) / ss
                pen = 0.5 * pt.sum(resid * resid)
                pm.Potential("shear_const_penalty", -sw * pen)

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
        if sampler_name in {"nutpie", "nuts-nutpie"}:
            idata = pm.sample(nuts_sampler="nutpie", nuts_sampler_kwargs=nk, **sampler_kwargs)
        elif sampler_name in {"numpyro", "jax"}:
            # NumPyro backend runs on CPU here; chain_method can be "vectorized"/"parallel"/"sequential"
            sampler_kwargs["cores"] = 1
            idata = pm.sample(nuts_sampler="numpyro", nuts_sampler_kwargs=nk, **sampler_kwargs)
        else:
            # Default PyMC NUTS
            idata = pm.sample(**sampler_kwargs)

    # --- Post-processing (mirror SMC outputs) ---
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
        Sigma_i = Sigma_samples[:, :, i]
        norm_i = np.max(np.abs(np.linalg.eigvalsh(Sigma_i)))
        Sigma_i = Sigma_i / (norm_i if norm_i > 0 else 1.0)
        ps_i, pd_i = stress_tensor_eigendecomposition(Sigma_i)
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
        "boot_principal_stresses": boot_ps,
        "boot_principal_directions": boot_pd,
        "R_bootstrap": R_samples,
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

    if return_plane_probabilities:
        if iterative_plane_selection:
            results["plane_probabilities"] = iterative_info.get("plane_probabilities")
            results["plane_selection_map"] = iterative_info.get("plane_map")
        else:
            if "p_plane2_post" not in idata.posterior:
                raise RuntimeError("Internal error: p_plane2_post missing from posterior")
            p2_s = idata.posterior["p_plane2_post"].stack(s=("chain", "draw")).values  # (N, S)
            plane_2_prob = np.mean(p2_s, axis=1)
            plane_1_prob = 1.0 - plane_2_prob
            plane_probabilities = np.stack([plane_1_prob, plane_2_prob], axis=1)
            plane_selection_map = (plane_2_prob > 0.5).astype(int)
            results["plane_probabilities"] = plane_probabilities
            results["plane_selection_map"] = plane_selection_map

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
    else:
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
        else:
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

    if tau0_samples is not None and tau0_samples.size > 0:
        results["tau0_median"] = float(np.median(tau0_samples))
        results["tau0_mean"] = float(np.mean(tau0_samples))
        results["tau0_std"] = float(np.std(tau0_samples))
        results["tau0_CI95"] = (
            float(np.quantile(tau0_samples, 0.025)),
            float(np.quantile(tau0_samples, 0.975)),
        )

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
