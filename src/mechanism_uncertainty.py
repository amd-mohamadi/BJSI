"""Local SDR measurement errors shared by both nodal planes of a mechanism."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def prepare_mechanism_errors(strike, dip, rake, strike_sigma, dip_sigma, rake_sigma):
    """Validate scalar/per-event standard deviations (degrees) for input plane 1."""
    angles = np.column_stack([strike, dip, rake]).astype(float)
    if not np.all(np.isfinite(angles)):
        raise ValueError("Mechanism angles must be finite")
    if np.any((angles[:, 1] < 0) | (angles[:, 1] > 90)):
        raise ValueError("Observed dip must be between 0 and 90 degrees")
    errors = []
    for name, value in zip(
        ("strike_sigma_deg", "dip_sigma_deg", "rake_sigma_deg"),
        (strike_sigma, dip_sigma, rake_sigma),
    ):
        a = np.asarray(value, dtype=float)
        if a.ndim == 0:
            a = np.full(len(angles), float(a))
        if a.shape != (len(angles),) or not np.all(np.isfinite(a)) or np.any(a < 0):
            raise ValueError(f"{name} must be a finite nonnegative scalar or array of shape (N,)")
        errors.append(a)
    return angles, np.column_stack(errors)


def validate_fixed_planes(value, n_events, iterative):
    if value is None:
        return None
    if iterative:
        raise ValueError("fixed_plane_indices and iterative_plane_selection are mutually exclusive")
    a = np.asarray(value)
    if a.shape != (n_events,) or not np.all(np.isin(a, [0, 1])):
        raise ValueError("fixed_plane_indices must have shape (N,) with 0 for plane 1 or 1 for plane 2")
    return a.astype(int)


def latent_mechanism_vectors(angles, errors, n1, s1, n2, s2):
    """Perturb input-plane-1 SDR once; derive the auxiliary plane exactly.

    Errors are independent local Gaussian offsets in degrees. Trigonometric
    conversion handles angle wrapping and crossings of horizontal/vertical
    dip without clipping or truncating the measurement distribution. Canonical
    posterior angles have strike in [0,360), dip in [0,90], rake in [-180,180].
    The uncertainty is tied to the supplied plane-1 SDR coordinate chart; it is
    not a rotation-invariant or correlated focal-mechanism error model.
    """
    active = np.flatnonzero(errors.ravel() > 0)
    if not len(active):
        # Exact legacy behavior, including supplied plane-2 rounding.
        return n1, s1, n2, s2
    z = pm.Normal("mechanism_error_z", 0.0, 1.0, shape=len(active))
    delta = pt.zeros(angles.size)
    delta = pt.set_subtensor(delta[active], z * errors.ravel()[active])
    a = pt.as_tensor_variable(angles) + delta.reshape(angles.shape)
    strike, dip, rake = (a[:, j] * (np.pi / 180.0) for j in range(3))
    n = pt.stack([
        -pt.sin(dip) * pt.sin(strike),
        -pt.sin(dip) * pt.cos(strike), pt.cos(dip),
    ], axis=1)
    s = pt.stack([
        pt.cos(rake) * pt.cos(strike) + pt.sin(rake) * pt.cos(dip) * pt.sin(strike),
        -pt.cos(rake) * pt.sin(strike) + pt.sin(rake) * pt.cos(dip) * pt.cos(strike),
        pt.sin(rake) * pt.sin(dip),
    ], axis=1)
    # Match the nominal auxiliary-plane sign; simultaneous n,s sign changes
    # preserve the double couple, shear misfit, and unsigned instability.
    aux_sign = np.where(np.sum(n2 * s1, axis=1) >= 0, 1.0, -1.0)[:, None]
    n_aux, s_aux = aux_sign * s, aux_sign * n

    sign = pt.where(n[:, 2] >= 0, 1.0, -1.0)[:, None]
    nc, sc = sign * n, sign * s
    phi = pt.arctan2(-nc[:, 0], -nc[:, 1])
    d = pt.arccos(pt.clip(nc[:, 2], -1.0, 1.0))
    along_strike = sc[:, 0] * pt.cos(phi) - sc[:, 1] * pt.sin(phi)
    along_dip = (sc[:, 0] * pt.cos(d) * pt.sin(phi)
                 + sc[:, 1] * pt.cos(d) * pt.cos(phi) + sc[:, 2] * pt.sin(d))
    canonical = pt.stack([
        (phi * (180.0 / np.pi)) % 360.0,
        d * (180.0 / np.pi),
        pt.arctan2(along_dip, along_strike) * (180.0 / np.pi),
    ], axis=1)
    pm.Deterministic("mechanism_angles_deg", canonical)
    return n, s, n_aux, s_aux
