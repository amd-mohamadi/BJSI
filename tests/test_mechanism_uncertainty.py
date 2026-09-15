import sys
from pathlib import Path

import numpy as np
import pymc as pm
import pytensor
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bjsi import normal_slip_vectors_batch, _build_joint_model, _slip_direction_logp
from mechanism_uncertainty import prepare_mechanism_errors, latent_mechanism_vectors, validate_fixed_planes
from utils_stress import aux_plane


def example():
    angles = np.array([[359., 89., 179.], [1., 2., -179.], [35., 50., 70.]])
    auxiliary = np.array([aux_plane(*a) for a in angles])
    n1, s1 = normal_slip_vectors_batch(*angles.T)
    n2, s2 = normal_slip_vectors_batch(*auxiliary.T)
    return angles, n1, s1, n2, s2


def build(beta=1., errors=0., fixed=None, **kw):
    a, n1, s1, n2, s2 = example()
    angles, sigmas = prepare_mechanism_errors(*a.T, errors, errors, errors)
    options = dict(
        n1=n1, n2=n2, s1=s1, s2=s2,
        mechanism_angles=angles, mechanism_errors=sigmas, fixed_plane_indices=fixed,
        q_prior_mu_arr=None, q_prior_sigma=1., R_prior_mu_val=None, R_prior_sigma=2.,
        infer_friction=True, infer_friction_method="sample", friction_fixed=None,
        friction_prior_params=(3., 3.), friction_range=(.2, 1.),
        iterative_plane_selection=False, iterative_info={}, event_weights=None,
        weighted_likelihood=False, likelihood_weight_mode="event", tau_weight_exponent=0.,
        normalize_tau_weights=True, tau_weight_clip=None,
        slip_likelihood_name="von_mises_fisher", slip_misfit_sigma=.35, slip_vmf_kappa_val=8.,
        instability_beta=beta, selection_beta=beta, signed_instability=False,
        clustering_prior_strength=0., plane2_prior_probs_arr=None, plane_prior_strength=0.,
        enforce_constant_shear=False, shear_weight=0., shear_sigma=.2,
        shear_center="mean", shear_target=None, shear_target_sigma=.3,
    )
    options.update(kw)
    return _build_joint_model(**options)[0]


def point(model):
    p = model.initial_point()
    p["q_raw"] = np.array([1., .2, -.3, .4])
    return p


def test_per_event_validation_and_zero_errors():
    a, n1, s1, n2, s2 = example()
    angles, sigmas = prepare_mechanism_errors(*a.T, [0., 2., 3.], 5., 0.)
    np.testing.assert_equal(sigmas[:, 0], [0., 2., 3.])
    with pytest.raises(ValueError):
        prepare_mechanism_errors(*a.T, [1., 2.], 5., 5.)
    for bad in [-1., np.nan, np.inf]:
        with pytest.raises(ValueError):
            prepare_mechanism_errors(*a.T, bad, 5., 5.)
    with pm.Model() as m:
        out = latent_mechanism_vectors(angles, np.zeros_like(sigmas), n1, s1, n2, s2)
    assert not m.free_RVs
    for actual, expected in zip(out, (n1, s1, n2, s2)):
        np.testing.assert_array_equal(actual, expected)


def test_shared_double_couple_and_angle_boundaries():
    a, n1, s1, n2, s2 = example()
    errors = np.array([[5., 5., 5.], [0., 5., 0.], [5., 0., 5.]])
    with pm.Model() as m:
        out = latent_mechanism_vectors(a, errors, n1, s1, n2, s2)
    f = pytensor.function([m["mechanism_error_z"]], [*out, m["mechanism_angles_deg"]])
    # Cross strike/rake wraps, vertical dip, and horizontal dip.
    n, s, na, sa, canonical = f(np.array([2., 2., 2., -2., 1., 1.]))
    for v in (n, s, na, sa):
        np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1., atol=1e-12)
    np.testing.assert_allclose(np.sum(n*s, axis=1), 0., atol=1e-12)
    mt = n[:, :, None]*s[:, None, :] + s[:, :, None]*n[:, None, :]
    mt_aux = na[:, :, None]*sa[:, None, :] + sa[:, :, None]*na[:, None, :]
    np.testing.assert_allclose(mt, mt_aux, atol=1e-12)
    nc, sc = normal_slip_vectors_batch(*canonical.T)
    np.testing.assert_allclose(mt, nc[:, :, None]*sc[:, None, :] + sc[:, :, None]*nc[:, None, :], atol=1e-12)
    assert np.all((canonical[:, 1] >= 0.) & (canonical[:, 1] <= 90.))
    assert np.all((canonical[:, 0] >= 0.) & (canonical[:, 0] < 360.))


@pytest.mark.parametrize("p_exponent", [0., 2.])
def test_fixed_planes_bypass_beta_and_match_directional_likelihood(p_exponent):
    labels = np.array([0, 1, 0])
    options = dict(fixed=labels, weighted_likelihood=p_exponent > 0,
                   tau_weight_exponent=p_exponent)
    low, high = build(beta=0., **options), build(beta=100., **options)
    p = point(low)
    np.testing.assert_allclose(low.compile_logp()(p), high.compile_logp()(p))
    sigma_value = low.replace_rvs_by_values([low["Sigma"]])[0]
    sigma = low.compile_fn(sigma_value, inputs=low.value_vars, on_unused_input="ignore")(p)
    _, n1, s1, n2, s2 = example()
    n, s = np.where(labels[:, None], n2, n1), np.where(labels[:, None], s2, s1)
    traction = n @ sigma
    shear = traction - (traction*n).sum(axis=1)[:, None]*n
    magnitudes = []
    for normals in (n1, n2):
        t = normals @ sigma
        magnitudes.append(np.linalg.norm(t - (t*normals).sum(axis=1)[:, None]*normals, axis=1))
    event_shear = .5*(magnitudes[0]+magnitudes[1])
    weight = (event_shear/event_shear.mean())**p_exponent
    expected = _slip_direction_logp(s, shear, family="von_mises_fisher", sigma=.35,
                                    vmf_kappa=8., weight=weight).eval().sum()
    actual = low.compile_logp(vars=low.potentials)(p)
    np.testing.assert_allclose(actual, expected)
    assert "mechanism_error_z" not in low.named_vars


def test_uncertainty_enters_likelihood_and_has_finite_gradients():
    m = build(errors=5.)
    p = point(m)
    f = m.compile_logp(vars=m.potentials)
    before = f(p)
    p["mechanism_error_z"] = np.full(9, .5)
    assert abs(f(p)-before) > 1e-5
    assert np.all(np.isfinite(m.compile_dlogp()(p)))
    for bad in [[0, 1], [0, 2, 1], [0, .5, 1]]:
        with pytest.raises(ValueError):
            validate_fixed_planes(bad, 3, False)
    with pytest.raises(ValueError):
        validate_fixed_planes([0, 1, 0], 3, True)


def test_zero_error_model_equals_legacy_graph():
    a = build()
    b = build(mechanism_angles=None, mechanism_errors=None)
    p = point(a)
    np.testing.assert_allclose(a.compile_logp()(p), b.compile_logp()(p), atol=1e-12)
