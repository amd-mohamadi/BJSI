"""Tests for the fault-population likelihood and its normalizer.

Run with ``pytest tests/test_fault_population.py`` from the repository root.
The tests build small models only; none of them samples a posterior.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTENSOR_FLAGS", "compiledir=/tmp/bjsi_pytensor_tests")

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from scipy.special import i0e, expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fault_population as fp  # noqa: E402
from bjsi import _build_joint_model, normal_slip_vectors_batch, _slip_direction_logp  # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def synthetic_catalog(n_events=40, seed=3):
    """Random mechanisms, adequate for structural tests of the log density."""
    rng = np.random.default_rng(seed)
    strike1 = rng.uniform(0, 360, n_events)
    dip1 = rng.uniform(20, 89, n_events)
    rake1 = rng.uniform(-180, 180, n_events)
    n1, s1 = normal_slip_vectors_batch(strike1, dip1, rake1)
    n2, s2 = s1.copy(), n1.copy()
    return n1, s1, n2, s2


def build(fault_population=None, **kwargs):
    n1, s1, n2, s2 = synthetic_catalog(kwargs.pop("n_events", 40))
    base = dict(
        n1=n1, n2=n2, s1=s1, s2=s2,
        q_prior_mu_arr=None, q_prior_sigma=1.0, R_prior_mu_val=None, R_prior_sigma=2.0,
        infer_friction=True, infer_friction_method="sample", friction_fixed=None,
        friction_prior_params=(3.0, 3.0), friction_range=(0.2, 1.0),
        iterative_plane_selection=False, iterative_info={}, event_weights=None,
        weighted_likelihood=False, likelihood_weight_mode="event", tau_weight_exponent=0.0,
        normalize_tau_weights=True, tau_weight_clip=None,
        slip_likelihood_name="von_mises_fisher", slip_misfit_sigma=0.35, slip_vmf_kappa_val=8.0,
        instability_beta=10.0, selection_beta=10.0, signed_instability=False,
        clustering_prior_strength=0.0, plane2_prior_probs_arr=None, plane_prior_strength=0.0,
        enforce_constant_shear=False, shear_weight=0.0, shear_sigma=0.2,
        shear_center="mean", shear_target=None, shear_target_sigma=0.3,
        fault_population=fault_population,
    )
    base.update(kwargs)
    return _build_joint_model(**base)[0]


def eval_vars(model, names, point=None):
    """Evaluate named deterministics and potentials at a point."""
    outs = model.replace_rvs_by_values(
        [model[name] if isinstance(name, str) else name for name in names]
    )
    fn = pytensor.function(model.value_vars, outs, on_unused_input="ignore")
    ip = model.initial_point() if point is None else point
    return fn(*[ip[v.name] for v in model.value_vars])


def at_point(model, **overrides):
    ip = model.initial_point()
    ip["q_raw"] = np.array([0.3, 0.5, -0.2, 0.7])
    ip.update(overrides)
    return ip


# ----------------------------------------------------------------------------
# item 6: the slip likelihood is a density on the great circle
# ----------------------------------------------------------------------------
def test_vmf_normalizer_is_circular():
    """exp(kappa cos) / (2 pi I0(kappa)) integrates to one over the rake."""
    kappa = 8.0
    n = np.array([[0.0, 0.0, 1.0]])
    angles = np.linspace(0.0, 2 * np.pi, 20001)
    s_obs = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])
    s_pred = np.tile(np.array([1.0, 0.0, 0.0]), (angles.size, 1))
    logp = _slip_direction_logp(
        pt.as_tensor_variable(s_obs), pt.as_tensor_variable(s_pred),
        family="von_mises_fisher", sigma=0.35, vmf_kappa=kappa, weight=1.0,
    ).eval()
    mass = np.trapezoid(np.exp(logp), angles)
    assert abs(mass - 1.0) < 1e-6, mass
    assert n.shape == (1, 3)


def test_vmf_matches_closed_form():
    kappa = 5.0
    s_obs = np.array([[1.0, 0.0, 0.0]])
    s_pred = np.array([[np.cos(0.4), np.sin(0.4), 0.0]])
    got = float(_slip_direction_logp(
        pt.as_tensor_variable(s_obs), pt.as_tensor_variable(s_pred),
        family="von_mises_fisher", sigma=0.35, vmf_kappa=kappa, weight=1.0,
    ).eval()[0])
    want = kappa * np.cos(0.4) - np.log(2 * np.pi) - (kappa + np.log(i0e(kappa)))
    assert abs(got - want) < 1e-10


def test_vmf_normalizer_is_circular_under_weighting():
    """The weighted likelihood is still a density, which is what p > 0 needs.

    With shear-traction weighting the effective concentration varies with the
    stress, so an incorrect normalizer no longer cancels between stress states.
    """
    angles = np.linspace(0.0, 2 * np.pi, 20001)
    s_obs = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])
    s_pred = np.tile(np.array([1.0, 0.0, 0.0]), (angles.size, 1))
    for weight in (0.25, 1.0, 4.0):
        logp = _slip_direction_logp(
            pt.as_tensor_variable(s_obs), pt.as_tensor_variable(s_pred),
            family="von_mises_fisher", sigma=0.35, vmf_kappa=8.0, weight=weight,
        ).eval()
        assert abs(np.trapezoid(np.exp(logp), angles) - 1.0) < 1e-6, weight


def test_vmf_stable_at_large_concentration():
    value = float(_slip_direction_logp(
        pt.as_tensor_variable(np.array([[1.0, 0.0, 0.0]])),
        pt.as_tensor_variable(np.array([[1.0, 0.0, 0.0]])),
        family="von_mises_fisher", sigma=0.35, vmf_kappa=50.0, weight=20.0,
    ).eval()[0])
    assert np.isfinite(value)


# ----------------------------------------------------------------------------
# item 1: the population reproduces the historical weights and plane posterior
# ----------------------------------------------------------------------------
def test_legacy_is_unchanged_by_default():
    model = build(None)
    names = [v.name for v in model.free_RVs]
    assert names == ["q_raw", "R", "mu_raw"]
    assert "log_Z_population" not in [d.name for d in model.deterministics]


def test_exp_population_matches_legacy_plane_prior():
    """g = exp(beta I) gives exactly the logistic weight of the legacy model."""
    legacy = build(None)
    pop = build({"family": "exp", "beta": 10.0, "normalize": False})
    p_legacy = eval_vars(legacy, ["p_plane2"], at_point(legacy))[0]
    p_pop = eval_vars(pop, ["p_plane2"], at_point(pop))[0]
    assert np.max(np.abs(p_legacy - p_pop)) < 1e-9


def test_exp_population_adds_the_orientation_evidence():
    """The population objective exceeds the legacy one by the orientation evidence.

    With g = exp(beta I) the legacy mixture equals the population mixture divided
    by exp(beta I_1) + exp(beta I_2). That factor is the density of the observed
    fault orientations under the population, and it depends on the stress, which
    is precisely the term the legacy objective omits.
    """
    n1, _, n2, _ = synthetic_catalog(40)
    legacy = build(None)
    pop = build({"family": "exp", "beta": 10.0, "normalize": False})
    for q in ([0.3, 0.5, -0.2, 0.7], [0.1, -0.6, 0.4, 0.2]):
        for r_logit, mu_logit in ((0.2, -0.5), (-0.8, 1.1)):
            kw = dict(q_raw=np.array(q), R_logodds__=r_logit, mu_raw_logodds__=mu_logit)
            a = float(eval_vars(legacy, [legacy.potentials[0]], at_point(legacy, **kw))[0])
            b = float(eval_vars(pop, [pop.potentials[0]], at_point(pop, **kw))[0])
            Sigma, mu = eval_vars(pop, ["Sigma", "mu"], at_point(pop, **kw))
            evidence = np.sum(np.logaddexp(
                10.0 * instability_from_sigma(n1, Sigma, float(mu)),
                10.0 * instability_from_sigma(n2, Sigma, float(mu)),
            ))
            assert abs((b - a) - evidence) < 1e-8, (b - a, evidence)


def instability_from_sigma(n, Sigma, mu):
    """Unsigned normalized instability, evaluated directly from a stress matrix."""
    t = n @ Sigma
    sigma_n = np.einsum("ij,ij->i", n, t)
    tau = np.linalg.norm(t - sigma_n[:, None] * n, axis=1)
    return (tau + mu * (1.0 + sigma_n)) / (np.sqrt(1.0 + mu * mu) + mu)


# ----------------------------------------------------------------------------
# item 2: the normalizer
# ----------------------------------------------------------------------------
def test_instability_numpy_matches_graph():
    rng = np.random.default_rng(0)
    n = rng.normal(size=(50, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    R, mu = 0.42, 0.63
    want = fp.instability_numpy(n, R, mu)
    Sigma = pt.as_tensor_variable(np.diag([-1.0, 2 * R - 1.0, 1.0]))
    got = fp.instability_pt(Sigma, pt.as_tensor_variable(n), mu).eval()
    assert np.max(np.abs(want - got)) < 1e-10


def test_normalizer_table_matches_direct_estimate():
    spec = fp.resolve_population_spec(
        {"family": "exp", "beta": 10.0, "table": {"R_points": 21, "mu_points": 17, "power": 14}},
        selection_beta=10.0, friction_range=(0.2, 1.0),
    )
    table = fp.normalizer_table(spec, verbose=False)
    normals = fp.sobol_unit_normals(16, 99)
    for R, mu in ((0.30, 0.40), (0.70, 0.80)):
        direct = np.log(np.mean(np.exp(10.0 * fp.instability_numpy(normals, R, mu))))
        got = float(fp.interp_normalizer_pt(table, pt.as_tensor_variable(R),
                                            pt.as_tensor_variable(mu)).eval())
        assert abs(direct - got) < 0.02, (R, mu, direct, got)


def test_ramp_normalizer_makes_the_density_integrate_to_one():
    """Mean of g(I)/Z over uniform normals is one, so p integrates to one."""
    spec = fp.resolve_population_spec(
        {"family": "ramp", "imin": 0.8, "table": {"R_points": 21, "mu_points": 17, "power": 14}},
        selection_beta=10.0, friction_range=(0.2, 1.0),
    )
    table = fp.normalizer_table(spec, verbose=False)
    normals = fp.sobol_unit_normals(16, 7)
    for R, mu in ((0.25, 0.30), (0.60, 0.90)):
        I = fp.instability_numpy(normals, R, mu)
        g = np.logaddexp(0.0, 100.0 * (I - 0.8)) / (100.0 * 0.2)
        logZ = float(fp.interp_normalizer_pt(table, pt.as_tensor_variable(R),
                                             pt.as_tensor_variable(mu)).eval())
        assert abs(np.mean(g) / np.exp(logZ) - 1.0) < 0.03


def test_three_dimensional_table_interpolates_imin():
    spec = fp.resolve_population_spec(
        {"family": "ramp", "imin": "infer",
         "table": {"R_points": 21, "mu_points": 17, "imin_points": 20, "power": 14}},
        selection_beta=10.0, friction_range=(0.2, 1.0),
    )
    table = fp.normalizer_table(spec, verbose=False)
    assert table["logZ"].ndim == 3
    normals = fp.sobol_unit_normals(16, 11)
    R, mu, imin = 0.55, 0.45, 0.62
    I = fp.instability_numpy(normals, R, mu)
    g = np.logaddexp(0.0, 100.0 * (I - imin)) / (100.0 * (1 - imin))
    got = float(fp.interp_normalizer_pt(table, pt.as_tensor_variable(R),
                                        pt.as_tensor_variable(mu),
                                        pt.as_tensor_variable(imin)).eval())
    assert abs(np.log(np.mean(g)) - got) < 0.05


def test_normalized_model_has_the_potential_and_is_finite():
    model = build({"family": "ramp", "imin": 0.8,
                   "table": {"R_points": 21, "mu_points": 17, "power": 14}})
    names = [p.name for p in model.potentials]
    assert "population_normalizer" in names
    values = eval_vars(model, model.potentials, at_point(model))
    assert all(np.isfinite(float(v)) for v in values)


def test_normalizer_removes_the_friction_trend():
    """The normalized objective must not reward mu through the total mass alone.

    For a catalog drawn uniformly at random the expected per-event log density is
    maximal at no particular mu. The unnormalized objective instead increases
    with mu, by construction of Z.
    """
    spec = fp.resolve_population_spec(
        {"family": "exp", "beta": 10.0, "table": {"R_points": 51, "mu_points": 41, "power": 15}},
        selection_beta=10.0, friction_range=(0.2, 1.0),
    )
    table = fp.normalizer_table(spec, verbose=False)
    normals = fp.sobol_unit_normals(15, 5)
    logZ = []
    for mu in (0.2, 0.6, 1.0):
        raw = np.log(np.mean(np.exp(10.0 * fp.instability_numpy(normals, 0.5, mu))))
        got = float(fp.interp_normalizer_pt(table, pt.as_tensor_variable(0.5),
                                            pt.as_tensor_variable(mu)).eval())
        logZ.append((raw, got))
    assert logZ[2][0] - logZ[0][0] > 0.1          # the mass does grow with mu
    for raw, got in logZ:                          # and the table tracks it
        assert abs(raw - got) < 0.02


# ----------------------------------------------------------------------------
# items 3 and 4: inferred sharpness, uniform and fabric components
# ----------------------------------------------------------------------------
def test_inferred_imin_adds_one_parameter():
    model = build({"family": "ramp", "imin": "infer",
                   "table": {"R_points": 21, "mu_points": 17, "imin_points": 20, "power": 14}})
    assert "I_min" in [v.name for v in model.free_RVs]


def test_uniform_mixture_adds_weight():
    model = build({"family": "ramp", "imin": 0.8, "mix_uniform": True,
                   "table": {"R_points": 21, "mu_points": 17, "power": 14}})
    assert "w_population" in [v.name for v in model.free_RVs]
    values = eval_vars(model, model.potentials, at_point(model))
    assert all(np.isfinite(float(v)) for v in values)


def test_fabric_adds_watson_parameters_and_is_finite():
    model = build({"family": "ramp", "imin": "infer", "fabric_K": 2,
                   "table": {"R_points": 21, "mu_points": 17, "imin_points": 20, "power": 14}})
    names = [v.name for v in model.free_RVs]
    for expected in ("w_population", "fabric_axis_raw", "fabric_kappa", "fabric_pi"):
        assert expected in names
    values = eval_vars(model, model.potentials, at_point(model))
    assert all(np.isfinite(float(v)) for v in values)


def test_watson_mixture_is_normalized():
    """Each Watson component has unit mean under the uniform measure."""
    normals = fp.sobol_unit_normals(16, 4)
    axes = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    for kappa in (3.0, 30.0, 90.0):
        log_h = fp.log_watson_mixture_pt(
            pt.as_tensor_variable(normals), pt.as_tensor_variable(axes),
            pt.as_tensor_variable(np.array([kappa, kappa])),
            pt.as_tensor_variable(np.log(np.array([0.5, 0.5]))),
        ).eval()
        assert abs(np.mean(np.exp(log_h)) - 1.0) < 2e-3, kappa


def test_fabric_and_uniform_are_exclusive():
    with pytest.raises(ValueError):
        build({"family": "ramp", "mix_uniform": True, "fabric_K": 2})


def test_population_rejects_incompatible_options():
    with pytest.raises(ValueError):
        build({"family": "exp"}, signed_instability=True)
    with pytest.raises(ValueError):
        build({"family": "exp"}, clustering_prior_strength=1.0)
    with pytest.raises(ValueError):
        build({"family": "exp"}, fixed_plane_indices=np.zeros(40, dtype=int))


def test_spec_validation():
    with pytest.raises(ValueError):
        fp.resolve_population_spec("nonsense", selection_beta=10.0, friction_range=(0.2, 1.0))
    with pytest.raises(ValueError):
        fp.resolve_population_spec({"family": "ramp", "imin": 1.5},
                                   selection_beta=10.0, friction_range=(0.2, 1.0))
    spec = fp.resolve_population_spec(None, selection_beta=10.0, friction_range=(0.2, 1.0))
    assert spec["family"] == fp.LEGACY and not spec["normalize"]


# ----------------------------------------------------------------------------
# item 5: basin diagnostics and per-basin evidence
# ----------------------------------------------------------------------------
def test_basin_report_separates_and_merges_chains():
    import basins

    class FakePosterior(dict):
        @property
        def sizes(self):
            shape = self["Sigma"].values.shape
            return {"chain": shape[0], "draw": shape[1]}

        def __contains__(self, key):
            return dict.__contains__(self, key)

    class Array:
        def __init__(self, values):
            self.values = values

    def stress(shmax_deg, R):
        az = np.radians(shmax_deg)
        e1 = np.array([np.cos(az), -np.sin(az), 0.0])
        e3 = np.array([np.sin(az), np.cos(az), 0.0])
        e2 = np.array([0.0, 0.0, 1.0])
        return (-1 * np.outer(e1, e1) + (2 * R - 1) * np.outer(e2, e2) + np.outer(e3, e3))

    draws = 50
    configs = [(80.0, 0.2), (80.4, 0.22), (101.0, 0.5)]
    Sigma = np.stack([np.tile(stress(s, r), (draws, 1, 1)) for s, r in configs])
    R = np.stack([np.full(draws, r) for _, r in configs])
    idata = type("I", (), {})()
    idata.posterior = FakePosterior(Sigma=Array(Sigma), R=Array(R))
    report = basins.basin_report(idata)
    assert report["n_basins"] == 2 and report["multimodal"]
    assert sorted(len(b["chains"]) for b in report["basins"]) == [1, 2]


def test_bridge_evidence_recovers_a_known_normalizer():
    """A model whose evidence is known analytically is recovered to < 0.1 nats."""
    import pymc as pm
    import basins

    sigma, n_obs = 2.0, 5
    rng = np.random.default_rng(1)
    y = rng.normal(1.0, 1.0, n_obs)
    with pm.Model() as model:
        x = pm.Normal("x", mu=0.0, sigma=sigma)
        pm.Normal("y", mu=x, sigma=1.0, observed=y)
        idata = pm.sample(draws=2000, tune=1000, chains=1, cores=1,
                          random_seed=5, progressbar=False)
    # Analytic: integral of N(x|0,sigma) prod N(y_i|x,1) dx
    var_post = 1.0 / (1.0 / sigma ** 2 + n_obs)
    mean_post = var_post * y.sum()
    want = (-0.5 * n_obs * np.log(2 * np.pi) - 0.5 * np.sum(y ** 2)
            + 0.5 * np.log(var_post / sigma ** 2) + 0.5 * mean_post ** 2 / var_post)
    got = basins.bridge_evidence(model, idata, 0, seed=2)["log_evidence"]
    assert abs(got - want) < 0.1, (got, want)


# ----------------------------------------------------------------------------
# Bingham fabric
# ----------------------------------------------------------------------------
def test_bingham_components_are_normalized():
    """Each Bingham component has unit mean under the uniform measure."""
    normals = fp.sobol_unit_normals(16, 4)
    axes = np.stack([np.eye(3), np.eye(3)])
    for k1, k2 in ((0.0, 0.0), (5.0, 5.0), (30.0, 0.0), (60.0, 20.0), (200.0, 3.0)):
        log_h = fp.log_bingham_mixture_pt(
            pt.as_tensor_variable(normals), pt.as_tensor_variable(axes),
            pt.as_tensor_variable(np.array([[k1, k2], [k1, k2]])),
            pt.as_tensor_variable(np.log(np.array([0.5, 0.5]))),
        ).eval()
        assert abs(np.mean(np.exp(log_h)) - 1.0) < 3e-3, (k1, k2)


def test_bingham_reduces_to_watson_when_concentrations_are_equal():
    normals = fp.sobol_unit_normals(14, 6)
    axis = np.array([0.0, 0.0, 1.0])
    kappa = 25.0
    rotation = np.eye(3)
    bingham = fp.log_bingham_mixture_pt(
        pt.as_tensor_variable(normals), pt.as_tensor_variable(rotation[None]),
        pt.as_tensor_variable(np.array([[kappa, kappa]])), pt.as_tensor_variable(np.zeros(1)),
    ).eval()
    watson = fp.log_watson_mixture_pt(
        pt.as_tensor_variable(normals), pt.as_tensor_variable(axis[None]),
        pt.as_tensor_variable(np.array([kappa])), pt.as_tensor_variable(np.zeros(1)),
    ).eval()
    assert np.max(np.abs(bingham - watson)) < 1e-6


def test_bingham_fabric_model_builds_and_is_finite():
    model = build({"family": "ramp", "imin": "infer", "fabric_K": 2,
                   "fabric_family": "bingham", "fabric_pi_alpha": 0.5,
                   "table": {"R_points": 21, "mu_points": 17, "imin_points": 20, "power": 14}})
    names = [v.name for v in model.free_RVs]
    for expected in ("w_population", "fabric_quat_raw", "fabric_kappa", "fabric_pi"):
        assert expected in names
    values = eval_vars(model, model.potentials, at_point(model))
    assert all(np.isfinite(float(v)) for v in values)
