"""Adaptive tempered SMC (BlackJAX) for the BJSI joint model.

The PyMC model built by :func:`bjsi._build_joint_model` is compiled to JAX by
PyTensor, so the target sampled here is the one NUTS samples, with every model
feature (fault populations, fabrics, mechanism errors, tau weights) carried
over unchanged.  The prior and the likelihood are separated by the names of the
model's potentials: the ``*_haar`` potentials correct the Gaussian chart priors
of the rotations and belong to the prior; every other potential is a data term
and is tempered.

Why SMC.  Fault-population posteriors are multimodal in the shape ratio and the
plane assignment, and independent NUTS chains land in different basins with no
way to weigh them.  A tempered particle population moves from the prior to the
posterior through a sequence of intermediate targets, populates every basin
and weighs the basins by their mass; the tempering path also gives the log
evidence, which model comparisons between populations and fabric sizes need.

The sampler is written from BlackJAX's building blocks rather than its
``adaptive_tempered_smc`` driver so that (i) the tempering increment is solved
on the particles that are actually weighted, (ii) the mutation kernel is tuned
from the particle cloud at every step, (iii) the population is mutated at the
final target after the last increment, and (iv) the increment of the log
evidence is recorded.  Mutation is HMC with a diagonal mass matrix measured
on the cloud (the default; it is what keeps the population at the posterior
density on the fabric models), NUTS, or a random-walk Metropolis kernel with
the particle covariance as proposal.  The number of moves per tempering step
is ``mcmc_steps``, or optionally adaptive (``move_tol``): the population is
moved until a move no longer carries the particles appreciably further from
their ancestors, with ``mcmc_steps`` as the cap.  The waste-free
scheme of Dau and Chopin (2022) keeps every intermediate MCMC state so that
``draws`` particles cost ``draws / mcmc_steps`` chains of ``mcmc_steps`` moves;
it is used for the tempering increments only, and the rounds at the posterior
move every particle.

Speed.  The likelihood is a sum over events, so ``vmap`` over particles makes
arrays of particles x events (tens of MB for thousands of events) that stream
through memory at every operation.  The particles are instead split across
host devices (one per CPU thread, set through ``XLA_FLAGS`` before JAX starts)
and each device evaluates its particles one at a time with the per-event
arrays in cache; on 1720 events this is 8x faster than ``vmap``.
"""
from __future__ import annotations

import math
import os
from functools import partial
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import arviz as az
import pymc as pm
import pytensor
import pytensor.tensor as pt

__all__ = ["sample_smc", "PRIOR_POTENTIAL_SUFFIX"]

# Potentials whose name ends with this are part of the prior (Haar corrections
# of the rotation charts); every other potential is a likelihood term.
PRIOR_POTENTIAL_SUFFIX = "_haar"
_GNOMONIC_SUFFIX = "_gnomonic"

KERNELS = ("rmh", "hmc", "nuts")


# ---------------------------------------------------------------------------
# JAX setup
_DEVICE_FLAG = "xla_force_host_platform_device_count"


def _import_jax():
    # One host device per CPU thread; only effective if JAX has not yet
    # started its CPU backend in this process (see _host_mesh).
    flags = os.environ.get("XLA_FLAGS", "")
    if _DEVICE_FLAG not in flags:
        os.environ["XLA_FLAGS"] = f"{flags} --{_DEVICE_FLAG}={os.cpu_count() or 1}".strip()
    import jax
    import jax.numpy as jnp
    import blackjax

    jax.config.update("jax_enable_x64", True)
    _register_jax_ops(jnp)
    return jax, jnp, blackjax


_REGISTERED = False


def _register_jax_ops(jnp):
    """JAX lowering of PyTensor ops that its default dispatch leaves to TFP."""
    global _REGISTERED
    if _REGISTERED:
        return
    from pytensor.link.jax.dispatch import jax_funcify
    from pytensor.scalar.math import Ive

    @jax_funcify.register(Ive)
    def _jax_funcify_ive(op, **kwargs):
        from jax.scipy.special import i0e, i1e

        def ive(v, x):
            # The fabric normalizer only needs order 0; order 1 is kept for safety.
            return jnp.where(v == 0, i0e(x), i1e(x))

        return ive

    _REGISTERED = True


# ---------------------------------------------------------------------------
# Model compilation
def _split_potentials(model):
    prior_pots = [p for p in model.potentials if str(p.name).endswith(PRIOR_POTENTIAL_SUFFIX)]
    lik_pots = [p for p in model.potentials if not str(p.name).endswith(PRIOR_POTENTIAL_SUFFIX)]
    return prior_pots, lik_pots


def _compile_model(model, jax, jnp):
    """Return flat-vector JAX functions of the unconstrained parameters.

    ``log_prior`` includes the transform Jacobians so that the particles live
    on the unconstrained space of the value variables, as they do for NUTS.
    ``outputs`` maps a particle to every unobserved variable of the model
    (constrained free variables and deterministics) for the ``InferenceData``.
    """
    from jax.flatten_util import ravel_pytree
    from pymc.sampling.jax import get_jaxified_graph
    from pymc.util import get_default_varnames

    prior_pots, lik_pots = _split_potentials(model)
    if model.observed_RVs:
        raise NotImplementedError("bjsi_smc expects the likelihood as potentials")
    if not lik_pots:
        raise ValueError("The model has no likelihood potential to temper")

    log_prior_expr = model.logp(vars=model.free_RVs + prior_pots, jacobian=True)
    log_lik_expr = model.logp(vars=lik_pots, jacobian=False)
    value_vars = list(model.value_vars)
    f_prior = get_jaxified_graph(inputs=value_vars, outputs=[log_prior_expr])
    f_lik = get_jaxified_graph(inputs=value_vars, outputs=[log_lik_expr])

    out_vars = list(get_default_varnames(model.unobserved_value_vars, include_transformed=False))
    f_out = get_jaxified_graph(inputs=value_vars, outputs=out_vars)
    out_names = [v.name for v in out_vars]

    template = [jnp.zeros(v.type.shape, dtype=jnp.float64) for v in value_vars]
    _, unravel = ravel_pytree(template)
    dim = int(sum(int(np.prod(v.type.shape)) for v in value_vars))

    def log_prior(z):
        return f_prior(*unravel(z))[0]

    def log_lik(z):
        return f_lik(*unravel(z))[0]

    def outputs(z):
        return dict(zip(out_names, f_out(*unravel(z))))

    return {
        "log_prior": log_prior,
        "log_lik": log_lik,
        "outputs": outputs,
        "dim": dim,
        "value_vars": value_vars,
        "out_names": out_names,
    }


def _prior_particles(model, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` particles from the prior on the unconstrained space.

    The rotation charts carry a Haar potential on top of a Gaussian, so their
    prior is the trivariate Cauchy pull-back of the uniform rotation and is
    drawn by mapping uniform unit quaternions through the chart.  The other
    free variables are drawn from their distributions and mapped through
    their transforms.
    """
    cols = []
    for rv in model.free_RVs:
        value_var = model.rvs_to_values[rv]
        shape = tuple(value_var.type.shape)
        if str(rv.name).endswith(_GNOMONIC_SUFFIX):
            q = rng.normal(size=(n,) + shape[:-1] + (4,))
            q /= np.linalg.norm(q, axis=-1, keepdims=True)
            draws = q[..., 1:] / q[..., :1]
        else:
            draws = np.asarray(pm.draw(rv, draws=n, random_seed=int(rng.integers(2**31))))
            transform = model.rvs_to_transforms.get(rv)
            if transform is not None:
                x = pt.tensor(dtype="float64", shape=(None,) + tuple(rv.type.shape), name="x")
                fwd = pytensor.function([x], transform.forward(x, *rv.owner.inputs))
                draws = np.asarray(fwd(draws.astype("float64")))
        draws = np.asarray(draws, dtype=np.float64).reshape(n, -1)
        if draws.shape[1] != int(np.prod(shape)):
            raise RuntimeError(f"Prior draw of {rv.name} has shape {draws.shape}, expected {shape}")
        cols.append(draws)
    return np.concatenate(cols, axis=1)


# ---------------------------------------------------------------------------
# Tempering schedule
def _log_ess(log_w: np.ndarray) -> float:
    m = np.max(log_w)
    lw = log_w - m
    return float(2.0 * np.log(np.sum(np.exp(lw))) - np.log(np.sum(np.exp(2.0 * lw))))


def _solve_delta(loglik: np.ndarray, lmbda: float, target_ess: float, max_iter: int = 60) -> float:
    """Largest increment of the tempering parameter keeping ESS >= target_ess * N."""
    ll = np.asarray(loglik, dtype=np.float64)
    ll = np.where(np.isfinite(ll), ll, -np.inf)
    n = ll.size
    target = math.log(target_ess * n)
    hi = 1.0 - lmbda
    if hi <= 0.0:
        return 0.0
    if _log_ess(hi * ll) >= target:
        return hi
    lo = 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if _log_ess(mid * ll) >= target:
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# Mutation kernels
def _build_mutation(jax, jnp, blackjax, kernel: str, *, log_prior, log_lik,
                    hmc_integration_steps: int, nuts_max_doublings: int):
    """Return (init_fn, step_fn, param_names) for the chosen kernel.

    ``step_fn(key, state, logdensity_fn, **params)`` follows the BlackJAX SMC
    convention; the parameters are the arrays tuned from the particle cloud.
    """
    from blackjax.mcmc import random_walk as rw

    if kernel == "rmh":
        # build_additive_step adds the Gaussian move to the position; build_rmh
        # alone would take the move itself as the proposed position.
        rmh_kernel = rw.build_additive_step()

        def step_fn(key, state, logdensity_fn, chol):
            return rmh_kernel(key, state, logdensity_fn, rw.normal(chol))

        return rw.init, step_fn, ("chol",)
    if kernel == "hmc":
        hmc_kernel = blackjax.mcmc.hmc.build_kernel()

        def step_fn(key, state, logdensity_fn, step_size, inverse_mass_matrix):
            return hmc_kernel(key, state, logdensity_fn, step_size, inverse_mass_matrix,
                              int(hmc_integration_steps))

        return blackjax.mcmc.hmc.init, step_fn, ("step_size", "inverse_mass_matrix")
    if kernel == "nuts":
        nuts_kernel = blackjax.mcmc.nuts.build_kernel()

        def step_fn(key, state, logdensity_fn, step_size, inverse_mass_matrix):
            return nuts_kernel(key, state, logdensity_fn, step_size, inverse_mass_matrix,
                               max_num_doublings=int(nuts_max_doublings))

        return blackjax.mcmc.nuts.init, step_fn, ("step_size", "inverse_mass_matrix")
    raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")


def _acceptance(info):
    if hasattr(info, "acceptance_rate"):
        return info.acceptance_rate
    return info.is_accepted.astype("float64")


class _KernelTuning:
    """Proposal parameters derived from the particle cloud and the acceptance rate.

    The shape of the proposal (covariance for the random walk, inverse mass
    for HMC/NUTS) is measured on the cloud once per tempering step; its size
    (a multiplier on the 2.38/sqrt(d) random-walk scale, or the step size) is
    adapted from the acceptance rate.
    """

    def __init__(self, kernel: str, dim: int, *, target_accept: float, adaptation_rate: float,
                 hmc_step_size: float, rng: np.random.Generator, jitter: float = 1e-10):
        self.kernel = kernel
        self.dim = dim
        self.target_accept = float(target_accept)
        self.rate = float(adaptation_rate)
        self.jitter = float(jitter)
        self.rng = rng
        self.base_scale = 2.38 / math.sqrt(dim)
        self.scale = 1.0 if kernel == "rmh" else float(hmc_step_size)
        self.bounds = (1e-3, 10.0) if kernel == "rmh" else (1e-5, 5.0)
        self._chol = None
        self._var = None
        self.gradient_particles = 1000

    def measure(self, particles: np.ndarray, weights: np.ndarray,
                gradient_fn: Optional[Callable] = None) -> None:
        """Robust weighted covariance of the cloud, and the local scale for HMC.

        The rotation charts have Cauchy tails under the prior and for weakly
        identified fabric components, so the per-dimension scale is taken from
        the interquartile range and the correlations from a winsorized cloud;
        a few far particles would otherwise set the proposal.

        The marginal width of a chart coordinate is not the width of the mode
        a particle sits in: the symmetric copies of a rotation and the
        exchangeable fabric components spread the cloud over several copies
        of the same mode, each a hundred times narrower than their spread.  An
        HMC mass built from the marginal variance then overshoots in those
        directions at any step size.  When ``gradient_fn`` is given, the mass
        is instead the local width ``1 / E[(d log pi / dz_i)^2]``, which the
        gradients of the particles measure inside their own modes (exact for
        a Gaussian mode), capped by the marginal variance.
        """
        n = particles.shape[0]
        idx = self.rng.choice(n, size=n, replace=True, p=weights)
        cloud = particles[idx]
        q25, q50, q75 = np.quantile(cloud, [0.25, 0.5, 0.75], axis=0)
        sigma = np.maximum((q75 - q25) / 1.349, math.sqrt(self.jitter))
        clipped = np.clip(cloud, q50 - 4.0 * sigma, q50 + 4.0 * sigma)
        cov = np.atleast_2d(np.cov(clipped, rowvar=False))
        d = np.sqrt(np.maximum(np.diag(cov), self.jitter))
        corr = cov / np.outer(d, d)
        cov = corr * np.outer(sigma, sigma) + self.jitter * np.eye(self.dim)
        self._var = sigma ** 2
        if gradient_fn is not None and self.kernel != "rmh":
            g = np.asarray(gradient_fn(cloud[: self.gradient_particles]))
            g2 = np.median(g * g, axis=0) / 0.4549   # median of chi2(1) is 0.4549
            local_var = 1.0 / np.maximum(g2, 1e-12)
            self._var = np.minimum(self._var, local_var)
        if self.kernel == "rmh":
            try:
                self._chol = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                self._chol = np.diag(sigma)

    def params(self) -> Dict[str, np.ndarray]:
        if self.kernel == "rmh":
            return {"chol": (self.base_scale * self.scale) * self._chol}
        return {"step_size": np.float64(self.scale), "inverse_mass_matrix": self._var}

    def adapt(self, acceptance: float) -> None:
        if not np.isfinite(acceptance):
            return
        self.scale = float(np.clip(
            self.scale * math.exp(self.rate * (acceptance - self.target_accept)),
            self.bounds[0], self.bounds[1],
        ))

    def on_target(self, acceptance: float, tol: float = 0.4) -> bool:
        return abs(acceptance - self.target_accept) <= tol * self.target_accept


# ---------------------------------------------------------------------------
# Particle-parallel evaluation
def _host_mesh(jax):
    """A one-axis mesh over the host devices, or ``None`` when there is one device."""
    devices = jax.devices()
    if len(devices) < 2:
        if devices[0].platform == "cpu" and (os.cpu_count() or 1) > 1:
            warnings.warn(
                "bjsi_smc runs on one CPU device: JAX started before bjsi_smc could set "
                f"--{_DEVICE_FLAG}.  Set XLA_FLAGS=--{_DEVICE_FLAG}=<threads> before "
                "importing JAX for the particle-parallel (several times faster) path.",
                RuntimeWarning, stacklevel=3,
            )
        return None
    from jax.sharding import Mesh
    return Mesh(np.array(devices), ("p",))


def _row_map(jax, mesh, fn, n_rows: int, n_shared: int):
    """Map ``fn`` over the leading axis of its first ``n_rows`` arguments.

    The remaining ``n_shared`` arguments are passed whole to every row.  With a
    mesh the rows are split across the devices and each device evaluates its
    rows one at a time, which keeps the per-event arrays of one particle in
    cache; the row count must be a multiple of the device count (``_pad``).
    Without a mesh this is ``vmap``.
    """
    if mesh is None:
        return jax.vmap(fn, in_axes=(0,) * n_rows + (None,) * n_shared)
    from jax.sharding import PartitionSpec as P

    def local(*args):
        rows, shared = args[:n_rows], args[n_rows:]
        return jax.lax.map(lambda r: fn(*r, *shared), rows)

    return jax.shard_map(local, mesh=mesh, in_specs=(P("p"),) * n_rows + (P(),) * n_shared,
                         out_specs=P("p"))


# ---------------------------------------------------------------------------
# One SMC run
def _run_one(jax, jnp, blackjax, compiled, *, particles0: np.ndarray, key, kernel: str,
             mcmc_steps: int, waste_free: bool, move_tol: Optional[float], target_ess: float,
             final_rounds: int, max_iterations: int, tuning: _KernelTuning,
             hmc_integration_steps: int, nuts_max_doublings: int, pilot_rounds: int,
             pilot_size: int, pilot_steps: int, mesh, progressbar: bool,
             tag: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    from blackjax.smc.resampling import systematic

    log_prior, log_lik = compiled["log_prior"], compiled["log_lik"]
    n, dim = particles0.shape
    init_fn, step_fn, _ = _build_mutation(
        jax, jnp, blackjax, kernel, log_prior=log_prior, log_lik=log_lik,
        hmc_integration_steps=hmc_integration_steps, nuts_max_doublings=nuts_max_doublings,
    )
    if waste_free:
        if n % mcmc_steps != 0:
            raise ValueError(f"waste-free SMC needs draws ({n}) divisible by mcmc_steps ({mcmc_steps})")
        if n // mcmc_steps < 1000:
            warnings.warn(
                f"waste-free SMC with {n // mcmc_steps} ancestors: the weights of posterior "
                "basins that the kernel cannot cross rest on that many draws.  Use at least "
                "1000 ancestors (draws >= 1000 * mcmc_steps).", RuntimeWarning, stacklevel=3,
            )
    n_devices = 1 if mesh is None else mesh.size

    def pad(tree):
        """Repeat the first row until the row count is a multiple of the device count."""
        def one(a):
            extra = -a.shape[0] % n_devices
            return a if extra == 0 else jnp.concatenate([a, jnp.repeat(a[:1], extra, axis=0)])
        return jax.tree_util.tree_map(one, tree)

    def tempered(lmbda):
        return lambda z: log_prior(z) + lmbda * log_lik(z)

    def init_row(z, lmbda):
        return init_fn(z, tempered(lmbda))

    def step_row(k, state, lmbda, params):
        state, info = step_fn(k, state, tempered(lmbda), **params)
        return state, _acceptance(info)

    lik_rows = _row_map(jax, mesh, log_lik, 1, 0)
    grad_rows = _row_map(jax, mesh, lambda z, lam: jax.grad(tempered(lam))(z), 1, 1)
    init_rows = _row_map(jax, mesh, init_row, 1, 1)
    step_rows = _row_map(jax, mesh, step_row, 2, 2)

    @jax.jit
    def batched_lik(z):
        return lik_rows(pad(z))[: z.shape[0]]

    @jax.jit
    def tempered_grad(z, lmbda):
        return grad_rows(pad(z), lmbda)[: z.shape[0]]

    def move(key, states, lmbda, params):
        """One kernel move of every particle (rows already padded)."""
        keys = jax.random.split(key, jax.tree_util.tree_leaves(states)[0].shape[0])
        return step_rows(keys, states, lmbda, params)

    def mutate_fixed(key, starts, lmbda, params, n_moves: int):
        """``n_moves`` moves from every row of ``starts``; returns the states after each move."""
        m = starts.shape[0]
        states = init_rows(pad(starts), lmbda)

        def body(states, k):
            states, accept = move(k, states, lmbda, params)
            return states, (states.position[:m], accept[:m])

        _, (history, accept) = jax.lax.scan(body, states, jax.random.split(key, n_moves))
        return jnp.swapaxes(history, 0, 1), jnp.mean(accept)

    def distance(a, b, scale_var):
        """Mean distance between matched rows, in units of the proposal scale."""
        return jnp.mean(jnp.sqrt(jnp.sum((a - b) ** 2 / scale_var, axis=1)))

    def mutate_adaptive(key, starts, lmbda, params, scale_var):
        """Move every row of ``starts`` while the moves still carry the particles away.

        The chain length follows the adaptive rule of Chopin's ``particles``
        package: after each move the mean distance of the particles from their
        ancestors is measured, and the population stops once a move increases
        it by less than ``move_tol`` of its previous value, or after
        ``mcmc_steps`` moves.  The rule compares the distance with itself, so it
        does not depend on how well ``scale_var`` matches the width of the
        target, and particles held in separate copies of a mode (the symmetric
        rotations) stop counting once they have explored their own copy.
        """
        m = starts.shape[0]

        def cond(carry):
            k, _, _, _, prev, dist = carry
            if move_tol is None:
                return k < mcmc_steps
            growing = (k < 2) | (dist - prev >= move_tol * prev)
            return (k < mcmc_steps) & growing

        def body(carry):
            k, states, key, accept_sum, _, dist = carry
            key, sub = jax.random.split(key)
            states, accept = move(sub, states, lmbda, params)
            new_dist = distance(states.position[:m], starts, scale_var)
            return k + 1, states, key, accept_sum + jnp.mean(accept[:m]), dist, new_dist

        zero = jnp.float64(0.0)
        carry = (jnp.int32(0), init_rows(pad(starts), lmbda), key, zero, zero, zero)
        k, states, _, accept_sum, _, dist = jax.lax.while_loop(cond, body, carry)
        return states.position[:m], accept_sum / k, k, dist

    @partial(jax.jit, static_argnames=("wf",))
    def iteration(key, particles, loglik, lmbda_old, lmbda_new, params, scale_var, wf):
        log_w = (lmbda_new - lmbda_old) * loglik
        log_w = jnp.where(jnp.isfinite(log_w), log_w, -jnp.inf)
        lse = jax.scipy.special.logsumexp(log_w)
        log_inc = lse - jnp.log(n)
        weights = jnp.exp(log_w - lse)
        k1, k2 = jax.random.split(key)
        if wf:
            ancestors = particles[systematic(k1, weights, n // mcmc_steps)]
            history, accept = mutate_fixed(k2, ancestors, lmbda_new, params, mcmc_steps - 1)
            new_particles = jnp.concatenate([ancestors[:, None, :], history], axis=1).reshape(n, dim)
            dist = distance(history[:, -1], ancestors, scale_var)
            n_moves = jnp.int32(mcmc_steps - 1)
        else:
            ancestors = particles[systematic(k1, weights, n)]
            new_particles, accept, n_moves, dist = mutate_adaptive(k2, ancestors, lmbda_new, params,
                                                                   scale_var)
        return new_particles, log_inc, accept, n_moves, dist

    @jax.jit
    def pilot(key, starts, lmbda, params):
        return mutate_fixed(key, starts, lmbda, params, pilot_steps)[1]

    def _to_jax_params(p):
        return {k: jnp.asarray(v) for k, v in p.items()}

    particles = jnp.asarray(particles0)
    loglik = np.asarray(batched_lik(particles))
    lmbda = 0.0
    log_evidence = 0.0
    schedule = [0.0]
    increments: List[float] = []
    accepts: List[float] = []
    ess_hist: List[float] = []
    moves_hist: List[int] = []
    t0 = time.perf_counter()
    n_iter = 0
    n_final = 0
    if progressbar:
        wf = f"waste-free(p={mcmc_steps})" if waste_free else f"steps<={mcmc_steps}"
        print(f"[bjsi_smc] {tag} start | particles={n} dim={dim} kernel={kernel} {wf} "
              f"move_tol={move_tol} target_ess={target_ess:.2f} devices={n_devices}",
              flush=True)
    while True:
        if lmbda >= 1.0:
            if n_final >= final_rounds:
                break
            lmbda_new = 1.0
            n_final += 1
        else:
            if n_iter >= max_iterations:
                break
            delta = _solve_delta(loglik, lmbda, target_ess)
            lmbda_new = min(1.0, lmbda + delta)
            n_iter += 1
        # Tune the proposal on the weighted cloud: it is the best available
        # picture of the target the particles are about to be moved under.
        log_w = (lmbda_new - lmbda) * loglik
        log_w = np.where(np.isfinite(log_w), log_w, -np.inf)
        w = np.exp(log_w - np.max(log_w))
        w /= w.sum()
        ess = 1.0 / float(np.sum(w * w))
        step_t0 = time.perf_counter()
        cloud = np.asarray(particles)
        tuning.measure(cloud, w, gradient_fn=lambda z: tempered_grad(jnp.asarray(z), lmbda_new))
        # Pilot moves on a subset of the weighted cloud bring the proposal size
        # onto the acceptance target before the population is moved.
        for _ in range(pilot_rounds):
            key, sub = jax.random.split(key)
            starts = jnp.asarray(cloud[tuning.rng.choice(n, size=pilot_size, replace=True, p=w)])
            acc = float(pilot(sub, starts, lmbda_new, _to_jax_params(tuning.params())))
            if tuning.on_target(acc):
                break
            tuning.adapt(acc)
        params = tuning.params()
        key, sub = jax.random.split(key)
        # Waste-free for the tempering increments; at the posterior every
        # particle is moved, so the final rounds do not thin the population
        # to draws / mcmc_steps ancestors.
        particles, log_inc, accept, n_moves, dist = iteration(
            sub, particles, jnp.asarray(loglik), lmbda, lmbda_new, _to_jax_params(params),
            jnp.asarray(tuning._var), wf=bool(waste_free and lmbda < 1.0),
        )
        particles.block_until_ready()
        loglik = np.asarray(batched_lik(particles))
        accept = float(accept)
        n_moves = int(n_moves)
        moves_hist.append(n_moves)
        log_inc = float(log_inc)
        log_evidence += log_inc
        increments.append(log_inc)
        accepts.append(accept)
        ess_hist.append(ess)
        schedule.append(float(lmbda_new))
        tuning.adapt(accept)
        if progressbar:
            phase = "final" if lmbda >= 1.0 else f"iter {n_iter:3d}"
            print(f"[bjsi_smc] {tag} {phase} | lambda={lmbda_new:.5f} | ess={ess:7.1f}/{n} | "
                  f"accept={accept:.2f} scale={tuning.scale:.3g} moves={n_moves} dist={float(dist):.2f} | "
                  f"logZ={log_evidence:9.2f} | "
                  f"step={time.perf_counter() - step_t0:5.1f}s elapsed={time.perf_counter() - t0:6.1f}s",
                  flush=True)
        lmbda = lmbda_new

    info = {
        "n_iterations": n_iter,
        "final_rounds": n_final,
        "converged": bool(lmbda >= 1.0),
        "lambda_schedule": schedule,
        "log_evidence": float(log_evidence),
        "log_evidence_increments": increments,
        "acceptance": accepts,
        "ess": ess_hist,
        "moves": moves_hist,
        "total_time_s": float(time.perf_counter() - t0),
    }
    return np.asarray(particles), info


# ---------------------------------------------------------------------------
# Public entry point
def sample_smc(
    model,
    *,
    draws: int = 4000,
    chains: int = 2,
    cores: int = 1,
    random_seed: Optional[int] = None,
    progressbar: bool = True,
    kernel: str = "hmc",
    mcmc_steps: int = 5,
    move_tol: Optional[float] = None,
    waste_free: bool = False,
    target_ess: float = 0.8,
    final_rounds: int = 2,
    max_iterations: int = 500,
    target_accept: Optional[float] = None,
    adaptation_rate: float = 1.0,
    hmc_step_size: float = 0.2,
    hmc_integration_steps: int = 10,
    nuts_max_doublings: int = 8,
    pilot_rounds: int = 8,
    pilot_size: int = 200,
    pilot_steps: int = 3,
    output_batch: int = 500,
) -> Tuple[az.InferenceData, Dict[str, Any]]:
    """Sample a PyMC model by adaptive tempered SMC and return an ``InferenceData``.

    Parameters
    ----------
    model : the PyMC model from :func:`bjsi._build_joint_model`
    draws : particles per run
    chains : independent SMC runs; they become the ``chain`` dimension so that
        the agreement of their evidences and summaries is the convergence check
    cores : runs executed concurrently.  One run already spreads its
        particles over every host device, so the gain is the idle share of the
        machine; the runs share the devices from Python threads, JAX releasing
        the GIL during computation
    kernel : ``"hmc"`` (default), ``"nuts"`` or ``"rmh"`` (random walk with the
        particle covariance).  On the Cushing fabric models the random walk
        leaves the population 20-40 nats below the posterior mode and gives
        run-to-run evidences that disagree by several nats; HMC with a
        diagonal mass from the cloud reaches the mode and the evidences of
        independent runs agree within a fraction of a nat.
    mcmc_steps : most moves per particle and tempering step (the waste-free ``p``)
    move_tol : the population stops moving once a move increases the mean
        distance of the particles from their ancestors by less than this
        fraction (at least 2 moves; the rule of Chopin's ``particles`` package).
        ``None`` (default) always makes ``mcmc_steps`` moves.  On Cape (1720
        events, K=1) 0.1 stopped at 4 of 5 moves and spread the log evidences
        of independent runs over 1.5 nats, against 0.4 nat over six runs with
        5 fixed moves: the HMC moves decorrelate slowly on these models.  Waste-free
        increments always make ``mcmc_steps - 1`` moves.
    waste_free : keep every intermediate state (``draws / mcmc_steps`` ancestors)
        during the tempering; the final rounds at the posterior move every particle
    target_ess : fraction of ``draws`` kept as ESS when choosing each increment
    final_rounds : extra mutation rounds at the posterior after the last increment
    target_accept : acceptance targeted by the scale adaptation
        (0.234 for rmh, 0.7 for hmc, 0.8 for nuts by default)
    hmc_step_size, hmc_integration_steps, nuts_max_doublings : gradient kernels
    pilot_rounds, pilot_size, pilot_steps : before each tempering step, up to
        ``pilot_rounds`` trials of ``pilot_steps`` moves on ``pilot_size``
        particles adjust the proposal size until the acceptance is on target

    The returned ``InferenceData`` has the posterior variables of the model
    with ``chain = run`` and ``draw = particle``; ``sample_stats`` carries the
    log likelihood and log prior of every particle, and ``attrs`` the log
    evidence of each run.  The second return value is the per-run diagnostics.
    """
    jax, jnp, blackjax = _import_jax()
    kernel = str(kernel).lower()
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")
    if target_accept is None:
        target_accept = {"rmh": 0.234, "hmc": 0.7, "nuts": 0.8}[kernel]
    draws, chains, mcmc_steps = int(draws), int(chains), int(mcmc_steps)
    if min(draws, chains, mcmc_steps) < 1:
        raise ValueError("draws, chains and mcmc_steps must be positive")

    rng = np.random.default_rng(random_seed)
    compiled = _compile_model(model, jax, jnp)
    dim = compiled["dim"]
    mesh = _host_mesh(jax)
    move_tol = None if move_tol is None else float(move_tol)
    key = jax.random.PRNGKey(int(rng.integers(2**31)))

    # Initial particles, keys and generators are drawn up front so that the
    # result does not depend on the order in which concurrent runs finish.
    starts = []
    for run in range(chains):
        particles0 = _prior_particles(model, draws, rng)
        key, sub = jax.random.split(key)
        run_rng = np.random.default_rng(int(rng.integers(2**31)))
        starts.append((particles0, sub, run_rng))

    def _one(run):
        particles0, sub, run_rng = starts[run]
        tuning = _KernelTuning(kernel, dim, target_accept=target_accept, rng=run_rng,
                               adaptation_rate=adaptation_rate, hmc_step_size=hmc_step_size)
        particles, info = _run_one(
            jax, jnp, blackjax, compiled, particles0=particles0, key=sub, kernel=kernel,
            mcmc_steps=mcmc_steps, waste_free=waste_free, move_tol=move_tol,
            target_ess=target_ess, final_rounds=final_rounds, max_iterations=max_iterations,
            tuning=tuning, hmc_integration_steps=hmc_integration_steps,
            nuts_max_doublings=nuts_max_doublings, pilot_rounds=int(pilot_rounds),
            pilot_size=min(int(pilot_size), draws), pilot_steps=int(pilot_steps), mesh=mesh,
            progressbar=progressbar, tag=f"run {run + 1}/{chains}",
        )
        if progressbar:
            status = "converged" if info["converged"] else "max_iterations"
            print(f"[bjsi_smc] run {run + 1}/{chains} done ({status}) | iters={info['n_iterations']} "
                  f"logZ={info['log_evidence']:.2f} | total={info['total_time_s']:.1f}s", flush=True)
        return particles, info

    cores = max(1, min(int(cores), chains))
    if cores == 1:
        results = [_one(run) for run in range(chains)]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=cores) as pool:
            results = list(pool.map(_one, range(chains)))
    runs = [r[0] for r in results]
    infos = [r[1] for r in results]

    idata = _build_inference_data(jax, jnp, compiled, runs, infos, output_batch)
    summary = {
        "sampler": "blackjax_smc",
        "kernel": kernel,
        "draws": draws,
        "chains": chains,
        "cores": cores,
        "mcmc_steps": mcmc_steps,
        "move_tol": move_tol,
        "waste_free": bool(waste_free),
        "target_ess": float(target_ess),
        "final_rounds": int(final_rounds),
        "log_evidence": [i["log_evidence"] for i in infos],
        "n_iterations": [i["n_iterations"] for i in infos],
        "runs": infos,
    }
    return idata, summary


def _build_inference_data(jax, jnp, compiled, runs, infos, batch: int) -> az.InferenceData:
    f_out = jax.jit(jax.vmap(compiled["outputs"]))
    f_prior = jax.jit(jax.vmap(compiled["log_prior"]))
    f_lik = jax.jit(jax.vmap(compiled["log_lik"]))
    posterior: Dict[str, List[np.ndarray]] = {name: [] for name in compiled["out_names"]}
    stats = {"log_likelihood": [], "log_prior": []}
    for particles in runs:
        z = jnp.asarray(particles)
        chunks = {name: [] for name in posterior}
        for start in range(0, z.shape[0], batch):
            out = f_out(z[start:start + batch])
            for name in posterior:
                chunks[name].append(np.asarray(out[name]))
        for name in posterior:
            posterior[name].append(np.concatenate(chunks[name], axis=0))
        stats["log_likelihood"].append(np.asarray(f_lik(z)))
        stats["log_prior"].append(np.asarray(f_prior(z)))
    posterior_arr = {name: np.stack(v, axis=0) for name, v in posterior.items()}
    stats_arr = {name: np.stack(v, axis=0) for name, v in stats.items()}
    stats_arr["lp"] = stats_arr["log_likelihood"] + stats_arr["log_prior"]
    idata = az.from_dict(posterior=posterior_arr, sample_stats=stats_arr)
    idata.attrs["sampler"] = "blackjax_smc"
    idata.attrs["log_evidence"] = [float(i["log_evidence"]) for i in infos]
    return idata
