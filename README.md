# BJSI: Bayesian Joint Stress Inversion

This repository is currently centered on two things only:

- the core library code in `src/`
- one end-to-end example workflow in `Geysers_inversion.py`

BJSI jointly estimates stress orientation, stress shape ratio `R`, friction `mu` (optional), and nodal-plane selection from focal mechanisms using Bayesian inference.

## Core code (`src/`)

- `src/bjsi.py`: Bayesian inversion engine (NUTS-based joint plane selection).
- `src/ilsi.py`: deterministic stress-inversion and instability utilities (ILSI).
- `src/utils_stress.py`: focal geometry, traction, and residual helper functions (ILSI).
- `src/plot_stress_output.py`: plotting and posterior diagnostic helpers used by the example script.

## Core Features

- **Joint Inversion**: Simultaneously solves for stress state ($R$, principal directions) and the true fault plane from ambiguous focal mechanisms.
- **Clustering Prior**: Incorporates a `clustering_prior_strength` parameter to resolve ambiguous plane selections by learning from unambiguous events in the same dataset. This probabilistic weighting mechanism calculates a certainty-weighted confident orientation tensor ($T_{conf}$) to guide the selection for ambiguous events based on the prevailing geometric trend of high-confidence selections.
- **Friction Estimation**: Optionally infers the macroscopic friction coefficient (`mu`) simultaneously with the stress state.

## Example workflow (`Geysers_inversion.py`)

`Geysers_inversion.py` runs a full inversion on the Geysers catalog and produces:

- posterior and summary outputs
- selected focal planes
- diagnostic and summary figures


## Installation

Use the cleaned conda environment file:

```bash
conda env create -f env.yml
conda activate pymc
```

If you already have a `pymc` environment and want to sync it to this file:

```bash
conda env update -f env.yml --prune
```

Optional extras:

- `cartopy` for basemap rendering in the map view
- `jax jaxlib` if you want to use numpyro as the NUTS backend

```bash
conda install -c conda-forge cartopy
conda install -c conda-forge jax jaxlib numpyro
```

## Run

```bash
python Geysers_inversion.py
```

## Outputs

The script writes results under `Geysers_output/`:

- `stress_out.pkl`: full inversion output dictionary
- `inv_out.csv`: scalar summary metrics
- `optimum_focal.csv`: selected nodal plane and instability per event
- `arviz.summary.txt`, `arviz.ess_rhat.csv`, `arviz.plot_posterior_hdi90.png`: posterior diagnostics
- `figures/`: Mohr/stereonet/PT/map figures

## Mechanism uncertainty and externally fixed fault planes

The PyMC NUTS and SMC entry points accept `strike_sigma_deg`, `dip_sigma_deg`,
and `rake_sigma_deg`. Each is a one-standard-deviation measurement error in
degrees, supplied either globally as a scalar or separately for each event as
an array of length N. **All three now default to 5 degrees.** Explicitly set all
three to zero to reproduce the previous exact-angle model.

The model draws one latent mechanism per event using independent local Gaussian
offsets from the first supplied nodal plane's strike, dip, and rake. It derives
the auxiliary plane from that same mechanism, preserving the double couple.
Angle wrapping and dip-boundary crossings are handled through the normal/slip
vectors. This is a local SDR error model, not a rotation-invariant distribution;
the 5-degree default is an assumption to test against actual catalog errors.
The directional likelihood retains its existing residual-scatter parameter.

To condition stress inference on externally supplied fault labels:

```python
result = Bayesian_joint_plane_selection_NUTS(
    strike1, dip1, rake1, strike2, dip2, rake2,
    fixed_plane_indices=labels_1_or_2 - 1,
    strike_sigma_deg=5.0, dip_sigma_deg=5.0, rake_sigma_deg=5.0,
    infer_friction=False, friction_fixed=0.6,
    weighted_likelihood=False, clustering_prior_strength=0.0,
    slip_likelihood="von_mises_fisher", slip_vmf_kappa=8.0,
)
```

`fixed_plane_indices` must contain 0 for input plane 1 or 1 for input plane 2
for every event. It bypasses instability and clustering in plane selection,
while still allowing uncertain geometry. It cannot be combined with iterative
preselection. Friction does not enter this directional likelihood; the fixed
value above is only an output placeholder, not an inferred friction estimate.
With fixed labels, reported plane probabilities are exactly zero or one and
agreement with those labels is imposed, not a validation result.

The result records `mechanism_sigma_deg` (N by 3) and `fixed_plane_indices`.
With nonzero errors, `result['idata'].posterior['mechanism_angles_deg']` contains
canonical latent plane-1 strike/dip/rake draws. Use these draws for analyses of
the inferred geometry; existing plotting helpers use nominal input geometry.
The separate BlackJAX implementation does not yet support these arguments.

For controlled starting-point tests, NUTS accepts `initvals` with the native
`nuts_sampler='pymc'` backend and uses `adapt_diag` without start-point jitter.


## References
- Michael (1984), stress inversion from slip data.
- Vavrycuk (2014), iterative joint stress-plane inversion.
- Beauce et al. (2022), ILSI with variable shear framework.
- PyMC (2023), Bayesian inference framework used by this project.
