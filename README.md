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


## References
- Michael (1984), stress inversion from slip data.
- Vavrycuk (2014), iterative joint stress-plane inversion.
- Beauce et al. (2022), ILSI with variable shear framework.
- PyMC (2023), Bayesian inference framework used by this project.
