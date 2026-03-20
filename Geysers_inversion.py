import os
from pathlib import Path
import sys

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"

REPO_ROOT = Path(__file__).resolve().parent
ILSI_SRC = REPO_ROOT / "src"
if str(ILSI_SRC) not in sys.path:
    sys.path.append(str(ILSI_SRC))

import joblib
import numpy as np
import pandas as pd

from bjsi import Bayesian_joint_plane_selection_NUTS
import ilsi
import utils_stress
from plot_stress_output import (
    generate_figures_hdi,
    arviz_uncertainty_metrics,
    plot_stereonet_planes_map,
    write_arviz_outputs,
    posterior_misfit_diagnostics,
    plot_focal_mechanisms_map,
)


EVAL_POSTERIOR_MISFIT = True
POSTERIOR_EVAL_SEED = 0

NUTS_DRAWS = 2000
NUTS_TUNE = 1000
NUTS_CHAINS = 4
NUTS_CORES = 4
NUTS_TARGET_ACCEPT = 0.8
NUTS_SAMPLER = "nutpie"
NUTS_SAMPLER_KWARGS = None
SELECTION_BETA = 10.0
TAU_WEIGHT_EXPONENT = 0.6
SLIP_LIKELIHOOD = "von_mises_fisher"
SLIP_VMF_KAPPA = 8.0
CLUSTERING_PRIOR_STRENGTH = 2.0


def _load_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "strike1" not in df.columns:
        required = {"strike", "dip", "rake"}
        if not required.issubset(df.columns):
            raise ValueError(
                "Catalog must contain either strike1/dip1/rake1 or strike/dip/rake columns."
            )
        df = df.rename(columns={"strike": "strike1", "dip": "dip1", "rake": "rake1"})

    if not {"strike2", "dip2", "rake2"}.issubset(df.columns):
        strikes_1 = df["strike1"].to_numpy(dtype=float)
        dips_1 = df["dip1"].to_numpy(dtype=float)
        rakes_1 = np.mod(df["rake1"].to_numpy(dtype=float), 360.0)
        strikes_2, dips_2, rakes_2 = np.asarray(
            list(map(utils_stress.aux_plane, strikes_1, dips_1, rakes_1))
        ).T
        df["strike2"] = strikes_2
        df["dip2"] = dips_2
        df["rake2"] = rakes_2

    return df
def main() -> None:
    catalog_path = REPO_ROOT / "examples" / "data" / "Geyser_2007_2020.csv"
    if not catalog_path.exists():
        fallback = REPO_ROOT / "Geyser_2007_2020.csv"
        if fallback.exists():
            catalog_path = fallback
        else:
            raise FileNotFoundError(
                "Catalog not found. Expected examples/data/Geyser_2007_2020.csv"
            )

    focals = _load_catalog(catalog_path)
    # focals = focals[:200]

    out_path = REPO_ROOT / "Geysers_output_gaussian"
    out_path.mkdir(exist_ok=True)

    strikes_1 = focals["strike1"].to_numpy(dtype=float)
    dips_1 = focals["dip1"].to_numpy(dtype=float)
    rakes_1 = np.mod(focals["rake1"].to_numpy(dtype=float), 360.0)

    strikes_2 = focals["strike2"].to_numpy(dtype=float)
    dips_2 = focals["dip2"].to_numpy(dtype=float)
    rakes_2 = focals["rake2"].to_numpy(dtype=float)

    inversion_output = Bayesian_joint_plane_selection_NUTS(
        strikes_1,
        dips_1,
        rakes_1,
        strikes_2,
        dips_2,
        rakes_2,
        infer_friction=True,
        infer_friction_method="sample",
        draws=NUTS_DRAWS,
        tune=NUTS_TUNE,
        chains=NUTS_CHAINS,
        cores=NUTS_CORES,
        target_accept=NUTS_TARGET_ACCEPT,
        nuts_sampler=NUTS_SAMPLER,
        nuts_sampler_kwargs=NUTS_SAMPLER_KWARGS,
        hdi_prob=0.90,
        weighted_likelihood=True,
        likelihood_weight_mode="event",
        tau_weight_exponent=TAU_WEIGHT_EXPONENT,
        selection_beta=SELECTION_BETA,
        slip_likelihood=SLIP_LIKELIHOOD,
        slip_vmf_kappa=SLIP_VMF_KAPPA,
        normalize_tau_weights=True,
        clustering_prior_strength=CLUSTERING_PRIOR_STRENGTH,
    )

    R = float(inversion_output["R_median"])
    plane_map = inversion_output["plane_selection_map"]
    fp_strikes = np.where(plane_map == 0, strikes_1, strikes_2)
    fp_dips = np.where(plane_map == 0, dips_1, dips_2)
    fp_rakes = np.where(plane_map == 0, rakes_1, rakes_2)

    I, _, _, _ = ilsi.compute_instability_parameter(
        inversion_output["principal_directions"],
        R,
        inversion_output["mu"],
        strikes_1,
        dips_1,
        rakes_1,
        strikes_2,
        dips_2,
        rakes_2,
        return_fault_planes=True,
    )

    inversion_output["misfit"] = float(
        np.mean(
            utils_stress.mean_angular_residual(
                inversion_output["stress_tensor"], fp_strikes, fp_dips, fp_rakes
            )
        )
    )

    posterior_diag = None
    if EVAL_POSTERIOR_MISFIT:
        posterior_diag = posterior_misfit_diagnostics(
            inversion_output,
            strikes_1,
            dips_1,
            rakes_1,
            strikes_2,
            dips_2,
            rakes_2,
            seed=POSTERIOR_EVAL_SEED,
            ilsi_src_path=str(ILSI_SRC),
        )
        if posterior_diag is not None:
            inversion_output["misfit_posterior"] = {
                k: v
                for k, v in posterior_diag.items()
                if k not in {"bestfit_stress_tensor", "bestfit_plane_map"}
            }

    output = {"misfit": [inversion_output["misfit"]], "R": [R], "mu": [inversion_output["mu"]]}

    if posterior_diag is not None:
        output["misfit_draw_hard_min"] = [posterior_diag["misfit_draw_hard_min"]]
        output["misfit_draw_hard_median"] = [posterior_diag["misfit_draw_hard_median"]]
        output["misfit_draw_expected_median"] = [
            posterior_diag["misfit_draw_expected_median"]
        ]
        output["bestfit_draw_index"] = [posterior_diag["bestfit_draw_index"]]

    az_metrics = arviz_uncertainty_metrics(
        inversion_output.get("idata"),
        var_names=["R", "mu", "tau0", "Sigma"],
    )
    for k, v in az_metrics.items():
        output[k] = [v]

    write_arviz_outputs(
        inversion_output.get("idata"),
        output_dir=str(out_path),
        var_names=["R", "mu", "tau0", "Sigma"],
        hdi_prob=0.9,
    )

    out_focals = focals.copy()
    out_focals["optimum_strike"] = fp_strikes
    out_focals["optimum_dip"] = fp_dips
    out_focals["optimum_rake"] = fp_rakes

    idx = plane_map.ravel()
    rows = np.arange(I.shape[0])
    out_focals["instability"] = I[rows, idx]
    out_focals["instability_max"] = np.max(I, axis=1)

    pd.DataFrame(output).to_csv(out_path / "inv_out.csv", index=False, float_format="%.4f")
    out_focals.to_csv(out_path / "optimum_focal.csv", index=False, float_format="%.4f")
    joblib.dump(inversion_output, out_path / "stress_out.pkl")

    generate_figures_hdi(
        stress_pkl=str(out_path / "stress_out.pkl"),
        optimum_focal_csv=str(out_path / "optimum_focal.csv"),
        mu=inversion_output["mu"],
        output_dir=str(out_path / "figures"),
        ilsi_src_path=str(ILSI_SRC),
        overwrite=True,
    )

    (out_path / "figures").mkdir(exist_ok=True)
    # plot_focal_mechanisms_map(
    #     all_events=focals,
    #     selected_events=out_focals,
    #     output_png=str(out_path / "figures" / "top_view.png"),
    #     color_column="instability",
    #     use_cartopy=True,
    # )

    plot_stereonet_planes_map(
        all_events=focals,
        selected_events=out_focals,
        output_png=out_path / "figures" / "selected_planes_topview.png",
        color_column="instability",
    )


if __name__ == "__main__":
    main()
