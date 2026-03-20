# plot_stress_output.py
import os
import sys
from pathlib import Path
from typing import Optional, Sequence, Dict, Any, Tuple

# Default src path is the directory containing this file
DEFAULT_ILSI_SRC = str(Path(__file__).parent)

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplstereonet
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# Lazy imports that depend on local dev code (ilsi and utils_stress)
def _ensure_ilsi_path(ilsi_src_path: Optional[str]):
    """Append ILSI dev path to sys.path if provided and not already present."""
    if ilsi_src_path:
        if ilsi_src_path not in sys.path:
            sys.path.append(ilsi_src_path)


def _safe_import_ilsi(ilsi_src_path: Optional[str]):
    _ensure_ilsi_path(ilsi_src_path)
    import ilsi  # type: ignore
    import utils_stress  # type: ignore

    return ilsi, utils_stress


# ---------------------------------------------------------------------------
# Helper computation functions (from your original script)
# ---------------------------------------------------------------------------
def calc_fault_angle(rake: float) -> float:
    """Return normalized fault angle in [-1,1] based on rake (consistent with old calc)."""
    if np.abs(rake) <= 90:
        return rake / 90.0
    else:
        return (180.0 - np.abs(rake)) * (rake / np.abs(rake)) / 90.0


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _plot_mohr(
    ax, stress_tensor, focals, mu, color_by, cmap_name, title_text, outname=None
):
    """
    Generic Mohr plot:
      - ax: matplotlib Axes
      - stress_tensor: 3x3 stress tensor (or as in your utils_stress input)
      - focals: pandas DataFrame with strike/dip/rake & other fields
      - color_by: array-like values for colormap (len == number of focal solutions)
      - cmap_name: string colormap
    """
    positive_compression = True

    p_stress, p_dir = utils_stress.stress_tensor_eigendecomposition(stress_tensor)
    if positive_compression:
        p_stress = p_stress * -1.0
    sig1, sig2, sig3 = p_stress

    # draw the three Mohr circles (sig1-sig3, sig1-sig2, sig2-sig3)
    for s1, s2 in [[sig1, sig3], [sig1, sig2], [sig2, sig3]]:
        center_x = (s1 + s2) / 2.0
        radius = (s2 - s1) / 2.0
        circle = plt.Circle((center_x, 0.0), radius, color="k", fill=False)
        ax.add_patch(circle)

    # compute normals/slips & traction
    strikes = focals["optimum_strike"].values
    dips = focals["optimum_dip"].values
    rakes = focals["optimum_rake"].values

    normals, slips = utils_stress.normal_slip_vectors(
        strikes, dips, np.zeros_like(strikes)
    )
    normals, slips = normals.T, slips.T
    traction, normal_traction, shear_traction = utils_stress.compute_traction(
        stress_tensor, normals
    )
    shear_mag = np.sqrt(np.sum(shear_traction**2, axis=-1))
    normal_comp = np.sum(normal_traction * normals, axis=-1)

    # label principal stresses on x-axis
    for i, s in enumerate([sig1, sig2, sig3]):
        label = r"$\sigma_{{{:d}}}$".format(i + 1)
        ax.text(s - 0.05, -0.07, label, fontsize=12)

    # colormap scatter
    if cmap_name == "cool_r":
        sc = ax.scatter(
            -normal_comp,
            shear_mag,
            c=color_by,
            alpha=0.8,
            vmin=-1,
            vmax=1,
            marker=".",
            cmap=cmap_name,
        )

        cbar = plt.colorbar(sc, ax=ax, fraction=0.07, pad=0.04, shrink=0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("Fault Angle", fontsize=14)

    elif cmap_name == "jet":
        sc = ax.scatter(
            -normal_comp,
            shear_mag,
            c=color_by,
            alpha=0.8,
            vmin=0,
            vmax=1,
            marker=".",
            cmap=cmap_name,
        )

        cbar = plt.colorbar(sc, ax=ax, fraction=0.07, pad=0.04, shrink=0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("Instability", fontsize=14)

    elif cmap_name == "seismic":
        sc = ax.scatter(
            -normal_comp,
            shear_mag,
            c=color_by,
            alpha=0.8,
            # vmin=0, vmax=1,
            marker=".",
            cmap=cmap_name,
        )

        cbar = plt.colorbar(sc, ax=ax, fraction=0.07, pad=0.04, shrink=0.5)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label("Pf [Mpa]", fontsize=14)

    else:
        sc = ax.scatter(
            -normal_comp,
            shear_mag,
            c=color_by,
            alpha=0.8,
            # vmin=-1, vmax=1,
            marker=".",
            cmap=cmap_name,
        )

    # friction/hydro strength curves (copied from your script)
    center_x = (sig1 + sig3) / 2.0
    radius = (sig1 - sig3) / 2.0
    s0 = 0.0
    Ph = -9.8 * 2.5
    eff_center_x = (sig1 + sig3) / 2.0 - Ph
    s0_required = radius * np.sqrt(mu**2 + 1.0) - eff_center_x * mu
    cnorm = np.arange(Ph, sig1 - 0.2, 0.1, dtype=float)
    cnorm2 = np.arange(sig3, sig1, 0.1, dtype=float)
    f_strength_hydro = mu * (cnorm - Ph) + s0_required
    f_strength_hydro2 = mu * (cnorm2 - sig3) + s0

    ax.plot(cnorm, f_strength_hydro, "k", linewidth=0.9)
    ax.plot(cnorm2, f_strength_hydro2, "k--", linewidth=0.9)

    x_intercept = Ph - (s0_required / mu)
    ax.set_xlim(x_intercept - 0.2, center_x + (radius + 0.3))
    ax.set_ylim(0.0, radius + 0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_ylabel(r"Shear stress $\tau$", fontsize=12)
    ax.set_xlabel(r"Normal stress $\sigma$", fontsize=12, labelpad=10)
    plt.title(title_text, fontsize=14)
    if outname:
        plt.savefig(outname, dpi=300, bbox_inches="tight")

    return sc


def _plot_PT_axes(outpath, strikes, dips, rakes, stress_tensor):
    """Create PT axes figure and save to outpath."""
    n, s = utils_stress.normal_slip_vectors(strikes, dips, rakes)
    p_or = np.zeros((len(strikes), 2), dtype=np.float32)
    t_or = np.zeros((len(strikes), 2), dtype=np.float32)
    for i in range(len(strikes)):
        p_vec, t_vec, b_vec = utils_stress.p_t_b_axes(n[:, i], s[:, i])
        p_or[i, :] = utils_stress.get_bearing_plunge(p_vec)
        t_or[i, :] = utils_stress.get_bearing_plunge(t_vec)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="stereonet")
    ax.line(t_or[:, 1], t_or[:, 0], marker="x", color="C0")
    ax.line(p_or[:, 1], p_or[:, 0], marker="o", color="C3")
    ax.line(t_or[0, 1], t_or[0, 0], marker="x", color="C0", label="T-axis")
    ax.line(p_or[0, 1], p_or[0, 0], marker="o", color="C3", label="P-axis")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0.90, 0.92))
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_stereonet_instability(outpath, strikes, dips, rakes, instability, title):
    """
    Plot planes on a stereonet, color-coded by instability.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="stereonet")

    # Create a ScalarMappable for the colormap to handle array-based coloring
    # Since ax.plane doesn't accept an array for color easily in one go (it returns a list of lines),
    # we iterate or we can use a collection if we want to be fancy. Iteration is clearest for now.

    cmap = plt.get_cmap("jet")
    norm = plt.Normalize(vmin=0, vmax=1)

    # We can plot all at once but we need to set colors manually after
    # ax.plane returns a list of Line2D objects
    planes = ax.plane(strikes, dips, alpha=0.5)

    # Set color for each plane based on instability
    for i, plane_line in enumerate(planes):
        color = cmap(norm(instability[i]))
        plane_line.set_color(color)

    # Add Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Instability")

    ax.grid(True)
    ax.set_title(title, y=1.02)

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_single_method_output_panels(
    inversion_output,
    focals: pd.DataFrame,
    utils_stress_mod,
    *,
    confidence_level: float,
    interval_label: str,
    hdi_info: Optional[Dict[str, Sequence[float]]],
    use_hdi_samples: bool,
    show_hdi_on_hist: bool,
    show_confidence_contours: bool,
    bootstrap_max_points: Optional[int],
    save_separate: bool,
    axes=None,
    marker: str = "o",
    color: str = "b",
):
    """Plot stereonet, shape-ratio histogram, and shear-stress histogram."""

    def _get_interval(
        info: Optional[Dict[str, Sequence[float]]], key: str
    ) -> Optional[Tuple[float, float]]:
        if not isinstance(info, dict):
            return None
        val = info.get(key)
        if val is None or len(val) < 2:
            return None
        try:
            return (float(val[0]), float(val[1]))
        except Exception:
            return None

    figures = []
    if save_separate:
        plt.rcParams["xtick.labelsize"] = 16
        plt.rcParams["ytick.labelsize"] = 16
        fig1 = plt.figure(figsize=(6, 6))
        ax1_local = fig1.add_subplot(111, projection="stereonet")
        fig2 = plt.figure(figsize=(8, 6))
        ax2_local = fig2.add_subplot(111)
        fig3 = plt.figure(figsize=(8, 6))
        ax3_local = fig3.add_subplot(111)
        figures = [fig1, fig2, fig3]
    elif axes is None:
        fig_local = plt.figure(figsize=(16, 5))
        gs_local = fig_local.add_gridspec(nrows=1, ncols=3, hspace=0.4, wspace=0.7)
        ax1_local = fig_local.add_subplot(gs_local[0, 0], projection="stereonet")
        ax2_local = fig_local.add_subplot(gs_local[0, 1])
        ax3_local = fig_local.add_subplot(gs_local[0, 2])
        figures = [fig_local]
    else:
        ax1_local, ax2_local, ax3_local = axes

    n_resamplings = inversion_output["boot_principal_directions"].shape[0]
    ax1_local.grid(True, kind="polar", linestyle=":")
    for label in ax1_local.get_xticklabels() + ax1_local.get_yticklabels():
        label.set_fontsize(16)

    principal_colors = ["r", "g", "b"]
    R = float(
        inversion_output.get("R_median", utils_stress_mod.R_(inversion_output["principal_stresses"]))
    )
    r_interval = _get_interval(hdi_info, "R")

    Rs_all = np.zeros(n_resamplings, dtype=np.float32)
    for b in range(n_resamplings):
        Rs_all[b] = utils_stress_mod.R_(inversion_output["boot_principal_stresses"][b, :])

    if use_hdi_samples and r_interval is not None:
        lo, hi = r_interval
        keep_idx = np.where((Rs_all >= lo) & (Rs_all <= hi))[0]
        if keep_idx.size == 0:
            keep_idx = np.arange(n_resamplings)
    else:
        keep_idx = np.arange(n_resamplings)

    if r_interval is not None:
        R_lo, R_hi = r_interval
    else:
        R_lo = np.percentile(Rs_all, (100.0 - confidence_level) / 2.0)
        R_hi = np.percentile(Rs_all, confidence_level + (100.0 - confidence_level) / 2.0)

    r_legend_label = "{:.0f}% {}, R={:.2f} ".format(confidence_level, interval_label, R)
    sigma_labels = []

    ms_boot = 5.0
    a_boot = 0.4
    m_boot = "."

    for k in range(3):
        az_best, pl_best = utils_stress_mod.get_bearing_plunge(
            inversion_output["principal_directions"][:, k]
        )
        ax1_local.line(
            pl_best,
            az_best,
            marker=marker,
            markeredgecolor="None",
            markeredgewidth=2,
            markerfacecolor=principal_colors[k],
            markersize=10,
            zorder=2,
        )

        boot_pd_stereo = np.zeros((n_resamplings, 2), dtype=np.float32)
        for b in range(n_resamplings):
            boot_pd_stereo[b, :] = utils_stress_mod.get_bearing_plunge(
                inversion_output["boot_principal_directions"][b, :, k]
            )

        count, lons_g, lats_g, ci_levels = utils_stress_mod.get_CI_levels(
            boot_pd_stereo[:, 0],
            boot_pd_stereo[:, 1],
            confidence_intervals=[confidence_level],
            nbins=200,
            smoothing_sig=1.5,
            return_count=True,
        )

        if show_confidence_contours:
            ax1_local.contourf(
                lons_g,
                lats_g,
                count,
                levels=[ci_levels[0], count.max() + 1e-9],
                colors=[principal_colors[k]],
                alpha=0.3,
                zorder=1.0,
            )
            ax1_local.contour(
                lons_g,
                lats_g,
                count,
                levels=ci_levels,
                vmin=0.0,
                linestyles=["solid", "dashed", "dashdot"][k],
                linewidths=0.75,
                colors=principal_colors[k],
                zorder=1.1,
            )
        else:
            boot_pd_to_plot = np.zeros((keep_idx.size, 2), dtype=np.float32)
            for j, b in enumerate(keep_idx):
                boot_pd_to_plot[j, :] = utils_stress_mod.get_bearing_plunge(
                    inversion_output["boot_principal_directions"][b, :, k]
                )
            azs = np.mod(boot_pd_to_plot[:, 0], 360.0)
            pls = boot_pd_to_plot[:, 1]

            mask = pls < 0.0
            if np.any(mask):
                pls[mask] = -pls[mask]
                azs[mask] = (azs[mask] + 180.0) % 360.0

            if bootstrap_max_points is not None and azs.size > bootstrap_max_points:
                sub_idx = np.random.choice(azs.size, bootstrap_max_points, replace=False)
                azs, pls = azs[sub_idx], pls[sub_idx]

            ax1_local.line(
                pls,
                azs,
                linestyle="None",
                marker=m_boot,
                markersize=ms_boot,
                markerfacecolor=principal_colors[k],
                markeredgecolor="none",
                alpha=a_boot,
                zorder=0.9,
            )

        sigma_labels.append(r"$\sigma_{{{}}}$".format(k + 1))

    ax2_local.hist(Rs_all, range=(0.0, 1.0), bins=20, lw=2.5, color=color, histtype="step")
    ax2_local.set_xlabel("Shape Ratio", fontsize=15)
    ax2_local.set_ylabel("Number of Bootstrap Samples", fontsize=15)
    if show_hdi_on_hist:
        ax2_local.axvspan(R_lo, R_hi, color=color, alpha=0.15, label=r_legend_label)
        ax2_local.legend(loc="upper right", fontsize=12)

    fp_strikes = focals["optimum_strike"].values
    fp_dips = focals["optimum_dip"].values
    fp_rakes = focals["optimum_rake"].values
    normals_loc, _ = utils_stress_mod.normal_slip_vectors(fp_strikes, fp_dips, fp_rakes)
    normals_loc = normals_loc.T

    best_stress_tensor = inversion_output["stress_tensor"]
    _, _, shear_loc_vec = utils_stress_mod.compute_traction(best_stress_tensor, normals_loc)
    shear_mags = np.sqrt(np.sum(shear_loc_vec**2, axis=-1))

    regime_colors = {
        "SS": "green",
        "TF": "blue",
        "NF": "red",
        "NS": "yellow",
        "TS": "orange",
        "U": "black",
    }

    final_data = []
    final_labels = []
    final_colors = []
    if "regime" in focals.columns:
        regime_series = focals["regime"].fillna("U").astype(str).str.strip().str.upper()
        regime_series = regime_series.replace("", "U")
        present_regimes = set(regime_series.values)
        ordered_regimes = [r for r in regime_colors if r in present_regimes]
        unknown_regimes = [r for r in regime_series.unique().tolist() if r not in regime_colors]
        ordered_regimes.extend(unknown_regimes)

        for regime in ordered_regimes:
            mask = regime_series.values == regime
            if np.any(mask):
                final_data.append(shear_mags[mask])
                final_labels.append(regime)
                final_colors.append(regime_colors.get(regime, "black"))
    else:
        fault_angles = np.array([calc_fault_angle(r) for r in fp_rakes])
        idx_normal = fault_angles < -0.5
        idx_reverse = fault_angles > 0.5
        idx_ss = ~(idx_normal | idx_reverse)
        data_list = [shear_mags[idx_normal], shear_mags[idx_ss], shear_mags[idx_reverse]]
        labels_list = ["Normal", "Strike-Slip", "Reverse"]
        colors_list = ["red", "green", "blue"]

        for data, label, c in zip(data_list, labels_list, colors_list):
            final_data.append(data)
            final_labels.append(label)
            final_colors.append(c)

    ax3_local.hist(
        final_data,
        bins=20,
        stacked=True,
        label=final_labels,
        color=final_colors,
        alpha=0.7,
        edgecolor="k",
        linewidth=0.5,
    )
    ax3_local.set_ylabel("Number of Events", fontsize=15)
    ax3_local.set_xlabel("Shear stress $\\vert \\tau \\vert$", fontsize=15)
    ax3_local.legend(loc="upper left", fontsize=13)
    if not save_separate:
        ax3_local.tick_params(axis="both", labelsize=13)

    return sigma_labels, figures


def _build_stereonet_legend(
    *, confidence_level: float, interval_label: str, show_confidence_contours: bool
):
    if show_confidence_contours:
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="r", markeredgecolor=None, markersize=10),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="g", markeredgecolor=None, markersize=10),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="b", markeredgecolor=None, markersize=10),
            Patch(facecolor="r", alpha=0.3, linewidth=0),
            Patch(facecolor="g", alpha=0.3, linewidth=0),
            Patch(facecolor="b", alpha=0.3, linewidth=0),
        ]
        labels = [
            r"$\sigma_1$",
            r"$\sigma_2$",
            r"$\sigma_3$",
            r"{:.0f}% {}".format(confidence_level, interval_label),
            r"{:.0f}% {}".format(confidence_level, interval_label),
            r"{:.0f}% {}".format(confidence_level, interval_label),
        ]
        return handles, labels

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="r", markeredgecolor=None, markersize=10),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="g", markeredgecolor=None, markersize=10),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="b", markeredgecolor=None, markersize=10),
    ]
    labels = [r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$"]
    return handles, labels


# ---------------------------------------------------------------------------
# Exported main function
# ---------------------------------------------------------------------------
def _generate_figures_core(
    stress_pkl: str,
    optimum_focal_csv: str,
    output_dir: str,
    mu: float,
    ilsi_src_path: Optional[str],
    bootstrap_max_points: Optional[int],
    confidence_level: float,
    show_hdi_on_hist: bool,
    show_confidence_contours: bool,
    save_separate: bool,
    overwrite: bool,
    *,
    hdi_info: Optional[Dict[str, Sequence[float]]] = None,
    hdi_source: Optional[str] = None,
    interval_label: str = "CI",
    use_hdi_samples: bool = False,
    show_hdi_on_mohr: bool = False,
) -> Dict[str, str]:
    """
    Generate Mohr plots, PT axes and the stereonet/cloud plot using the provided data files.

    Returns a dict mapping logical figure names -> saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ensure local ilsi/utils_stress available
    global ilsi, utils_stress
    ilsi, utils_stress = _safe_import_ilsi(ilsi_src_path)

    # load data
    stress_out = joblib.load(stress_pkl)
    focals = pd.read_csv(optimum_focal_csv)
    if hdi_info is None and hdi_source:
        hdi_info = stress_out.get(hdi_source)

    def _fmt_interval(x: Optional[Sequence[float]], nd: int = 2) -> str:
        if not x or len(x) < 2:
            return ""
        try:
            lo, hi = float(x[0]), float(x[1])
        except Exception:
            return ""
        return f"[{lo:.{nd}f}, {hi:.{nd}f}]"

    def _get_interval(
        info: Optional[Dict[str, Sequence[float]]], key: str
    ) -> Optional[Tuple[float, float]]:
        if not isinstance(info, dict):
            return None
        val = info.get(key)
        if val is None or len(val) < 2:
            return None
        try:
            return (float(val[0]), float(val[1]))
        except Exception:
            return None

    stress_tensor = stress_out["stress_tensor"]

    # compute fault angle array (same logic as original)
    fault_angles = focals["optimum_rake"].apply(calc_fault_angle).values.flatten()

    # 1) Mohr plot colored by instability
    fig1_path = os.path.join(output_dir, "Mohr_average_stress_instability.png")
    if overwrite or not os.path.exists(fig1_path):
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        title_lines = [f"mu={mu:.2f}"]
        if show_hdi_on_mohr and isinstance(hdi_info, dict):
            mu_int = _fmt_interval(hdi_info.get("mu"))
            r_int = _fmt_interval(hdi_info.get("R"))
            if mu_int:
                title_lines[0] = f"mu={mu:.2f} {mu_int}"
            if "R_median" in stress_out:
                r_med = float(stress_out["R_median"])
                if r_int:
                    title_lines.append(
                        f"R={r_med:.2f}  {confidence_level:.0f}% {interval_label} {r_int}"
                    )
                else:
                    title_lines.append(f"R={r_med:.2f}")
        _plot_mohr(
            ax,
            stress_tensor,
            focals,
            mu,
            color_by=focals["instability"].values,
            cmap_name="jet",
            title_text="\n".join(title_lines),
            outname=fig1_path,
        )
        plt.close(fig)

    # 4) PT axes stereonet
    pt_path = os.path.join(output_dir, "PT_axes.png")
    if overwrite or not os.path.exists(pt_path):
        _plot_PT_axes(
            pt_path,
            focals["optimum_strike"].values,
            focals["optimum_dip"].values,
            focals["optimum_rake"].values,
            stress_tensor,
        )

    # 4b) Selected and Aux planes with instability
    sel_planes_path = os.path.join(output_dir, "selected_planes.png")
    aux_planes_path = os.path.join(output_dir, "aux_planes.png")

    if (
        overwrite
        or not os.path.exists(sel_planes_path)
        or not os.path.exists(aux_planes_path)
    ):
        # Data for selected
        sel_strikes = focals["optimum_strike"].values
        sel_dips = focals["optimum_dip"].values
        sel_rakes = focals["optimum_rake"].values
        sel_instability = focals["instability"].values

        # Plot Selected
        _plot_stereonet_instability(
            sel_planes_path,
            sel_strikes,
            sel_dips,
            sel_rakes,
            sel_instability,
            "Selected Planes Instability",
        )

        # Prepare Data for Aux
        # 1. Identify aux parameters
        # If optimum matches plane1, aux is plane2. Else aux is plane1.
        s1 = focals["strike1"].values
        d1 = focals["dip1"].values
        r1 = focals["rake1"].values
        s2 = focals["strike2"].values
        d2 = focals["dip2"].values
        r2 = focals["rake2"].values

        # Simple proximity check to decide selected index
        # (Assuming optimum comes from one of them)
        dist1 = np.abs(sel_strikes - s1) + np.abs(sel_dips - d1)
        dist2 = np.abs(sel_strikes - s2) + np.abs(sel_dips - d2)

        is_p1 = dist1 < dist2

        aux_strikes = np.where(is_p1, s2, s1)
        aux_dips = np.where(is_p1, d2, d1)
        aux_rakes = np.where(is_p1, r2, r1)

        # 2. Compute instability for aux
        # Need PD, R, mu
        pd_val = stress_out.get("principal_directions")
        if pd_val is None:
            _, pd_val = utils_stress.stress_tensor_eigendecomposition(stress_tensor)

        R_val = float(stress_out.get("R_median", 0.5))
        mu_val = float(stress_out.get("mu", mu))

        # Use ilsi to compute instability
        # We pass aux array as both set1 and set2 to get I for it
        # (compute_instability_parameter returns I for shape (N,2))
        try:
            I_aux_pair = ilsi.compute_instability_parameter(
                pd_val,
                R_val,
                mu_val,
                aux_strikes,
                aux_dips,
                aux_rakes,
                aux_strikes,
                aux_dips,
                aux_rakes,
                return_fault_planes=False,
            )
            aux_instability = I_aux_pair[:, 0]
        except Exception:
            # Fallback if calculation fails
            aux_instability = np.zeros_like(aux_strikes)

        # Plot Aux
        _plot_stereonet_instability(
            aux_planes_path,
            aux_strikes,
            aux_dips,
            aux_rakes,
            aux_instability,
            "Auxiliary Planes Instability",
        )

        generated_paths = {
            "mohr_instability": fig1_path,
            "pt_axes": pt_path,
            "selected_planes": sel_planes_path,
            "aux_planes": aux_planes_path,
        }
    else:
        generated_paths = {
            "mohr_instability": fig1_path,
            "pt_axes": pt_path,
        }

    # 5) The stereonet + bootstrap cloud and histograms
    separate_paths = {
        "stereonet_directions": os.path.join(output_dir, "stereonet_directions.png"),
        "shape_ratio_histogram": os.path.join(output_dir, "shape_ratio_histogram.png"),
        "shear_stress_histogram": os.path.join(output_dir, "shear_stress_histogram.png"),
    }
    fig4_path = os.path.join(output_dir, "average_stress_directions.png")

    do_separate = save_separate and (
        overwrite or not all(os.path.exists(p) for p in separate_paths.values())
    )
    do_combined = (not save_separate) and (overwrite or not os.path.exists(fig4_path))

    if do_separate or do_combined:
        fig = None
        axes = None
        if do_combined:
            fig = plt.figure("inverted_stress_tensors", figsize=(15, 5))
            gs = fig.add_gridspec(
                nrows=1,
                ncols=3,
                top=0.8,
                bottom=0.15,
                left=0.05,
                right=0.95,
                hspace=0.3,
                wspace=0.3,
            )
            axes = [
                fig.add_subplot(gs[0, 0], projection="stereonet"),
                fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[0, 2]),
            ]

        _, figs = _plot_single_method_output_panels(
            stress_out,
            focals,
            utils_stress,
            confidence_level=confidence_level,
            interval_label=interval_label,
            hdi_info=hdi_info,
            use_hdi_samples=use_hdi_samples,
            show_hdi_on_hist=show_hdi_on_hist,
            show_confidence_contours=show_confidence_contours,
            bootstrap_max_points=bootstrap_max_points,
            save_separate=do_separate,
            axes=axes,
        )

        handles, labels = _build_stereonet_legend(
            confidence_level=confidence_level,
            interval_label=interval_label,
            show_confidence_contours=show_confidence_contours,
        )

        if do_combined and fig is not None and axes is not None:
            axes[0].legend(
                handles=handles,
                labels=labels,
                bbox_to_anchor=(-0.2, 0.15),
                loc="center",
                ncol=1,
                borderaxespad=0.0,
                frameon=True,
                fontsize=14,
            )
            fig.savefig(fig4_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            generated_paths["stereonet_directions"] = fig4_path

        elif do_separate and len(figs) >= 3:
            ax_stereo = figs[0].axes[0]
            ax_stereo.legend(
                handles=handles,
                labels=labels,
                bbox_to_anchor=(1.1, 0.15),
                loc="center",
                ncol=1,
                borderaxespad=0.0,
                frameon=True,
                fontsize=14,
            )
            figs[0].savefig(separate_paths["stereonet_directions"], dpi=200, bbox_inches="tight")
            figs[1].savefig(separate_paths["shape_ratio_histogram"], dpi=200, bbox_inches="tight")
            figs[2].savefig(separate_paths["shear_stress_histogram"], dpi=200, bbox_inches="tight")

            for f in figs:
                plt.close(f)

            generated_paths.update(separate_paths)

    return generated_paths


def generate_figures(
    stress_pkl: str = "stress.pkl",
    optimum_focal_csv: str = "optimum_focal.csv",
    output_dir: str = "figures",
    mu: float = 0.65,
    ilsi_src_path: Optional[str] = DEFAULT_ILSI_SRC,
    bootstrap_max_points: Optional[int] = 5000,
    confidence_level: float = 95.0,
    show_hdi_on_hist: bool = True,
    show_confidence_contours: bool = True,
    save_separate: bool = True,
    overwrite: bool = True,
) -> Dict[str, str]:
    """
    Generate Mohr plots, PT axes and the stereonet/cloud plot using the provided data files.

    This wrapper targets bootstrap-based intervals (CI-style) from deterministic/SMC outputs.
    """
    return _generate_figures_core(
        stress_pkl=stress_pkl,
        optimum_focal_csv=optimum_focal_csv,
        output_dir=output_dir,
        mu=mu,
        ilsi_src_path=ilsi_src_path,
        bootstrap_max_points=bootstrap_max_points,
        confidence_level=confidence_level,
        show_hdi_on_hist=show_hdi_on_hist,
        show_confidence_contours=show_confidence_contours,
        save_separate=save_separate,
        overwrite=overwrite,
        interval_label="CI",
        use_hdi_samples=False,
        show_hdi_on_mohr=False,
    )


def generate_figures_hdi(
    stress_pkl: str = "stress.pkl",
    optimum_focal_csv: str = "optimum_focal.csv",
    output_dir: str = "figures",
    mu: float = 0.65,
    ilsi_src_path: Optional[str] = DEFAULT_ILSI_SRC,
    bootstrap_max_points: Optional[int] = 5000,
    hdi_prob: float = 0.95,
    hdi_info: Optional[Dict[str, Sequence[float]]] = None,
    show_hdi_on_hist: bool = True,
    show_confidence_contours: bool = True,
    save_separate: bool = True,
    overwrite: bool = True,
    use_hdi_samples: bool = True,
) -> Dict[str, str]:
    """
    Generate Mohr plots, PT axes and stereonet outputs using Bayesian HDI summaries.

    This wrapper prefers HDI intervals stored in the Bayesian inversion output.
    """
    confidence_level = float(hdi_prob) * 100.0
    hdi_source = None
    if hdi_info is None:
        if abs(float(hdi_prob) - 0.9) < 1e-6:
            hdi_source = "hdi_90"
        else:
            hdi_source = "hdi"
    return _generate_figures_core(
        stress_pkl=stress_pkl,
        optimum_focal_csv=optimum_focal_csv,
        output_dir=output_dir,
        mu=mu,
        ilsi_src_path=ilsi_src_path,
        bootstrap_max_points=bootstrap_max_points,
        confidence_level=confidence_level,
        show_hdi_on_hist=show_hdi_on_hist,
        show_confidence_contours=show_confidence_contours,
        save_separate=save_separate,
        overwrite=overwrite,
        hdi_info=hdi_info,
        hdi_source=hdi_source,
        interval_label="HDI",
        use_hdi_samples=use_hdi_samples,
        show_hdi_on_mohr=True,
    )


def _sanitize_arviz_name(name: str) -> str:
    name = str(name).strip()
    name = name.replace("[", "_").replace("]", "")
    name = name.replace(", ", "_").replace(",", "_")
    name = name.replace(" ", "_")
    return name


def arviz_uncertainty_metrics(idata, *, var_names: list[str]) -> dict[str, float]:
    """Return ArviZ uncertainty and convergence metrics for selected variables."""
    if idata is None or not hasattr(idata, "posterior"):
        return {}

    posterior_vars = set(getattr(idata.posterior, "data_vars", {}).keys())
    present = [v for v in var_names if v in posterior_vars]
    if not present:
        return {}

    try:
        import arviz as az

        df = az.summary(idata, var_names=present)
    except Exception:
        return {}

    out: dict[str, float] = {}
    for idx, row in df.iterrows():
        base = _sanitize_arviz_name(idx)
        for col, prefix in (
            ("r_hat", "az_rhat"),
            ("ess_bulk", "az_ess_bulk"),
            ("ess_tail", "az_ess_tail"),
        ):
            if col not in df.columns:
                continue
            try:
                out[f"{prefix}_{base}"] = float(row[col])
            except Exception:
                out[f"{prefix}_{base}"] = float("nan")

    out["az_n_chains"] = float(idata.posterior.sizes.get("chain", 1))
    out["az_n_draws"] = float(idata.posterior.sizes.get("draw", 0))
    return out


def write_arviz_outputs(
    idata,
    *,
    output_dir: str,
    var_names: list[str],
    hdi_prob: float = 0.9,
) -> None:
    """Write ArviZ summary text and posterior plot to output_dir."""
    if idata is None or not hasattr(idata, "posterior"):
        return

    posterior_vars = set(getattr(idata.posterior, "data_vars", {}).keys())
    present = [v for v in var_names if v in posterior_vars]
    if not present:
        return

    try:
        import arviz as az

        df = az.summary(idata, var_names=present)
        with open(os.path.join(output_dir, "arviz.summary.txt"), "w", encoding="utf-8") as f:
            f.write(df.to_string())
            f.write("\n")
    except Exception:
        pass

    try:
        import arviz as az

        numba_disable_prev = None
        try:
            import numba

            numba_disable_prev = numba.config.DISABLE_JIT
            numba.config.DISABLE_JIT = True
        except Exception:
            numba_disable_prev = None

        try:
            axes = az.plot_posterior(idata, var_names=present, hdi_prob=float(hdi_prob))
            try:
                fig = axes.ravel()[0].figure
            except Exception:
                fig = plt.gcf()
            fig.tight_layout()
            fig.savefig(
                os.path.join(output_dir, "arviz.plot_posterior_hdi90.png"),
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)
        finally:
            if numba_disable_prev is not None:
                try:
                    import numba

                    numba.config.DISABLE_JIT = numba_disable_prev
                except Exception:
                    pass
    except Exception:
        pass


def posterior_misfit_diagnostics(
    inversion_output,
    strikes_1,
    dips_1,
    rakes_1,
    strikes_2,
    dips_2,
    rakes_2,
    *,
    n_draws: int = 500,
    seed: int = 0,
    ilsi_src_path: Optional[str] = DEFAULT_ILSI_SRC,
):
    """Evaluate angular misfit statistics across posterior draws."""
    _, utils_mod = _safe_import_ilsi(ilsi_src_path)

    idata = inversion_output.get("idata")
    if idata is None or "Sigma" not in idata.posterior:
        return None

    Sigma_s = idata.posterior["Sigma"].stack(s=("chain", "draw")).values
    n_samples = Sigma_s.shape[2]
    if n_samples <= 0:
        return None

    rng = np.random.default_rng(seed)
    idx = rng.choice(n_samples, size=min(int(n_draws), n_samples), replace=False)

    p2_s = None
    if "p_plane2_post" in idata.posterior:
        p2_s = idata.posterior["p_plane2_post"].stack(s=("chain", "draw")).values

    plane_map_global = inversion_output.get("plane_selection_map")
    if p2_s is None and plane_map_global is None:
        return None

    misfit_draw_hard = np.empty(idx.size, dtype=float)
    misfit_draw_expected = np.empty(idx.size, dtype=float)

    for j, k in enumerate(idx):
        Sigma = Sigma_s[:, :, k]
        norm = np.max(np.abs(np.linalg.eigvalsh(Sigma)))
        Sigma = Sigma / (norm if norm > 0 else 1.0)

        if p2_s is not None:
            p2 = p2_s[:, k]
            sel2 = p2 > 0.5
            fp_strikes = np.where(sel2, strikes_2, strikes_1)
            fp_dips = np.where(sel2, dips_2, dips_1)
            fp_rakes = np.where(sel2, rakes_2, rakes_1)
        else:
            fp_strikes = np.where(plane_map_global == 0, strikes_1, strikes_2)
            fp_dips = np.where(plane_map_global == 0, dips_1, dips_2)
            fp_rakes = np.where(plane_map_global == 0, rakes_1, rakes_2)
            p2 = None

        misfit_draw_hard[j] = float(
            utils_mod.mean_angular_residual(Sigma, fp_strikes, fp_dips, fp_rakes)
        )

        if p2_s is not None:
            angles1 = np.abs(
                utils_mod.angular_residual(Sigma, strikes_1, dips_1, rakes_1)
            )
            angles2 = np.abs(
                utils_mod.angular_residual(Sigma, strikes_2, dips_2, rakes_2)
            )
            misfit_draw_expected[j] = float(np.mean((1.0 - p2) * angles1 + p2 * angles2))
        else:
            misfit_draw_expected[j] = misfit_draw_hard[j]

    best_j = int(np.argmin(misfit_draw_hard))
    best_k = int(idx[best_j])

    Sigma_best = Sigma_s[:, :, best_k]
    norm = np.max(np.abs(np.linalg.eigvalsh(Sigma_best)))
    Sigma_best = Sigma_best / (norm if norm > 0 else 1.0)

    if p2_s is not None:
        plane_map_best = (p2_s[:, best_k] > 0.5).astype(int)
    else:
        plane_map_best = np.asarray(plane_map_global, dtype=int)

    return {
        "n_eval_draws": int(idx.size),
        "misfit_draw_hard_min": float(np.min(misfit_draw_hard)),
        "misfit_draw_hard_median": float(np.median(misfit_draw_hard)),
        "misfit_draw_hard_mean": float(np.mean(misfit_draw_hard)),
        "misfit_draw_expected_min": float(np.min(misfit_draw_expected)),
        "misfit_draw_expected_median": float(np.median(misfit_draw_expected)),
        "misfit_draw_expected_mean": float(np.mean(misfit_draw_expected)),
        "bestfit_draw_index": best_k,
        "bestfit_misfit": float(misfit_draw_hard[best_j]),
        "bestfit_stress_tensor": Sigma_best,
        "bestfit_plane_map": plane_map_best,
    }


def plot_stereonet_planes_map(
    all_events: pd.DataFrame,
    selected_events: pd.DataFrame,
    output_png,
    color_column: str = "instability",
    hemisphere: str = "lower",
) -> bool:
    """Plot selected planes as map-positioned stereonet traces."""
    required = {"optimum_strike", "optimum_dip", color_column}
    if not required.issubset(selected_events.columns):
        return False

    coord_options = (
        ("easting", "northing", "Easting [m]", "Northing [m]", 100.0, 100.0),
        ("lon", "lat", "Longitude", "Latitude", None, None),
        ("longitude", "latitude", "Longitude", "Latitude", None, None),
    )

    coords = None
    for x_col, y_col, x_label, y_label, x_pad, y_pad in coord_options:
        if {x_col, y_col}.issubset(all_events.columns) and {x_col, y_col}.issubset(
            selected_events.columns
        ):
            coords = (x_col, y_col, x_label, y_label, x_pad, y_pad)
            break

    if coords is None:
        return False

    hemi = str(hemisphere or "lower").strip().lower()
    if hemi not in {"lower", "upper"}:
        raise ValueError("hemisphere must be either 'lower' or 'upper'")

    x_col, y_col, x_label, y_label, x_pad, y_pad = coords

    from matplotlib.cm import ScalarMappable
    from matplotlib.markers import MarkerStyle
    from matplotlib.path import Path as MplPath
    from matplotlib.transforms import Affine2D

    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot()

    ax.scatter(
        all_events[x_col].to_numpy(dtype=float),
        all_events[y_col].to_numpy(dtype=float),
        color="k",
        marker=".",
        s=1,
        alpha=0.35,
        zorder=-9,
    )

    plot_df = selected_events.copy()
    if "depth_km" in plot_df.columns:
        plot_df = plot_df.sort_values(by="depth_km", ascending=False)
    elif "depth" in plot_df.columns:
        plot_df = plot_df.sort_values(by="depth", ascending=False)

    cmap = plt.cm.jet
    cvals = plot_df[color_column].astype(float).values
    if np.all(np.isnan(cvals)):
        cvals = np.zeros_like(cvals)
    vmin = np.nanmin(cvals)
    vmax = np.nanmax(cvals)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        vmin, vmax = 0.0, 1.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    def get_stereonet_path(dip: float) -> MplPath:
        dip = max(0.0, min(90.0, dip))
        dip_rad = np.radians(dip)
        lambdas = np.linspace(-np.pi / 2, np.pi / 2, 50)
        factor = 0.5 / (1.0 + np.cos(lambdas) * np.sin(dip_rad))
        x_vals = np.cos(lambdas) * np.cos(dip_rad) * factor
        y_vals = np.sin(lambdas) * factor
        if hemi == "upper":
            x_vals = -x_vals
        verts = list(zip(x_vals, y_vals))
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 1)
        return MplPath(verts, codes)

    for _, event in plot_df.iterrows():
        if pd.isna(event.get(x_col)) or pd.isna(event.get(y_col)):
            continue
        if pd.isna(event.get("optimum_strike")) or pd.isna(event.get("optimum_dip")):
            continue

        mag = _row_magnitude(event, default=0.0)
        size = ((mag + 4.0) ** 2.0) * 12.0
        rgba = cmap(norm(float(event[color_column])))

        strike = float(event["optimum_strike"])
        dip = float(event["optimum_dip"])

        base_path = get_stereonet_path(dip)
        rotated_path = base_path.transformed(Affine2D().rotate_deg(-strike))
        marker = MarkerStyle(marker=rotated_path)

        ax.scatter(
            float(event[x_col]),
            float(event[y_col]),
            marker=marker,
            s=size,
            edgecolors=rgba,
            facecolors="none",
            linewidths=1.5,
            zorder=3,
        )

    x_all = all_events[x_col].to_numpy(dtype=float)
    y_all = all_events[y_col].to_numpy(dtype=float)
    x_min, x_max = np.nanmin(x_all), np.nanmax(x_all)
    y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)

    if x_pad is None:
        x_pad = 0.02 * max(x_max - x_min, 0.1)
    if y_pad is None:
        y_pad = 0.02 * max(y_max - y_min, 0.1)

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, shrink=0.3)
    cbar.set_label(color_column.replace("_", " ").title(), fontsize=12)

    ax.set_aspect("equal")
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title("Selected Planes", fontsize=16)
    try:
        ax.ticklabel_format(useOffset=False, style="plain")
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
    return True


def _row_magnitude(row: pd.Series, default: float = 3.0) -> float:
    for key in ("mw", "ml", "md", "ma", "magnitude"):
        if key in row.index:
            val = row[key]
            if pd.notna(val):
                return float(val)
    return float(default)


def plot_focal_mechanisms_map(
    all_events: pd.DataFrame,
    selected_events: pd.DataFrame,
    output_png: str,
    color_column: str = "instability",
    use_cartopy: bool = True,
) -> None:
    """Plot focal mechanisms on map, with optional Cartopy basemap."""
    if "lon" not in selected_events.columns or "lat" not in selected_events.columns:
        print("Skipping map plot: lon/lat columns are missing.")
        return
    if color_column not in selected_events.columns:
        print(f"Skipping map plot: '{color_column}' column is missing.")
        return

    lon_min = np.nanmin(all_events["lon"].values) if "lon" in all_events.columns else np.nanmin(selected_events["lon"].values)
    lon_max = np.nanmax(all_events["lon"].values) if "lon" in all_events.columns else np.nanmax(selected_events["lon"].values)
    lat_min = np.nanmin(all_events["lat"].values) if "lat" in all_events.columns else np.nanmin(selected_events["lat"].values)
    lat_max = np.nanmax(all_events["lat"].values) if "lat" in all_events.columns else np.nanmax(selected_events["lat"].values)

    lon_pad = 0.02 * max(lon_max - lon_min, 0.1)
    lat_pad = 0.02 * max(lat_max - lat_min, 0.1)

    data_crs = None
    projection = None
    fig = None
    ax = None
    if use_cartopy:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature

            projection = ccrs.Mercator(
                central_longitude=0.5 * (lon_min + lon_max),
                min_latitude=max(-80.0, lat_min - 1.0),
                max_latitude=min(84.0, lat_max + 1.0),
            )
            data_crs = ccrs.PlateCarree()

            fig = plt.figure(figsize=(10, 12))
            ax = fig.add_subplot(111, projection=projection)
            ax.set_extent(
                [lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad],
                crs=data_crs,
            )

            ax.add_feature(cfeature.LAND, facecolor="0.95", zorder=-20)
            ax.add_feature(cfeature.OCEAN, facecolor="0.90", zorder=-21)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=-19)
            ax.add_feature(cfeature.BORDERS, linewidth=0.6, zorder=-19)
            ax.add_feature(
                cfeature.NaturalEarthFeature(
                    "cultural",
                    "admin_1_states_provinces_lines",
                    "50m",
                    edgecolor="0.45",
                    facecolor="none",
                ),
                zorder=-18,
            )
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle="--")
            gl.top_labels = False
            gl.right_labels = False
        except Exception as exc:
            print(f"Cartopy unavailable ({exc}). Falling back to plain lon/lat map.")
            use_cartopy = False

    if not use_cartopy:
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
        ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)

    if fig is None or ax is None:
        return

    if "lon" in all_events.columns and "lat" in all_events.columns:
        scatter_kwargs = dict(color="k", marker=".", s=2, alpha=0.35, zorder=-10)
        if use_cartopy and data_crs is not None:
            scatter_kwargs["transform"] = data_crs
        ax.scatter(all_events["lon"].values, all_events["lat"].values, **scatter_kwargs)

    plot_df = selected_events.copy()
    if "depth" in plot_df.columns:
        plot_df = plot_df.sort_values(by="depth", ascending=False)

    cmap = plt.cm.jet
    cvals = plot_df[color_column].astype(float).values
    if np.all(np.isnan(cvals)):
        cvals = np.zeros_like(cvals)
    vmin = np.nanmin(cvals)
    vmax = np.nanmax(cvals)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        vmin, vmax = 0.0, 1.0
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    from matplotlib.cm import ScalarMappable
    from pyrocko import moment_tensor as pmt
    from pyrocko.plot import beachball

    for _, event in plot_df.iterrows():
        if pd.isna(event.get("optimum_strike")) or pd.isna(event.get("optimum_dip")) or pd.isna(event.get("optimum_rake")):
            continue
        if pd.isna(event.get("lon")) or pd.isna(event.get("lat")):
            continue

        mt = pmt.MomentTensor(
            strike=float(event["optimum_strike"]),
            dip=float(event["optimum_dip"]),
            rake=float(event["optimum_rake"]),
        )

        mag = _row_magnitude(event, default=3.0)
        size = (mag + 1.0) ** 1.5
        rgba = cmap(norm(float(event[color_column])))

        if use_cartopy and projection is not None and data_crs is not None:
            x_map, y_map = projection.transform_point(float(event["lon"]), float(event["lat"]), data_crs)
            position = (x_map, y_map)
        else:
            position = (float(event["lon"]), float(event["lat"]))

        beachball.plot_beachball_mpl(
            mt,
            ax,
            beachball_type="full",
            size=size,
            position=position,
            color_t=rgba[:3],
            linewidth=0.85,
        )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, shrink=0.9)
    cbar.set_label(color_column.replace("_", " ").title(), fontsize=12)

    ax.set_aspect("equal")
    if not use_cartopy:
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title("Map view", fontsize=16)
    try:
        ax.ticklabel_format(useOffset=False, style="plain")
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
