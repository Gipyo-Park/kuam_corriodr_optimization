import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# Main pipeline currently uses these hardcoded bounds.
CODE_LAT_LIM = [35.5446, 35.6427]
CODE_LON_LIM = [129.0514, 129.1436]


def _load_dict_npy(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    data = np.load(path, allow_pickle=True).item()
    return data


def _to_latlon_from_5179(x_2d, y_2d):
    try:
        import pyproj
    except Exception as e:
        raise RuntimeError(
            "pyproj is required for georeferenced comparison. Install with: pip install pyproj"
        ) from e

    tr = pyproj.Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)
    lon_2d, lat_2d = tr.transform(x_2d, y_2d)
    return np.asarray(lat_2d, dtype=float), np.asarray(lon_2d, dtype=float)


def _pick_alt_index(z_vec, target_alt_m):
    z = np.asarray(z_vec, dtype=float).ravel()
    return int(np.argmin(np.abs(z - float(target_alt_m))))


def _draw_code_extent_box(ax, lon_lim, lat_lim, color="cyan", lw=2.0):
    w = float(lon_lim[1] - lon_lim[0])
    h = float(lat_lim[1] - lat_lim[0])
    rect = Rectangle(
        (float(lon_lim[0]), float(lat_lim[0])),
        w,
        h,
        fill=False,
        edgecolor=color,
        linewidth=float(lw),
        linestyle="--",
        label="Main code lat/lon extent",
    )
    ax.add_patch(rect)


def _report_extent(name, lat_2d, lon_2d):
    lat_min, lat_max = float(np.nanmin(lat_2d)), float(np.nanmax(lat_2d))
    lon_min, lon_max = float(np.nanmin(lon_2d)), float(np.nanmax(lon_2d))
    print(f"[{name}] data lat_lim: [{lat_min:.6f}, {lat_max:.6f}]")
    print(f"[{name}] data lon_lim: [{lon_min:.6f}, {lon_max:.6f}]")
    print(f"[{name}] code lat_lim: {CODE_LAT_LIM}")
    print(f"[{name}] code lon_lim: {CODE_LON_LIM}")
    lat_cov = (CODE_LAT_LIM[1] - CODE_LAT_LIM[0]) / (lat_max - lat_min)
    lon_cov = (CODE_LON_LIM[1] - CODE_LON_LIM[0]) / (lon_max - lon_min)
    print(f"[{name}] coverage ratio (lat/lon): {lat_cov:.3f} / {lon_cov:.3f}")


def _apply_geo_view(ax, use_roi):
    if use_roi:
        ax.set_xlim(CODE_LON_LIM)
        ax.set_ylim(CODE_LAT_LIM)


def _plot_main_eval_grid_georef(ax, layer_2d, cmap, title, vmin=None, vmax=None):
    # Main pipeline evaluates risks on a fixed Ny x Nx ROI grid, not on source X/Y georef.
    im = ax.imshow(
        layer_2d,
        origin="lower",
        extent=[CODE_LON_LIM[0], CODE_LON_LIM[1], CODE_LAT_LIM[0], CODE_LAT_LIM[1]],
        interpolation="nearest",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return im


def compare_alignment(
    bird_path="air_risk_data/bird_riskmap_springfall_3d.npy",
    moc_path="air_risk_data/UAM_MOC_3D_Risk_Map.npy",
    grc_path="high_res_affected_population_GRC.npy",
    target_alt_m=600.0,
    heading_index=0,
    out_path="figure/compare_risk_alignment.png",
    show_plot=True,
):
    bird = _load_dict_npy(bird_path)
    moc = _load_dict_npy(moc_path)

    bird_x = np.asarray(bird["X_2d"], dtype=float)
    bird_y = np.asarray(bird["Y_2d"], dtype=float)
    bird_r = np.asarray(bird["Risk_3d"], dtype=float)
    bird_z = np.asarray(bird["altitude_vec"], dtype=float)

    moc_x = np.asarray(moc["X_2d"], dtype=float)
    moc_y = np.asarray(moc["Y_2d"], dtype=float)
    moc_r = np.asarray(moc["Risk_3d"], dtype=float)
    moc_z = np.asarray(moc["z_vec"], dtype=float)

    bird_ai = _pick_alt_index(bird_z, target_alt_m)
    moc_ai = _pick_alt_index(moc_z, target_alt_m)

    bird_layer = bird_r[:, :, bird_ai]
    moc_layer = moc_r[:, :, moc_ai]

    bird_lat, bird_lon = _to_latlon_from_5179(bird_x, bird_y)
    moc_lat, moc_lon = _to_latlon_from_5179(moc_x, moc_y)

    print(f"Bird altitude requested={target_alt_m}m, selected={bird_z[bird_ai]}m")
    print(f"MOC altitude requested={target_alt_m}m, selected={moc_z[moc_ai]}m")
    _report_extent("Bird", bird_lat, bird_lon)
    _report_extent("MOC", moc_lat, moc_lon)

    # Ground risk slice used by main pipeline (Ny, Nx, heading)
    grc = np.load(grc_path, allow_pickle=True)
    selected = grc[:, :, 0, 3:]
    Ny, Nx, H = selected.shape
    hi = int(np.clip(heading_index, 0, H - 1))
    grc_layer = selected[:, :, hi]

    def _build_single_figure(use_roi, figure_title, save_path):
        fig, axes = plt.subplots(3, 2, figsize=(16, 18), constrained_layout=True)

        # 1) Bird existing visualize style (index-based)
        ax = axes[0, 0]
        ax.imshow(bird_layer.T, origin="lower", cmap="inferno", interpolation="nearest")
        ax.set_title(f"Bird index view (existing style) | alt={int(bird_z[bird_ai])}m")
        ax.set_xlabel("X index")
        ax.set_ylabel("Y index")

        # 2) Bird georeferenced view (lat/lon)
        ax = axes[0, 1]
        pcm = ax.pcolormesh(bird_lon, bird_lat, bird_layer, shading="auto", cmap="inferno")
        _draw_code_extent_box(ax, CODE_LON_LIM, CODE_LAT_LIM)
        _apply_geo_view(ax, use_roi)
        ax.set_title("Bird georeferenced view (EPSG:5179 -> WGS84)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(loc="upper right")
        fig.colorbar(pcm, ax=ax, shrink=0.85, label="Bird risk")

        # 3) MOC existing visualize style (index-based)
        ax = axes[1, 0]
        ax.imshow(moc_layer.T, origin="lower", cmap="tab10", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(f"MOC index view (existing style) | alt={int(moc_z[moc_ai])}m")
        ax.set_xlabel("X index")
        ax.set_ylabel("Y index")

        # 4) MOC georeferenced view (lat/lon)
        ax = axes[1, 1]
        pcm = ax.pcolormesh(moc_lon, moc_lat, moc_layer, shading="auto", cmap="tab10", vmin=0, vmax=1)
        _draw_code_extent_box(ax, CODE_LON_LIM, CODE_LAT_LIM)
        _apply_geo_view(ax, use_roi)
        ax.set_title("MOC georeferenced view (EPSG:5179 -> WGS84)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(loc="upper right")
        fig.colorbar(pcm, ax=ax, shrink=0.85, label="MOC binary")

        # 5) Ground current style (no rotation)
        ax = axes[2, 0]
        ax.imshow(grc_layer, origin="lower", cmap="jet")
        ax.set_title(f"Ground index view (no rotation) | heading idx={hi}")
        ax.set_xlabel("X index")
        ax.set_ylabel("Y index")

        # 6) Ground transpose view for orientation check
        ax = axes[2, 1]
        ax.imshow(grc_layer.T, origin="lower", cmap="jet")
        ax.set_title(f"Ground alternative view (transpose) | heading idx={hi}")
        ax.set_xlabel("X index")
        ax.set_ylabel("Y index")

        fig.suptitle(figure_title, fontsize=18)
        fig.savefig(save_path, dpi=150)
        print(f"Saved comparison figure: {save_path}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def _build_side_by_side_figure(save_path):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

        ax = axes[0, 0]
        pcm = ax.pcolormesh(bird_lon, bird_lat, bird_layer, shading="auto", cmap="inferno")
        _draw_code_extent_box(ax, CODE_LON_LIM, CODE_LAT_LIM)
        ax.set_title("Bird georef - Full extent")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(loc="upper right")
        fig.colorbar(pcm, ax=ax, shrink=0.85, label="Bird risk")

        ax = axes[0, 1]
        pcm = ax.pcolormesh(bird_lon, bird_lat, bird_layer, shading="auto", cmap="inferno")
        _draw_code_extent_box(ax, CODE_LON_LIM, CODE_LAT_LIM)
        _apply_geo_view(ax, use_roi=True)
        ax.set_title("Bird georef - ROI only")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(loc="upper right")
        fig.colorbar(pcm, ax=ax, shrink=0.85, label="Bird risk")

        ax = axes[1, 0]
        pcm = ax.pcolormesh(moc_lon, moc_lat, moc_layer, shading="auto", cmap="tab10", vmin=0, vmax=1)
        _draw_code_extent_box(ax, CODE_LON_LIM, CODE_LAT_LIM)
        ax.set_title("MOC georef - Full extent")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(loc="upper right")
        fig.colorbar(pcm, ax=ax, shrink=0.85, label="MOC binary")

        ax = axes[1, 1]
        pcm = ax.pcolormesh(moc_lon, moc_lat, moc_layer, shading="auto", cmap="tab10", vmin=0, vmax=1)
        _draw_code_extent_box(ax, CODE_LON_LIM, CODE_LAT_LIM)
        _apply_geo_view(ax, use_roi=True)
        ax.set_title("MOC georef - ROI only")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(loc="upper right")
        fig.colorbar(pcm, ax=ax, shrink=0.85, label="MOC binary")

        fig.suptitle("Full extent vs ROI-only direct comparison", fontsize=18)
        fig.savefig(save_path, dpi=150)
        print(f"Saved comparison figure: {save_path}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    def _build_main_eval_grid_figure(save_path):
        fig, axes = plt.subplots(3, 2, figsize=(16, 16), constrained_layout=True)

        ax = axes[0, 0]
        pcm = ax.pcolormesh(bird_lon, bird_lat, bird_layer, shading="auto", cmap="inferno")
        _draw_code_extent_box(ax, CODE_LON_LIM, CODE_LAT_LIM)
        _apply_geo_view(ax, use_roi=True)
        ax.set_title("Bird source georef (ROI clip)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(loc="upper right")
        fig.colorbar(pcm, ax=ax, shrink=0.85, label="Bird risk")

        ax = axes[0, 1]
        im = _plot_main_eval_grid_georef(
            ax,
            bird_layer,
            cmap="inferno",
            title="Bird main-eval georef (fills ROI grid)",
        )
        fig.colorbar(im, ax=ax, shrink=0.85, label="Bird risk")

        ax = axes[1, 0]
        pcm = ax.pcolormesh(moc_lon, moc_lat, moc_layer, shading="auto", cmap="tab10", vmin=0, vmax=1)
        _draw_code_extent_box(ax, CODE_LON_LIM, CODE_LAT_LIM)
        _apply_geo_view(ax, use_roi=True)
        ax.set_title("MOC source georef (ROI clip)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend(loc="upper right")
        fig.colorbar(pcm, ax=ax, shrink=0.85, label="MOC binary")

        ax = axes[1, 1]
        im = _plot_main_eval_grid_georef(
            ax,
            moc_layer,
            cmap="tab10",
            title="MOC main-eval georef (fills ROI grid)",
            vmin=0,
            vmax=1,
        )
        fig.colorbar(im, ax=ax, shrink=0.85, label="MOC binary")

        ax = axes[2, 0]
        im = _plot_main_eval_grid_georef(
            ax,
            grc_layer,
            cmap="jet",
            title=f"Ground main-eval georef (no rotation) | heading idx={hi}",
        )
        fig.colorbar(im, ax=ax, shrink=0.85, label="Ground risk")

        ax = axes[2, 1]
        im = _plot_main_eval_grid_georef(
            ax,
            grc_layer.T,
            cmap="jet",
            title=f"Ground transpose georef (orientation check) | heading idx={hi}",
        )
        fig.colorbar(im, ax=ax, shrink=0.85, label="Ground risk")

        fig.suptitle("Source georef vs Main evaluation-grid georef", fontsize=18)
        fig.savefig(save_path, dpi=150)
        print(f"Saved comparison figure: {save_path}")
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_base, out_ext = os.path.splitext(out_path)
    if not out_ext:
        out_ext = ".png"

    full_out = f"{out_base}_full_extent{out_ext}"
    roi_out = f"{out_base}_roi_only{out_ext}"
    side_by_side_out = f"{out_base}_side_by_side{out_ext}"
    eval_grid_out = f"{out_base}_main_eval_grid{out_ext}"

    _build_single_figure(
        use_roi=False,
        figure_title="Risk Map Alignment Check - Full extent",
        save_path=full_out,
    )
    _build_single_figure(
        use_roi=True,
        figure_title="Risk Map Alignment Check - ROI only",
        save_path=roi_out,
    )
    _build_side_by_side_figure(save_path=side_by_side_out)
    _build_main_eval_grid_figure(save_path=eval_grid_out)

    # Keep backward-compatible output path by copying the side-by-side result naming convention.
    print(
        "Primary comparison outputs: "
        f"{full_out}, {roi_out}, {side_by_side_out}, {eval_grid_out}"
    )


if __name__ == "__main__":
    compare_alignment(
        bird_path="air_risk_data/bird_riskmap_springfall_3d.npy",
        moc_path="air_risk_data/UAM_MOC_3D_Risk_Map.npy",
        grc_path="high_res_affected_population_GRC.npy",
        target_alt_m=600.0,
        heading_index=0,
        out_path="figure/compare_risk_alignment.png",
        show_plot=True,
    )
