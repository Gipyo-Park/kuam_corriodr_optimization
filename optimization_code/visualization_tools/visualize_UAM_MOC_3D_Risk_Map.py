import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_LAT_LIM = [35.535, 35.652]
CODE_LON_LIM = [129.020, 129.150]
GROUND_RISK_PATH = PROJECT_ROOT / "ground_risk_data" / "Modified_high_res_affected_population_GRC.npy"
MOC_DIR = PROJECT_ROOT / "260608_MOC"
V20_VERTIPORT_ELEVATION_MSL_M = 150.0
V20_CRUISE_ALTITUDE_M = 750.0


def load_eval_grid_shape(grc_path=GROUND_RISK_PATH):
    data = np.load(str(grc_path), allow_pickle=True)
    return int(data.shape[0]), int(data.shape[1])


def _to_latlon_from_5179(x_2d, y_2d):
    try:
        import pyproj
    except Exception as e:
        raise RuntimeError("pyproj is required. Install with: pip install pyproj") from e

    tr = pyproj.Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)
    lon_2d, lat_2d = tr.transform(x_2d, y_2d)
    return np.asarray(lat_2d, dtype=float), np.asarray(lon_2d, dtype=float)


def _draw_roi_box(ax):
    rect = Rectangle(
        (CODE_LON_LIM[0], CODE_LAT_LIM[0]),
        CODE_LON_LIM[1] - CODE_LON_LIM[0],
        CODE_LAT_LIM[1] - CODE_LAT_LIM[0],
        fill=False,
        edgecolor="cyan",
        linewidth=2.0,
        linestyle="--",
        label="v20 eval extent",
    )
    ax.add_patch(rect)


def _fixed_agl_files(moc_dir):
    moc_dir = Path(moc_dir)
    prefix = "UAM_MOC_XYZ_risk_fixedAGL"
    files = {}
    for path in moc_dir.glob(f"{prefix}*.npy"):
        suffix = path.stem[len(prefix):]
        if suffix.isdigit():
            files[int(suffix)] = path
    if not files:
        raise FileNotFoundError(f"No fixed-AGL MOC files found in {moc_dir}")
    return dict(sorted(files.items()))


def load_fixed_agl_xyz(path):
    xyzr = np.asarray(np.load(str(path), allow_pickle=False), dtype=float)
    if xyzr.ndim != 2 or xyzr.shape[1] != 4:
        raise ValueError(f"{path} must have shape (N, 4), got {xyzr.shape}")
    x, y, z, risk = (xyzr[:, i] for i in range(4))
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    if unique_x.size * unique_y.size != xyzr.shape[0]:
        raise ValueError(f"{path} does not form a complete rectangular grid")
    ix = np.searchsorted(unique_x, x)
    iy = np.searchsorted(unique_y, y)
    grid = np.empty((unique_y.size, unique_x.size), dtype=np.uint8)
    grid[iy, ix] = (risk >= 0.5).astype(np.uint8)
    x_2d, y_2d = np.meshgrid(unique_x, unique_y)
    return {
        "grid": grid,
        "x_2d": x_2d,
        "y_2d": y_2d,
        "unique_x": unique_x,
        "unique_y": unique_y,
        "z_m": float(np.unique(z)[0]),
    }


def select_agl_for_v20(cruise_altitude_m=V20_CRUISE_ALTITUDE_M, vertiport_elevation_m=V20_VERTIPORT_ELEVATION_MSL_M, moc_dir=MOC_DIR):
    files = _fixed_agl_files(moc_dir)
    requested_agl = float(cruise_altitude_m) - float(vertiport_elevation_m)
    available = np.asarray(sorted(files), dtype=float)
    eligible = available[available <= requested_agl + 1e-9]
    selected_agl = int(eligible[-1] if eligible.size else available[0])
    return selected_agl, files[selected_agl], requested_agl, files


def align_fixed_agl_to_v20_eval_grid(source, ny, nx):
    try:
        import pyproj
    except Exception as e:
        raise RuntimeError("pyproj is required. Install with: pip install pyproj") from e

    target_lats = np.linspace(CODE_LAT_LIM[0], CODE_LAT_LIM[1], int(ny))
    target_lons = np.linspace(CODE_LON_LIM[0], CODE_LON_LIM[1], int(nx))
    target_lon_2d, target_lat_2d = np.meshgrid(target_lons, target_lats)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
    target_x, target_y = transformer.transform(target_lon_2d, target_lat_2d)

    unique_x = source["unique_x"]
    unique_y = source["unique_y"]
    dx = float(unique_x[1] - unique_x[0])
    dy = float(unique_y[1] - unique_y[0])
    source_i = np.rint((target_x - float(unique_x[0])) / dx).astype(int)
    source_j = np.rint((target_y - float(unique_y[0])) / dy).astype(int)
    source_i = np.clip(source_i, 0, unique_x.size - 1)
    source_j = np.clip(source_j, 0, unique_y.size - 1)
    inside = (
        (target_x >= float(unique_x[0]) - 0.5 * dx)
        & (target_x <= float(unique_x[-1]) + 0.5 * dx)
        & (target_y >= float(unique_y[0]) - 0.5 * dy)
        & (target_y <= float(unique_y[-1]) + 0.5 * dy)
    )

    eval_grid = np.ones((int(ny), int(nx)), dtype=np.uint8)
    eval_grid[inside] = source["grid"][source_j[inside], source_i[inside]]
    return eval_grid


def analyze_fixed_agl_maps(moc_dir=MOC_DIR):
    files = _fixed_agl_files(moc_dir)
    print("[MOC fixed-AGL files]")
    counts = {}
    for agl, path in files.items():
        src = load_fixed_agl_xyz(path)
        count = int(np.count_nonzero(src["grid"]))
        counts[agl] = count
        print(f"- AGL {agl:>3}m | {path.name} | shape={src['grid'].shape} | ones={count}")
    return counts


def visualize_v20_moc_fixed_agl(
    moc_dir=MOC_DIR,
    cruise_altitude_m=V20_CRUISE_ALTITUDE_M,
    vertiport_elevation_m=V20_VERTIPORT_ELEVATION_MSL_M,
    show_plot=True,
    save_path=None,
):
    ny, nx = load_eval_grid_shape()
    selected_agl, selected_path, requested_agl, files = select_agl_for_v20(
        cruise_altitude_m=cruise_altitude_m,
        vertiport_elevation_m=vertiport_elevation_m,
        moc_dir=moc_dir,
    )
    source = load_fixed_agl_xyz(selected_path)
    eval_grid = align_fixed_agl_to_v20_eval_grid(source, ny, nx)
    lat_2d, lon_2d = _to_latlon_from_5179(source["x_2d"], source["y_2d"])

    print(
        f"v20 MOC selection: cruise={float(cruise_altitude_m):.1f}m MSL, "
        f"vertiport={float(vertiport_elevation_m):.1f}m MSL, "
        f"requested_agl={requested_agl:.1f}m, selected_agl={selected_agl}m"
    )
    print(f"Selected MOC file: {selected_path}")
    print(f"v20 eval grid shape: {(ny, nx)}, ones_ratio={float(np.mean(eval_grid)):.4f}")

    cmap = ListedColormap(["#1f77b4", "#ff7f0e"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    ax = axes[0]
    pcm = ax.pcolormesh(lon_2d, lat_2d, source["grid"], shading="auto", cmap=cmap, norm=norm)
    _draw_roi_box(ax)
    ax.set_title(f"MOC source georef | fixed AGL {selected_agl}m")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right")
    fig.colorbar(pcm, ax=ax, shrink=0.9, label="MOC binary")

    ax = axes[1]
    pcm = ax.pcolormesh(lon_2d, lat_2d, source["grid"], shading="auto", cmap=cmap, norm=norm)
    _draw_roi_box(ax)
    ax.set_xlim(CODE_LON_LIM)
    ax.set_ylim(CODE_LAT_LIM)
    ax.set_title("MOC source georef - v20 ROI")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right")
    fig.colorbar(pcm, ax=ax, shrink=0.9, label="MOC binary")

    ax = axes[2]
    im = ax.imshow(
        eval_grid,
        origin="lower",
        extent=[CODE_LON_LIM[0], CODE_LON_LIM[1], CODE_LAT_LIM[0], CODE_LAT_LIM[1]],
        interpolation="nearest",
        aspect="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.set_title("MOC v20 eval grid")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(im, ax=ax, shrink=0.9, label="MOC binary")

    fig.suptitle("MOC Fixed-AGL Map Aligned to v20", fontsize=16)
    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def visualize_all_fixed_agl_index_maps(moc_dir=MOC_DIR, show_plot=True, save_path=None):
    files = _fixed_agl_files(moc_dir)
    rows = 2
    cols = math.ceil(len(files) / rows)
    cmap = ListedColormap(["#1f77b4", "#ff7f0e"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.8, rows * 4.5), squeeze=False, constrained_layout=True)
    axes = axes.flatten()
    im = None
    for ax, (agl, path) in zip(axes, files.items()):
        src = load_fixed_agl_xyz(path)
        im = ax.imshow(src["grid"], origin="lower", interpolation="nearest", cmap=cmap, norm=norm)
        ax.set_title(f"fixed AGL {agl}m")
        ax.set_xlabel("X index")
        ax.set_ylabel("Y index")
    for ax in axes[len(files):]:
        ax.axis("off")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.82)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["0", "1"])
        cbar.set_label("MOC binary")
    fig.suptitle("MOC Fixed-AGL Source Index Maps", fontsize=16)
    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    out_dir = PROJECT_ROOT / "figure"
    analyze_fixed_agl_maps(MOC_DIR)
    visualize_v20_moc_fixed_agl(
        moc_dir=MOC_DIR,
        show_plot=True,
        save_path=str(out_dir / "moc_fixed_agl_v20_alignment.png"),
    )
    visualize_all_fixed_agl_index_maps(
        moc_dir=MOC_DIR,
        show_plot=False,
        save_path=str(out_dir / "moc_fixed_agl_index_maps.png"),
    )
