import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


CODE_LAT_LIM = [35.5446, 35.6427]
CODE_LON_LIM = [129.0514, 129.1436]


def load_bird_risk_data(file_path):
    """Load and validate bird spring/fall 3D risk map structure."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = np.load(file_path, allow_pickle=True).item()
    required_keys = {"X_2d", "Y_2d", "Z_terrain_2d", "Risk_3d", "altitude_vec"}
    missing = required_keys - set(data.keys())
    if missing:
        raise KeyError(f"Missing required keys: {sorted(missing)}")

    risk_3d = data["Risk_3d"]
    altitude_vec = data["altitude_vec"]

    if risk_3d.ndim != 3:
        raise ValueError(f"Risk_3d must be 3D, got {risk_3d.ndim}D")

    if risk_3d.shape[2] != len(altitude_vec):
        raise ValueError(
            "Risk_3d third axis and altitude_vec length mismatch: "
            f"{risk_3d.shape[2]} vs {len(altitude_vec)}"
        )

    return data


def analyze_bird_risk_data(data):
    """Print detailed statistics to choose a stable visualization scale."""
    risk_3d = data["Risk_3d"].astype(np.float64)
    altitude_vec = data["altitude_vec"]

    finite = risk_3d[np.isfinite(risk_3d)]
    nonzero = finite[finite > 0]

    print("\n[Bird Risk Data Analysis]")
    print(f"- keys: {list(data.keys())}")
    print(f"- Risk_3d shape: {risk_3d.shape}, dtype: {data['Risk_3d'].dtype}")
    print(f"- altitude_vec: {altitude_vec}")
    print(f"- global min/max: {finite.min():.8f} / {finite.max():.8f}")
    print(f"- zero ratio (global): {(finite == 0).mean():.4f}")

    if nonzero.size == 0:
        # All-zero dataset fallback
        scale = {
            "vmin": 0.0,
            "vmax": 1.0,
            "has_nonzero": False,
        }
        print("- non-zero values: none (all values are 0)")
    else:
        p05, p50, p95, p99 = np.quantile(nonzero, [0.05, 0.50, 0.95, 0.99])
        vmax = float(p99)
        vmin = float(max(np.min(nonzero), p05 * 0.2))

        if vmin >= vmax:
            vmin = float(np.min(nonzero))

        scale = {
            "vmin": vmin,
            "vmax": vmax,
            "has_nonzero": True,
        }

        print(f"- non-zero count: {nonzero.size}")
        print(
            "- non-zero quantiles "
            f"(p05/p50/p95/p99): {p05:.8f} / {p50:.8f} / {p95:.8f} / {p99:.8f}"
        )
        print(f"- display scale (non-zero based): vmin={vmin:.8f}, vmax={vmax:.8f}")

    for i, altitude in enumerate(altitude_vec):
        layer = risk_3d[:, :, i]
        lv = layer[np.isfinite(layer)]
        lz = np.mean(lv == 0)
        lnz = lv[lv > 0]

        if lnz.size > 0:
            lq95 = np.quantile(lnz, 0.95)
            lmax = lnz.max()
            print(
                f"  altitude {int(altitude):>4} m | zero={lz:.4f} "
                f"nonzero_count={lnz.size:>6} nz_p95={lq95:.8f} nz_max={lmax:.8f}"
            )
        else:
            print(
                f"  altitude {int(altitude):>4} m | zero={lz:.4f} "
                "nonzero_count=     0 (all zero)"
            )

    return scale


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
        label="Main code lat/lon extent",
    )
    ax.add_patch(rect)


def plot_bird_alignment_compare(data, target_alt_m=600.0, show_plot=True, save_path=None):
    risk_3d = data["Risk_3d"].astype(np.float64)
    altitude_vec = np.asarray(data["altitude_vec"], dtype=float)
    x_2d = np.asarray(data["X_2d"], dtype=float)
    y_2d = np.asarray(data["Y_2d"], dtype=float)

    ai = int(np.argmin(np.abs(altitude_vec - float(target_alt_m))))
    alt_sel = float(altitude_vec[ai])
    layer = risk_3d[:, :, ai]
    lat_2d, lon_2d = _to_latlon_from_5179(x_2d, y_2d)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    ax = axes[0]
    pcm = ax.pcolormesh(lon_2d, lat_2d, layer, shading="auto", cmap="inferno")
    _draw_roi_box(ax)
    ax.set_title(f"Bird source georef - Full extent | alt={int(alt_sel)}m")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right")
    fig.colorbar(pcm, ax=ax, shrink=0.9, label="Bird risk")

    ax = axes[1]
    pcm = ax.pcolormesh(lon_2d, lat_2d, layer, shading="auto", cmap="inferno")
    _draw_roi_box(ax)
    ax.set_xlim(CODE_LON_LIM)
    ax.set_ylim(CODE_LAT_LIM)
    ax.set_title("Bird source georef - ROI only")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right")
    fig.colorbar(pcm, ax=ax, shrink=0.9, label="Bird risk")

    ax = axes[2]
    im = ax.imshow(
        layer,
        origin="lower",
        extent=[CODE_LON_LIM[0], CODE_LON_LIM[1], CODE_LAT_LIM[0], CODE_LAT_LIM[1]],
        interpolation="nearest",
        aspect="auto",
        cmap="inferno",
    )
    ax.set_title("Bird main-eval georef (fills ROI grid)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(im, ax=ax, shrink=0.9, label="Bird risk")

    fig.suptitle("Bird Risk Alignment Quick Compare", fontsize=16)

    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved compare figure: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def visualize_bird_risk_springfall(file_path, show_plot=True, save_path=None):
    data = load_bird_risk_data(file_path)
    scale = analyze_bird_risk_data(data)

    risk_3d = data["Risk_3d"].astype(np.float64)
    altitude_vec = data["altitude_vec"]

    num_altitudes = len(altitude_vec)
    rows = 2
    cols = math.ceil(num_altitudes / rows)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 5.5, rows * 4.8),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes.flatten()

    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(color="#d9d9d9")

    im = None
    for i, altitude in enumerate(altitude_vec):
        ax = axes[i]
        layer = risk_3d[:, :, i].T

        # Zero-dominant layers are clearer when zero is masked to background color.
        layer_masked = np.ma.masked_where(layer <= 0, layer)

        if np.ma.count(layer_masked) == 0 or not scale["has_nonzero"]:
            ax.imshow(
                np.zeros_like(layer),
                cmap="Greys",
                origin="lower",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            ax.text(
                0.5,
                0.5,
                "All values are 0",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="black",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
        else:
            im = ax.imshow(
                layer_masked,
                cmap=cmap,
                origin="lower",
                vmin=scale["vmin"],
                vmax=scale["vmax"],
                interpolation="nearest",
            )

        ax.set_title(f"Altitude: {int(altitude)} m")
        ax.set_xlabel("X-axis Grid Index")
        ax.set_ylabel("Y-axis Grid Index")

    for j in range(num_altitudes, len(axes)):
        axes[j].axis("off")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.82)
        cbar.set_label("Bird Risk (non-zero percentile scaled)")

    fig.suptitle("Bird Risk Map (Spring/Fall) by Altitude", fontsize=18)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    compare_out = os.path.join(os.path.dirname(save_path), "bird_risk_alignment_compare.png") if save_path else os.path.join("figure", "bird_risk_alignment_compare.png")
    plot_bird_alignment_compare(
        data,
        target_alt_m=600.0,
        show_plot=show_plot,
        save_path=compare_out,
    )


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    npy_file = os.path.join(base_dir, "air_risk_data", "bird_riskmap_springfall_3d.npy")
    output_img = os.path.join(base_dir, "figure", "bird_riskmap_springfall_3d.png")

    visualize_bird_risk_springfall(npy_file, show_plot=True, save_path=output_img)
