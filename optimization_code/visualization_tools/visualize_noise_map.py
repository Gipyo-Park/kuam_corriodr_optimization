import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_LAT_LIM = [35.535, 35.652]
CODE_LON_LIM = [129.020, 129.150]
V20_CRUISE_ALTITUDE_M = 750.0


def infer_target_altitudes(
    air_risk_path=PROJECT_ROOT / "air_risk_data" / "bird_riskmap_springfall_3d.npy",
    default_altitudes=None,
):
    """
    Return the altitude layer convention used by v20.

    v20 evaluates the current run at altitude_levels=[750.0] by default and
    selects nearest layers from each 3D map during loading.
    """
    if default_altitudes is None:
        default_altitudes = np.array([V20_CRUISE_ALTITUDE_M], dtype=float)
    default_altitudes = np.asarray(default_altitudes, dtype=float).ravel()
    if default_altitudes.size > 0:
        print(f"Altitude reference set from v20 altitude_levels: {default_altitudes.tolist()}")
        return default_altitudes

    candidates = [Path(air_risk_path)]
    for p in candidates:
        if not p.exists():
            continue
        try:
            raw = np.load(str(p), allow_pickle=True).item()
            if "altitude_vec" in raw:
                z = np.asarray(raw["altitude_vec"], dtype=float).ravel()
            elif "z_vec" in raw:
                z = np.asarray(raw["z_vec"], dtype=float).ravel()
            else:
                continue
            if z.size > 0:
                print(f"Altitude reference loaded from {p}: {z.tolist()}")
                return z.astype(float)
        except Exception as e:
            print(f"Altitude reference load failed ({p}): {e}")
    print(f"Altitude reference not found. Fallback to v20 cruise altitude [{V20_CRUISE_ALTITUDE_M}].")
    return np.array([V20_CRUISE_ALTITUDE_M], dtype=float)


def load_ground_risk_shape(grc_path=PROJECT_ROOT / "ground_risk_data" / "Modified_high_res_affected_population_GRC.npy"):
    if not os.path.exists(grc_path):
        raise FileNotFoundError(f"GRC file not found: {grc_path}")
    data = np.load(grc_path, allow_pickle=True)
    if data.ndim < 2:
        raise ValueError(f"Unexpected GRC ndim={data.ndim}, expected >=2")
    ny, nx = int(data.shape[0]), int(data.shape[1])
    return ny, nx


def _draw_roi_box(ax):
    rect = Rectangle(
        (CODE_LON_LIM[0], CODE_LAT_LIM[0]),
        CODE_LON_LIM[1] - CODE_LON_LIM[0],
        CODE_LAT_LIM[1] - CODE_LAT_LIM[0],
        fill=False,
        edgecolor="cyan",
        linewidth=2.0,
        linestyle="--",
        label="Main CODE extent",
    )
    ax.add_patch(rect)


def _validate_columns(df, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def remap_noise_with_coords(csv_path, ny, nx, metric_col="Lden_db"):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    _validate_columns(df, ["receiver_id", "order", "grid_i", "grid_j", "lat", "lon", metric_col])

    # Prefer explicit grid_i/grid_j mapping from with_coords CSV.
    work = df.copy()
    for col in ["receiver_id", "order", "grid_i", "grid_j"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in ["lat", "lon", metric_col]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    rows_total = int(len(work))
    invalid_numeric = int(
        work[["grid_i", "grid_j", "lat", "lon", metric_col]].isna().any(axis=1).sum()
    )
    work = work.dropna(subset=["grid_i", "grid_j", "lat", "lon", metric_col]).copy()

    work["grid_i"] = work["grid_i"].astype(int)
    work["grid_j"] = work["grid_j"].astype(int)
    # receiver_id and order should match for this dataset contract.
    order_receiver_mismatch = int(np.sum(work["receiver_id"].astype(int) != work["order"].astype(int)))

    oob_mask = (
        (work["grid_i"] < 0) | (work["grid_i"] >= nx) |
        (work["grid_j"] < 0) | (work["grid_j"] >= ny)
    )
    rows_oob = int(oob_mask.sum())
    work = work.loc[~oob_mask].copy()

    # Keep last row for duplicated cells.
    dup_cell_mask = work.duplicated(subset=["grid_i", "grid_j"], keep="last")
    rows_dup_cells = int(dup_cell_mask.sum())
    work = work.loc[~dup_cell_mask].copy()

    noise_grid = np.full((ny, nx), np.nan, dtype=float)
    lat_grid = np.full((ny, nx), np.nan, dtype=float)
    lon_grid = np.full((ny, nx), np.nan, dtype=float)
    rid_grid = np.full((ny, nx), np.nan, dtype=float)
    order_grid = np.full((ny, nx), np.nan, dtype=float)

    gi = work["grid_i"].to_numpy(dtype=int)
    gj = work["grid_j"].to_numpy(dtype=int)
    vv = work[metric_col].to_numpy(dtype=float)
    la = work["lat"].to_numpy(dtype=float)
    lo = work["lon"].to_numpy(dtype=float)
    rid = work["receiver_id"].to_numpy(dtype=float)
    od = work["order"].to_numpy(dtype=float)

    noise_grid[gj, gi] = vv
    lat_grid[gj, gi] = la
    lon_grid[gj, gi] = lo
    rid_grid[gj, gi] = rid
    order_grid[gj, gi] = od

    rows_mapped = int(len(work))
    rows_unmatched = int(rows_total - rows_mapped - invalid_numeric - rows_oob)

    finite = noise_grid[np.isfinite(noise_grid)]
    nan_ratio = float(np.isnan(noise_grid).sum() / noise_grid.size)
    neg_count = int(np.sum(finite < 0.0)) if finite.size else 0
    pvals = np.percentile(finite, [1, 25, 50, 75, 99]).tolist() if finite.size else [np.nan] * 5

    metadata = {
        "schema_version": "noise_3d_v1",
        "source_csv": str(csv_path),
        "csv_path": str(csv_path),
        "metric_col": str(metric_col),
        "grid_shape": [int(ny), int(nx)],
        "rows_total": rows_total,
        "mapped_rows": rows_mapped,
        "rows_mapped": rows_mapped,
        "rows_unmatched": rows_unmatched,
        "rows_invalid": invalid_numeric,
        "rows_invalid_numeric": invalid_numeric,
        "rows_oob": rows_oob,
        "rows_dup_dropped": rows_dup_cells,
        "rows_duplicate_cells_dropped": rows_dup_cells,
        "order_receiver_mismatch_rows": order_receiver_mismatch,
        "nan_ratio": nan_ratio,
        "noise_min_db": float(np.min(finite)) if finite.size else float("nan"),
        "noise_max_db": float(np.max(finite)) if finite.size else float("nan"),
        "noise_mean_db": float(np.mean(finite)) if finite.size else float("nan"),
        "noise_percentiles_db": {
            "p1": float(pvals[0]),
            "p25": float(pvals[1]),
            "p50": float(pvals[2]),
            "p75": float(pvals[3]),
            "p99": float(pvals[4]),
        },
        "negative_db_count": neg_count,
    }

    return noise_grid, lat_grid, lon_grid, rid_grid, order_grid, metadata


def log_mapping_summary(meta):
    print("\n[Noise Mapping Summary]")
    print(f"- rows_total: {meta['rows_total']}")
    print(f"- rows_mapped: {meta['rows_mapped']}")
    print(f"- rows_unmatched: {meta['rows_unmatched']}")
    print(f"- rows_invalid_numeric: {meta['rows_invalid_numeric']}")
    print(f"- rows_oob: {meta['rows_oob']}")
    print(f"- rows_duplicate_cells_dropped: {meta['rows_duplicate_cells_dropped']}")
    print(f"- order_receiver_mismatch_rows: {meta['order_receiver_mismatch_rows']}")
    print(f"- grid_shape: {meta['grid_shape']}")
    print(f"- nan_ratio: {meta['nan_ratio']:.4f}")
    print(
        f"- noise min/max/mean: "
        f"{meta['noise_min_db']:.3f} / {meta['noise_max_db']:.3f} / {meta['noise_mean_db']:.3f} dB"
    )
    q = meta["noise_percentiles_db"]
    print(f"- percentiles p1/p25/p50/p75/p99: {q['p1']:.3f} / {q['p25']:.3f} / {q['p50']:.3f} / {q['p75']:.3f} / {q['p99']:.3f}")
    print(f"- negative_db_count: {meta['negative_db_count']}")
    print(
        f"- sanity: rows_total == rows_mapped + rows_unmatched + rows_invalid_numeric + rows_oob ? "
        f"{meta['rows_total'] == (meta['rows_mapped'] + meta['rows_unmatched'] + meta['rows_invalid_numeric'] + meta['rows_oob'])}"
    )


def visualize_noise_heatmap_3view(
    noise_grid,
    lat_grid,
    lon_grid,
    save_path,
    show_plot=False,
):
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)

    ax = axes[0]
    im0 = ax.imshow(noise_grid, origin="lower", interpolation="nearest", cmap="RdYlGn_r")
    ax.set_title("Index/Grid View (grid_j, grid_i)")
    ax.set_xlabel("grid_i (X index)")
    ax.set_ylabel("grid_j (Y index)")
    fig.colorbar(im0, ax=ax, label="Lden (dB)")

    ax = axes[1]
    valid = np.isfinite(noise_grid) & np.isfinite(lat_grid) & np.isfinite(lon_grid)
    if np.any(valid):
        pcm = ax.scatter(
            lon_grid[valid],
            lat_grid[valid],
            c=noise_grid[valid],
            s=8,
            cmap="RdYlGn_r",
            linewidths=0.0,
        )
        fig.colorbar(pcm, ax=ax, label="Lden (dB)")
    _draw_roi_box(ax)
    ax.set_title("Georef Full View (lat/lon from CSV)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right")

    ax = axes[2]
    im2 = ax.imshow(
        noise_grid,
        origin="lower",
        extent=[CODE_LON_LIM[0], CODE_LON_LIM[1], CODE_LAT_LIM[0], CODE_LAT_LIM[1]],
        interpolation="nearest",
        aspect="auto",
        cmap="RdYlGn_r",
    )
    ax.set_xlim(CODE_LON_LIM)
    ax.set_ylim(CODE_LAT_LIM)
    ax.set_title("ROI View (main eval extent)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(im2, ax=ax, label="Lden (dB)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def visualize_noise_interpolation(noise_grid, save_path, show_plot=False):
    ny, nx = noise_grid.shape
    lat = np.linspace(CODE_LAT_LIM[0], CODE_LAT_LIM[1], ny)
    lon = np.linspace(CODE_LON_LIM[0], CODE_LON_LIM[1], nx)
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    lat_fine = np.linspace(CODE_LAT_LIM[0], CODE_LAT_LIM[1], ny * 3)
    lon_fine = np.linspace(CODE_LON_LIM[0], CODE_LON_LIM[1], nx * 3)
    lon_fine_2d, lat_fine_2d = np.meshgrid(lon_fine, lat_fine)

    vals = noise_grid.ravel()
    pts = np.column_stack([lat_2d.ravel(), lon_2d.ravel()])
    valid = np.isfinite(vals)
    if int(np.sum(valid)) < 3:
        print("Interpolation skipped: not enough valid points")
        return

    interp = griddata(
        pts[valid],
        vals[valid],
        (lat_fine_2d, lon_fine_2d),
        method="linear",
        fill_value=np.nan,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    ax = axes[0]
    im0 = ax.imshow(
        noise_grid,
        origin="lower",
        extent=[CODE_LON_LIM[0], CODE_LON_LIM[1], CODE_LAT_LIM[0], CODE_LAT_LIM[1]],
        interpolation="nearest",
        aspect="auto",
        cmap="RdYlGn_r",
    )
    ax.set_title("Original Discrete Grid")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(im0, ax=ax, label="Lden (dB)")

    ax = axes[1]
    im1 = ax.imshow(
        interp,
        origin="lower",
        extent=[CODE_LON_LIM[0], CODE_LON_LIM[1], CODE_LAT_LIM[0], CODE_LAT_LIM[1]],
        interpolation="bilinear",
        aspect="auto",
        cmap="RdYlGn_r",
    )
    ax.set_title("Bilinear Interpolated")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(im1, ax=ax, label="Lden (dB)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def export_noise_grid_npy(
    noise_grid,
    metadata,
    output_path=PROJECT_ROOT / "noise_data" / "noise_lden_grid.npy",
    target_altitudes=None,
):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    z_vec = (
        np.asarray(target_altitudes, dtype=float).ravel()
        if target_altitudes is not None
        else np.array([0.0], dtype=float)
    )
    if z_vec.size == 0:
        z_vec = np.array([0.0], dtype=float)
    base_2d = np.asarray(noise_grid, dtype=float)
    risk_3d = np.repeat(base_2d[:, :, np.newaxis], int(z_vec.size), axis=2)
    payload = {
        "Risk_3d": risk_3d,
        "z_vec": z_vec,
        "altitude_vec": z_vec.copy(),
        "lat_lim": CODE_LAT_LIM,
        "lon_lim": CODE_LON_LIM,
        "metadata": metadata,
    }
    np.save(str(out), payload, allow_pickle=True)
    loaded = np.load(str(out), allow_pickle=True).item()
    has_keys = all(k in loaded for k in ("Risk_3d", "z_vec", "lat_lim", "lon_lim", "metadata"))
    shape_ok = tuple(loaded["Risk_3d"].shape) == tuple(risk_3d.shape)
    dtype_ok = str(loaded["Risk_3d"].dtype)
    print(f"Saved: {out}")
    print(
        "NPY reload check: "
        f"keys_ok={has_keys}, "
        f"Risk_3d shape={loaded['Risk_3d'].shape}, expected={risk_3d.shape}, "
        f"dtype={dtype_ok}, "
        f"z_vec={loaded['z_vec'].tolist()}, "
        f"min/max={float(np.nanmin(loaded['Risk_3d'])):.3f}/{float(np.nanmax(loaded['Risk_3d'])):.3f}, "
        f"ok={bool(has_keys and shape_ok)}"
    )


def main():
    csv_path = PROJECT_ROOT / "noise_data" / "noise_output_lden_with_coords.csv"
    grc_path = PROJECT_ROOT / "ground_risk_data" / "Modified_high_res_affected_population_GRC.npy"
    output_dir = PROJECT_ROOT / "figure" / "noise_analysis"

    if not csv_path.exists():
        raise FileNotFoundError(f"Noise CSV not found: {csv_path}")
    ny, nx = load_ground_risk_shape(str(grc_path))
    print(f"Grid dimensions from GRC: Ny={ny}, Nx={nx}")

    noise_grid, lat_grid, lon_grid, _, _, meta = remap_noise_with_coords(
        str(csv_path), ny, nx, metric_col="Lden_db"
    )
    log_mapping_summary(meta)
    target_altitudes = infer_target_altitudes()
    risk_3d = np.repeat(np.asarray(noise_grid, dtype=float)[:, :, np.newaxis], int(target_altitudes.size), axis=2)
    print(f"Noise Risk_3d layer count set to {risk_3d.shape[2]} (match reference altitude layers).")
    noise_layer = risk_3d[:, :, 0]

    visualize_noise_heatmap_3view(
        noise_layer,
        lat_grid,
        lon_grid,
        save_path=str(output_dir / "noise_lden_heatmap_3view.png"),
        show_plot=False,
    )
    visualize_noise_interpolation(
        noise_layer,
        save_path=str(output_dir / "noise_lden_interpolation.png"),
        show_plot=False,
    )
    export_noise_grid_npy(
        noise_layer,
        meta,
        output_path=PROJECT_ROOT / "noise_data" / "noise_lden_grid.npy",
        target_altitudes=target_altitudes,
    )


if __name__ == "__main__":
    main()
