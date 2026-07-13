from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib_vertiport_wind"),
)

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pyproj import Transformer
from scipy.io import loadmat


VERTIPORT_LAT = 35.6033860
VERTIPORT_LON = 129.0780250
VERTIPORT_ALT_M = 150.0
TARGET_ALT_MSL = 550.0
CONE_TOP_ALT_MSL = 550.0
CLIMB_ANGLE_DEG = 8.0
ANALYSIS_RADIUS_M = 1000.0
N_SECTORS = 12
DEFAULT_Z_VEC = np.arange(0.0, 1000.0, 100.0)

SEASON_NAMES = ["spring", "summer", "autumn", "winter"]
SEASON_MONTHS = {
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
    "winter": [12, 1, 2],
}
SEASON_TITLES = {
    "spring": "봄 (3~5월)",
    "summer": "여름 (6~8월)",
    "autumn": "가을 (9~11월)",
    "winter": "겨울 (12~2월)",
}


def resolve_dirs(base_dir: Path) -> tuple[Path, Path]:
    return base_dir, base_dir / "python_outputs"


def month_mat_paths(data_dir: Path) -> list[Path]:
    paths = [data_dir / f"AirRisk_Data_{month}.mat" for month in range(1, 13)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing monthly MAT files:\n" + "\n".join(str(path) for path in missing)
        )
    return paths


def bearing_from_vec(u_east: float, v_north: float) -> float:
    """Return clockwise bearing from north for a vector pointing toward."""
    return float((np.degrees(np.arctan2(u_east, v_north)) + 360.0) % 360.0)


def sector_index_from_bearing(bearing_deg: float) -> int:
    """Match plot_new_moc_top6: S1=0~30 deg, S2=30~60 deg, ..."""
    return int(np.clip(bearing_deg // (360.0 / N_SECTORS), 0, N_SECTORS - 1))


def build_half_plane_mask(center_sector: int) -> list[int]:
    mask = [0] * N_SECTORS
    for offset in range(-2, 4):
        mask[(center_sector + offset) % N_SECTORS] = 1
    return mask


def safe_nanmean(arr: np.ndarray) -> float:
    if not np.any(np.isfinite(arr)):
        return 0.0
    return float(np.nanmean(arr))


def interpolate_z(arr_3d: np.ndarray, z_vec: np.ndarray, target_z: float) -> np.ndarray:
    if target_z <= z_vec[0]:
        return arr_3d[:, :, 0].astype(float)
    if target_z >= z_vec[-1]:
        return arr_3d[:, :, -1].astype(float)
    upper = int(np.searchsorted(z_vec, target_z, side="right"))
    lower = upper - 1
    weight = (target_z - z_vec[lower]) / (z_vec[upper] - z_vec[lower])
    return (
        (1.0 - weight) * arr_3d[:, :, lower].astype(float)
        + weight * arr_3d[:, :, upper].astype(float)
    )


def load_monthly_wind(paths: list[Path]):
    monthly = []
    reference_x = None
    reference_y = None
    reference_z = None
    reference_shape = None

    for month, path in enumerate(paths, 1):
        data = loadmat(
            str(path),
            variable_names=["X_2d", "Y_2d", "z_vec", "U3d", "V3d", "theta3d"],
        )
        x_2d = np.asarray(data["X_2d"], dtype=float)
        y_2d = np.asarray(data["Y_2d"], dtype=float)
        u = np.asarray(data["U3d"], dtype=float)
        v = np.asarray(data["V3d"], dtype=float)
        theta = np.asarray(data["theta3d"], dtype=float)
        z_vec = (
            np.asarray(data["z_vec"], dtype=float).ravel()
            if "z_vec" in data
            else DEFAULT_Z_VEC.copy()
        )

        expected_shape = (*x_2d.shape, len(z_vec))
        if u.shape != expected_shape or v.shape != expected_shape:
            raise ValueError(
                f"{path.name}: U/V shape must be {expected_shape}, "
                f"got {u.shape}/{v.shape}"
            )
        if theta.shape != expected_shape:
            raise ValueError(f"{path.name}: theta3d shape differs from U/V")

        if reference_x is None:
            reference_x = x_2d
            reference_y = y_2d
            reference_z = z_vec
            reference_shape = expected_shape
        else:
            if expected_shape != reference_shape:
                raise ValueError(f"{path.name}: wind array shape differs from month 1")
            if not np.array_equal(x_2d, reference_x) or not np.array_equal(
                y_2d, reference_y
            ):
                raise ValueError(f"{path.name}: X_2d/Y_2d differs from month 1")
            if not np.array_equal(z_vec, reference_z):
                raise ValueError(f"{path.name}: z_vec differs from month 1")

        valid = (
            np.isfinite(u)
            & np.isfinite(v)
            & np.isfinite(theta)
            & (u != -1)
            & (u != 0)
            & (theta != -1)
            & (theta != 0)
        )
        monthly.append(
            {
                "month": month,
                "u": np.where(valid, u, np.nan),
                "v": np.where(valid, v, np.nan),
            }
        )

    return reference_x, reference_y, reference_z, monthly


def mean_vector_fields(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    u_stack = np.stack([row["u"] for row in rows], axis=0)
    v_stack = np.stack([row["v"] for row in rows], axis=0)
    u_count = np.count_nonzero(np.isfinite(u_stack), axis=0)
    v_count = np.count_nonzero(np.isfinite(v_stack), axis=0)
    u_mean = np.divide(
        np.nansum(u_stack, axis=0),
        u_count,
        out=np.full(u_count.shape, np.nan, dtype=float),
        where=u_count > 0,
    )
    v_mean = np.divide(
        np.nansum(v_stack, axis=0),
        v_count,
        out=np.full(v_count.shape, np.nan, dtype=float),
        where=v_count > 0,
    )
    return u_mean, v_mean


def make_recommendation(u_field: np.ndarray, v_field: np.ndarray, roi_mask: np.ndarray):
    u_mean = safe_nanmean(np.where(roi_mask, u_field, np.nan))
    v_mean = safe_nanmean(np.where(roi_mask, v_field, np.nan))
    toward_deg = bearing_from_vec(u_mean, v_mean)
    from_deg = (toward_deg + 180.0) % 360.0

    takeoff_center = sector_index_from_bearing(from_deg)
    landing_center = (takeoff_center + N_SECTORS // 2) % N_SECTORS
    return {
        "wind_u_mps": u_mean,
        "wind_v_mps": v_mean,
        "wind_speed_mps": float(np.hypot(u_mean, v_mean)),
        "wind_toward_deg": toward_deg,
        "wind_from_deg": from_deg,
        "takeoff_center_sector_1based": takeoff_center + 1,
        "landing_center_sector_1based": landing_center + 1,
        "takeoff_mask": build_half_plane_mask(takeoff_center),
        "landing_mask": build_half_plane_mask(landing_center),
    }


def and_masks(mask_list: list[list[int]]) -> list[int]:
    arr = np.asarray(mask_list, dtype=np.int32)
    return np.all(arr == 1, axis=0).astype(int).tolist()


def sample_mask_for_quiver(mask, target_arrows_across=9):
    active = np.asarray(mask, dtype=bool)
    rows, cols = np.where(active)
    sampled = np.zeros_like(active, dtype=bool)
    if rows.size == 0:
        return sampled
    row_span = int(rows.max() - rows.min() + 1)
    col_span = int(cols.max() - cols.min() + 1)
    step = max(
        1,
        int(np.floor(max(row_span, col_span) / target_arrows_across)),
    )
    sampled[
        rows.min() : rows.max() + 1 : step,
        cols.min() : cols.max() + 1 : step,
    ] = True
    return active & sampled


def add_sector_guide(ax, x_center: float, y_center: float, radius_m: float):
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(
        x_center + radius_m * np.cos(theta),
        y_center + radius_m * np.sin(theta),
        "r--",
        linewidth=1.1,
    )
    for sector in range(N_SECTORS):
        bearing = np.deg2rad(sector * 360.0 / N_SECTORS)
        ax.plot(
            [x_center, x_center + radius_m * np.sin(bearing)],
            [y_center, y_center + radius_m * np.cos(bearing)],
            color="#777777",
            linewidth=0.45,
            alpha=0.55,
        )
    ax.plot(
        x_center,
        y_center,
        "P",
        markersize=8,
        markerfacecolor="gold",
        markeredgecolor="black",
    )


def setup_map_axis(ax, x_center: float, y_center: float, radius_m: float):
    margin = radius_m * 1.08
    ax.set_xlim(x_center - margin, x_center + margin)
    ax.set_ylim(y_center - margin, y_center + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22)
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)


def plot_wind_field(
    ax,
    x_2d,
    y_2d,
    u,
    v,
    roi_mask,
    x_center,
    y_center,
    radius_m,
    vmax,
    direction_mode,
    title,
):
    valid = roi_mask & np.isfinite(u) & np.isfinite(v)
    if not np.any(valid):
        add_sector_guide(ax, x_center, y_center, radius_m)
        setup_map_axis(ax, x_center, y_center, radius_m)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.text(
            0.5,
            0.5,
            "유효 바람 자료 없음",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        return None

    speed = np.hypot(u, v)
    mesh = ax.pcolormesh(
        x_2d,
        y_2d,
        np.ma.masked_where(~valid, speed),
        cmap="viridis",
        norm=Normalize(vmin=0.0, vmax=vmax),
        shading="auto",
    )

    arrows = sample_mask_for_quiver(valid)
    arrow_sign = 1.0 if direction_mode == "to" else -1.0
    ax.quiver(
        x_2d[arrows],
        y_2d[arrows],
        arrow_sign * u[arrows],
        arrow_sign * v[arrows],
        color="black",
        angles="xy",
        scale_units="xy",
        scale=0.018,
        width=0.0025,
        headwidth=4.2,
        headlength=5.2,
        headaxislength=4.6,
        alpha=0.82,
    )

    mean_u = float(np.nanmean(u[valid]))
    mean_v = float(np.nanmean(v[valid]))
    toward = bearing_from_vec(mean_u, mean_v)
    from_deg = (toward + 180.0) % 360.0
    ax.text(
        0.02,
        0.02,
        f"평균 TO {toward:.1f}° / FROM {from_deg:.1f}°\n"
        f"평균 벡터풍속 {np.hypot(mean_u, mean_v):.2f} m/s",
        transform=ax.transAxes,
        fontsize=7.5,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.88},
    )
    add_sector_guide(ax, x_center, y_center, radius_m)
    setup_map_axis(ax, x_center, y_center, radius_m)
    ax.set_title(title, fontsize=10, fontweight="bold")
    return mesh


def save_altitude_figure(
    output_path: Path,
    title: str,
    x_2d,
    y_2d,
    z_vec,
    u_3d,
    v_3d,
    roi_mask,
    x_center,
    y_center,
    radius_m,
    vmax,
    direction_mode,
):
    fig, axes = plt.subplots(
        2,
        5,
        figsize=(22, 10.5),
        layout="constrained",
    )
    fig.set_constrained_layout_pads(
        w_pad=0.05,
        h_pad=0.08,
        wspace=0.04,
        hspace=0.08,
    )
    last_mesh = None
    for altitude_index, (ax, altitude) in enumerate(zip(axes.ravel(), z_vec)):
        mesh = plot_wind_field(
            ax,
            x_2d,
            y_2d,
            u_3d[:, :, altitude_index],
            v_3d[:, :, altitude_index],
            roi_mask,
            x_center,
            y_center,
            radius_m,
            vmax,
            direction_mode,
            f"{altitude:.0f} m MSL",
        )
        if mesh is not None:
            last_mesh = mesh
        if altitude_index % 5 == 0:
            ax.set_ylabel("EPSG:5179 Y [m]")
        if altitude_index >= 5:
            ax.set_xlabel("EPSG:5179 X [m]")

    if last_mesh is not None:
        colorbar = fig.colorbar(
            last_mesh,
            ax=axes,
            orientation="horizontal",
            fraction=0.035,
            pad=0.035,
            aspect=55,
        )
        colorbar.set_label("수평 풍속 sqrt(U²+V²) [m/s]")

    direction_text = (
        "TO 화살표: 공기가 실제로 향하는 방향"
        if direction_mode == "to"
        else "FROM 화살표: 바람이 불어오는 방향(TO의 정확한 반대)"
    )
    fig.suptitle(
        f"{title}\n{direction_text} | 동일 색상 범위 0~{vmax:.2f} m/s",
        fontsize=17,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_summary_figure(
    output_path,
    title,
    fields,
    layout,
    x_2d,
    y_2d,
    roi_mask,
    x_center,
    y_center,
    radius_m,
    vmax,
):
    fig, axes = plt.subplots(*layout, figsize=(18, 13), layout="constrained")
    fig.set_constrained_layout_pads(
        w_pad=0.05,
        h_pad=0.08,
        wspace=0.04,
        hspace=0.08,
    )
    last_mesh = None
    for ax, (panel_title, u, v) in zip(np.asarray(axes).ravel(), fields):
        mesh = plot_wind_field(
            ax,
            x_2d,
            y_2d,
            u,
            v,
            roi_mask,
            x_center,
            y_center,
            radius_m,
            vmax,
            "to",
            panel_title,
        )
        if mesh is not None:
            last_mesh = mesh
    if last_mesh is not None:
        colorbar = fig.colorbar(
            last_mesh,
            ax=axes,
            orientation="horizontal",
            fraction=0.035,
            pad=0.035,
            aspect=55,
        )
        colorbar.set_label("수평 풍속 sqrt(U²+V²) [m/s]")
    fig.suptitle(
        f"{title}\n화살표=바람이 향하는 방향(TO), 표기에는 FROM도 병기",
        fontsize=17,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    data_dir, out_dir = resolve_dirs(base_dir)
    altitude_dir = out_dir / "monthly_altitude_wind"
    out_dir.mkdir(parents=True, exist_ok=True)
    altitude_dir.mkdir(parents=True, exist_ok=True)

    paths = month_mat_paths(data_dir)
    x_2d, y_2d, z_vec, monthly = load_monthly_wind(paths)

    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:5179", always_xy=True
    )
    x_center, y_center = transformer.transform(VERTIPORT_LON, VERTIPORT_LAT)
    physical_cone_radius_m = (
        (CONE_TOP_ALT_MSL - VERTIPORT_ALT_M)
        / np.tan(np.deg2rad(CLIMB_ANGLE_DEG))
    )
    analysis_radius_m = ANALYSIS_RADIUS_M
    roi_mask = np.hypot(x_2d - x_center, y_2d - y_center) <= analysis_radius_m

    annual_u, annual_v = mean_vector_fields(monthly)
    all_speed = np.hypot(
        np.stack([row["u"] for row in monthly], axis=0),
        np.stack([row["v"] for row in monthly], axis=0),
    )
    roi_3d = np.broadcast_to(roi_mask[:, :, None], annual_u.shape)
    roi_4d = np.broadcast_to(roi_3d, all_speed.shape[1:])
    finite_speed = all_speed[:, roi_4d]
    vmax = float(np.nanmax(finite_speed))
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("No valid wind speed exists inside the evaluation cone")

    save_altitude_figure(
        out_dir / "annual_wind_by_altitude_to.png",
        "12개월 연평균 고도별 바람",
        x_2d,
        y_2d,
        z_vec,
        annual_u,
        annual_v,
        roi_mask,
        x_center,
        y_center,
        analysis_radius_m,
        vmax,
        "to",
    )
    save_altitude_figure(
        out_dir / "annual_wind_by_altitude_from.png",
        "12개월 연평균 고도별 바람",
        x_2d,
        y_2d,
        z_vec,
        annual_u,
        annual_v,
        roi_mask,
        x_center,
        y_center,
        analysis_radius_m,
        vmax,
        "from",
    )

    for row in monthly:
        month = row["month"]
        for direction_mode in ("to", "from"):
            save_altitude_figure(
                altitude_dir
                / f"month_{month:02d}_wind_by_altitude_{direction_mode}.png",
                f"{month}월 고도별 바람",
                x_2d,
                y_2d,
                z_vec,
                row["u"],
                row["v"],
                roi_mask,
                x_center,
                y_center,
                analysis_radius_m,
                vmax,
                direction_mode,
            )

    monthly_550 = []
    for row in monthly:
        monthly_550.append(
            (
                f"{row['month']}월",
                interpolate_z(row["u"], z_vec, TARGET_ALT_MSL),
                interpolate_z(row["v"], z_vec, TARGET_ALT_MSL),
            )
        )
    summary_vmax = max(
        float(np.nanmax(np.hypot(u[roi_mask], v[roi_mask])))
        for _, u, v in monthly_550
    )
    save_summary_figure(
        out_dir / "monthly_wind_plot_py.png",
        f"월별 평균 바람 ({TARGET_ALT_MSL:.0f} m MSL)",
        monthly_550,
        (3, 4),
        x_2d,
        y_2d,
        roi_mask,
        x_center,
        y_center,
        analysis_radius_m,
        summary_vmax,
    )

    seasonal_fields = []
    seasonal_recommendations = {}
    for season in SEASON_NAMES:
        rows = [monthly[month - 1] for month in SEASON_MONTHS[season]]
        season_u, season_v = mean_vector_fields(rows)
        u_550 = interpolate_z(season_u, z_vec, TARGET_ALT_MSL)
        v_550 = interpolate_z(season_v, z_vec, TARGET_ALT_MSL)
        seasonal_fields.append((SEASON_TITLES[season], u_550, v_550))
        seasonal_recommendations[season] = make_recommendation(
            u_550, v_550, roi_mask
        )
    save_summary_figure(
        out_dir / "seasonal_wind_plot_py.png",
        f"계절별 평균 바람 ({TARGET_ALT_MSL:.0f} m MSL)",
        seasonal_fields,
        (2, 2),
        x_2d,
        y_2d,
        roi_mask,
        x_center,
        y_center,
        analysis_radius_m,
        summary_vmax,
    )

    annual_550_u = interpolate_z(annual_u, z_vec, TARGET_ALT_MSL)
    annual_550_v = interpolate_z(annual_v, z_vec, TARGET_ALT_MSL)
    annual_recommendation = make_recommendation(
        annual_550_u, annual_550_v, roi_mask
    )
    monthly_recommendations = {
        str(row["month"]): make_recommendation(
            interpolate_z(row["u"], z_vec, TARGET_ALT_MSL),
            interpolate_z(row["v"], z_vec, TARGET_ALT_MSL),
            roi_mask,
        )
        for row in monthly
    }

    seasonal_takeoff_masks = [
        seasonal_recommendations[season]["takeoff_mask"] for season in SEASON_NAMES
    ]
    seasonal_landing_masks = [
        seasonal_recommendations[season]["landing_mask"] for season in SEASON_NAMES
    ]
    monthly_takeoff_masks = [
        monthly_recommendations[str(month)]["takeoff_mask"]
        for month in range(1, 13)
    ]
    monthly_landing_masks = [
        monthly_recommendations[str(month)]["landing_mask"]
        for month in range(1, 13)
    ]

    recommendations = {
        "metadata": {
            "source": "AirRisk_Data_1.mat ... AirRisk_Data_12.mat",
            "altitude_msl_m": TARGET_ALT_MSL,
            "analysis_radius_m": analysis_radius_m,
            "physical_cone_radius_m": physical_cone_radius_m,
            "direction_definition": {
                "TO": "wind vector direction using U3d/V3d",
                "FROM": "TO + 180 degrees",
            },
            "sector_definition": "S1=0~30 deg, ..., S12=330~360 deg",
        },
        "seasonal_annual": {
            "annual": annual_recommendation,
            **seasonal_recommendations,
        },
        "monthly": monthly_recommendations,
        "common_masks": {
            "seasonal_4way": {
                "takeoff": and_masks(seasonal_takeoff_masks),
                "landing": and_masks(seasonal_landing_masks),
            },
            "monthly_12way": {
                "takeoff": and_masks(monthly_takeoff_masks),
                "landing": and_masks(monthly_landing_masks),
            },
        },
    }
    json_path = out_dir / "wind_sector_recommendations.json"
    json_path.write_text(
        json.dumps(recommendations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Wind direction convention",
        "TO: wind is moving toward this bearing",
        "FROM: meteorological source direction = TO + 180 deg",
        "",
        f"Annual 550 m TO={annual_recommendation['wind_toward_deg']:.3f} deg",
        f"Annual 550 m FROM={annual_recommendation['wind_from_deg']:.3f} deg",
        f"Annual 550 m speed={annual_recommendation['wind_speed_mps']:.3f} m/s",
        "",
        "# Monthly recommendations",
    ]
    for month in range(1, 13):
        row = monthly_recommendations[str(month)]
        lines.append(
            f"{month:02d}: TO={row['wind_toward_deg']:.2f}, "
            f"FROM={row['wind_from_deg']:.2f}, "
            f"takeoff_center=S{row['takeoff_center_sector_1based']}, "
            f"landing_center=S{row['landing_center_sector_1based']}"
        )
    report_path = out_dir / "wind_sector_recommendations.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(
        "Annual 550 m vector: "
        f"TO={annual_recommendation['wind_toward_deg']:.2f} deg, "
        f"FROM={annual_recommendation['wind_from_deg']:.2f} deg, "
        f"speed={annual_recommendation['wind_speed_mps']:.3f} m/s"
    )
    print(f"[OK] altitude figures: {2 + 2 * len(monthly)}")
    print(f"[OK] saved: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
