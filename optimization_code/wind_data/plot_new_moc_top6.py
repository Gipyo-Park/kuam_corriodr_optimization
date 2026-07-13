"""
12개 이착륙 섹터의 MOC, 연평균 바람, 지상/공중 위험도를 재평가한다.

결과는 wind_data/python_outputs 아래에 지도, 대시보드, CSV, 요약문으로 저장한다.
회랑폭과 사용자 지정 제외 섹터는 이 진단 평가에 사용하지 않는다.
"""
from __future__ import annotations

import csv
from io import BytesIO
import math
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "matplotlib_sector_evaluation"),
)

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle, Wedge
from PIL import Image
import pyproj
from scipy.io import loadmat


VERTIPORT_LAT = 35.6033860
VERTIPORT_LON = 129.0780250
VERTIPORT_ALT_M = 150.0
TARGET_ALT_MSL = 550.0
CONE_TOP_ALT_MSL = 550.0
CLIMB_ANGLE_DEG = 8.0
ANALYSIS_RADIUS_M = 1000.0
N_SECTORS = 12
WIND_THRESHOLD = 0.9
TOP_N = 8
REFERENCE_TAKEOFF_SECTOR = 7
REFERENCE_LANDING_SECTOR = 5
SHOW_PLOTS = False

MOC_SCORE_FORMULA = "MOC 안전점수 = 1 - (차단 셀 수 / 전체 셀 수)"
WIND_SCORE_FORMULA = "바람점수 = 해당 섹터 역풍성분 / 12개 섹터 중 최대 역풍성분"
RISK_SCORE_FORMULA = "통합위험도 = 0.5 x 지상위험도 + 0.5 x 공중위험도"
MOC_ISSUE_FORMULA = "MOC 문제점수 = (이륙 MOC 비율 + 착륙 MOC 비율) / 2"
WIND_GAP_FORMULA = (
    "바람 미달점수 = [max(0.9-이륙점수, 0)/0.9 + "
    "max(0.9-착륙점수, 0)/0.9] / 2"
)
CONDITION_GAP_FORMULA = (
    "조건 미달점수 = 0.5 x MOC 문제점수 + 0.5 x 바람 미달점수"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WIND_DIR = PROJECT_ROOT / "wind_data"
OUTPUT_DIR = WIND_DIR / "python_outputs"
MOC_DIR = PROJECT_ROOT / "260608_MOC"
MOC_FIXED_AGL400 = MOC_DIR / "UAM_MOC_XYZ_risk_fixedAGL400.npy"

AGL_MOC = MOC_FIXED_AGL400
VERTIPORT_MOC = MOC_FIXED_AGL400
AIR_RISK_PATH = PROJECT_ROOT / "air_risk_data" / "bird_riskmap_springfall_3d.npy"
GRC_PATH = PROJECT_ROOT / "ground_risk_data" / "Modified_high_res_affected_population_GRC.npy"


def to_5179(lat, lon):
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:5179", always_xy=True
    )
    x, y = transformer.transform(lon, lat)
    return float(x), float(y)


def sector_idx_for(rx, ry):
    bearing = (np.rad2deg(np.arctan2(rx, ry)) + 360.0) % 360.0
    return int(np.clip(bearing // (360.0 / N_SECTORS), 0, N_SECTORS - 1))


def sector_bearing(k):
    return ((k + 0.5) * 360.0 / N_SECTORS) % 360.0


def normalize01(values):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr)
    lo = float(np.nanmin(arr[finite]))
    hi = float(np.nanmax(arr[finite]))
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def sample_mask_for_quiver(mask, target_arrows_across=9):
    """Sample relative to the active mask extent, not the full source grid."""
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


def _grid_spacing_m(x_2d, y_2d):
    unique_x = np.unique(np.asarray(x_2d, dtype=float))
    unique_y = np.unique(np.asarray(y_2d, dtype=float))
    if unique_x.size < 2 or unique_y.size < 2:
        raise ValueError("Projected grid must contain at least two X/Y coordinates")
    dx = np.diff(unique_x)
    dy = np.diff(unique_y)
    if not np.allclose(dx, dx[0]) or not np.allclose(dy, dy[0]):
        raise ValueError("Projected grid spacing is not uniform")
    return float(dx[0]), float(dy[0])


def _validate_projected_grid(name, x_2d, y_2d, expected_shape=None):
    x_arr = np.asarray(x_2d, dtype=float)
    y_arr = np.asarray(y_2d, dtype=float)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"{name}: X/Y shapes differ: {x_arr.shape} vs {y_arr.shape}")
    if expected_shape is not None and x_arr.shape != tuple(expected_shape):
        raise ValueError(
            f"{name}: grid shape {x_arr.shape} != expected {tuple(expected_shape)}"
        )
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError(f"{name}: projected coordinates contain NaN or Inf")
    dx, dy = _grid_spacing_m(x_arr, y_arr)
    if not np.isclose(dx, 100.0) or not np.isclose(dy, 100.0):
        raise ValueError(f"{name}: expected 100 m spacing, got dx={dx}, dy={dy}")
    return dx, dy


def _alignment_row(
    layer,
    shape,
    x_values,
    y_values,
    dx_m,
    dy_m,
    cells_in_cone,
    note,
):
    return {
        "layer": layer,
        "shape": "x".join(str(v) for v in shape),
        "crs": "EPSG:5179",
        "x_min_m": float(np.nanmin(x_values)),
        "x_max_m": float(np.nanmax(x_values)),
        "y_min_m": float(np.nanmin(y_values)),
        "y_max_m": float(np.nanmax(y_values)),
        "dx_m": float(dx_m),
        "dy_m": float(dy_m),
        "cells_in_cone": int(cells_in_cone),
        "note": note,
    }


def _interp_along_z(arr_3d, z_vec, target_z):
    if target_z <= z_vec[0]:
        return arr_3d[:, :, 0].astype(float)
    if target_z >= z_vec[-1]:
        return arr_3d[:, :, -1].astype(float)
    k_hi = int(np.searchsorted(z_vec, target_z, side="right"))
    k_lo = max(k_hi - 1, 0)
    weight = (target_z - z_vec[k_lo]) / (z_vec[k_hi] - z_vec[k_lo])
    return (
        (1.0 - weight) * arr_3d[:, :, k_lo].astype(float)
        + weight * arr_3d[:, :, k_hi].astype(float)
    )


def _monthly_wind_paths():
    paths = [WIND_DIR / f"AirRisk_Data_{month}.mat" for month in range(1, 13)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing monthly wind files:\n"
            + "\n".join(str(path) for path in missing)
        )
    return paths


def load_annual_wind():
    paths = _monthly_wind_paths()

    base = loadmat(paths[0], variable_names=["X_2d", "Y_2d", "z_vec"])
    x_2d = np.asarray(base["X_2d"], dtype=float)
    y_2d = np.asarray(base["Y_2d"], dtype=float)
    z_vec = np.asarray(base["z_vec"], dtype=float).ravel()
    _validate_projected_grid("Annual wind base grid", x_2d, y_2d)
    shape_3d = (*x_2d.shape, len(z_vec))
    sum_u = np.zeros(shape_3d)
    sum_v = np.zeros(shape_3d)
    count = np.zeros(shape_3d, dtype=int)

    for path in paths:
        data = loadmat(
            path,
            variable_names=[
                "X_2d",
                "Y_2d",
                "z_vec",
                "U3d",
                "V3d",
                "theta3d",
            ],
        )
        month_x = np.asarray(data["X_2d"], dtype=float)
        month_y = np.asarray(data["Y_2d"], dtype=float)
        month_z = np.asarray(data["z_vec"], dtype=float).ravel()
        _validate_projected_grid(
            f"Wind grid {Path(path).name}",
            month_x,
            month_y,
            expected_shape=x_2d.shape,
        )
        if not np.array_equal(month_x, x_2d) or not np.array_equal(month_y, y_2d):
            raise ValueError(
                f"{Path(path).name}: X_2d/Y_2d differ from the annual base grid"
            )
        if not np.array_equal(month_z, z_vec):
            raise ValueError(f"{Path(path).name}: z_vec differs from the base file")
        u = np.asarray(data["U3d"], dtype=float)
        v = np.asarray(data["V3d"], dtype=float)
        theta = np.asarray(data["theta3d"], dtype=float)
        if u.shape != shape_3d or v.shape != shape_3d or theta.shape != shape_3d:
            raise ValueError(
                f"{Path(path).name}: wind array shape does not match {shape_3d}"
            )
        valid = (u != -1) & (u != 0) & (theta != -1) & (theta != 0)
        sum_u += np.where(valid, u, 0.0)
        sum_v += np.where(valid, v, 0.0)
        count += valid

    u_annual = np.divide(
        sum_u, count, out=np.zeros_like(sum_u), where=count > 0
    )
    v_annual = np.divide(
        sum_v, count, out=np.zeros_like(sum_v), where=count > 0
    )
    return (
        x_2d,
        y_2d,
        _interp_along_z(u_annual, z_vec, TARGET_ALT_MSL),
        _interp_along_z(v_annual, z_vec, TARGET_ALT_MSL),
        len(paths),
    )


def load_monthly_cone_mean_wind(vx, vy, cone_r):
    """Return monthly mean vectors for presentation; scoring remains cell-based."""
    paths = _monthly_wind_paths()
    monthly = []
    reference_x = None
    reference_y = None
    for month, path in enumerate(paths, 1):
        data = loadmat(
            path,
            variable_names=["X_2d", "Y_2d", "z_vec", "U3d", "V3d", "theta3d"],
        )
        x_2d = np.asarray(data["X_2d"], dtype=float)
        y_2d = np.asarray(data["Y_2d"], dtype=float)
        _validate_projected_grid(f"Monthly wind grid {month}", x_2d, y_2d)
        if reference_x is None:
            reference_x = x_2d
            reference_y = y_2d
        elif not np.array_equal(x_2d, reference_x) or not np.array_equal(
            y_2d, reference_y
        ):
            raise ValueError(f"Monthly wind grid {month} differs from month 1")
        z_vec = np.asarray(data["z_vec"], dtype=float).ravel()
        u = np.asarray(data["U3d"], dtype=float)
        v = np.asarray(data["V3d"], dtype=float)
        theta = np.asarray(data["theta3d"], dtype=float)
        valid = (u != -1) & (u != 0) & (theta != -1) & (theta != 0)
        u_valid = np.where(valid, u, np.nan)
        v_valid = np.where(valid, v, np.nan)
        u_target = _interp_along_z(u_valid, z_vec, TARGET_ALT_MSL)
        v_target = _interp_along_z(v_valid, z_vec, TARGET_ALT_MSL)
        in_cone = np.hypot(x_2d - vx, y_2d - vy) <= cone_r
        monthly.append(
            {
                "month": month,
                "u_mps": float(np.nanmean(u_target[in_cone])),
                "v_mps": float(np.nanmean(v_target[in_cone])),
            }
        )
    return monthly


def evaluate_sectors():
    vx, vy = to_5179(VERTIPORT_LAT, VERTIPORT_LON)
    physical_cone_r = (
        (CONE_TOP_ALT_MSL - VERTIPORT_ALT_M)
        / np.tan(np.deg2rad(CLIMB_ANGLE_DEG))
    )
    cone_r = ANALYSIS_RADIUS_M

    cruise_moc = np.load(AGL_MOC)
    xm, ym, cruise_risk = (
        cruise_moc[:, 0],
        cruise_moc[:, 1],
        cruise_moc[:, 3],
    )
    xu = np.sort(np.unique(xm))
    yu = np.sort(np.unique(ym))
    cruise_dx, cruise_dy = _grid_spacing_m(xu, yu)
    if not np.isclose(cruise_dx, 100.0) or not np.isclose(cruise_dy, 100.0):
        raise ValueError(
            f"Cruise MOC: expected 100 m spacing, got {cruise_dx}, {cruise_dy}"
        )
    cruise_grid = np.full((len(yu), len(xu)), np.nan)
    x_index = {value: idx for idx, value in enumerate(xu)}
    y_index = {value: idx for idx, value in enumerate(yu)}
    for row in cruise_moc:
        cruise_grid[y_index[row[1]], x_index[row[0]]] = float(row[3])
    cruise_x_grid, cruise_y_grid = np.meshgrid(xu, yu)
    cruise_grid_in_analysis = (
        np.hypot(cruise_x_grid - vx, cruise_y_grid - vy) <= cone_r
    ) & np.isfinite(cruise_grid)

    vmoc = np.load(VERTIPORT_MOC)
    vmoc_dx, vmoc_dy = _grid_spacing_m(vmoc[:, 0], vmoc[:, 1])
    if not np.isclose(vmoc_dx, 100.0) or not np.isclose(vmoc_dy, 100.0):
        raise ValueError(
            f"Vertiport MOC: expected 100 m spacing, got {vmoc_dx}, {vmoc_dy}"
        )
    vmoc_xu = np.sort(np.unique(vmoc[:, 0]))
    vmoc_yu = np.sort(np.unique(vmoc[:, 1]))
    vmoc_grid = np.full((vmoc_yu.size, vmoc_xu.size), np.nan)
    vmoc_ix = np.searchsorted(vmoc_xu, vmoc[:, 0])
    vmoc_iy = np.searchsorted(vmoc_yu, vmoc[:, 1])
    vmoc_grid[vmoc_iy, vmoc_ix] = vmoc[:, 3]
    vmoc_x_grid, vmoc_y_grid = np.meshgrid(vmoc_xu, vmoc_yu)
    vmoc_grid_in_cone = (
        np.hypot(vmoc_x_grid - vx, vmoc_y_grid - vy) <= cone_r
    ) & np.isfinite(vmoc_grid)
    rx_v = vmoc[:, 0] - vx
    ry_v = vmoc[:, 1] - vy
    in_cone_v = np.hypot(rx_v, ry_v) <= cone_r
    moc_total = np.zeros(N_SECTORS, dtype=int)
    moc_blocked = np.zeros(N_SECTORS, dtype=int)
    for idx in np.where(in_cone_v)[0]:
        sector = sector_idx_for(rx_v[idx], ry_v[idx])
        moc_total[sector] += 1
        if vmoc[idx, 3] >= 0.5:
            moc_blocked[sector] += 1
    moc_ratio = np.divide(
        moc_blocked,
        moc_total,
        out=np.zeros(N_SECTORS, dtype=float),
        where=moc_total > 0,
    )
    moc_safety = 1.0 - moc_ratio
    moc_requirement_met = moc_blocked == 0

    air_raw = np.load(AIR_RISK_PATH, allow_pickle=True).item()
    x_risk = np.asarray(air_raw["X_2d"], dtype=float)
    y_risk = np.asarray(air_raw["Y_2d"], dtype=float)
    risk_3d = np.asarray(air_raw["Risk_3d"], dtype=float)
    z_key = "z_vec" if "z_vec" in air_raw else "altitude_vec"
    z_risk = np.asarray(air_raw[z_key], dtype=float).ravel()
    risk_dx, risk_dy = _validate_projected_grid(
        "Air-risk reference grid", x_risk, y_risk
    )
    if risk_3d.shape[:2] != x_risk.shape:
        raise ValueError(
            f"Air risk shape {risk_3d.shape[:2]} != grid shape {x_risk.shape}"
        )
    in_cone_risk = np.hypot(x_risk - vx, y_risk - vy) <= cone_r
    air_cumulative = risk_3d[:, :, z_risk <= TARGET_ALT_MSL + 1e-6].sum(axis=2)
    air_grid_norm = normalize01(np.where(in_cone_risk, air_cumulative, np.nan))

    ground_raw = np.asarray(np.load(GRC_PATH, allow_pickle=True), dtype=float)
    if ground_raw.ndim != 4:
        raise ValueError(f"Ground risk must be 4D, got shape {ground_raw.shape}")
    ground = np.max(ground_raw, axis=(2, 3)).T
    if ground.shape != x_risk.shape:
        raise ValueError(
            "Ground risk must align after one spatial transpose: "
            f"{ground.shape} != air-risk grid {x_risk.shape}"
        )
    ground_grid_norm = normalize01(np.where(in_cone_risk, ground, np.nan))

    sector_ground_sum = np.zeros(N_SECTORS)
    sector_air_sum = np.zeros(N_SECTORS)
    sector_risk_count = np.zeros(N_SECTORS, dtype=int)
    for i, j in zip(*np.where(in_cone_risk)):
        sector = sector_idx_for(x_risk[i, j] - vx, y_risk[i, j] - vy)
        sector_ground_sum[sector] += ground_grid_norm[i, j]
        sector_air_sum[sector] += air_grid_norm[i, j]
        sector_risk_count[sector] += 1
    sector_ground_mean = np.divide(
        sector_ground_sum,
        sector_risk_count,
        out=np.zeros(N_SECTORS, dtype=float),
        where=sector_risk_count > 0,
    )
    sector_air_mean = np.divide(
        sector_air_sum,
        sector_risk_count,
        out=np.zeros(N_SECTORS, dtype=float),
        where=sector_risk_count > 0,
    )
    ground_score = normalize01(sector_ground_mean)
    air_score = normalize01(sector_air_mean)
    combined_risk = 0.5 * ground_score + 0.5 * air_score

    x_wind, y_wind, u_annual, v_annual, wind_file_count = load_annual_wind()
    _validate_projected_grid(
        "Annual wind grid", x_wind, y_wind, expected_shape=x_risk.shape
    )
    if not np.array_equal(x_wind, x_risk) or not np.array_equal(y_wind, y_risk):
        raise ValueError("Air-risk and annual-wind X_2d/Y_2d grids do not match")
    if u_annual.shape != x_risk.shape or v_annual.shape != x_risk.shape:
        raise ValueError("Annual wind U/V shapes do not match the reference grid")
    in_cone_wind = np.hypot(x_wind - vx, y_wind - vy) <= cone_r
    sum_u = np.zeros(N_SECTORS)
    sum_v = np.zeros(N_SECTORS)
    wind_count = np.zeros(N_SECTORS, dtype=int)
    for i, j in zip(*np.where(in_cone_wind)):
        sector = sector_idx_for(x_wind[i, j] - vx, y_wind[i, j] - vy)
        sum_u[sector] += u_annual[i, j]
        sum_v[sector] += v_annual[i, j]
        wind_count[sector] += 1
    u_avg = np.divide(
        sum_u,
        wind_count,
        out=np.zeros(N_SECTORS, dtype=float),
        where=wind_count > 0,
    )
    v_avg = np.divide(
        sum_v,
        wind_count,
        out=np.zeros(N_SECTORS, dtype=float),
        where=wind_count > 0,
    )

    wind_along = np.zeros(N_SECTORS)
    for sector in range(N_SECTORS):
        bearing_rad = np.deg2rad(sector_bearing(sector))
        wind_along[sector] = (
            u_avg[sector] * np.sin(bearing_rad)
            + v_avg[sector] * np.cos(bearing_rad)
        )
    takeoff_headwind = np.maximum(-wind_along, 0.0)
    landing_headwind = np.maximum(wind_along, 0.0)
    takeoff_wind_score = takeoff_headwind / max(takeoff_headwind.max(), 1e-12)
    landing_wind_score = landing_headwind / max(landing_headwind.max(), 1e-12)

    metrics = []
    for sector in range(N_SECTORS):
        speed = float(np.hypot(u_avg[sector], v_avg[sector]))
        toward = (
            np.rad2deg(np.arctan2(u_avg[sector], v_avg[sector])) + 360.0
        ) % 360.0
        metrics.append(
            {
                "sector": sector + 1,
                "bearing_deg": sector_bearing(sector),
                "moc_blocked_cells": int(moc_blocked[sector]),
                "moc_total_cells": int(moc_total[sector]),
                "moc_blocked_ratio": float(moc_ratio[sector]),
                "moc_safety_score": float(moc_safety[sector]),
                "moc_requirement_met": bool(moc_requirement_met[sector]),
                "wind_u_mps": float(u_avg[sector]),
                "wind_v_mps": float(v_avg[sector]),
                "wind_cell_count": int(wind_count[sector]),
                "wind_speed_mps": speed,
                "wind_toward_deg": float(toward),
                "wind_from_deg": float((toward + 180.0) % 360.0),
                "radial_wind_mps": float(wind_along[sector]),
                "takeoff_headwind_mps": float(takeoff_headwind[sector]),
                "landing_headwind_mps": float(landing_headwind[sector]),
                "takeoff_wind_score": float(takeoff_wind_score[sector]),
                "landing_wind_score": float(landing_wind_score[sector]),
                "ground_risk_score": float(ground_score[sector]),
                "air_risk_score": float(air_score[sector]),
                "risk_cell_count": int(sector_risk_count[sector]),
                "combined_risk_score": float(combined_risk[sector]),
            }
        )

    alignment_rows = [
        _alignment_row(
            "air_risk",
            x_risk.shape,
            x_risk,
            y_risk,
            risk_dx,
            risk_dy,
            np.count_nonzero(in_cone_risk),
            "Reference X_2d/Y_2d grid",
        ),
        _alignment_row(
            "annual_wind",
            x_wind.shape,
            x_wind,
            y_wind,
            risk_dx,
            risk_dy,
            np.count_nonzero(in_cone_wind),
            f"Exact coordinate match across {wind_file_count} monthly files",
        ),
        _alignment_row(
            "ground_risk",
            ground.shape,
            x_risk,
            y_risk,
            risk_dx,
            risk_dy,
            np.count_nonzero(in_cone_risk),
            "Spatial axes transposed once; coordinates inherited from air-risk grid",
        ),
        _alignment_row(
            "cruise_moc_fixedAGL400",
            cruise_grid.shape,
            xu,
            yu,
            cruise_dx,
            cruise_dy,
            np.count_nonzero(np.hypot(xm - vx, ym - vy) <= cone_r),
            "Separate cruise MOC source grid",
        ),
        _alignment_row(
            "vertiport_moc_angle08_alt600",
            (len(np.unique(vmoc[:, 1])), len(np.unique(vmoc[:, 0]))),
            vmoc[:, 0],
            vmoc[:, 1],
            vmoc_dx,
            vmoc_dy,
            np.count_nonzero(in_cone_v),
            "8 deg cone MOC clipped to the 1 km analysis radius",
        ),
    ]

    plot_data = {
        "vx": vx,
        "vy": vy,
        "cone_r": cone_r,
        "physical_cone_r": physical_cone_r,
        "cruise_x_km": (xu - vx) / 1000.0,
        "cruise_y_km": (yu - vy) / 1000.0,
        "cruise_grid": cruise_grid,
        "cruise_grid_in_analysis": cruise_grid_in_analysis,
        "vmoc_x_km": rx_v / 1000.0,
        "vmoc_y_km": ry_v / 1000.0,
        "vmoc_grid_x_km": (vmoc_x_grid - vx) / 1000.0,
        "vmoc_grid_y_km": (vmoc_y_grid - vy) / 1000.0,
        "vmoc_grid": vmoc_grid,
        "vmoc_grid_in_cone": vmoc_grid_in_cone,
        "vmoc_risk": vmoc[:, 3],
        "vmoc_in_cone": in_cone_v,
        "vmoc_blocked_mask": in_cone_v & (vmoc[:, 3] >= 0.5),
        "wind_x_km": (x_wind - vx) / 1000.0,
        "wind_y_km": (y_wind - vy) / 1000.0,
        "wind_u": u_annual,
        "wind_v": v_annual,
        "wind_in_cone": in_cone_wind,
        "wind_file_count": wind_file_count,
        "monthly_cone_wind": load_monthly_cone_mean_wind(vx, vy, cone_r),
        "risk_x_km": (x_risk - vx) / 1000.0,
        "risk_y_km": (y_risk - vy) / 1000.0,
        "risk_x": x_risk,
        "risk_y": y_risk,
        "ground_grid_norm": ground_grid_norm,
        "air_grid_norm": air_grid_norm,
        "risk_in_cone": in_cone_risk,
        "alignment_rows": alignment_rows,
    }
    return metrics, plot_data


def build_combination_ranking(metrics):
    combinations = []
    for takeoff in range(N_SECTORS):
        for landing in range(N_SECTORS):
            if takeoff == landing:
                continue
            t = metrics[takeoff]
            l = metrics[landing]
            combination_risk = 0.5 * (
                t["combined_risk_score"] + l["combined_risk_score"]
            )
            moc_issue_score = 0.5 * (
                t["moc_blocked_ratio"] + l["moc_blocked_ratio"]
            )
            takeoff_wind_gap = max(
                WIND_THRESHOLD - t["takeoff_wind_score"], 0.0
            ) / WIND_THRESHOLD
            landing_wind_gap = max(
                WIND_THRESHOLD - l["landing_wind_score"], 0.0
            ) / WIND_THRESHOLD
            wind_gap_score = 0.5 * (takeoff_wind_gap + landing_wind_gap)
            condition_gap_score = 0.5 * moc_issue_score + 0.5 * wind_gap_score
            required_conditions_met = (
                t["moc_requirement_met"]
                and l["moc_requirement_met"]
                and t["takeoff_wind_score"] >= WIND_THRESHOLD
                and l["landing_wind_score"] >= WIND_THRESHOLD
            )
            combinations.append(
                {
                    "takeoff_sector": takeoff + 1,
                    "landing_sector": landing + 1,
                    "takeoff_moc_ratio": t["moc_blocked_ratio"],
                    "landing_moc_ratio": l["moc_blocked_ratio"],
                    "takeoff_wind_score": t["takeoff_wind_score"],
                    "landing_wind_score": l["landing_wind_score"],
                    "ground_risk_score": 0.5
                    * (t["ground_risk_score"] + l["ground_risk_score"]),
                    "air_risk_score": 0.5
                    * (t["air_risk_score"] + l["air_risk_score"]),
                    "combined_risk_score": combination_risk,
                    "moc_issue_score": moc_issue_score,
                    "wind_gap_score": wind_gap_score,
                    "condition_gap_score": condition_gap_score,
                    "required_conditions_met": required_conditions_met,
                    "is_reference_s7_s5": (
                        takeoff + 1 == REFERENCE_TAKEOFF_SECTOR
                        and landing + 1 == REFERENCE_LANDING_SECTOR
                    ),
                }
            )

    combinations.sort(
        key=lambda row: (
            row["condition_gap_score"],
            row["combined_risk_score"],
            row["takeoff_sector"],
            row["landing_sector"],
        )
    )
    for rank, row in enumerate(combinations, 1):
        row["comparison_rank"] = rank
    return combinations


def validate_results(metrics, combinations):
    if len(metrics) != N_SECTORS:
        raise AssertionError(f"Expected {N_SECTORS} sector rows, got {len(metrics)}")
    if len(combinations) != N_SECTORS * (N_SECTORS - 1):
        raise AssertionError(
            f"Expected 132 takeoff/landing combinations, got {len(combinations)}"
        )

    score_fields = (
        "moc_blocked_ratio",
        "moc_safety_score",
        "takeoff_wind_score",
        "landing_wind_score",
        "ground_risk_score",
        "air_risk_score",
        "combined_risk_score",
    )
    for row in metrics:
        for field in score_fields:
            value = row[field]
            if not -1e-9 <= value <= 1.0 + 1e-9:
                raise AssertionError(
                    f"S{row['sector']} {field} outside [0, 1]: {value}"
                )

    for row in metrics:
        if row["moc_total_cells"] <= 0:
            raise AssertionError(
                f"S{row['sector']} has no MOC cells inside the analysis radius"
            )


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(metrics, combinations, plot_data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sector_path = OUTPUT_DIR / "sector_metrics.csv"
    combination_path = OUTPUT_DIR / "sector_combination_ranking.csv"
    summary_path = OUTPUT_DIR / "sector_selection_summary.txt"
    alignment_path = OUTPUT_DIR / "spatial_alignment_report.csv"

    sector_csv_rows = [
        {
            "섹터": row["sector"],
            "중심방향_deg": row["bearing_deg"],
            "MOC_차단셀수": row["moc_blocked_cells"],
            "MOC_전체셀수": row["moc_total_cells"],
            "MOC_차단비율": row["moc_blocked_ratio"],
            "MOC_안전점수": row["moc_safety_score"],
            "MOC_필수조건통과": row["moc_requirement_met"],
            "평균풍속_mps": row["wind_speed_mps"],
            "바람_셀수": row["wind_cell_count"],
            "바람이향하는방향_deg": row["wind_toward_deg"],
            "바람이불어오는방향_deg": row["wind_from_deg"],
            "이륙_역풍성분_mps": row["takeoff_headwind_mps"],
            "착륙_역풍성분_mps": row["landing_headwind_mps"],
            "이륙_바람점수": row["takeoff_wind_score"],
            "착륙_바람점수": row["landing_wind_score"],
            "지상위험도": row["ground_risk_score"],
            "공중위험도": row["air_risk_score"],
            "위험도_셀수": row["risk_cell_count"],
            "통합위험도": row["combined_risk_score"],
        }
        for row in metrics
    ]
    combination_csv_rows = [
        {
            "비교순위": row["comparison_rank"],
            "이륙섹터": row["takeoff_sector"],
            "착륙섹터": row["landing_sector"],
            "이륙_MOC차단비율": row["takeoff_moc_ratio"],
            "착륙_MOC차단비율": row["landing_moc_ratio"],
            "이륙_바람점수": row["takeoff_wind_score"],
            "착륙_바람점수": row["landing_wind_score"],
            "지상위험도": row["ground_risk_score"],
            "공중위험도": row["air_risk_score"],
            "통합위험도": row["combined_risk_score"],
            "MOC_문제점수": row["moc_issue_score"],
            "바람_기준미달점수": row["wind_gap_score"],
            "조건미달점수": row["condition_gap_score"],
            "필수조건통과": row["required_conditions_met"],
            "현재_S7_S5조합": row["is_reference_s7_s5"],
            "MOC_문제점수_계산식": MOC_ISSUE_FORMULA,
            "바람_기준미달점수_계산식": WIND_GAP_FORMULA,
            "조건미달점수_계산식": CONDITION_GAP_FORMULA,
        }
        for row in combinations
    ]
    write_csv(sector_path, sector_csv_rows, list(sector_csv_rows[0].keys()))
    write_csv(
        combination_path,
        combination_csv_rows,
        list(combination_csv_rows[0].keys()),
    )
    write_csv(
        alignment_path,
        plot_data["alignment_rows"],
        list(plot_data["alignment_rows"][0].keys()),
    )

    eligible_combinations = sorted(
        (row for row in combinations if row["required_conditions_met"]),
        key=lambda row: row["combined_risk_score"],
    )
    reference = next(
        row for row in combinations if row["is_reference_s7_s5"]
    )
    top_rows = combinations[:TOP_N]

    lines = [
        "12개 이착륙 섹터 재평가 결과",
        "=" * 42,
        f"평가 고도: {TARGET_ALT_MSL:.0f} m MSL",
        f"분석 반경: {plot_data['cone_r']:.1f} m",
        f"물리 원추 반경(550 m MSL, 8°): "
        f"{plot_data['physical_cone_r']:.1f} m",
        f"연평균 바람 자료: AirRisk_Data_*.mat {plot_data['wind_file_count']}개월",
        "바람점수 의미: 해당 방향에서 받을 수 있는 역풍의 상대적 크기",
        "0점은 해당 방향을 사용할 수 없다는 뜻이 아니라, 연평균 바람이 "
        "역풍으로 도움을 주지 않는다는 뜻",
        WIND_SCORE_FORMULA,
        f"바람 필수조건: 이륙/착륙 점수 각각 {WIND_THRESHOLD:.1f} 이상",
        "MOC 필수조건: 해당 섹터 차단 셀 0개",
        MOC_SCORE_FORMULA,
        RISK_SCORE_FORMULA,
        "회랑폭 및 사용자 수동 제외 섹터: 미적용",
        "",
        "필수조건 적용 결과",
        "-" * 42,
    ]
    if eligible_combinations:
        best = eligible_combinations[0]
        lines.append(
            f"필수조건 통과 최적 조합: 이륙 S{best['takeoff_sector']} / "
            f"착륙 S{best['landing_sector']}, "
            f"통합위험도={best['combined_risk_score']:.3f}"
        )
        lines.append(f"필수조건을 모두 만족한 조합 수: {len(eligible_combinations)}")
    else:
        lines.append("필수조건을 모두 만족하는 이착륙 조합이 없습니다.")

    lines.extend(
        [
            "",
            "조건 미달이 적은 순위 (비교용)",
            MOC_ISSUE_FORMULA,
            WIND_GAP_FORMULA,
            CONDITION_GAP_FORMULA,
            "모든 문제점수는 0에 가까울수록 유리",
            "-" * 42,
        ]
    )
    for row in top_rows:
        lines.append(
            f"{row['comparison_rank']:2d}. 이륙 S{row['takeoff_sector']}, "
            f"착륙 S{row['landing_sector']} | "
            f"조건 미달 점수={row['condition_gap_score']:.3f}, "
            f"통합위험도={row['combined_risk_score']:.3f}, "
            f"바람점수(이륙/착륙)={row['takeoff_wind_score']:.3f}/"
            f"{row['landing_wind_score']:.3f}, "
            f"MOC 비율(이륙/착륙)={100*row['takeoff_moc_ratio']:.1f}%/"
            f"{100*row['landing_moc_ratio']:.1f}%"
        )

    s7 = metrics[REFERENCE_TAKEOFF_SECTOR - 1]
    s5 = metrics[REFERENCE_LANDING_SECTOR - 1]
    lines.extend(
        [
            "",
            "현재 설정 S7 이륙 / S5 착륙",
            "-" * 42,
            f"비교 순위: {reference['comparison_rank']} / {len(combinations)}",
            f"S7: MOC={s7['moc_blocked_cells']}/{s7['moc_total_cells']} "
            f"({100*s7['moc_blocked_ratio']:.1f}%), "
            f"이륙 바람점수={s7['takeoff_wind_score']:.3f}, "
            f"통합위험도={s7['combined_risk_score']:.3f}",
            f"S5: MOC={s5['moc_blocked_cells']}/{s5['moc_total_cells']} "
            f"({100*s5['moc_blocked_ratio']:.1f}%), "
            f"착륙 바람점수={s5['landing_wind_score']:.3f}, "
            f"통합위험도={s5['combined_risk_score']:.3f}",
            f"조합: 조건 미달 점수={reference['condition_gap_score']:.3f}, "
            f"통합위험도={reference['combined_risk_score']:.3f}",
            "판정: 필수조건 미통과. S7은 MOC 0개 조건과 바람 0.9 조건에 "
            "조금 못 미치며, S5는 착륙 바람점수가 낮습니다.",
        ]
    )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_sector_wedges(ax, metrics, cone_r_km):
    step = 360.0 / N_SECTORS
    for idx, row in enumerate(metrics):
        bearing_start = idx * step
        bearing_end = bearing_start + step
        theta1 = (90.0 - bearing_end) % 360.0
        theta2 = (90.0 - bearing_start) % 360.0
        clear = row["moc_requirement_met"]
        face = (0.30, 0.75, 0.35, 0.10) if clear else (0.85, 0.25, 0.20, 0.12)
        edge = "forestgreen" if clear else "firebrick"
        linewidth = 2.4 if row["sector"] in {
            REFERENCE_TAKEOFF_SECTOR,
            REFERENCE_LANDING_SECTOR,
        } else 0.9
        ax.add_patch(
            Wedge(
                (0, 0),
                cone_r_km,
                theta1,
                theta2,
                facecolor=face,
                edgecolor=edge,
                linewidth=linewidth,
                zorder=5,
            )
        )
        angle = np.deg2rad(row["bearing_deg"])
        label_r = cone_r_km * 0.72
        x = label_r * np.sin(angle)
        y = label_r * np.cos(angle)
        label = (
            f"S{row['sector']}\n"
            f"MOC {100*row['moc_blocked_ratio']:.1f}%\n"
            f"이륙 {row['takeoff_wind_score']:.2f} / "
            f"착륙 {row['landing_wind_score']:.2f}"
        )
        box_edge = edge
        if row["sector"] == REFERENCE_TAKEOFF_SECTOR:
            box_edge = "darkgreen"
        elif row["sector"] == REFERENCE_LANDING_SECTOR:
            box_edge = "darkorange"
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": box_edge,
                "linewidth": 1.5,
                "alpha": 0.92,
            },
            zorder=10,
        )


def add_reference_arrows(ax, cone_r_km):
    def endpoint(sector, fraction):
        angle = np.deg2rad(sector_bearing(sector - 1))
        radius = cone_r_km * fraction
        return radius * np.sin(angle), radius * np.cos(angle)

    tx, ty = endpoint(REFERENCE_TAKEOFF_SECTOR, 0.95)
    ax.add_patch(
        FancyArrowPatch(
            (0, 0),
            (tx, ty),
            arrowstyle="-|>",
            mutation_scale=20,
            color="darkgreen",
            linewidth=3.0,
            zorder=12,
            label="현재 이륙 S7",
        )
    )
    lx, ly = endpoint(REFERENCE_LANDING_SECTOR, 0.95)
    ax.add_patch(
        FancyArrowPatch(
            (lx, ly),
            (0, 0),
            arrowstyle="-|>",
            mutation_scale=20,
            color="darkorange",
            linewidth=3.0,
            zorder=12,
            label="현재 착륙 S5",
        )
    )


def plot_sector_map(metrics, plot_data):
    fig = plt.figure(figsize=(17, 11), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[4.3, 1.25])
    ax = fig.add_subplot(grid[0, 0])
    ax_info = fig.add_subplot(grid[0, 1])
    cone_binary = np.where(plot_data["vmoc_grid"] >= 0.5, 1.0, 0.0)
    cone_display = np.ma.masked_where(
        ~plot_data["vmoc_grid_in_cone"],
        cone_binary,
    )
    ax.pcolormesh(
        plot_data["vmoc_grid_x_km"],
        plot_data["vmoc_grid_y_km"],
        cone_display,
        cmap=ListedColormap(["#d9efd5", "#e58b86"]),
        shading="auto",
        vmin=0,
        vmax=1,
        alpha=0.78,
        zorder=1,
    )

    blocked = plot_data["vmoc_blocked_mask"]
    ax.scatter(
        plot_data["vmoc_x_km"][blocked],
        plot_data["vmoc_y_km"][blocked],
        s=10,
        c="#7a0019",
        marker="s",
        alpha=0.95,
        label="원추 MOC 차단 셀 중심",
        zorder=6,
    )

    wind_mask = plot_data["wind_in_cone"]
    cone_r_km = plot_data["cone_r"] / 1000.0
    sector_angles = np.deg2rad(
        [row["bearing_deg"] for row in metrics]
    )
    mean_anchor_r = cone_r_km * 0.40
    mean_x = mean_anchor_r * np.sin(sector_angles)
    mean_y = mean_anchor_r * np.cos(sector_angles)
    sector_speed = np.array([row["wind_speed_mps"] for row in metrics])
    speed_limit = max(1.0, float(np.nanmax(sector_speed)))
    sector_quiver = ax.quiver(
        mean_x,
        mean_y,
        [row["wind_u_mps"] for row in metrics],
        [row["wind_v_mps"] for row in metrics],
        sector_speed,
        cmap="viridis",
        norm=matplotlib.colors.Normalize(vmin=0.0, vmax=speed_limit),
        angles="xy",
        scale_units="xy",
        scale=30,
        width=0.0045,
        alpha=0.95,
        zorder=9,
    )
    colorbar = fig.colorbar(
        sector_quiver,
        ax=ax,
        orientation="horizontal",
        fraction=0.035,
        pad=0.06,
    )
    colorbar.set_label("섹터별 12개월 연평균 풍속 [m/s]")
    add_sector_wedges(ax, metrics, cone_r_km)
    add_reference_arrows(ax, cone_r_km)
    circle = plt.Circle(
        (0, 0),
        cone_r_km,
        fill=False,
        linestyle="--",
        linewidth=1.8,
        color="black",
        zorder=8,
    )
    ax.add_patch(circle)
    ax.plot(0, 0, "kP", markersize=13, markerfacecolor="gold", zorder=13)

    half = cone_r_km * 1.16
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("버티포트 기준 X (km)")
    ax.set_ylabel("버티포트 기준 Y (km)")
    ax.set_title(
        "12개 섹터 원추 MOC 및 연평균 바람 진단 지도\n"
        "연한 배경=원추 MOC 셀 판정, 진한 사각형=차단 셀 중심",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        0.01,
        0.01,
        "연한 녹색: 원추 경로 통과 셀\n"
        "연한 분홍색: 원추 경로 차단 셀\n"
        "진한 사각형: 같은 차단 셀의 중심 위치\n"
        "화살표: 섹터별 12개월 연평균 바람\n"
        "화살표 방향=TO(바람이 향하는 방향), 길이·색=풍속\n"
        "기상학적 FROM 방향은 TO의 반대\n"
        "짙은 녹색/주황 화살표: 현재 S7 이륙/S5 착륙",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "gray"},
        zorder=20,
    )
    ax.legend(
        handles=[
            Patch(facecolor="#d9efd5", label="원추 MOC 통과 셀"),
            Patch(facecolor="#e58b86", label="원추 MOC 차단 셀"),
            Patch(facecolor="#7a0019", label="차단 셀 중심"),
            Patch(facecolor="#777777", label="화살표 = 섹터 연평균 바람"),
        ],
        loc="upper right",
        fontsize=9,
    )

    valid_speed = np.hypot(
        plot_data["wind_u"][wind_mask],
        plot_data["wind_v"][wind_mask],
    )
    nonzero = valid_speed > 1e-9
    annual_u = float(np.mean(plot_data["wind_u"][wind_mask][nonzero]))
    annual_v = float(np.mean(plot_data["wind_v"][wind_mask][nonzero]))
    annual_speed = float(np.hypot(annual_u, annual_v))
    annual_to = float((np.rad2deg(np.arctan2(annual_u, annual_v)) + 360.0) % 360.0)
    annual_from = (annual_to + 180.0) % 360.0
    ax_info.axis("off")
    ax_info.set_title(
        "섹터별 실제 평균 바람\n"
        "(TO=향하는 방향, FROM=불어오는 방향)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    table_rows = [
        [
            f"S{row['sector']}",
            f"{row['wind_toward_deg']:.1f}°",
            f"{row['wind_from_deg']:.1f}°",
            f"{row['wind_speed_mps']:.2f}",
            f"{row['wind_u_mps']:.2f}",
            f"{row['wind_v_mps']:.2f}",
        ]
        for row in metrics
    ]
    wind_table = ax_info.table(
        cellText=table_rows,
        colLabels=["섹터", "TO", "FROM", "풍속", "U", "V"],
        cellLoc="center",
        loc="upper center",
        bbox=[0.0, 0.32, 1.0, 0.62],
    )
    wind_table.auto_set_font_size(False)
    wind_table.set_fontsize(8.5)
    ax_info.text(
        0.0,
        0.27,
        "1 km 분석영역 연평균 바람",
        fontsize=11,
        fontweight="bold",
        transform=ax_info.transAxes,
    )
    ax_info.text(
        0.0,
        0.02,
        f"유효 셀 수: {int(nonzero.sum()):,}개\n"
        f"전체 평균 U/V: {annual_u:.2f} / {annual_v:.2f} m/s\n"
        f"전체 평균 풍속: {annual_speed:.2f} m/s\n"
        f"전체 평균 TO/FROM: {annual_to:.1f}° / {annual_from:.1f}°\n"
        f"섹터 평균 TO 방향 범위: "
        f"{min(row['wind_toward_deg'] for row in metrics):.1f}°~"
        f"{max(row['wind_toward_deg'] for row in metrics):.1f}°\n\n"
        "지도에는 셀별 화살표를 표시하지 않습니다.\n"
        "각 화살표는 해당 섹터 내부 셀의 연평균 U/V입니다.\n"
        "이 섹터 연평균 U/V가 바람 점수 계산에 사용됩니다.",
        fontsize=9.3,
        linespacing=1.45,
        va="bottom",
        transform=ax_info.transAxes,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#f7f7f7",
            "edgecolor": "gray",
        },
    )
    fig.savefig(
        OUTPUT_DIR / "sector_map_diagnostics.png",
        dpi=300,
        bbox_inches="tight",
    )
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_sector_annual_mean_wind(metrics, plot_data):
    """Plot one annual-mean U/V vector for each sector in a map-style figure."""
    cone_r_km = plot_data["cone_r"] / 1000.0
    sector_angles = np.deg2rad([row["bearing_deg"] for row in metrics])
    anchor_radius = cone_r_km * 0.48
    arrow_x = anchor_radius * np.sin(sector_angles)
    arrow_y = anchor_radius * np.cos(sector_angles)
    sector_u = np.array([row["wind_u_mps"] for row in metrics])
    sector_v = np.array([row["wind_v_mps"] for row in metrics])
    sector_speed = np.hypot(sector_u, sector_v)
    speed_limit = max(1.0, float(np.ceil(np.max(sector_speed))))

    fig, ax = plt.subplots(figsize=(9, 9), layout="constrained")
    for sector in range(N_SECTORS):
        boundary = np.deg2rad(sector * 360.0 / N_SECTORS)
        ax.plot(
            [0.0, cone_r_km * np.sin(boundary)],
            [0.0, cone_r_km * np.cos(boundary)],
            color="#999999",
            linewidth=0.8,
            alpha=0.8,
            zorder=1,
        )

    wind_quiver = ax.quiver(
        arrow_x,
        arrow_y,
        sector_u,
        sector_v,
        sector_speed,
        cmap="viridis",
        norm=matplotlib.colors.Normalize(vmin=0.0, vmax=speed_limit),
        angles="xy",
        scale_units="xy",
        scale=18,
        width=0.005,
        zorder=4,
    )

    for row in metrics:
        angle = np.deg2rad(row["bearing_deg"])
        label_r = cone_r_km * 0.88
        ax.text(
            label_r * np.sin(angle),
            label_r * np.cos(angle),
            f"S{row['sector']}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox={
                "facecolor": "white",
                "edgecolor": "#777777",
                "boxstyle": "round,pad=0.15",
                "alpha": 0.95,
            },
            zorder=7,
        )

    ax.add_patch(
        plt.Circle(
            (0.0, 0.0),
            cone_r_km,
            fill=False,
            color="black",
            linestyle="--",
            linewidth=1.8,
            zorder=6,
        )
    )
    ax.plot(
        0.0,
        0.0,
        "P",
        color="gold",
        markeredgecolor="black",
        markersize=12,
        zorder=8,
    )
    colorbar = fig.colorbar(
        wind_quiver,
        ax=ax,
        fraction=0.046,
        pad=0.035,
    )
    colorbar.set_label("섹터 연평균 풍속 [m/s]")

    half = cone_r_km * 1.12
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.set_xlabel("버티포트 기준 X [km]")
    ax.set_ylabel("버티포트 기준 Y [km]")
    ax.grid(True, alpha=0.22)
    ax.set_title(
        "섹터별 12개월 연평균 바람\n"
        "화살표=TO(향하는 방향), FROM=TO+180°, 색=풍속\n"
        "각 섹터 중심에 해당 섹터 평균 U/V 화살표 1개 표시",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    fig.savefig(
        OUTPUT_DIR / "sector_annual_mean_wind.png",
        dpi=300,
        bbox_inches="tight",
    )
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_sector_map_panels(metrics, plot_data):
    fig, axes = plt.subplots(2, 2, figsize=(17, 15))
    ax_moc, ax_cell_wind, ax_sector_wind, ax_scores = axes.ravel()
    cone_r_km = plot_data["cone_r"] / 1000.0
    half = cone_r_km * 1.12
    sector_step = 360.0 / N_SECTORS

    def setup_map_axis(ax, title, show_sector_labels=True):
        for sector in range(N_SECTORS):
            boundary = np.deg2rad(sector * sector_step)
            ax.plot(
                [0, cone_r_km * np.sin(boundary)],
                [0, cone_r_km * np.cos(boundary)],
                color="#777777",
                linewidth=0.8,
                alpha=0.75,
                zorder=3,
            )
            if show_sector_labels:
                center = np.deg2rad(sector_bearing(sector))
                label_r = cone_r_km * 0.83
                ax.text(
                    label_r * np.sin(center),
                    label_r * np.cos(center),
                    f"S{sector + 1}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": "#555555",
                        "alpha": 0.9,
                    },
                    zorder=10,
                )
        ax.add_patch(
            plt.Circle(
                (0, 0),
                cone_r_km,
                fill=False,
                linestyle="--",
                linewidth=1.5,
                color="black",
                zorder=8,
            )
        )
        ax.plot(0, 0, "P", color="gold", markeredgecolor="black", markersize=10)
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.22)
        ax.set_xlabel("버티포트 기준 X [km]")
        ax.set_ylabel("버티포트 기준 Y [km]")
        ax.set_title(title, fontsize=13, fontweight="bold")

    cruise_binary = np.where(plot_data["cruise_grid"] >= 0.5, 1.0, 0.0)
    cruise_display = np.ma.masked_where(
        ~plot_data["cruise_grid_in_analysis"],
        cruise_binary,
    )
    ax_moc.pcolormesh(
        plot_data["cruise_x_km"],
        plot_data["cruise_y_km"],
        cruise_display,
        cmap=ListedColormap(["#d9efd5", "#dc5a52"]),
        shading="auto",
        vmin=0,
        vmax=1,
        alpha=0.78,
        zorder=1,
    )
    blocked = plot_data["vmoc_blocked_mask"]
    ax_moc.scatter(
        plot_data["vmoc_x_km"][blocked],
        plot_data["vmoc_y_km"][blocked],
        s=15,
        c="#7a0019",
        marker="s",
        alpha=0.85,
        zorder=5,
    )
    setup_map_axis(
        ax_moc,
        "① MOC만 표시\n배경 적색=순항 MOC 차단, 진한 사각형=원추 내부 차단 셀",
    )
    for row in metrics:
        angle = np.deg2rad(row["bearing_deg"])
        label_r = cone_r_km * 0.60
        ax_moc.text(
            label_r * np.sin(angle),
            label_r * np.cos(angle),
            f"{100*row['moc_blocked_ratio']:.1f}%",
            ha="center",
            va="center",
            fontsize=8,
            color="firebrick" if row["moc_blocked_ratio"] > 0 else "forestgreen",
            fontweight="bold",
            zorder=11,
        )
    ax_moc.legend(
        handles=[
            Patch(facecolor="#d9efd5", label="순항 MOC 안전"),
            Patch(facecolor="#dc5a52", label="순항 MOC 차단"),
            Patch(facecolor="#7a0019", label="원추 내부 차단 셀"),
        ],
        loc="upper right",
        fontsize=8,
    )

    wind_mask = plot_data["wind_in_cone"]
    quiver_mask = sample_mask_for_quiver(wind_mask)
    sampled_speed = np.hypot(
        plot_data["wind_u"][quiver_mask],
        plot_data["wind_v"][quiver_mask],
    )
    speed_limit = max(1.0, float(np.nanpercentile(sampled_speed, 95)))
    cell_quiver = ax_cell_wind.quiver(
        plot_data["wind_x_km"][quiver_mask],
        plot_data["wind_y_km"][quiver_mask],
        plot_data["wind_u"][quiver_mask],
        plot_data["wind_v"][quiver_mask],
        sampled_speed,
        cmap="viridis",
        norm=matplotlib.colors.Normalize(vmin=0.0, vmax=speed_limit),
        angles="xy",
        scale_units="xy",
        scale=13,
        width=0.004,
        alpha=0.9,
        zorder=5,
    )
    setup_map_axis(
        ax_cell_wind,
        "② 셀별 실제 연평균 바람만 표시\n"
        "화살표=TO(향하는 방향), FROM=TO+180°, 길이·색=풍속",
    )
    ax_cell_wind.quiverkey(
        cell_quiver,
        0.68,
        0.05,
        5.0,
        "5 m/s",
        coordinates="axes",
        labelpos="E",
        fontproperties={"size": 9},
    )
    cell_colorbar = fig.colorbar(
        cell_quiver,
        ax=ax_cell_wind,
        orientation="horizontal",
        fraction=0.045,
        pad=0.08,
    )
    cell_colorbar.set_label("셀별 12개월 평균 풍속 [m/s]")

    sector_angles = np.deg2rad([row["bearing_deg"] for row in metrics])
    anchor_r = cone_r_km * 0.43
    anchor_x = anchor_r * np.sin(sector_angles)
    anchor_y = anchor_r * np.cos(sector_angles)
    sector_speed = np.array([row["wind_speed_mps"] for row in metrics])
    sector_quiver = ax_sector_wind.quiver(
        anchor_x,
        anchor_y,
        [row["wind_u_mps"] for row in metrics],
        [row["wind_v_mps"] for row in metrics],
        sector_speed,
        cmap="plasma",
        norm=matplotlib.colors.Normalize(
            vmin=float(sector_speed.min()), vmax=float(sector_speed.max())
        ),
        angles="xy",
        scale_units="xy",
        scale=11,
        width=0.009,
        zorder=6,
    )
    setup_map_axis(
        ax_sector_wind,
        "③ 섹터별 평균 바람만 표시\n각 섹터 내부 셀의 U/V를 평균한 대표 벡터",
    )
    for row, x, y in zip(metrics, anchor_x, anchor_y):
        ax_sector_wind.text(
            x,
            y - 0.24,
            f"{row['wind_toward_deg']:.1f}°\n{row['wind_speed_mps']:.2f} m/s",
            ha="center",
            va="top",
            fontsize=7.5,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "#8e44ad",
                "alpha": 0.88,
            },
            zorder=10,
        )
    ax_sector_wind.quiverkey(
        sector_quiver,
        0.68,
        0.05,
        5.0,
        "5 m/s",
        coordinates="axes",
        labelpos="E",
        fontproperties={"size": 9},
    )
    sector_colorbar = fig.colorbar(
        sector_quiver,
        ax=ax_sector_wind,
        orientation="horizontal",
        fraction=0.045,
        pad=0.08,
    )
    sector_colorbar.set_label("섹터 평균 풍속 [m/s]")

    for idx, row in enumerate(metrics):
        bearing_start = idx * sector_step
        bearing_end = bearing_start + sector_step
        theta1 = (90.0 - bearing_end) % 360.0
        theta2 = (90.0 - bearing_start) % 360.0
        takeoff_score = row["takeoff_wind_score"]
        landing_score = row["landing_wind_score"]
        if takeoff_score >= landing_score:
            face = (0.10, 0.65, 0.25, 0.12 + 0.38 * takeoff_score)
            edge = "darkgreen"
        else:
            face = (1.0, 0.55, 0.0, 0.12 + 0.38 * landing_score)
            edge = "darkorange"
        ax_scores.add_patch(
            Wedge(
                (0, 0),
                cone_r_km,
                theta1,
                theta2,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.2,
                zorder=2,
            )
        )
        angle = np.deg2rad(row["bearing_deg"])
        label_r = cone_r_km * 0.63
        ax_scores.text(
            label_r * np.sin(angle),
            label_r * np.cos(angle),
            f"S{row['sector']}\n이륙 {takeoff_score:.2f}\n착륙 {landing_score:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": edge,
                "alpha": 0.9,
            },
            zorder=9,
        )
    setup_map_axis(
        ax_scores,
        "④ 이륙·착륙 바람점수만 표시\n녹색=이륙 역풍 우세, 주황=착륙 역풍 우세",
        show_sector_labels=False,
    )
    add_reference_arrows(ax_scores, cone_r_km)
    ax_scores.legend(
        handles=[
            Patch(facecolor="#37a957", label="이륙 바람점수 우세"),
            Patch(facecolor="#f39c32", label="착륙 바람점수 우세"),
        ],
        loc="upper right",
        fontsize=8,
    )

    fig.suptitle(
        "섹터 진단 요소별 분리 보기\n"
        "종합 지도와 동일한 데이터를 MOC·셀 바람·섹터 평균·점수로 분리",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(
        OUTPUT_DIR / "sector_map_diagnostics_panels.png",
        dpi=300,
        bbox_inches="tight",
    )
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def _setup_spatial_axis(ax, plot_data, title, show_labels=True):
    cone_r_km = plot_data["cone_r"] / 1000.0
    half = cone_r_km * 1.12
    for sector in range(N_SECTORS):
        boundary = np.deg2rad(sector * 360.0 / N_SECTORS)
        ax.plot(
            [0.0, cone_r_km * np.sin(boundary)],
            [0.0, cone_r_km * np.cos(boundary)],
            color="#666666",
            linewidth=0.7,
            alpha=0.65,
            zorder=6,
        )
        if show_labels:
            center = np.deg2rad(sector_bearing(sector))
            label_r = cone_r_km * 0.86
            ax.text(
                label_r * np.sin(center),
                label_r * np.cos(center),
                f"S{sector + 1}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "#555555",
                    "alpha": 0.88,
                },
                zorder=10,
            )
    ax.add_patch(
        plt.Circle(
            (0, 0),
            cone_r_km,
            fill=False,
            linestyle="--",
            linewidth=1.5,
            color="black",
            zorder=9,
        )
    )
    ax.plot(0, 0, "P", color="gold", markeredgecolor="black", markersize=10, zorder=12)
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.20)
    ax.set_xlabel("버티포트 기준 X [km]")
    ax.set_ylabel("버티포트 기준 Y [km]")
    ax.set_title(title, fontsize=12, fontweight="bold")


def plot_spatial_alignment_panels(metrics, plot_data):
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(22, 14.5),
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.96,
        bottom=0.065,
        top=0.84,
        wspace=0.28,
        hspace=0.40,
    )
    (
        ax_cruise,
        ax_cone,
        ax_wind,
        ax_ground,
        ax_air,
        ax_combined,
    ) = axes.ravel()

    cruise_binary = np.where(plot_data["cruise_grid"] >= 0.5, 1.0, 0.0)
    cruise_display = np.ma.masked_where(
        ~plot_data["cruise_grid_in_analysis"],
        cruise_binary,
    )
    cruise_mesh = ax_cruise.pcolormesh(
        plot_data["cruise_x_km"],
        plot_data["cruise_y_km"],
        cruise_display,
        cmap=ListedColormap(["#d9efd5", "#dc5a52"]),
        shading="auto",
        vmin=0,
        vmax=1,
        alpha=0.82,
    )
    _setup_spatial_axis(
        ax_cruise,
        plot_data,
        "① 순항 MOC\nfixedAGL400, 원본 100 m EPSG:5179 격자",
    )
    ax_cruise.legend(
        handles=[
            Patch(facecolor="#d9efd5", label="통과 셀"),
            Patch(facecolor="#dc5a52", label="차단 셀"),
        ],
        loc="upper right",
        fontsize=8,
    )

    cone_binary = np.where(plot_data["vmoc_grid"] >= 0.5, 1.0, 0.0)
    cone_display = np.ma.masked_where(
        ~plot_data["vmoc_grid_in_cone"],
        cone_binary,
    )
    ax_cone.pcolormesh(
        plot_data["vmoc_grid_x_km"],
        plot_data["vmoc_grid_y_km"],
        cone_display,
        cmap=ListedColormap(["#d9efd5", "#e58b86"]),
        shading="auto",
        vmin=0,
        vmax=1,
        alpha=0.85,
    )
    blocked = plot_data["vmoc_blocked_mask"]
    ax_cone.scatter(
        plot_data["vmoc_x_km"][blocked],
        plot_data["vmoc_y_km"][blocked],
        c="#7a0019",
        s=9,
        marker="s",
        linewidths=0,
        alpha=0.95,
        zorder=5,
    )
    _setup_spatial_axis(
        ax_cone,
        plot_data,
        "② 이착륙 원추 MOC\n연한 분홍=차단 셀, 진한 사각형=같은 차단 셀 중심",
    )
    ax_cone.legend(
        handles=[
            Patch(facecolor="#d9efd5", label="통과 셀"),
            Patch(facecolor="#e58b86", label="차단 셀"),
            Patch(facecolor="#7a0019", label="차단 셀 중심"),
        ],
        loc="upper right",
        fontsize=8,
    )

    wind_mask = plot_data["wind_in_cone"]
    quiver_mask = sample_mask_for_quiver(wind_mask)
    wind_speed = np.hypot(
        plot_data["wind_u"][quiver_mask],
        plot_data["wind_v"][quiver_mask],
    )
    speed_limit = max(1.0, float(np.nanpercentile(wind_speed, 95)))
    wind_quiver = ax_wind.quiver(
        plot_data["wind_x_km"][quiver_mask],
        plot_data["wind_y_km"][quiver_mask],
        plot_data["wind_u"][quiver_mask],
        plot_data["wind_v"][quiver_mask],
        wind_speed,
        cmap="viridis",
        norm=matplotlib.colors.Normalize(vmin=0.0, vmax=speed_limit),
        angles="xy",
        scale_units="xy",
        scale=13,
        width=0.004,
    )
    _setup_spatial_axis(
        ax_wind,
        plot_data,
        "③ 셀별 12개월 연평균 바람\n"
        "화살표=TO(향하는 방향), FROM=TO+180°, 색=풍속",
    )
    fig.colorbar(wind_quiver, ax=ax_wind, fraction=0.045, pad=0.03).set_label(
        "풍속 [m/s]"
    )

    risk_mask = ~plot_data["risk_in_cone"]
    ground_mesh = ax_ground.pcolormesh(
        plot_data["risk_x_km"],
        plot_data["risk_y_km"],
        np.ma.masked_where(risk_mask, plot_data["ground_grid_norm"]),
        shading="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
    )
    _setup_spatial_axis(
        ax_ground,
        plot_data,
        "④ 지상위험도\n1 km 내부 셀만 0-1 정규화",
    )
    fig.colorbar(ground_mesh, ax=ax_ground, fraction=0.045, pad=0.03).set_label(
        "정규화 위험도 [0-1]"
    )

    air_mesh = ax_air.pcolormesh(
        plot_data["risk_x_km"],
        plot_data["risk_y_km"],
        np.ma.masked_where(risk_mask, plot_data["air_grid_norm"]),
        shading="auto",
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    _setup_spatial_axis(
        ax_air,
        plot_data,
        "⑤ 공중위험도\n550 m MSL 이하 누적, 1 km 내부 0-1 정규화",
    )
    fig.colorbar(air_mesh, ax=ax_air, fraction=0.045, pad=0.03).set_label(
        "정규화 위험도 [0-1]"
    )

    combined_grid = 0.5 * (
        plot_data["ground_grid_norm"] + plot_data["air_grid_norm"]
    )
    combined_mesh = ax_combined.pcolormesh(
        plot_data["risk_x_km"],
        plot_data["risk_y_km"],
        np.ma.masked_where(risk_mask, combined_grid),
        shading="auto",
        cmap="inferno",
        vmin=0,
        vmax=1,
        alpha=0.88,
    )
    blocked = plot_data["vmoc_blocked_mask"]
    ax_combined.scatter(
        plot_data["vmoc_x_km"][blocked],
        plot_data["vmoc_y_km"][blocked],
        s=11,
        c="#00d4ff",
        marker="s",
        label="원추 MOC 차단 셀",
        zorder=7,
    )
    _setup_spatial_axis(
        ax_combined,
        plot_data,
        "⑥ 주요 레이어 종합 비교\n배경=0.5×지상+0.5×공중, 청색=원추 MOC 차단",
    )
    fig.colorbar(combined_mesh, ax=ax_combined, fraction=0.045, pad=0.03).set_label(
        "셀 단위 비교용 통합값 [0-1]"
    )
    ax_combined.legend(loc="upper right", fontsize=8)

    half = plot_data["cone_r"] / 1000.0 * 1.12
    fig.suptitle(
        "섹터 지도 공간 정합 진단\n"
        f"모든 패널: EPSG:5179, 동일 중심·동일 범위 ±{half:.3f} km, "
        "동일 종횡비",
        fontsize=17,
        fontweight="bold",
        y=0.965,
    )
    fig.savefig(
        OUTPUT_DIR / "sector_map_diagnostics_panels.png",
        dpi=300,
        bbox_inches="tight",
    )
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def _deg_to_tile(lon, lat, zoom):
    lat = float(np.clip(lat, -85.05112878, 85.05112878))
    n = 2**zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (
        1.0
        - math.asinh(math.tan(lat_rad)) / math.pi
    ) / 2.0 * n
    return x, y


def _tile_to_deg(x, y, zoom):
    n = 2**zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lon, lat


def _download_osm_background(lon_lim, lat_lim, zoom=13):
    x0f, y1f = _deg_to_tile(lon_lim[0], lat_lim[0], zoom)
    x1f, y0f = _deg_to_tile(lon_lim[1], lat_lim[1], zoom)
    x0, x1 = math.floor(x0f), math.floor(x1f)
    y0, y1 = math.floor(y0f), math.floor(y1f)
    width = (x1 - x0 + 1) * 256
    height = (y1 - y0 + 1) * 256
    mosaic = Image.new("RGB", (width, height), "white")
    try:
        for tile_y in range(y0, y1 + 1):
            for tile_x in range(x0, x1 + 1):
                url = f"https://tile.openstreetmap.org/{zoom}/{tile_x}/{tile_y}.png"
                request = Request(url, headers={"User-Agent": "sector-alignment/1.0"})
                with urlopen(request, timeout=2.5) as response:
                    tile = Image.open(BytesIO(response.read())).convert("RGB")
                mosaic.paste(tile, ((tile_x - x0) * 256, (tile_y - y0) * 256))
    except Exception as exc:
        return None, None, str(exc)

    west, north = _tile_to_deg(x0, y0, zoom)
    east, south = _tile_to_deg(x1 + 1, y1 + 1, zoom)
    return np.asarray(mosaic), [west, east, south, north], None


def _draw_lonlat_guides(ax, plot_data, transformer, lon_lim, lat_lim):
    angles = np.linspace(0.0, 2.0 * np.pi, 361)
    radius = plot_data["cone_r"]
    circle_x = plot_data["vx"] + radius * np.cos(angles)
    circle_y = plot_data["vy"] + radius * np.sin(angles)
    circle_lon, circle_lat = transformer.transform(circle_x, circle_y)
    ax.plot(circle_lon, circle_lat, "k--", linewidth=1.3, zorder=8)
    ax.plot(
        VERTIPORT_LON,
        VERTIPORT_LAT,
        "P",
        color="gold",
        markeredgecolor="black",
        markersize=9,
        zorder=10,
    )
    ax.set_xlim(lon_lim)
    ax.set_ylim(lat_lim)
    mean_lat = 0.5 * (lat_lim[0] + lat_lim[1])
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)), adjustable="box")
    ax.set_xlabel("경도")
    ax.set_ylabel("위도")
    ax.grid(True, alpha=0.18)


def plot_osm_overview(plot_data):
    half_m = plot_data["cone_r"] * 1.12
    to_wgs84 = pyproj.Transformer.from_crs(
        "EPSG:5179", "EPSG:4326", always_xy=True
    )
    corner_x = np.array(
        [plot_data["vx"] - half_m, plot_data["vx"] + half_m] * 2
    )
    corner_y = np.array(
        [plot_data["vy"] - half_m] * 2 + [plot_data["vy"] + half_m] * 2
    )
    corner_lon, corner_lat = to_wgs84.transform(corner_x, corner_y)
    lon_lim = [float(np.min(corner_lon)), float(np.max(corner_lon))]
    lat_lim = [float(np.min(corner_lat)), float(np.max(corner_lat))]
    background, background_extent, osm_error = _download_osm_background(
        lon_lim, lat_lim
    )

    fig, axes = plt.subplots(2, 2, figsize=(17, 16))
    fig.subplots_adjust(
        left=0.06,
        right=0.94,
        bottom=0.06,
        top=0.86,
        wspace=0.24,
        hspace=0.30,
    )
    for ax in axes.ravel():
        if background is not None:
            ax.imshow(background, extent=background_extent, origin="upper")
        else:
            ax.set_facecolor("#f2f2f2")
        _draw_lonlat_guides(ax, plot_data, to_wgs84, lon_lim, lat_lim)

    vmoc_grid_x = (
        plot_data["vmoc_grid_x_km"] * 1000.0 + plot_data["vx"]
    )
    vmoc_grid_y = (
        plot_data["vmoc_grid_y_km"] * 1000.0 + plot_data["vy"]
    )
    vmoc_grid_lon, vmoc_grid_lat = to_wgs84.transform(
        vmoc_grid_x,
        vmoc_grid_y,
    )
    cone_binary = np.where(plot_data["vmoc_grid"] >= 0.5, 1.0, 0.0)
    cone_display = np.ma.masked_where(
        ~plot_data["vmoc_grid_in_cone"],
        cone_binary,
    )
    axes[0, 0].pcolormesh(
        vmoc_grid_lon,
        vmoc_grid_lat,
        cone_display,
        cmap=ListedColormap(["#83c77d", "#e58b86"]),
        shading="auto",
        vmin=0,
        vmax=1,
        alpha=0.42,
    )
    blocked = plot_data["vmoc_blocked_mask"]
    vmoc_lon, vmoc_lat = to_wgs84.transform(
        plot_data["vmoc_x_km"][blocked] * 1000.0 + plot_data["vx"],
        plot_data["vmoc_y_km"][blocked] * 1000.0 + plot_data["vy"],
    )
    axes[0, 0].scatter(
        vmoc_lon,
        vmoc_lat,
        s=6,
        c="#6d0017",
        marker="s",
        alpha=0.78,
        zorder=7,
    )
    axes[0, 0].set_title(
        "원추 MOC 정합 지도\n"
        "연한 분홍=차단 셀, 진한 사각형=같은 차단 셀 중심",
        fontweight="bold",
    )

    wind_mask = plot_data["wind_in_cone"]
    quiver_mask = sample_mask_for_quiver(wind_mask)
    wind_lon, wind_lat = to_wgs84.transform(
        plot_data["wind_x_km"][quiver_mask] * 1000.0 + plot_data["vx"],
        plot_data["wind_y_km"][quiver_mask] * 1000.0 + plot_data["vy"],
    )
    wind_u = plot_data["wind_u"][quiver_mask]
    wind_v = plot_data["wind_v"][quiver_mask]
    wind_norm = np.hypot(wind_u, wind_v)
    axes[0, 1].quiver(
        wind_lon,
        wind_lat,
        np.divide(wind_u, wind_norm, out=np.zeros_like(wind_u), where=wind_norm > 0),
        np.divide(wind_v, wind_norm, out=np.zeros_like(wind_v), where=wind_norm > 0),
        wind_norm,
        cmap="viridis",
        angles="uv",
        scale=24,
        width=0.003,
        zorder=7,
    )
    axes[0, 1].set_title(
        "연평균 바람 정합 지도\n방향=U/V 벡터, 색=풍속",
        fontweight="bold",
    )

    risk_lon, risk_lat = to_wgs84.transform(
        plot_data["risk_x"], plot_data["risk_y"]
    )
    risk_mask = ~plot_data["risk_in_cone"]
    ground = axes[1, 0].pcolormesh(
        risk_lon,
        risk_lat,
        np.ma.masked_where(risk_mask, plot_data["ground_grid_norm"]),
        shading="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        alpha=0.58,
    )
    fig.colorbar(ground, ax=axes[1, 0], fraction=0.045, pad=0.03)
    axes[1, 0].set_title(
        "지상위험도 정합 지도\n1 km 내부 0-1 정규화",
        fontweight="bold",
    )

    air = axes[1, 1].pcolormesh(
        risk_lon,
        risk_lat,
        np.ma.masked_where(risk_mask, plot_data["air_grid_norm"]),
        shading="auto",
        cmap="magma",
        vmin=0,
        vmax=1,
        alpha=0.58,
    )
    fig.colorbar(air, ax=axes[1, 1], fraction=0.045, pad=0.03)
    axes[1, 1].set_title(
        "공중위험도 정합 지도\n550 m 이하 누적, 1 km 내부 0-1 정규화",
        fontweight="bold",
    )

    background_status = (
        "OpenStreetMap 배경 사용"
        if background is not None
        else "네트워크 접근 불가: 동일 위경도 좌표 배경으로 대체"
    )
    fig.suptitle(
        "동일 지도 범위의 MOC·바람·지상·공중 위험도 비교\n"
        f"{background_status}",
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )
    fig.savefig(
        OUTPUT_DIR / "sector_map_osm_overview.png",
        dpi=300,
        bbox_inches="tight",
    )
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_dashboard(metrics, combinations):
    sectors = np.arange(1, N_SECTORS + 1)
    moc_safety = np.array([row["moc_safety_score"] for row in metrics])
    takeoff_wind = np.array([row["takeoff_wind_score"] for row in metrics])
    landing_wind = np.array([row["landing_wind_score"] for row in metrics])
    ground_risk = np.array([row["ground_risk_score"] for row in metrics])
    air_risk = np.array([row["air_risk_score"] for row in metrics])
    combined_risk = np.array([row["combined_risk_score"] for row in metrics])

    fig = plt.figure(figsize=(19, 22))
    grid = fig.add_gridspec(
        4, 2, height_ratios=[1.0, 1.0, 1.15, 1.05]
    )
    ax_moc = fig.add_subplot(grid[0, 0])
    ax_wind_bar = fig.add_subplot(grid[0, 1])
    ax_wind_line = fig.add_subplot(grid[1, 0])
    ax_risk = fig.add_subplot(grid[1, 1])
    ax_combination = fig.add_subplot(grid[2, 0])
    ax_text = fig.add_subplot(grid[2, 1])
    ax_table = fig.add_subplot(grid[3, :])

    moc_colors = ["#2ca25f" if value == 1.0 else "#de2d26" for value in moc_safety]
    ax_moc.bar(sectors, moc_safety, color=moc_colors)
    ax_moc.axhline(1.0, color="black", linestyle="--", linewidth=1.2)
    for x, row in zip(sectors, metrics):
        ax_moc.text(
            x,
            min(row["moc_safety_score"] + 0.025, 1.04),
            f"{100*row['moc_blocked_ratio']:.1f}%",
            ha="center",
            fontsize=8,
        )
    ax_moc.set_ylim(0, 1.10)
    ax_moc.set_xticks(sectors)
    ax_moc.set_title(
        "MOC 안전점수 (막대 위 숫자 = 차단비율)\n"
        "안전점수 = 1 - 차단 셀 수 / 전체 셀 수"
    )
    ax_moc.set_ylabel("안전점수 (1 = 차단 셀 0개)")
    ax_moc.grid(axis="y", alpha=0.25)

    wind_width = 0.36
    takeoff_x = sectors - wind_width / 2
    landing_x = sectors + wind_width / 2
    ax_wind_bar.bar(
        takeoff_x,
        takeoff_wind,
        wind_width,
        color="#72c68c",
        alpha=0.85,
        label="이륙 바람점수",
    )
    ax_wind_bar.bar(
        landing_x,
        landing_wind,
        wind_width,
        color="#f7b267",
        alpha=0.85,
        label="착륙 바람점수",
    )
    for takeoff_pos, landing_pos, takeoff_score, landing_score in zip(
        takeoff_x, landing_x, takeoff_wind, landing_wind
    ):
        ax_wind_bar.annotate(
            f"{takeoff_score:.2f}",
            (takeoff_pos, takeoff_score),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="#176b35",
        )
        ax_wind_bar.annotate(
            f"{landing_score:.2f}",
            (landing_pos, landing_score),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="#a65100",
        )
    ax_wind_bar.axhline(
        WIND_THRESHOLD,
        color="crimson",
        linestyle="--",
        linewidth=1.6,
        label=f"필수조건 {WIND_THRESHOLD:.1f}",
    )
    ax_wind_bar.set_ylim(-0.04, 1.12)
    ax_wind_bar.set_xticks(sectors)
    ax_wind_bar.set_title(
        "연평균 바람 적합도 - 막대그래프\n"
        "점수 = 해당 섹터 역풍성분 / 12개 섹터 중 최대 역풍성분"
    )
    ax_wind_bar.set_ylabel("0-1 정규화 점수")
    ax_wind_bar.legend(fontsize=9)
    ax_wind_bar.grid(axis="y", alpha=0.25)

    ax_wind_line.plot(
        sectors,
        takeoff_wind,
        marker="o",
        markersize=7,
        linewidth=2.2,
        color="#238b45",
        label="이륙 바람점수",
    )
    ax_wind_line.plot(
        sectors,
        landing_wind,
        marker="s",
        markersize=7,
        linewidth=2.2,
        color="#f28e2b",
        label="착륙 바람점수",
    )
    for sector, takeoff_score, landing_score in zip(
        sectors, takeoff_wind, landing_wind
    ):
        ax_wind_line.annotate(
            f"{takeoff_score:.2f}",
            (sector, takeoff_score),
            xytext=(-9, 8 if takeoff_score > 0 else 6),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="#176b35",
        )
        ax_wind_line.annotate(
            f"{landing_score:.2f}",
            (sector, landing_score),
            xytext=(9, -13 if landing_score > 0 else -14),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="#a65100",
        )
    ax_wind_line.axhline(
        WIND_THRESHOLD,
        color="crimson",
        linestyle="--",
        linewidth=1.6,
        label=f"필수조건 {WIND_THRESHOLD:.1f}",
    )
    ax_wind_line.set_ylim(-0.08, 1.12)
    ax_wind_line.set_xticks(sectors)
    ax_wind_line.set_title(
        "연평균 바람 적합도 - 점·선 그래프\n"
        "0점 = 사용 금지가 아니라 연평균 역풍 도움 없음"
    )
    ax_wind_line.set_ylabel("0-1 정규화 점수")
    ax_wind_line.legend(fontsize=9)
    ax_wind_line.grid(axis="y", alpha=0.25)

    risk_width = 0.25
    ax_risk.bar(
        sectors - risk_width,
        ground_risk,
        risk_width,
        color="#4e79a7",
        label="지상위험",
    )
    ax_risk.bar(
        sectors,
        air_risk,
        risk_width,
        color="#e15759",
        label="공중위험",
    )
    ax_risk.bar(
        sectors + risk_width,
        combined_risk,
        risk_width,
        color="#8064a2",
        label="통합위험 (1:1)",
    )
    ax_risk.set_ylim(0, 1.08)
    ax_risk.set_xticks(sectors)
    ax_risk.set_title(
        "섹터별 위험도 (낮을수록 유리)\n"
        "통합위험도 = 0.5 x 지상위험도 + 0.5 x 공중위험도"
    )
    ax_risk.set_ylabel("0-1 정규화 위험도")
    ax_risk.legend(fontsize=9)
    ax_risk.grid(axis="y", alpha=0.25)

    condition_gap_matrix = np.full((N_SECTORS, N_SECTORS), np.nan)
    requirement_matrix = np.zeros((N_SECTORS, N_SECTORS), dtype=bool)
    for row in combinations:
        t = row["takeoff_sector"] - 1
        l = row["landing_sector"] - 1
        condition_gap_matrix[l, t] = row["condition_gap_score"]
        requirement_matrix[l, t] = row["required_conditions_met"]
    image = ax_combination.imshow(
        condition_gap_matrix,
        cmap="YlOrRd",
        vmin=0,
        vmax=max(0.5, float(np.nanmax(condition_gap_matrix))),
        origin="lower",
    )
    for landing in range(N_SECTORS):
        for takeoff in range(N_SECTORS):
            value = condition_gap_matrix[landing, takeoff]
            if np.isfinite(value):
                ax_combination.text(
                    takeoff,
                    landing,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="black" if value < 0.45 else "white",
                )
            if requirement_matrix[landing, takeoff]:
                ax_combination.plot(takeoff, landing, "c*", markersize=10)
    ax_combination.add_patch(
        Rectangle(
            (REFERENCE_TAKEOFF_SECTOR - 1.5, REFERENCE_LANDING_SECTOR - 1.5),
            1,
            1,
            fill=False,
            edgecolor="lime",
            linewidth=3,
        )
    )
    ax_combination.set_xticks(
        np.arange(N_SECTORS), labels=[f"S{i}" for i in sectors]
    )
    ax_combination.set_yticks(
        np.arange(N_SECTORS), labels=[f"S{i}" for i in sectors]
    )
    ax_combination.set_xlabel("이륙 섹터")
    ax_combination.set_ylabel("착륙 섹터")
    ax_combination.set_title(
        "이착륙 조합별 조건 미달 점수 (낮을수록 유리)\n"
        "조건 미달 = 0.5 x MOC 문제 + 0.5 x 바람 미달\n"
        "MOC 문제=(이륙비율+착륙비율)/2, "
        "바람 미달=두 방향의 0.9 미달률 평균",
        fontsize=10,
    )
    fig.colorbar(image, ax=ax_combination, fraction=0.046, pad=0.04)

    ax_table.axis("off")
    top_rows = combinations[:TOP_N]
    table_data = [
        [
            row["comparison_rank"],
            f"S{row['takeoff_sector']}",
            f"S{row['landing_sector']}",
            f"{row['moc_issue_score']:.3f}",
            f"{row['wind_gap_score']:.3f}",
            f"{row['condition_gap_score']:.3f}",
            f"{row['combined_risk_score']:.3f}",
            f"{row['takeoff_wind_score']:.2f}/"
            f"{row['landing_wind_score']:.2f}",
            f"{100*row['takeoff_moc_ratio']:.1f}/"
            f"{100*row['landing_moc_ratio']:.1f}",
        ]
        for row in top_rows
    ]
    table = ax_table.table(
        cellText=table_data,
        colLabels=[
            "순위",
            "이륙",
            "착륙",
            "MOC 문제",
            "바람 미달",
            "조건 미달\n(최종)",
            "위험도",
            "바람 이륙/착륙",
            "MOC% 이륙/착륙",
        ],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.6)
    table.scale(1.0, 1.75)
    ax_table.set_title(
        f"조건 미달이 적은 상위 {TOP_N}개 조합\n"
        "MOC 문제=(이륙 MOC%+착륙 MOC%)/2\n"
        "바람 미달=[max(0.9-이륙,0)/0.9 + "
        "max(0.9-착륙,0)/0.9]/2",
        fontsize=9.5,
        fontweight="bold",
        pad=10,
    )

    eligible_combinations = sorted(
        (row for row in combinations if row["required_conditions_met"]),
        key=lambda row: row["combined_risk_score"],
    )
    reference = next(
        row for row in combinations if row["is_reference_s7_s5"]
    )
    s7 = metrics[REFERENCE_TAKEOFF_SECTOR - 1]
    s5 = metrics[REFERENCE_LANDING_SECTOR - 1]
    if eligible_combinations:
        best = eligible_combinations[0]
        requirement_text = (
            f"필수조건 통과 최적: 이륙 S{best['takeoff_sector']}, "
            f"착륙 S{best['landing_sector']}\n"
            f"통합위험도 {best['combined_risk_score']:.3f}, "
            f"통과 조합 {len(eligible_combinations)}개"
        )
    else:
        requirement_text = "필수조건을 모두 만족하는 조합 없음"
    explanation = (
        "선정 절차\n"
        "1. MOC 차단 셀이 0개인지 확인\n"
        f"2. 이륙/착륙 바람점수가 각각 {WIND_THRESHOLD:.1f} 이상인지 확인\n"
        "3. 통과 조합 중 지상·공중 위험도 1:1 합이 가장 낮은 조합 선택\n\n"
        "점수 계산식\n"
        "MOC 문제 = (이륙 MOC 비율 + 착륙 MOC 비율) / 2\n"
        "바람 미달 = 이륙·착륙의 0.9 기준 미달률 평균\n"
        "조건 미달 = 0.5 x MOC 문제 + 0.5 x 바람 미달\n"
        "통합위험 = 0.5 x 지상위험 + 0.5 x 공중위험\n\n"
        "바람점수는 사용 가능/불가능 판정값이 아닙니다.\n"
        "0점은 그 방향에서 연평균 역풍의 도움을 받지 못한다는 뜻입니다.\n\n"
        f"{requirement_text}\n\n"
        "현재 설정: 이륙 S7 / 착륙 S5\n"
        f"S7: MOC {100*s7['moc_blocked_ratio']:.1f}% "
        f"({s7['moc_blocked_cells']}/{s7['moc_total_cells']}), "
        f"이륙 바람 {s7['takeoff_wind_score']:.3f}, "
        f"위험 {s7['combined_risk_score']:.3f}\n"
        f"S5: MOC {100*s5['moc_blocked_ratio']:.1f}% "
        f"({s5['moc_blocked_cells']}/{s5['moc_total_cells']}), "
        f"착륙 바람 {s5['landing_wind_score']:.3f}, "
        f"위험 {s5['combined_risk_score']:.3f}\n"
        f"S7/S5 비교 순위: {reference['comparison_rank']}/"
        f"{len(combinations)}, "
        f"조건 미달 점수 {reference['condition_gap_score']:.3f}\n\n"
        "해석: S7은 낮은 위험도와 높은 이륙 바람점수가 장점이지만\n"
        "MOC 0개 및 바람 0.9 필수조건에는 조금 미달합니다.\n"
        "S5는 MOC가 깨끗하고 위험도가 낮지만 착륙 바람점수가 낮습니다."
    )
    ax_text.axis("off")
    ax_text.text(
        0.01,
        0.98,
        explanation,
        va="top",
        fontsize=11,
        linespacing=1.45,
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "#f7f7f7",
            "edgecolor": "#555555",
        },
    )

    fig.suptitle(
        "12개 이착륙 섹터 종합 재평가 대시보드\n"
        f"{TARGET_ALT_MSL:.0f} m MSL, 12개월 연평균 바람, "
        "MOC + 지상/공중 위험도",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(
        OUTPUT_DIR / "sector_evaluation_dashboard.png",
        dpi=300,
        bbox_inches="tight",
    )
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_wind_scoring_presentation(metrics, plot_data):
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    ax_month, ax_map, ax_dot, ax_score = axes.ravel()

    monthly = plot_data["monthly_cone_wind"]
    colors = plt.cm.hsv(np.linspace(0, 1, len(monthly), endpoint=False))
    max_month_speed = max(
        np.hypot(row["u_mps"], row["v_mps"]) for row in monthly
    )
    vector_limit = max(1.0, max_month_speed * 1.25)
    label_offsets = [
        (22, 0),
        (18, 12),
        (12, 22),
        (0, 26),
        (-12, 22),
        (-20, 12),
        (-24, 0),
        (-20, -12),
        (-12, -22),
        (0, -26),
        (12, -22),
        (20, -12),
    ]
    for row, color, offset in zip(monthly, colors, label_offsets):
        ax_month.arrow(
            0,
            0,
            row["u_mps"],
            row["v_mps"],
            color=color,
            width=0.025,
            head_width=0.18,
            length_includes_head=True,
            alpha=0.78,
        )
        ax_month.annotate(
            f"{row['month']}월",
            (row["u_mps"], row["v_mps"]),
            xytext=offset,
            textcoords="offset points",
            color=color,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7},
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": color,
                "alpha": 0.85,
            },
        )
    mean_u = float(np.mean([row["u_mps"] for row in monthly]))
    mean_v = float(np.mean([row["v_mps"] for row in monthly]))
    ax_month.arrow(
        0,
        0,
        mean_u,
        mean_v,
        color="black",
        width=0.045,
        head_width=0.25,
        length_includes_head=True,
        zorder=10,
    )
    ax_month.text(
        mean_u * 1.06,
        mean_v * 1.06,
        "12개월 평균",
        color="black",
        fontsize=10,
        fontweight="bold",
    )
    ax_month.axhline(0, color="gray", linewidth=0.8)
    ax_month.axvline(0, color="gray", linewidth=0.8)
    ax_month.set_xlim(-vector_limit, vector_limit)
    ax_month.set_ylim(-vector_limit, vector_limit)
    ax_month.set_aspect("equal")
    ax_month.grid(True, alpha=0.25)
    ax_month.set_xlabel("U: 동쪽(+) / 서쪽(-) 바람성분 [m/s]")
    ax_month.set_ylabel("V: 북쪽(+) / 남쪽(-) 바람성분 [m/s]")
    ax_month.set_title(
        "1. 연평균 바람벡터 계산\n"
        "각 셀·고도에서 U와 V를 12개월 유효자료로 각각 평균",
        fontweight="bold",
    )
    ax_month.text(
        0.02,
        0.02,
        "실제 점수 계산은 원추 전체를 한 번에 평균하지 않습니다.\n"
        "① 각 격자 셀별 12개월 U,V 평균\n"
        "② 500 m와 600 m 자료를 선형보간하여 550 m 계산\n"
        "③ 그 뒤 각 섹터 내부 셀들의 U,V를 평균\n"
        "※ 이 패널의 월별 화살표는 발표를 위한 원추 영역 요약",
        transform=ax_month.transAxes,
        fontsize=9,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "gray"},
    )

    wind_mask = plot_data["wind_in_cone"]
    quiver_mask = sample_mask_for_quiver(wind_mask)
    ax_map.quiver(
        plot_data["wind_x_km"][quiver_mask],
        plot_data["wind_y_km"][quiver_mask],
        plot_data["wind_u"][quiver_mask],
        plot_data["wind_v"][quiver_mask],
        color="#1769aa",
        angles="xy",
        scale_units="xy",
        scale=13,
        width=0.004,
        alpha=0.8,
    )
    cone_r_km = plot_data["cone_r"] / 1000.0
    theta = np.linspace(0, 2 * np.pi, 300)
    ax_map.plot(
        cone_r_km * np.cos(theta),
        cone_r_km * np.sin(theta),
        "k--",
        linewidth=1.5,
    )
    for sector in range(N_SECTORS):
        boundary = np.deg2rad(sector * 360.0 / N_SECTORS)
        ax_map.plot(
            [0, cone_r_km * np.sin(boundary)],
            [0, cone_r_km * np.cos(boundary)],
            color="gray",
            linewidth=0.8,
        )
        bearing = np.deg2rad(sector_bearing(sector))
        label_r = cone_r_km * 0.80
        color = "black"
        if sector + 1 == REFERENCE_TAKEOFF_SECTOR:
            color = "darkgreen"
        elif sector + 1 == REFERENCE_LANDING_SECTOR:
            color = "darkorange"
        ax_map.text(
            label_r * np.sin(bearing),
            label_r * np.cos(bearing),
            f"S{sector + 1}",
            ha="center",
            va="center",
            color=color,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": color,
                "alpha": 0.9,
            },
        )
    ax_map.plot(0, 0, "P", color="gold", markeredgecolor="black", markersize=12)
    ax_map.set_xlim(-cone_r_km * 1.1, cone_r_km * 1.1)
    ax_map.set_ylim(-cone_r_km * 1.1, cone_r_km * 1.1)
    ax_map.set_aspect("equal")
    ax_map.grid(True, alpha=0.25)
    ax_map.set_xlabel("버티포트 기준 X [km]")
    ax_map.set_ylabel("버티포트 기준 Y [km]")
    ax_map.set_title(
        "2. 550 m 연평균 바람장을 12개 섹터로 분류\n"
        "각 셀은 위치 방위각에 따라 S1-S12 중 하나에 포함",
        fontweight="bold",
    )
    ax_map.text(
        0.02,
        0.02,
        "각 섹터에서 파란 화살표들의 U,V를 평균하여\n"
        "섹터 대표 바람벡터 (U_avg, V_avg)를 계산",
        transform=ax_map.transAxes,
        fontsize=9,
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "gray"},
    )

    ax_dot.set_xlim(-5.5, 5.5)
    ax_dot.set_ylim(-3.1, 3.1)
    ax_dot.set_aspect("equal")
    ax_dot.axhline(0, color="#dddddd", linewidth=0.8)
    ax_dot.axis("off")
    wind_scale = 0.38
    heading_length = 2.1
    examples = [
        (-3.0, metrics[REFERENCE_TAKEOFF_SECTOR - 1], "S7 이륙 예시", "darkgreen"),
        (3.0, metrics[REFERENCE_LANDING_SECTOR - 1], "S5 착륙 예시", "darkorange"),
    ]
    for origin_x, row, title, color in examples:
        bearing = np.deg2rad(row["bearing_deg"])
        direction = np.array([np.sin(bearing), np.cos(bearing)])
        wind = np.array([row["wind_u_mps"], row["wind_v_mps"]])
        projection = row["radial_wind_mps"] * direction
        origin = np.array([origin_x, 0.0])
        heading_end = origin + heading_length * direction
        wind_end = origin + wind_scale * wind
        projection_end = origin + wind_scale * projection
        ax_dot.annotate(
            "",
            xy=heading_end,
            xytext=origin,
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.5},
        )
        ax_dot.annotate(
            "",
            xy=wind_end,
            xytext=origin,
            arrowprops={"arrowstyle": "-|>", "color": "#1769aa", "lw": 2.5},
        )
        ax_dot.annotate(
            "",
            xy=projection_end,
            xytext=origin,
            arrowprops={"arrowstyle": "-|>", "color": "crimson", "lw": 3},
        )
        ax_dot.plot(origin_x, 0, "ko", markersize=5)
        ax_dot.text(
            origin_x,
            2.75,
            title,
            ha="center",
            fontsize=11,
            fontweight="bold",
            color=color,
        )
        ax_dot.text(
            origin_x,
            -2.65,
            f"섹터방향 e = ({direction[0]:+.2f}, {direction[1]:+.2f})\n"
            f"평균바람 W = ({wind[0]:+.2f}, {wind[1]:+.2f}) m/s\n"
            f"내적 p = W·e = {row['radial_wind_mps']:+.3f} m/s",
            ha="center",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.92},
        )
    ax_dot.text(
        0,
        1.6,
        "파랑: 평균 바람벡터 W\n"
        "섹터색: 바깥쪽 섹터방향 e\n"
        "빨강: 바람의 섹터방향 성분 p = W·e",
        ha="center",
        fontsize=9,
        bbox={"facecolor": "#f5f5f5", "edgecolor": "gray"},
    )
    ax_dot.set_title(
        "3. 역풍 판정은 외적이 아니라 내적(투영) 사용\n"
        "p<0: 바깥쪽 이륙방향의 역풍, p>0: 안쪽 착륙방향의 역풍",
        fontweight="bold",
    )

    sectors = np.arange(1, N_SECTORS + 1)
    takeoff_scores = np.array([row["takeoff_wind_score"] for row in metrics])
    landing_scores = np.array([row["landing_wind_score"] for row in metrics])
    width = 0.38
    ax_score.bar(
        sectors - width / 2,
        takeoff_scores,
        width,
        color="#4caf6a",
        label="이륙점수",
    )
    ax_score.bar(
        sectors + width / 2,
        landing_scores,
        width,
        color="#f4a340",
        label="착륙점수",
    )
    ax_score.axhline(
        WIND_THRESHOLD,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label=f"현재 비교 기준 {WIND_THRESHOLD:.1f}",
    )
    ax_score.set_xticks(sectors)
    ax_score.set_ylim(0, 1.12)
    ax_score.set_ylabel("0-1 바람점수")
    ax_score.grid(axis="y", alpha=0.25)
    ax_score.legend(loc="upper right")
    ax_score.set_title(
        "4. 역풍성분을 0-1 점수로 변환\n"
        "이륙=max(-p,0)/이륙 최대값, 착륙=max(p,0)/착륙 최대값",
        fontweight="bold",
    )
    s7 = metrics[REFERENCE_TAKEOFF_SECTOR - 1]
    s5 = metrics[REFERENCE_LANDING_SECTOR - 1]
    ax_score.text(
        0.02,
        0.96,
        f"S7 이륙: p={s7['radial_wind_mps']:+.3f} -> "
        f"역풍={s7['takeoff_headwind_mps']:.3f} m/s -> "
        f"점수={s7['takeoff_wind_score']:.3f}\n"
        f"S5 착륙: p={s5['radial_wind_mps']:+.3f} -> "
        f"역풍={s5['landing_headwind_mps']:.3f} m/s -> "
        f"점수={s5['landing_wind_score']:.3f}\n\n"
        "0점은 운항 금지가 아니라 연평균 역풍 이점이 없다는 의미\n"
        "횡풍성분과 순간 최대풍속은 현재 점수에 포함하지 않음",
        transform=ax_score.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "gray"},
    )

    fig.suptitle(
        "연평균 바람으로 이착륙 섹터 점수를 계산하는 방법\n"
        "월별 U/V 벡터 평균 -> 섹터별 평균 -> 내적(방향 투영) -> 0-1 정규화",
        fontsize=18,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.015,
        "중요: theta3d 풍향값은 현재 유효하지 않은 셀을 판별하는 데만 사용하고, "
        "실제 점수는 U3d와 V3d 벡터성분으로 계산합니다. 외적은 사용하지 않습니다.",
        ha="center",
        fontsize=10,
        bbox={"facecolor": "#fff4cc", "edgecolor": "#c28b00", "pad": 6},
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    fig.savefig(
        OUTPUT_DIR / "wind_scoring_method_presentation.png",
        dpi=300,
        bbox_inches="tight",
    )
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def print_console_summary(metrics, combinations, plot_data):
    eligible_combinations = [
        row for row in combinations if row["required_conditions_met"]
    ]
    reference = next(
        row for row in combinations if row["is_reference_s7_s5"]
    )
    print(
        f"analysis r = {plot_data['cone_r']:.0f} m "
        f"(physical cone r = {plot_data['physical_cone_r']:.0f} m), "
        f"evaluation altitude = {TARGET_ALT_MSL:.0f} m MSL"
    )
    print(
        f"sector rows = {len(metrics)}, takeoff/landing combinations = "
        f"{len(combinations)}, required-condition matches = "
        f"{len(eligible_combinations)}"
    )
    print("\nSector metrics:")
    print(
        " S  MOC blocked   MOC%  takeoff/landing wind   "
        "ground air combined"
    )
    for row in metrics:
        print(
            f"{row['sector']:2d}  "
            f"{row['moc_blocked_cells']:3d}/{row['moc_total_cells']:<3d}  "
            f"{100*row['moc_blocked_ratio']:5.1f}  "
            f"{row['takeoff_wind_score']:.3f}/"
            f"{row['landing_wind_score']:.3f}  "
            f"{row['ground_risk_score']:.3f} "
            f"{row['air_risk_score']:.3f} "
            f"{row['combined_risk_score']:.3f}"
        )
    print(f"\nTop {TOP_N} combinations with the smallest condition gap:")
    for row in combinations[:TOP_N]:
        print(
            f"{row['comparison_rank']:2d}. takeoff=S{row['takeoff_sector']} "
            f"landing=S{row['landing_sector']} "
            f"condition_gap={row['condition_gap_score']:.3f} "
            f"risk={row['combined_risk_score']:.3f}"
        )
    print(
        f"\nS7/S5: rank={reference['comparison_rank']}, "
        f"condition_gap={reference['condition_gap_score']:.3f}, "
        f"risk={reference['combined_risk_score']:.3f}"
    )
    print(f"\nSaved outputs to: {OUTPUT_DIR}")


def main():
    metrics, plot_data = evaluate_sectors()
    combinations = build_combination_ranking(metrics)
    validate_results(metrics, combinations)
    write_outputs(metrics, combinations, plot_data)
    plot_sector_map(metrics, plot_data)
    plot_sector_annual_mean_wind(metrics, plot_data)
    plot_spatial_alignment_panels(metrics, plot_data)
    plot_osm_overview(plot_data)
    plot_dashboard(metrics, combinations)
    plot_wind_scoring_presentation(metrics, plot_data)
    print_console_summary(metrics, combinations, plot_data)


if __name__ == "__main__":
    main()
