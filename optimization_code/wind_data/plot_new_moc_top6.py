"""전이 회랑을 따라 MOC·바람·지상·공중 위험을 평가해 이착륙 섹터를 선정한다."""
from __future__ import annotations

import csv
from io import BytesIO
import json
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
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch, Polygon
from PIL import Image
import pyproj
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat


# ============================== 평가 파라미터 ==============================
VERTIPORT_LAT = 35.6033860             # 평가 기준 버티포트 위도(deg)
VERTIPORT_LON = 129.0780250            # 평가 기준 버티포트 경도(deg)
VERTIPORT_ALT_MSL_M = 150.0            # 버티포트 표고(MSL m), MOC의 AGL 기준고도
TARGET_ALT_MSL_M = 600.0               # 전이 종료 목표고도(MSL m)
CLIMB_DESCENT_ANGLE_DEG = 6.0          # 전이 중심선의 상승·하강각(deg)
TRANSITION_CORRIDOR_HALF_WIDTH_M = 100.0  # 전이 회랑 좌우 반폭(m), 총폭은 200m
TRANSITION_DOWNWARD_CLEARANCE_M = 100.0   # 중심선 아래로 요구하는 최대 MOC 이격(m)
ALONG_TRACK_SAMPLE_STEP_M = 80.0       # 회랑 종방향 표본의 목표 간격(m)
N_SECTORS = 12                         # 버티포트 주변을 나누는 동일 각도 섹터 수
TOP_N = 6                              # 표·요약에 표시할 상위 이착륙 조합 수
WIND_TAIL_WEIGHT = 0.5                 # 바람위험 중 순풍 성분 비중
WIND_CROSS_WEIGHT = 0.5                # 바람위험 중 측풍 성분 비중
WIND_RISK_WEIGHT = 1.0 / 3.0           # 최종 점수의 바람위험 비중
GROUND_RISK_WEIGHT = 1.0 / 3.0         # 최종 점수의 지상위험 비중
AIR_RISK_WEIGHT = 1.0 / 3.0            # 최종 점수의 공중위험 비중
MOC_REFERENCE_MSL_M = VERTIPORT_ALT_MSL_M  # fixed-AGL MOC 지도의 기준 MSL(m)
MOC_AGL_LEVELS_M = np.arange(100.0, 1000.0, 100.0)  # 사용 가능한 MOC AGL 층(m)
REFERENCE_TAKEOFF_SECTOR = 7           # 기존 설정과 비교할 참조 이륙 섹터
REFERENCE_LANDING_SECTOR = 5           # 기존 설정과 비교할 참조 착륙 섹터
SHOW_PLOTS = False                     # True면 저장 후 화면에도 그림을 표시
OSM_ZOOM = 12                          # OSM 배경 타일 확대 수준; 실패 시 회색 배경 사용

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WIND_DIR = PROJECT_ROOT / "wind_data"  # AirRisk_Data_1~12.mat 입력 폴더
OUTPUT_DIR = WIND_DIR / "python_outputs"  # 이 스크립트의 결과 덮어쓰기 폴더
MOC_DIR = PROJECT_ROOT / "260608_MOC"  # fixed AGL100~900 MOC 입력 폴더
AIR_RISK_PATH = (
    PROJECT_ROOT / "air_risk_data" / "bird_riskmap_springfall_3d.npy"
)  # 봄·가을 조류 공중위험 3D 입력
GROUND_RISK_PATH = (
    PROJECT_ROOT
    / "ground_risk_data"
    / "Modified_high_res_affected_population_GRC.npy"
)  # 방향별 지상위험 입력


def _validate_parameters():
    if not TARGET_ALT_MSL_M > VERTIPORT_ALT_MSL_M:
        raise ValueError("TARGET_ALT_MSL_M must be greater than vertiport altitude")
    if not 0.0 < CLIMB_DESCENT_ANGLE_DEG < 90.0:
        raise ValueError("CLIMB_DESCENT_ANGLE_DEG must be between 0 and 90")
    if TRANSITION_CORRIDOR_HALF_WIDTH_M <= 0.0:
        raise ValueError("TRANSITION_CORRIDOR_HALF_WIDTH_M must be positive")
    if TRANSITION_DOWNWARD_CLEARANCE_M < 0.0:
        raise ValueError("TRANSITION_DOWNWARD_CLEARANCE_M cannot be negative")
    if ALONG_TRACK_SAMPLE_STEP_M <= 0.0:
        raise ValueError("ALONG_TRACK_SAMPLE_STEP_M must be positive")
    if N_SECTORS < 2:
        raise ValueError("N_SECTORS must be at least 2")
    if TOP_N <= 0:
        raise ValueError("TOP_N must be positive")
    if not np.isclose(WIND_TAIL_WEIGHT + WIND_CROSS_WEIGHT, 1.0):
        raise ValueError("Wind component weights must sum to 1")
    if not np.isclose(
        WIND_RISK_WEIGHT + GROUND_RISK_WEIGHT + AIR_RISK_WEIGHT, 1.0
    ):
        raise ValueError("Selection risk weights must sum to 1")


def _to_5179(lat, lon):
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:5179", always_xy=True
    )
    x, y = transformer.transform(lon, lat)
    return float(x), float(y)


def _sector_bearing(sector_zero_based):
    return ((float(sector_zero_based) + 0.5) * 360.0 / N_SECTORS) % 360.0


def _heading_unit(bearing_deg):
    angle = np.deg2rad(float(bearing_deg))
    return np.array([np.sin(angle), np.cos(angle)], dtype=float)


def _normalize01(values):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    out = np.ones_like(arr)
    if not np.any(finite):
        return out, None, None
    lo = float(np.min(arr[finite]))
    hi = float(np.max(arr[finite]))
    if hi - lo <= 1e-12:
        out[finite] = 0.0
    else:
        out[finite] = (arr[finite] - lo) / (hi - lo)
    return out, lo, hi


def _grid_axes(x_2d, y_2d, name):
    x_arr = np.asarray(x_2d, dtype=float)
    y_arr = np.asarray(y_2d, dtype=float)
    if x_arr.shape != y_arr.shape or x_arr.ndim != 2:
        raise ValueError(f"{name}: X/Y must be equal-shape 2D arrays")
    standard_x = np.asarray(x_arr[0, :], dtype=float)
    standard_y = np.asarray(y_arr[:, 0], dtype=float)
    if np.allclose(x_arr, standard_x[np.newaxis, :]) and np.allclose(
        y_arr, standard_y[:, np.newaxis]
    ):
        x_axis, y_axis = standard_x, standard_y
        transpose_spatial_axes = False
    else:
        transposed_x = np.asarray(x_arr[:, 0], dtype=float)
        transposed_y = np.asarray(y_arr[0, :], dtype=float)
        if not (
            np.allclose(x_arr, transposed_x[:, np.newaxis])
            and np.allclose(y_arr, transposed_y[np.newaxis, :])
        ):
            raise ValueError(f"{name}: X/Y grid is not rectilinear")
        x_axis, y_axis = transposed_x, transposed_y
        transpose_spatial_axes = True
    if np.any(np.diff(x_axis) <= 0.0) or np.any(np.diff(y_axis) <= 0.0):
        raise ValueError(f"{name}: grid axes must be strictly increasing")
    return x_axis, y_axis, transpose_spatial_axes


def _grid_spacing(axis, name):
    diffs = np.diff(np.asarray(axis, dtype=float))
    if diffs.size == 0 or not np.allclose(diffs, diffs[0]):
        raise ValueError(f"{name}: grid spacing must be uniform")
    return float(diffs[0])


def _interp2(x_axis, y_axis, values, x_query, y_query):
    interpolator = RegularGridInterpolator(
        (y_axis, x_axis),
        np.asarray(values, dtype=float),
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    points = np.column_stack([y_query, x_query])
    return np.asarray(interpolator(points), dtype=float)


def _monthly_wind_paths():
    paths = [WIND_DIR / f"AirRisk_Data_{month}.mat" for month in range(1, 13)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing monthly wind files:\n" + "\n".join(str(path) for path in missing)
        )
    return paths


def _load_wind_data():
    paths = _monthly_wind_paths()
    months = []
    base_x = base_y = base_z = None
    display_u_sum = display_v_sum = display_count = None
    display_altitude = 0.5 * (VERTIPORT_ALT_MSL_M + TARGET_ALT_MSL_M)

    for month, path in enumerate(paths, 1):
        data = loadmat(
            path,
            variable_names=["X_2d", "Y_2d", "z_vec", "U3d", "V3d", "theta3d"],
        )
        x_2d = np.asarray(data["X_2d"], dtype=float)
        y_2d = np.asarray(data["Y_2d"], dtype=float)
        x_axis, y_axis, transpose_spatial_axes = _grid_axes(x_2d, y_2d, path.name)
        z_axis = np.asarray(data["z_vec"], dtype=float).ravel()
        u = np.asarray(data["U3d"], dtype=float)
        v = np.asarray(data["V3d"], dtype=float)
        theta = np.asarray(data["theta3d"], dtype=float)
        if transpose_spatial_axes:
            u = np.transpose(u, (1, 0, 2))
            v = np.transpose(v, (1, 0, 2))
            theta = np.transpose(theta, (1, 0, 2))
        expected_shape = (y_axis.size, x_axis.size, z_axis.size)
        if u.shape != expected_shape or v.shape != expected_shape or theta.shape != expected_shape:
            raise ValueError(f"{path.name}: wind shape does not match {expected_shape}")
        if base_x is None:
            base_x, base_y, base_z = x_axis, y_axis, z_axis
            display_u_sum = np.zeros(expected_shape[:2], dtype=float)
            display_v_sum = np.zeros(expected_shape[:2], dtype=float)
            display_count = np.zeros(expected_shape[:2], dtype=int)
        elif (
            not np.array_equal(x_axis, base_x)
            or not np.array_equal(y_axis, base_y)
            or not np.array_equal(z_axis, base_z)
        ):
            raise ValueError(f"{path.name}: monthly wind grids are not identical")

        valid = (
            np.isfinite(u)
            & np.isfinite(v)
            & (u != -1.0)
            & (v != -1.0)
            & (theta != -1.0)
            & ~((u == 0.0) & (v == 0.0))
        )
        points_axes = (y_axis, x_axis, z_axis)
        months.append(
            {
                "month": month,
                "u_num": RegularGridInterpolator(
                    points_axes,
                    np.where(valid, u, 0.0),
                    bounds_error=False,
                    fill_value=0.0,
                ),
                "v_num": RegularGridInterpolator(
                    points_axes,
                    np.where(valid, v, 0.0),
                    bounds_error=False,
                    fill_value=0.0,
                ),
                "weight": RegularGridInterpolator(
                    points_axes,
                    valid.astype(float),
                    bounds_error=False,
                    fill_value=0.0,
                ),
            }
        )

        display_idx = int(np.argmin(np.abs(z_axis - display_altitude)))
        display_valid = valid[:, :, display_idx]
        display_u_sum += np.where(display_valid, u[:, :, display_idx], 0.0)
        display_v_sum += np.where(display_valid, v[:, :, display_idx], 0.0)
        display_count += display_valid

    display_u = np.divide(
        display_u_sum,
        display_count,
        out=np.full_like(display_u_sum, np.nan),
        where=display_count > 0,
    )
    display_v = np.divide(
        display_v_sum,
        display_count,
        out=np.full_like(display_v_sum, np.nan),
        where=display_count > 0,
    )
    return {
        "x_axis": base_x,
        "y_axis": base_y,
        "z_axis": base_z,
        "months": months,
        "display_u": display_u,
        "display_v": display_v,
        "display_altitude_msl_m": float(base_z[display_idx]),
    }


def _sample_wind_month(month_data, x, y, altitude_msl):
    points = np.column_stack([y, x, altitude_msl])
    weight = np.asarray(month_data["weight"](points), dtype=float)
    valid = weight >= 0.5
    u = np.full(weight.shape, np.nan, dtype=float)
    v = np.full(weight.shape, np.nan, dtype=float)
    u[valid] = np.asarray(month_data["u_num"](points), dtype=float)[valid] / weight[valid]
    v[valid] = np.asarray(month_data["v_num"](points), dtype=float)[valid] / weight[valid]
    return u, v, valid


def _load_air_risk(reference_x, reference_y):
    raw = np.load(AIR_RISK_PATH, allow_pickle=True).item()
    x_2d = np.asarray(raw["X_2d"], dtype=float)
    y_2d = np.asarray(raw["Y_2d"], dtype=float)
    x_axis, y_axis, transpose_spatial_axes = _grid_axes(x_2d, y_2d, "Air risk")
    if not np.array_equal(x_axis, reference_x) or not np.array_equal(y_axis, reference_y):
        raise ValueError("Air-risk and wind grids do not match")
    z_key = "z_vec" if "z_vec" in raw else "altitude_vec"
    z_axis = np.asarray(raw[z_key], dtype=float).ravel()
    risk = np.asarray(raw["Risk_3d"], dtype=float)
    if transpose_spatial_axes:
        risk = np.transpose(risk, (1, 0, 2))
    if risk.shape != (y_axis.size, x_axis.size, z_axis.size):
        raise ValueError("Air-risk array shape does not match its X/Y/Z grid")
    finite = risk[np.isfinite(risk)]
    vmax = float(np.max(finite)) if finite.size else 0.0
    normalized = np.divide(risk, vmax, out=np.zeros_like(risk), where=vmax > 0.0)
    return {
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "risk": normalized,
        "raw_max": vmax,
        "display": np.max(
            normalized[:, :, (z_axis >= VERTIPORT_ALT_MSL_M) & (z_axis <= TARGET_ALT_MSL_M)],
            axis=2,
        ),
    }


def _sample_air_risk(air_data, x, y, altitude_msl):
    result = np.full(np.asarray(x).shape, np.nan, dtype=float)
    layer_indices = np.argmin(
        np.abs(np.asarray(altitude_msl)[:, None] - air_data["z_axis"][None, :]),
        axis=1,
    )
    for layer_idx in np.unique(layer_indices):
        mask = layer_indices == layer_idx
        result[mask] = _interp2(
            air_data["x_axis"],
            air_data["y_axis"],
            air_data["risk"][:, :, int(layer_idx)],
            np.asarray(x)[mask],
            np.asarray(y)[mask],
        )
    return result, layer_indices


def _load_ground_risk(reference_x, reference_y):
    raw = np.asarray(np.load(GROUND_RISK_PATH, allow_pickle=True), dtype=float)
    if raw.ndim != 4 or raw.shape[2] < 1 or raw.shape[3] < 11:
        raise ValueError(f"Unexpected ground-risk shape: {raw.shape}")
    selected = np.asarray(raw[:, :, 0, 3:], dtype=float)
    if selected.shape != (reference_y.size, reference_x.size, 8):
        raise ValueError(
            "Ground risk must align to the wind grid and contain 8 heading maps"
        )
    minimum = float(np.nanmin(selected))
    shifted = selected - minimum
    maximum = float(np.nanmax(shifted))
    normalized = shifted / maximum if maximum > 0.0 else shifted
    return {
        "risk": normalized,
        "raw_min": minimum,
        "shifted_max": maximum,
        "display": np.mean(normalized, axis=2),
    }


def _load_moc_layers():
    layers = {}
    reference_x = reference_y = None
    for agl in MOC_AGL_LEVELS_M.astype(int):
        path = MOC_DIR / f"UAM_MOC_XYZ_risk_fixedAGL{agl}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing MOC layer: {path}")
        raw = np.asarray(np.load(path), dtype=float)
        if raw.ndim != 2 or raw.shape[1] < 4:
            raise ValueError(f"Invalid MOC layer shape: {path.name} {raw.shape}")
        x_axis = np.sort(np.unique(raw[:, 0]))
        y_axis = np.sort(np.unique(raw[:, 1]))
        if reference_x is None:
            reference_x, reference_y = x_axis, y_axis
        elif not np.array_equal(x_axis, reference_x) or not np.array_equal(y_axis, reference_y):
            raise ValueError("MOC layers do not share an identical grid")
        grid = np.full((y_axis.size, x_axis.size), np.nan, dtype=float)
        ix = np.searchsorted(x_axis, raw[:, 0])
        iy = np.searchsorted(y_axis, raw[:, 1])
        grid[iy, ix] = raw[:, 3]
        layers[int(agl)] = grid
    return {
        "x_axis": reference_x,
        "y_axis": reference_y,
        "dx": _grid_spacing(reference_x, "MOC X"),
        "dy": _grid_spacing(reference_y, "MOC Y"),
        "layers": layers,
        "display_union": np.max(
            np.stack([layers[100], layers[200], layers[300]], axis=2), axis=2
        ),
    }


def _moc_floor_agl(query_msl):
    query_agl = np.asarray(query_msl, dtype=float) - MOC_REFERENCE_MSL_M
    indices = np.searchsorted(MOC_AGL_LEVELS_M, query_agl + 1e-6, side="right") - 1
    return MOC_AGL_LEVELS_M[np.clip(indices, 0, MOC_AGL_LEVELS_M.size - 1)].astype(int)


def _sample_moc(moc_data, x, y, query_msl):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    selected_agl = _moc_floor_agl(query_msl)
    ix = np.rint((x - moc_data["x_axis"][0]) / moc_data["dx"]).astype(int)
    iy = np.rint((y - moc_data["y_axis"][0]) / moc_data["dy"]).astype(int)
    inside = (
        (x >= moc_data["x_axis"][0] - 0.5 * moc_data["dx"])
        & (x <= moc_data["x_axis"][-1] + 0.5 * moc_data["dx"])
        & (y >= moc_data["y_axis"][0] - 0.5 * moc_data["dy"])
        & (y <= moc_data["y_axis"][-1] + 0.5 * moc_data["dy"])
        & (ix >= 0)
        & (ix < moc_data["x_axis"].size)
        & (iy >= 0)
        & (iy < moc_data["y_axis"].size)
    )
    values = np.full(x.shape, np.nan, dtype=float)
    for agl in np.unique(selected_agl):
        mask = inside & (selected_agl == agl)
        values[mask] = moc_data["layers"][int(agl)][iy[mask], ix[mask]]
    blocked = inside & (values >= 0.5)
    return values, selected_agl, inside, blocked, iy, ix


def _transition_samples(vx, vy, sector, direction, total_distance_m):
    bearing_out = _sector_bearing(sector - 1)
    outward = _heading_unit(bearing_out)
    if direction == "takeoff":
        flight_heading = bearing_out
        flight_unit = outward
    elif direction == "landing":
        flight_heading = (bearing_out + 180.0) % 360.0
        flight_unit = -outward
    else:
        raise ValueError(f"Unknown direction: {direction}")
    normal = np.array([-flight_unit[1], flight_unit[0]], dtype=float)

    station_count = max(2, int(np.ceil(total_distance_m / ALONG_TRACK_SAMPLE_STEP_M)) + 1)
    along = np.linspace(0.0, total_distance_m, station_count)
    cross_step = float(np.clip(TRANSITION_CORRIDOR_HALF_WIDTH_M / 3.0, 20.0, 100.0))
    cross_count = max(
        3,
        int(np.ceil(2.0 * TRANSITION_CORRIDOR_HALF_WIDTH_M / cross_step)) + 1,
    )
    cross = np.linspace(
        -TRANSITION_CORRIDOR_HALF_WIDTH_M,
        TRANSITION_CORRIDOR_HALF_WIDTH_M,
        cross_count,
    )
    if direction == "takeoff":
        radial = along
        altitude = VERTIPORT_ALT_MSL_M + along * np.tan(
            np.deg2rad(CLIMB_DESCENT_ANGLE_DEG)
        )
    else:
        radial = total_distance_m - along
        altitude = TARGET_ALT_MSL_M - along * np.tan(
            np.deg2rad(CLIMB_DESCENT_ANGLE_DEG)
        )
    altitude = np.clip(altitude, VERTIPORT_ALT_MSL_M, TARGET_ALT_MSL_M)
    center_x = vx + radial * outward[0]
    center_y = vy + radial * outward[1]

    station_index = np.repeat(np.arange(station_count), cross_count)
    cross_flat = np.tile(cross, station_count)
    x = np.repeat(center_x, cross_count) + cross_flat * normal[0]
    y = np.repeat(center_y, cross_count) + cross_flat * normal[1]
    return {
        "sector": int(sector),
        "direction": direction,
        "bearing_out_deg": float(bearing_out),
        "flight_heading_deg": float(flight_heading),
        "flight_unit": flight_unit,
        "station_index": station_index,
        "along_m": np.repeat(along, cross_count),
        "radial_m": np.repeat(radial, cross_count),
        "cross_m": cross_flat,
        "center_x": np.repeat(center_x, cross_count),
        "center_y": np.repeat(center_y, cross_count),
        "x": x,
        "y": y,
        "altitude_msl": np.repeat(altitude, cross_count),
        "station_count": station_count,
        "cross_count": cross_count,
        "centerline_x": center_x,
        "centerline_y": center_y,
        "centerline_altitude_msl": altitude,
    }


def _mean_or_nan(values):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    return float(np.mean(arr[finite])) if np.any(finite) else float("nan")


def _evaluate_direction(samples, moc_data, ground_data, air_data, wind_data):
    altitude = samples["altitude_msl"]
    effective_clearance = np.minimum(
        TRANSITION_DOWNWARD_CLEARANCE_M,
        np.maximum(0.0, altitude - VERTIPORT_ALT_MSL_M),
    )
    lower_face_msl = altitude - effective_clearance
    moc_value, moc_agl, moc_inside, moc_blocked, moc_row, moc_col = _sample_moc(
        moc_data, samples["x"], samples["y"], lower_face_msl
    )

    heading_index = int(round(samples["flight_heading_deg"] / 45.0) % 8)
    ground_value = _interp2(
        wind_data["x_axis"],
        wind_data["y_axis"],
        ground_data["risk"][:, :, heading_index],
        samples["x"],
        samples["y"],
    )
    air_value, air_layer_index = _sample_air_risk(
        air_data, samples["x"], samples["y"], altitude
    )

    monthly_rows = []
    monthly_tail = []
    monthly_cross = []
    monthly_head = []
    monthly_u = []
    monthly_v = []
    valid_wind_total = 0
    for month_data in wind_data["months"]:
        u, v, wind_valid = _sample_wind_month(
            month_data, samples["x"], samples["y"], altitude
        )
        along_component = u * samples["flight_unit"][0] + v * samples["flight_unit"][1]
        cross_component = np.abs(
            u * samples["flight_unit"][1] - v * samples["flight_unit"][0]
        )
        tailwind = np.maximum(along_component, 0.0)
        headwind = np.maximum(-along_component, 0.0)
        valid_wind_total += int(np.count_nonzero(wind_valid))
        monthly_rows.append(
            {
                "Sector": samples["sector"],
                "Direction": samples["direction"],
                "Flight_Heading_deg": samples["flight_heading_deg"],
                "Month": month_data["month"],
                "Mean_U_mps": _mean_or_nan(u),
                "Mean_V_mps": _mean_or_nan(v),
                "Mean_Tailwind_mps": _mean_or_nan(tailwind),
                "Mean_Crosswind_mps": _mean_or_nan(cross_component),
                "Mean_Headwind_mps": _mean_or_nan(headwind),
                "Valid_Sample_Count": int(np.count_nonzero(wind_valid)),
                "Total_Sample_Count": int(wind_valid.size),
                "Coverage_Ratio": float(np.mean(wind_valid)),
            }
        )
        monthly_tail.append(tailwind)
        monthly_cross.append(cross_component)
        monthly_head.append(headwind)
        monthly_u.append(u)
        monthly_v.append(v)

    tail_stack = np.asarray(monthly_tail, dtype=float)
    cross_stack = np.asarray(monthly_cross, dtype=float)
    head_stack = np.asarray(monthly_head, dtype=float)
    u_stack = np.asarray(monthly_u, dtype=float)
    v_stack = np.asarray(monthly_v, dtype=float)
    tail_sample_mean = np.divide(
        np.nansum(tail_stack, axis=0),
        np.sum(np.isfinite(tail_stack), axis=0),
        out=np.full(tail_stack.shape[1], np.nan),
        where=np.sum(np.isfinite(tail_stack), axis=0) > 0,
    )
    cross_sample_mean = np.divide(
        np.nansum(cross_stack, axis=0),
        np.sum(np.isfinite(cross_stack), axis=0),
        out=np.full(cross_stack.shape[1], np.nan),
        where=np.sum(np.isfinite(cross_stack), axis=0) > 0,
    )
    head_sample_mean = np.divide(
        np.nansum(head_stack, axis=0),
        np.sum(np.isfinite(head_stack), axis=0),
        out=np.full(head_stack.shape[1], np.nan),
        where=np.sum(np.isfinite(head_stack), axis=0) > 0,
    )

    tested_cells = {
        (int(agl), int(row), int(col))
        for agl, row, col, inside in zip(moc_agl, moc_row, moc_col, moc_inside)
        if bool(inside)
    }
    blocked_cells = {
        (int(agl), int(row), int(col))
        for agl, row, col, blocked in zip(moc_agl, moc_row, moc_col, moc_blocked)
        if bool(blocked)
    }
    total_cells = len(tested_cells)
    blocked_cell_count = len(blocked_cells)
    out_of_grid_count = int(np.count_nonzero(~moc_inside))
    moc_pass = blocked_cell_count == 0 and out_of_grid_count == 0 and total_cells > 0
    total_wind_samples = int(samples["x"].size * len(wind_data["months"]))
    ground_coverage = float(np.mean(np.isfinite(ground_value)))
    air_coverage = float(np.mean(np.isfinite(air_value)))
    wind_coverage = float(valid_wind_total / max(total_wind_samples, 1))

    result = {
        "sector": samples["sector"],
        "direction": samples["direction"],
        "bearing_out_deg": samples["bearing_out_deg"],
        "flight_heading_deg": samples["flight_heading_deg"],
        "ground_heading_index": heading_index,
        "moc_blocked_cells": blocked_cell_count,
        "moc_tested_cells": total_cells,
        "moc_blocked_samples": int(np.count_nonzero(moc_blocked)),
        "moc_tested_samples": int(np.count_nonzero(moc_inside)),
        "moc_out_of_grid_samples": out_of_grid_count,
        "moc_blocked_ratio": float(blocked_cell_count / max(total_cells, 1)),
        "moc_safety_score": float(1.0 - blocked_cell_count / max(total_cells, 1)),
        "moc_requirement_met": bool(moc_pass),
        "ground_risk_raw": _mean_or_nan(ground_value),
        "air_risk_raw": _mean_or_nan(air_value),
        "tailwind_mps_raw": _mean_or_nan(tail_stack),
        "crosswind_mps_raw": _mean_or_nan(cross_stack),
        "headwind_mps_raw": _mean_or_nan(head_stack),
        "wind_u_mps": _mean_or_nan(u_stack),
        "wind_v_mps": _mean_or_nan(v_stack),
        "ground_coverage_ratio": ground_coverage,
        "air_coverage_ratio": air_coverage,
        "wind_coverage_ratio": wind_coverage,
        "data_coverage_complete": bool(
            ground_coverage >= 0.999
            and air_coverage >= 0.999
            and wind_coverage >= 0.999
        ),
        "centerline_x": samples["centerline_x"],
        "centerline_y": samples["centerline_y"],
        "centerline_altitude_msl": samples["centerline_altitude_msl"],
    }

    audit_rows = []
    for idx in range(samples["x"].size):
        if moc_blocked[idx]:
            moc_status = "MOC_INTERSECTION"
        elif not moc_inside[idx]:
            moc_status = "OUT_OF_GRID"
        else:
            moc_status = "CLEAR"
        audit_rows.append(
            {
                "Sector": samples["sector"],
                "Direction": samples["direction"],
                "Flight_Heading_deg": samples["flight_heading_deg"],
                "Ground_Heading_Index": heading_index,
                "Station_Index": int(samples["station_index"][idx]),
                "Along_Track_m": float(samples["along_m"][idx]),
                "Radial_Distance_From_Vertiport_m": float(samples["radial_m"][idx]),
                "Cross_Track_m": float(samples["cross_m"][idx]),
                "X_EPSG5179_m": float(samples["x"][idx]),
                "Y_EPSG5179_m": float(samples["y"][idx]),
                "Center_X_EPSG5179_m": float(samples["center_x"][idx]),
                "Center_Y_EPSG5179_m": float(samples["center_y"][idx]),
                "Center_Altitude_MSL_m": float(altitude[idx]),
                "Effective_Downward_Clearance_m": float(effective_clearance[idx]),
                "Corridor_Lower_Face_MSL_m": float(lower_face_msl[idx]),
                "Selected_MOC_AGL_m": int(moc_agl[idx]),
                "MOC_Grid_Row": int(moc_row[idx]),
                "MOC_Grid_Col": int(moc_col[idx]),
                "MOC_Value": None if not np.isfinite(moc_value[idx]) else float(moc_value[idx]),
                "MOC_Status": moc_status,
                "Ground_Risk_Raw": None if not np.isfinite(ground_value[idx]) else float(ground_value[idx]),
                "Air_Risk_Raw": None if not np.isfinite(air_value[idx]) else float(air_value[idx]),
                "Air_Risk_Source_MSL_m": float(air_data["z_axis"][air_layer_index[idx]]),
                "Mean_12Month_Tailwind_mps": None if not np.isfinite(tail_sample_mean[idx]) else float(tail_sample_mean[idx]),
                "Mean_12Month_Crosswind_mps": None if not np.isfinite(cross_sample_mean[idx]) else float(cross_sample_mean[idx]),
                "Mean_12Month_Headwind_mps": None if not np.isfinite(head_sample_mean[idx]) else float(head_sample_mean[idx]),
            }
        )
    return result, audit_rows, monthly_rows


def evaluate_sectors():
    _validate_parameters()
    vx, vy = _to_5179(VERTIPORT_LAT, VERTIPORT_LON)
    vertical_height_m = TARGET_ALT_MSL_M - VERTIPORT_ALT_MSL_M
    total_distance_m = vertical_height_m / np.tan(np.deg2rad(CLIMB_DESCENT_ANGLE_DEG))

    wind_data = _load_wind_data()
    air_data = _load_air_risk(wind_data["x_axis"], wind_data["y_axis"])
    ground_data = _load_ground_risk(wind_data["x_axis"], wind_data["y_axis"])
    moc_data = _load_moc_layers()

    direction_results = []
    audit_rows = []
    monthly_rows = []
    for sector in range(1, N_SECTORS + 1):
        for direction in ("takeoff", "landing"):
            samples = _transition_samples(vx, vy, sector, direction, total_distance_m)
            result, sample_rows, month_rows = _evaluate_direction(
                samples, moc_data, ground_data, air_data, wind_data
            )
            direction_results.append(result)
            audit_rows.extend(sample_rows)
            monthly_rows.extend(month_rows)

    normalization = {}
    raw_fields = {
        "tailwind": "tailwind_mps_raw",
        "crosswind": "crosswind_mps_raw",
        "ground": "ground_risk_raw",
        "air": "air_risk_raw",
    }
    normalized_values = {}
    for label, field in raw_fields.items():
        normalized, lo, hi = _normalize01([row[field] for row in direction_results])
        normalized_values[label] = normalized
        normalization[label] = {"minimum": lo, "maximum": hi}

    for idx, row in enumerate(direction_results):
        row["tailwind_risk_score"] = float(normalized_values["tailwind"][idx])
        row["crosswind_risk_score"] = float(normalized_values["crosswind"][idx])
        row["wind_risk_score"] = float(
            WIND_TAIL_WEIGHT * row["tailwind_risk_score"]
            + WIND_CROSS_WEIGHT * row["crosswind_risk_score"]
        )
        row["wind_safety_score"] = float(1.0 - row["wind_risk_score"])
        row["ground_risk_score"] = float(normalized_values["ground"][idx])
        row["air_risk_score"] = float(normalized_values["air"][idx])
        row["combined_risk_score"] = float(
            WIND_RISK_WEIGHT * row["wind_risk_score"]
            + GROUND_RISK_WEIGHT * row["ground_risk_score"]
            + AIR_RISK_WEIGHT * row["air_risk_score"]
        )

    lookup = {
        (row["sector"], row["direction"]): row for row in direction_results
    }
    metrics = []
    for sector in range(1, N_SECTORS + 1):
        takeoff = lookup[(sector, "takeoff")]
        landing = lookup[(sector, "landing")]
        mean_u = 0.5 * (takeoff["wind_u_mps"] + landing["wind_u_mps"])
        mean_v = 0.5 * (takeoff["wind_v_mps"] + landing["wind_v_mps"])
        toward = float((np.rad2deg(np.arctan2(mean_u, mean_v)) + 360.0) % 360.0)
        metrics.append(
            {
                "sector": sector,
                "bearing_deg": takeoff["bearing_out_deg"],
                "moc_blocked_cells": max(takeoff["moc_blocked_cells"], landing["moc_blocked_cells"]),
                "moc_total_cells": min(takeoff["moc_tested_cells"], landing["moc_tested_cells"]),
                "moc_blocked_ratio": max(takeoff["moc_blocked_ratio"], landing["moc_blocked_ratio"]),
                "moc_safety_score": min(takeoff["moc_safety_score"], landing["moc_safety_score"]),
                "moc_requirement_met": bool(takeoff["moc_requirement_met"] and landing["moc_requirement_met"]),
                "takeoff_moc_requirement_met": takeoff["moc_requirement_met"],
                "landing_moc_requirement_met": landing["moc_requirement_met"],
                "takeoff_moc_blocked_cells": takeoff["moc_blocked_cells"],
                "landing_moc_blocked_cells": landing["moc_blocked_cells"],
                "takeoff_moc_out_of_grid_samples": takeoff["moc_out_of_grid_samples"],
                "landing_moc_out_of_grid_samples": landing["moc_out_of_grid_samples"],
                "wind_u_mps": float(mean_u),
                "wind_v_mps": float(mean_v),
                "wind_speed_mps": float(np.hypot(mean_u, mean_v)),
                "wind_toward_deg": toward,
                "wind_from_deg": float((toward + 180.0) % 360.0),
                "takeoff_headwind_mps": takeoff["headwind_mps_raw"],
                "landing_headwind_mps": landing["headwind_mps_raw"],
                "takeoff_tailwind_mps": takeoff["tailwind_mps_raw"],
                "landing_tailwind_mps": landing["tailwind_mps_raw"],
                "takeoff_crosswind_mps": takeoff["crosswind_mps_raw"],
                "landing_crosswind_mps": landing["crosswind_mps_raw"],
                "takeoff_wind_score": takeoff["wind_safety_score"],
                "landing_wind_score": landing["wind_safety_score"],
                "takeoff_wind_risk_score": takeoff["wind_risk_score"],
                "landing_wind_risk_score": landing["wind_risk_score"],
                "takeoff_ground_risk_score": takeoff["ground_risk_score"],
                "landing_ground_risk_score": landing["ground_risk_score"],
                "takeoff_air_risk_score": takeoff["air_risk_score"],
                "landing_air_risk_score": landing["air_risk_score"],
                "takeoff_combined_risk_score": takeoff["combined_risk_score"],
                "landing_combined_risk_score": landing["combined_risk_score"],
                "ground_risk_score": float(0.5 * (takeoff["ground_risk_score"] + landing["ground_risk_score"])),
                "air_risk_score": float(0.5 * (takeoff["air_risk_score"] + landing["air_risk_score"])),
                "combined_risk_score": float(0.5 * (takeoff["combined_risk_score"] + landing["combined_risk_score"])),
                "takeoff_coverage_complete": takeoff["data_coverage_complete"],
                "landing_coverage_complete": landing["data_coverage_complete"],
                "takeoff_wind_coverage_ratio": takeoff["wind_coverage_ratio"],
                "landing_wind_coverage_ratio": landing["wind_coverage_ratio"],
                "takeoff_ground_coverage_ratio": takeoff["ground_coverage_ratio"],
                "landing_ground_coverage_ratio": landing["ground_coverage_ratio"],
                "takeoff_air_coverage_ratio": takeoff["air_coverage_ratio"],
                "landing_air_coverage_ratio": landing["air_coverage_ratio"],
            }
        )

    plot_data = {
        "vx": vx,
        "vy": vy,
        "total_distance_m": float(total_distance_m),
        "vertical_height_m": float(vertical_height_m),
        "moc": moc_data,
        "ground": ground_data,
        "air": air_data,
        "wind": wind_data,
        "directions": direction_results,
        "audit_rows": audit_rows,
        "monthly_rows": monthly_rows,
        "normalization": normalization,
    }
    return metrics, plot_data


def build_combination_ranking(metrics):
    combinations = []
    for takeoff_idx in range(N_SECTORS):
        for landing_idx in range(N_SECTORS):
            if takeoff_idx == landing_idx:
                continue
            takeoff = metrics[takeoff_idx]
            landing = metrics[landing_idx]
            moc_clear = bool(
                takeoff["takeoff_moc_requirement_met"]
                and landing["landing_moc_requirement_met"]
            )
            coverage_complete = bool(
                takeoff["takeoff_coverage_complete"]
                and landing["landing_coverage_complete"]
            )
            # MOC is the requested hard constraint.  Wind/ground/air coverage
            # is retained as an audit warning and does not silently veto a
            # sector when finite samples are available for its risk means.
            required_conditions_met = bool(moc_clear)
            moc_issue_score = float(
                0.5 * (
                    takeoff["moc_blocked_ratio"] + landing["moc_blocked_ratio"]
                )
            )
            wind_risk = float(
                0.5 * (
                    takeoff["takeoff_wind_risk_score"]
                    + landing["landing_wind_risk_score"]
                )
            )
            ground_risk = float(
                0.5 * (
                    takeoff["takeoff_ground_risk_score"]
                    + landing["landing_ground_risk_score"]
                )
            )
            air_risk = float(
                0.5 * (
                    takeoff["takeoff_air_risk_score"]
                    + landing["landing_air_risk_score"]
                )
            )
            combined_risk = float(
                WIND_RISK_WEIGHT * wind_risk
                + GROUND_RISK_WEIGHT * ground_risk
                + AIR_RISK_WEIGHT * air_risk
            )
            combinations.append(
                {
                    "takeoff_sector": takeoff_idx + 1,
                    "landing_sector": landing_idx + 1,
                    "takeoff_moc_pass": takeoff["takeoff_moc_requirement_met"],
                    "landing_moc_pass": landing["landing_moc_requirement_met"],
                    "takeoff_moc_blocked_cells": takeoff["takeoff_moc_blocked_cells"],
                    "landing_moc_blocked_cells": landing["landing_moc_blocked_cells"],
                    "takeoff_moc_out_of_grid_samples": takeoff["takeoff_moc_out_of_grid_samples"],
                    "landing_moc_out_of_grid_samples": landing["landing_moc_out_of_grid_samples"],
                    "takeoff_wind_risk_score": takeoff["takeoff_wind_risk_score"],
                    "landing_wind_risk_score": landing["landing_wind_risk_score"],
                    "takeoff_ground_risk_score": takeoff["takeoff_ground_risk_score"],
                    "landing_ground_risk_score": landing["landing_ground_risk_score"],
                    "takeoff_air_risk_score": takeoff["takeoff_air_risk_score"],
                    "landing_air_risk_score": landing["landing_air_risk_score"],
                    "wind_risk_score": wind_risk,
                    "ground_risk_score": ground_risk,
                    "air_risk_score": air_risk,
                    "combined_risk_score": combined_risk,
                    "moc_issue_score": moc_issue_score,
                    "moc_clear": moc_clear,
                    "coverage_complete": coverage_complete,
                    "required_conditions_met": required_conditions_met,
                    "is_reference_s7_s5": bool(
                        takeoff_idx + 1 == REFERENCE_TAKEOFF_SECTOR
                        and landing_idx + 1 == REFERENCE_LANDING_SECTOR
                    ),
                }
            )

    combinations.sort(
        key=lambda row: (
            not row["required_conditions_met"],
            row["moc_issue_score"] if not row["required_conditions_met"] else 0.0,
            row["combined_risk_score"],
            row["takeoff_sector"],
            row["landing_sector"],
        )
    )
    eligible_rank = 0
    for rank, row in enumerate(combinations, 1):
        row["comparison_rank"] = rank
        if row["required_conditions_met"]:
            eligible_rank += 1
            row["eligible_rank"] = eligible_rank
        else:
            row["eligible_rank"] = None
    return combinations


def _best_combination(combinations):
    return next((row for row in combinations if row["required_conditions_met"]), None)


def validate_results(metrics, combinations, plot_data):
    if len(metrics) != N_SECTORS:
        raise AssertionError(f"Expected {N_SECTORS} sector metrics")
    if len(combinations) != N_SECTORS * (N_SECTORS - 1):
        raise AssertionError("Expected 132 distinct takeoff/landing combinations")
    expected_distance = (TARGET_ALT_MSL_M - VERTIPORT_ALT_MSL_M) / np.tan(
        np.deg2rad(CLIMB_DESCENT_ANGLE_DEG)
    )
    if not np.isclose(plot_data["total_distance_m"], expected_distance, atol=1e-6):
        raise AssertionError("Transition horizontal-distance calculation mismatch")
    if int(_moc_floor_agl(np.array([350.0]))[0]) != 200:
        raise AssertionError("MSL350 lower face must select AGL200 MOC")
    if not np.isclose(
        min(TRANSITION_DOWNWARD_CLEARANCE_M, 450.0 - VERTIPORT_ALT_MSL_M),
        100.0,
    ):
        raise AssertionError("MSL450 effective clearance must be 100m")
    for sector in range(1, N_SECTORS + 1):
        takeoff = next(
            row for row in plot_data["directions"]
            if row["sector"] == sector and row["direction"] == "takeoff"
        )
        landing = next(
            row for row in plot_data["directions"]
            if row["sector"] == sector and row["direction"] == "landing"
        )
        if (
            takeoff["moc_blocked_cells"] != landing["moc_blocked_cells"]
            or takeoff["moc_out_of_grid_samples"] != landing["moc_out_of_grid_samples"]
        ):
            raise AssertionError(f"S{sector}: takeoff/landing MOC must be symmetric")
        if not np.all(np.diff(takeoff["centerline_altitude_msl"]) >= -1e-9):
            raise AssertionError(f"S{sector}: takeoff altitude is not monotonic")
        if not np.all(np.diff(landing["centerline_altitude_msl"]) <= 1e-9):
            raise AssertionError(f"S{sector}: landing altitude is not monotonic")
    for row in metrics:
        for key, value in row.items():
            if key.endswith("_score") and not -1e-9 <= float(value) <= 1.0 + 1e-9:
                raise AssertionError(f"S{row['sector']} {key} outside [0,1]: {value}")
    best = _best_combination(combinations)
    if best is not None and not (
        best["takeoff_moc_pass"] and best["landing_moc_pass"]
    ):
        raise AssertionError("An infeasible combination was selected")


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_outputs(metrics, combinations, plot_data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best = _best_combination(combinations)
    top_rows = combinations[:TOP_N]

    sector_rows = []
    for row in metrics:
        sector_rows.append(
            {
                "섹터": row["sector"],
                "중심방향_deg": row["bearing_deg"],
                "MOC_필수조건통과": row["moc_requirement_met"],
                "MOC_차단셀수": row["moc_blocked_cells"],
                "MOC_전체셀수": row["moc_total_cells"],
                "MOC_차단비율": row["moc_blocked_ratio"],
                "MOC_안전점수": row["moc_safety_score"],
                "이륙_MOC통과": row["takeoff_moc_requirement_met"],
                "착륙_MOC통과": row["landing_moc_requirement_met"],
                "이륙_MOC범위밖표본": row["takeoff_moc_out_of_grid_samples"],
                "착륙_MOC범위밖표본": row["landing_moc_out_of_grid_samples"],
                "평균풍속_mps": row["wind_speed_mps"],
                "바람이향하는방향_deg": row["wind_toward_deg"],
                "바람이불어오는방향_deg": row["wind_from_deg"],
                "이륙_순풍_mps": row["takeoff_tailwind_mps"],
                "착륙_순풍_mps": row["landing_tailwind_mps"],
                "이륙_측풍_mps": row["takeoff_crosswind_mps"],
                "착륙_측풍_mps": row["landing_crosswind_mps"],
                "이륙_역풍_mps": row["takeoff_headwind_mps"],
                "착륙_역풍_mps": row["landing_headwind_mps"],
                "이륙_바람위험도": row["takeoff_wind_risk_score"],
                "착륙_바람위험도": row["landing_wind_risk_score"],
                "이륙_바람안전점수": row["takeoff_wind_score"],
                "착륙_바람안전점수": row["landing_wind_score"],
                "이륙_지상위험도": row["takeoff_ground_risk_score"],
                "착륙_지상위험도": row["landing_ground_risk_score"],
                "이륙_공중위험도": row["takeoff_air_risk_score"],
                "착륙_공중위험도": row["landing_air_risk_score"],
                "이륙_종합위험도": row["takeoff_combined_risk_score"],
                "착륙_종합위험도": row["landing_combined_risk_score"],
                "평균_지상위험도": row["ground_risk_score"],
                "평균_공중위험도": row["air_risk_score"],
                "평균_종합위험도": row["combined_risk_score"],
                "이륙_바람coverage": row["takeoff_wind_coverage_ratio"],
                "착륙_바람coverage": row["landing_wind_coverage_ratio"],
                "이륙_지상coverage": row["takeoff_ground_coverage_ratio"],
                "착륙_지상coverage": row["landing_ground_coverage_ratio"],
                "이륙_공중coverage": row["takeoff_air_coverage_ratio"],
                "착륙_공중coverage": row["landing_air_coverage_ratio"],
            }
        )
    combination_rows = []
    for row in combinations:
        combination_rows.append(
            {
                "비교순위": row["comparison_rank"],
                "MOC통과순위": row["eligible_rank"],
                "이륙섹터": row["takeoff_sector"],
                "착륙섹터": row["landing_sector"],
                "이륙_MOC통과": row["takeoff_moc_pass"],
                "착륙_MOC통과": row["landing_moc_pass"],
                "이륙_MOC차단셀": row["takeoff_moc_blocked_cells"],
                "착륙_MOC차단셀": row["landing_moc_blocked_cells"],
                "이륙_MOC범위밖표본": row["takeoff_moc_out_of_grid_samples"],
                "착륙_MOC범위밖표본": row["landing_moc_out_of_grid_samples"],
                "바람위험도": row["wind_risk_score"],
                "지상위험도": row["ground_risk_score"],
                "공중위험도": row["air_risk_score"],
                "종합위험도": row["combined_risk_score"],
                "MOC_문제점수": row["moc_issue_score"],
                "MOC필수조건통과": row["moc_clear"],
                "데이터coverage통과": row["coverage_complete"],
                "최종선정가능": row["required_conditions_met"],
                "현재_S7_S5조합": row["is_reference_s7_s5"],
            }
        )
    _write_csv(OUTPUT_DIR / "sector_metrics.csv", sector_rows)
    _write_csv(OUTPUT_DIR / "sector_combination_ranking.csv", combination_rows)
    _write_csv(OUTPUT_DIR / "sector_transition_samples.csv", plot_data["audit_rows"])
    _write_csv(OUTPUT_DIR / "sector_monthly_wind_metrics.csv", plot_data["monthly_rows"])

    xw, yw = plot_data["wind"]["x_axis"], plot_data["wind"]["y_axis"]
    xm, ym = plot_data["moc"]["x_axis"], plot_data["moc"]["y_axis"]
    alignment_rows = [
        {
            "Layer": "wind/ground/air",
            "CRS": "EPSG:5179",
            "Shape": f"{yw.size}x{xw.size}",
            "X_Min_m": float(xw[0]),
            "X_Max_m": float(xw[-1]),
            "Y_Min_m": float(yw[0]),
            "Y_Max_m": float(yw[-1]),
            "dx_m": _grid_spacing(xw, "risk X"),
            "dy_m": _grid_spacing(yw, "risk Y"),
            "Note": "Exact shared projected grid; path-aware sampling",
        },
        {
            "Layer": "MOC AGL100-900",
            "CRS": "EPSG:5179",
            "Shape": f"{ym.size}x{xm.size}",
            "X_Min_m": float(xm[0]),
            "X_Max_m": float(xm[-1]),
            "Y_Min_m": float(ym[0]),
            "Y_Max_m": float(ym[-1]),
            "dx_m": plot_data["moc"]["dx"],
            "dy_m": plot_data["moc"]["dy"],
            "Note": "Nearest cell; layer selected at corridor lower face",
        },
    ]
    _write_csv(OUTPUT_DIR / "spatial_alignment_report.csv", alignment_rows)

    eligible_count = sum(row["required_conditions_met"] for row in combinations)
    lines = [
        "전이 회랑 기반 이착륙 섹터 선정 결과",
        "=" * 52,
        f"버티포트/목표고도: {VERTIPORT_ALT_MSL_M:.0f} -> {TARGET_ALT_MSL_M:.0f} m MSL",
        f"상승·하강각: {CLIMB_DESCENT_ANGLE_DEG:.1f}°",
        f"전이 수평거리: {plot_data['total_distance_m']:.3f} m",
        f"전이 회랑 반폭/총폭: {TRANSITION_CORRIDOR_HALF_WIDTH_M:.1f} / {2*TRANSITION_CORRIDOR_HALF_WIDTH_M:.1f} m",
        f"MOC 하방 이격: 최대 {TRANSITION_DOWNWARD_CLEARANCE_M:.1f} m",
        "MOC 사용층: AGL100, AGL200, AGL300 (하단면 floor 규칙)",
        "MOC 판정: 차단 0개이며 OUT_OF_GRID 0개인 방향만 통과",
        "바람위험: 0.5×12개월 평균 순풍위험 + 0.5×12개월 평균 측풍위험",
        "종합위험: 1/3×바람 + 1/3×지상 + 1/3×공중",
        "공중위험: 봄·가을 지도에서 실제 경로 MSL의 최근접 고도층 사용",
        "지상위험: 실제 비행 heading의 45° 방향층 사용",
        "",
        f"선정 가능한 조합 수: {eligible_count} / {len(combinations)}",
    ]
    if best is None:
        lines.append("최종 선정: MOC를 통과한 이착륙 조합이 없습니다.")
    else:
        lines.extend(
            [
                f"최종 선정: 이륙 S{best['takeoff_sector']} / 착륙 S{best['landing_sector']}",
                f"최종 종합위험도: {best['combined_risk_score']:.6f}",
                f"  바람/지상/공중: {best['wind_risk_score']:.6f} / {best['ground_risk_score']:.6f} / {best['air_risk_score']:.6f}",
            ]
        )
    lines.extend(["", f"상위 {TOP_N}개 조합", "-" * 52])
    for row in top_rows:
        status = "PASS" if row["required_conditions_met"] else "INFEASIBLE"
        lines.append(
            f"{row['comparison_rank']:2d}. 이륙 S{row['takeoff_sector']} / 착륙 S{row['landing_sector']} | "
            f"{status} | 종합={row['combined_risk_score']:.4f}, "
            f"바람={row['wind_risk_score']:.4f}, 지상={row['ground_risk_score']:.4f}, "
            f"공중={row['air_risk_score']:.4f}"
        )
    (OUTPUT_DIR / "sector_selection_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    result_json = {
        "configuration": {
            "vertiport": {
                "lat": VERTIPORT_LAT,
                "lon": VERTIPORT_LON,
                "altitude_msl_m": VERTIPORT_ALT_MSL_M,
            },
            "target_altitude_msl_m": TARGET_ALT_MSL_M,
            "climb_descent_angle_deg": CLIMB_DESCENT_ANGLE_DEG,
            "transition_horizontal_distance_m": plot_data["total_distance_m"],
            "transition_corridor_half_width_m": TRANSITION_CORRIDOR_HALF_WIDTH_M,
            "transition_downward_clearance_m": TRANSITION_DOWNWARD_CLEARANCE_M,
            "along_track_sample_step_m": ALONG_TRACK_SAMPLE_STEP_M,
            "moc_reference_msl_m": MOC_REFERENCE_MSL_M,
            "moc_available_agl_layers_m": MOC_AGL_LEVELS_M,
            "moc_layers_used_m": sorted(
                {row["Selected_MOC_AGL_m"] for row in plot_data["audit_rows"]}
            ),
            "weights": {
                "wind": WIND_RISK_WEIGHT,
                "ground": GROUND_RISK_WEIGHT,
                "air": AIR_RISK_WEIGHT,
                "tailwind_within_wind": WIND_TAIL_WEIGHT,
                "crosswind_within_wind": WIND_CROSS_WEIGHT,
            },
            "air_risk_season": "springfall",
            "top_n": TOP_N,
        },
        "input_files": {
            "wind": [str(path) for path in _monthly_wind_paths()],
            "ground_risk": str(GROUND_RISK_PATH),
            "air_risk": str(AIR_RISK_PATH),
            "moc_directory": str(MOC_DIR),
        },
        "normalization": plot_data["normalization"],
        "eligible_combination_count": eligible_count,
        "selected_combination": best,
        "top_combinations": top_rows,
        "sector_metrics": metrics,
        "validation_policy": {
            "moc_is_hard_constraint": True,
            "out_of_grid_is_not_collision_but_is_ineligible": True,
            "risk_data_coverage_is_diagnostic_not_hard_constraint": True,
            "effective_clearance_formula": "min(100, max(0, center_MSL-150))",
            "lower_face_formula": "center_MSL-effective_clearance",
            "moc_layer_selection": "floor_at_corridor_lower_face",
            "ground_heading_policy": "nearest_45_degree_heading_map",
            "air_altitude_policy": "nearest_MSL_layer",
            "wind_policy": "12_month_mean_tailwind_and_absolute_crosswind",
        },
    }
    with (OUTPUT_DIR / "sector_selection_results.json").open("w", encoding="utf-8") as file:
        json.dump(_json_value(result_json), file, ensure_ascii=False, indent=2, allow_nan=False)


def _corridor_polygon(vx, vy, bearing_deg, distance_m):
    heading = _heading_unit(bearing_deg)
    normal = np.array([-heading[1], heading[0]])
    start = np.array([vx, vy])
    end = start + distance_m * heading
    width = TRANSITION_CORRIDOR_HALF_WIDTH_M
    return np.vstack(
        [start + width * normal, end + width * normal, end - width * normal, start - width * normal]
    )


def _setup_local_axis(ax, plot_data, title):
    limit = plot_data["total_distance_m"] / 1000.0 * 1.08
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East [km]")
    ax.set_ylabel("North [km]")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.2)


def _draw_candidates(ax, metrics, plot_data, component=None, show_width=True):
    vx, vy = plot_data["vx"], plot_data["vy"]
    distance = plot_data["total_distance_m"]
    cmap = plt.get_cmap("viridis")
    for row in metrics:
        bearing = row["bearing_deg"]
        heading = _heading_unit(bearing)
        end = np.array([vx, vy]) + distance * heading
        if component is None:
            color = "#2f8f46" if row["moc_requirement_met"] else "#c43d3d"
        else:
            color = cmap(float(np.clip(component(row), 0.0, 1.0)))
        if show_width:
            polygon = (_corridor_polygon(vx, vy, bearing, distance) - [vx, vy]) / 1000.0
            ax.add_patch(
                Polygon(
                    polygon,
                    closed=True,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.08,
                    zorder=2,
                )
            )
        ax.plot(
            [0.0, (end[0] - vx) / 1000.0],
            [0.0, (end[1] - vy) / 1000.0],
            color=color,
            linewidth=1.3,
            alpha=0.9,
            zorder=4,
        )
        label_xy = 1.035 * (end - [vx, vy]) / 1000.0
        ax.text(label_xy[0], label_xy[1], f"S{row['sector']}", ha="center", va="center", fontsize=8)
    ax.plot(0.0, 0.0, marker="P", color="gold", markeredgecolor="black", markersize=10, zorder=10)


def _highlight_selection(ax, best, plot_data):
    if best is None:
        return
    distance = plot_data["total_distance_m"]
    for sector, color, marker, label in (
        (best["takeoff_sector"], "blue", "^", "Selected takeoff"),
        (best["landing_sector"], "green", "v", "Selected landing"),
    ):
        bearing = _sector_bearing(sector - 1)
        heading = _heading_unit(bearing)
        end = distance * heading / 1000.0
        ax.plot([0.0, end[0]], [0.0, end[1]], color=color, linewidth=4.0, label=label, zorder=8)
        ax.plot(end[0], end[1], marker=marker, color=color, markeredgecolor="black", markersize=10, zorder=9)


def _background_mesh(ax, plot_data, x_axis, y_axis, values, cmap, label):
    x_km = (x_axis - plot_data["vx"]) / 1000.0
    y_km = (y_axis - plot_data["vy"]) / 1000.0
    mesh = ax.pcolormesh(x_km, y_km, values, shading="auto", cmap=cmap, vmin=0.0, vmax=1.0, alpha=0.58)
    plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03, label=label)


def plot_sector_map(metrics, combinations, plot_data):
    best = _best_combination(combinations)
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), layout="constrained")
    ax_map, ax_rank = axes
    _background_mesh(
        ax_map,
        plot_data,
        plot_data["moc"]["x_axis"],
        plot_data["moc"]["y_axis"],
        plot_data["moc"]["display_union"],
        ListedColormap(["#e9f5e6", "#d9467b"]),
        "MOC blocked union (AGL100/200/300)",
    )
    _draw_candidates(ax_map, metrics, plot_data)
    _highlight_selection(ax_map, best, plot_data)
    _setup_local_axis(
        ax_map,
        plot_data,
        "전이 회랑 MOC 진단\n배경은 사용층 합집합, 판정은 위치·하단면 고도별",
    )
    ax_map.legend(
        handles=[
            Patch(color="#2f8f46", label="MOC PASS"),
            Patch(color="#c43d3d", label="MOC FAIL/OUT_OF_GRID"),
            Patch(color="blue", label="Selected takeoff"),
            Patch(color="green", label="Selected landing"),
        ],
        loc="upper right",
        fontsize=8,
    )

    top = combinations[:TOP_N]
    labels = [f"S{row['takeoff_sector']}/S{row['landing_sector']}" for row in top][::-1]
    values = [row["combined_risk_score"] for row in top][::-1]
    colors = ["#4472c4" if row["required_conditions_met"] else "#c43d3d" for row in top][::-1]
    ax_rank.barh(labels, values, color=colors)
    for idx, value in enumerate(values):
        ax_rank.text(value + 0.01, idx, f"{value:.3f}", va="center")
    ax_rank.set_xlim(0.0, max(1.0, max(values, default=1.0) * 1.2))
    ax_rank.set_xlabel("종합위험도 (낮을수록 유리)")
    ax_rank.set_title(f"상위 {TOP_N}개 이착륙 조합", fontweight="bold")
    ax_rank.grid(axis="x", alpha=0.25)
    fig.suptitle(
        f"전이 회랑 기반 섹터 선정 | {VERTIPORT_ALT_MSL_M:.0f}→{TARGET_ALT_MSL_M:.0f}m MSL, "
        f"{CLIMB_DESCENT_ANGLE_DEG:.1f}°, 반폭 {TRANSITION_CORRIDOR_HALF_WIDTH_M:.0f}m",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(OUTPUT_DIR / "sector_map_diagnostics.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_sector_annual_mean_wind(metrics, combinations, plot_data):
    best = _best_combination(combinations)
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), layout="constrained")
    ax_map, ax_bar = axes
    wind = plot_data["wind"]
    x_grid, y_grid = np.meshgrid(wind["x_axis"], wind["y_axis"])
    mask = np.hypot(x_grid - plot_data["vx"], y_grid - plot_data["vy"]) <= plot_data["total_distance_m"]
    stride = 5
    sample = np.zeros(mask.shape, dtype=bool)
    sample[::stride, ::stride] = True
    sample &= mask & np.isfinite(wind["display_u"]) & np.isfinite(wind["display_v"])
    speed = np.hypot(wind["display_u"][sample], wind["display_v"][sample])
    q = ax_map.quiver(
        (x_grid[sample] - plot_data["vx"]) / 1000.0,
        (y_grid[sample] - plot_data["vy"]) / 1000.0,
        wind["display_u"][sample],
        wind["display_v"][sample],
        speed,
        cmap="viridis",
        scale=80,
        width=0.003,
        zorder=3,
    )
    plt.colorbar(q, ax=ax_map, fraction=0.046, pad=0.03, label="Wind speed [m/s]")
    _draw_candidates(ax_map, metrics, plot_data, show_width=False)
    _highlight_selection(ax_map, best, plot_data)
    _setup_local_axis(
        ax_map,
        plot_data,
        f"12개월 평균 바람 벡터 ({wind['display_altitude_msl_m']:.0f}m MSL 참고층)\n점수는 전 경로 위치·고도·12개월 표본으로 계산",
    )
    sectors = np.arange(1, N_SECTORS + 1)
    width = 0.38
    ax_bar.bar(
        sectors - width / 2,
        [row["takeoff_wind_risk_score"] for row in metrics],
        width,
        color="blue",
        alpha=0.72,
        label="Takeoff wind risk",
    )
    ax_bar.bar(
        sectors + width / 2,
        [row["landing_wind_risk_score"] for row in metrics],
        width,
        color="green",
        alpha=0.72,
        label="Landing wind risk",
    )
    ax_bar.set_xticks(sectors)
    ax_bar.set_ylim(0.0, 1.05)
    ax_bar.set_xlabel("Sector")
    ax_bar.set_ylabel("Wind risk [0-1]")
    ax_bar.set_title("순풍 50% + 측풍 50% 바람위험", fontweight="bold")
    ax_bar.legend()
    ax_bar.grid(axis="y", alpha=0.25)
    fig.suptitle("전이 회랑 월별·고도별 바람 평가", fontsize=16, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "sector_annual_mean_wind.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_spatial_alignment_panels(metrics, combinations, plot_data):
    best = _best_combination(combinations)
    fig, axes = plt.subplots(2, 2, figsize=(17, 15), layout="constrained")
    ax_moc, ax_wind, ax_ground, ax_air = axes.ravel()

    _background_mesh(ax_moc, plot_data, plot_data["moc"]["x_axis"], plot_data["moc"]["y_axis"], plot_data["moc"]["display_union"], ListedColormap(["#e9f5e6", "#d9467b"]), "Blocked")
    _draw_candidates(ax_moc, metrics, plot_data)
    _highlight_selection(ax_moc, best, plot_data)
    _setup_local_axis(ax_moc, plot_data, "① MOC 사용층 합집합과 실제 회랑 판정")

    _draw_candidates(
        ax_wind,
        metrics,
        plot_data,
        component=lambda row: 0.5 * (row["takeoff_wind_risk_score"] + row["landing_wind_risk_score"]),
    )
    _highlight_selection(ax_wind, best, plot_data)
    _setup_local_axis(ax_wind, plot_data, "② 방향별 평균 바람위험 (viridis: 낮음→높음)")

    _background_mesh(ax_ground, plot_data, plot_data["wind"]["x_axis"], plot_data["wind"]["y_axis"], plot_data["ground"]["display"], "YlOrRd", "8-heading mean context")
    _draw_candidates(ax_ground, metrics, plot_data, component=lambda row: row["ground_risk_score"], show_width=False)
    _highlight_selection(ax_ground, best, plot_data)
    _setup_local_axis(ax_ground, plot_data, "③ 지상위험 배경과 heading별 회랑 점수")

    _background_mesh(ax_air, plot_data, plot_data["air"]["x_axis"], plot_data["air"]["y_axis"], plot_data["air"]["display"], "magma", "150-600m max context")
    _draw_candidates(ax_air, metrics, plot_data, component=lambda row: row["air_risk_score"], show_width=False)
    _highlight_selection(ax_air, best, plot_data)
    _setup_local_axis(ax_air, plot_data, "④ 봄·가을 공중위험과 실제 고도별 회랑 점수")

    fig.suptitle(
        "MOC·바람·지상·공중 위험 공간 정합 진단\n모든 회랑은 EPSG:5179, 반폭 100m, 전이거리 4.281km",
        fontsize=17,
        fontweight="bold",
    )
    fig.savefig(OUTPUT_DIR / "sector_map_diagnostics_panels.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def _deg_to_tile(lon, lat, zoom):
    lat = float(np.clip(lat, -85.05112878, 85.05112878))
    n = 2**zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def _tile_to_deg(x, y, zoom):
    n = 2**zoom
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lon, lat


def _download_osm_background(lon_lim, lat_lim):
    x0f, y1f = _deg_to_tile(lon_lim[0], lat_lim[0], OSM_ZOOM)
    x1f, y0f = _deg_to_tile(lon_lim[1], lat_lim[1], OSM_ZOOM)
    x0, x1 = math.floor(x0f), math.floor(x1f)
    y0, y1 = math.floor(y0f), math.floor(y1f)
    mosaic = Image.new("RGB", ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256), "white")
    try:
        for tile_y in range(y0, y1 + 1):
            for tile_x in range(x0, x1 + 1):
                request = Request(
                    f"https://tile.openstreetmap.org/{OSM_ZOOM}/{tile_x}/{tile_y}.png",
                    headers={"User-Agent": "uam-sector-evaluation/1.0"},
                )
                with urlopen(request, timeout=2.5) as response:
                    tile = Image.open(BytesIO(response.read())).convert("RGB")
                mosaic.paste(tile, ((tile_x - x0) * 256, (tile_y - y0) * 256))
    except Exception as exc:
        return None, None, str(exc)
    west, north = _tile_to_deg(x0, y0, OSM_ZOOM)
    east, south = _tile_to_deg(x1 + 1, y1 + 1, OSM_ZOOM)
    return np.asarray(mosaic), [west, east, south, north], None


def plot_osm_overview(metrics, combinations, plot_data):
    best = _best_combination(combinations)
    to_wgs84 = pyproj.Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)
    pad = plot_data["total_distance_m"] * 1.08
    corner_x = np.array([plot_data["vx"] - pad, plot_data["vx"] + pad] * 2)
    corner_y = np.array([plot_data["vy"] - pad] * 2 + [plot_data["vy"] + pad] * 2)
    corner_lon, corner_lat = to_wgs84.transform(corner_x, corner_y)
    lon_lim = [float(np.min(corner_lon)), float(np.max(corner_lon))]
    lat_lim = [float(np.min(corner_lat)), float(np.max(corner_lat))]
    background, extent, error = _download_osm_background(lon_lim, lat_lim)

    fig, ax = plt.subplots(figsize=(11, 10), layout="constrained")
    if background is not None:
        ax.imshow(background, extent=extent, origin="upper")
        background_text = "OpenStreetMap background"
    else:
        ax.set_facecolor("#eeeeee")
        background_text = "OSM 배경 사용 불가 - 동일 위경도 좌표 배경으로 대체"
    for row in metrics:
        bearing = row["bearing_deg"]
        heading = _heading_unit(bearing)
        end_x = plot_data["vx"] + plot_data["total_distance_m"] * heading[0]
        end_y = plot_data["vy"] + plot_data["total_distance_m"] * heading[1]
        lons, lats = to_wgs84.transform(
            [plot_data["vx"], end_x], [plot_data["vy"], end_y]
        )
        color = "#2f8f46" if row["moc_requirement_met"] else "#c43d3d"
        ax.plot(lons, lats, color=color, linewidth=1.5, alpha=0.8)
        ax.text(lons[-1], lats[-1], f"S{row['sector']}", fontsize=8, ha="center")
    if best is not None:
        for sector, color, marker, label in (
            (best["takeoff_sector"], "blue", "^", "Selected takeoff"),
            (best["landing_sector"], "green", "v", "Selected landing"),
        ):
            heading = _heading_unit(_sector_bearing(sector - 1))
            end_x = plot_data["vx"] + plot_data["total_distance_m"] * heading[0]
            end_y = plot_data["vy"] + plot_data["total_distance_m"] * heading[1]
            lons, lats = to_wgs84.transform(
                [plot_data["vx"], end_x], [plot_data["vy"], end_y]
            )
            ax.plot(lons, lats, color=color, linewidth=4.0, label=label)
            ax.plot(lons[-1], lats[-1], marker=marker, color=color, markeredgecolor="black", markersize=10)
    ax.plot(VERTIPORT_LON, VERTIPORT_LAT, marker="P", color="gold", markeredgecolor="black", markersize=11, label="Vertiport")
    ax.set_xlim(lon_lim)
    ax.set_ylim(lat_lim)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(np.mean(lat_lim))), adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")
    ax.set_title(
        "전이 회랑 섹터 선정 지도\n"
        + background_text,
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(OUTPUT_DIR / "sector_map_osm_overview.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_dashboard(metrics, combinations, plot_data):
    best = _best_combination(combinations)
    fig = plt.figure(figsize=(18, 16), layout="constrained")
    grid = fig.add_gridspec(2, 2)
    ax_components = fig.add_subplot(grid[0, 0])
    ax_moc = fig.add_subplot(grid[0, 1])
    ax_matrix = fig.add_subplot(grid[1, 0])
    ax_table = fig.add_subplot(grid[1, 1])

    sectors = np.arange(1, N_SECTORS + 1)
    width = 0.25
    mean_wind = [0.5 * (row["takeoff_wind_risk_score"] + row["landing_wind_risk_score"]) for row in metrics]
    ax_components.bar(sectors - width, mean_wind, width, label="Wind", color="#4e79a7")
    ax_components.bar(sectors, [row["ground_risk_score"] for row in metrics], width, label="Ground", color="#f28e2b")
    ax_components.bar(sectors + width, [row["air_risk_score"] for row in metrics], width, label="Air", color="#e15759")
    ax_components.set_xticks(sectors)
    ax_components.set_ylim(0.0, 1.05)
    ax_components.set_title("섹터별 3개 선정 점수 (이륙·착륙 평균)", fontweight="bold")
    ax_components.set_ylabel("Normalized risk [0-1]")
    ax_components.legend()
    ax_components.grid(axis="y", alpha=0.25)

    colors = ["#2f8f46" if row["moc_requirement_met"] else "#c43d3d" for row in metrics]
    ax_moc.bar(sectors, [row["moc_safety_score"] for row in metrics], color=colors)
    for row in metrics:
        ax_moc.text(row["sector"], row["moc_safety_score"] + 0.025, "PASS" if row["moc_requirement_met"] else "FAIL", ha="center", fontsize=7)
    ax_moc.set_xticks(sectors)
    ax_moc.set_ylim(0.0, 1.12)
    ax_moc.set_title("MOC 필수조건", fontweight="bold")
    ax_moc.set_ylabel("MOC safety score")
    ax_moc.grid(axis="y", alpha=0.25)

    matrix = np.full((N_SECTORS, N_SECTORS), np.nan)
    feasible = np.zeros((N_SECTORS, N_SECTORS), dtype=bool)
    for row in combinations:
        t = row["takeoff_sector"] - 1
        l = row["landing_sector"] - 1
        matrix[l, t] = row["combined_risk_score"]
        feasible[l, t] = row["required_conditions_met"]
    image = ax_matrix.imshow(matrix, origin="lower", cmap="viridis_r", vmin=0.0, vmax=1.0)
    for landing in range(N_SECTORS):
        for takeoff in range(N_SECTORS):
            if np.isfinite(matrix[landing, takeoff]):
                text_value = f"{matrix[landing, takeoff]:.2f}" if feasible[landing, takeoff] else "×"
                ax_matrix.text(takeoff, landing, text_value, ha="center", va="center", fontsize=6.5, color="black" if matrix[landing, takeoff] < 0.55 else "white")
    ax_matrix.set_xticks(np.arange(N_SECTORS), [f"S{i}" for i in sectors])
    ax_matrix.set_yticks(np.arange(N_SECTORS), [f"S{i}" for i in sectors])
    ax_matrix.set_xlabel("Takeoff sector")
    ax_matrix.set_ylabel("Landing sector")
    ax_matrix.set_title("조합 종합위험도 | × = MOC 불통과", fontweight="bold")
    fig.colorbar(image, ax=ax_matrix, fraction=0.046, pad=0.03)

    ax_table.axis("off")
    top = combinations[:TOP_N]
    table_data = [
        [
            row["comparison_rank"],
            f"S{row['takeoff_sector']}",
            f"S{row['landing_sector']}",
            "PASS" if row["required_conditions_met"] else "FAIL",
            f"{row['wind_risk_score']:.3f}",
            f"{row['ground_risk_score']:.3f}",
            f"{row['air_risk_score']:.3f}",
            f"{row['combined_risk_score']:.3f}",
        ]
        for row in top
    ]
    table = ax_table.table(
        cellText=table_data,
        colLabels=["Rank", "TO", "LD", "MOC", "Wind", "Ground", "Air", "Total"],
        cellLoc="center",
        loc="upper center",
        bbox=[0.0, 0.48, 1.0, 0.46],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    selection_text = (
        "선정 가능한 조합 없음"
        if best is None
        else (
            f"최종 선정: 이륙 S{best['takeoff_sector']} / 착륙 S{best['landing_sector']}\n"
            f"종합 {best['combined_risk_score']:.4f}\n"
            f"바람 {best['wind_risk_score']:.4f} | 지상 {best['ground_risk_score']:.4f} | 공중 {best['air_risk_score']:.4f}"
        )
    )
    ax_table.text(
        0.02,
        0.36,
        selection_text
        + "\n\n선정 절차\n1. 전이 회랑 전체 MOC PASS\n2. 바람·지상·공중 위험 1:1:1 최소\n3. 데이터 coverage는 별도 진단",
        va="top",
        fontsize=11,
        linespacing=1.45,
        bbox={"facecolor": "#f5f5f5", "edgecolor": "#777777", "boxstyle": "round,pad=0.5"},
    )
    ax_table.set_title(f"상위 {TOP_N}개 조합", fontweight="bold")
    fig.suptitle("전이 회랑 기반 섹터 선정 대시보드", fontsize=18, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "sector_evaluation_dashboard.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_wind_scoring_presentation(metrics, combinations, plot_data):
    best = _best_combination(combinations)
    fig, axes = plt.subplots(2, 2, figsize=(17, 13), layout="constrained")
    ax_month, ax_sector, ax_scatter, ax_text = axes.ravel()
    monthly = plot_data["monthly_rows"]
    if best is not None:
        selections = [
            (best["takeoff_sector"], "takeoff", "blue", "Selected takeoff"),
            (best["landing_sector"], "landing", "green", "Selected landing"),
        ]
        for sector, direction, color, label in selections:
            rows = [row for row in monthly if row["Sector"] == sector and row["Direction"] == direction]
            months = [row["Month"] for row in rows]
            ax_month.plot(months, [row["Mean_Tailwind_mps"] for row in rows], color=color, marker="o", label=f"{label} tail")
            ax_month.plot(months, [row["Mean_Crosswind_mps"] for row in rows], color=color, marker="s", linestyle="--", label=f"{label} cross")
    ax_month.set_xticks(np.arange(1, 13))
    ax_month.set_xlabel("Month")
    ax_month.set_ylabel("Mean component [m/s]")
    ax_month.set_title("선정 조합의 월별 순풍·측풍", fontweight="bold")
    ax_month.grid(True, alpha=0.25)
    ax_month.legend(fontsize=8)

    sectors = np.arange(1, N_SECTORS + 1)
    ax_sector.plot(sectors, [row["takeoff_wind_risk_score"] for row in metrics], color="blue", marker="^", label="Takeoff")
    ax_sector.plot(sectors, [row["landing_wind_risk_score"] for row in metrics], color="green", marker="v", label="Landing")
    ax_sector.set_xticks(sectors)
    ax_sector.set_ylim(0.0, 1.05)
    ax_sector.set_xlabel("Sector")
    ax_sector.set_ylabel("Wind risk [0-1]")
    ax_sector.set_title("섹터별 바람위험", fontweight="bold")
    ax_sector.grid(True, alpha=0.25)
    ax_sector.legend()

    for row in plot_data["directions"]:
        color = "blue" if row["direction"] == "takeoff" else "green"
        marker = "^" if row["direction"] == "takeoff" else "v"
        ax_scatter.scatter(row["tailwind_mps_raw"], row["crosswind_mps_raw"], color=color, marker=marker, alpha=0.75)
        ax_scatter.text(row["tailwind_mps_raw"], row["crosswind_mps_raw"], f"S{row['sector']}", fontsize=7)
    ax_scatter.set_xlabel("12-month mean tailwind [m/s]")
    ax_scatter.set_ylabel("12-month mean crosswind [m/s]")
    ax_scatter.set_title("원자료 성분 분포", fontweight="bold")
    ax_scatter.grid(True, alpha=0.25)

    ax_text.axis("off")
    ax_text.text(
        0.02,
        0.98,
        "바람위험 계산\n\n"
        "1. 각 회랑 표본에서 실제 MSL의 월별 U/V를 보간\n"
        "2. 비행 heading에 투영해 순풍·역풍·절대 측풍 계산\n"
        "3. 12개월과 회랑 전체 표본의 평균 계산\n"
        "4. 24개 방향 후보에서 순풍·측풍을 각각 0~1 정규화\n"
        "5. 바람위험 = 0.5×순풍위험 + 0.5×측풍위험\n\n"
        "역풍은 감사자료에 기록하지만 벌점으로 사용하지 않습니다.\n"
        "바람 임계값은 없으며 MOC만 필수 제약입니다.",
        va="top",
        fontsize=11,
        linespacing=1.55,
        bbox={"facecolor": "#fff4cc", "edgecolor": "#c28b00", "boxstyle": "round,pad=0.6"},
    )
    fig.suptitle("월별·고도별 바람위험 계산 방법", fontsize=18, fontweight="bold")
    fig.savefig(OUTPUT_DIR / "wind_scoring_method_presentation.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def print_console_summary(metrics, combinations, plot_data):
    best = _best_combination(combinations)
    eligible = sum(row["required_conditions_met"] for row in combinations)
    print(
        f"transition horizontal distance = {plot_data['total_distance_m']:.3f} m, "
        f"corridor half width = {TRANSITION_CORRIDOR_HALF_WIDTH_M:.1f} m"
    )
    print(f"sector rows = {len(metrics)}, combinations = {len(combinations)}, eligible = {eligible}")
    if best is None:
        print("selected combination = NONE")
    else:
        print(
            f"selected takeoff S{best['takeoff_sector']} / landing S{best['landing_sector']} "
            f"combined risk = {best['combined_risk_score']:.6f}"
        )
    print(f"Saved outputs to: {OUTPUT_DIR}")


def main():
    metrics, plot_data = evaluate_sectors()
    combinations = build_combination_ranking(metrics)
    validate_results(metrics, combinations, plot_data)
    write_outputs(metrics, combinations, plot_data)
    plot_sector_map(metrics, combinations, plot_data)
    plot_sector_annual_mean_wind(metrics, combinations, plot_data)
    plot_spatial_alignment_panels(metrics, combinations, plot_data)
    plot_osm_overview(metrics, combinations, plot_data)
    plot_dashboard(metrics, combinations, plot_data)
    plot_wind_scoring_presentation(metrics, combinations, plot_data)
    print_console_summary(metrics, combinations, plot_data)


if __name__ == "__main__":
    main()
