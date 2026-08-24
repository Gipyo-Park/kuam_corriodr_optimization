import sys
import os
import csv
import json
from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd

import pickle
from tqdm import tqdm

import matplotlib
_CLICK_MODE_ENV = os.environ.get("WP_CLICK_MODE", "1").strip().lower()
USE_INTERACTIVE_BACKEND = _CLICK_MODE_ENV in ("1", "true", "yes", "on")
# Keep the main optimization/render pipeline on Agg to avoid Tk shutdown issues.
matplotlib.use("Agg")
if USE_INTERACTIVE_BACKEND:
    try:
        import tkinter  # noqa: F401
    except Exception:
        USE_INTERACTIVE_BACKEND = False
import matplotlib.pyplot as plt
from matplotlib._pylab_helpers import Gcf
from matplotlib.patches import Patch
from scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from pyproj import Transformer
from shapely.geometry import LineString
from shapely.ops import transform as shapely_transform

from crossover_GP import crossover_gp
from mutation_GP import mutation_gp
from fast_non_dominated_sort import fast_non_dominated_sort
from generate_initial_population_GP import generate_initial_population_gp
from evaluate_objectives_with_constraints_GP import (
    evaluate_objectives_with_constraints_gp as _evaluate_constraints_shared,
    _corridor_violates_nfz_with_width as _corridor_violates_nfz_with_width_shared,
    _segment_to_segment_min_distance_m as _segment_to_segment_min_distance_m_shared,
)
from rf_turn import apply_rf_turns


TAKEOFF_TRANSITION_PROFILE = None
LANDING_TRANSITION_PROFILE_DESC = None
TRANSITION_CONTEXT = None
RF_ALLOW_TANGENT_CLAMP = True
RF_CORNER_FIT_MARGIN = 0.95
RF_CORNER_MIN_TANGENT_M = 1.0
RF_MIN_TURN_ANGLE_DEG = 0.5

TAKEOFF_TRANSITION_COLOR = "blue"
LANDING_TRANSITION_COLOR = "green"
FLIGHT_PHASE_VERTIPORT = "vertiport"
FLIGHT_PHASE_TAKEOFF_STAGE1 = "takeoff_stage1"
FLIGHT_PHASE_TAKEOFF_STAGE2 = "takeoff_stage2"
FLIGHT_PHASE_CRUISE = "cruise"
FLIGHT_PHASE_LANDING_STAGE2 = "landing_stage2"
FLIGHT_PHASE_LANDING_STAGE1 = "landing_stage1"
TRANSITION_STRUCTURE_FIXED_ONLY = "fixed_straight_only"
TRANSITION_STRUCTURE_FIXED_PLUS_OPTIMIZED = "fixed_straight_plus_optimized"
TRANSITION_STRUCTURE_OPTIMIZED_ONLY = "optimized_only"
TRANSITION_STRUCTURE_MODES = (
    TRANSITION_STRUCTURE_FIXED_ONLY,
    TRANSITION_STRUCTURE_FIXED_PLUS_OPTIMIZED,
    TRANSITION_STRUCTURE_OPTIMIZED_ONLY,
)


class SectorSelectionInfeasibleError(ValueError):
    """No MOC-safe automatic takeoff/landing sector pair can be selected."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = list(details or [])


MOC_REFERENCE_MSL_M = 150.0
MOC_AGL_LEVELS_M = np.arange(100.0, 1000.0, 100.0, dtype=float)
TRANSITION_PHASE_NAMES = (
    FLIGHT_PHASE_TAKEOFF_STAGE1,
    FLIGHT_PHASE_TAKEOFF_STAGE2,
    FLIGHT_PHASE_LANDING_STAGE2,
    FLIGHT_PHASE_LANDING_STAGE1,
)

_CORRIDOR_TO_EPSG5179 = Transformer.from_crs(
    "EPSG:4326", "EPSG:5179", always_xy=True
)
_CORRIDOR_FROM_EPSG5179 = Transformer.from_crs(
    "EPSG:5179", "EPSG:4326", always_xy=True
)


def _title_with_altitude(title, altitude_levels, vertiport):
    altitude_msl = float(np.asarray(altitude_levels, dtype=float).ravel()[0])
    altitude_agl = altitude_msl - float(np.asarray(vertiport, dtype=float).ravel()[2])
    return (
        f"{title}\n"
        f"Altitude: {altitude_msl:.1f}m MSL "
        f"(= {altitude_agl:.1f}m AGL above vertiport)"
    )


def cleanup_matplotlib_tk():
    """Best-effort cleanup for TkAgg resources to avoid shutdown-time Tk errors."""
    try:
        for manager in list(Gcf.get_all_fig_managers()):
            try:
                manager.destroy()
            except Exception:
                pass
    except Exception:
        pass

    try:
        plt.close("all")
    except Exception:
        pass

    # Explicitly destroy Tk default root if it exists.
    if USE_INTERACTIVE_BACKEND:
        try:
            import tkinter as tk
            root = tk._default_root
            if root is not None:
                try:
                    root.update_idletasks()
                except Exception:
                    pass
                try:
                    root.destroy()
                except Exception:
                    pass
                tk._default_root = None
        except Exception:
            pass

def filter_nodes_in_strip(a, b, cand, W_strip_m, end_buffer_ratio=-0.3):
    if cand is None or cand.size == 0:
        return cand
    mean_lat_rad = np.deg2rad(0.5 * (a[0] + b[0]))
    m_per_lat = 111000.0
    m_per_lon = 111000.0 * np.cos(mean_lat_rad)
    ax, ay = a[1] * m_per_lon, a[0] * m_per_lat
    bx, by = b[1] * m_per_lon, b[0] * m_per_lat
    vx, vy = bx - ax, by - ay
    vv = vx * vx + vy * vy
    if vv < 1e-9:
        return np.empty((0, 3))
    cx = cand[:, 1] * m_per_lon
    cy = cand[:, 0] * m_per_lat
    wx, wy = cx - ax, cy - ay
    t = (wx * vx + wy * vy) / vv
    t_min = -end_buffer_ratio
    t_max = 1.0 + end_buffer_ratio
    t_clip = np.clip(t, 0.0, 1.0)
    px = ax + t_clip * vx
    py = ay + t_clip * vy
    dx = cx - px
    dy = cy - py
    dist = np.sqrt(dx * dx + dy * dy)
    mask = (t >= t_min) & (t <= t_max) & (dist <= W_strip_m)
    return cand[mask]


def _segment_strip_end_buffer_ratio(seg_idx, seg_count, boundary_ratio=-0.1, interior_ratio=-0.3):
    if seg_count <= 1:
        return float(boundary_ratio)
    if seg_idx == 0 or seg_idx == seg_count - 1:
        return float(boundary_ratio)
    return float(interior_ratio)


def generate_nodes_3d_segment(p1, p2, W_buf, resolution_m, lat_lim, lon_lim,
                              Ny, Nx, forbidden_zones, altitude_levels=None):
    minLat, maxLat = lat_lim
    minLon, maxLon = lon_lim
    dLat_deg = (maxLat - minLat) / (Ny - 1)
    dLon_deg = (maxLon - minLon) / (Nx - 1)
    j1, i1 = (p1[0] - minLat) / dLat_deg, (p1[1] - minLon) / dLon_deg
    j2, i2 = (p2[0] - minLat) / dLat_deg, (p2[1] - minLon) / dLon_deg
    p1g = np.array([i1, j1], dtype=float)
    p2g = np.array([i2, j2], dtype=float)
    vec = p2g - p1g
    length = float(np.linalg.norm(vec))
    if length < 1e-9:
        return np.empty((0, 3)), np.empty((0, 3))
    u = vec / length
    v = np.array([-u[1], u[0]], dtype=float)
    mean_lat_rad = np.deg2rad(float(np.mean([p1[0], p2[0]])))
    m_lon = 111000.0 * np.cos(mean_lat_rad)
    m_lat = 111000.0
    mpu = float(np.sqrt((u[0] * dLon_deg * m_lon) ** 2 + (u[1] * dLat_deg * m_lat) ** 2))
    mpv = float(np.sqrt((v[0] * dLon_deg * m_lon) ** 2 + (v[1] * dLat_deg * m_lat) ** 2))
    len_m = length * mpu
    n_s = max(2, int(round(len_m / resolution_m)) + 1)
    s_m = np.linspace(0.0, len_m, n_s)
    n_t = max(3, int(round(2.0 * W_buf / resolution_m)) + 1)
    t_m = np.linspace(-W_buf, W_buf, n_t)
    S, T = np.meshgrid(s_m, t_m)
    Si = S / mpu
    Ti = T / mpv
    I = p1g[0] + Si * u[0] + Ti * v[0]
    J = p1g[1] + Si * u[1] + Ti * v[1]
    ok = (I >= 0) & (I < Nx) & (J >= 0) & (J < Ny)
    Ii, Ji = I[ok].ravel(), J[ok].ravel()
    lons = minLon + Ii * dLon_deg
    lats = minLat + Ji * dLat_deg

    if altitude_levels is None or len(np.atleast_1d(altitude_levels)) == 0:
        alts = np.full_like(lats, p1[2], dtype=float)
        all_grid = np.column_stack([lats, lons, alts])
    else:
        altitude_levels = np.asarray(altitude_levels, dtype=float).ravel()
        all_grid = np.vstack([
            np.column_stack([lats, lons, np.full_like(lats, alt, dtype=float)])
            for alt in altitude_levels
        ])
    nodes = all_grid
    if forbidden_zones is not None and forbidden_zones.size > 0 and all_grid.size > 0:
        mask = np.ones(all_grid.shape[0], dtype=bool)
        for rect in forbidden_zones:
            mn_lon, mx_lon, mn_lat, mx_lat = rect
            mask &= ~((lons >= mn_lon) & (lons <= mx_lon) & (lats >= mn_lat) & (lats <= mx_lat))
        nodes = all_grid[mask]
    return nodes, all_grid


def _horiz_dist(angle_deg, alt_m):
    a = np.clip(np.deg2rad(angle_deg), 1e-6, np.pi / 2 - 1e-6)
    return float(alt_m / np.tan(a))

def _sector_angle(sector_1, n=12):
    i = int(sector_1) - 1
    w = 2.0 * np.pi / n
    return np.deg2rad(90.0) - (i + 0.5) * w

def _move_latlon(lat0, lon0, heading_rad, dist_m):
    m_lat = 111000.0
    m_lon = 111000.0 * np.cos(np.deg2rad(lat0))
    return float(lat0 + dist_m * np.sin(heading_rad) / m_lat), \
           float(lon0 + dist_m * np.cos(heading_rad) / m_lon)


def _build_sector_wedge_lonlat(vertiport_lla, center_heading_deg, half_width_deg, radius_m, n_pts=24):
    lat0, lon0 = float(vertiport_lla[0]), float(vertiport_lla[1])
    r = max(1.0, float(radius_m))
    half = max(0.5, float(half_width_deg))
    h0 = float(center_heading_deg) - half
    h1 = float(center_heading_deg) + half
    hs = np.linspace(h0, h1, int(max(6, n_pts)))
    poly_lon = [lon0]
    poly_lat = [lat0]
    for h in hs:
        heading_rad = np.deg2rad(h)
        lat_i, lon_i = _move_latlon(lat0, lon0, heading_rad, r)
        poly_lon.append(float(lon_i))
        poly_lat.append(float(lat_i))
    poly_lon.append(lon0)
    poly_lat.append(lat0)
    return np.asarray(poly_lon, dtype=float), np.asarray(poly_lat, dtype=float)


def _heading_deg_from_segment(p0, p1):
    p0 = np.asarray(p0, dtype=float).reshape(3)
    p1 = np.asarray(p1, dtype=float).reshape(3)
    mean_lat = float(0.5 * (p0[0] + p1[0]))
    m_lat = 111000.0
    m_lon = 111000.0 * np.cos(np.deg2rad(mean_lat))
    d_north = (p1[0] - p0[0]) * m_lat
    d_east = (p1[1] - p0[1]) * m_lon
    if abs(d_north) < 1e-9 and abs(d_east) < 1e-9:
        return None
    return float((np.degrees(np.arctan2(d_east, d_north)) + 360.0) % 360.0)

def build_takeoff_landing(vertiport, angle_deg=25.0, alt_delta_m=350.0,
                          takeoff_sector=11, landing_sector=6,
                          takeoff_target_alt_m=None, landing_target_alt_m=None):
    lat0, lon0, alt0 = float(vertiport[0]), float(vertiport[1]), float(vertiport[2])
    d = _horiz_dist(angle_deg, alt_delta_m)
    to_lat, to_lon = _move_latlon(lat0, lon0, _sector_angle(takeoff_sector), d)
    ld_lat, ld_lon = _move_latlon(lat0, lon0, _sector_angle(landing_sector), d)
    to_alt = float(alt0 if takeoff_target_alt_m is None else takeoff_target_alt_m)
    ld_alt = float(alt0 if landing_target_alt_m is None else landing_target_alt_m)
    return np.array([to_lat, to_lon, to_alt]), np.array([ld_lat, ld_lon, ld_alt]), d


def build_transition_point(port_lla, angle_deg=25.0, alt_delta_m=350.0,
                           sector=11, target_alt_m=None):
    """Build one transition point from one vertiport-like LLA point."""
    lat0, lon0, alt0 = float(port_lla[0]), float(port_lla[1]), float(port_lla[2])
    d = _horiz_dist(angle_deg, alt_delta_m)
    lat, lon = _move_latlon(lat0, lon0, _sector_angle(sector), d)
    alt = float(alt0 if target_alt_m is None else target_alt_m)
    return np.array([lat, lon, alt]), d


def _calculate_transition_geometry(
    height_m,
    transition_mode,
    distance_m=None,
    angle_deg=None,
):
    """
    전이 경로 삼각형의 밑변과 경사각을 선택한 모드에 따라 계산한다.

    - distance 모드: 사용자가 밑변을 입력하고 경사각을 자동 계산한다.
    - angle 모드: 사용자가 경사각을 입력하고 밑변을 자동 계산한다.
    - height_m: 순항 고도와 버티포트 고도의 차이(삼각형 높이)
    """
    mode = str(transition_mode).strip().lower()
    height = float(abs(height_m))

    if mode == "distance":
        if distance_m is None:
            raise ValueError("distance mode requires distance_m.")
        distance = float(distance_m)
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError("distance mode requires distance_m > 0.")
        angle = float(np.rad2deg(np.arctan2(height, distance)))
    elif mode == "angle":
        if angle_deg is None:
            raise ValueError("angle mode requires angle_deg.")
        angle = float(angle_deg)
        if not np.isfinite(angle) or not 0.0 < angle < 90.0:
            raise ValueError("angle mode requires 0 < angle_deg < 90.")
        distance = float(height / np.tan(np.deg2rad(angle)))
    else:
        raise ValueError(
            f"transition_mode must be 'distance' or 'angle', got {transition_mode!r}."
        )

    return {
        "mode": mode,
        "height_m": height,
        "distance_m": distance,
        "angle_deg": angle,
    }


def build_transition_profile_linear(
    start_lla,
    end_lla,
    sample_spacing_m=50.0,
):
    """Build linear 3D transition samples (lat/lon/alt all vary linearly)."""
    s = np.asarray(start_lla, dtype=float).ravel()
    e = np.asarray(end_lla, dtype=float).ravel()
    if s.size != 3 or e.size != 3:
        raise ValueError("start_lla and end_lla must be [lat, lon, alt].")

    dist_h = _seg_dist_m(s, e)
    ds = float(sample_spacing_m)
    if ds <= 0.0:
        raise ValueError("sample_spacing_m must be > 0.")
    n_pts = int(max(2, np.ceil(max(dist_h, 1e-9) / ds) + 1))
    t = np.linspace(0.0, 1.0, n_pts)
    prof = (s[None, :] * (1.0 - t[:, None]) + e[None, :] * t[:, None]).astype(float)
    return prof


def build_transition_profile_by_mode(
    start_lla,
    target_alt_m,
    heading_deg=None,
    sector=None,
    transition_mode="distance",
    distance_m=None,
    angle_deg=None,
    sample_spacing_m=50.0,
    mode_label="transition",
):
    """Build a linear 3D transition profile using distance or angle mode."""
    start = np.asarray(start_lla, dtype=float).ravel()
    if start.size != 3:
        raise ValueError("start_lla must be [lat, lon, alt].")
    target_altitude_m = float(target_alt_m)
    geometry = _calculate_transition_geometry(
        height_m=target_altitude_m - start[2],
        transition_mode=transition_mode,
        distance_m=distance_m,
        angle_deg=angle_deg,
    )

    if heading_deg is None:
        if sector is None:
            raise ValueError("Either heading_deg or sector must be provided.")
        heading_rad = _sector_angle(sector)
        heading_deg_used = float(np.rad2deg(heading_rad))
    else:
        heading_deg_used = float(heading_deg)
        heading_rad = np.deg2rad(heading_deg_used)

    path_distance_m = float(geometry["distance_m"])
    end_lat, end_lon = _move_latlon(
        float(start[0]),
        float(start[1]),
        heading_rad,
        path_distance_m,
    )
    end_lla = np.array([end_lat, end_lon, target_altitude_m], dtype=float)
    profile = build_transition_profile_linear(
        start,
        end_lla,
        sample_spacing_m=sample_spacing_m,
    )

    print(
        f"[{mode_label}] mode={geometry['mode']} | "
        f"height={geometry['height_m']:.1f}m | "
        f"distance={geometry['distance_m']:.1f}m | "
        f"angle={geometry['angle_deg']:.2f}deg | "
        f"heading={heading_deg_used:.2f}deg | "
        f"samples={profile.shape[0]} | "
        f"sample_spacing={float(sample_spacing_m):.1f}m"
    )

    geometry.update({
        "heading_deg": float(heading_deg_used),
        "sample_spacing_m": float(sample_spacing_m),
        "end_lla": end_lla.astype(float),
    })
    return end_lla.astype(float), path_distance_m, profile.astype(float), geometry


def _validate_transition_angle(angle_deg, label):
    try:
        angle = float(angle_deg)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} must satisfy 0 < angle < 90 degrees."
        ) from exc
    if not np.isfinite(angle) or not 0.0 < angle < 90.0:
        raise ValueError(f"{label} must satisfy 0 < angle < 90 degrees.")
    return angle


def build_stage1_transition_profile(
    port_lla,
    target_alt_m,
    heading_deg,
    transition_structure_mode,
    transition_mode,
    straight_distance_m,
    total_transition_horizontal_distance_m,
    angle_deg,
    sample_spacing_m,
    mode_label,
):
    """Build one direction's fixed prefix and complete transition geometry."""
    port = np.asarray(port_lla, dtype=float).reshape(3)
    target_alt = float(target_alt_m)
    structure_mode = str(transition_structure_mode).strip().lower()
    if structure_mode not in TRANSITION_STRUCTURE_MODES:
        raise ValueError(
            "transition_structure_mode must be one of "
            f"{TRANSITION_STRUCTURE_MODES}, got {transition_structure_mode!r}."
        )
    heading = float(heading_deg)
    if not np.isfinite(heading):
        raise ValueError(f"{mode_label}_heading_deg must be finite.")
    height = target_alt - float(port[2])
    if height < -1e-9:
        raise ValueError(f"{mode_label} target altitude must not be below the vertiport.")

    geometry_mode = str(transition_mode).strip().lower()
    configured_angle = angle_deg
    configured_total_distance = total_transition_horizontal_distance_m
    configured_stage1 = straight_distance_m
    if height <= 1e-9:
        angle = 0.0
        total_distance = 0.0
        requested_stage1 = 0.0
        if structure_mode == TRANSITION_STRUCTURE_OPTIMIZED_ONLY:
            geometry_mode = "ignored"
    elif structure_mode == TRANSITION_STRUCTURE_OPTIMIZED_ONLY:
        angle = _validate_transition_angle(angle_deg, f"{mode_label}_angle_deg")
        total_distance = float(height / np.tan(np.deg2rad(angle)))
        requested_stage1 = 0.0
        geometry_mode = "ignored"
    else:
        geometry = _calculate_transition_geometry(
            height_m=height,
            transition_mode=geometry_mode,
            distance_m=total_transition_horizontal_distance_m,
            angle_deg=angle_deg,
        )
        angle = float(geometry["angle_deg"])
        total_distance = float(geometry["distance_m"])
        if structure_mode == TRANSITION_STRUCTURE_FIXED_ONLY:
            requested_stage1 = total_distance
        else:
            configured_stage1 = float(straight_distance_m)
            if not np.isfinite(configured_stage1) or configured_stage1 < 0.0:
                raise ValueError(
                    f"{mode_label}_stage1_straight_distance_m must be finite and >= 0."
                )
            requested_stage1 = configured_stage1

    stage1_distance = (
        total_distance
        if structure_mode == TRANSITION_STRUCTURE_FIXED_ONLY
        else min(float(requested_stage1), total_distance)
    )
    stage1_clamped = bool(
        structure_mode == TRANSITION_STRUCTURE_FIXED_PLUS_OPTIMIZED
        and float(requested_stage1) > total_distance
    )
    stage2_distance = max(0.0, total_distance - stage1_distance)
    stage2_actual = bool(
        structure_mode != TRANSITION_STRUCTURE_FIXED_ONLY
        and stage2_distance > 0.0
    )
    stage1_alt = float(port[2] + stage1_distance * np.tan(np.deg2rad(angle)))
    if (
        structure_mode != TRANSITION_STRUCTURE_OPTIMIZED_ONLY
        and stage2_distance == 0.0
    ):
        stage1_alt = target_alt

    heading_rad = np.deg2rad(heading)
    end_lat, end_lon = _move_latlon(
        float(port[0]),
        float(port[1]),
        heading_rad,
        stage1_distance,
    )
    stage1_end = np.array([end_lat, end_lon, stage1_alt], dtype=float)
    if stage1_distance <= 0.0:
        profile = port.reshape(1, 3).copy()
    else:
        profile = build_transition_profile_linear(
            port,
            stage1_end,
            sample_spacing_m=sample_spacing_m,
        )
        if profile.shape[0] == 2:
            profile = np.vstack([
                profile[0],
                0.5 * (profile[0] + profile[1]),
                profile[1],
            ]).astype(float)

    return stage1_end, profile, {
        "mode": f"{structure_mode}_{geometry_mode}",
        "transition_structure_mode": structure_mode,
        "transition_mode": geometry_mode,
        "angle_deg": angle,
        "actual_angle_deg": angle,
        "configured_angle_deg": (
            (
                None if configured_angle is None else float(configured_angle)
            )
            if (
                height > 1e-9
                and (
                    structure_mode == TRANSITION_STRUCTURE_OPTIMIZED_ONLY
                    or geometry_mode == "angle"
                )
            )
            else configured_angle
        ),
        "altitude_profile_input": "angle",
        "height_m": float(height),
        "distance_m": total_distance,
        "total_horizontal_distance_m": total_distance,
        "configured_total_horizontal_distance_m": (
            (
                None
                if configured_total_distance is None
                else float(configured_total_distance)
            )
            if height > 1e-9 and geometry_mode == "distance"
            else configured_total_distance
        ),
        "stage1_requested_straight_distance_m": float(requested_stage1),
        "stage1_straight_distance_m": stage1_distance,
        "stage1_effective_straight_distance_m": stage1_distance,
        "stage1_clamped_to_cruise": stage1_clamped,
        "stage2_horizontal_distance_m": stage2_distance,
        "optimized_transition_actual": stage2_actual,
        "stage2_collapsed_at_cruise": bool(
            structure_mode == TRANSITION_STRUCTURE_FIXED_PLUS_OPTIMIZED
            and stage2_distance == 0.0
        ),
        "fixed_straight_actual": bool(stage1_distance > 0.0),
        "stage1_end_lla": stage1_end.copy(),
        "heading_deg": heading,
        "sample_spacing_m": float(sample_spacing_m),
    }


def _angular_difference_deg(a_deg, b_deg):
    return float(abs((float(a_deg) - float(b_deg) + 180.0) % 360.0 - 180.0))


def _polyline_cumulative_horizontal_m(points):
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    out = np.zeros(pts.shape[0], dtype=float)
    for i in range(1, pts.shape[0]):
        out[i] = out[i - 1] + _seg_dist_m(pts[i - 1], pts[i])
    return out


def _insert_polyline_distances(points, requested_local_distances_m):
    """Insert exact along-track points into a polyline and return local distances."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if pts.shape[0] <= 1:
        return pts.copy(), np.zeros(pts.shape[0], dtype=float)

    cumulative = _polyline_cumulative_horizontal_m(pts)
    total = float(cumulative[-1])
    requested = sorted({
        float(np.clip(v, 0.0, total))
        for v in requested_local_distances_m
        if np.isfinite(v) and -1e-6 <= float(v) <= total + 1e-6
    })
    target_distances = list(cumulative.astype(float))
    for requested_s in requested:
        if not target_distances:
            target_distances.append(requested_s)
            continue
        nearest_idx = int(np.argmin(np.abs(np.asarray(target_distances) - requested_s)))
        nearest_gap = abs(float(target_distances[nearest_idx]) - requested_s)
        nearest_is_endpoint = bool(
            nearest_idx == 0 or nearest_idx == len(target_distances) - 1
        )
        strictly_interior = bool(0.0 < requested_s < total)
        if nearest_gap <= 0.05 and not (
            nearest_is_endpoint and strictly_interior
        ):
            # Preserve the exact requested phase boundary without creating a
            # near-duplicate RF/TF point beside an existing sample.
            target_distances[nearest_idx] = requested_s
        else:
            target_distances.append(requested_s)
    target_distances = sorted(set(target_distances))

    out = []
    for s in target_distances:
        if s <= 0.0:
            out.append(pts[0].copy())
            continue
        if s >= total:
            out.append(pts[-1].copy())
            continue
        hi = int(np.searchsorted(cumulative, s, side="right"))
        hi = int(np.clip(hi, 1, pts.shape[0] - 1))
        lo = hi - 1
        span = float(cumulative[hi] - cumulative[lo])
        tau = 0.0 if span <= 1e-12 else float((s - cumulative[lo]) / span)
        out.append(pts[lo] * (1.0 - tau) + pts[hi] * tau)
    return np.asarray(out, dtype=float), np.asarray(target_distances, dtype=float)


def _profile_rf_segments_for_transition(rf, transition_context):
    """Assign phase and angle-based altitude to RF/TF points after RF geometry."""
    if not bool(transition_context.get("enabled", False)):
        path = np.asarray(rf.get("path", np.empty((0, 3))), dtype=float).reshape(-1, 3)
        phases = np.full(path.shape[0], FLIGHT_PHASE_CRUISE, dtype=object)
        start_vertiport = transition_context.get("start_vertiport_lla")
        end_vertiport = transition_context.get("end_vertiport_lla")
        if path.shape[0] > 0:
            if start_vertiport is not None and _seg_dist_3d_m(path[0], start_vertiport) <= 0.05:
                phases[0] = FLIGHT_PHASE_VERTIPORT
            if end_vertiport is not None and _seg_dist_3d_m(path[-1], end_vertiport) <= 0.05:
                phases[-1] = FLIGHT_PHASE_VERTIPORT
        rf["path"] = path
        rf["flight_phases"] = phases
        disabled_segments = list(rf.get("segments", []))
        for seg_idx, seg in enumerate(disabled_segments):
            seg_pts = np.asarray(seg.get("points", np.empty((0, 3))), dtype=float).reshape(-1, 3)
            seg_phases = np.full(seg_pts.shape[0], FLIGHT_PHASE_CRUISE, dtype=object)
            if seg_pts.shape[0] > 0:
                if (
                    seg_idx == 0
                    and start_vertiport is not None
                    and _seg_dist_3d_m(seg_pts[0], start_vertiport) <= 0.05
                ):
                    seg_phases[0] = FLIGHT_PHASE_VERTIPORT
                if (
                    seg_idx == len(disabled_segments) - 1
                    and end_vertiport is not None
                    and _seg_dist_3d_m(seg_pts[-1], end_vertiport) <= 0.05
                ):
                    seg_phases[-1] = FLIGHT_PHASE_VERTIPORT
            seg["point_phases"] = seg_phases
        rf["transition_feasible"] = True
        rf["transition_fail_reason"] = "ok"
        return rf

    raw_segments = list(rf.get("segments", []))
    if not raw_segments:
        raw_path = np.asarray(rf.get("path", np.empty((0, 3))), dtype=float).reshape(-1, 3)
        if raw_path.shape[0] >= 2:
            raw_segments = [{"type": "TF", "points": raw_path}]

    seg_lengths = []
    for seg in raw_segments:
        seg_pts = np.asarray(seg.get("points", np.empty((0, 3))), dtype=float).reshape(-1, 3)
        seg_lengths.append(float(_polyline_cumulative_horizontal_m(seg_pts)[-1]) if seg_pts.shape[0] else 0.0)
    core_total = float(np.sum(seg_lengths))
    station_eps = max(
        1e-12,
        16.0 * np.finfo(float).eps * max(1.0, abs(core_total)),
    )
    takeoff_remaining = float(transition_context["takeoff"]["stage2_horizontal_distance_m"])
    landing_remaining = float(transition_context["landing"]["stage2_horizontal_distance_m"])
    takeoff_stage2_actual = bool(
        transition_context["takeoff"].get(
            "optimized_transition_actual", takeoff_remaining > 0.0
        )
        and takeoff_remaining > 0.0
    )
    landing_stage2_actual = bool(
        transition_context["landing"].get(
            "optimized_transition_actual", landing_remaining > 0.0
        )
        and landing_remaining > 0.0
    )
    distance_feasible = bool(
        core_total + station_eps >= takeoff_remaining + landing_remaining
    )
    fail_reasons = []
    if not distance_feasible:
        fail_reasons.append("transition_distance_overlap")

    takeoff_boundary_s = float(takeoff_remaining)
    landing_boundary_s = float(core_total - landing_remaining)
    cruise_alt = float(transition_context["cruise_altitude_m"])
    takeoff_start_alt = float(transition_context["takeoff"]["stage1_end_lla"][2])
    landing_end_alt = float(transition_context["landing"]["stage1_end_lla"][2])
    tan_takeoff = float(np.tan(np.deg2rad(transition_context["takeoff"]["angle_deg"])))
    tan_landing = float(np.tan(np.deg2rad(transition_context["landing"]["angle_deg"])))

    profiled_segments = []
    offset = 0.0
    takeoff_end_point = None
    landing_end_point = None
    takeoff_end_error_m = float("inf")
    landing_end_error_m = float("inf")
    requested_core_breaks = [takeoff_boundary_s, landing_boundary_s]
    if landing_stage2_actual:
        requested_core_breaks.append(
            landing_boundary_s + 0.5 * landing_remaining
        )
    for seg, seg_len in zip(raw_segments, seg_lengths):
        seg_pts = np.asarray(seg.get("points", np.empty((0, 3))), dtype=float).reshape(-1, 3)
        local_breaks = []
        for boundary in requested_core_breaks:
            if offset - station_eps <= boundary <= offset + seg_len + station_eps:
                local_breaks.append(boundary - offset)
        pts_i, local_s = _insert_polyline_distances(seg_pts, local_breaks)
        global_s = offset + local_s

        takeoff_alt = takeoff_start_alt + global_s * tan_takeoff
        landing_alt = landing_end_alt + (core_total - global_s) * tan_landing
        pts_i[:, 2] = np.minimum(cruise_alt, np.minimum(takeoff_alt, landing_alt))

        phases_i = np.full(pts_i.shape[0], FLIGHT_PHASE_CRUISE, dtype=object)
        if takeoff_stage2_actual:
            phases_i[global_s < takeoff_boundary_s] = FLIGHT_PHASE_TAKEOFF_STAGE2
        # The exact boundary still terminates the preceding edge. Mark only
        # samples beyond it as landing so the boundary-to-next edge is green.
        if landing_stage2_actual:
            phases_i[global_s > landing_boundary_s] = FLIGHT_PHASE_LANDING_STAGE2

        if global_s.size:
            takeoff_idx = int(np.argmin(np.abs(global_s - takeoff_boundary_s)))
            takeoff_error = float(abs(global_s[takeoff_idx] - takeoff_boundary_s))
            if takeoff_error <= station_eps:
                if takeoff_stage2_actual:
                    phases_i[takeoff_idx] = FLIGHT_PHASE_TAKEOFF_STAGE2
                if takeoff_error < takeoff_end_error_m:
                    takeoff_end_error_m = takeoff_error
                    takeoff_end_point = pts_i[takeoff_idx].copy()

            landing_idx = int(np.argmin(np.abs(global_s - landing_boundary_s)))
            landing_error = float(abs(global_s[landing_idx] - landing_boundary_s))
            if landing_error <= station_eps:
                if not (
                    takeoff_stage2_actual
                    and abs(landing_boundary_s - takeoff_boundary_s)
                    <= station_eps
                ):
                    phases_i[landing_idx] = FLIGHT_PHASE_CRUISE
                if landing_error < landing_end_error_m:
                    landing_end_error_m = landing_error
                    landing_end_point = pts_i[landing_idx].copy()

        seg_new = dict(seg)
        seg_new["points"] = pts_i
        seg_new["point_phases"] = phases_i
        seg_new["point_cumulative_core_m"] = global_s
        profiled_segments.append(seg_new)
        offset += seg_len

    if takeoff_end_point is None and profiled_segments:
        takeoff_end_point = np.asarray(profiled_segments[0]["points"], dtype=float)[0].copy()
    if landing_end_point is None and profiled_segments:
        landing_end_point = np.asarray(profiled_segments[-1]["points"], dtype=float)[-1].copy()

    full_segments = []
    takeoff_stage1_profile = np.asarray(
        transition_context["takeoff"].get("stage1_profile", np.empty((0, 3))),
        dtype=float,
    ).reshape(-1, 3)
    if takeoff_stage1_profile.shape[0] >= 2:
        full_segments.append({
            "type": "TF",
            "points": takeoff_stage1_profile,
            "point_phases": np.full(
                takeoff_stage1_profile.shape[0], FLIGHT_PHASE_TAKEOFF_STAGE1, dtype=object
            ),
            "is_fixed_transition_stage1": True,
        })
    full_segments.extend(profiled_segments)

    landing_stage1_profile_desc = np.asarray(
        transition_context["landing"].get("stage1_profile_desc", np.empty((0, 3))),
        dtype=float,
    ).reshape(-1, 3)
    if landing_stage1_profile_desc.shape[0] >= 2:
        full_segments.append({
            "type": "TF",
            "points": landing_stage1_profile_desc,
            "point_phases": np.full(
                landing_stage1_profile_desc.shape[0], FLIGHT_PHASE_LANDING_STAGE1, dtype=object
            ),
            "is_fixed_transition_stage1": True,
        })

    if full_segments:
        first_phases = np.asarray(full_segments[0]["point_phases"], dtype=object).copy()
        if first_phases.size:
            first_phases[0] = FLIGHT_PHASE_VERTIPORT
            full_segments[0]["point_phases"] = first_phases
        last_phases = np.asarray(full_segments[-1]["point_phases"], dtype=object).copy()
        if last_phases.size:
            last_phases[-1] = FLIGHT_PHASE_VERTIPORT
            full_segments[-1]["point_phases"] = last_phases

    path_parts = []
    phase_parts = []
    for seg in full_segments:
        pts_i = np.asarray(seg["points"], dtype=float).reshape(-1, 3)
        phases_i = np.asarray(seg["point_phases"], dtype=object).reshape(-1)
        if pts_i.shape[0] == 0:
            continue
        if not path_parts:
            path_parts.append(pts_i)
            phase_parts.append(phases_i)
        else:
            skip = 1 if _seg_dist_3d_m(path_parts[-1][-1], pts_i[0]) <= 0.05 else 0
            if pts_i[skip:].shape[0] > 0:
                path_parts.append(pts_i[skip:])
                phase_parts.append(phases_i[skip:])

    if path_parts:
        full_path = np.vstack([p for p in path_parts if p.size > 0])
        full_phases = np.concatenate([p for p in phase_parts if p.size > 0])
    else:
        full_path = np.empty((0, 3), dtype=float)
        full_phases = np.empty((0,), dtype=object)

    core_path_parts = []
    for seg in profiled_segments:
        pts_i = np.asarray(seg["points"], dtype=float).reshape(-1, 3)
        if pts_i.shape[0] == 0:
            continue
        if not core_path_parts:
            core_path_parts.append(pts_i)
        else:
            skip = 1 if _seg_dist_3d_m(core_path_parts[-1][-1], pts_i[0]) <= 0.05 else 0
            if pts_i[skip:].shape[0] > 0:
                core_path_parts.append(pts_i[skip:])
    core_path = (
        np.vstack(core_path_parts).astype(float)
        if core_path_parts else np.empty((0, 3), dtype=float)
    )

    takeoff_continuity_ok = False
    landing_continuity_ok = False
    if core_path.shape[0] == 0:
        fail_reasons.append("transition_core_path_missing")
    else:
        expected_takeoff_boundary = (
            takeoff_stage1_profile[-1]
            if takeoff_stage1_profile.shape[0] >= 2
            else np.asarray(
                transition_context["takeoff"]["stage1_end_lla"], dtype=float
            ).reshape(3)
        )
        expected_landing_boundary = (
            landing_stage1_profile_desc[0]
            if landing_stage1_profile_desc.shape[0] >= 2
            else np.asarray(
                transition_context["landing"]["stage1_end_lla"], dtype=float
            ).reshape(3)
        )
        takeoff_continuity_ok = bool(
            _seg_dist_3d_m(expected_takeoff_boundary, core_path[0]) <= 0.05
        )
        landing_continuity_ok = bool(
            _seg_dist_3d_m(core_path[-1], expected_landing_boundary) <= 0.05
        )
        if not takeoff_continuity_ok:
            fail_reasons.append("takeoff_stage1_core_discontinuity")
        if not landing_continuity_ok:
            fail_reasons.append("landing_core_stage1_discontinuity")

    duplicate_endpoint_ok = True
    for i in range(1, full_path.shape[0]):
        if np.allclose(
            full_path[i - 1],
            full_path[i],
            rtol=0.0,
            atol=1e-12,
        ):
            duplicate_endpoint_ok = False
            break
    if not duplicate_endpoint_ok:
        fail_reasons.append("duplicate_consecutive_endpoint")

    phase_order = {
        FLIGHT_PHASE_TAKEOFF_STAGE1: 1,
        FLIGHT_PHASE_TAKEOFF_STAGE2: 2,
        FLIGHT_PHASE_CRUISE: 3,
        FLIGHT_PHASE_LANDING_STAGE2: 4,
        FLIGHT_PHASE_LANDING_STAGE1: 5,
    }
    phase_order_ok = bool(full_path.shape[0] == full_phases.shape[0] and full_path.shape[0] > 0)
    phase_ranks = []
    if phase_order_ok:
        last_idx = full_phases.shape[0] - 1
        for i, phase in enumerate(full_phases):
            if phase == FLIGHT_PHASE_VERTIPORT:
                rank = 0 if i == 0 else (6 if i == last_idx else None)
            else:
                rank = phase_order.get(str(phase))
            if rank is None:
                phase_order_ok = False
                break
            phase_ranks.append(int(rank))
        if phase_order_ok and any(
            phase_ranks[i] < phase_ranks[i - 1]
            for i in range(1, len(phase_ranks))
        ):
            phase_order_ok = False
    if not phase_order_ok:
        fail_reasons.append("flight_phase_order_violation")

    takeoff_indices = np.flatnonzero(
        np.isin(
            full_phases,
            [FLIGHT_PHASE_TAKEOFF_STAGE1, FLIGHT_PHASE_TAKEOFF_STAGE2],
        )
    ).tolist()
    if full_path.shape[0] > 0 and (not takeoff_indices or takeoff_indices[0] != 0):
        takeoff_indices.insert(0, 0)
    takeoff_monotonic_ok = True
    if len(takeoff_indices) >= 2:
        takeoff_alts = full_path[np.asarray(takeoff_indices, dtype=int), 2]
        takeoff_monotonic_ok = bool(np.all(np.diff(takeoff_alts) >= -1e-6))
    if not takeoff_monotonic_ok:
        fail_reasons.append("takeoff_altitude_not_monotonic")

    landing_indices = np.flatnonzero(
        np.isin(
            full_phases,
            [FLIGHT_PHASE_LANDING_STAGE2, FLIGHT_PHASE_LANDING_STAGE1],
        )
    ).tolist()
    if full_path.shape[0] > 0 and (not landing_indices or landing_indices[-1] != full_path.shape[0] - 1):
        landing_indices.append(full_path.shape[0] - 1)
    landing_monotonic_ok = True
    if len(landing_indices) >= 2:
        landing_alts = full_path[np.asarray(landing_indices, dtype=int), 2]
        landing_monotonic_ok = bool(np.all(np.diff(landing_alts) <= 1e-6))
    if not landing_monotonic_ok:
        fail_reasons.append("landing_altitude_not_monotonic")

    first_heading = None
    last_heading = None
    for i in range(core_path.shape[0] - 1):
        if _seg_dist_m(core_path[i], core_path[i + 1]) > station_eps:
            first_heading = _heading_deg_from_segment(core_path[i], core_path[i + 1])
            break
    for i in range(core_path.shape[0] - 1, 0, -1):
        if _seg_dist_m(core_path[i - 1], core_path[i]) > station_eps:
            last_heading = _heading_deg_from_segment(core_path[i - 1], core_path[i])
            break

    target_takeoff = float(transition_context["takeoff_heading_deg"])
    target_landing_inbound = (
        float(transition_context["landing_heading_deg"]) + 180.0
    ) % 360.0
    half_width = float(transition_context["sector_half_width_deg"])
    require_takeoff_sector_heading = bool(
        transition_context.get(
            "require_takeoff_sector_heading",
            transition_context.get("require_sector_heading", True),
        )
    )
    require_landing_sector_heading = bool(
        transition_context.get(
            "require_landing_sector_heading",
            transition_context.get("require_sector_heading", True),
        )
    )
    takeoff_stage1_expected_m = float(
        transition_context["takeoff"]["stage1_straight_distance_m"]
    )
    landing_stage1_expected_m = float(
        transition_context["landing"]["stage1_straight_distance_m"]
    )
    takeoff_stage1_actual_m = (
        float(_polyline_cumulative_horizontal_m(takeoff_stage1_profile)[-1])
        if takeoff_stage1_profile.shape[0] >= 2 else 0.0
    )
    landing_stage1_actual_m = (
        float(_polyline_cumulative_horizontal_m(landing_stage1_profile_desc)[-1])
        if landing_stage1_profile_desc.shape[0] >= 2 else 0.0
    )
    takeoff_stage1_distance_tolerance_m = max(
        0.5, abs(takeoff_stage1_expected_m) * 5e-4
    )
    landing_stage1_distance_tolerance_m = max(
        0.5, abs(landing_stage1_expected_m) * 5e-4
    )
    takeoff_stage1_distance_ok = bool(
        abs(takeoff_stage1_actual_m - takeoff_stage1_expected_m)
        <= takeoff_stage1_distance_tolerance_m
    )
    landing_stage1_distance_ok = bool(
        abs(landing_stage1_actual_m - landing_stage1_expected_m)
        <= landing_stage1_distance_tolerance_m
    )
    takeoff_stage1_heading_ok = True
    landing_stage1_heading_ok = True
    if takeoff_stage1_profile.shape[0] >= 2:
        stage1_heading = _heading_deg_from_segment(
            takeoff_stage1_profile[0], takeoff_stage1_profile[-1]
        )
        takeoff_stage1_heading_ok = bool(
            stage1_heading is not None
            and _angular_difference_deg(stage1_heading, target_takeoff) <= 0.1
        )
    if landing_stage1_profile_desc.shape[0] >= 2:
        stage1_heading = _heading_deg_from_segment(
            landing_stage1_profile_desc[0], landing_stage1_profile_desc[-1]
        )
        landing_stage1_heading_ok = bool(
            stage1_heading is not None
            and _angular_difference_deg(stage1_heading, target_landing_inbound) <= 0.1
        )
    if not takeoff_stage1_distance_ok:
        fail_reasons.append("takeoff_stage1_distance_mismatch")
    if not landing_stage1_distance_ok:
        fail_reasons.append("landing_stage1_distance_mismatch")
    if not takeoff_stage1_heading_ok:
        fail_reasons.append("takeoff_stage1_heading_violation")
    if not landing_stage1_heading_ok:
        fail_reasons.append("landing_stage1_heading_violation")
    takeoff_sector_ok = True
    landing_sector_ok = True
    if require_takeoff_sector_heading:
        takeoff_sector_ok = bool(
            first_heading is not None
            and _angular_difference_deg(first_heading, target_takeoff)
            <= half_width + 1e-6
        )
        if first_heading is None:
            fail_reasons.append("takeoff_sector_heading_unavailable")
        elif not takeoff_sector_ok:
            fail_reasons.append("takeoff_sector_heading_violation")
    if require_landing_sector_heading:
        landing_sector_ok = bool(
            last_heading is not None
            and _angular_difference_deg(last_heading, target_landing_inbound)
            <= half_width + 1e-6
        )
        if last_heading is None:
            fail_reasons.append("landing_sector_heading_unavailable")
        elif not landing_sector_ok:
            fail_reasons.append("landing_sector_heading_violation")

    takeoff_transition_endpoint_ok = bool(
        not distance_feasible
        or (
            takeoff_end_point is not None
            and abs(float(takeoff_end_point[2]) - cruise_alt) <= 1e-4
        )
    )
    landing_transition_endpoint_ok = bool(
        not distance_feasible
        or (
            landing_end_point is not None
            and abs(float(landing_end_point[2]) - cruise_alt) <= 1e-4
        )
    )
    if not takeoff_transition_endpoint_ok:
        fail_reasons.append("takeoff_transition_endpoint_missing_or_wrong_altitude")
    if not landing_transition_endpoint_ok:
        fail_reasons.append("landing_transition_endpoint_missing_or_wrong_altitude")

    transition_feasible = bool(not fail_reasons)
    fail_reason = fail_reasons[0] if fail_reasons else "ok"
    validation_checks = {
        "distance_non_overlapping": bool(distance_feasible),
        "takeoff_stage1_core_continuity": bool(takeoff_continuity_ok),
        "landing_core_stage1_continuity": bool(landing_continuity_ok),
        "no_duplicate_consecutive_endpoint": bool(duplicate_endpoint_ok),
        "flight_phase_order": bool(phase_order_ok),
        "takeoff_altitude_monotonic": bool(takeoff_monotonic_ok),
        "landing_altitude_monotonic": bool(landing_monotonic_ok),
        "takeoff_stage1_distance": bool(takeoff_stage1_distance_ok),
        "landing_stage1_distance": bool(landing_stage1_distance_ok),
        "takeoff_stage1_heading": bool(takeoff_stage1_heading_ok),
        "landing_stage1_heading": bool(landing_stage1_heading_ok),
        "takeoff_transition_endpoint": bool(takeoff_transition_endpoint_ok),
        "landing_transition_endpoint": bool(landing_transition_endpoint_ok),
        "takeoff_sector_heading": bool(takeoff_sector_ok),
        "landing_sector_heading": bool(landing_sector_ok),
    }

    rf["segments"] = full_segments
    rf["path"] = full_path
    rf["flight_phases"] = full_phases
    rf["transition_feasible"] = bool(transition_feasible)
    rf["transition_fail_reason"] = str(fail_reason)
    rf["transition_fail_reasons"] = list(fail_reasons)
    rf["feasible"] = bool(rf.get("feasible", True) and transition_feasible)
    rf["transition_meta"] = {
        "enabled": True,
        "two_stage_enabled": bool(transition_context.get("two_stage_enabled", False)),
        "transition_structure_mode": str(
            transition_context.get(
                "transition_structure_mode",
                TRANSITION_STRUCTURE_FIXED_PLUS_OPTIMIZED,
            )
        ),
        "core_horizontal_distance_m": core_total,
        "validation_checks": validation_checks,
        "takeoff_transition_end": None if takeoff_end_point is None else takeoff_end_point,
        "landing_transition_end": None if landing_end_point is None else landing_end_point,
        "takeoff_transition_total_horizontal_distance_m": float(
            transition_context["takeoff"]["total_horizontal_distance_m"]
        ),
        "landing_transition_total_horizontal_distance_m": float(
            transition_context["landing"]["total_horizontal_distance_m"]
        ),
        "takeoff_stage1_straight_distance_m": float(
            transition_context["takeoff"]["stage1_straight_distance_m"]
        ),
        "landing_stage1_straight_distance_m": float(
            transition_context["landing"]["stage1_straight_distance_m"]
        ),
        "takeoff_stage1_distance_tolerance_m": float(
            takeoff_stage1_distance_tolerance_m
        ),
        "landing_stage1_distance_tolerance_m": float(
            landing_stage1_distance_tolerance_m
        ),
        "takeoff_stage1_requested_straight_distance_m": float(
            transition_context["takeoff"].get(
                "stage1_requested_straight_distance_m",
                transition_context["takeoff"]["stage1_straight_distance_m"],
            )
        ),
        "landing_stage1_requested_straight_distance_m": float(
            transition_context["landing"].get(
                "stage1_requested_straight_distance_m",
                transition_context["landing"]["stage1_straight_distance_m"],
            )
        ),
        "takeoff_stage2_horizontal_distance_m": float(takeoff_remaining),
        "landing_stage2_horizontal_distance_m": float(landing_remaining),
        "takeoff_optimized_transition_actual": bool(takeoff_stage2_actual),
        "landing_optimized_transition_actual": bool(landing_stage2_actual),
        "takeoff_stage2_collapsed_at_cruise": bool(
            transition_context["takeoff"].get("stage2_collapsed_at_cruise", False)
        ),
        "landing_stage2_collapsed_at_cruise": bool(
            transition_context["landing"].get("stage2_collapsed_at_cruise", False)
        ),
        "takeoff_stage1_clamped_to_cruise": bool(
            transition_context["takeoff"].get("stage1_clamped_to_cruise", False)
        ),
        "landing_stage1_clamped_to_cruise": bool(
            transition_context["landing"].get("stage1_clamped_to_cruise", False)
        ),
        "fixed_transition_general_output_suppressed": bool(
            transition_context.get("transition_structure_mode")
            == TRANSITION_STRUCTURE_FIXED_ONLY
        ),
        "fixed_transition_evaluated_but_not_exported": bool(
            transition_context.get("transition_structure_mode")
            == TRANSITION_STRUCTURE_FIXED_ONLY
        ),
        "takeoff_climb_angle_deg": float(transition_context["takeoff"]["angle_deg"]),
        "landing_descent_angle_deg": float(transition_context["landing"]["angle_deg"]),
        "takeoff_stage1_end": np.asarray(
            transition_context["takeoff"]["stage1_end_lla"], dtype=float
        ).copy(),
        "landing_stage1_start": np.asarray(
            transition_context["landing"]["stage1_end_lla"], dtype=float
        ).copy(),
    }
    rf["takeoff_stage1_end"] = rf["transition_meta"]["takeoff_stage1_end"]
    rf["takeoff_transition_end"] = rf["transition_meta"]["takeoff_transition_end"]
    rf["landing_transition_start"] = rf["transition_meta"]["landing_transition_end"]
    rf["landing_stage1_start"] = rf["transition_meta"]["landing_stage1_start"]
    return rf


def _build_output_rf_view_v1(
    rf,
    transition_structure_mode,
    transition_enabled=True,
):
    """Return the public corridor view without exposing fixed-only straight legs."""
    structure_mode = str(transition_structure_mode).strip().lower()
    suppress_fixed = bool(
        transition_enabled
        and structure_mode == TRANSITION_STRUCTURE_FIXED_ONLY
    )
    if not suppress_fixed:
        return rf

    output_rf = dict(rf)
    output_segments = []
    for seg in rf.get("segments", []):
        if bool(seg.get("is_fixed_transition_stage1", False)):
            continue
        seg_copy = dict(seg)
        pts = np.asarray(
            seg.get("points", np.empty((0, 3))), dtype=float
        ).reshape(-1, 3).copy()
        phases = np.asarray(
            seg.get(
                "point_phases",
                np.full(pts.shape[0], FLIGHT_PHASE_CRUISE, dtype=object),
            ),
            dtype=object,
        ).reshape(-1).copy()
        if phases.size != pts.shape[0]:
            raise RuntimeError(
                "Output RF segment phase count mismatch: "
                f"points={pts.shape[0]}, phases={phases.size}."
            )
        phases[:] = FLIGHT_PHASE_CRUISE
        seg_copy["points"] = pts
        seg_copy["point_phases"] = phases
        output_segments.append(seg_copy)

    path_parts = []
    phase_parts = []
    for seg in output_segments:
        pts = np.asarray(seg["points"], dtype=float).reshape(-1, 3)
        phases = np.asarray(seg["point_phases"], dtype=object).reshape(-1)
        if pts.shape[0] == 0:
            continue
        skip = 0
        if path_parts and _seg_dist_3d_m(path_parts[-1][-1], pts[0]) <= 0.05:
            skip = 1
        if pts[skip:].shape[0] > 0:
            path_parts.append(pts[skip:])
            phase_parts.append(phases[skip:])

    output_path = (
        np.vstack(path_parts).astype(float)
        if path_parts else np.empty((0, 3), dtype=float)
    )
    output_phases = (
        np.concatenate(phase_parts).astype(object)
        if phase_parts else np.empty((0,), dtype=object)
    )
    if output_path.shape[0] != output_phases.size:
        raise RuntimeError(
            "Output RF path/phase mismatch after fixed-transition suppression."
        )

    output_rf["segments"] = output_segments
    output_rf["path"] = output_path
    output_rf["flight_phases"] = output_phases
    output_rf["fixed_transition_general_output_suppressed"] = True
    output_rf["fixed_transition_evaluated_but_not_exported"] = True
    return output_rf


def build_full_corridor_path(start_vertiport, takeoff_complete, path_core, landing_entry, end_vertiport):
    return np.vstack([start_vertiport, takeoff_complete, path_core, landing_entry, end_vertiport]).astype(float)


def _stitch_full_corridor_from_profiles(takeoff_profile, core_path, landing_profile_desc):
    takeoff_profile = np.asarray(takeoff_profile, dtype=float).reshape(-1, 3)
    core_path = np.asarray(core_path, dtype=float).reshape(-1, 3)
    landing_profile_desc = np.asarray(landing_profile_desc, dtype=float).reshape(-1, 3)

    pieces = [takeoff_profile]
    if core_path.size > 0:
        pieces.append(core_path[1:-1] if core_path.shape[0] >= 2 else core_path)
    if landing_profile_desc.size > 0:
        pieces.append(landing_profile_desc[1:])
    if not pieces:
        return np.empty((0, 3), dtype=float)

    stitched = np.vstack([p for p in pieces if np.size(p) > 0]).astype(float)
    return stitched


# Apply RF turns on cruise-only span and stitch straight transitions
def apply_rf_turns_full_corridor(
    path_core,
    start_vertiport,
    end_vertiport,
    ground_speed_mps,
    bank_angle_deg,
    num_arc_points,
    look_ahead,
    look_ahead_threshold_m,
    look_ahead_min_scale,
    look_ahead_window,
    use_boundary_heading=False,
    rf_debug_level="off",
    allow_tangent_clamp=None,
    corner_fit_margin=None,
    corner_min_tangent_m=None,
    min_turn_angle_deg=None,
    transition_context=None,
):
    """Apply RF to the optimizable span, then apply the v1 transition profile."""
    core = np.asarray(path_core, dtype=float)
    if core.size == 0:
        core = np.empty((0, 3), dtype=float)
    else:
        core = core.reshape(-1, 3)

    optimization_start = np.asarray(start_vertiport, dtype=float).reshape(3)
    optimization_end = np.asarray(end_vertiport, dtype=float).reshape(3)
    if transition_context is None:
        transition_context = globals().get("TRANSITION_CONTEXT", None)
    if transition_context is None:
        transition_context = {"enabled": False}

    backbone_parts = []
    if core.shape[0] == 0 or _seg_dist_3d_m(core[0], optimization_start) > 0.05:
        backbone_parts.append(optimization_start.reshape(1, 3))
    if core.shape[0] > 0:
        backbone_parts.append(core)
    if core.shape[0] == 0 or _seg_dist_3d_m(core[-1], optimization_end) > 0.05:
        backbone_parts.append(optimization_end.reshape(1, 3))
    backbone = np.vstack(backbone_parts).astype(float)

    # Optional RF boundary headings follow the actual candidate tangent rather
    # than forcing a sector-center tangent. Sector compliance is validated on
    # the resulting first/last non-zero legs independently.
    entry_heading_deg = None
    exit_heading_deg = None
    if bool(use_boundary_heading):
        for i in range(backbone.shape[0] - 1):
            if _seg_dist_m(backbone[i], backbone[i + 1]) > 0.05:
                entry_heading_deg = _heading_deg_from_segment(backbone[i], backbone[i + 1])
                break
        for i in range(backbone.shape[0] - 1, 0, -1):
            if _seg_dist_m(backbone[i - 1], backbone[i]) > 0.05:
                exit_heading_deg = _heading_deg_from_segment(backbone[i - 1], backbone[i])
                break

    if allow_tangent_clamp is None:
        allow_tangent_clamp = bool(RF_ALLOW_TANGENT_CLAMP)
    if corner_fit_margin is None:
        corner_fit_margin = float(RF_CORNER_FIT_MARGIN)
    if corner_min_tangent_m is None:
        corner_min_tangent_m = float(RF_CORNER_MIN_TANGENT_M)
    if min_turn_angle_deg is None:
        min_turn_angle_deg = float(RF_MIN_TURN_ANGLE_DEG)
    rf = apply_rf_turns(
        backbone,
        ground_speed_mps,
        bank_angle_deg,
        num_arc_points,
        look_ahead=look_ahead,
        look_ahead_threshold_m=look_ahead_threshold_m,
        look_ahead_min_scale=look_ahead_min_scale,
        look_ahead_window=look_ahead_window,
        rf_debug_level=rf_debug_level,
        entry_heading_deg=entry_heading_deg,
        exit_heading_deg=exit_heading_deg,
        allow_tangent_clamp=allow_tangent_clamp,
        corner_fit_margin=corner_fit_margin,
        corner_min_tangent_m=corner_min_tangent_m,
        min_turn_angle_deg=min_turn_angle_deg,
    )
    # Preserve RF-only geometry feasibility before transition profiling folds
    # its independent validation result into rf["feasible"].
    rf["rf_geometry_feasible"] = bool(rf.get("feasible", False))

    return _profile_rf_segments_for_transition(rf, transition_context)


def draw_vertiport_radius_rings(gx, center_lla, radii_m=(4500.0, 5000.0, 5500.0), n_pts=240):
    if gx is None or center_lla is None:
        return
    lat0 = float(center_lla[0])
    lon0 = float(center_lla[1])
    m_lat = 111000.0
    m_lon = 111000.0 * np.cos(np.deg2rad(lat0))
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_pts), endpoint=True)

    radii = np.atleast_1d(np.asarray(radii_m, dtype=float)).ravel()
    if radii.size == 0:
        return
    for idx, rad_m in enumerate(sorted(radii.tolist())):
        label = f"Airspace Radius {rad_m / 1000.0:.1f} km"
        lat_ring = lat0 + (rad_m * np.sin(theta)) / m_lat
        lon_ring = lon0 + (rad_m * np.cos(theta)) / m_lon
        gx.plot(
            lon_ring,
            lat_ring,
            "-",
            color="deepskyblue" if idx == len(radii) - 1 else "steelblue",
            linewidth=1.6 if idx == len(radii) - 1 else 1.0,
            alpha=0.65 if idx == len(radii) - 1 else 0.35,
            transform=ccrs.Geodetic(),
            zorder=1,
            label=label,
        )


def compute_centered_map_extent(latlon_points, vertiport, ring_radii_m=(4500.0, 5000.0, 5500.0), pad_ratio=0.10):
    pts = np.asarray(latlon_points, dtype=float)
    if pts.size == 0:
        raise ValueError("latlon_points must not be empty")

    lat0 = float(vertiport[0])
    lon0 = float(vertiport[1])
    m_lat = 111000.0
    m_lon0 = 111000.0 * np.cos(np.deg2rad(lat0))

    max_ring = float(np.max(ring_radii_m)) if len(ring_radii_m) > 0 else 0.0
    ring_lat = max_ring / m_lat
    ring_lon = max_ring / m_lon0 if m_lon0 > 1e-9 else ring_lat

    min_lat = min(float(np.min(pts[:, 0])), lat0 - ring_lat)
    max_lat = max(float(np.max(pts[:, 0])), lat0 + ring_lat)
    min_lon = min(float(np.min(pts[:, 1])), lon0 - ring_lon)
    max_lon = max(float(np.max(pts[:, 1])), lon0 + ring_lon)

    c_lat = 0.5 * (min_lat + max_lat)
    c_lon = 0.5 * (min_lon + max_lon)
    m_lon = 111000.0 * np.cos(np.deg2rad(c_lat))

    half_lat_m = 0.5 * (max_lat - min_lat) * m_lat
    half_lon_m = 0.5 * (max_lon - min_lon) * m_lon
    half_m = max(half_lat_m, half_lon_m) * (1.0 + float(pad_ratio))

    half_lat = half_m / m_lat
    half_lon = half_m / m_lon if m_lon > 1e-9 else half_lat
    return [c_lon - half_lon, c_lon + half_lon, c_lat - half_lat, c_lat + half_lat]


def build_circle_lla(center_lla, radius_m, n_pts=120):
    """Return circle boundary points as LLA list (lat, lon, alt)."""
    lat0 = float(center_lla[0])
    lon0 = float(center_lla[1])
    alt0 = float(center_lla[2]) if len(center_lla) >= 3 else 0.0
    m_lat = 111000.0
    m_lon = 111000.0 * np.cos(np.deg2rad(lat0))
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_pts), endpoint=False)
    pts = []
    for t in theta:
        lat = lat0 + (float(radius_m) * np.sin(t)) / m_lat
        lon = lon0 + (float(radius_m) * np.cos(t)) / m_lon
        pts.append([float(lat), float(lon), alt0])
    if pts:
        pts.append(pts[0])
    return pts


def bbox_to_polygon_lla(rect, alt_m=0.0):
    """Convert bbox [lon_min, lon_max, lat_min, lat_max] to closed LLA polygon."""
    lon_min, lon_max, lat_min, lat_max = [float(v) for v in rect]
    a = [lat_min, lon_min, float(alt_m)]
    b = [lat_min, lon_max, float(alt_m)]
    c = [lat_max, lon_max, float(alt_m)]
    d = [lat_max, lon_min, float(alt_m)]
    return [a, b, c, d, a]


def load_fixed_agl_moc_maps(
    moc_dir,
    altitude_levels,
    vertiport_elevation_msl_m,
    Ny,
    Nx,
    lat_lim,
    lon_lim,
):
    """Load fixed-AGL XYZ MOC maps and align them to the evaluation grid."""
    moc_dir = Path(moc_dir)
    if not moc_dir.is_dir():
        raise FileNotFoundError(f"MOC directory not found: {moc_dir}")

    expected_agl_levels = np.arange(100, 1000, 100, dtype=int)
    files_by_agl = {}
    prefix = "UAM_MOC_XYZ_risk_fixedAGL"
    for path in moc_dir.glob(f"{prefix}*.npy"):
        suffix = path.stem[len(prefix):]
        if suffix.isdigit():
            files_by_agl[int(suffix)] = path

    missing = [int(v) for v in expected_agl_levels if int(v) not in files_by_agl]
    if missing:
        raise FileNotFoundError(f"Missing fixed-AGL MOC maps: {missing}")

    source_by_agl = {}
    risk_counts = {}
    ref_x = None
    ref_y = None
    for agl_m in expected_agl_levels:
        path = files_by_agl[int(agl_m)]
        xyzr = np.asarray(np.load(str(path), allow_pickle=False), dtype=float)
        if xyzr.ndim != 2 or xyzr.shape[1] != 4:
            raise ValueError(f"{path.name} must have shape (N, 4), got {xyzr.shape}")
        if not np.all(np.isfinite(xyzr)):
            raise ValueError(f"{path.name} contains NaN or Inf")

        x, y, z, risk = (xyzr[:, i] for i in range(4))
        unique_risk = np.unique(risk)
        if not np.all(np.isin(unique_risk, [0.0, 1.0])):
            raise ValueError(f"{path.name} risk column must contain only 0/1")
        if np.unique(z).size != 1:
            raise ValueError(f"{path.name} must contain one fixed flight altitude")

        unique_x = np.unique(x)
        unique_y = np.unique(y)
        if unique_x.size * unique_y.size != xyzr.shape[0]:
            raise ValueError(f"{path.name} does not form a complete rectangular grid")
        if unique_x.size < 2 or unique_y.size < 2:
            raise ValueError(f"{path.name} grid must have at least 2 cells per axis")
        if not np.allclose(np.diff(unique_x), np.diff(unique_x)[0]):
            raise ValueError(f"{path.name} X spacing is not uniform")
        if not np.allclose(np.diff(unique_y), np.diff(unique_y)[0]):
            raise ValueError(f"{path.name} Y spacing is not uniform")

        if ref_x is None:
            ref_x = unique_x
            ref_y = unique_y
        elif not np.array_equal(unique_x, ref_x) or not np.array_equal(unique_y, ref_y):
            raise ValueError(f"{path.name} grid coordinates differ from the other MOC maps")

        ix = np.searchsorted(unique_x, x)
        iy = np.searchsorted(unique_y, y)
        risk_grid = np.empty((unique_y.size, unique_x.size), dtype=np.uint8)
        risk_grid[iy, ix] = risk.astype(np.uint8)
        source_by_agl[int(agl_m)] = risk_grid
        risk_counts[int(agl_m)] = int(np.count_nonzero(risk_grid))

    counts = np.array([risk_counts[int(v)] for v in expected_agl_levels], dtype=int)
    if np.any(np.diff(counts) > 0):
        raise ValueError("MOC risk-cell counts must not increase as AGL increases")
    for low_agl, high_agl in zip(expected_agl_levels[:-1], expected_agl_levels[1:]):
        low = source_by_agl[int(low_agl)]
        high = source_by_agl[int(high_agl)]
        if np.any((high == 1) & (low == 0)):
            raise ValueError("Higher-AGL MOC risk cells must be a subset of lower-AGL cells")

    requested_agl = (
        np.asarray(altitude_levels, dtype=float).ravel()
        - float(vertiport_elevation_msl_m)
    )
    if requested_agl.size == 0:
        raise ValueError("altitude_levels must contain at least one MSL altitude")
    if np.any(requested_agl < float(expected_agl_levels[0]) - 1e-9):
        raise ValueError(
            f"Requested AGL {requested_agl.tolist()}m is below the minimum supported "
            f"AGL {int(expected_agl_levels[0])}m"
        )

    selected_agl = []
    for agl_m in requested_agl:
        eligible = expected_agl_levels[expected_agl_levels <= float(agl_m) + 1e-9]
        selected_agl.append(int(eligible[-1]) if eligible.size else int(expected_agl_levels[0]))
    selected_agl = np.asarray(selected_agl, dtype=int)

    try:
        import pyproj
    except Exception as exc:
        raise RuntimeError("pyproj is required to align EPSG:5179 MOC maps") from exc

    target_lats = np.linspace(float(lat_lim[0]), float(lat_lim[1]), int(Ny))
    target_lons = np.linspace(float(lon_lim[0]), float(lon_lim[1]), int(Nx))
    target_lon_2d, target_lat_2d = np.meshgrid(target_lons, target_lats)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
    target_x, target_y = transformer.transform(target_lon_2d, target_lat_2d)

    dx = float(ref_x[1] - ref_x[0])
    dy = float(ref_y[1] - ref_y[0])
    source_i = np.rint((target_x - float(ref_x[0])) / dx).astype(int)
    source_j = np.rint((target_y - float(ref_y[0])) / dy).astype(int)
    source_i = np.clip(source_i, 0, ref_x.size - 1)
    source_j = np.clip(source_j, 0, ref_y.size - 1)
    inside = (
        (target_x >= float(ref_x[0]) - 0.5 * dx)
        & (target_x <= float(ref_x[-1]) + 0.5 * dx)
        & (target_y >= float(ref_y[0]) - 0.5 * dy)
        & (target_y <= float(ref_y[-1]) + 0.5 * dy)
    )

    moc_risk = np.ones((int(Ny), int(Nx), selected_agl.size), dtype=np.uint8)
    for layer_idx, agl_m in enumerate(selected_agl):
        source = source_by_agl[int(agl_m)]
        moc_risk[:, :, layer_idx][inside] = source[
            source_j[inside],
            source_i[inside],
        ]

    selected_names = [files_by_agl[int(v)].name for v in selected_agl]
    print(f"Selected MOC file(s): {', '.join(selected_names)}")

    meta = {
        "requested_agl_m": [float(v) for v in requested_agl],
        "selected_agl_m": [int(v) for v in selected_agl],
        "selection_policy": "floor_to_available_agl_then_cap_at_900m",
        "available_agl_m": [int(v) for v in expected_agl_levels],
        "source_risk_cell_counts": {
            str(int(v)): int(risk_counts[int(v)]) for v in expected_agl_levels
        },
        "selected_ones_ratio_on_evaluation_grid": float(np.mean(moc_risk)),
        "outside_source_grid_is_blocked": True,
    }
    return moc_risk, np.max(moc_risk, axis=2).astype(float), meta


def load_noise_risk_from_npy(
    npy_path,
    Ny,
    Nx,
    altitude_levels,
    noise_floor_db=0.0,
):
    """Load 3D noise npy and align to (Ny, Nx, len(altitude_levels))."""
    npy_path = Path(npy_path)
    if not npy_path.exists():
        raise FileNotFoundError(f"Noise NPY not found: {npy_path}")

    raw = np.load(str(npy_path), allow_pickle=True).item()
    if "Risk_3d" not in raw:
        raise KeyError(f"'Risk_3d' key not found in {npy_path.name}")

    risk_3d = np.asarray(raw["Risk_3d"], dtype=float)
    if risk_3d.ndim != 3:
        raise ValueError(f"Risk_3d must be 3D, got ndim={risk_3d.ndim}")

    if risk_3d.shape[0] == Nx and risk_3d.shape[1] == Ny:
        risk_3d = np.transpose(risk_3d, (1, 0, 2))
        transposed = True
    elif risk_3d.shape[0] == Ny and risk_3d.shape[1] == Nx:
        transposed = False
    else:
        raise RuntimeError(
            f"Noise Risk_3d shape {risk_3d.shape} incompatible with expected "
            f"(Ny,Nx,Nz)=({Ny},{Nx},Nz) or ({Nx},{Ny},Nz)"
        )

    if "z_vec" in raw:
        z_vec = np.asarray(raw["z_vec"], dtype=float).ravel()
    elif "altitude_vec" in raw:
        z_vec = np.asarray(raw["altitude_vec"], dtype=float).ravel()
    else:
        z_vec = np.array([0.0], dtype=float)

    if z_vec.size != risk_3d.shape[2]:
        raise ValueError(
            f"Noise z-vector length {z_vec.size} != Risk_3d Nz {risk_3d.shape[2]}"
        )

    A = int(len(altitude_levels))
    noise_db_stack = np.zeros((Ny, Nx, A), dtype=float)
    selected_idx = []
    for i, alt in enumerate(np.asarray(altitude_levels, dtype=float).ravel()):
        src_idx = int(np.argmin(np.abs(z_vec - float(alt))))
        selected_idx.append(src_idx)
        layer = risk_3d[:, :, src_idx]
        layer_active = np.where(
            np.isfinite(layer) & (layer > float(noise_floor_db)),
            layer,
            0.0,
        )
        noise_db_stack[:, :, i] = layer_active

    vmax = float(np.max(noise_db_stack)) if noise_db_stack.size > 0 else 0.0
    noise_norm_stack = (noise_db_stack / vmax) if vmax > 1e-12 else np.zeros_like(noise_db_stack)

    lat_lim_meta = raw.get("lat_lim", None)
    lon_lim_meta = raw.get("lon_lim", None)
    nan_ratio_raw = float(np.mean(~np.isfinite(risk_3d)))
    finite_raw = risk_3d[np.isfinite(risk_3d)]
    negative_raw_count = int(np.sum(finite_raw < 0.0)) if finite_raw.size > 0 else 0

    meta = {
        "source_type": "npy",
        "npy_path": str(npy_path),
        "risk3d_shape_raw": [int(v) for v in raw["Risk_3d"].shape],
        "risk3d_shape_aligned": [int(v) for v in risk_3d.shape],
        "transposed_to_ny_nx": bool(transposed),
        "z_vec_source": [float(v) for v in z_vec.tolist()],
        "selected_layer_idx": [int(v) for v in selected_idx],
        "noise_floor_db": float(noise_floor_db),
        "noise_max_db_after_floor": float(vmax),
        "nan_ratio_raw": float(nan_ratio_raw),
        "negative_count_raw": int(negative_raw_count),
        "lat_lim_meta": lat_lim_meta,
        "lon_lim_meta": lon_lim_meta,
        "metadata_in_npy": raw.get("metadata", None),
    }
    return noise_norm_stack.astype(float), noise_db_stack.astype(float), meta


def save_clicked_waypoints(latlon_points, fixed_alt_m, out_dir, base_name="clicked_waypoints"):
    """Save clicked waypoints into JSON and CSV files, preserving click order."""
    pts = np.asarray(latlon_points, dtype=float)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, p in enumerate(pts, start=1):
        rows.append({
            "order": int(i),
            "lat": float(p[0]),
            "lon": float(p[1]),
            "alt_m": float(fixed_alt_m),
        })

    json_path = out_path / f"{base_name}.json"
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(
            {
                "count": int(len(rows)),
                "fixed_altitude_m": float(fixed_alt_m),
                "waypoints": rows,
            },
            f_json,
            ensure_ascii=False,
            indent=2,
        )

    csv_path = out_path / f"{base_name}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=["order", "lat", "lon", "alt_m"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved clicked waypoints JSON: {json_path}")
    print(f"Saved clicked waypoints CSV : {csv_path}")
    return json_path, csv_path


def collect_waypoints_from_clicks(
    vertiport,
    lat_lim,
    lon_lim,
    request,
    altitude_levels,
    map_zoom=13,
    start_vertiport=None,
    end_vertiport=None,
    takeoff_complete=None,
    landing_entry=None,
    use_takeoff_landing_transition=False,
    use_two_stage_transition=False,
    transition_structure_mode=None,
    takeoff_optimized_transition_actual=None,
    landing_optimized_transition_actual=None,
    takeoff_heading_deg=None,
    landing_heading_deg=None,
    takeoff_sector_user=None,
    landing_sector_user=None,
    sector_half_width_deg=15.0,
    emergency_points=None,
    forbidden_zones=None,
    moc_binary_2d=None,
    ring_radii_m=(4500.0, 5000.0, 5500.0),
    extent_pad_ratio=0.18,
):
    """
    Collect waypoints by click order on map.
    - Left click: add WP
    - Right click or Backspace/Delete: remove last WP
    - Enter: finish
    """
    if not USE_INTERACTIVE_BACKEND:
        print("Interactive click backend is unavailable (Tk not available).")
        return np.empty((0, 2), dtype=float)

    # Switch to TkAgg only for click capture.
    try:
        plt.switch_backend("TkAgg")
        matplotlib.rcParams["toolbar"] = "None"
    except Exception:
        print("Failed to activate TkAgg for click input. Falling back to default waypoints.")
        return np.empty((0, 2), dtype=float)

    fig = plt.figure("Waypoint Click Input", figsize=(11, 8))
    fig.subplots_adjust(left=0.06, right=0.72)
    ax = fig.add_subplot(1, 1, 1, projection=request.crs)

    # Ensure full airspace rings are visible with margin, plus key overlays.
    extent_points = [np.asarray(vertiport[:2], dtype=float)]
    if start_vertiport is not None:
        extent_points.append(np.asarray(start_vertiport[:2], dtype=float))
    if end_vertiport is not None:
        extent_points.append(np.asarray(end_vertiport[:2], dtype=float))
    if takeoff_complete is not None:
        extent_points.append(np.asarray(takeoff_complete[:2], dtype=float))
    if landing_entry is not None:
        extent_points.append(np.asarray(landing_entry[:2], dtype=float))
    if emergency_points is not None:
        em = np.asarray(emergency_points, dtype=float)
        if em.size > 0:
            extent_points.extend(em[:, :2].tolist())
    if forbidden_zones is not None:
        fz = np.asarray(forbidden_zones, dtype=float)
        if fz.size > 0:
            for rect in fz:
                lon_min, lon_max, lat_min, lat_max = [float(v) for v in rect]
                extent_points.extend([
                    [lat_min, lon_min], [lat_min, lon_max],
                    [lat_max, lon_min], [lat_max, lon_max],
                ])

    extent_points = np.asarray(extent_points, dtype=float)
    click_extent = compute_centered_map_extent(
        extent_points,
        vertiport,
        ring_radii_m=ring_radii_m,
        pad_ratio=float(extent_pad_ratio),
    )
    click_lon_min, click_lon_max, click_lat_min, click_lat_max = [float(v) for v in click_extent]
    ax.set_extent(click_extent)
    ax.add_image(request, int(map_zoom))
    ax.set_title(_title_with_altitude(
        "Click WPs in order | Left: add, Right/Delete: undo, Enter: finish",
        altitude_levels,
        vertiport,
    ))

    draw_vertiport_radius_rings(ax, vertiport, radii_m=ring_radii_m)

    wedge_radius_m = float(np.clip(max(ring_radii_m) * 0.12, 250.0, 700.0)) if ring_radii_m else 500.0
    if start_vertiport is not None and takeoff_heading_deg is not None:
        to_lon, to_lat = _build_sector_wedge_lonlat(
            start_vertiport,
            takeoff_heading_deg,
            sector_half_width_deg,
            wedge_radius_m,
        )
        ax.fill(
            to_lon, to_lat,
            color="royalblue", alpha=0.24,
            edgecolor="navy", linewidth=0.8,
            transform=ccrs.PlateCarree(), zorder=6,
            label=(
                f"Takeoff Sector S{int(takeoff_sector_user)}"
                if takeoff_sector_user is not None else "Takeoff Sector"
            ),
        )
    if end_vertiport is not None and landing_heading_deg is not None:
        ld_lon, ld_lat = _build_sector_wedge_lonlat(
            end_vertiport,
            landing_heading_deg,
            sector_half_width_deg,
            wedge_radius_m,
        )
        ax.fill(
            ld_lon, ld_lat,
            color="seagreen", alpha=0.24,
            edgecolor="darkgreen", linewidth=0.8,
            transform=ccrs.PlateCarree(), zorder=6,
            label=(
                f"Landing Sector S{int(landing_sector_user)}"
                if landing_sector_user is not None else "Landing Sector"
            ),
        )

    # Optional MOC overlay: 1-cells indicate obstacle-risk regions to avoid.
    if moc_binary_2d is not None and np.size(moc_binary_2d) > 0:
        plot_moc_binary_overlay(
            ax,
            moc_binary_2d,
            lat_lim,
            lon_lim,
            label="MOC=1 (Corridor-Prohibited)",
            fill_color="magenta",
            fill_alpha=0.24,
        )

    if takeoff_complete is not None:
        if (
            bool(use_takeoff_landing_transition)
            and transition_structure_mode == TRANSITION_STRUCTURE_FIXED_ONLY
        ):
            ax.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90,
                       c=TAKEOFF_TRANSITION_COLOR, marker="^", transform=ccrs.Geodetic(),
                       zorder=10, label="Takeoff Transition End")
        elif (
            bool(use_takeoff_landing_transition)
            and bool(use_two_stage_transition)
            and start_vertiport is not None
            and _seg_dist_m(start_vertiport, takeoff_complete) > 0.5
        ):
            if takeoff_optimized_transition_actual is not False:
                ax.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90, facecolors="none",
                           edgecolors=TAKEOFF_TRANSITION_COLOR, linewidths=1.4, marker="o",
                           transform=ccrs.Geodetic(), zorder=10, label="Takeoff Stage1 End")
            else:
                ax.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90,
                           c=TAKEOFF_TRANSITION_COLOR, edgecolors="k", linewidths=0.7, marker="^",
                           transform=ccrs.Geodetic(), zorder=10, label="Takeoff Transition End")
        elif not bool(use_takeoff_landing_transition):
            ax.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90,
                       c=TAKEOFF_TRANSITION_COLOR, marker="^", transform=ccrs.Geodetic(),
                       zorder=10, label="Takeoff_End")
    if landing_entry is not None:
        if (
            bool(use_takeoff_landing_transition)
            and transition_structure_mode == TRANSITION_STRUCTURE_FIXED_ONLY
        ):
            ax.scatter([landing_entry[1]], [landing_entry[0]], s=90,
                       c=LANDING_TRANSITION_COLOR, marker="v", transform=ccrs.Geodetic(),
                       zorder=10, label="Landing Transition Start")
        elif (
            bool(use_takeoff_landing_transition)
            and bool(use_two_stage_transition)
            and end_vertiport is not None
            and _seg_dist_m(end_vertiport, landing_entry) > 0.5
        ):
            if landing_optimized_transition_actual is not False:
                ax.scatter([landing_entry[1]], [landing_entry[0]], s=90, facecolors="none",
                           edgecolors=LANDING_TRANSITION_COLOR, linewidths=1.4, marker="o",
                           transform=ccrs.Geodetic(), zorder=10, label="Landing Stage1 Start")
            else:
                ax.scatter([landing_entry[1]], [landing_entry[0]], s=90,
                           c=LANDING_TRANSITION_COLOR, edgecolors="k", linewidths=0.7, marker="v",
                           transform=ccrs.Geodetic(), zorder=10, label="Landing Transition Start")
        elif not bool(use_takeoff_landing_transition):
            ax.scatter([landing_entry[1]], [landing_entry[0]], s=90,
                       c=LANDING_TRANSITION_COLOR, marker="v", transform=ccrs.Geodetic(),
                       zorder=10, label="Landing_End")

    if start_vertiport is not None:
        ax.scatter([start_vertiport[1]], [start_vertiport[0]], s=120, c="red",
                   edgecolors="k", marker="s", transform=ccrs.Geodetic(),
                   zorder=10, label="Start Vertiport")
    if end_vertiport is not None:
        ax.scatter([end_vertiport[1]], [end_vertiport[0]], s=120, c="crimson",
                   edgecolors="k", marker="D", transform=ccrs.Geodetic(),
                   zorder=10, label="End Vertiport")

    if emergency_points is not None:
        em = np.asarray(emergency_points, dtype=float)
        if em.size > 0:
            ax.scatter(em[:, 1], em[:, 0], s=75, c="lime", edgecolors="k",
                       marker="P", transform=ccrs.Geodetic(), zorder=9,
                       label="Emergency Landing")

    if forbidden_zones is not None:
        fz = np.asarray(forbidden_zones, dtype=float)
        if fz.size > 0:
            for zi, rect in enumerate(fz):
                lon_min, lon_max, lat_min, lat_max = [float(v) for v in rect]
                poly = np.array([
                    [lon_min, lat_min],
                    [lon_max, lat_min],
                    [lon_max, lat_max],
                    [lon_min, lat_max],
                    [lon_min, lat_min],
                ], dtype=float)
                ax.plot(
                    poly[:, 0], poly[:, 1],
                    "-", color="red", linewidth=1.4,
                    transform=ccrs.Geodetic(), zorder=9,
                    label=("No-Fly Zone" if zi == 0 else None),
                )
                ax.fill(
                    poly[:, 0], poly[:, 1],
                    color="red", alpha=0.12,
                    transform=ccrs.Geodetic(), zorder=8,
                )

    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, framealpha=0.9)

    clicked_latlon = []
    click_markers = []
    click_texts = []

    def _redraw_clicks():
        while click_markers:
            click_markers.pop().remove()
        while click_texts:
            click_texts.pop().remove()
        if clicked_latlon:
            arr = np.array(clicked_latlon, dtype=float)
            mk = ax.scatter(arr[:, 1], arr[:, 0], s=60, c="orange", edgecolors="k",
                            linewidths=0.5, marker="o", transform=ccrs.Geodetic(), zorder=11)
            click_markers.append(mk)
            if arr.shape[0] >= 2:
                ln = ax.plot(arr[:, 1], arr[:, 0], "-", color="orange", linewidth=1.1,
                             transform=ccrs.Geodetic(), zorder=10)[0]
                click_markers.append(ln)
            for i, p in enumerate(arr, start=1):
                tx = ax.text(p[1], p[0], f"{i}", color="black", fontsize=8,
                             transform=ccrs.Geodetic(), zorder=12)
                click_texts.append(tx)
        fig.canvas.draw_idle()

    def _onclick(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return

        # event x/y are in map projection; convert to lon/lat
        lon, lat = ccrs.PlateCarree().transform_point(event.xdata, event.ydata, ax.projection)

        if event.button == 1:
            if click_lat_min <= lat <= click_lat_max and click_lon_min <= lon <= click_lon_max:
                clicked_latlon.append([float(lat), float(lon)])
                print(f"[WP click] #{len(clicked_latlon)}  lat={lat:.7f}, lon={lon:.7f}")
                _redraw_clicks()
            else:
                print("[WP click ignored] outside current click-map extent")
        elif event.button == 3:
            if clicked_latlon:
                removed = clicked_latlon.pop()
                print(f"[WP remove] lat={removed[0]:.7f}, lon={removed[1]:.7f}")
                _redraw_clicks()

    def _onkey(event):
        if event.key in ("enter", "return"):
            plt.close(fig)
        elif event.key in ("backspace", "delete"):
            if clicked_latlon:
                removed = clicked_latlon.pop()
                print(f"[WP remove] lat={removed[0]:.7f}, lon={removed[1]:.7f}")
                _redraw_clicks()

    fig.canvas.mpl_connect("button_press_event", _onclick)
    fig.canvas.mpl_connect("key_press_event", _onkey)
    plt.show()

    # Cleanup: explicitly destroy Tk figure manager/window when available.
    try:
        manager = getattr(fig.canvas, "manager", None)
        if manager is not None:
            manager.destroy()
    except Exception:
        pass
    try:
        plt.close(fig)
    except Exception:
        pass
    cleanup_matplotlib_tk()
    try:
        plt.switch_backend("Agg")
    except Exception:
        pass
    
    return np.array(clicked_latlon, dtype=float)


def _validate_objective_weights_v1(objective_weights, objective_count=None):
    """Validate non-negative preference weights and return normalized weights."""
    weights = np.asarray(objective_weights, dtype=float).reshape(-1)
    if objective_count is not None and weights.size != int(objective_count):
        raise ValueError(
            "objective_weight_length_mismatch: "
            f"objectives={int(objective_count)}, weights={weights.size}"
        )
    if weights.size == 0:
        raise ValueError("invalid_objective_weights: at least one active weight is required")
    if not np.all(np.isfinite(weights)):
        raise ValueError("invalid_objective_weights: every active weight must be finite")
    if np.any(weights < 0.0):
        raise ValueError("invalid_objective_weights: every active weight must be >= 0")
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("invalid_objective_weights: active weights must not all be zero")
    return weights, weights / weight_sum


def _weighted_normalized_objective_analysis_v1(f_vals, objective_weights):
    """Return per-objective min-max values, contributions, and weighted score J."""
    values = np.asarray(f_vals, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("invalid_objective_values: expected a non-empty N x M array")
    if not np.all(np.isfinite(values)):
        raise ValueError("invalid_objective_values: all values must be finite")
    configured_weights, normalized_weights = _validate_objective_weights_v1(
        objective_weights, values.shape[1]
    )
    minimum = np.min(values, axis=0)
    maximum = np.max(values, axis=0)
    value_range = maximum - minimum
    normalized = np.zeros_like(values, dtype=float)
    varying = value_range >= 1e-10
    if np.any(varying):
        normalized[:, varying] = (
            values[:, varying] - minimum[varying]
        ) / value_range[varying]
    contributions = normalized * normalized_weights[np.newaxis, :]
    scores = np.sum(contributions, axis=1)
    return {
        "configured_weights": configured_weights,
        "normalized_weights": normalized_weights,
        "minimum": minimum,
        "maximum": maximum,
        "range": value_range,
        "normalized_values": normalized,
        "weighted_contributions": contributions,
        "scores": scores,
    }


def selection_nsga3(population, f_vals, feasible, N, objective_weights):
    """Pareto-front selection with weighted preference on the truncated front."""
    fronts = fast_non_dominated_sort(f_vals)
    next_idx = []
    for front in fronts:
        valid = [i for i in front if feasible[i]]
        if not valid:
            continue
        if len(next_idx) + len(valid) <= N:
            next_idx.extend(valid)
        else:
            rem = N - len(next_idx)
            lf = np.array(valid, dtype=int)
            if lf.size > 0 and rem > 0:
                preference = _weighted_normalized_objective_analysis_v1(
                    np.asarray(f_vals, dtype=float)[lf], objective_weights
                )
                # Stable index tie-break keeps repeated runs deterministic when
                # two candidates have the same normalized weighted score.
                order = np.lexsort((lf, preference["scores"]))
                next_idx.extend(lf[order[:rem]].tolist())
            break
    return [population[i] for i in next_idx[:N]]


def variation_nsga3(pop, nodes, ratio, node_risks=None, mutation_cfg=None):
    if not pop:
        return []
    n_off = int(round(len(pop) * ratio))
    n = len(pop)
    offspring = []
    cfg = mutation_cfg if mutation_cfg is not None else {}
    for _ in range(n_off):
        i1, i2 = (np.random.choice(n, 2, replace=False) if n >= 2 else (0, 0))
        child = crossover_gp(pop[i1], pop[i2])
        child = mutation_gp(child, nodes, node_risks=node_risks, **cfg)
        offspring.append(child)
    return offspring


def _solution_signature(sol, decimals=7):
    """Hashable signature for one solution path (for diversity logging only)."""
    arr = np.asarray(sol, dtype=float)
    if arr.size == 0:
        return (0, 0, b"")
    arr_q = np.round(arr, decimals=decimals)
    return (int(arr_q.shape[0]), int(arr_q.shape[1]), arr_q.tobytes())


def _unique_solution_count(population, decimals=7):
    if not population:
        return 0
    return len({_solution_signature(sol, decimals=decimals) for sol in population})


def _seg_dist_m(a, b):
    """lat/lon 두 점 사이 수평 거리(미터)."""
    m_lat = 111000.0
    m_lon = 111000.0 * np.cos(np.deg2rad(float(0.5 * (a[0] + b[0]))))
    return float(np.sqrt(((b[0] - a[0]) * m_lat) ** 2 + ((b[1] - a[1]) * m_lon) ** 2))


def _seg_dist_3d_m(a, b):
    d2 = _seg_dist_m(a, b)
    dz = float(b[2] - a[2])
    return float(np.sqrt(d2 * d2 + dz * dz))


def _path_total_3d_distance_m(path):
    if path is None or path.shape[0] < 2:
        return 0.0
    total = 0.0
    for i in range(1, path.shape[0]):
        total += _seg_dist_3d_m(path[i - 1], path[i])
    return float(total)


def _evaluate_objectives_altitude_aware_v1(
    path,
    Norm_RT,
    AirRisk,
    use_heading_map,
    altitude_levels,
    cell_size,
    refine_scales,
    air_risk_threshold,
    w_dist,
    w_ground,
    w_air,
    lat_lim,
    lon_lim,
    NoiseRisk=None,
    noise_floor_db=0.0,
    w_noise=1.0,
):
    """Return raw objectives; user weights are applied once during preference selection."""
    p = np.asarray(path, dtype=float).reshape(-1, 3)
    levels = np.asarray(altitude_levels, dtype=float).ravel()
    active_weights = [w_dist, w_ground, w_air]
    if NoiseRisk is not None and np.size(NoiseRisk) > 0:
        active_weights.append(w_noise)
    _validate_objective_weights_v1(active_weights, len(active_weights))
    if p.shape[0] < 2 or levels.size == 0:
        n_obj = 4 if NoiseRisk is not None and np.size(NoiseRisk) > 0 else 3
        return np.full(n_obj, 1e6, dtype=float)

    total_dist = float(_path_total_3d_distance_m(p))
    total_ground = 0.0
    total_air = 0.0
    total_noise = 0.0
    _, _, Ny, Nx = np.asarray(Norm_RT).shape
    min_lat, max_lat = [float(v) for v in lat_lim]
    min_lon, max_lon = [float(v) for v in lon_lim]
    d_lat = (max_lat - min_lat) / (Ny - 1) if Ny > 1 else 1.0
    d_lon = (max_lon - min_lon) / (Nx - 1) if Nx > 1 else 1.0
    noise = None if NoiseRisk is None or np.size(NoiseRisk) == 0 else np.asarray(NoiseRisk, dtype=float)
    if noise is not None and noise.ndim == 2:
        noise = noise[:, :, np.newaxis]

    for i in range(p.shape[0] - 1):
        p1 = p[i]
        p2 = p[i + 1]
        dist_2d_m = _seg_dist_m(p1, p2)
        if dist_2d_m <= 1e-9:
            continue

        vec = p2[:2] - p1[:2]
        if bool(use_heading_map):
            theta = float(np.rad2deg(np.arctan2(vec[1], vec[0])))
            if theta < 0.0:
                theta += 360.0
            heading_idx = int(round(theta / 45.0) % 8)
        else:
            heading_idx = 0

        if dist_2d_m < 200.0:
            refine_scale = float(refine_scales[3])
        elif dist_2d_m < 500.0:
            refine_scale = float(refine_scales[2])
        elif dist_2d_m < 1000.0:
            refine_scale = float(refine_scales[1])
        else:
            refine_scale = float(refine_scales[0])
        n_samples = max(2, int(np.ceil(dist_2d_m / (float(cell_size) * refine_scale))))

        tau = np.linspace(0.0, 1.0, n_samples)
        sample_lat = p1[0] + tau * (p2[0] - p1[0])
        sample_lon = p1[1] + tau * (p2[1] - p1[1])
        sample_alt = p1[2] + tau * (p2[2] - p1[2])
        sample_i = (sample_lon - min_lon) / d_lon
        sample_j = (sample_lat - min_lat) / d_lat
        coords = np.vstack([sample_j, sample_i])
        altitude_idx = np.argmin(np.abs(sample_alt[:, None] - levels[None, :]), axis=1)

        segment_ground = np.zeros(n_samples, dtype=float)
        segment_air = np.zeros(n_samples, dtype=float)
        segment_noise = np.zeros(n_samples, dtype=float)
        for alt_idx in np.unique(altitude_idx):
            mask = altitude_idx == alt_idx
            coords_i = coords[:, mask]
            segment_ground[mask] = map_coordinates(
                Norm_RT[int(alt_idx), heading_idx], coords_i, order=1, cval=0.0
            )
            segment_air[mask] = map_coordinates(
                AirRisk[:, :, int(alt_idx)], coords_i, order=1, cval=0.0
            )
            if noise is not None:
                noise_idx = int(np.clip(int(alt_idx), 0, noise.shape[2] - 1))
                segment_noise[mask] = map_coordinates(
                    noise[:, :, noise_idx], coords_i, order=1, cval=0.0
                )

        total_ground += float(np.sum(segment_ground))
        total_air += float(np.sum(segment_air))
        if noise is not None:
            # The loader applies the dB floor before normalizing to [0, 1].
            # Do not compare normalized values against a dB threshold again.
            active_noise = np.where(segment_noise > 0.0, segment_noise, 0.0)
            total_noise += float(np.sum(active_noise))

    values = [
        total_dist,
        total_ground,
        total_air,
    ]
    if noise is not None:
        values.append(total_noise)
    return np.asarray(values, dtype=float)


def _moc_floor_layer_index_v1(altitude_msl_m, layer_count):
    agl_m = float(altitude_msl_m) - float(MOC_REFERENCE_MSL_M)
    # Treat a numerically exact layer boundary as that layer.  This keeps an
    # exact clearance-limit contact feasible instead of selecting the layer
    # below it because of sub-micrometre floating-point drift.
    idx = int(np.searchsorted(MOC_AGL_LEVELS_M, agl_m + 1e-6, side="right") - 1)
    return int(np.clip(idx, 0, max(0, int(layer_count) - 1)))


def _validate_transition_corridor_cfg_v1(transition_corridor_cfg):
    cfg = transition_corridor_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return cfg
    for key in ("half_width_m", "downward_clearance_m"):
        value = cfg.get(key)
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = float("nan")
        if not np.isfinite(numeric_value) or numeric_value <= 0.0:
            raise ValueError(f"invalid_transition_corridor_{key}: must be finite and > 0")
    if abs(float(cfg["half_width_m"]) - float(cfg["downward_clearance_m"])) > 1e-9:
        raise ValueError(
            "transition_corridor_width_clearance_mismatch: horizontal half-width "
            "and downward clearance must use the same parameter"
        )
    for key in ("start_vertiport_msl_m", "end_vertiport_msl_m"):
        value = cfg.get(key)
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = float("nan")
        if not np.isfinite(numeric_value):
            raise ValueError(f"invalid_transition_corridor_{key}: must be finite")
    return cfg


def _edge_phase_values_v1(path, flight_phases, transition_corridor_cfg=None):
    """Return one phase per edge using the exported destination-phase rule."""
    points = np.asarray(path, dtype=float).reshape(-1, 3)
    cfg = _validate_transition_corridor_cfg_v1(transition_corridor_cfg)
    enabled = bool(cfg.get("enabled", False))

    if flight_phases is None:
        if enabled:
            raise ValueError(
                "flight_phase_length_mismatch: "
                f"path_points={points.shape[0]}, flight_phases=0."
            )
        return np.full(max(0, points.shape[0] - 1), FLIGHT_PHASE_CRUISE, dtype=object)

    phases = np.asarray(flight_phases, dtype=object).reshape(-1)
    if phases.size != points.shape[0]:
        raise ValueError(
            "flight_phase_length_mismatch: "
            f"path_points={points.shape[0]}, flight_phases={phases.size}."
        )
    if not enabled:
        return np.full(max(0, points.shape[0] - 1), FLIGHT_PHASE_CRUISE, dtype=object)

    valid_phases = set(TRANSITION_PHASE_NAMES) | {
        FLIGHT_PHASE_CRUISE,
        FLIGHT_PHASE_VERTIPORT,
    }
    invalid_phases = sorted({str(value) for value in phases if str(value) not in valid_phases})
    if invalid_phases:
        raise ValueError(
            "invalid_flight_phase: " + ", ".join(invalid_phases)
        )
    vertiport_indices = np.flatnonzero(phases == FLIGHT_PHASE_VERTIPORT)
    if any(int(idx) not in (0, points.shape[0] - 1) for idx in vertiport_indices):
        raise ValueError("invalid_flight_phase: vertiport is allowed only at path endpoints")
    edge_phases = phases[1:].copy()
    vertiport_edges = edge_phases == FLIGHT_PHASE_VERTIPORT
    edge_phases[vertiport_edges] = phases[:-1][vertiport_edges]
    return edge_phases.astype(object)


def _transition_direction_from_phase_value_v1(phase):
    phase_text = str(phase)
    if phase_text.startswith("takeoff_"):
        return "takeoff"
    if phase_text.startswith("landing_"):
        return "landing"
    return None


def _edge_corridor_half_width_v1(phase, cruise_half_width_m, transition_corridor_cfg=None):
    cfg = _validate_transition_corridor_cfg_v1(transition_corridor_cfg)
    if bool(cfg.get("enabled", False)) and str(phase) in TRANSITION_PHASE_NAMES:
        return float(cfg["half_width_m"])
    return float(cruise_half_width_m)


def _transition_reference_msl_v1(phase, transition_corridor_cfg=None):
    cfg = _validate_transition_corridor_cfg_v1(transition_corridor_cfg)
    direction = _transition_direction_from_phase_value_v1(phase)
    if direction == "takeoff":
        return float(cfg["start_vertiport_msl_m"])
    if direction == "landing":
        return float(cfg["end_vertiport_msl_m"])
    return None


def _effective_downward_clearance_v1(
    center_msl_m,
    phase,
    transition_corridor_cfg=None,
):
    cfg = _validate_transition_corridor_cfg_v1(transition_corridor_cfg)
    if not bool(cfg.get("enabled", False)) or str(phase) not in TRANSITION_PHASE_NAMES:
        return 0.0
    configured = float(cfg["downward_clearance_m"])
    reference_msl = _transition_reference_msl_v1(phase, cfg)
    return float(min(configured, max(0.0, float(center_msl_m) - float(reference_msl))))


def _iter_corridor_moc_samples_v1(
    p1,
    p2,
    half_width_m,
    moc_risk,
    lat_lim,
    lon_lim,
    along_step_m=80.0,
    phase=FLIGHT_PHASE_CRUISE,
    transition_corridor_cfg=None,
):
    """Yield the exact along/cross-track samples used by the v1 MOC checker."""
    if moc_risk is None or np.size(moc_risk) == 0:
        return
    cfg = _validate_transition_corridor_cfg_v1(transition_corridor_cfg)
    phase = str(phase)
    valid_edge_phases = set(TRANSITION_PHASE_NAMES) | {FLIGHT_PHASE_CRUISE}
    if bool(cfg.get("enabled", False)) and phase not in valid_edge_phases:
        raise ValueError(f"invalid_flight_phase: {phase}")
    is_transition = bool(
        cfg.get("enabled", False) and phase in TRANSITION_PHASE_NAMES
    )
    try:
        sampling_half_width_m = float(half_width_m)
    except (TypeError, ValueError):
        sampling_half_width_m = float("nan")
    if is_transition and (
        not np.isfinite(sampling_half_width_m)
        or sampling_half_width_m <= 0.0
        or abs(sampling_half_width_m - float(cfg["half_width_m"])) > 1e-9
    ):
        raise ValueError(
            "transition_corridor_half_width_mismatch: MOC sampling width must "
            "match transition_corridor_half_width_m"
        )
    configured_clearance_m = (
        float(cfg["downward_clearance_m"]) if is_transition else 0.0
    )
    reference_msl_m = (
        float(_transition_reference_msl_v1(phase, cfg)) if is_transition else None
    )
    moc = np.asarray(moc_risk, dtype=float)
    if moc.ndim == 2:
        moc = moc[:, :, np.newaxis]
    Ny, Nx, Nz = moc.shape
    min_lat, max_lat = [float(v) for v in lat_lim]
    min_lon, max_lon = [float(v) for v in lon_lim]
    d_lat = (max_lat - min_lat) / (Ny - 1) if Ny > 1 else 1.0
    d_lon = (max_lon - min_lon) / (Nx - 1) if Nx > 1 else 1.0

    p1 = np.asarray(p1, dtype=float).reshape(3)
    p2 = np.asarray(p2, dtype=float).reshape(3)
    mean_lat = 0.5 * (float(p1[0]) + float(p2[0]))
    m_lat = 111000.0
    m_lon = 111000.0 * np.cos(np.deg2rad(mean_lat))
    dx = (float(p2[1]) - float(p1[1])) * m_lon
    dy = (float(p2[0]) - float(p1[0])) * m_lat
    seg_len = float(np.hypot(dx, dy))
    if seg_len <= 1e-9:
        return
    ux, uy = dx / seg_len, dy / seg_len
    nx, ny = -uy, ux
    n_along = max(2, int(np.ceil(seg_len / max(10.0, float(along_step_m)))) + 1)
    s_values = np.linspace(0.0, seg_len, n_along)
    half_width = max(0.0, sampling_half_width_m)
    if half_width <= 1e-9:
        t_values = np.array([0.0], dtype=float)
    else:
        cross_step = float(np.clip(half_width / 3.0, 20.0, 100.0))
        n_cross = max(3, int(np.ceil(2.0 * half_width / cross_step)) + 1)
        t_values = np.linspace(-half_width, half_width, n_cross)

    for s in s_values:
        tau = float(s / seg_len)
        center_alt_msl = float(p1[2]) + tau * (float(p2[2]) - float(p1[2]))
        center_layer_idx = _moc_floor_layer_index_v1(center_alt_msl, Nz)
        effective_clearance_m = (
            min(
                configured_clearance_m,
                max(0.0, center_alt_msl - reference_msl_m),
            )
            if is_transition else 0.0
        )
        query_alt_msl = float(center_alt_msl - effective_clearance_m)
        layer_idx = _moc_floor_layer_index_v1(query_alt_msl, Nz)
        center_x = float(s) * ux
        center_y = float(s) * uy
        for t in t_values:
            qx = center_x + float(t) * nx
            qy = center_y + float(t) * ny
            lon = float(p1[1]) + qx / m_lon
            lat = float(p1[0]) + qy / m_lat
            grid_i = int(np.round((lon - min_lon) / d_lon))
            grid_j = int(np.round((lat - min_lat) / d_lat))
            in_grid = bool(0 <= grid_i < Nx and 0 <= grid_j < Ny)
            blocked = bool(
                in_grid and float(moc[grid_j, grid_i, layer_idx]) >= 0.5
            )
            yield (
                float(s),
                float(t),
                tau,
                lat,
                lon,
                center_alt_msl,
                int(center_layer_idx),
                float(effective_clearance_m),
                float(query_alt_msl),
                int(layer_idx),
                int(grid_j),
                int(grid_i),
                in_grid,
                blocked,
            )


def _corridor_hits_moc_v1(
    p1,
    p2,
    half_width_m,
    moc_risk,
    lat_lim,
    lon_lim,
    along_step_m=80.0,
    phase=FLIGHT_PHASE_CRUISE,
    transition_corridor_cfg=None,
):
    """MOC footprint check using conservative fixed-AGL floor selection."""
    for sample in _iter_corridor_moc_samples_v1(
        p1,
        p2,
        half_width_m,
        moc_risk,
        lat_lim,
        lon_lim,
        along_step_m=along_step_m,
        phase=phase,
        transition_corridor_cfg=transition_corridor_cfg,
    ):
        if bool(sample[-1]):
            return True
    return False


def _phase_specific_nfz_hit_v1(
    path,
    edge_phases,
    cruise_half_width_m,
    transition_corridor_cfg,
    forbidden_zones,
    direction_filter=None,
):
    points = np.asarray(path, dtype=float).reshape(-1, 3)
    if forbidden_zones is None or np.size(forbidden_zones) == 0:
        return True, "ok"
    for edge_idx in range(points.shape[0] - 1):
        phase = str(edge_phases[edge_idx])
        direction = _transition_direction_from_phase_value_v1(phase)
        if direction_filter is not None and direction != str(direction_filter):
            continue
        half_width_m = _edge_corridor_half_width_v1(
            phase,
            cruise_half_width_m,
            transition_corridor_cfg,
        )
        if half_width_m <= 0.0:
            continue
        for rect in forbidden_zones:
            if _corridor_violates_nfz_with_width_shared(
                points[edge_idx],
                points[edge_idx + 1],
                half_width_m,
                rect,
            ):
                if direction is not None:
                    return False, f"{direction}_transition_nfz_corridor_width_intersection"
                return False, "nfz_corridor_width_intersection"
    return True, "ok"


def _phase_specific_nfz_centerline_reason_v1(
    path,
    edge_phases,
    forbidden_zones,
):
    """Return the phase-aware reason for the first actual centerline NFZ hit."""
    points = np.asarray(path, dtype=float).reshape(-1, 3)
    if forbidden_zones is None or np.size(forbidden_zones) == 0:
        return "ok"
    for edge_idx in range(points.shape[0] - 1):
        phase = str(edge_phases[edge_idx])
        direction = _transition_direction_from_phase_value_v1(phase)
        for rect in forbidden_zones:
            if _corridor_violates_nfz_with_width_shared(
                points[edge_idx], points[edge_idx + 1], 0.0, rect
            ):
                if direction is not None:
                    return f"{direction}_transition_nfz_centerline_intersection"
                return "nfz_centerline_intersection"
    return "ok"


def _phase_specific_self_overlap_v1(
    path,
    edge_phases,
    cruise_half_width_m,
    transition_corridor_cfg,
    eps_m=1.0,
    direction_filter=None,
):
    points = np.asarray(path, dtype=float).reshape(-1, 3)
    if points.shape[0] < 4:
        return False, "ok"
    mean_lat = float(np.mean(points[:, 0]))
    m_lat = 111000.0
    m_lon = 111000.0 * np.cos(np.deg2rad(mean_lat))
    xy = np.column_stack([points[:, 1] * m_lon, points[:, 0] * m_lat])
    widths = np.asarray([
        _edge_corridor_half_width_v1(
            phase,
            cruise_half_width_m,
            transition_corridor_cfg,
        )
        for phase in edge_phases
    ], dtype=float)
    for i in range(points.shape[0] - 1):
        for j in range(i + 2, points.shape[0] - 1):
            threshold_m = float(widths[i] + widths[j] - max(0.0, float(eps_m)))
            if threshold_m <= 0.0:
                continue
            distance_m = _segment_to_segment_min_distance_m_shared(
                xy[i], xy[i + 1], xy[j], xy[j + 1]
            )
            if distance_m < threshold_m:
                directions = {
                    value
                    for value in (
                        _transition_direction_from_phase_value_v1(edge_phases[i]),
                        _transition_direction_from_phase_value_v1(edge_phases[j]),
                    )
                    if value is not None
                }
                if (
                    direction_filter is not None
                    and str(direction_filter) not in directions
                ):
                    continue
                if directions == {"takeoff"}:
                    reason = "takeoff_transition_self_corridor_width_overlap"
                elif directions == {"landing"}:
                    reason = "landing_transition_self_corridor_width_overlap"
                elif directions == {"takeoff", "landing"}:
                    reason = "takeoff_landing_transition_self_corridor_width_overlap"
                else:
                    reason = "self_corridor_width_overlap"
                return True, reason
    return False, "ok"


def evaluate_objectives_with_constraints_gp(
    path,
    Norm_RT,
    AirRisk,
    use_heading_map,
    flight_dist_limit,
    forbidden_zones,
    delta_z_max,
    altitude_levels,
    cell_size,
    refine_scales,
    air_risk_threshold,
    w_dist,
    w_ground,
    w_air,
    lat_lim,
    lon_lim,
    NoiseRisk=None,
    noise_floor_db=0.0,
    w_noise=1.0,
    W_half=None,
    check_corridor_nfz=False,
    MOCRisk=None,
    check_corridor_moc=False,
    check_corridor_self_overlap=True,
    vertiport=None,
    landing_entry=None,
    takeoff_complete=None,
    return_reason=False,
    flight_phases=None,
    transition_corridor_cfg=None,
):
    """v1-only wrapper: shared geometric checks plus altitude-aware objectives/MOC."""
    transition_width_enabled = bool(
        (transition_corridor_cfg or {}).get("enabled", False)
    )
    cruise_half_width_m = float(W_half) if W_half is not None else 0.0
    shared_result = _evaluate_constraints_shared(
        path,
        Norm_RT,
        AirRisk,
        use_heading_map,
        flight_dist_limit,
        forbidden_zones,
        delta_z_max,
        altitude_levels,
        1.0e12,
        np.ones(4, dtype=float),
        air_risk_threshold,
        w_dist,
        w_ground,
        w_air,
        lat_lim,
        lon_lim,
        NoiseRisk=NoiseRisk,
        noise_floor_db=noise_floor_db,
        w_noise=w_noise,
        W_half=W_half,
        check_corridor_nfz=(bool(check_corridor_nfz) and not transition_width_enabled),
        MOCRisk=None,
        check_corridor_moc=False,
        check_corridor_self_overlap=(
            bool(check_corridor_self_overlap) and not transition_width_enabled
        ),
        vertiport=vertiport,
        landing_entry=landing_entry,
        takeoff_complete=takeoff_complete,
        return_reason=True,
    )
    _, shared_ok, shared_reason = shared_result
    objective_values = _evaluate_objectives_altitude_aware_v1(
        path,
        Norm_RT,
        AirRisk,
        use_heading_map,
        altitude_levels,
        cell_size,
        refine_scales,
        air_risk_threshold,
        w_dist,
        w_ground,
        w_air,
        lat_lim,
        lon_lim,
        NoiseRisk=NoiseRisk,
        noise_floor_db=noise_floor_db,
        w_noise=w_noise,
    )

    ok = bool(shared_ok)
    reason = str(shared_reason)
    check_path = np.asarray(path, dtype=float).reshape(-1, 3)
    try:
        edge_phases = _edge_phase_values_v1(
            check_path,
            flight_phases,
            transition_corridor_cfg,
        )
    except ValueError as exc:
        edge_phases = np.empty((0,), dtype=object)
        ok = False
        reason = str(exc).split(":", 1)[0]

    if (
        not ok
        and transition_width_enabled
        and reason == "nfz_centerline_intersection"
        and edge_phases.size == max(0, check_path.shape[0] - 1)
    ):
        phase_nfz_reason = _phase_specific_nfz_centerline_reason_v1(
            check_path,
            edge_phases,
            forbidden_zones,
        )
        if phase_nfz_reason != "ok":
            reason = phase_nfz_reason

    if (
        ok and transition_width_enabled and bool(check_corridor_nfz)
    ):
        ok, reason = _phase_specific_nfz_hit_v1(
            check_path,
            edge_phases,
            cruise_half_width_m,
            transition_corridor_cfg,
            forbidden_zones,
        )

    if ok and bool(check_corridor_moc) and (
        transition_width_enabled or cruise_half_width_m > 0.0
    ):
        for i in range(check_path.shape[0] - 1):
            phase = str(edge_phases[i])
            half_width_m = _edge_corridor_half_width_v1(
                phase,
                cruise_half_width_m,
                transition_corridor_cfg,
            )
            if half_width_m <= 0.0:
                continue
            if _corridor_hits_moc_v1(
                check_path[i],
                check_path[i + 1],
                half_width_m,
                MOCRisk,
                lat_lim,
                lon_lim,
                phase=phase,
                transition_corridor_cfg=transition_corridor_cfg,
            ):
                ok = False
                direction = _transition_direction_from_phase_value_v1(phase)
                reason = (
                    f"{direction}_transition_moc_3d_intersection"
                    if direction is not None else "moc_corridor_width_intersection"
                )
                break

    if (
        ok and transition_width_enabled and bool(check_corridor_self_overlap)
    ):
        overlap, overlap_reason = _phase_specific_self_overlap_v1(
            check_path,
            edge_phases,
            cruise_half_width_m,
            transition_corridor_cfg,
        )
        if overlap:
            ok = False
            reason = overlap_reason

    if not ok:
        objective_values = np.full(objective_values.shape, 1e6, dtype=float)
    if return_reason:
        return objective_values, ok, reason
    return objective_values, ok


def _enforce_mandatory_wp_order(path, mandatory_backbone, xy_tol_m=5.0):
    """
    Keep mandatory waypoints in fixed order and reinsert optional nodes by segment projection.
    This does not modify crossover/mutation logic; it repairs path order in main pipeline.
    """
    mandatory = np.asarray(mandatory_backbone, dtype=float)
    if mandatory.ndim != 2 or mandatory.shape[0] < 2:
        return np.asarray(path, dtype=float)

    p = np.asarray(path, dtype=float)
    if p.ndim != 2 or p.shape[0] == 0:
        return mandatory.copy()

    ref_lat = float(np.mean(mandatory[:, 0]))
    m_lat = 111000.0
    m_lon = 111000.0 * np.cos(np.deg2rad(ref_lat))

    def _xy(arr):
        return np.column_stack([arr[:, 1] * m_lon, arr[:, 0] * m_lat]).astype(float)

    man_xy = _xy(mandatory[:, :2])
    p_xy = _xy(p[:, :2])

    # Filter out points that are effectively mandatory points.
    extras = []
    xy_tol = float(max(0.0, xy_tol_m))
    for i in range(p.shape[0]):
        d = np.linalg.norm(man_xy - p_xy[i][None, :], axis=1)
        if float(np.min(d)) > xy_tol:
            extras.append(p[i].astype(float))

    per_seg = [[] for _ in range(mandatory.shape[0] - 1)]
    for q in extras:
        q_xy = np.array([q[1] * m_lon, q[0] * m_lat], dtype=float)
        best_k = 0
        best_cost = np.inf
        best_t = 0.0

        for k in range(mandatory.shape[0] - 1):
            a = man_xy[k]
            b = man_xy[k + 1]
            v = b - a
            vv = float(np.dot(v, v))
            if vv < 1e-12:
                continue
            t = float(np.dot(q_xy - a, v) / vv)
            t_clip = float(np.clip(t, 0.0, 1.0))
            proj = a + t_clip * v
            d_perp = float(np.linalg.norm(q_xy - proj))
            outside_penalty = 0.0
            if t < 0.0:
                outside_penalty = -t
            elif t > 1.0:
                outside_penalty = t - 1.0
            cost = d_perp + 1000.0 * outside_penalty
            if cost < best_cost:
                best_cost = cost
                best_k = k
                best_t = t_clip

        per_seg[best_k].append((best_t, q))

    rebuilt = [mandatory[0].astype(float)]
    for k in range(mandatory.shape[0] - 1):
        if per_seg[k]:
            per_seg[k].sort(key=lambda x: x[0])
            for _, q in per_seg[k]:
                rebuilt.append(q)
        rebuilt.append(mandatory[k + 1].astype(float))

    return np.asarray(rebuilt, dtype=float)


def _sample_point_risks(path, Norm_RT, AirRisk, altitude_levels,
                        use_heading_map, air_risk_threshold, lat_lim, lon_lim):
    """
    경로 각 점의 Ground/Air/Combined risk를 샘플링한다.
    주의: 현재 path 고도는 MSL 기준으로 처리한다.
    """
    if path is None or path.shape[0] == 0:
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float), np.empty((0,), dtype=float)

    Ny, Nx = AirRisk.shape[0], AirRisk.shape[1]
    minLat, maxLat = lat_lim
    minLon, maxLon = lon_lim
    dLat_deg = (maxLat - minLat) / (Ny - 1) if Ny > 1 else 1.0
    dLon_deg = (maxLon - minLon) / (Nx - 1) if Nx > 1 else 1.0

    p = np.asarray(path, dtype=float)
    n = p.shape[0]
    g = np.zeros(n, dtype=float)
    a = np.zeros(n, dtype=float)
    c = np.zeros(n, dtype=float)

    for i in range(n):
        if n == 1:
            vec = np.array([0.0, 1.0], dtype=float)
        elif i < n - 1:
            vec = p[i + 1, :2] - p[i, :2]
        else:
            vec = p[i, :2] - p[i - 1, :2]

        if use_heading_map:
            theta = np.rad2deg(np.arctan2(vec[1], vec[0]))
            if theta < 0:
                theta += 360.0
            head_idx = int(round(theta / 45.0) % 8)
        else:
            head_idx = 0

        alt_idx = int(np.argmin(np.abs(altitude_levels - p[i, 2])))
        I = int(np.clip(round((p[i, 1] - minLon) / dLon_deg), 0, Nx - 1))
        J = int(np.clip(round((p[i, 0] - minLat) / dLat_deg), 0, Ny - 1))

        gi = float(Norm_RT[alt_idx, head_idx, J, I])
        ai = float(AirRisk[J, I, alt_idx])
        ci = (gi * ai) + (ai if ai > float(air_risk_threshold) else 0.0)

        g[i] = gi
        a[i] = ai
        c[i] = ci

    return g, a, c


def _aggregate_path_risks(path, Norm_RT, AirRisk, altitude_levels,
                          use_heading_map, cell_size, refine_scales,
                          air_risk_threshold, lat_lim, lon_lim):
    """
    경로 전체의 누적 Ground/Air/Combined risk를 계산한다.
    주의: 현재 path 고도는 MSL 기준으로 처리한다.
    """
    if path is None or path.shape[0] < 2:
        return 0.0, 0.0, 0.0

    _, _, Ny, Nx = Norm_RT.shape
    minLat, maxLat = lat_lim
    minLon, maxLon = lon_lim
    dLat_deg = (maxLat - minLat) / (Ny - 1) if Ny > 1 else 1.0
    dLon_deg = (maxLon - minLon) / (Nx - 1) if Nx > 1 else 1.0
    total_ground = 0.0
    total_air = 0.0
    total_combined = 0.0

    for i in range(path.shape[0] - 1):
        p1 = path[i, :]
        p2 = path[i + 1, :]
        vec = p2[:2] - p1[:2]

        if use_heading_map:
            theta = np.rad2deg(np.arctan2(vec[1], vec[0]))
            if theta < 0:
                theta += 360.0
            head_idx = int(round(theta / 45.0) % 8)
        else:
            head_idx = 0

        dist_2d_m = _seg_dist_m(p1, p2)
        if dist_2d_m < 1e-6:
            continue

        if dist_2d_m < 200:
            refine_scale = refine_scales[3]
        elif dist_2d_m < 500:
            refine_scale = refine_scales[2]
        elif dist_2d_m < 1000:
            refine_scale = refine_scales[1]
        else:
            refine_scale = refine_scales[0]

        num_samples = int(np.ceil(dist_2d_m / (cell_size * refine_scale)))
        if num_samples < 2:
            num_samples = 2

        yq_lat = np.linspace(p1[0], p2[0], num_samples)
        xq_lon = np.linspace(p1[1], p2[1], num_samples)
        yq_alt = np.linspace(p1[2], p2[2], num_samples)

        Iq = (xq_lon - minLon) / dLon_deg
        Jq = (yq_lat - minLat) / dLat_deg
        coords = np.vstack((Jq, Iq))

        altitude_idx = np.argmin(
            np.abs(yq_alt[:, None] - np.asarray(altitude_levels, dtype=float)[None, :]),
            axis=1,
        )
        interp_ground = np.zeros(num_samples, dtype=float)
        interp_air = np.zeros(num_samples, dtype=float)
        for alt_idx in np.unique(altitude_idx):
            mask = altitude_idx == alt_idx
            interp_ground[mask] = map_coordinates(
                Norm_RT[int(alt_idx), head_idx], coords[:, mask], order=1, cval=0.0
            )
            interp_air[mask] = map_coordinates(
                AirRisk[:, :, int(alt_idx)], coords[:, mask], order=1, cval=0.0
            )
        additive_air = np.where(interp_air > air_risk_threshold, interp_air, 0.0)
        interp_combined = (interp_ground * interp_air) + additive_air

        total_ground += float(np.sum(interp_ground))
        total_air += float(np.sum(interp_air))
        total_combined += float(np.sum(interp_combined))

    return total_ground, total_air, total_combined


def _sample_point_noise(path, NoiseMap, altitude_levels, lat_lim, lon_lim):
    """
    경로 각 점의 소음값을 샘플링한다.
    현재 NoiseMap은 고도 영향이 거의 없는 2D/준2D 형태를 기본 가정한다.
    """
    if path is None or path.shape[0] == 0 or NoiseMap is None or np.size(NoiseMap) == 0:
        return np.empty((0,), dtype=float)

    nm = np.asarray(NoiseMap, dtype=float)
    if nm.ndim == 2:
        nm = nm[:, :, np.newaxis]

    Ny, Nx = nm.shape[0], nm.shape[1]
    minLat, maxLat = lat_lim
    minLon, maxLon = lon_lim
    dLat_deg = (maxLat - minLat) / (Ny - 1) if Ny > 1 else 1.0
    dLon_deg = (maxLon - minLon) / (Nx - 1) if Nx > 1 else 1.0

    p = np.asarray(path, dtype=float)
    out = np.zeros(p.shape[0], dtype=float)
    for i in range(p.shape[0]):
        alt_idx = int(np.argmin(np.abs(altitude_levels - p[i, 2]))) if nm.shape[2] > 1 else 0
        I = int(np.clip(round((p[i, 1] - minLon) / dLon_deg), 0, Nx - 1))
        J = int(np.clip(round((p[i, 0] - minLat) / dLat_deg), 0, Ny - 1))
        out[i] = float(nm[J, I, alt_idx])
    return out


def _aggregate_path_noise(path, NoiseMap, altitude_levels, cell_size, refine_scales, lat_lim, lon_lim):
    """
    경로 전체의 누적 소음 리스크를 계산한다.
    현재 NoiseMap은 고도 영향이 거의 없는 형태를 기본 가정한다.
    """
    if path is None or path.shape[0] < 2 or NoiseMap is None or np.size(NoiseMap) == 0:
        return 0.0

    nm = np.asarray(NoiseMap, dtype=float)
    if nm.ndim == 2:
        nm = nm[:, :, np.newaxis]

    Ny, Nx = nm.shape[0], nm.shape[1]
    minLat, maxLat = lat_lim
    minLon, maxLon = lon_lim
    dLat_deg = (maxLat - minLat) / (Ny - 1) if Ny > 1 else 1.0
    dLon_deg = (maxLon - minLon) / (Nx - 1) if Nx > 1 else 1.0
    total_noise = 0.0
    for i in range(path.shape[0] - 1):
        p1 = path[i, :]
        p2 = path[i + 1, :]
        dist_2d_m = _seg_dist_m(p1, p2)
        if dist_2d_m < 1e-6:
            continue

        if dist_2d_m < 200:
            refine_scale = refine_scales[3]
        elif dist_2d_m < 500:
            refine_scale = refine_scales[2]
        elif dist_2d_m < 1000:
            refine_scale = refine_scales[1]
        else:
            refine_scale = refine_scales[0]

        num_samples = int(np.ceil(dist_2d_m / (cell_size * refine_scale)))
        if num_samples < 2:
            num_samples = 2

        yq_lat = np.linspace(p1[0], p2[0], num_samples)
        xq_lon = np.linspace(p1[1], p2[1], num_samples)
        yq_alt = np.linspace(p1[2], p2[2], num_samples)
        Iq = (xq_lon - minLon) / dLon_deg
        Jq = (yq_lat - minLat) / dLat_deg
        coords = np.vstack((Jq, Iq))

        if nm.shape[2] > 1:
            altitude_idx = np.argmin(
                np.abs(yq_alt[:, None] - np.asarray(altitude_levels, dtype=float)[None, :]),
                axis=1,
            )
        else:
            altitude_idx = np.zeros(num_samples, dtype=int)
        interp_noise = np.zeros(num_samples, dtype=float)
        for alt_idx in np.unique(altitude_idx):
            mask = altitude_idx == alt_idx
            interp_noise[mask] = map_coordinates(
                nm[:, :, int(alt_idx)], coords[:, mask], order=1, cval=0.0
            )
        total_noise += float(np.sum(interp_noise))

    return float(total_noise)


def _dist_to_center_m(points_latlon, center_latlon):
    pts = np.asarray(points_latlon, dtype=float)
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]
    c = np.asarray(center_latlon, dtype=float).ravel()
    mean_lat = float(0.5 * (np.mean(pts[:, 0]) + c[0]))
    m_lat = 111000.0
    m_lon = 111000.0 * np.cos(np.deg2rad(mean_lat))
    dlat = (pts[:, 0] - c[0]) * m_lat
    dlon = (pts[:, 1] - c[1]) * m_lon
    return np.sqrt(dlat * dlat + dlon * dlon)


def filter_nodes_in_airspace(cand, center_latlon, radius_m, alt_min_m=None, alt_max_m=None):
    if cand is None or cand.size == 0:
        return cand
    d = _dist_to_center_m(cand[:, :2], center_latlon)
    mask = d <= float(radius_m)
    if alt_min_m is not None:
        mask &= cand[:, 2] >= float(alt_min_m)
    if alt_max_m is not None:
        mask &= cand[:, 2] <= float(alt_max_m)
    return cand[mask]


def is_path_inside_airspace(path, center_latlon, radius_m, alt_min_m=None, alt_max_m=None):
    if path is None or path.size == 0:
        return False
    d = _dist_to_center_m(path[:, :2], center_latlon)
    mask = d <= float(radius_m)
    if alt_min_m is not None:
        mask &= path[:, 2] >= float(alt_min_m)
    if alt_max_m is not None:
        mask &= path[:, 2] <= float(alt_max_m)
    return bool(np.all(mask))


def _edge_lateral_boundary_points_v1(p1, p2, half_width_m):
    p1 = np.asarray(p1, dtype=float).reshape(3)
    p2 = np.asarray(p2, dtype=float).reshape(3)
    half_width = float(max(0.0, half_width_m))
    if half_width <= 1e-9:
        return np.vstack([p1[:2], p2[:2]])
    mean_lat = 0.5 * (float(p1[0]) + float(p2[0]))
    m_lat = 111000.0
    m_lon = 111000.0 * np.cos(np.deg2rad(mean_lat))
    dx = (float(p2[1]) - float(p1[1])) * m_lon
    dy = (float(p2[0]) - float(p1[0])) * m_lat
    seg_len = float(np.hypot(dx, dy))
    if seg_len <= 1e-9:
        lat_off = half_width / m_lat
        lon_off = half_width / max(1e-9, abs(m_lon))
        return np.asarray([
            [p1[0] + lat_off, p1[1]],
            [p1[0] - lat_off, p1[1]],
            [p1[0], p1[1] + lon_off],
            [p1[0], p1[1] - lon_off],
        ], dtype=float)
    nx = -dy / seg_len
    ny = dx / seg_len
    lon_off = nx * half_width / m_lon
    lat_off = ny * half_width / m_lat
    return np.asarray([
        [p1[0] + lat_off, p1[1] + lon_off],
        [p1[0] - lat_off, p1[1] - lon_off],
        [p2[0] + lat_off, p2[1] + lon_off],
        [p2[0] - lat_off, p2[1] - lon_off],
    ], dtype=float)


def _is_path_inside_airspace_envelope_v1(
    path,
    flight_phases,
    center_latlon,
    radius_m,
    cruise_half_width_m,
    alt_min_m=None,
    alt_max_m=None,
    transition_corridor_cfg=None,
):
    """Check the phase-specific horizontal corridor and transition lower envelope."""
    points = np.asarray(path, dtype=float).reshape(-1, 3)
    audit = {
        "status": "FAIL",
        "reason": "path_too_short",
        "horizontal_min_margin_m": None,
        "vertical_lower_min_margin_m": None,
        "vertical_upper_min_margin_m": None,
        "directions": {},
    }
    if points.shape[0] < 2:
        return False, "path_too_short", audit
    try:
        edge_phases = _edge_phase_values_v1(
            points,
            flight_phases,
            transition_corridor_cfg,
        )
    except ValueError as exc:
        failure_reason = str(exc).split(":", 1)[0]
        audit["reason"] = failure_reason
        return False, failure_reason, audit

    first_reason = "ok"
    horizontal_margins = []
    lower_margins = []
    upper_margins = []
    direction_values = {"takeoff": [], "landing": []}
    for edge_idx in range(points.shape[0] - 1):
        p1 = points[edge_idx]
        p2 = points[edge_idx + 1]
        phase = str(edge_phases[edge_idx])
        direction = _transition_direction_from_phase_value_v1(phase)
        is_transition = bool(
            (transition_corridor_cfg or {}).get("enabled", False)
            and direction is not None
        )
        edge_half_width_m = _edge_corridor_half_width_v1(
            phase,
            cruise_half_width_m,
            transition_corridor_cfg,
        )
        horizontal_points = _edge_lateral_boundary_points_v1(
            p1,
            p2,
            edge_half_width_m,
        )
        max_distance_m = float(np.max(_dist_to_center_m(horizontal_points, center_latlon)))
        horizontal_margin_m = float(radius_m) - max_distance_m
        horizontal_margins.append(horizontal_margin_m)

        if alt_min_m is None:
            lower_margin_m = float("inf")
        else:
            lower_values = []
            for point in (p1, p2):
                effective = _effective_downward_clearance_v1(
                    point[2], phase, transition_corridor_cfg
                ) if is_transition else 0.0
                lower_values.append(float(point[2]) - float(effective))
            lower_margin_m = float(min(lower_values) - float(alt_min_m))
            lower_margins.append(lower_margin_m)

        if alt_max_m is None:
            upper_margin_m = float("inf")
        else:
            upper_margin_m = float(float(alt_max_m) - max(float(p1[2]), float(p2[2])))
            upper_margins.append(upper_margin_m)

        edge_ok = True
        edge_reason = "ok"
        if horizontal_margin_m <= 1e-6:
            edge_ok = False
            edge_reason = (
                f"{direction}_transition_airspace_horizontal_envelope_outside"
                if direction is not None
                else "cruise_airspace_horizontal_envelope_contact_or_outside"
            )
        elif lower_margin_m < -1e-6:
            edge_ok = False
            edge_reason = (
                f"{direction}_transition_airspace_vertical_envelope_outside"
                if direction is not None else "airspace_altitude_centerline_outside"
            )
        elif upper_margin_m < -1e-6:
            edge_ok = False
            edge_reason = (
                f"{direction}_transition_airspace_vertical_envelope_outside"
                if direction is not None else "airspace_altitude_centerline_outside"
            )
        if first_reason == "ok" and not edge_ok:
            first_reason = edge_reason
        if direction is not None:
            direction_values[direction].append({
                "horizontal_margin_m": horizontal_margin_m,
                "vertical_lower_margin_m": lower_margin_m,
                "vertical_upper_margin_m": upper_margin_m,
                "ok": bool(edge_ok),
                "reason": str(edge_reason),
            })

    for direction, values in direction_values.items():
        if not values:
            audit["directions"][direction] = {"status": "NOT APPLICABLE"}
            continue
        audit["directions"][direction] = {
            "status": "PASS" if all(v["ok"] for v in values) else "FAIL",
            "reason": next(
                (str(v["reason"]) for v in values if not v["ok"]),
                "ok",
            ),
            "horizontal_min_margin_m": float(min(v["horizontal_margin_m"] for v in values)),
            "vertical_lower_min_margin_m": float(min(v["vertical_lower_margin_m"] for v in values)),
            "vertical_upper_min_margin_m": float(min(v["vertical_upper_margin_m"] for v in values)),
        }

    ok = bool(first_reason == "ok")
    audit.update({
        "status": "PASS" if ok else "FAIL",
        "reason": first_reason,
        "horizontal_min_margin_m": (
            float(min(horizontal_margins)) if horizontal_margins else None
        ),
        "vertical_lower_min_margin_m": (
            float(min(lower_margins)) if lower_margins else None
        ),
        "vertical_upper_min_margin_m": (
            float(min(upper_margins)) if upper_margins else None
        ),
    })
    return ok, first_reason, audit


def generate_single_initial_solution(
    backbone,                 # (K, 3) 고정/필수 WP
    wp_perturb_radius_m,      # WP 교란 반경 (m)
    min_extra_nodes_per_seg,  # int | list[int], 세그먼트별 최소 extra node 수
    max_extra_nodes_per_seg,  # int | list[int], 세그먼트별 최대 extra node 수
    safe_nodes_by_seg,        # list of ndarray, 세그먼트별 안전 노드 후보
    emergency_points,         # (E, 3) 비상착륙점
    emergency_strip_m,        # emergency 포함 완화 strip 폭(m)
    is_fixed,                 # (K,) bool, True면 해당 WP 교란 금지
    wp_perturb_steps=1,       # WP 교란 반복 횟수
    min_seg_for_extra_nodes_m=2000.0,  # 이 값보다 짧은 세그먼트는 extra node 미생성
):
    K = backbone.shape[0]
    m_lat = 111000.0
    m_lon = 111000.0 * np.cos(np.deg2rad(float(np.mean(backbone[:, 0]))))
    r_lat = wp_perturb_radius_m / m_lat
    r_lon = wp_perturb_radius_m / m_lon
    n_perturb = max(1, int(wp_perturb_steps))
    step_r_lat = r_lat / n_perturb
    step_r_lon = r_lon / n_perturb

    perturbed = backbone.copy()
    for i in range(K):
        if is_fixed[i]:
            continue
        for _ in range(n_perturb):
            ang = np.random.uniform(0, 2 * np.pi)
            d = np.sqrt(np.random.uniform()) * 1.0   # within unit circle
            perturbed[i, 0] += d * step_r_lat * np.sin(ang)
            perturbed[i, 1] += d * step_r_lon * np.cos(ang)

    # min/max를 세그먼트별 리스트 형태로 정규화
    if isinstance(min_extra_nodes_per_seg, int):
        mn_list = [min_extra_nodes_per_seg] * (K - 1)
    else:
        mn_list = list(min_extra_nodes_per_seg)
    if isinstance(max_extra_nodes_per_seg, int):
        mx_list = [max_extra_nodes_per_seg] * (K - 1)
    else:
        mx_list = list(max_extra_nodes_per_seg)

    path_pts = [perturbed[0]]
    for k in range(K - 1):
        a = perturbed[k]
        b = perturbed[k + 1]
        inserts = []
        end_buffer_ratio = _segment_strip_end_buffer_ratio(k, K - 1)

        seg_m = _seg_dist_m(a, b)
        seg_long_enough = seg_m >= min_seg_for_extra_nodes_m

        if seg_long_enough:
            lo = mn_list[k] if k < len(mn_list) else 0
            hi = mx_list[k] if k < len(mx_list) else 0
            m = int(np.random.randint(lo, hi + 1)) if hi >= lo else 0
            cand = safe_nodes_by_seg[k] if k < len(safe_nodes_by_seg) else np.empty((0, 3))
            if m > 0 and cand.size > 0:
                idx = np.random.choice(cand.shape[0], size=min(m, cand.shape[0]), replace=False)
                inserts.extend(cand[idx].tolist())

        if emergency_points is not None and emergency_points.size > 0:
            em_in_strip = filter_nodes_in_strip(a, b, emergency_points, emergency_strip_m,
                                                 end_buffer_ratio=end_buffer_ratio)
            if em_in_strip.size > 0:
                inserts.extend(em_in_strip.tolist())

        if inserts:
            inserts_arr = np.array(inserts, dtype=float)
            ab = b[:2] - a[:2]
            denom = float(np.dot(ab, ab)) + 1e-12
            t_vals = ((inserts_arr[:, :2] - a[:2]) @ ab) / denom
            order = np.argsort(t_vals)
            for idx in order:
                path_pts.append(inserts_arr[idx])

        path_pts.append(b)

    return np.array(path_pts, dtype=float)


def generate_single_initial_solution_with_skip(
    full_waypoints,           # (M, 3) 전체 WP
    takeoff_wp,               # (3,) 필수 시작점(takeoff_complete)
    landing_wp,               # (3,) 필수 도착점(landing_entry)
    wp_perturb_radius_m,      # WP 교란 반경
    min_extra_nodes_per_seg,  # int | list[int]
    max_extra_nodes_per_seg,  # int | list[int]
    safe_nodes_by_seg_full,   # list, full_waypoints 기준 safe nodes
    emergency_points,         # (E, 3)
    emergency_strip_m,        # float
    wp_perturb_steps=1,       # WP 교란 반복 횟수
    wp_skip_prob=0.25,        # 중간 WP skip 확률 (0~1)
    min_seg_for_extra_nodes_m=2000.0,  # 이 값보다 짧은 세그먼트는 extra node 미생성
):
    """
    WP skip 초기해 생성:
    - 필수: takeoff_wp, landing_wp
    - 선택: full_waypoints[1:-1]를 skip 확률로 샘플링
    """
    M = full_waypoints.shape[0]
    m_lat = 111000.0
    m_lon = 111000.0 * np.cos(np.deg2rad(float(np.mean(full_waypoints[:, 0]))))
    r_lat = wp_perturb_radius_m / m_lat
    r_lon = wp_perturb_radius_m / m_lon
    n_perturb = max(1, int(wp_perturb_steps))
    step_r_lat = r_lat / n_perturb
    step_r_lon = r_lon / n_perturb

    selected_indices = [0]
    for i in range(1, M - 1):
        if np.random.uniform() > wp_skip_prob:
            selected_indices.append(i)
    selected_indices.append(M - 1)

    selected_waypoints = full_waypoints[selected_indices]
    selected_wps = np.vstack([takeoff_wp, selected_waypoints, landing_wp])
    K = selected_wps.shape[0]

    perturbed = selected_wps.copy()
    for i in range(1, K - 1):
        for _ in range(n_perturb):
            ang = np.random.uniform(0, 2 * np.pi)
            d = np.sqrt(np.random.uniform())
            perturbed[i, 0] += d * step_r_lat * np.sin(ang)
            perturbed[i, 1] += d * step_r_lon * np.cos(ang)

    # min/max를 세그먼트별 리스트 형태로 정규화
    if isinstance(min_extra_nodes_per_seg, int):
        mn_list = [min_extra_nodes_per_seg] * (K - 1)
    else:
        mn_list = list(min_extra_nodes_per_seg)
    if isinstance(max_extra_nodes_per_seg, int):
        mx_list = [max_extra_nodes_per_seg] * (K - 1)
    else:
        mx_list = list(max_extra_nodes_per_seg)

    path_pts = [perturbed[0]]
    for k in range(K - 1):
        a = perturbed[k]
        b = perturbed[k + 1]
        inserts = []
        end_buffer_ratio = _segment_strip_end_buffer_ratio(k, K - 1)

        seg_m = _seg_dist_m(a, b)
        seg_long_enough = seg_m >= min_seg_for_extra_nodes_m

        if seg_long_enough:
            lo = mn_list[k] if k < len(mn_list) else 0
            hi = mx_list[k] if k < len(mx_list) else 0
            m = int(np.random.randint(lo, hi + 1)) if hi >= lo else 0
            cand = safe_nodes_by_seg_full[k] if k < len(safe_nodes_by_seg_full) else np.empty((0, 3))
            if m > 0 and cand.size > 0:
                idx = np.random.choice(cand.shape[0], size=min(m, cand.shape[0]), replace=False)
                inserts.extend(cand[idx].tolist())

        if emergency_points is not None and emergency_points.size > 0:
            em_in_strip = filter_nodes_in_strip(a, b, emergency_points, emergency_strip_m,
                                                 end_buffer_ratio=end_buffer_ratio)
            if em_in_strip.size > 0:
                inserts.extend(em_in_strip.tolist())

        # t-projection
        if inserts:
            inserts_arr = np.array(inserts, dtype=float)
            ab = b[:2] - a[:2]
            denom = float(np.dot(ab, ab)) + 1e-12
            t_vals = ((inserts_arr[:, :2] - a[:2]) @ ab) / denom
            order = np.argsort(t_vals)
            for idx in order:
                path_pts.append(inserts_arr[idx])

        path_pts.append(b)

    return np.array(path_pts, dtype=float)


def _build_corridor_buffer_geometry_v1(path, half_width_m):
    """Build a valid metric corridor buffer around a WGS84 centerline."""
    if path is None:
        return None
    points = np.asarray(path, dtype=float)
    width = float(half_width_m)
    if (
        points.ndim != 2
        or points.shape[0] < 2
        or points.shape[1] < 2
        or not np.isfinite(width)
        or width <= 0.0
    ):
        return None
    if not np.all(np.isfinite(points[:, :2])):
        return None

    x, y = _CORRIDOR_TO_EPSG5179.transform(points[:, 1], points[:, 0])
    xy = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]
    xy = xy[keep]
    if xy.shape[0] < 2:
        return None

    geometry = LineString(xy).buffer(
        width,
        cap_style="flat",
        join_style="round",
    )
    if geometry.is_empty:
        return None
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    return None if geometry.is_empty else geometry


def plot_corridor_width(gx, path, W_half, color="yellow", alpha=0.2):
    geometry_m = _build_corridor_buffer_geometry_v1(path, W_half)
    if geometry_m is None:
        return
    geometry_lonlat = shapely_transform(
        _CORRIDOR_FROM_EPSG5179.transform,
        geometry_m,
    )
    gx.add_geometries(
        [geometry_lonlat],
        crs=ccrs.PlateCarree(),
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
        zorder=2,
    )


def _plot_corridor_width_by_phase_v1(
    gx,
    rf,
    cruise_half_width_m,
    transition_corridor_cfg,
    color="yellow",
    alpha=0.2,
):
    """Draw the same corridor fill style with the width selected per flight phase."""
    if not bool((transition_corridor_cfg or {}).get("enabled", False)):
        path = np.asarray(rf.get("path", np.empty((0, 3))), dtype=float).reshape(-1, 3)
        plot_corridor_width(gx, path, cruise_half_width_m, color=color, alpha=alpha)
        return
    for seg in rf.get("segments", []):
        points = np.asarray(seg.get("points", np.empty((0, 3))), dtype=float).reshape(-1, 3)
        if points.shape[0] < 2:
            continue
        phases = np.asarray(
            seg.get("point_phases", np.empty((0,), dtype=object)), dtype=object
        ).reshape(-1)
        edge_phases = _edge_phase_values_v1(
            points,
            phases,
            transition_corridor_cfg,
        )
        widths = np.asarray([
            _edge_corridor_half_width_v1(
                phase,
                cruise_half_width_m,
                transition_corridor_cfg,
            )
            for phase in edge_phases
        ], dtype=float)
        start = 0
        while start < widths.size:
            end = start + 1
            while end < widths.size and abs(widths[end] - widths[start]) <= 1e-9:
                end += 1
            plot_corridor_width(
                gx,
                points[start:end + 1],
                float(widths[start]),
                color=color,
                alpha=alpha,
            )
            start = end


def plot_forbidden_zones(gx, forbidden_zones,
                         edge_color="firebrick", face_color="tomato",
                         edge_alpha=0.85, face_alpha=0.12,
                         edge_width=1.2, zorder_fill=2, zorder_edge=9,
                         label="No-Fly Zone"):
    """Overlay NFZ rectangles on map axes with subtle transparency."""
    if gx is None or forbidden_zones is None:
        return
    fz = np.asarray(forbidden_zones, dtype=float)
    if fz.size == 0:
        return
    for zi, rect in enumerate(fz):
        lon_min, lon_max, lat_min, lat_max = [float(v) for v in rect]
        poly = np.array([
            [lon_min, lat_min],
            [lon_max, lat_min],
            [lon_max, lat_max],
            [lon_min, lat_max],
            [lon_min, lat_min],
        ], dtype=float)
        gx.fill(
            poly[:, 0], poly[:, 1],
            color=face_color, alpha=float(face_alpha),
            transform=ccrs.Geodetic(), zorder=int(zorder_fill),
        )
        gx.plot(
            poly[:, 0], poly[:, 1],
            "-", color=edge_color, linewidth=float(edge_width),
            alpha=float(edge_alpha),
            transform=ccrs.Geodetic(), zorder=int(zorder_edge),
            label=None,
        )

    # Add one explicit legend handle so NFZ label is always visible in map legends.
    gx.plot(
        [], [],
        "-", color=edge_color, linewidth=float(edge_width),
        alpha=float(edge_alpha),
        label=label,
    )


def plot_moc_binary_overlay(gx, moc_2d, lat_lim, lon_lim,
                            label="MOC=1 (Obstacle Risk)",
                            fill_color="fuchsia", fill_alpha=0.22):
    """Overlay binary MOC mask (1=blocked/risky for corridor) on map axes."""
    if gx is None or moc_2d is None or np.size(moc_2d) == 0:
        return
    mm = (np.asarray(moc_2d, dtype=float) >= 0.5).astype(float)
    if mm.ndim != 2:
        return

    Ny, Nx = mm.shape
    lats = np.linspace(float(lat_lim[0]), float(lat_lim[1]), Ny)
    lons = np.linspace(float(lon_lim[0]), float(lon_lim[1]), Nx)
    LON, LAT = np.meshgrid(lons, lats)

    if np.any(mm > 0.5):
        gx.contourf(
            LON,
            LAT,
            mm,
            levels=[0.5, 1.5],
            colors=[fill_color],
            alpha=float(fill_alpha),
            transform=ccrs.PlateCarree(),
            zorder=2,
        )

    gx.plot(
        [], [],
        "-",
        color=fill_color,
        linewidth=3.0,
        alpha=float(fill_alpha),
        label=label,
    )


def _sector_wind_months_v1(season):
    """Return the wind months represented by the configured sector season."""
    season = str(season).strip().lower()
    month_groups = {
        "annual": tuple(range(1, 13)),
        "spring": (3, 4, 5),
        "summer": (6, 7, 8),
        "autumn": (9, 10, 11),
        "winter": (12, 1, 2),
    }
    if season not in month_groups:
        raise ValueError(
            f"sector_season must be one of {tuple(month_groups)}, got {season!r}."
        )
    return month_groups[season]


def _validate_sector_1based_v1(value, label):
    sector = int(value)
    if sector < 1 or sector > 12:
        raise ValueError(f"{label} must be in [1, 12], got {value}")
    return sector


def _sector_rectilinear_axes_v1(x_2d, y_2d, source_name):
    """Extract increasing X/Y axes and report whether spatial data need transposing."""
    x_arr = np.asarray(x_2d, dtype=float)
    y_arr = np.asarray(y_2d, dtype=float)
    if x_arr.ndim != 2 or x_arr.shape != y_arr.shape:
        raise ValueError(f"{source_name}: X_2d/Y_2d must be equal-shape 2D arrays.")

    x_axis = np.asarray(x_arr[0, :], dtype=float)
    y_axis = np.asarray(y_arr[:, 0], dtype=float)
    transpose_spatial = False
    if not (
        np.allclose(x_arr, x_axis[np.newaxis, :])
        and np.allclose(y_arr, y_axis[:, np.newaxis])
    ):
        x_axis = np.asarray(x_arr[:, 0], dtype=float)
        y_axis = np.asarray(y_arr[0, :], dtype=float)
        transpose_spatial = True
        if not (
            np.allclose(x_arr, x_axis[:, np.newaxis])
            and np.allclose(y_arr, y_axis[np.newaxis, :])
        ):
            raise ValueError(f"{source_name}: wind grid must be rectilinear.")
    if np.any(np.diff(x_axis) <= 0.0) or np.any(np.diff(y_axis) <= 0.0):
        raise ValueError(f"{source_name}: wind grid axes must be strictly increasing.")
    return x_axis, y_axis, transpose_spatial


def _load_sector_wind_data_v1(wind_data_dir, months):
    """Load only the seasonal monthly U/V fields needed by automatic sector selection."""
    try:
        import pyproj
    except Exception as exc:
        raise RuntimeError("pyproj is required for automatic sector wind sampling.") from exc

    wind_dir = Path(wind_data_dir)
    month_numbers = tuple(int(value) for value in months)
    paths = [wind_dir / f"AirRisk_Data_{month}.mat" for month in month_numbers]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing sector wind file(s):\n" + "\n".join(missing))

    base_x = base_y = base_z = None
    loaded_months = []
    for month, path in zip(month_numbers, paths):
        raw = loadmat(
            str(path),
            variable_names=["X_2d", "Y_2d", "z_vec", "U3d", "V3d", "theta3d"],
        )
        x_axis, y_axis, transpose_spatial = _sector_rectilinear_axes_v1(
            raw["X_2d"], raw["Y_2d"], path.name
        )
        z_axis = np.asarray(raw["z_vec"], dtype=float).reshape(-1)
        u = np.asarray(raw["U3d"], dtype=float)
        v = np.asarray(raw["V3d"], dtype=float)
        theta = np.asarray(raw["theta3d"], dtype=float)
        if transpose_spatial:
            u = np.transpose(u, (1, 0, 2))
            v = np.transpose(v, (1, 0, 2))
            theta = np.transpose(theta, (1, 0, 2))
        expected_shape = (y_axis.size, x_axis.size, z_axis.size)
        if u.shape != expected_shape or v.shape != expected_shape or theta.shape != expected_shape:
            raise ValueError(
                f"{path.name}: U/V/theta shape must be {expected_shape}, "
                f"got U={u.shape}, V={v.shape}, theta={theta.shape}."
            )
        if base_x is None:
            base_x, base_y, base_z = x_axis, y_axis, z_axis
        elif (
            not np.array_equal(x_axis, base_x)
            or not np.array_equal(y_axis, base_y)
            or not np.array_equal(z_axis, base_z)
        ):
            raise ValueError("Monthly sector wind grids must have identical X/Y/Z axes.")

        valid = (
            np.isfinite(u)
            & np.isfinite(v)
            & np.isfinite(theta)
            & (u != -1.0)
            & (v != -1.0)
            & (theta != -1.0)
            & ~((u == 0.0) & (v == 0.0))
        )
        axes = (y_axis, x_axis, z_axis)
        loaded_months.append({
            "month": int(month),
            "u_numerator": RegularGridInterpolator(
                axes, np.where(valid, u, 0.0), bounds_error=False, fill_value=0.0
            ),
            "v_numerator": RegularGridInterpolator(
                axes, np.where(valid, v, 0.0), bounds_error=False, fill_value=0.0
            ),
            "valid_weight": RegularGridInterpolator(
                axes, valid.astype(float), bounds_error=False, fill_value=0.0
            ),
        })

    return {
        "months": loaded_months,
        "month_numbers": [int(value) for value in month_numbers],
        "x_axis": np.asarray(base_x, dtype=float),
        "y_axis": np.asarray(base_y, dtype=float),
        "z_axis": np.asarray(base_z, dtype=float),
        "to_epsg5179": pyproj.Transformer.from_crs(
            "EPSG:4326", "EPSG:5179", always_xy=True
        ),
    }


def _sample_sector_wind_v1(wind_data, latitudes, longitudes, altitudes_msl):
    """Sample seasonal U/V fields with validity-weighted trilinear interpolation."""
    lats = np.asarray(latitudes, dtype=float)
    lons = np.asarray(longitudes, dtype=float)
    alts = np.asarray(altitudes_msl, dtype=float)
    x, y = wind_data["to_epsg5179"].transform(lons, lats)
    query = np.column_stack([np.asarray(y), np.asarray(x), alts])
    samples = []
    for month_data in wind_data["months"]:
        weight = np.asarray(month_data["valid_weight"](query), dtype=float)
        valid = weight >= 0.5
        u = np.full(weight.shape, np.nan, dtype=float)
        v = np.full(weight.shape, np.nan, dtype=float)
        u_num = np.asarray(month_data["u_numerator"](query), dtype=float)
        v_num = np.asarray(month_data["v_numerator"](query), dtype=float)
        u[valid] = u_num[valid] / weight[valid]
        v[valid] = v_num[valid] / weight[valid]
        samples.append((u, v, valid))
    return samples


def _sector_nominal_geometry_v1(
    port_altitude_msl,
    target_altitude_msl,
    transition_structure_mode,
    transition_mode,
    configured_total_distance_m,
    angle_deg,
    direction_label,
):
    """Resolve the full nominal port-to-cruise slope used for sector screening."""
    height = float(target_altitude_msl) - float(port_altitude_msl)
    if height < -1e-9:
        raise ValueError(f"{direction_label} target altitude is below the vertiport.")
    if height <= 1e-9:
        return {"height_m": 0.0, "distance_m": 0.0, "angle_deg": 0.0, "mode": "zero_height"}
    if str(transition_structure_mode) == TRANSITION_STRUCTURE_OPTIMIZED_ONLY:
        angle = _validate_transition_angle(angle_deg, f"{direction_label}_angle_deg")
        return {
            "height_m": float(height),
            "distance_m": float(height / np.tan(np.deg2rad(angle))),
            "angle_deg": float(angle),
            "mode": "optimized_only_angle",
        }
    return _calculate_transition_geometry(
        height_m=height,
        transition_mode=transition_mode,
        distance_m=configured_total_distance_m,
        angle_deg=angle_deg,
    )


def _sector_normalize01_v1(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    normalized = np.ones(values.shape, dtype=float)
    if not np.any(finite):
        return normalized, None, None
    minimum = float(np.min(values[finite]))
    maximum = float(np.max(values[finite]))
    if maximum - minimum <= 1e-12:
        normalized[finite] = 0.0
    else:
        normalized[finite] = (values[finite] - minimum) / (maximum - minimum)
    return normalized, minimum, maximum


def _sector_finite_or_none_v1(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _sector_json_safe_v1(value):
    if isinstance(value, dict):
        return {str(key): _sector_json_safe_v1(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sector_json_safe_v1(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_sector_json_safe_v1(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


def _evaluate_sector_direction_v1(
    sector,
    direction,
    port_lla,
    target_altitude_msl,
    geometry,
    corridor_half_width_m,
    along_track_step_m,
    transition_corridor_cfg,
    Norm_RT,
    AirRisk,
    risk_altitude_levels,
    MOCRisk,
    lat_lim,
    lon_lim,
    use_heading_map,
    wind_data,
):
    """Evaluate one direction/sector over the same 3D footprint used by the v1 MOC checker."""
    port = np.asarray(port_lla, dtype=float).reshape(3)
    total_distance_m = float(geometry["distance_m"])
    bearing_out_deg = float(((int(sector) - 0.5) * 30.0) % 360.0)
    heading_math_deg = float(90.0 - bearing_out_deg)
    outer_lat, outer_lon = _move_latlon(
        float(port[0]), float(port[1]), np.deg2rad(heading_math_deg), total_distance_m
    )
    outer = np.array([outer_lat, outer_lon, float(target_altitude_msl)], dtype=float)
    if direction == "takeoff":
        p1, p2 = port, outer
        phase = FLIGHT_PHASE_TAKEOFF_STAGE2
        flight_heading_deg = bearing_out_deg
    elif direction == "landing":
        p1, p2 = outer, port
        phase = FLIGHT_PHASE_LANDING_STAGE2
        flight_heading_deg = float((bearing_out_deg + 180.0) % 360.0)
    else:
        raise ValueError(f"Unknown sector direction: {direction}")

    samples = list(_iter_corridor_moc_samples_v1(
        p1,
        p2,
        corridor_half_width_m,
        MOCRisk,
        lat_lim,
        lon_lim,
        along_step_m=along_track_step_m,
        phase=phase,
        transition_corridor_cfg=transition_corridor_cfg,
    ))
    if not samples:
        raise RuntimeError(f"No MOC samples generated for {direction} sector S{sector}.")

    lats = np.asarray([row[3] for row in samples], dtype=float)
    lons = np.asarray([row[4] for row in samples], dtype=float)
    center_altitudes = np.asarray([row[5] for row in samples], dtype=float)
    layer_indices = np.asarray([row[9] for row in samples], dtype=int)
    grid_rows = np.asarray([row[10] for row in samples], dtype=int)
    grid_cols = np.asarray([row[11] for row in samples], dtype=int)
    in_grid = np.asarray([row[12] for row in samples], dtype=bool)
    blocked = np.asarray([row[13] for row in samples], dtype=bool)
    effective_clearance = np.asarray([row[7] for row in samples], dtype=float)
    lower_face_msl = np.asarray([row[8] for row in samples], dtype=float)

    tested_cells = {
        (int(layer), int(row), int(col))
        for layer, row, col, valid in zip(layer_indices, grid_rows, grid_cols, in_grid)
        if bool(valid)
    }
    blocked_cells = {
        (int(layer), int(row), int(col))
        for layer, row, col, hit in zip(layer_indices, grid_rows, grid_cols, blocked)
        if bool(hit)
    }
    blocked_sample_count = int(np.count_nonzero(blocked))
    out_of_grid_count = int(np.count_nonzero(~in_grid))
    moc_pass = bool(blocked_sample_count == 0 and out_of_grid_count == 0 and tested_cells)
    moc_issue_ratio = float(
        (blocked_sample_count + out_of_grid_count) / max(1, len(samples))
    )

    _, heading_count, Ny, Nx = Norm_RT.shape
    risk_inside = (
        (lats >= float(lat_lim[0]))
        & (lats <= float(lat_lim[1]))
        & (lons >= float(lon_lim[0]))
        & (lons <= float(lon_lim[1]))
    )
    Iq = (lons - float(lon_lim[0])) / (
        (float(lon_lim[1]) - float(lon_lim[0])) / max(1, Nx - 1)
    )
    Jq = (lats - float(lat_lim[0])) / (
        (float(lat_lim[1]) - float(lat_lim[0])) / max(1, Ny - 1)
    )
    altitude_indices = np.argmin(
        np.abs(
            center_altitudes[:, None]
            - np.asarray(risk_altitude_levels, dtype=float)[None, :]
        ),
        axis=1,
    )
    heading_index = (
        int(round(flight_heading_deg / 45.0) % heading_count)
        if bool(use_heading_map) else 0
    )
    ground_values = np.full(len(samples), np.nan, dtype=float)
    air_values = np.full(len(samples), np.nan, dtype=float)
    coords = np.vstack([Jq, Iq])
    for altitude_index in np.unique(altitude_indices):
        mask = risk_inside & (altitude_indices == int(altitude_index))
        if not np.any(mask):
            continue
        ground_values[mask] = map_coordinates(
            Norm_RT[int(altitude_index), heading_index],
            coords[:, mask],
            order=1,
            mode="constant",
            cval=np.nan,
        )
        air_values[mask] = map_coordinates(
            AirRisk[:, :, int(altitude_index)],
            coords[:, mask],
            order=1,
            mode="constant",
            cval=np.nan,
        )

    wind_samples = _sample_sector_wind_v1(
        wind_data, lats, lons, center_altitudes
    )
    bearing_rad = np.deg2rad(flight_heading_deg)
    east_unit = float(np.sin(bearing_rad))
    north_unit = float(np.cos(bearing_rad))
    tailwind_values = []
    crosswind_values = []
    headwind_values = []
    valid_wind_count = 0
    for u, v, valid in wind_samples:
        along = u * east_unit + v * north_unit
        cross = np.abs(u * north_unit - v * east_unit)
        tailwind_values.append(np.where(valid, np.maximum(along, 0.0), np.nan))
        crosswind_values.append(np.where(valid, cross, np.nan))
        headwind_values.append(np.where(valid, np.maximum(-along, 0.0), np.nan))
        valid_wind_count += int(np.count_nonzero(valid))
    tailwind_stack = np.asarray(tailwind_values, dtype=float)
    crosswind_stack = np.asarray(crosswind_values, dtype=float)
    headwind_stack = np.asarray(headwind_values, dtype=float)

    def _finite_mean(values):
        arr = np.asarray(values, dtype=float)
        finite = np.isfinite(arr)
        return float(np.mean(arr[finite])) if np.any(finite) else float("nan")

    used_layers_agl = sorted({
        int(MOC_AGL_LEVELS_M[int(np.clip(index, 0, len(MOC_AGL_LEVELS_M) - 1))])
        for index in layer_indices.tolist()
    })
    return {
        "sector": int(sector),
        "direction": str(direction),
        "bearing_out_deg": bearing_out_deg,
        "flight_heading_deg": float(flight_heading_deg),
        "ground_heading_index": int(heading_index),
        "nominal_total_distance_m": float(total_distance_m),
        "nominal_angle_deg": float(geometry["angle_deg"]),
        "sample_count": int(len(samples)),
        "moc_tested_sample_count": int(np.count_nonzero(in_grid)),
        "moc_blocked_sample_count": blocked_sample_count,
        "moc_out_of_grid_sample_count": out_of_grid_count,
        "moc_tested_cell_count": int(len(tested_cells)),
        "moc_blocked_cell_count": int(len(blocked_cells)),
        "moc_issue_ratio": moc_issue_ratio,
        "moc_pass": moc_pass,
        "used_moc_agl_layers_m": used_layers_agl,
        "minimum_effective_clearance_m": float(np.min(effective_clearance)),
        "maximum_effective_clearance_m": float(np.max(effective_clearance)),
        "minimum_lower_face_msl_m": float(np.min(lower_face_msl)),
        "maximum_lower_face_msl_m": float(np.max(lower_face_msl)),
        "tailwind_mps_raw": _finite_mean(tailwind_stack),
        "crosswind_mps_raw": _finite_mean(crosswind_stack),
        "headwind_mps_raw": _finite_mean(headwind_stack),
        "ground_risk_raw": _finite_mean(ground_values),
        "air_risk_raw": _finite_mean(air_values),
        "wind_coverage_ratio": float(
            valid_wind_count / max(1, len(samples) * len(wind_samples))
        ),
        "ground_coverage_ratio": float(np.mean(np.isfinite(ground_values))),
        "air_coverage_ratio": float(np.mean(np.isfinite(air_values))),
        "risk_data_available": bool(
            np.any(np.isfinite(tailwind_stack))
            and np.any(np.isfinite(crosswind_stack))
            and np.any(np.isfinite(ground_values))
            and np.any(np.isfinite(air_values))
        ),
        "_blocked_lats": lats[blocked].copy(),
        "_blocked_lons": lons[blocked].copy(),
    }


def _plot_sector_selection_diagnostics_v1(
    analysis,
    moc_risk,
    lat_lim,
    lon_lim,
    start_vertiport,
    end_vertiport,
    output_path,
    request=None,
    map_zoom=13,
):
    """Save one PNG containing map-backed takeoff and landing sector wheels."""
    output_path = Path(output_path)
    direction_metrics = analysis["direction_metrics"]
    selected_pair = analysis["selected_pair"]

    def _osm_preflight():
        if request is None:
            return False, "map_request_unavailable"
        try:
            zoom = int(map_zoom)
            lon = float(start_vertiport[1])
            lat = float(np.clip(start_vertiport[0], -85.05112878, 85.05112878))
            tile_count = 2 ** zoom
            tile_x = int(np.floor((lon + 180.0) / 360.0 * tile_count))
            lat_rad = np.deg2rad(lat)
            tile_y = int(np.floor(
                (1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) * 0.5 * tile_count
            ))
            tile_image, _, _ = request.get_image((tile_x, tile_y, zoom))
            tile_pixels = np.asarray(tile_image, dtype=float)
            if tile_pixels.size == 0 or float(np.nanstd(tile_pixels)) < 1e-6:
                raise RuntimeError("OSM returned an empty placeholder tile")
            return True, None
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def _render(use_osm):
        projection = request.crs if request is not None else ccrs.PlateCarree()
        fig = plt.figure(figsize=(20, 11))
        grid = fig.add_gridspec(2, 2, height_ratios=[7.5, 1.6], hspace=0.08, wspace=0.05)
        axes = {
            "takeoff": fig.add_subplot(grid[0, 0], projection=projection),
            "landing": fig.add_subplot(grid[0, 1], projection=projection),
        }
        summary_ax = fig.add_subplot(grid[1, :])
        summary_ax.axis("off")
        cmap = plt.get_cmap("RdYlGn_r")

        for direction, port, selected_sector, selected_color in (
            ("takeoff", start_vertiport, selected_pair["takeoff_sector"], "blue"),
            ("landing", end_vertiport, selected_pair["landing_sector"], "green"),
        ):
            ax = axes[direction]
            metrics = [row for row in direction_metrics if row["direction"] == direction]
            radius_m = max(float(row["nominal_total_distance_m"]) for row in metrics)
            circle_points = np.asarray(build_circle_lla(port, radius_m * 1.08), dtype=float)
            extent = compute_centered_map_extent(
                circle_points[:, :2], port, ring_radii_m=(radius_m * 1.08,), pad_ratio=0.02
            )
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            if use_osm and request is not None:
                ax.add_image(request, int(map_zoom))
            else:
                ax.set_facecolor("#eef2f3")
            ax.gridlines(
                crs=ccrs.PlateCarree(), draw_labels=False,
                linewidth=0.45, color="gray", alpha=0.35, linestyle="--"
            )

            used_layer_indices = sorted({
                int(np.where(MOC_AGL_LEVELS_M == float(agl))[0][0])
                for row in metrics
                for agl in row["used_moc_agl_layers_m"]
                if np.any(MOC_AGL_LEVELS_M == float(agl))
            })
            if used_layer_indices:
                moc_union = np.max(
                    np.asarray(moc_risk, dtype=float)[:, :, used_layer_indices], axis=2
                )
                Ny, Nx = moc_union.shape
                map_lats = np.linspace(float(lat_lim[0]), float(lat_lim[1]), Ny)
                map_lons = np.linspace(float(lon_lim[0]), float(lon_lim[1]), Nx)
                LON, LAT = np.meshgrid(map_lons, map_lats)
                m_lat = 111000.0
                m_lon = 111000.0 * np.cos(np.deg2rad(float(port[0])))
                radial = np.hypot(
                    (LAT - float(port[0])) * m_lat,
                    (LON - float(port[1])) * m_lon,
                )
                visible_moc = np.where(radial <= radius_m * 1.02, moc_union, 0.0)
                if np.any(visible_moc >= 0.5):
                    ax.contourf(
                        LON, LAT, visible_moc,
                        levels=[0.5, 1.5], colors=["magenta"], alpha=0.18,
                        transform=ccrs.PlateCarree(), zorder=2,
                    )

            for row in metrics:
                sector = int(row["sector"])
                heading_math_deg = float(np.rad2deg(_sector_angle(sector)))
                wedge_lon, wedge_lat = _build_sector_wedge_lonlat(
                    port,
                    heading_math_deg,
                    float(analysis["sector_half_width_deg"]),
                    row["nominal_total_distance_m"],
                    n_pts=30,
                )
                risk_score = float(np.clip(row["combined_risk_score"], 0.0, 1.0))
                ax.fill(
                    wedge_lon, wedge_lat,
                    facecolor=cmap(risk_score), edgecolor="black", linewidth=0.75,
                    alpha=0.36, transform=ccrs.PlateCarree(), zorder=4,
                )
                if not bool(row["moc_pass"]):
                    ax.fill(
                        wedge_lon, wedge_lat, facecolor="none", edgecolor="red",
                        linewidth=1.1, hatch="///", transform=ccrs.PlateCarree(), zorder=6,
                    )
                if int(row["moc_out_of_grid_sample_count"]) > 0:
                    ax.fill(
                        wedge_lon, wedge_lat, facecolor="none", edgecolor="darkorange",
                        linewidth=1.1, hatch="..", transform=ccrs.PlateCarree(), zorder=7,
                    )
                if sector == int(selected_sector):
                    ax.plot(
                        wedge_lon, wedge_lat, color=selected_color, linewidth=4.0,
                        transform=ccrs.PlateCarree(), zorder=10,
                    )

                blocked_lats = np.asarray(row.get("_blocked_lats", []), dtype=float)
                blocked_lons = np.asarray(row.get("_blocked_lons", []), dtype=float)
                if blocked_lats.size:
                    unique_points = np.unique(
                        np.round(np.column_stack([blocked_lats, blocked_lons]), 7), axis=0
                    )
                    if unique_points.shape[0] > 120:
                        pick = np.linspace(0, unique_points.shape[0] - 1, 120).astype(int)
                        unique_points = unique_points[pick]
                    ax.scatter(
                        unique_points[:, 1], unique_points[:, 0], marker="x", s=12,
                        color="red", linewidths=0.8, transform=ccrs.PlateCarree(), zorder=9,
                    )

                label_lat, label_lon = _move_latlon(
                    float(port[0]), float(port[1]), np.deg2rad(heading_math_deg),
                    0.68 * float(row["nominal_total_distance_m"]),
                )
                label = (
                    f"S{sector}  R {row['combined_risk_score']:.2f}\n"
                    f"W/G/A {row['wind_risk_score']:.2f}/"
                    f"{row['ground_risk_score']:.2f}/{row['air_risk_score']:.2f}\n"
                    f"MOC {row['moc_blocked_cell_count']}/{row['moc_tested_cell_count']}"
                )
                if int(row["moc_out_of_grid_sample_count"]) > 0:
                    label += f"  OOG {row['moc_out_of_grid_sample_count']}"
                ax.text(
                    label_lon, label_lat, label,
                    fontsize=6.7, fontweight="bold", ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.76, edgecolor="none", pad=1.2),
                    transform=ccrs.PlateCarree(), zorder=12,
                )

            ax.scatter(
                [float(port[1])], [float(port[0])], marker="s", s=85,
                facecolor="red", edgecolor="black", linewidth=1.2,
                transform=ccrs.PlateCarree(), zorder=15,
            )
            ax.set_title(
                f"{direction.title()} sectors | selected S{int(selected_sector)} | "
                f"nominal radius {radius_m / 1000.0:.2f} km",
                fontsize=13, fontweight="bold",
            )

        color_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.0, 1.0))
        color_mappable.set_array([])
        fig.colorbar(
            color_mappable, ax=list(axes.values()), orientation="horizontal",
            fraction=0.035, pad=0.03, label="Normalized combined risk (lower is better)",
        )
        fig.legend(
            handles=[
                Patch(facecolor="magenta", alpha=0.22, label="MOC blocked union (used layers)"),
                Patch(facecolor="none", edgecolor="red", hatch="///", label="MOC fail"),
                Patch(facecolor="none", edgecolor="darkorange", hatch="..", label="OUT_OF_GRID"),
                Patch(facecolor="none", edgecolor="blue", linewidth=3.0, label="Selected takeoff"),
                Patch(facecolor="none", edgecolor="green", linewidth=3.0, label="Selected landing"),
            ],
            loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=5, fontsize=9,
        )

        summary = (
            f"Selection: S{selected_pair['takeoff_sector']} takeoff / "
            f"S{selected_pair['landing_sector']} landing    |    "
            f"Status: {analysis['status']}    |    "
            f"Wind period: {analysis['season']} (months {analysis['wind_months']})\n"
            f"Pair risk={selected_pair['combined_risk_score']:.4f}  "
            f"[wind={selected_pair['wind_risk_score']:.4f}, "
            f"ground={selected_pair['ground_risk_score']:.4f}, "
            f"air={selected_pair['air_risk_score']:.4f}]    |    "
            f"MOC issue={selected_pair['moc_issue_ratio']:.4f}    |    "
            f"MOC-safe pairs={analysis['moc_safe_pair_count']}/{analysis['combination_count']}"
        )
        summary_ax.text(
            0.5, 0.52, summary, ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.65", facecolor="#f7f7f7", edgecolor="#666666"),
        )
        fig.suptitle(
            "Automatic sector selection: seasonal wind + ground risk + air risk, with mandatory MOC screening",
            fontsize=16, fontweight="bold", y=0.995,
        )
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    osm_available, preflight_reason = _osm_preflight()
    if not osm_available:
        if request is not None:
            print(
                "Warning: sector diagnostic OSM background unavailable; "
                f"using neutral fallback: {preflight_reason}"
            )
        _render(False)
        return {"used_osm": False, "fallback_reason": preflight_reason}
    try:
        _render(True)
        return {"used_osm": True, "fallback_reason": None}
    except Exception as exc:
        plt.close("all")
        print(
            "Warning: sector diagnostic OSM background failed; using neutral fallback: "
            f"{type(exc).__name__}: {exc}"
        )
        _render(False)
        return {
            "used_osm": False,
            "fallback_reason": f"{type(exc).__name__}: {exc}",
        }


def _automatic_sector_selection_v1(
    start_vertiport,
    end_vertiport,
    target_altitude_msl,
    transition_structure_mode,
    transition_mode,
    takeoff_total_distance_m,
    landing_total_distance_m,
    takeoff_angle_deg,
    landing_angle_deg,
    corridor_half_width_m,
    sector_half_width_deg,
    along_track_step_m,
    sector_season,
    wind_tail_weight,
    wind_cross_weight,
    wind_risk_weight,
    ground_risk_weight,
    air_risk_weight,
    wind_data_dir,
    Norm_RT,
    AirRisk,
    risk_altitude_levels,
    MOCRisk,
    lat_lim,
    lon_lim,
    use_heading_map,
    transition_corridor_cfg,
    output_png_path,
    request=None,
):
    """Evaluate 24 directional sectors, rank 132 pairs, and save one diagnostic PNG."""
    if not np.isclose(float(wind_tail_weight) + float(wind_cross_weight), 1.0):
        raise ValueError("sector wind tail/cross weights must sum to 1.0.")
    if not np.isclose(
        float(wind_risk_weight) + float(ground_risk_weight) + float(air_risk_weight),
        1.0,
    ):
        raise ValueError("sector wind/ground/air weights must sum to 1.0.")
    if not np.isfinite(float(along_track_step_m)) or float(along_track_step_m) <= 0.0:
        raise ValueError("sector along-track sample spacing must be finite and > 0.")
    if not np.isfinite(float(sector_half_width_deg)) or not 0.0 < float(sector_half_width_deg) <= 15.0:
        raise ValueError("sector_half_width_deg must satisfy 0 < value <= 15 degrees.")

    season = str(sector_season).strip().lower()
    wind_months = _sector_wind_months_v1(season)
    wind_data = _load_sector_wind_data_v1(wind_data_dir, wind_months)
    target_altitude_msl = float(target_altitude_msl)
    takeoff_geometry = _sector_nominal_geometry_v1(
        start_vertiport[2], target_altitude_msl,
        transition_structure_mode, transition_mode,
        takeoff_total_distance_m, takeoff_angle_deg, "takeoff",
    )
    landing_geometry = _sector_nominal_geometry_v1(
        end_vertiport[2], target_altitude_msl,
        transition_structure_mode, transition_mode,
        landing_total_distance_m, landing_angle_deg, "landing",
    )

    direction_metrics = []
    for sector in range(1, 13):
        direction_metrics.append(_evaluate_sector_direction_v1(
            sector, "takeoff", start_vertiport, target_altitude_msl,
            takeoff_geometry, corridor_half_width_m, along_track_step_m,
            transition_corridor_cfg,
            Norm_RT, AirRisk, risk_altitude_levels, MOCRisk,
            lat_lim, lon_lim, use_heading_map, wind_data,
        ))
        direction_metrics.append(_evaluate_sector_direction_v1(
            sector, "landing", end_vertiport, target_altitude_msl,
            landing_geometry, corridor_half_width_m, along_track_step_m,
            transition_corridor_cfg,
            Norm_RT, AirRisk, risk_altitude_levels, MOCRisk,
            lat_lim, lon_lim, use_heading_map, wind_data,
        ))

    normalization = {}
    raw_fields = {
        "tailwind": "tailwind_mps_raw",
        "crosswind": "crosswind_mps_raw",
        "ground": "ground_risk_raw",
        "air": "air_risk_raw",
    }
    normalized_by_field = {}
    for label, field in raw_fields.items():
        normalized, minimum, maximum = _sector_normalize01_v1(
            [row[field] for row in direction_metrics]
        )
        normalized_by_field[label] = normalized
        normalization[label] = {"minimum": minimum, "maximum": maximum}

    for index, row in enumerate(direction_metrics):
        row["tailwind_risk_score"] = float(normalized_by_field["tailwind"][index])
        row["crosswind_risk_score"] = float(normalized_by_field["crosswind"][index])
        row["wind_risk_score"] = float(
            float(wind_tail_weight) * row["tailwind_risk_score"]
            + float(wind_cross_weight) * row["crosswind_risk_score"]
        )
        row["ground_risk_score"] = float(normalized_by_field["ground"][index])
        row["air_risk_score"] = float(normalized_by_field["air"][index])
        row["combined_risk_score"] = float(
            float(wind_risk_weight) * row["wind_risk_score"]
            + float(ground_risk_weight) * row["ground_risk_score"]
            + float(air_risk_weight) * row["air_risk_score"]
        )

    metric_lookup = {
        (int(row["sector"]), str(row["direction"])): row
        for row in direction_metrics
    }
    combinations = []
    for takeoff_sector in range(1, 13):
        for landing_sector in range(1, 13):
            if takeoff_sector == landing_sector:
                continue
            takeoff = metric_lookup[(takeoff_sector, "takeoff")]
            landing = metric_lookup[(landing_sector, "landing")]
            moc_pass = bool(takeoff["moc_pass"] and landing["moc_pass"])
            risk_data_available = bool(
                takeoff["risk_data_available"] and landing["risk_data_available"]
            )
            combinations.append({
                "takeoff_sector": int(takeoff_sector),
                "landing_sector": int(landing_sector),
                "moc_pass": moc_pass,
                "risk_data_available": risk_data_available,
                "eligible": bool(moc_pass and risk_data_available),
                "moc_issue_ratio": float(
                    0.5 * (takeoff["moc_issue_ratio"] + landing["moc_issue_ratio"])
                ),
                "wind_risk_score": float(
                    0.5 * (takeoff["wind_risk_score"] + landing["wind_risk_score"])
                ),
                "ground_risk_score": float(
                    0.5 * (takeoff["ground_risk_score"] + landing["ground_risk_score"])
                ),
                "air_risk_score": float(
                    0.5 * (takeoff["air_risk_score"] + landing["air_risk_score"])
                ),
                "combined_risk_score": float(
                    0.5 * (takeoff["combined_risk_score"] + landing["combined_risk_score"])
                ),
                "takeoff_moc_blocked_cell_count": int(takeoff["moc_blocked_cell_count"]),
                "landing_moc_blocked_cell_count": int(landing["moc_blocked_cell_count"]),
                "takeoff_moc_tested_cell_count": int(takeoff["moc_tested_cell_count"]),
                "landing_moc_tested_cell_count": int(landing["moc_tested_cell_count"]),
                "takeoff_moc_out_of_grid_sample_count": int(
                    takeoff["moc_out_of_grid_sample_count"]
                ),
                "landing_moc_out_of_grid_sample_count": int(
                    landing["moc_out_of_grid_sample_count"]
                ),
            })
    if len(combinations) != 132:
        raise RuntimeError(f"Expected 132 sector pairs, generated {len(combinations)}.")

    eligible = [row for row in combinations if row["eligible"]]
    if not eligible:
        moc_safe_pair_count = int(sum(bool(row["moc_pass"]) for row in combinations))
        risk_covered_pair_count = int(
            sum(bool(row["risk_data_available"]) for row in combinations)
        )
        raise SectorSelectionInfeasibleError(
            "No MOC-safe takeoff/landing sector pair has enough wind/ground/air "
            "data; automatic selection will not choose an MOC-violating pair.",
            details=[
                {
                    "field": "sector_selection",
                    "message": f"0 of {len(combinations)} sector pairs are eligible.",
                    "type": "no_eligible_sector_pair",
                },
                {
                    "field": "sector_selection.moc",
                    "message": (
                        f"{moc_safe_pair_count} of {len(combinations)} pairs passed "
                        "mandatory MOC screening."
                    ),
                    "type": "moc_screening_summary",
                },
                {
                    "field": "sector_selection.risk_coverage",
                    "message": (
                        f"{risk_covered_pair_count} of {len(combinations)} pairs have "
                        "wind, ground-risk, and air-risk coverage."
                    ),
                    "type": "risk_coverage_summary",
                },
            ],
        )
    selected_pair = min(
        eligible,
        key=lambda row: (
            row["combined_risk_score"], row["takeoff_sector"], row["landing_sector"]
        ),
    )
    selection_status = "OPTIMAL_MOC_FEASIBLE"

    for rank, row in enumerate(sorted(
        combinations,
        key=lambda value: (
            not value["eligible"],
            value["moc_issue_ratio"] if not value["eligible"] else 0.0,
            value["combined_risk_score"],
            value["takeoff_sector"], value["landing_sector"],
        ),
    ), start=1):
        row["comparison_rank"] = int(rank)

    selected_pair = dict(selected_pair)
    selected_pair["selection_status"] = selection_status
    analysis = {
        "enabled": True,
        "mode": "automatic",
        "status": selection_status,
        "season": season,
        "wind_months": [int(value) for value in wind_months],
        "weights": {
            "wind": float(wind_risk_weight),
            "ground": float(ground_risk_weight),
            "air": float(air_risk_weight),
            "tailwind_within_wind": float(wind_tail_weight),
            "crosswind_within_wind": float(wind_cross_weight),
        },
        "moc_selection_enforced_independently_of_optimizer_switch": True,
        "selection_priority": "MOC_PASS_FIRST_THEN_MINIMUM_COMBINED_RISK",
        "nominal_screening_path_policy": "sector_center_straight_from_vertiport_to_cruise",
        "along_track_sample_spacing_m": float(along_track_step_m),
        "corridor_half_width_m": float(corridor_half_width_m),
        "sector_half_width_deg": float(sector_half_width_deg),
        "takeoff_nominal_geometry": {
            key: _sector_finite_or_none_v1(value) if key != "mode" else str(value)
            for key, value in takeoff_geometry.items()
        },
        "landing_nominal_geometry": {
            key: _sector_finite_or_none_v1(value) if key != "mode" else str(value)
            for key, value in landing_geometry.items()
        },
        "normalization": normalization,
        "direction_count": int(len(direction_metrics)),
        "combination_count": int(len(combinations)),
        "eligible_pair_count": int(len(eligible)),
        "moc_safe_pair_count": int(len(eligible)),
        "selected_pair": selected_pair,
        "direction_metrics": direction_metrics,
        "combinations": combinations,
        "diagnostic_figure": str(Path(output_png_path)),
    }
    plot_status = _plot_sector_selection_diagnostics_v1(
        analysis,
        MOCRisk,
        lat_lim,
        lon_lim,
        start_vertiport,
        end_vertiport,
        output_png_path,
        request=request,
    )
    analysis["diagnostic_figure_used_osm"] = bool(plot_status["used_osm"])
    analysis["diagnostic_figure_fallback_reason"] = plot_status["fallback_reason"]
    analysis["direction_metrics"] = [
        {key: value for key, value in row.items() if not str(key).startswith("_")}
        for row in direction_metrics
    ]
    return _sector_json_safe_v1(analysis)


def _plot_masked_chunks(ax, lons, lats, mask, **plot_kwargs):
    mask = np.asarray(mask, dtype=bool).ravel()
    if lons is None or lats is None:
        return
    if mask.size == 0 or lons.size == 0 or lats.size == 0:
        return
    if not (mask.size == lons.size == lats.size):
        return

    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return

    label = plot_kwargs.pop("label", None)
    cuts = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, idx.size - 1]
    first_label_used = False

    for s, e in zip(starts, ends):
        seg_idx = idx[s : e + 1]
        if seg_idx.size < 2:
            continue
        kwargs_i = dict(plot_kwargs)
        if label is not None and not first_label_used:
            kwargs_i["label"] = label
            first_label_used = True
        ax.plot(lons[seg_idx], lats[seg_idx], **kwargs_i)


def _plot_masked_edges(ax, lons, lats, edge_mask, **plot_kwargs):
    """Plot contiguous path edges without dropping one-edge phase spans."""
    edge_mask = np.asarray(edge_mask, dtype=bool).ravel()
    if lons is None or lats is None:
        return
    if edge_mask.size == 0 or lons.size < 2 or lats.size < 2:
        return
    if not (edge_mask.size == lons.size - 1 == lats.size - 1):
        return

    idx = np.flatnonzero(edge_mask)
    if idx.size == 0:
        return

    label = plot_kwargs.pop("label", None)
    cuts = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, idx.size - 1]
    first_label_used = False

    for s, e in zip(starts, ends):
        first_edge = int(idx[s])
        last_edge = int(idx[e])
        point_slice = slice(first_edge, last_edge + 2)
        kwargs_i = dict(plot_kwargs)
        if label is not None and not first_label_used:
            kwargs_i["label"] = label
            first_label_used = True
        ax.plot(lons[point_slice], lats[point_slice], **kwargs_i)


def save_excel_route_map_figure(xlsx_path, out_png_path=None, figure_title=None, use_takeoff_landing_transition=True):
    """
    Route_Data 시트의 경로를 지도 위에 시각화해 PNG로 저장한다.
    표시 요소:
    - Flight_Phase-based cruise, takeoff transition, landing transition
    - TF_End, RF_Start, RF arc points, Arc center
    - NFZ, Airspace boundary
    """
    xlsx_path = Path(xlsx_path)
    out_png_path = Path(out_png_path) if out_png_path is not None else (xlsx_path.parent / "fig_route_from_excel_map.png")

    try:
        route_df = pd.read_excel(str(xlsx_path), sheet_name="Route_Data")
    except Exception as e:
        print(f"Excel route map skipped (Route_Data read failed): {e}")
        return None

    if route_df is None or route_df.empty:
        print("Excel route map skipped (Route_Data is empty).")
        return None

    lat = pd.to_numeric(route_df.get("Lat"), errors="coerce")
    lon = pd.to_numeric(route_df.get("Lon"), errors="coerce")
    alt = pd.to_numeric(route_df.get("Altitude_MSL_m"), errors="coerce")
    valid = (~lat.isna()) & (~lon.isna())
    route = route_df.loc[valid].copy()
    if route.empty:
        print("Excel route map skipped (no valid Lat/Lon rows).")
        return None

    route["Lat"] = lat[valid].to_numpy(dtype=float)
    route["Lon"] = lon[valid].to_numpy(dtype=float)
    route["Altitude_MSL_m"] = alt[valid].to_numpy(dtype=float)

    input_points = pd.DataFrame(columns=["Point_Name", "Lat", "Lon", "Alt_m"])
    try:
        input_points_raw = pd.read_excel(str(xlsx_path), sheet_name="Input_Points")
        required_cols = {"Point_Name", "Lat", "Lon"}
        if required_cols.issubset(input_points_raw.columns):
            input_points = input_points_raw.copy()
            input_points["Lat"] = pd.to_numeric(input_points["Lat"], errors="coerce")
            input_points["Lon"] = pd.to_numeric(input_points["Lon"], errors="coerce")
            input_points = input_points.dropna(subset=["Lat", "Lon"])
    except Exception:
        pass

    input_point_map = {}
    for _, row in input_points.iterrows():
        point_name = str(row["Point_Name"]).strip().lower()
        input_point_map[point_name] = (float(row["Lon"]), float(row["Lat"]))

    
    fig, ax = plt.subplots(figsize=(12, 10))

    # NFZ overlay from Excel sheet
    try:
        df_nfz = pd.read_excel(str(xlsx_path), sheet_name="NFZ_Info")
        if (not df_nfz.empty) and {"Zone_ID", "Lon", "Lat", "Point_No"}.issubset(df_nfz.columns):
            for _, grp in df_nfz.groupby("Zone_ID"):
                g = grp.sort_values("Point_No")
                ax.fill(g["Lon"], g["Lat"], color="tomato", alpha=0.14, zorder=1)
                ax.plot(g["Lon"], g["Lat"], "-", color="firebrick", linewidth=1.1, alpha=0.8, zorder=2)
            ax.plot([], [], "-", color="firebrick", linewidth=1.2, label="NFZ")
    except Exception:
        pass

    # Airspace boundary overlay from Excel sheet
    try:
        df_air = pd.read_excel(str(xlsx_path), sheet_name="Airspace_Info")
        if (not df_air.empty) and {"Type", "Lon", "Lat"}.issubset(df_air.columns):
            bnd = df_air[df_air["Type"].astype(str) == "Boundary"]
            cen = df_air[df_air["Type"].astype(str) == "Center"]
            if not bnd.empty:
                ax.plot(bnd["Lon"], bnd["Lat"], "-", color="deepskyblue", linewidth=1.8, alpha=0.85,
                        label="Airspace Boundary", zorder=2)
            if not cen.empty:
                ax.scatter(cen["Lon"], cen["Lat"], c="deepskyblue", s=45, marker="x", zorder=3)
    except Exception:
        pass

    plot_route = route
    if not bool(use_takeoff_landing_transition):
        type_text_all = route["Type"].astype(str).str.lower() if "Type" in route.columns else pd.Series("", index=route.index)
        non_vertiport_mask = ~type_text_all.eq("vertiport").to_numpy(dtype=bool)
        plot_route = route.loc[non_vertiport_mask].copy()
        if plot_route.empty:
            plot_route = route.copy()
    if plot_route is None or plot_route.empty or plot_route.shape[0] < 2:
        print(
            "Excel route map skipped (insufficient points after mode filter). "
            f"len(route)={len(route)}, len(plot_route)={0 if plot_route is None else len(plot_route)}"
        )
        plt.close(fig)
        return None

    type_text = plot_route["Type"].astype(str).str.lower() if "Type" in plot_route.columns else pd.Series("", index=plot_route.index)
    has_flight_phase = "Flight_Phase" in plot_route.columns
    if has_flight_phase:
        phase_text = plot_route["Flight_Phase"].astype(str).str.strip().str.lower()
        mask_takeoff = phase_text.isin([
            FLIGHT_PHASE_TAKEOFF_STAGE1,
            FLIGHT_PHASE_TAKEOFF_STAGE2,
        ]).to_numpy(dtype=bool)
        mask_landing = phase_text.isin([
            FLIGHT_PHASE_LANDING_STAGE2,
            FLIGHT_PHASE_LANDING_STAGE1,
        ]).to_numpy(dtype=bool)
        mask_vertiport = phase_text.eq(FLIGHT_PHASE_VERTIPORT).to_numpy(dtype=bool)
        mask_cruise = phase_text.eq(FLIGHT_PHASE_CRUISE).to_numpy(dtype=bool)
    else:
        # Backward-compatible fallback for workbooks created before Flight_Phase.
        phase_text = pd.Series("", index=plot_route.index)
        mask_takeoff = type_text.str.contains("takeoff", na=False).to_numpy(dtype=bool)
        mask_landing = type_text.str.contains("landing", na=False).to_numpy(dtype=bool)
        mask_vertiport = type_text.eq("vertiport").to_numpy(dtype=bool)
        mask_cruise = ~(mask_takeoff | mask_landing | mask_vertiport)
    mask_rf = type_text.str.contains("rf_arc", na=False).to_numpy(dtype=bool)
    mask_tf = (type_text.str.contains("tf_point|takeoff_path_point|landing_path_point", na=False, regex=True)).to_numpy(dtype=bool)

    def _flag_mask(col_name):
        if col_name not in plot_route.columns:
            return np.zeros(plot_route.shape[0], dtype=bool)
        return plot_route[col_name].astype(str).str.strip().str.upper().eq("O").to_numpy(dtype=bool)

    tf_start_mask = _flag_mask("TF_Start")
    tf_end_mask = _flag_mask("TF_End")
    rf_start_mask = _flag_mask("RF_Start")
    rf_end_mask = _flag_mask("RF_End")

    lons = plot_route["Lon"].to_numpy(dtype=float)
    lats = plot_route["Lat"].to_numpy(dtype=float)
    transition_marker_plot_mask = np.zeros(plot_route.shape[0], dtype=bool)
    for point_name in (
        "takeoff_stage1_end",
        "landing_stage1_start",
        "takeoff_transition_end",
        "landing_transition_start",
    ):
        boundary_lonlat = input_point_map.get(point_name)
        if boundary_lonlat is None:
            continue
        boundary_lla = np.array([boundary_lonlat[1], boundary_lonlat[0], 0.0], dtype=float)
        for point_idx in range(plot_route.shape[0]):
            route_lla = np.array([lats[point_idx], lons[point_idx], 0.0], dtype=float)
            if _seg_dist_m(route_lla, boundary_lla) <= 0.5:
                transition_marker_plot_mask[point_idx] = True
    tf_start_mask &= ~transition_marker_plot_mask
    tf_end_mask &= ~transition_marker_plot_mask
    rf_start_mask &= ~transition_marker_plot_mask
    rf_end_mask &= ~transition_marker_plot_mask

    
    ax.plot(lons, lats, "-", color="gray", linewidth=2.5, alpha=0.4, zorder=3, label="Route Outline")
    
    if has_flight_phase:
        phase_values = phase_text.to_numpy(dtype=object)
        edge_phases = phase_values[1:].copy()
        # The final edge terminates at a vertiport; retain its source landing phase.
        vertiport_edges = edge_phases == FLIGHT_PHASE_VERTIPORT
        edge_phases[vertiport_edges] = phase_values[:-1][vertiport_edges]
        _plot_masked_edges(
            ax, lons, lats, edge_phases == FLIGHT_PHASE_CRUISE,
            color="black", linewidth=2.8, alpha=1.0, zorder=5, label="Cruise Section",
        )
        _plot_masked_edges(
            ax, lons, lats,
            np.isin(edge_phases, [FLIGHT_PHASE_TAKEOFF_STAGE1, FLIGHT_PHASE_TAKEOFF_STAGE2]),
            color=TAKEOFF_TRANSITION_COLOR, linewidth=3.2, alpha=0.95,
            zorder=6, label="Takeoff Transition",
        )
        _plot_masked_edges(
            ax, lons, lats,
            np.isin(edge_phases, [FLIGHT_PHASE_LANDING_STAGE2, FLIGHT_PHASE_LANDING_STAGE1]),
            color=LANDING_TRANSITION_COLOR, linewidth=3.2, alpha=0.95,
            zorder=6, label="Landing Transition",
        )
    else:
        _plot_masked_chunks(ax, lons, lats, mask_cruise, color="black", linewidth=2.8, alpha=1.0,
                            zorder=5, label="Cruise Section")
        _plot_masked_chunks(ax, lons, lats, mask_takeoff, color=TAKEOFF_TRANSITION_COLOR,
                            linewidth=3.2, alpha=0.95, zorder=6, label="Takeoff Transition")
        _plot_masked_chunks(ax, lons, lats, mask_landing, color=LANDING_TRANSITION_COLOR,
                            linewidth=3.2, alpha=0.95, zorder=6, label="Landing Transition")
    
    if np.any(mask_rf):
        ax.scatter(lons[mask_rf], lats[mask_rf], s=18, c="gold", alpha=0.90, edgecolors="none",
                   label="RF Arc Points", zorder=9)
    if np.any(mask_tf):
        ax.scatter(lons[mask_tf], lats[mask_tf], s=16, c="deepskyblue", alpha=0.80, edgecolors="none",
                   label="TF Points", zorder=9)

    if np.any(tf_end_mask):
        ax.scatter(lons[tf_end_mask], lats[tf_end_mask], s=90, c="blue", marker="v", edgecolors="navy",
                   linewidths=0.7, label="TF End", zorder=12)
    
    if np.any(rf_start_mask):
        ax.scatter(lons[rf_start_mask], lats[rf_start_mask], s=95, c="orange", marker=">", edgecolors="darkorange",
                   linewidths=0.7, label="RF Start", zorder=12)
    
    if np.any(rf_end_mask):
        ax.scatter(lons[rf_end_mask], lats[rf_end_mask], s=60, c="gold", marker="s", edgecolors="darkgoldenrod",
                   linewidths=0.7, label="RF End", zorder=12)

    if {"Arc_Center_Lat", "Arc_Center_Lon"}.issubset(plot_route.columns):
        aclat = pd.to_numeric(plot_route["Arc_Center_Lat"], errors="coerce").to_numpy(dtype=float)
        aclon = pd.to_numeric(plot_route["Arc_Center_Lon"], errors="coerce").to_numpy(dtype=float)
        ac_mask = (~np.isnan(aclat)) & (~np.isnan(aclon))
        if np.any(ac_mask):
            for idx in np.where(ac_mask)[0]:
                ax.scatter(aclon[idx], aclat[idx], s=50, c="black", marker="x", linewidths=2.8,
                          zorder=11)
            ax.scatter([], [], s=50, c="black", marker="x", linewidths=2.8,
                      label="RF Arc Center", zorder=11)

    waypoint_rows = input_points[
        input_points["Point_Name"].astype(str).str.startswith("WP_", na=False)
    ]
    if not waypoint_rows.empty:
        ax.scatter(
            waypoint_rows["Lon"], waypoint_rows["Lat"],
            s=60, c="orange", edgecolors="k", linewidths=0.5, marker="o",
            label="Waypoints", zorder=10,
        )

    route_type_all = route["Type"].astype(str).str.strip().str.lower()
    route_segment_all = (
        route["Segment"].astype(str).str.strip().str.lower()
        if "Segment" in route.columns
        else pd.Series("", index=route.index)
    )

    def _vertiport_from_route(segment_name):
        mask = route_type_all.eq("vertiport") & route_segment_all.eq(segment_name)
        rows = route.loc[mask]
        if rows.empty:
            return None
        return float(rows.iloc[0]["Lon"]), float(rows.iloc[0]["Lat"])

    start_vertiport_point = input_point_map.get(
        "start_vertiport", _vertiport_from_route("start")
    )
    end_vertiport_point = input_point_map.get(
        "end_vertiport", _vertiport_from_route("end")
    )
    takeoff_stage1_end_point = input_point_map.get("takeoff_stage1_end")
    landing_stage1_start_point = input_point_map.get("landing_stage1_start")
    takeoff_end_point = input_point_map.get(
        "takeoff_transition_end",
        input_point_map.get("takeoff_point", (float(lons[0]), float(lats[0]))),
    )
    landing_end_point = input_point_map.get(
        "landing_transition_start",
        input_point_map.get("landing_point", (float(lons[-1]), float(lats[-1]))),
    )

    if end_vertiport_point is not None:
        ax.scatter(
            end_vertiport_point[0], end_vertiport_point[1],
            s=180, c="crimson", edgecolors="k", linewidths=0.8, marker="D",
            label="End Vertiport", zorder=13,
        )
    if start_vertiport_point is not None:
        ax.scatter(
            start_vertiport_point[0], start_vertiport_point[1],
            s=100, c="red", edgecolors="k", linewidths=0.8, marker="s",
            label="Start Vertiport", zorder=14,
        )
    if takeoff_stage1_end_point is not None:
        ax.scatter(
            takeoff_stage1_end_point[0], takeoff_stage1_end_point[1],
            s=70, facecolors="none", edgecolors=TAKEOFF_TRANSITION_COLOR,
            linewidths=1.5, marker="o", label="Takeoff Stage1 End", zorder=15,
        )
    if landing_stage1_start_point is not None:
        ax.scatter(
            landing_stage1_start_point[0], landing_stage1_start_point[1],
            s=70, facecolors="none", edgecolors=LANDING_TRANSITION_COLOR,
            linewidths=1.5, marker="o", label="Landing Stage1 Start", zorder=15,
        )
    ax.scatter(
        takeoff_end_point[0], takeoff_end_point[1],
        s=90, c=TAKEOFF_TRANSITION_COLOR, edgecolors="k", linewidths=0.5, marker="^",
        label="Takeoff_End", zorder=15,
    )
    ax.scatter(
        landing_end_point[0], landing_end_point[1],
        s=90, c=LANDING_TRANSITION_COLOR, edgecolors="k", linewidths=0.5, marker="v",
        label="Landing_End", zorder=15,
    )

    if figure_title is None:
        figure_title = "Balanced Optimal Corridor Replotted from Excel Route_Data"
    ax.set_title(figure_title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.grid(True, alpha=0.35, linestyle="--")

    mean_lat = float(np.mean(lats))
    ax.set_aspect(1.0 / max(1e-8, np.cos(np.deg2rad(mean_lat))))
    
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    marker_label_order = [
        "Waypoints", "Start Vertiport", "End Vertiport",
        "Takeoff Stage1 End", "Landing Stage1 Start",
        "Takeoff_End", "Landing_End",
    ]
    non_marker_indices = [
        i for i, label in enumerate(legend_labels)
        if label not in marker_label_order
    ]
    marker_indices = [
        legend_labels.index(label)
        for label in marker_label_order
        if label in legend_labels
    ]
    legend_indices = non_marker_indices + marker_indices
    ax.legend(
        [legend_handles[i] for i in legend_indices],
        [legend_labels[i] for i in legend_indices],
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        fontsize=8, framealpha=0.9, frameon=True,
    )

    fig.savefig(out_png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png_path}")
    return out_png_path


# NSGA-III loop with RF-preprocessed path evaluation
def run_nsga3(
    nodes_pool, node_risk_pool, population, N_pop, Nmax, ratio,
    mutation_cfg,
    require_rf_for_parent_selection,
    mandatory_backbone,
    Norm_RT, AirRisk, use_map, f_limit, f_zones,
    alt, cs, scales, air_thr, dz,
    w_d, w_g, w_a, lat_lim, lon_lim,
    NoiseRisk, noise_floor_db, w_n,
    objective_weights,
    ground_speed_mps, bank_angle_deg, num_arc_points,
    look_ahead, look_ahead_threshold_m, look_ahead_min_scale, look_ahead_window,
    use_boundary_heading,
    W_half, check_corridor_nfz, check_corridor_moc, check_corridor_self_overlap,
    MOCRisk,
    start_vertiport, end_vertiport, landing_entry, takeoff_complete,
    airspace_center_latlon, airspace_radius_m,
    airspace_alt_min_m, airspace_alt_max_m,
    min_corridor_distance_m,
    transition_corridor_cfg,
):
    """NSGA-III with RF-turn preprocessing."""

    dummy = np.vstack([population[0][0], population[0][-1]])
    temp_f, _ = evaluate_objectives_with_constraints_gp(
        dummy, Norm_RT, AirRisk, use_map, f_limit, f_zones, dz, alt, cs, scales,
        air_thr, w_d, w_g, w_a, lat_lim, lon_lim,
        NoiseRisk=NoiseRisk, noise_floor_db=noise_floor_db, w_noise=w_n,
        W_half=W_half, check_corridor_nfz=check_corridor_nfz,
        MOCRisk=MOCRisk, check_corridor_moc=check_corridor_moc,
        check_corridor_self_overlap=check_corridor_self_overlap,
        vertiport=None, landing_entry=None, takeoff_complete=None,
        flight_phases=np.full(dummy.shape[0], FLIGHT_PHASE_CRUISE, dtype=object),
        transition_corridor_cfg=transition_corridor_cfg,
    )
    num_obj = len(temp_f)
    _validate_objective_weights_v1(objective_weights, num_obj)

    def _evaluate_one(chromo):
        chromo = _enforce_mandatory_wp_order(chromo, mandatory_backbone)
        rf = apply_rf_turns_full_corridor(
            chromo,
            start_vertiport,
            end_vertiport,
            ground_speed_mps,
            bank_angle_deg,
            num_arc_points,
            look_ahead,
            look_ahead_threshold_m,
            look_ahead_min_scale,
            look_ahead_window,
            use_boundary_heading=use_boundary_heading,
        )
        full_path = rf["path"]
        air_ok, _, _ = _is_path_inside_airspace_envelope_v1(
            full_path,
            rf.get("flight_phases"),
            airspace_center_latlon,
            airspace_radius_m,
            cruise_half_width_m=W_half,
            alt_min_m=airspace_alt_min_m,
            alt_max_m=airspace_alt_max_m,
            transition_corridor_cfg=transition_corridor_cfg,
        )
        if not air_ok:
            f_pen = np.asarray(temp_f, dtype=float) + 1e6
            return f_pen, False

        if float(min_corridor_distance_m) > 0.0:
            full_dist_m = _path_total_3d_distance_m(full_path)
            if full_dist_m + 1e-6 < float(min_corridor_distance_m):
                f_pen = np.asarray(temp_f, dtype=float) + 1e6
                return f_pen, False

        f_val, feas = evaluate_objectives_with_constraints_gp(
            full_path, Norm_RT, AirRisk, use_map, f_limit, f_zones, dz, alt, cs, scales,
            air_thr, w_d, w_g, w_a, lat_lim, lon_lim,
            NoiseRisk=NoiseRisk, noise_floor_db=noise_floor_db, w_noise=w_n,
            W_half=W_half, check_corridor_nfz=check_corridor_nfz,
            MOCRisk=MOCRisk, check_corridor_moc=check_corridor_moc,
            check_corridor_self_overlap=check_corridor_self_overlap,
            vertiport=None, landing_entry=None, takeoff_complete=None,
            flight_phases=rf.get("flight_phases"),
            transition_corridor_cfg=transition_corridor_cfg,
        )
        if not rf["feasible"]:
            f_val = np.asarray(f_val, dtype=float) + 1e6
        return f_val, bool(feas and rf["feasible"])

    pop = list(population[:N_pop]) if len(population) > N_pop else list(population)
    pop = [_enforce_mandatory_wp_order(p, mandatory_backbone) for p in pop]
    last_success_pop = []
    gen_history = []

    for gen in range(1, Nmax + 1):
        Np = len(pop)
        parents_count = int(Np)
        parents_unique = _unique_solution_count(pop)
        f_vals = np.zeros((Np, num_obj), dtype=float)
        feasible = np.zeros(Np, dtype=bool)

        for i in range(Np):
            f_vals[i], feasible[i] = _evaluate_one(pop[i])

        num_feas = int(np.sum(feasible))
        rf_mask = (f_vals[:, 0] < 1e6)  # +1e6 페널티가 없으면 RF 기하적으로 feasible
        rf_feas = int(np.sum(rf_mask))
        if require_rf_for_parent_selection:
            selection_mask = feasible & rf_mask
        else:
            selection_mask = feasible
        sel_feas = int(np.sum(selection_mask))
        print(f"[Gen {gen}] pop {Np}  |  constraint_feasible: {num_feas}/{Np}  |  RF_feasible: {rf_feas}/{Np}")
        if require_rf_for_parent_selection:
            print(f"[Gen {gen}] parent_selection_feasible(constraint AND RF): {sel_feas}/{Np}")
        else:
            print(f"[Gen {gen}] parent_selection_feasible(constraint only): {sel_feas}/{Np}")

        new_pop = selection_nsga3(
            pop, f_vals, selection_mask, N_pop, objective_weights
        )
        carry_over_used = False

        if new_pop:
            last_success_pop = list(new_pop)
        else:
            new_pop = list(last_success_pop)
            carry_over_used = True
            print(f"[Gen {gen}] no parent-selectable solution in current generation; carrying over {len(new_pop)} previous feasible parent(s).")

        if not new_pop:
            return [], np.empty((0, num_obj)), gen_history

        gp = list(new_pop)
        gf = np.zeros((len(gp), num_obj), dtype=float)
        gfeas = np.zeros(len(gp), dtype=bool)
        for gi in range(len(gp)):
            gf[gi], gfeas[gi] = _evaluate_one(gp[gi])
        gen_history.append({
            "gen": gen,
            "population": gp,
            "f_vals": gf,
            "feasible": gfeas,
        })

        selected_count = len(new_pop)
        selected_unique = _unique_solution_count(new_pop)

        if gen < Nmax:
            offspring = variation_nsga3(
                new_pop,
                nodes_pool,
                ratio,
                node_risks=node_risk_pool,
                mutation_cfg=mutation_cfg,
            )
            offspring = [_enforce_mandatory_wp_order(ch, mandatory_backbone) for ch in offspring]
            offspring_count = len(offspring)
            offspring_unique = _unique_solution_count(offspring)
            pop_next = new_pop + offspring
        else:
            offspring = []
            offspring_count = 0
            offspring_unique = 0
            pop_next = new_pop

        next_count = len(pop_next)
        next_unique = _unique_solution_count(pop_next)

        print(
            f"[Gen {gen}] parents: {parents_count} (unique {parents_unique}) | "
            f"selected(new_pop): {selected_count} (unique {selected_unique}) | "
            f"offspring: {offspring_count} (unique {offspring_unique}) | "
            f"next_pop: {next_count} (unique {next_unique}) | "
            f"carry_over: {int(carry_over_used)}"
        )
        pop = [_enforce_mandatory_wp_order(pn, mandatory_backbone) for pn in pop_next]

    if not pop:
        return [], np.empty((0, num_obj)), gen_history

    f_final = np.zeros((len(pop), num_obj), dtype=float)
    for i in range(len(pop)):
        f_final[i], _ = _evaluate_one(pop[i])

    return pop, f_final, gen_history


def _representative_indices_v1(f_vals, objective_weights, feasible=None):
    values = np.asarray(f_vals, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        return []
    if feasible is None:
        valid_indices = np.arange(values.shape[0], dtype=int)
    else:
        feasible_mask = np.asarray(feasible, dtype=bool).reshape(-1)
        if feasible_mask.size != values.shape[0]:
            raise ValueError(
                "objective_feasibility_length_mismatch: "
                f"objectives={values.shape[0]}, feasible={feasible_mask.size}"
            )
        valid_indices = np.flatnonzero(feasible_mask)
    if valid_indices.size == 0:
        return []

    valid_values = values[valid_indices]
    representative_indices = [
        int(valid_indices[int(np.argmin(valid_values[:, objective_idx]))])
        for objective_idx in range(values.shape[1])
    ]
    local_fronts = fast_non_dominated_sort(valid_values)
    if local_fronts and local_fronts[0]:
        front_local = np.asarray(local_fronts[0], dtype=int)
        front_global = valid_indices[front_local]
        preference = _weighted_normalized_objective_analysis_v1(
            values[front_global], objective_weights
        )
        balanced_local = int(np.argmin(preference["scores"]))
        representative_indices.append(int(front_global[balanced_local]))
    else:
        representative_indices.append(representative_indices[0])
    return representative_indices


def _balanced_objective_audit_v1(
    f_vals,
    objective_weights,
    objective_names,
    feasible,
    balanced_index,
):
    """Build JSON-safe audit data for the final weighted Balanced selection."""
    values = np.asarray(f_vals, dtype=float)
    feasible_mask = np.asarray(feasible, dtype=bool).reshape(-1)
    valid_indices = np.flatnonzero(feasible_mask)
    if valid_indices.size == 0 or balanced_index is None:
        return {
            "status": "NOT AVAILABLE",
            "formula": "sum(normalized_weight_i * minmax_normalized_objective_i)",
        }
    valid_values = values[valid_indices]
    local_fronts = fast_non_dominated_sort(valid_values)
    front_local = np.asarray(local_fronts[0], dtype=int)
    front_global = valid_indices[front_local]
    preference = _weighted_normalized_objective_analysis_v1(
        values[front_global], objective_weights
    )
    selected_matches = np.flatnonzero(front_global == int(balanced_index))
    if selected_matches.size != 1:
        raise ValueError("balanced_index_not_in_first_pareto_front")
    selected_local = int(selected_matches[0])
    names = [str(value) for value in objective_names]
    return {
        "status": "APPLIED",
        "formula": "sum(normalized_weight_i * minmax_normalized_objective_i)",
        "normalization_scope": "final_feasible_first_pareto_front",
        "objective_values_are_raw": True,
        "objective_names": names,
        "configured_weights": {
            name: float(preference["configured_weights"][idx])
            for idx, name in enumerate(names)
        },
        "normalized_weights": {
            name: float(preference["normalized_weights"][idx])
            for idx, name in enumerate(names)
        },
        "normalization_minimum": {
            name: float(preference["minimum"][idx])
            for idx, name in enumerate(names)
        },
        "normalization_maximum": {
            name: float(preference["maximum"][idx])
            for idx, name in enumerate(names)
        },
        "selected_raw_objectives": {
            name: float(values[int(balanced_index), idx])
            for idx, name in enumerate(names)
        },
        "selected_normalized_objectives": {
            name: float(preference["normalized_values"][selected_local, idx])
            for idx, name in enumerate(names)
        },
        "selected_weighted_contributions": {
            name: float(preference["weighted_contributions"][selected_local, idx])
            for idx, name in enumerate(names)
        },
        "weighted_normalized_objective_score": float(
            preference["scores"][selected_local]
        ),
        "balanced_population_index": int(balanced_index),
        "pareto_front_population_indices": [int(value) for value in front_global],
    }


def pick_representatives(population, f_vals, objective_weights, feasible=None):
    if not population or f_vals.size == 0:
        return []
    indices = _representative_indices_v1(
        f_vals, objective_weights, feasible=feasible
    )
    return [population[index] for index in indices]


def _plot_rf_segments_by_phase(
    ax,
    rf,
    cruise_color="black",
    tf_lw=1.5,
    rf_lw=2.0,
    transform=None,
    zorder=8,
    transition_labels=False,
    draw_rf_markers=False,
    linestyle="-",
    include_fixed_stage1=True,
):
    """Plot TF/RF geometry while keeping transition phases blue/green."""
    label_used = {"takeoff": False, "landing": False}
    plot_transform = ccrs.Geodetic() if transform is None else transform

    for seg in rf.get("segments", []):
        if (
            not include_fixed_stage1
            and bool(seg.get("is_fixed_transition_stage1", False))
        ):
            continue
        pts = np.asarray(seg.get("points", np.empty((0, 3))), dtype=float).reshape(-1, 3)
        if pts.shape[0] < 2:
            continue
        phases = np.asarray(
            seg.get("point_phases", np.full(pts.shape[0], FLIGHT_PHASE_CRUISE, dtype=object)),
            dtype=object,
        ).reshape(-1)
        if phases.size != pts.shape[0]:
            phases = np.full(pts.shape[0], FLIGHT_PHASE_CRUISE, dtype=object)

        edge_phases = []
        for i in range(pts.shape[0] - 1):
            phase = str(phases[i + 1])
            if phase == FLIGHT_PHASE_VERTIPORT:
                phase = str(phases[i])
            edge_phases.append(phase)

        start = 0
        while start < len(edge_phases):
            phase = edge_phases[start]
            end = start + 1
            while end < len(edge_phases) and edge_phases[end] == phase:
                end += 1
            if phase.startswith("takeoff_"):
                color = TAKEOFF_TRANSITION_COLOR
                phase_group = "takeoff"
                label = "Takeoff Transition" if transition_labels and not label_used[phase_group] else None
            elif phase.startswith("landing_"):
                color = LANDING_TRANSITION_COLOR
                phase_group = "landing"
                label = "Landing Transition" if transition_labels and not label_used[phase_group] else None
            else:
                color = cruise_color
                phase_group = None
                label = None
            ax.plot(
                pts[start:end + 1, 1],
                pts[start:end + 1, 0],
                linestyle,
                color=color,
                linewidth=rf_lw if seg.get("type") == "RF" else tf_lw,
                transform=plot_transform,
                zorder=zorder + (1 if phase != FLIGHT_PHASE_CRUISE else 0),
                label=label,
            )
            if phase_group is not None:
                label_used[phase_group] = True
            start = end

        if draw_rf_markers and seg.get("type") == "RF":
            ax.scatter(
                pts[0, 1], pts[0, 0], s=40, c="yellow", marker=">",
                edgecolors="k", linewidths=0.6, transform=plot_transform, zorder=zorder + 2,
            )
            ax.scatter(
                pts[-1, 1], pts[-1, 0], s=40, c="yellow", marker="s",
                edgecolors="k", linewidths=0.6, transform=plot_transform, zorder=zorder + 2,
            )
            arc_center = np.asarray(seg.get("arc_center", np.empty(0)), dtype=float).reshape(-1)
            if arc_center.size >= 2:
                ax.scatter(
                    arc_center[1], arc_center[0], s=50, c="white", marker="x",
                    linewidths=1.5, transform=plot_transform, zorder=zorder + 2,
                )


def _plot_transition_phase_markers(
    ax,
    rf,
    transform=None,
    zorder=12,
    labels=False,
    include_stage1=True,
    direction=None,
    general_output=False,
):
    """Plot stage-1 boundaries and dynamic final transition endpoints."""
    plot_transform = ccrs.Geodetic() if transform is None else transform
    meta = rf.get("transition_meta", {}) if isinstance(rf, dict) else {}
    structure_mode = str(meta.get("transition_structure_mode", "")).strip().lower()

    takeoff_final = meta.get("takeoff_transition_end")
    landing_final = meta.get("landing_transition_end")
    if general_output:
        if structure_mode == TRANSITION_STRUCTURE_FIXED_ONLY:
            # General fixed-only figures already draw the single static endpoint pair.
            takeoff_final = None
            landing_final = None
        else:
            if not bool(meta.get("takeoff_optimized_transition_actual", takeoff_final is not None)):
                takeoff_final = None
            if not bool(meta.get("landing_optimized_transition_actual", landing_final is not None)):
                landing_final = None

    takeoff_stage1 = (
        meta.get("takeoff_stage1_end")
        if include_stage1 and float(meta.get("takeoff_stage1_straight_distance_m", 0.0)) > 0.5
        else None
    )
    landing_stage1 = (
        meta.get("landing_stage1_start")
        if include_stage1 and float(meta.get("landing_stage1_straight_distance_m", 0.0)) > 0.5
        else None
    )
    marker_specs = [
        (takeoff_stage1, TAKEOFF_TRANSITION_COLOR, "o", True, "Takeoff Stage1 End"),
        (landing_stage1, LANDING_TRANSITION_COLOR, "o", True, "Landing Stage1 Start"),
        (takeoff_final, TAKEOFF_TRANSITION_COLOR, "^", False, "Takeoff Transition End"),
        (landing_final, LANDING_TRANSITION_COLOR, "v", False, "Landing Transition Start"),
    ]
    if direction == "takeoff":
        marker_specs = [marker_specs[0], marker_specs[2]]
    elif direction == "landing":
        marker_specs = [marker_specs[1], marker_specs[3]]
    solid_points = []
    for point, _, _, hollow, _ in marker_specs:
        if hollow or point is None:
            continue
        p = np.asarray(point, dtype=float).reshape(-1)
        if p.size >= 2 and np.all(np.isfinite(p[:2])):
            solid_points.append(p)
    seen = []
    for point, color, marker, hollow, label in marker_specs:
        if point is None:
            continue
        p = np.asarray(point, dtype=float).reshape(-1)
        if p.size < 2 or not np.all(np.isfinite(p[:2])):
            continue
        if hollow and any(
            _seg_dist_m(np.r_[p[:2], 0.0], np.r_[q[:2], 0.0]) <= 0.5
            for q in solid_points
        ):
            continue
        duplicate = any(_seg_dist_m(np.r_[p[:2], 0.0], np.r_[q[:2], 0.0]) <= 0.5 for q in seen)
        if duplicate and hollow:
            continue
        seen.append(p)
        kwargs = {
            "s": 90,
            "marker": marker,
            "linewidths": 1.4 if hollow else 0.7,
            "transform": plot_transform,
            "zorder": zorder,
            "label": label if labels else None,
        }
        if hollow:
            kwargs.update(facecolors="none", edgecolors=color)
        else:
            kwargs.update(c=color, edgecolors="k")
        ax.scatter([p[1]], [p[0]], **kwargs)


def _edge_phase_for_transition_v1(point_phases, edge_idx):
    phases = np.asarray(point_phases, dtype=object).reshape(-1)
    phase = str(phases[int(edge_idx) + 1])
    if phase == FLIGHT_PHASE_VERTIPORT:
        phase = str(phases[int(edge_idx)])
    return phase


def _transition_direction_from_phase_v1(phase):
    return _transition_direction_from_phase_value_v1(phase)


def _moc_envelope_for_cells_v1(moc_risk, cells):
    """Return the discrete first-clear boundary for unique in-grid MOC cells."""
    moc = np.asarray(moc_risk, dtype=float)
    if moc.ndim == 2:
        moc = moc[:, :, np.newaxis]
    unique_cells = sorted({(int(row), int(col)) for row, col in cells})
    if not unique_cells:
        return {
            "class": "OUT_OF_GRID",
            "highest_blocked_agl_m": None,
            "first_clear_agl_m": None,
            "required_safe_msl_m": None,
        }
    blocked_by_layer = np.zeros(moc.shape[2], dtype=bool)
    for row, col in unique_cells:
        blocked_by_layer |= np.asarray(moc[row, col, :] >= 0.5, dtype=bool)
    blocked_indices = np.flatnonzero(blocked_by_layer)
    if blocked_indices.size == 0:
        return {
            "class": "ALL_CLEAR",
            "highest_blocked_agl_m": None,
            "first_clear_agl_m": float(MOC_AGL_LEVELS_M[0]),
            "required_safe_msl_m": None,
        }
    highest_idx = int(blocked_indices[-1])
    highest_agl_m = float(MOC_AGL_LEVELS_M[highest_idx])
    if highest_idx >= moc.shape[2] - 1:
        return {
            "class": "NO_CLEAR_LAYER",
            "highest_blocked_agl_m": highest_agl_m,
            "first_clear_agl_m": None,
            "required_safe_msl_m": None,
        }
    first_clear_agl_m = float(MOC_AGL_LEVELS_M[highest_idx + 1])
    return {
        "class": "FINITE_CEILING",
        "highest_blocked_agl_m": highest_agl_m,
        "first_clear_agl_m": first_clear_agl_m,
        "required_safe_msl_m": float(MOC_REFERENCE_MSL_M + first_clear_agl_m),
    }


def _collect_transition_moc_samples_v1(
    rf,
    half_width_m,
    moc_risk,
    lat_lim,
    lon_lim,
    along_step_m=80.0,
    transition_corridor_cfg=None,
):
    """Collect auditable MOC footprint samples for transition TF/RF edges."""
    records = []
    cumulative_m = 0.0

    for segment_idx, seg in enumerate(rf.get("segments", []), start=1):
        pts = np.asarray(seg.get("points", np.empty((0, 3))), dtype=float).reshape(-1, 3)
        if pts.shape[0] < 2:
            continue
        phases = np.asarray(
            seg.get(
                "point_phases",
                np.full(pts.shape[0], FLIGHT_PHASE_CRUISE, dtype=object),
            ),
            dtype=object,
        ).reshape(-1)
        if phases.size != pts.shape[0]:
            raise ValueError(
                "flight_phase_length_mismatch in transition MOC collection: "
                f"points={pts.shape[0]}, phases={phases.size}."
            )
        valid_point_phases = set(TRANSITION_PHASE_NAMES) | {
            FLIGHT_PHASE_CRUISE,
            FLIGHT_PHASE_VERTIPORT,
        }
        invalid_phases = sorted({
            str(value) for value in phases
            if str(value) not in valid_point_phases
        })
        if invalid_phases:
            raise ValueError(
                "invalid_flight_phase in transition MOC collection: "
                + ", ".join(invalid_phases)
            )

        segment_type = str(seg.get("type", "TF"))
        turn_radius_m = (
            float(seg.get("turn_radius"))
            if segment_type == "RF" and seg.get("turn_radius") is not None
            else float("nan")
        )
        turn_angle_deg = (
            float(np.rad2deg(float(seg.get("turn_angle", 0.0))))
            if segment_type == "RF" else float("nan")
        )

        for edge_idx in range(pts.shape[0] - 1):
            p1 = pts[edge_idx]
            p2 = pts[edge_idx + 1]
            edge_length_m = _seg_dist_m(p1, p2)
            source_point_phase = str(phases[edge_idx])
            destination_point_phase = str(phases[edge_idx + 1])
            phase = _edge_phase_for_transition_v1(phases, edge_idx)
            direction = _transition_direction_from_phase_v1(phase)
            if direction is not None:
                along_sample_idx = -1
                previous_edge_s_m = None
                for sample in (
                    _iter_corridor_moc_samples_v1(
                        p1,
                        p2,
                        half_width_m,
                        moc_risk,
                        lat_lim,
                        lon_lim,
                        along_step_m=along_step_m,
                        phase=phase,
                        transition_corridor_cfg=transition_corridor_cfg,
                    )
                ):
                    (
                        edge_s_m,
                        cross_track_m,
                        tau,
                        sample_lat,
                        sample_lon,
                        altitude_msl_m,
                        center_layer_idx,
                        effective_clearance_m,
                        query_altitude_msl_m,
                        layer_idx,
                        grid_row,
                        grid_col,
                        in_grid,
                        blocked,
                    ) = sample
                    if (
                        previous_edge_s_m is None
                        or abs(float(edge_s_m) - float(previous_edge_s_m)) > 1e-9
                    ):
                        along_sample_idx += 1
                        previous_edge_s_m = float(edge_s_m)
                    center_lat = float(p1[0] + tau * (p2[0] - p1[0]))
                    center_lon = float(p1[1] + tau * (p2[1] - p1[1]))
                    moc_agl_m = float(
                        MOC_AGL_LEVELS_M[
                            int(np.clip(layer_idx, 0, len(MOC_AGL_LEVELS_M) - 1))
                        ]
                    )
                    center_moc_agl_m = float(
                        MOC_AGL_LEVELS_M[
                            int(np.clip(center_layer_idx, 0, len(MOC_AGL_LEVELS_M) - 1))
                        ]
                    )
                    cell_envelope = _moc_envelope_for_cells_v1(
                        moc_risk,
                        [(grid_row, grid_col)] if in_grid else [],
                    )
                    records.append({
                        "Direction": direction,
                        "Flight_Phase": phase,
                        "Source_Point_Phase": source_point_phase,
                        "Destination_Point_Phase": destination_point_phase,
                        "Segment_Type": segment_type,
                        "Segment_Index": int(segment_idx),
                        "Edge_Index": int(edge_idx),
                        "Along_Sample_Index": int(along_sample_idx),
                        "Along_Track_m": float(cumulative_m + edge_s_m),
                        "Direction_Along_Track_m": 0.0,
                        "Edge_Along_Track_m": float(edge_s_m),
                        "Cross_Track_Offset_m": float(cross_track_m),
                        "Center_Lat": center_lat,
                        "Center_Lon": center_lon,
                        "Lat": float(sample_lat),
                        "Lon": float(sample_lon),
                        "Altitude_MSL_m": float(altitude_msl_m),
                        "Altitude_AGL_m": float(altitude_msl_m - MOC_REFERENCE_MSL_M),
                        "Center_Altitude_MSL_m": float(altitude_msl_m),
                        "Center_Altitude_AGL_m": float(
                            altitude_msl_m - MOC_REFERENCE_MSL_M
                        ),
                        "Selected_MOC_Layer_Index": int(layer_idx),
                        "Selected_MOC_AGL_m": moc_agl_m,
                        "MOC_Reference_MSL_m": float(MOC_REFERENCE_MSL_M + moc_agl_m),
                        "Grid_Row": int(grid_row),
                        "Grid_Col": int(grid_col),
                        "Grid_Status": "INSIDE" if in_grid else "OUT_OF_GRID",
                        "Sample_Status": (
                            "MOC_INTERSECTION"
                            if blocked else (
                                "CLEAR" if in_grid else "OUT_OF_GRID/WARN"
                            )
                        ),
                        "Blocked": bool(blocked),
                        "RF_Turn_Radius_m": turn_radius_m,
                        "RF_Turn_Angle_deg": turn_angle_deg,
                        "Fixed_Stage1": bool(seg.get("is_fixed_transition_stage1", False)),
                        "Transition_Corridor_Half_Width_m": float(half_width_m),
                        "Configured_Downward_Clearance_m": float(
                            (transition_corridor_cfg or {}).get("downward_clearance_m", 0.0)
                        ),
                        "Effective_Downward_Clearance_m": float(effective_clearance_m),
                        "Clearance_Taper_Active": bool(
                            effective_clearance_m + 1e-9
                            < float((transition_corridor_cfg or {}).get("downward_clearance_m", 0.0))
                        ),
                        "Corridor_Lower_Face_MSL_m": float(query_altitude_msl_m),
                        "Corridor_Lower_Face_AGL_m": float(
                            query_altitude_msl_m - MOC_REFERENCE_MSL_M
                        ),
                        "MOC_Query_MSL_m": float(query_altitude_msl_m),
                        "MOC_Query_AGL_m": float(
                            query_altitude_msl_m - MOC_REFERENCE_MSL_M
                        ),
                        "Center_MOC_Layer_Index": int(center_layer_idx),
                        "Center_MOC_AGL_m": center_moc_agl_m,
                        "Center_MOC_Reference_MSL_m": float(
                            MOC_REFERENCE_MSL_M + center_moc_agl_m
                        ),
                        "Clearance_MOC_Layer_Index": int(layer_idx),
                        "Clearance_MOC_AGL_m": moc_agl_m,
                        "Clearance_MOC_Reference_MSL_m": float(
                            MOC_REFERENCE_MSL_M + moc_agl_m
                        ),
                        "Cell_MOC_Envelope_Class": str(cell_envelope["class"]),
                        "Cell_Highest_Blocked_MOC_AGL_m": cell_envelope[
                            "highest_blocked_agl_m"
                        ],
                        "Cell_First_Clear_MOC_AGL_m": cell_envelope[
                            "first_clear_agl_m"
                        ],
                        "Cell_Required_Safe_MSL_m": cell_envelope[
                            "required_safe_msl_m"
                        ],
                    })
            cumulative_m += edge_length_m

    columns = [
        "Direction", "Flight_Phase", "Source_Point_Phase",
        "Destination_Point_Phase", "Segment_Type", "Segment_Index",
        "Edge_Index", "Along_Sample_Index", "Along_Track_m",
        "Direction_Along_Track_m", "Edge_Along_Track_m",
        "Cross_Track_Offset_m", "Center_Lat", "Center_Lon", "Lat", "Lon",
        "Altitude_MSL_m", "Altitude_AGL_m", "Selected_MOC_Layer_Index",
        "Center_Altitude_MSL_m", "Center_Altitude_AGL_m",
        "Selected_MOC_AGL_m", "MOC_Reference_MSL_m", "Grid_Row", "Grid_Col",
        "Grid_Status", "Sample_Status", "Blocked", "RF_Turn_Radius_m",
        "RF_Turn_Angle_deg", "Fixed_Stage1",
        "Transition_Corridor_Half_Width_m",
        "Configured_Downward_Clearance_m",
        "Effective_Downward_Clearance_m", "Clearance_Taper_Active",
        "Corridor_Lower_Face_MSL_m", "Corridor_Lower_Face_AGL_m",
        "MOC_Query_MSL_m", "MOC_Query_AGL_m",
        "Center_MOC_Layer_Index", "Center_MOC_AGL_m",
        "Center_MOC_Reference_MSL_m", "Clearance_MOC_Layer_Index",
        "Clearance_MOC_AGL_m", "Clearance_MOC_Reference_MSL_m",
        "Cell_MOC_Envelope_Class", "Cell_Highest_Blocked_MOC_AGL_m",
        "Cell_First_Clear_MOC_AGL_m", "Cell_Required_Safe_MSL_m",
    ]
    df = pd.DataFrame(records, columns=columns)
    if not df.empty:
        # Adjacent TF/RF edges legitimately use different cross-track normals.
        # Remove only physically identical endpoint samples; keep distinct RF
        # boundary footprint samples even when they round to the same grid cell.
        df["_Dedup_Along_m"] = np.round(df["Along_Track_m"].to_numpy(dtype=float), 3)
        df["_Dedup_Lat"] = np.round(df["Lat"].to_numpy(dtype=float), 10)
        df["_Dedup_Lon"] = np.round(df["Lon"].to_numpy(dtype=float), 10)
        df["_Dedup_Alt_m"] = np.round(df["Altitude_MSL_m"].to_numpy(dtype=float), 6)
        df = df.drop_duplicates(
            [
                "Direction",
                "_Dedup_Along_m",
                "_Dedup_Lat",
                "_Dedup_Lon",
                "_Dedup_Alt_m",
                "Selected_MOC_Layer_Index",
                "Grid_Status",
            ],
            keep="first",
        ).drop(
            columns=["_Dedup_Along_m", "_Dedup_Lat", "_Dedup_Lon", "_Dedup_Alt_m"]
        ).reset_index(drop=True)
        for direction in ("takeoff", "landing"):
            mask = df["Direction"].eq(direction)
            if bool(mask.any()):
                start_m = float(df.loc[mask, "Along_Track_m"].min())
                df.loc[mask, "Direction_Along_Track_m"] = (
                    df.loc[mask, "Along_Track_m"] - start_m
                )
        df["_Station_Key_m"] = np.round(
            df["Direction_Along_Track_m"].to_numpy(dtype=float), 3
        )
        df["Direction_Station_Index"] = -1
        df["Station_MOC_Envelope_Class"] = "OUT_OF_GRID"
        df["Station_Highest_Blocked_MOC_AGL_m"] = np.nan
        df["Station_First_Clear_MOC_AGL_m"] = np.nan
        df["Station_Required_Safe_MSL_m"] = np.nan
        df["Station_Vertical_Margin_m"] = np.nan
        df["Station_3D_Status"] = "OUT_OF_GRID/WARN"
        for direction in ("takeoff", "landing"):
            station_no = 0
            direction_rows = df.loc[df["Direction"].eq(direction)]
            for station_key, station_rows in direction_rows.groupby(
                "_Station_Key_m", sort=True
            ):
                station_no += 1
                station_index = station_rows.index
                cells = [
                    (row, col)
                    for row, col, in_grid in zip(
                        station_rows["Grid_Row"],
                        station_rows["Grid_Col"],
                        station_rows["Grid_Status"].eq("INSIDE"),
                    )
                    if bool(in_grid)
                ]
                envelope = _moc_envelope_for_cells_v1(moc_risk, cells)
                lower_face_msl = float(
                    station_rows["Corridor_Lower_Face_MSL_m"].min()
                )
                required_safe_msl = envelope["required_safe_msl_m"]
                margin_m = (
                    None
                    if required_safe_msl is None
                    else float(lower_face_msl - float(required_safe_msl))
                )
                any_blocked = bool(station_rows["Blocked"].any())
                any_out = bool(station_rows["Grid_Status"].ne("INSIDE").any())
                if envelope["class"] == "NO_CLEAR_LAYER":
                    station_status = "NO_CLEAR_LAYER/FAIL"
                elif any_blocked:
                    station_status = "MOC_INTERSECTION/FAIL"
                elif any_out:
                    station_status = "OUT_OF_GRID/WARN"
                elif margin_m is not None and margin_m < -1e-6:
                    station_status = "MOC_INTERSECTION/FAIL"
                elif margin_m is not None and abs(margin_m) <= 1e-6:
                    station_status = "EXACT_LIMIT/PASS"
                elif envelope["class"] == "ALL_CLEAR":
                    station_status = "ALL_CLEAR/PASS"
                else:
                    station_status = "CLEAR/PASS"
                df.loc[station_index, "Direction_Station_Index"] = int(station_no)
                df.loc[station_index, "Station_MOC_Envelope_Class"] = str(
                    envelope["class"]
                )
                for column_name, value in (
                    ("Station_Highest_Blocked_MOC_AGL_m", envelope["highest_blocked_agl_m"]),
                    ("Station_First_Clear_MOC_AGL_m", envelope["first_clear_agl_m"]),
                    ("Station_Required_Safe_MSL_m", required_safe_msl),
                    ("Station_Vertical_Margin_m", margin_m),
                ):
                    if value is not None:
                        df.loc[station_index, column_name] = float(value)
                df.loc[station_index, "Station_3D_Status"] = station_status
        df = df.drop(columns=["_Station_Key_m"])
    return df


def _transition_moc_checker_hit_v1(
    rf,
    half_width_m,
    moc_risk,
    lat_lim,
    lon_lim,
    along_step_m=80.0,
    transition_corridor_cfg=None,
):
    """Re-run the binary checker on transition edges only for audit matching."""
    for seg in rf.get("segments", []):
        pts = np.asarray(seg.get("points", np.empty((0, 3))), dtype=float).reshape(-1, 3)
        if pts.shape[0] < 2:
            continue
        phases = np.asarray(
            seg.get(
                "point_phases",
                np.full(pts.shape[0], FLIGHT_PHASE_CRUISE, dtype=object),
            ),
            dtype=object,
        ).reshape(-1)
        if phases.size != pts.shape[0]:
            raise ValueError(
                "flight_phase_length_mismatch in transition MOC checker."
            )
        for edge_idx in range(pts.shape[0] - 1):
            phase = _edge_phase_for_transition_v1(phases, edge_idx)
            if _transition_direction_from_phase_v1(phase) is None:
                continue
            if _corridor_hits_moc_v1(
                pts[edge_idx],
                pts[edge_idx + 1],
                half_width_m,
                moc_risk,
                lat_lim,
                lon_lim,
                along_step_m=along_step_m,
                phase=phase,
                transition_corridor_cfg=transition_corridor_cfg,
            ):
                return True
    return False


def _full_path_moc_checker_hit_v1(
    rf,
    cruise_half_width_m,
    transition_corridor_cfg,
    moc_risk,
    lat_lim,
    lon_lim,
    along_step_m=80.0,
):
    for seg in rf.get("segments", []):
        points = np.asarray(seg.get("points", np.empty((0, 3))), dtype=float).reshape(-1, 3)
        if points.shape[0] < 2:
            continue
        phases = np.asarray(seg.get("point_phases", np.empty((0,), dtype=object)), dtype=object).reshape(-1)
        if phases.size != points.shape[0]:
            raise ValueError("flight_phase_length_mismatch in full-path MOC checker.")
        for edge_idx in range(points.shape[0] - 1):
            phase = _edge_phase_for_transition_v1(phases, edge_idx)
            half_width_m = _edge_corridor_half_width_v1(
                phase,
                cruise_half_width_m,
                transition_corridor_cfg,
            )
            if _corridor_hits_moc_v1(
                points[edge_idx],
                points[edge_idx + 1],
                half_width_m,
                moc_risk,
                lat_lim,
                lon_lim,
                along_step_m=along_step_m,
                phase=phase,
                transition_corridor_cfg=transition_corridor_cfg,
            ):
                return True
    return False


def _moc_diagnostic_status_v1(layer_rows, moc_enforced):
    hits = int(np.count_nonzero(layer_rows["Blocked"].to_numpy(dtype=bool)))
    out_of_grid = int(np.count_nonzero(layer_rows["Grid_Status"].ne("INSIDE")))
    if not bool(moc_enforced):
        return "NOT ENFORCED", hits, out_of_grid
    if hits > 0:
        return "FAIL", hits, out_of_grid
    if out_of_grid > 0:
        return "WARN", hits, out_of_grid
    return "PASS", hits, out_of_grid


def _build_transition_3d_validation_v1(
    rf,
    cruise_half_width_m,
    transition_corridor_cfg,
    moc_risk,
    moc_enforced,
    lat_lim,
    lon_lim,
    forbidden_zones,
    check_corridor_nfz,
    check_corridor_self_overlap,
    airspace_audit,
):
    transition_corridor_cfg = _validate_transition_corridor_cfg_v1(
        transition_corridor_cfg
    )
    if not bool(transition_corridor_cfg.get("enabled", False)):
        direction_status = {
            "status": "NOT APPLICABLE",
            "moc_status": "NOT APPLICABLE",
            "nfz_status": "NOT APPLICABLE",
            "airspace_status": "NOT APPLICABLE",
            "self_overlap_status": "NOT APPLICABLE",
            "fail_reasons": [],
        }
        return {
            "status": "NOT APPLICABLE",
            "reason": "transition_disabled",
            "moc_status": "NOT APPLICABLE",
            "nfz_status": "NOT APPLICABLE",
            "airspace_status": "NOT APPLICABLE",
            "self_overlap_status": "NOT APPLICABLE",
            "directions": {
                "takeoff": dict(direction_status),
                "landing": dict(direction_status),
            },
        }
    half_width_m = float(transition_corridor_cfg["half_width_m"])
    samples = _collect_transition_moc_samples_v1(
        rf,
        half_width_m,
        moc_risk,
        lat_lim,
        lon_lim,
        transition_corridor_cfg=transition_corridor_cfg,
    )
    if samples.empty:
        return {
            "status": "FAIL",
            "reason": "transition_moc_samples_missing",
            "directions": {},
        }
    moc_status, hit_count, out_count = _moc_diagnostic_status_v1(samples, moc_enforced)
    transition_checker_hit = _transition_moc_checker_hit_v1(
        rf,
        half_width_m,
        moc_risk,
        lat_lim,
        lon_lim,
        transition_corridor_cfg=transition_corridor_cfg,
    )
    checker_match = bool(bool(samples["Blocked"].any()) == bool(transition_checker_hit))
    full_path = np.asarray(rf.get("path", np.empty((0, 3))), dtype=float).reshape(-1, 3)
    edge_phases = _edge_phase_values_v1(
        full_path, rf.get("flight_phases"), transition_corridor_cfg
    )
    if bool(check_corridor_nfz):
        nfz_ok, nfz_reason = _phase_specific_nfz_hit_v1(
            full_path,
            edge_phases,
            cruise_half_width_m,
            transition_corridor_cfg,
            forbidden_zones,
        )
        nfz_status = "PASS" if nfz_ok else "FAIL"
    else:
        nfz_status, nfz_reason = "NOT ENFORCED", "not_enforced"
    if bool(check_corridor_self_overlap):
        self_hit, self_reason = _phase_specific_self_overlap_v1(
            full_path,
            edge_phases,
            cruise_half_width_m,
            transition_corridor_cfg,
        )
        self_status = "FAIL" if self_hit else "PASS"
    else:
        self_status, self_reason = "NOT ENFORCED", "not_enforced"
    air_status = str((airspace_audit or {}).get("status", "UNKNOWN"))

    if (
        (bool(moc_enforced) and moc_status == "FAIL")
        or nfz_status == "FAIL"
        or self_status == "FAIL" or air_status == "FAIL"
        or not checker_match
    ):
        overall_status = "FAIL"
    elif not bool(moc_enforced):
        overall_status = "NOT ENFORCED"
    elif moc_status == "WARN" or air_status in ("WARN", "UNKNOWN"):
        overall_status = "WARN"
    else:
        overall_status = "PASS"

    directions = {}
    for direction in ("takeoff", "landing"):
        direction_rows = samples.loc[samples["Direction"].eq(direction)]
        station_rows = _transition_station_rows_v1(samples, direction)
        direction_moc_status, direction_hits, direction_out = _moc_diagnostic_status_v1(
            direction_rows, moc_enforced
        )
        finite_margins = pd.to_numeric(
            station_rows["Station_Vertical_Margin_m"], errors="coerce"
        ).dropna()
        no_clear_rows = station_rows.loc[
            station_rows["Station_MOC_Envelope_Class"].eq("NO_CLEAR_LAYER")
        ]
        minimum_margin_location = None
        minimum_margin_value = None
        minimum_margin_status = "UNAVAILABLE"
        if not no_clear_rows.empty:
            worst_row = no_clear_rows.iloc[0]
            minimum_margin_status = "NO_CLEAR_LAYER/FAIL"
            minimum_margin_location = {
                "direction_along_track_m": float(
                    worst_row["Direction_Along_Track_m"]
                ),
                "center_lat": float(worst_row["Center_Lat"]),
                "center_lon": float(worst_row["Center_Lon"]),
                "center_msl_m": float(worst_row["Center_Altitude_MSL_m"]),
                "lower_face_msl_m": float(
                    worst_row["Corridor_Lower_Face_MSL_m"]
                ),
                "required_safe_msl_m": None,
                "envelope_class": "NO_CLEAR_LAYER",
            }
        elif not finite_margins.empty:
            worst_idx = finite_margins.idxmin()
            worst_row = station_rows.loc[worst_idx]
            minimum_margin_value = float(finite_margins.loc[worst_idx])
            minimum_margin_status = str(worst_row["Station_3D_Status"])
            minimum_margin_location = {
                "direction_along_track_m": float(
                    worst_row["Direction_Along_Track_m"]
                ),
                "center_lat": float(worst_row["Center_Lat"]),
                "center_lon": float(worst_row["Center_Lon"]),
                "center_msl_m": float(worst_row["Center_Altitude_MSL_m"]),
                "lower_face_msl_m": float(
                    worst_row["Corridor_Lower_Face_MSL_m"]
                ),
                "required_safe_msl_m": float(
                    worst_row["Station_Required_Safe_MSL_m"]
                ),
            }
        direction_air_audit = (
            (airspace_audit or {}).get("directions", {}).get(direction, {})
        )
        direction_air_status = str(direction_air_audit.get("status", "UNKNOWN"))
        direction_air_reason = str(direction_air_audit.get("reason", "unknown"))
        if not bool(check_corridor_nfz):
            direction_nfz_status, direction_nfz_reason = (
                "NOT ENFORCED", "not_enforced"
            )
        else:
            direction_nfz_ok, direction_nfz_reason = _phase_specific_nfz_hit_v1(
                full_path,
                edge_phases,
                cruise_half_width_m,
                transition_corridor_cfg,
                forbidden_zones,
                direction_filter=direction,
            )
            direction_nfz_status = "PASS" if direction_nfz_ok else "FAIL"
        if not bool(check_corridor_self_overlap):
            direction_self_status, direction_self_reason = (
                "NOT ENFORCED", "not_enforced"
            )
        else:
            direction_self_hit, direction_self_reason = (
                _phase_specific_self_overlap_v1(
                    full_path,
                    edge_phases,
                    cruise_half_width_m,
                    transition_corridor_cfg,
                    direction_filter=direction,
                )
            )
            direction_self_status = "FAIL" if direction_self_hit else "PASS"
        direction_fail_reasons = []
        for status_value, reason_value in (
            (direction_moc_status, f"{direction}_transition_moc_3d_intersection"),
            (direction_nfz_status, direction_nfz_reason),
            (direction_air_status, direction_air_reason),
            (direction_self_status, direction_self_reason),
        ):
            if status_value == "FAIL":
                direction_fail_reasons.append(str(reason_value))
        if (
            direction_moc_status == "FAIL"
            or direction_nfz_status == "FAIL"
            or direction_air_status == "FAIL"
            or direction_self_status == "FAIL"
        ):
            direction_status = "FAIL"
        elif direction_moc_status in ("WARN", "NOT ENFORCED") or direction_air_status == "UNKNOWN":
            direction_status = direction_moc_status
        else:
            direction_status = "PASS"
        effective_clearances = pd.to_numeric(
            station_rows["Effective_Downward_Clearance_m"], errors="coerce"
        ).dropna()
        directions[direction] = {
            "status": str(direction_status),
            "moc_status": str(direction_moc_status),
            "nfz_status": str(direction_nfz_status),
            "airspace_status": str(direction_air_status),
            "self_overlap_status": str(direction_self_status),
            "fail_reasons": direction_fail_reasons,
            "station_count": int(station_rows.shape[0]),
            "sample_count": int(direction_rows.shape[0]),
            "hit_count": int(direction_hits),
            "out_of_grid_count": int(direction_out),
            "minimum_vertical_margin_m": minimum_margin_value,
            "minimum_vertical_margin_status": minimum_margin_status,
            "minimum_vertical_margin_location": minimum_margin_location,
            "effective_downward_clearance_min_m": (
                None if effective_clearances.empty else float(effective_clearances.min())
            ),
            "effective_downward_clearance_max_m": (
                None if effective_clearances.empty else float(effective_clearances.max())
            ),
        }
    return {
        "status": str(overall_status),
        "reason": "ok" if overall_status == "PASS" else "see_constraint_status",
        "transition_corridor_half_width_m": half_width_m,
        "transition_vertical_clearance_m": float(
            transition_corridor_cfg["downward_clearance_m"]
        ),
        "moc_status": str(moc_status),
        "moc_hit_count": int(hit_count),
        "moc_out_of_grid_count": int(out_count),
        "moc_checker_match": bool(checker_match),
        "effective_clearance_formula": (
            "min(configured_clearance, max(0, center_MSL - direction_vertiport_MSL))"
        ),
        "lower_face_formula": "center_MSL - effective_clearance",
        "applied_phases": [str(value) for value in TRANSITION_PHASE_NAMES],
        "applied_constraints": ["MOC", "NFZ", "airspace", "self_overlap"],
        "nfz_status": str(nfz_status),
        "nfz_reason": str(nfz_reason),
        "airspace_status": str(air_status),
        "airspace_reason": str((airspace_audit or {}).get("reason", "unknown")),
        "self_overlap_status": str(self_status),
        "self_overlap_reason": str(self_reason),
        "directions": directions,
    }


def _transition_centerline_rows_v1(sample_df, direction=None, layer_idx=None):
    rows = sample_df
    if direction is not None:
        rows = rows.loc[rows["Direction"].eq(str(direction))]
    if layer_idx is not None:
        rows = rows.loc[rows["Selected_MOC_Layer_Index"].eq(int(layer_idx))]
    if rows.empty:
        return rows.copy()
    keys = ["Direction", "Segment_Index", "Edge_Index", "Along_Sample_Index"]
    return (
        rows.sort_values(["Along_Track_m", "Segment_Index", "Edge_Index", "Along_Sample_Index"])
        .drop_duplicates(keys, keep="first")
        .reset_index(drop=True)
    )


def _transition_station_rows_v1(sample_df, direction):
    rows = sample_df.loc[sample_df["Direction"].eq(str(direction))]
    if rows.empty:
        return rows.copy()
    return (
        rows.sort_values(
            ["Direction_Along_Track_m", "Segment_Index", "Edge_Index", "Along_Sample_Index"]
        )
        .drop_duplicates(["Direction", "Direction_Station_Index"], keep="first")
        .reset_index(drop=True)
    )


def _transition_profile_edges_v1(rf, direction):
    """Return exact transition-edge distance/phase geometry for the side profile."""
    direction = str(direction)
    cumulative_m = 0.0
    records = []
    for segment_idx, seg in enumerate(rf.get("segments", []), start=1):
        points = np.asarray(
            seg.get("points", np.empty((0, 3))), dtype=float
        ).reshape(-1, 3)
        if points.shape[0] < 2:
            continue
        phases = np.asarray(
            seg.get("point_phases", np.empty((0,), dtype=object)), dtype=object
        ).reshape(-1)
        if phases.size != points.shape[0]:
            raise ValueError("flight_phase_length_mismatch in transition profile edges")
        for edge_idx in range(points.shape[0] - 1):
            edge_length_m = float(_seg_dist_m(points[edge_idx], points[edge_idx + 1]))
            phase = _edge_phase_for_transition_v1(phases, edge_idx)
            if _transition_direction_from_phase_v1(phase) == direction:
                records.append({
                    "segment_index": int(segment_idx),
                    "segment_type": str(seg.get("type", "TF")),
                    "flight_phase": str(phase),
                    "global_start_m": float(cumulative_m),
                    "global_end_m": float(cumulative_m + edge_length_m),
                    "start_msl_m": float(points[edge_idx, 2]),
                    "end_msl_m": float(points[edge_idx + 1, 2]),
                })
            cumulative_m += edge_length_m
    if not records:
        return records
    origin_m = float(min(item["global_start_m"] for item in records))
    for item in records:
        item["direction_start_m"] = float(item["global_start_m"] - origin_m)
        item["direction_end_m"] = float(item["global_end_m"] - origin_m)
    return records


def _transition_moc_extent_v1(sample_df, direction, fallback_extent):
    rows = sample_df.loc[sample_df["Direction"].eq(str(direction))]
    if rows.empty:
        return [float(v) for v in fallback_extent]
    lat_min = float(rows["Lat"].min())
    lat_max = float(rows["Lat"].max())
    lon_min = float(rows["Lon"].min())
    lon_max = float(rows["Lon"].max())
    mean_lat = 0.5 * (lat_min + lat_max)
    span_lat_m = max(1.0, (lat_max - lat_min) * 111000.0)
    span_lon_m = max(
        1.0,
        (lon_max - lon_min) * 111000.0 * np.cos(np.deg2rad(mean_lat)),
    )
    pad_m = max(300.0, 0.08 * max(span_lat_m, span_lon_m))
    pad_lat = pad_m / 111000.0
    pad_lon = pad_m / max(1.0, 111000.0 * np.cos(np.deg2rad(mean_lat)))
    return [lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat]


def _plot_moc_transition_layer_axis_v1(
    ax,
    request,
    extent,
    rf,
    sample_df,
    direction,
    layer_idx,
    moc_risk,
    moc_enforced,
    half_width_m,
    lat_lim,
    lon_lim,
    airspace_center_lla,
    airspace_radius_m,
    forbidden_zones,
    start_vertiport,
    end_vertiport,
    compact=False,
):
    """Draw one direction/layer MOC footprint panel from checker samples."""
    direction = str(direction)
    layer_idx = int(layer_idx)
    direction_label = "Takeoff" if direction == "takeoff" else "Landing"
    direction_color = (
        TAKEOFF_TRANSITION_COLOR
        if direction == "takeoff" else LANDING_TRANSITION_COLOR
    )
    moc = np.asarray(moc_risk, dtype=float)
    if moc.ndim == 2:
        moc = moc[:, :, np.newaxis]
    layer_idx = int(np.clip(layer_idx, 0, moc.shape[2] - 1))
    moc_agl_m = float(
        MOC_AGL_LEVELS_M[int(np.clip(layer_idx, 0, len(MOC_AGL_LEVELS_M) - 1))]
    )
    layer_rows = sample_df.loc[
        sample_df["Direction"].eq(direction)
        & sample_df["Selected_MOC_Layer_Index"].eq(layer_idx)
    ].copy()
    center_all = _transition_centerline_rows_v1(sample_df, direction=direction)
    center_active = _transition_centerline_rows_v1(
        sample_df, direction=direction, layer_idx=layer_idx
    )
    status, hit_count, out_of_grid_count = _moc_diagnostic_status_v1(
        layer_rows, moc_enforced
    )
    tested_count = int(np.count_nonzero(layer_rows["Grid_Status"].eq("INSIDE")))

    ax.set_extent([float(v) for v in extent])
    ax.add_image(request, 13)
    draw_vertiport_radius_rings(
        ax, airspace_center_lla, radii_m=(float(airspace_radius_m),)
    )
    plot_forbidden_zones(
        ax, forbidden_zones, face_alpha=0.10, edge_alpha=0.80
    )
    plot_moc_binary_overlay(
        ax,
        moc[:, :, layer_idx],
        lat_lim,
        lon_lim,
        label=f"MOC=1 AGL{int(moc_agl_m)}m",
        fill_color="magenta",
        fill_alpha=0.24,
    )

    if not center_all.empty:
        ax.plot(
            center_all["Center_Lon"].to_numpy(dtype=float),
            center_all["Center_Lat"].to_numpy(dtype=float),
            color="dimgray",
            linewidth=1.1,
            alpha=0.55,
            transform=ccrs.Geodetic(),
            zorder=5,
            label=f"Full {direction_label} Transition" if not compact else None,
        )

    if not center_active.empty:
        for (_, _, _), edge_rows in center_active.groupby(
            ["Segment_Index", "Segment_Type", "Flight_Phase"], sort=False
        ):
            edge_rows = edge_rows.sort_values("Direction_Along_Track_m")
            pts = np.column_stack([
                edge_rows["Center_Lat"].to_numpy(dtype=float),
                edge_rows["Center_Lon"].to_numpy(dtype=float),
                edge_rows["Altitude_MSL_m"].to_numpy(dtype=float),
            ])
            segment_type = str(edge_rows["Segment_Type"].iloc[0])
            phase = str(edge_rows["Flight_Phase"].iloc[0])
            linestyle = "-"
            linewidth = 3.0 if segment_type == "RF" else 1.9
            if pts.shape[0] >= 2:
                plot_corridor_width(
                    ax,
                    pts,
                    float(half_width_m),
                    color=direction_color,
                    alpha=0.12,
                )
                ax.plot(
                    pts[:, 1],
                    pts[:, 0],
                    linestyle=linestyle,
                    color=direction_color,
                    linewidth=linewidth,
                    transform=ccrs.Geodetic(),
                    zorder=9,
                )
            else:
                ax.scatter(
                    pts[:, 1],
                    pts[:, 0],
                    s=18,
                    c=direction_color,
                    marker="o",
                    transform=ccrs.Geodetic(),
                    zorder=9,
                )

        ax.plot(
            [], [], "-", color=direction_color, linewidth=1.9,
            label=f"{direction_label} TF / Stage2" if not compact else None,
        )
        ax.plot(
            [], [], "--", color=direction_color, linewidth=1.9,
            label=f"{direction_label} Stage1" if not compact else None,
        )
        ax.plot(
            [], [], "-", color=direction_color, linewidth=3.0,
            label=f"{direction_label} RF" if not compact else None,
        )

    inside_clear = layer_rows.loc[
        layer_rows["Grid_Status"].eq("INSIDE") & ~layer_rows["Blocked"]
    ]
    if not inside_clear.empty:
        ax.scatter(
            inside_clear["Lon"], inside_clear["Lat"],
            s=5, c="slategray", alpha=0.28, marker=".",
            transform=ccrs.Geodetic(), zorder=7,
            label="Checked footprint" if not compact else None,
        )
    blocked_rows = layer_rows.loc[layer_rows["Blocked"]]
    if not blocked_rows.empty:
        grid_lat_step = (
            (float(lat_lim[1]) - float(lat_lim[0])) / (moc.shape[0] - 1)
            if moc.shape[0] > 1 else 0.0
        )
        grid_lon_step = (
            (float(lon_lim[1]) - float(lon_lim[0])) / (moc.shape[1] - 1)
            if moc.shape[1] > 1 else 0.0
        )
        for _, cell in blocked_rows.drop_duplicates(
            ["Grid_Row", "Grid_Col"]
        ).iterrows():
            cell_lat = float(lat_lim[0]) + float(cell["Grid_Row"]) * grid_lat_step
            cell_lon = float(lon_lim[0]) + float(cell["Grid_Col"]) * grid_lon_step
            ax.fill(
                [
                    cell_lon - 0.5 * grid_lon_step,
                    cell_lon + 0.5 * grid_lon_step,
                    cell_lon + 0.5 * grid_lon_step,
                    cell_lon - 0.5 * grid_lon_step,
                ],
                [
                    cell_lat - 0.5 * grid_lat_step,
                    cell_lat - 0.5 * grid_lat_step,
                    cell_lat + 0.5 * grid_lat_step,
                    cell_lat + 0.5 * grid_lat_step,
                ],
                facecolor="red",
                edgecolor="red",
                linewidth=0.8,
                alpha=0.24,
                transform=ccrs.PlateCarree(),
                zorder=12,
            )
        ax.scatter(
            blocked_rows["Lon"], blocked_rows["Lat"],
            s=44, c="red", marker="x", linewidths=1.2,
            transform=ccrs.Geodetic(), zorder=13,
            label="MOC hit" if not compact else None,
        )
    outside_rows = layer_rows.loc[layer_rows["Grid_Status"].ne("INSIDE")]
    if not outside_rows.empty:
        ax.scatter(
            outside_rows["Lon"], outside_rows["Lat"],
            s=30, c="darkorange", marker="^", edgecolors="k", linewidths=0.3,
            transform=ccrs.Geodetic(), zorder=13,
            label="Out of evaluation grid" if not compact else None,
        )

    active_rf_indices = sorted({
        int(v)
        for v in center_active.loc[
            center_active["Segment_Type"].eq("RF"), "Segment_Index"
        ].tolist()
    })
    for marker_idx, segment_idx in enumerate(active_rf_indices):
        if not 1 <= segment_idx <= len(rf.get("segments", [])):
            continue
        seg = rf["segments"][segment_idx - 1]
        pts = np.asarray(seg.get("points", np.empty((0, 3))), dtype=float).reshape(-1, 3)
        if pts.shape[0] == 0:
            continue
        center = np.asarray(seg.get("arc_center", np.empty((0,))), dtype=float).reshape(-1)
        ax.scatter(
            [pts[0, 1]], [pts[0, 0]], s=38, c="yellow", marker=">",
            edgecolors="k", linewidths=0.5, transform=ccrs.Geodetic(), zorder=12,
            label=("RF segment start" if marker_idx == 0 and not compact else None),
        )
        ax.scatter(
            [pts[-1, 1]], [pts[-1, 0]], s=38, c="yellow", marker="s",
            edgecolors="k", linewidths=0.5, transform=ccrs.Geodetic(), zorder=12,
            label=("RF segment end" if marker_idx == 0 and not compact else None),
        )
        if center.size >= 2:
            active_seg_rows = center_active.loc[
                center_active["Segment_Index"].eq(segment_idx)
            ]
            anchor = active_seg_rows.iloc[0]
            ax.plot(
                [center[1], float(anchor["Center_Lon"])],
                [center[0], float(anchor["Center_Lat"])],
                ":", color="goldenrod", linewidth=0.8,
                transform=ccrs.Geodetic(), zorder=10,
            )
            ax.scatter(
                [center[1]], [center[0]], s=44, c="white", marker="x",
                linewidths=1.3, transform=ccrs.Geodetic(), zorder=12,
                label=("RF center" if marker_idx == 0 and not compact else None),
            )
            if not compact:
                ax.text(
                    center[1], center[0],
                    f" R={float(seg.get('turn_radius', 0.0)):.0f}m",
                    fontsize=6, color="black", transform=ccrs.Geodetic(), zorder=13,
                )

    _plot_transition_phase_markers(
        ax,
        rf,
        transform=ccrs.Geodetic(),
        zorder=14,
        labels=not compact,
        include_stage1=True,
        direction=direction,
    )
    port = start_vertiport if direction == "takeoff" else end_vertiport
    ax.scatter(
        [port[1]], [port[0]],
        s=88 if not compact else 45,
        c="red" if direction == "takeoff" else "crimson",
        edgecolors="k",
        marker="s" if direction == "takeoff" else "D",
        transform=ccrs.Geodetic(),
        zorder=14,
        label=(f"{direction_label} Vertiport" if not compact else None),
    )

    alt_min = float(layer_rows["Altitude_MSL_m"].min())
    alt_max = float(layer_rows["Altitude_MSL_m"].max())
    lower_min = float(layer_rows["Corridor_Lower_Face_MSL_m"].min())
    lower_max = float(layer_rows["Corridor_Lower_Face_MSL_m"].max())
    ax.set_title(
        (
            f"{direction_label} | MOC AGL{int(moc_agl_m)}m\n"
            f"Applied path MSL {alt_min:.1f}-{alt_max:.1f}m | "
            f"lower face MSL {lower_min:.1f}-{lower_max:.1f}m | "
            f"{status}: hits {hit_count}/{tested_count}, out {out_of_grid_count}"
        ),
        fontsize=8 if compact else 10,
    )
    if not compact:
        ax.legend(
            loc="center left", bbox_to_anchor=(1.01, 0.5),
            fontsize=7, framealpha=0.9,
        )


def _plot_transition_moc_profile_axis_v1(
    ax,
    sample_df,
    direction,
    rf,
    moc_enforced,
):
    direction = str(direction)
    direction_label = "Takeoff" if direction == "takeoff" else "Landing"
    direction_color = (
        TAKEOFF_TRANSITION_COLOR
        if direction == "takeoff" else LANDING_TRANSITION_COLOR
    )
    station_rows = _transition_station_rows_v1(sample_df, direction)
    if station_rows.empty:
        ax.text(0.5, 0.5, f"No {direction_label.lower()} transition samples", ha="center", va="center")
        ax.set_axis_off()
        return

    station_rows = station_rows.sort_values("Direction_Along_Track_m")
    profile_edges = _transition_profile_edges_v1(rf, direction)
    x = station_rows["Direction_Along_Track_m"].to_numpy(dtype=float)
    center_msl = station_rows["Altitude_MSL_m"].to_numpy(dtype=float)
    lower_msl = station_rows["Corridor_Lower_Face_MSL_m"].to_numpy(dtype=float)
    required_msl = pd.to_numeric(
        station_rows["Station_Required_Safe_MSL_m"], errors="coerce"
    ).to_numpy(dtype=float)
    statuses = station_rows["Station_3D_Status"].astype(str).to_numpy(dtype=object)
    envelope_classes = station_rows["Station_MOC_Envelope_Class"].astype(str).to_numpy(dtype=object)
    x_max = max(1.0, float(np.max(x)))
    finite_required = np.isfinite(required_msl)
    values_for_limits = [center_msl, lower_msl]
    if np.any(finite_required):
        values_for_limits.append(required_msl[finite_required])
    if np.any(envelope_classes == "NO_CLEAR_LAYER"):
        values_for_limits.append(np.asarray([
            MOC_REFERENCE_MSL_M + float(MOC_AGL_LEVELS_M[-1])
        ]))
    y_min = float(min(np.min(values) for values in values_for_limits))
    y_max = float(max(np.max(values) for values in values_for_limits))
    pad_y = max(10.0, 0.05 * max(1.0, y_max - y_min))
    plot_bottom = y_min - pad_y
    plot_top = y_max + pad_y

    layer_indices = station_rows["Selected_MOC_Layer_Index"].to_numpy(dtype=int)
    if x.size == 1:
        station_left = np.asarray([0.0], dtype=float)
        station_right = np.asarray([x_max], dtype=float)
    else:
        station_mid = 0.5 * (x[:-1] + x[1:])
        station_left = np.r_[0.0, station_mid]
        station_right = np.r_[station_mid, x_max]
    band_start = 0
    band_no = 0
    while band_start < layer_indices.size:
        band_end = band_start + 1
        while (
            band_end < layer_indices.size
            and int(layer_indices[band_end]) == int(layer_indices[band_start])
        ):
            band_end += 1
        layer_idx = int(layer_indices[band_start])
        span_start = float(station_left[band_start])
        span_end = float(station_right[band_end - 1])
        ax.axvspan(
            span_start,
            max(span_start + 1e-6, span_end),
            color=("whitesmoke" if band_no % 2 == 0 else "lightsteelblue"),
            alpha=0.25,
            zorder=0,
        )
        ax.text(
            0.5 * (span_start + span_end),
            0.995,
            f"AGL{int(MOC_AGL_LEVELS_M[layer_idx])}",
            transform=ax.get_xaxis_transform(),
            fontsize=6,
            color="dimgray",
            ha="center",
            va="top",
            zorder=8,
        )
        band_no += 1
        band_start = band_end

    ax.fill_between(
        x,
        lower_msl,
        center_msl,
        color=direction_color,
        alpha=0.12,
        label="Downward transition envelope",
        zorder=2,
    )
    ax.plot(
        x,
        lower_msl,
        "--",
        color=direction_color,
        linewidth=1.5,
        label="Corridor lower face",
        zorder=4,
    )
    ax.plot(
        x,
        center_msl,
        color=direction_color,
        linewidth=1.2,
        alpha=0.65,
        label=f"{direction_label} centerline",
        zorder=4,
    )
    if np.any(finite_required):
        envelope_plot = np.where(finite_required, required_msl, np.nan)
        ax.step(
            x,
            envelope_plot,
            where="mid",
            color="magenta",
            linewidth=1.8,
            label="Discrete MOC required-safe envelope (not terrain)",
            zorder=3,
        )
        ax.fill_between(
            x,
            plot_bottom,
            envelope_plot,
            where=finite_required,
            step="mid",
            color="magenta",
            alpha=0.10,
            zorder=1,
        )
    elif np.all(envelope_classes == "ALL_CLEAR"):
        ax.text(
            0.02,
            0.92,
            (
                "MOC envelope: ALL_CLEAR"
                if moc_enforced else "MOC envelope: ALL_CLEAR (NOT ENFORCED)"
            ),
            transform=ax.transAxes,
            color="magenta",
            fontsize=8,
            weight="bold",
            ha="left",
            va="top",
        )

    fail_mask = np.asarray(["FAIL" in value for value in statuses], dtype=bool)
    exact_mask = np.asarray([value.startswith("EXACT_LIMIT") for value in statuses], dtype=bool)
    out_mask = np.asarray([value.startswith("OUT_OF_GRID") for value in statuses], dtype=bool)
    no_clear_mask = envelope_classes == "NO_CLEAR_LAYER"
    if np.any(fail_mask):
        ax.scatter(
            x[fail_mask], lower_msl[fail_mask],
            c="red", marker="x", s=42, linewidths=1.3,
            label=(
                "3D MOC intersection"
                if moc_enforced else "3D MOC intersection (diagnostic only)"
            ),
            zorder=7,
        )
        finite_fail = fail_mask & finite_required
        if np.any(finite_fail):
            ax.fill_between(
                x,
                lower_msl,
                required_msl,
                where=finite_fail,
                color="red",
                alpha=0.22,
                zorder=5,
            )
    if np.any(exact_mask):
        ax.scatter(
            x[exact_mask], lower_msl[exact_mask],
            facecolors="none", edgecolors="gold", marker="o", s=45,
            linewidths=1.2,
            label=(
                "Exact clearance limit (PASS)"
                if moc_enforced else "Exact clearance limit (NOT ENFORCED)"
            ),
            zorder=7,
        )

    unique_x = np.unique(x)
    if unique_x.size > 1:
        station_half_span_m = max(
            1.0, 0.5 * float(np.median(np.diff(unique_x)))
        )
    else:
        station_half_span_m = 1.0
    first_out = True
    first_no_clear = True
    for idx in range(x.size):
        x0 = max(0.0, float(x[idx] - station_half_span_m))
        x1 = min(x_max, float(x[idx] + station_half_span_m))
        if fail_mask[idx]:
            ax.axvspan(x0, x1, color="red", alpha=0.06, zorder=1)
        if out_mask[idx]:
            ax.axvspan(
                x0, x1, color="darkorange", alpha=0.12, zorder=1,
                label="OUT_OF_GRID / WARN" if first_out else None,
            )
            first_out = False
        if no_clear_mask[idx]:
            ax.axvspan(
                x0, x1, facecolor="magenta", edgecolor="red", hatch="///",
                alpha=0.12, zorder=2,
                label=(
                    (
                        "No clear layer through AGL900"
                        if moc_enforced
                        else "No clear layer through AGL900 (diagnostic only)"
                    )
                    if first_no_clear else None
                ),
            )
            first_no_clear = False

    first_rf_interval = True
    rf_segment_indices = sorted({
        int(item["segment_index"])
        for item in profile_edges
        if item["segment_type"] == "RF"
    })
    for segment_idx in rf_segment_indices:
        rf_edges = [
            item for item in profile_edges
            if int(item["segment_index"]) == segment_idx
        ]
        x0 = float(min(item["direction_start_m"] for item in rf_edges))
        x1 = float(max(item["direction_end_m"] for item in rf_edges))
        ax.axvspan(
            x0,
            max(x0 + 1e-6, x1),
            color="gold",
            alpha=0.12,
            zorder=1,
            label="RF interval" if first_rf_interval else None,
        )
        first_rf_interval = False

    phase_order = (
        [FLIGHT_PHASE_TAKEOFF_STAGE1, FLIGHT_PHASE_TAKEOFF_STAGE2]
        if direction == "takeoff"
        else [FLIGHT_PHASE_LANDING_STAGE2, FLIGHT_PHASE_LANDING_STAGE1]
    )
    for phase in phase_order:
        phase_edges = [
            item for item in profile_edges
            if item["flight_phase"] == phase
        ]
        if not phase_edges:
            continue
        for edge_no, item in enumerate(phase_edges):
            ax.plot(
                [item["direction_start_m"], item["direction_end_m"]],
                [item["start_msl_m"], item["end_msl_m"]],
                "-",
                color=direction_color,
                linewidth=2.2,
                label=phase if edge_no == 0 else None,
                zorder=4,
            )

    if profile_edges and direction == "takeoff":
        stage1_edges = [
            item for item in profile_edges
            if item["flight_phase"] == FLIGHT_PHASE_TAKEOFF_STAGE1
        ]
        if stage1_edges:
            boundary = max(stage1_edges, key=lambda item: item["direction_end_m"])
            ax.scatter(
                [boundary["direction_end_m"]], [boundary["end_msl_m"]],
                s=55, facecolors="none", edgecolors=direction_color,
                marker="o", linewidths=1.3, zorder=6,
            )
        transition_end = max(
            profile_edges, key=lambda item: item["direction_end_m"]
        )
        ax.scatter(
            [transition_end["direction_end_m"]], [transition_end["end_msl_m"]],
            s=60, c=direction_color, edgecolors="k", marker="^", zorder=6,
        )
    elif profile_edges:
        transition_start = min(
            profile_edges, key=lambda item: item["direction_start_m"]
        )
        ax.scatter(
            [transition_start["direction_start_m"]], [transition_start["start_msl_m"]],
            s=60, c=direction_color, edgecolors="k", marker="v", zorder=6,
        )
        stage1_edges = [
            item for item in profile_edges
            if item["flight_phase"] == FLIGHT_PHASE_LANDING_STAGE1
        ]
        if stage1_edges:
            boundary = min(stage1_edges, key=lambda item: item["direction_start_m"])
            ax.scatter(
                [boundary["direction_start_m"]], [boundary["start_msl_m"]],
                s=55, facecolors="none", edgecolors=direction_color,
                marker="o", linewidths=1.3, zorder=6,
            )

    taper_values = station_rows["Clearance_Taper_Active"].to_numpy(dtype=bool)
    taper_changes = np.flatnonzero(taper_values[1:] != taper_values[:-1]) + 1
    if taper_changes.size > 0:
        taper_x = float(x[int(taper_changes[0])])
        configured_clearance_m = float(
            station_rows["Configured_Downward_Clearance_m"].iloc[0]
        )
        ax.axvline(taper_x, color="dimgray", linestyle=":", linewidth=1.0, zorder=3)
        ax.text(
            taper_x,
            plot_top,
            f"{configured_clearance_m:.0f}m downward clearance boundary",
            fontsize=7, color="dimgray", rotation=90, va="top", ha="right",
        )

    ax.set_xlim(0.0, x_max)
    ax.set_ylim(plot_bottom, plot_top)
    ax.set_title(
        f"{direction_label} 3D transition MOC clearance"
        + ("" if moc_enforced else " | NOT ENFORCED")
    )
    ax.set_xlabel("Transition ground-track distance (m)")
    ax.set_ylabel("Altitude MSL (m)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=7, framealpha=0.9)


def _save_moc_transition_snapshots_v1(
    rf,
    out_dir,
    request,
    map_extent,
    moc_risk,
    moc_enforced,
    half_width_m,
    cruise_half_width_m,
    transition_corridor_cfg,
    lat_lim,
    lon_lim,
    airspace_center_lla,
    airspace_radius_m,
    forbidden_zones,
    start_vertiport,
    end_vertiport,
    overall_constraint_ok,
    overall_constraint_reason,
    airspace_ok,
    check_corridor_nfz,
    check_corridor_self_overlap,
    min_distance_enforced,
    min_distance_ok,
    rf_min_allowed_radius_m,
):
    """Save final Balanced-path MOC/RF transition audit artifacts."""
    transition_structure_mode = str(
        rf.get("transition_meta", {}).get("transition_structure_mode", "unknown")
    )
    audit_only_fixed_transition_geometry = bool(
        transition_structure_mode == TRANSITION_STRUCTURE_FIXED_ONLY
    )
    moc_audit_includes_fixed_transition = bool(any(
        bool(seg.get("is_fixed_transition_stage1", False))
        for seg in rf.get("segments", [])
    ))
    audit_only_notice = (
        "audit-only fixed transition geometry; omitted from general corridor outputs"
    )
    sample_df = _collect_transition_moc_samples_v1(
        rf,
        half_width_m,
        moc_risk,
        lat_lim,
        lon_lim,
        transition_corridor_cfg=transition_corridor_cfg,
    )
    if sample_df.empty:
        raise RuntimeError(
            "Transition MOC visualization requested, but no transition samples were collected."
        )
    for direction in ("takeoff", "landing"):
        if not bool(sample_df["Direction"].eq(direction).any()):
            raise RuntimeError(
                f"Transition MOC visualization has no {direction} samples."
            )

    diagnostic_hit = bool(sample_df["Blocked"].any())
    transition_checker_hit = bool(_transition_moc_checker_hit_v1(
        rf,
        half_width_m,
        moc_risk,
        lat_lim,
        lon_lim,
        transition_corridor_cfg=transition_corridor_cfg,
    ))
    checker_match = bool(diagnostic_hit == transition_checker_hit)
    if not checker_match:
        raise RuntimeError(
            "Transition MOC diagnostic samples disagree with the constraint checker."
        )

    full_path_moc_hit = bool(_full_path_moc_checker_hit_v1(
        rf,
        cruise_half_width_m,
        transition_corridor_cfg,
        moc_risk,
        lat_lim,
        lon_lim,
    ))
    full_path = np.asarray(rf.get("path", np.empty((0, 3))), dtype=float).reshape(-1, 3)
    full_edge_phases = _edge_phase_values_v1(
        full_path,
        rf.get("flight_phases"),
        transition_corridor_cfg,
    )
    if bool(check_corridor_nfz):
        nfz_ok, nfz_reason = _phase_specific_nfz_hit_v1(
            full_path,
            full_edge_phases,
            cruise_half_width_m,
            transition_corridor_cfg,
            forbidden_zones,
        )
        nfz_status = "PASS" if nfz_ok else "FAIL"
    else:
        nfz_ok, nfz_reason, nfz_status = True, "not_enforced", "NOT ENFORCED"
    if bool(check_corridor_self_overlap):
        self_overlap_hit, self_overlap_reason = _phase_specific_self_overlap_v1(
            full_path,
            full_edge_phases,
            cruise_half_width_m,
            transition_corridor_cfg,
        )
        self_overlap_status = "FAIL" if self_overlap_hit else "PASS"
    else:
        self_overlap_hit = False
        self_overlap_reason = "not_enforced"
        self_overlap_status = "NOT ENFORCED"
    transition_airspace_validation = dict(
        rf.get("transition_airspace_validation", {})
    )

    snapshot_dir = Path(out_dir) / "moc_transition_snapshots"
    working_dir = Path(out_dir) / "_moc_transition_snapshots_incomplete"
    working_dir.mkdir(parents=True, exist_ok=False)
    csv_name = "moc_transition_samples.csv"
    json_name = "moc_transition_summary.json"
    sample_df.to_csv(working_dir / csv_name, index=False, encoding="utf-8-sig")

    overall_moc_status, total_hits, total_out_of_grid = _moc_diagnostic_status_v1(
        sample_df, moc_enforced
    )
    total_samples = int(sample_df.shape[0])
    total_tested = int(np.count_nonzero(sample_df["Grid_Status"].eq("INSIDE")))

    direction_summary = {}
    used_layers_by_direction = {}
    layer_sample_sum = 0
    for direction in ("takeoff", "landing"):
        direction_rows = sample_df.loc[sample_df["Direction"].eq(direction)]
        direction_stations = _transition_station_rows_v1(sample_df, direction)
        used_layers = []
        for value in direction_stations["Selected_MOC_Layer_Index"].tolist():
            layer_idx_value = int(value)
            if layer_idx_value not in used_layers:
                used_layers.append(layer_idx_value)
        used_layers_by_direction[direction] = used_layers
        layer_statistics = []
        for layer_idx in used_layers:
            layer_rows = direction_rows.loc[
                direction_rows["Selected_MOC_Layer_Index"].eq(layer_idx)
            ]
            layer_status, hit_count, out_count = _moc_diagnostic_status_v1(
                layer_rows, moc_enforced
            )
            sample_count = int(layer_rows.shape[0])
            tested_count = int(np.count_nonzero(layer_rows["Grid_Status"].eq("INSIDE")))
            layer_sample_sum += sample_count
            moc_agl_m = int(MOC_AGL_LEVELS_M[layer_idx])
            layer_statistics.append({
                "layer_index": int(layer_idx),
                "moc_agl_m": moc_agl_m,
                "moc_reference_msl_m": float(MOC_REFERENCE_MSL_M + moc_agl_m),
                "applied_path_msl_min_m": float(layer_rows["Altitude_MSL_m"].min()),
                "applied_path_msl_max_m": float(layer_rows["Altitude_MSL_m"].max()),
                "applied_lower_face_msl_min_m": float(
                    layer_rows["Corridor_Lower_Face_MSL_m"].min()
                ),
                "applied_lower_face_msl_max_m": float(
                    layer_rows["Corridor_Lower_Face_MSL_m"].max()
                ),
                "sample_count": sample_count,
                "tested_count": tested_count,
                "hit_count": int(hit_count),
                "out_of_grid_count": int(out_count),
                "status": str(layer_status),
            })
        direction_summary[direction] = {
            "flight_layer_order": (
                "ascending_agl" if direction == "takeoff" else "descending_agl"
            ),
            "used_layers_agl_m": [int(MOC_AGL_LEVELS_M[idx]) for idx in used_layers],
            "sample_count": int(direction_rows.shape[0]),
            "station_count": int(direction_stations.shape[0]),
            "tested_count": int(np.count_nonzero(direction_rows["Grid_Status"].eq("INSIDE"))),
            "hit_count": int(np.count_nonzero(direction_rows["Blocked"])),
            "out_of_grid_count": int(np.count_nonzero(
                direction_rows["Grid_Status"].ne("INSIDE")
            )),
            "layer_sample_count_sum": int(sum(
                item["sample_count"] for item in layer_statistics
            )),
            "layer_statistics": layer_statistics,
        }
        direction_status, _, _ = _moc_diagnostic_status_v1(
            direction_rows, moc_enforced
        )
        direction_summary[direction]["moc_3d_status"] = str(direction_status)
        direction_validation = (
            rf.get("transition_3d_validation", {})
            .get("directions", {})
            .get(direction, {})
        )
        direction_summary[direction]["nfz_status"] = str(
            direction_validation.get("nfz_status", nfz_status)
        )
        direction_summary[direction]["airspace_status"] = str(
            direction_validation.get(
                "airspace_status",
                transition_airspace_validation.get("directions", {})
                .get(direction, {})
                .get("status", "UNKNOWN"),
            )
        )
        direction_summary[direction]["self_overlap_status"] = str(
            direction_validation.get("self_overlap_status", self_overlap_status)
        )
        direction_summary[direction]["constraint_fail_reasons"] = [
            str(value) for value in direction_validation.get("fail_reasons", [])
        ]
        finite_margins = pd.to_numeric(
            direction_stations["Station_Vertical_Margin_m"], errors="coerce"
        )
        finite_mask = finite_margins.notna()
        no_clear_rows = direction_stations.loc[
            direction_stations["Station_MOC_Envelope_Class"].eq("NO_CLEAR_LAYER")
        ]
        if not no_clear_rows.empty:
            worst_row = no_clear_rows.iloc[0]
            direction_summary[direction]["minimum_vertical_margin_m"] = None
            direction_summary[direction]["minimum_vertical_margin_status"] = (
                "NO_CLEAR_LAYER/FAIL"
            )
            direction_summary[direction]["minimum_vertical_margin_location"] = {
                "direction_along_track_m": float(worst_row["Direction_Along_Track_m"]),
                "center_lat": float(worst_row["Center_Lat"]),
                "center_lon": float(worst_row["Center_Lon"]),
                "center_msl_m": float(worst_row["Center_Altitude_MSL_m"]),
                "lower_face_msl_m": float(worst_row["Corridor_Lower_Face_MSL_m"]),
                "required_safe_msl_m": None,
                "envelope_class": "NO_CLEAR_LAYER",
            }
        elif bool(finite_mask.any()):
            worst_idx = finite_margins.loc[finite_mask].idxmin()
            worst_row = direction_stations.loc[worst_idx]
            direction_summary[direction]["minimum_vertical_margin_m"] = float(
                finite_margins.loc[worst_idx]
            )
            direction_summary[direction]["minimum_vertical_margin_status"] = str(
                worst_row["Station_3D_Status"]
            )
            direction_summary[direction]["minimum_vertical_margin_location"] = {
                "direction_along_track_m": float(worst_row["Direction_Along_Track_m"]),
                "center_lat": float(worst_row["Center_Lat"]),
                "center_lon": float(worst_row["Center_Lon"]),
                "center_msl_m": float(worst_row["Center_Altitude_MSL_m"]),
                "lower_face_msl_m": float(worst_row["Corridor_Lower_Face_MSL_m"]),
                "required_safe_msl_m": float(
                    worst_row["Station_Required_Safe_MSL_m"]
                ),
                "envelope_class": str(worst_row["Station_MOC_Envelope_Class"]),
            }
        else:
            direction_summary[direction]["minimum_vertical_margin_m"] = None
            direction_summary[direction]["minimum_vertical_margin_status"] = (
                "UNAVAILABLE"
            )
            direction_summary[direction]["minimum_vertical_margin_location"] = None
        effective_clearances = pd.to_numeric(
            direction_stations["Effective_Downward_Clearance_m"], errors="coerce"
        ).dropna()
        direction_summary[direction]["effective_downward_clearance_min_m"] = (
            None if effective_clearances.empty else float(effective_clearances.min())
        )
        direction_summary[direction]["effective_downward_clearance_max_m"] = (
            None if effective_clearances.empty else float(effective_clearances.max())
        )
        direction_summary[direction]["station_status_counts"] = {
            str(key): int(value)
            for key, value in direction_stations["Station_3D_Status"].value_counts().items()
        }

    all_rf_segments = [
        seg for seg in rf.get("segments", []) if str(seg.get("type", "TF")) == "RF"
    ]
    transition_rf_segment_indices = {
        int(value)
        for value in sample_df.loc[
            sample_df["Segment_Type"].eq("RF"), "Segment_Index"
        ].tolist()
    }
    rf_segments = [
        seg
        for segment_idx, seg in enumerate(rf.get("segments", []), start=1)
        if segment_idx in transition_rf_segment_indices
        and str(seg.get("type", "TF")) == "RF"
    ]
    transition_rf_radii = [
        float(seg["turn_radius"])
        for seg in rf_segments
        if seg.get("turn_radius") is not None and np.isfinite(float(seg["turn_radius"]))
    ]
    all_rf_radii = [
        float(seg["turn_radius"])
        for seg in all_rf_segments
        if seg.get("turn_radius") is not None
        and np.isfinite(float(seg["turn_radius"]))
    ]
    min_actual_rf_radius_m = min(all_rf_radii) if all_rf_radii else None
    transition_min_actual_rf_radius_m = (
        min(transition_rf_radii) if transition_rf_radii else None
    )
    rf_radius_pass = bool(
        min_actual_rf_radius_m is None
        or min_actual_rf_radius_m + 1e-6 >= float(rf_min_allowed_radius_m)
    )
    transition_rf_radius_pass = (
        None
        if transition_min_actual_rf_radius_m is None else bool(
            transition_min_actual_rf_radius_m + 1e-6
            >= float(rf_min_allowed_radius_m)
        )
    )
    rf_geometry_feasible = bool(
        rf.get("rf_geometry_feasible", rf.get("feasible", True))
    )
    combined_rf_transition_feasible = bool(rf.get("feasible", True))
    rf_had_clamp = bool(rf.get("had_clamp", False))
    if not rf_geometry_feasible or not rf_radius_pass:
        rf_status = "FAIL"
    elif not all_rf_segments:
        rf_status = "NOT APPLICABLE"
    elif rf_had_clamp:
        rf_status = "WARN"
    else:
        rf_status = "PASS"

    validation_checks = {
        str(key): bool(value)
        for key, value in rf.get("transition_meta", {}).get(
            "validation_checks", {}
        ).items()
    }
    validation_pass_count = int(sum(validation_checks.values()))
    validation_total_count = int(len(validation_checks))

    file_names = [
        "00_transition_moc_validation.png",
        "01_takeoff_moc_layers_overview.png",
        "02_landing_moc_layers_overview.png",
    ]
    for direction in ("takeoff", "landing"):
        ordered_layers = list(used_layers_by_direction[direction])
        for sequence_no, layer_idx in enumerate(ordered_layers, start=1):
            moc_agl_m = int(MOC_AGL_LEVELS_M[layer_idx])
            file_names.append(
                f"{direction}_{sequence_no:03d}_agl{moc_agl_m:04d}.png"
            )
    file_names.extend([csv_name, json_name])

    if (
        (bool(moc_enforced) and overall_moc_status == "FAIL")
        or nfz_status == "FAIL"
        or self_overlap_status == "FAIL"
        or not bool(airspace_ok)
    ):
        transition_3d_status = "FAIL"
    elif not bool(moc_enforced):
        transition_3d_status = "NOT ENFORCED"
    elif overall_moc_status == "WARN":
        transition_3d_status = "WARN"
    else:
        transition_3d_status = "PASS"

    summary = {
        "enabled": True,
        "generated": True,
        "reason": "generated",
        "transition_structure_mode": transition_structure_mode,
        "audit_only_fixed_transition_geometry": audit_only_fixed_transition_geometry,
        "moc_audit_includes_fixed_transition": moc_audit_includes_fixed_transition,
        "output_policy_notice": (
            audit_only_notice if audit_only_fixed_transition_geometry else None
        ),
        "folder": "moc_transition_snapshots",
        "files": [
            (Path("moc_transition_snapshots") / name).as_posix()
            for name in file_names
        ],
        "moc_enforced": bool(moc_enforced),
        "status": str(overall_moc_status),
        "sample_count": total_samples,
        "tested_count": total_tested,
        "hit_count": int(total_hits),
        "out_of_grid_count": int(total_out_of_grid),
        "layer_sample_count_sum": int(layer_sample_sum),
        "layer_sample_count_matches_total": bool(layer_sample_sum == total_samples),
        "diagnostic_hit": bool(diagnostic_hit),
        "transition_checker_hit": bool(transition_checker_hit),
        "checker_match": bool(checker_match),
        "checker_status": (
            "NOT ENFORCED"
            if not moc_enforced else ("PASS" if checker_match else "FAIL")
        ),
        "full_path_checker_hit": bool(full_path_moc_hit),
        "corridor_half_width_m": float(half_width_m),
        "transition_corridor_half_width_m": float(half_width_m),
        "configured_downward_clearance_m": float(
            transition_corridor_cfg["downward_clearance_m"]
        ),
        "transition_vertical_clearance_m": float(
            transition_corridor_cfg["downward_clearance_m"]
        ),
        "cruise_corridor_half_width_m": float(cruise_half_width_m),
        "transition_3d_status": str(transition_3d_status),
        "transition_3d_validation": dict(
            rf.get("transition_3d_validation", {})
        ),
        "transition_3d_corridor_policy": {
            "effective_clearance_formula": (
                "min(configured_clearance, max(0, center_MSL - direction_vertiport_MSL))"
            ),
            "lower_face_formula": "center_MSL - effective_clearance",
            "moc_layer_selection": "floor_at_corridor_lower_face",
            "envelope_semantics": (
                "Discrete fixed-AGL MOC required-safe envelope; not terrain or obstacle height"
            ),
            "vertical_safety_assumption": (
                "blocked cells at higher MOC layers are subsets of lower-layer blocked cells"
            ),
            "all_clear_first_clear_policy": (
                "AGL100 is recorded as the first clear available layer; required-safe MSL stays null"
            ),
            "layer_interpolation": "none",
            "transition_phases": [str(value) for value in TRANSITION_PHASE_NAMES],
        },
        "spatial_constraint_status": {
            "MOC": str(overall_moc_status),
            "NFZ": str(nfz_status),
            "NFZ_reason": str(nfz_reason),
            "airspace": "PASS" if airspace_ok else "FAIL",
            "airspace_reason": str(
                transition_airspace_validation.get("reason", "ok" if airspace_ok else "failed")
            ),
            "self_overlap": str(self_overlap_status),
            "self_overlap_reason": str(self_overlap_reason),
        },
        "along_track_sample_spacing_max_m": 80.0,
        "sample_deduplication_policy": (
            "physically identical shared-edge endpoint samples are counted once; "
            "distinct TF/RF cross-track-normal samples are preserved"
        ),
        "flight_phase_recording_policy": {
            "Flight_Phase": (
                "edge destination phase with vertiport source-phase fallback"
            ),
            "point_phase_columns": [
                "Source_Point_Phase", "Destination_Point_Phase"
            ],
            "landing_transition_start_rule": (
                "boundary point remains cruise; following edge is landing_stage2"
            ),
        },
        "overall_constraint_pass": bool(overall_constraint_ok),
        "overall_constraint_reason": str(overall_constraint_reason),
        "airspace_pass": bool(airspace_ok),
        "min_corridor_distance_enforced": bool(min_distance_enforced),
        "min_corridor_distance_pass": (
            bool(min_distance_ok) if min_distance_enforced else None
        ),
        "transition_feasible": bool(rf.get("transition_feasible", True)),
        "transition_fail_reason": str(rf.get("transition_fail_reason", "ok")),
        "transition_fail_reasons": [
            str(value) for value in rf.get("transition_fail_reasons", [])
        ],
        "transition_validation_checks": validation_checks,
        "rf_feasible": rf_geometry_feasible,
        "combined_rf_transition_feasible": combined_rf_transition_feasible,
        "rf_had_clamp": rf_had_clamp,
        "rf_status": rf_status,
        "rf_arc_count": int(len(all_rf_segments)),
        "transition_rf_arc_count": int(len(rf_segments)),
        "full_path_rf_arc_count": int(len(all_rf_segments)),
        "rf_min_allowed_radius_m": float(rf_min_allowed_radius_m),
        "rf_min_actual_radius_m": (
            None if min_actual_rf_radius_m is None else float(min_actual_rf_radius_m)
        ),
        "rf_radius_pass": (
            bool(rf_radius_pass) if all_rf_segments else None
        ),
        "rf_radius_status": (
            "NOT APPLICABLE"
            if not all_rf_segments else ("PASS" if rf_radius_pass else "FAIL")
        ),
        "transition_rf_min_actual_radius_m": (
            None
            if transition_min_actual_rf_radius_m is None
            else float(transition_min_actual_rf_radius_m)
        ),
        "transition_rf_radius_pass": transition_rf_radius_pass,
        "transition_rf_radius_status": (
            "NOT APPLICABLE"
            if transition_rf_radius_pass is None
            else ("PASS" if transition_rf_radius_pass else "FAIL")
        ),
        "rf_fail_reason_counts": {
            str(key): int(value)
            for key, value in rf.get("fail_reason_counts", {}).items()
        },
        "directions": direction_summary,
    }
    if not summary["layer_sample_count_matches_total"]:
        raise RuntimeError(
            "MOC layer sample counts do not sum to the full transition sample count."
        )

    fig = plt.figure("Transition MOC Validation", figsize=(16, 10))
    grid = fig.add_gridspec(2, 2, width_ratios=(2.25, 1.0), hspace=0.34, wspace=0.24)
    ax_takeoff = fig.add_subplot(grid[0, 0])
    ax_landing = fig.add_subplot(grid[1, 0])
    ax_status = fig.add_subplot(grid[:, 1])
    _plot_transition_moc_profile_axis_v1(
        ax_takeoff, sample_df, "takeoff", rf, moc_enforced
    )
    _plot_transition_moc_profile_axis_v1(
        ax_landing, sample_df, "landing", rf, moc_enforced
    )
    ax_status.set_axis_off()
    min_radius_text = (
        "n/a" if min_actual_rf_radius_m is None else f"{min_actual_rf_radius_m:.1f} m"
    )
    transition_min_radius_text = (
        "n/a"
        if transition_min_actual_rf_radius_m is None
        else f"{transition_min_actual_rf_radius_m:.1f} m"
    )
    min_distance_status = (
        ("PASS" if min_distance_ok else "FAIL")
        if min_distance_enforced else "NOT ENFORCED"
    )
    rf_radius_status = (
        "NOT APPLICABLE"
        if not all_rf_segments else ("PASS" if rf_radius_pass else "FAIL")
    )
    transition_rf_radius_status = (
        "NOT APPLICABLE"
        if transition_rf_radius_pass is None
        else ("PASS" if transition_rf_radius_pass else "FAIL")
    )
    takeoff_sector_status = (
        "NOT AVAILABLE"
        if "takeoff_sector_heading" not in validation_checks
        else ("PASS" if validation_checks["takeoff_sector_heading"] else "FAIL")
    )
    landing_sector_status = (
        "NOT AVAILABLE"
        if "landing_sector_heading" not in validation_checks
        else ("PASS" if validation_checks["landing_sector_heading"] else "FAIL")
    )
    station_rows_all = pd.concat([
        _transition_station_rows_v1(sample_df, "takeoff"),
        _transition_station_rows_v1(sample_df, "landing"),
    ], ignore_index=True)
    station_margins = pd.to_numeric(
        station_rows_all.get("Station_Vertical_Margin_m"), errors="coerce"
    )
    has_no_clear_layer = bool(
        station_rows_all["Station_MOC_Envelope_Class"].eq(
            "NO_CLEAR_LAYER"
        ).any()
    )
    if has_no_clear_layer:
        minimum_margin_text = "NO CLEAR LAYER"
    elif not bool(station_margins.notna().any()):
        minimum_margin_text = "n/a"
    else:
        minimum_margin_text = f"{float(station_margins.min()):.1f} m"
    status_lines = [
        "FINAL BALANCED TRANSITION AUDIT",
        *(
            [audit_only_notice]
            if audit_only_fixed_transition_geometry else []
        ),
        "",
        f"MOC enforcement       : {'ON' if moc_enforced else 'OFF'}",
        f"MOC transition status : {overall_moc_status}",
        f"MOC samples           : {total_samples}",
        f"Inside / hits / out   : {total_tested} / {total_hits} / {total_out_of_grid}",
        (
            "Checker agreement     : NOT ENFORCED"
            if not moc_enforced
            else f"Checker agreement     : {'PASS' if checker_match else 'FAIL'}"
        ),
        f"Full-path MOC hit     : {'YES' if full_path_moc_hit else 'NO'}",
        f"Transition half-width : +/-{float(half_width_m):.1f} m",
        f"Downward clearance    : {float(transition_corridor_cfg['downward_clearance_m']):.1f} m max",
        f"Minimum MOC margin    : {minimum_margin_text}",
        (
            "Clearance taper       : 0m at port -> configured at center AGL"
            f"{float(transition_corridor_cfg['downward_clearance_m']):.0f}"
        ),
        "",
        f"Overall constraints   : {'PASS' if overall_constraint_ok else 'FAIL'}",
        f"Constraint reason     : {overall_constraint_reason}",
        f"Airspace              : {'PASS' if airspace_ok else 'FAIL'}",
        f"NFZ envelope          : {nfz_status}",
        f"Self-overlap envelope : {self_overlap_status}",
        f"Transition 3D status  : {transition_3d_status}",
        f"Minimum distance      : {min_distance_status}",
        f"Transition validation : {validation_pass_count}/{validation_total_count} PASS",
        f"Transition feasible   : {'PASS' if rf.get('transition_feasible', True) else 'FAIL'}",
        f"Takeoff sector heading: {takeoff_sector_status}",
        f"Landing sector heading: {landing_sector_status}",
        "",
        f"RF geometry (all path): {rf_status}",
        f"RF feasible / clamp   : {rf_geometry_feasible} / {rf_had_clamp}",
        f"RF arcs (all path)    : {len(all_rf_segments)}",
        f"RF min actual         : {min_radius_text}",
        f"RF min allowed        : {float(rf_min_allowed_radius_m):.1f} m",
        f"RF radius check       : {rf_radius_status}",
        f"Transition RF arcs    : {len(rf_segments)}",
        f"Transition RF min     : {transition_min_radius_text}",
        f"Transition RF radius  : {transition_rf_radius_status}",
        "",
        "Discrete fixed-AGL MOC envelope; not terrain.",
        "Below AGL200 query altitude, AGL100 is the enforced proxy layer.",
        "OUT_OF_GRID samples are WARN, never safe samples.",
    ]
    ax_status.text(
        0.02, 0.98, "\n".join(status_lines),
        transform=ax_status.transAxes, ha="left", va="top",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="gray", alpha=0.92),
    )
    fig.suptitle(
        (
            "Final Balanced Path: 3D transition MOC clearance and RF validation"
            + (
                f"\n{audit_only_notice}"
                if audit_only_fixed_transition_geometry else ""
            )
        ),
        fontsize=14,
    )
    fig.savefig(
        working_dir / "00_transition_moc_validation.png",
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(fig)

    for direction, overview_name in (
        ("takeoff", "01_takeoff_moc_layers_overview.png"),
        ("landing", "02_landing_moc_layers_overview.png"),
    ):
        ordered_layers = list(used_layers_by_direction[direction])
        panel_count = len(ordered_layers)
        column_count = min(3, max(1, panel_count))
        row_count = int(np.ceil(panel_count / column_count))
        overview_fig = plt.figure(
            f"{direction.title()} MOC Layers Overview",
            figsize=(5.2 * column_count, 4.7 * row_count),
        )
        extent = _transition_moc_extent_v1(sample_df, direction, map_extent)
        for panel_no, layer_idx in enumerate(ordered_layers, start=1):
            panel_ax = overview_fig.add_subplot(
                row_count, column_count, panel_no, projection=request.crs
            )
            _plot_moc_transition_layer_axis_v1(
                panel_ax,
                request,
                extent,
                rf,
                sample_df,
                direction,
                layer_idx,
                moc_risk,
                moc_enforced,
                half_width_m,
                lat_lim,
                lon_lim,
                airspace_center_lla,
                airspace_radius_m,
                forbidden_zones,
                start_vertiport,
                end_vertiport,
                compact=True,
            )
        overview_fig.suptitle(
            (
                f"{direction.title()} transition MOC layers in flight order"
                + (
                    f"\n{audit_only_notice}"
                    if audit_only_fixed_transition_geometry else ""
                )
            ),
            fontsize=13,
        )
        overview_fig.tight_layout(rect=[0, 0, 1, 0.96])
        overview_fig.savefig(working_dir / overview_name, dpi=165, bbox_inches="tight")
        plt.close(overview_fig)

        for sequence_no, layer_idx in enumerate(ordered_layers, start=1):
            moc_agl_m = int(MOC_AGL_LEVELS_M[layer_idx])
            frame_name = f"{direction}_{sequence_no:03d}_agl{moc_agl_m:04d}.png"
            frame_fig = plt.figure(
                f"{direction.title()} MOC AGL{moc_agl_m}", figsize=(14, 10)
            )
            frame_fig.subplots_adjust(left=0.05, right=0.76)
            frame_ax = frame_fig.add_subplot(1, 1, 1, projection=request.crs)
            _plot_moc_transition_layer_axis_v1(
                frame_ax,
                request,
                extent,
                rf,
                sample_df,
                direction,
                layer_idx,
                moc_risk,
                moc_enforced,
                half_width_m,
                lat_lim,
                lon_lim,
                airspace_center_lla,
                airspace_radius_m,
                forbidden_zones,
                start_vertiport,
                end_vertiport,
                compact=False,
            )
            if audit_only_fixed_transition_geometry:
                frame_fig.suptitle(audit_only_notice, fontsize=12)
            frame_fig.savefig(working_dir / frame_name, dpi=175, bbox_inches="tight")
            plt.close(frame_fig)

    with open(working_dir / json_name, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)
    working_dir.rename(snapshot_dir)
    return summary


def _save_generation_snapshots(
    gen_history,
    out_dir,
    request,
    map_extent,
    airspace_center_lla,
    airspace_radius_m,
    forbidden_zones,
    bb_full,
    waypoints,
    start_vertiport,
    end_vertiport,
    takeoff_complete,
    landing_entry,
    use_takeoff_landing_transition,
    use_two_stage_transition,
    transition_structure_mode,
    takeoff_optimized_transition_actual,
    landing_optimized_transition_actual,
    objective_names,
    objective_weights,
    altitude_levels,
    apply_rf_corridor_fn,
    output_rf_view_fn,
    W_half,
    transition_corridor_cfg,
    moc_plot_2d,
    lat_lim,
    lon_lim,
):
    """Save per-generation corridor snapshot figures from NSGA history."""
    gen_snap_dir = out_dir / "gen_snapshots"
    gen_snap_dir.mkdir(parents=True, exist_ok=True)

    for gh in gen_history:
        gno = int(gh["gen"])
        gpop = gh["population"]
        gf = gh["f_vals"]
        if not gpop or gf.size == 0:
            continue

        greps = pick_representatives(
            gpop,
            gf,
            objective_weights,
            feasible=gh.get("feasible"),
        )

        figg = plt.figure(f"Generation {gno}: Evolved Corridor", figsize=(14, 10))
        figg.subplots_adjust(left=0.05, right=0.72)
        gxg = figg.add_subplot(1, 1, 1, projection=request.crs)
        gxg.set_extent(map_extent)
        gxg.add_image(request, 13)
        gxg.set_title(_title_with_altitude(
            f"Generation {gno} Corridor (RF Turn applied)",
            altitude_levels,
            start_vertiport,
        ))
        draw_vertiport_radius_rings(gxg, airspace_center_lla, radii_m=(airspace_radius_m,))
        plot_forbidden_zones(gxg, forbidden_zones, face_alpha=0.10, edge_alpha=0.80)
        plot_moc_binary_overlay(
            gxg, moc_plot_2d, lat_lim, lon_lim,
            label="MOC=1 (Corridor-Prohibited)",
            fill_color="magenta", fill_alpha=0.18,
        )

        gxg.plot(bb_full[:, 1], bb_full[:, 0], "r--", linewidth=1.5, transform=ccrs.Geodetic(),
                 label="Backbone", zorder=4)
        gxg.scatter(waypoints[:, 1], waypoints[:, 0], s=60, c="orange", edgecolors="k",
                    linewidths=0.5, marker="o", transform=ccrs.Geodetic(), label="Waypoints", zorder=6)
        gxg.scatter([start_vertiport[1]], [start_vertiport[0]], s=120, c="red", edgecolors="k",
                    marker="s", transform=ccrs.Geodetic(), label="Start Vertiport", zorder=7)
        gxg.scatter([end_vertiport[1]], [end_vertiport[0]], s=120, c="crimson", edgecolors="k",
                    marker="D", transform=ccrs.Geodetic(), label="End Vertiport", zorder=7)
        if not use_takeoff_landing_transition:
            gxg.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90, c=TAKEOFF_TRANSITION_COLOR,
                        marker="^", transform=ccrs.Geodetic(), label="Takeoff_End", zorder=7)
            gxg.scatter([landing_entry[1]], [landing_entry[0]], s=90, c=LANDING_TRANSITION_COLOR,
                        marker="v", transform=ccrs.Geodetic(), label="Landing_End", zorder=7)
        elif transition_structure_mode == TRANSITION_STRUCTURE_FIXED_ONLY:
            gxg.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90,
                        c=TAKEOFF_TRANSITION_COLOR, marker="^", transform=ccrs.Geodetic(),
                        label="Takeoff Transition End", zorder=7)
            gxg.scatter([landing_entry[1]], [landing_entry[0]], s=90,
                        c=LANDING_TRANSITION_COLOR, marker="v", transform=ccrs.Geodetic(),
                        label="Landing Transition Start", zorder=7)
        elif use_two_stage_transition and _seg_dist_m(start_vertiport, takeoff_complete) > 0.5:
            if bool(takeoff_optimized_transition_actual):
                gxg.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90, facecolors="none",
                            edgecolors=TAKEOFF_TRANSITION_COLOR, linewidths=1.4, marker="o",
                            transform=ccrs.Geodetic(), label="Takeoff Stage1 End", zorder=7)
            else:
                gxg.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90,
                            c=TAKEOFF_TRANSITION_COLOR, edgecolors="k", linewidths=0.7, marker="^",
                            transform=ccrs.Geodetic(), label="Takeoff Transition End", zorder=7)
        if use_takeoff_landing_transition and use_two_stage_transition and _seg_dist_m(end_vertiport, landing_entry) > 0.5:
            if bool(landing_optimized_transition_actual):
                gxg.scatter([landing_entry[1]], [landing_entry[0]], s=90, facecolors="none",
                            edgecolors=LANDING_TRANSITION_COLOR, linewidths=1.4, marker="o",
                            transform=ccrs.Geodetic(), label="Landing Stage1 Start", zorder=7)
            else:
                gxg.scatter([landing_entry[1]], [landing_entry[0]], s=90,
                            c=LANDING_TRANSITION_COLOR, edgecolors="k", linewidths=0.7, marker="v",
                            transform=ccrs.Geodetic(), label="Landing Transition Start", zorder=7)

        rep_labels = objective_names + ["Balanced"]
        rep_colors = [plt.cm.tab10(i % 10) for i in range(len(objective_names))] + ["black"]
        for ri, rep in enumerate(greps):
            rf = output_rf_view_fn(apply_rf_corridor_fn(rep))
            rp = rf["path"]
            segs = rf["segments"]
            col = rep_colors[ri] if ri < len(rep_colors) else rep_colors[-1]
            lab = rep_labels[ri] if ri < len(rep_labels) else f"Rep{ri}"

            _plot_corridor_width_by_phase_v1(
                gxg,
                rf,
                W_half,
                transition_corridor_cfg,
                color=col,
                alpha=0.08,
            )

            _plot_rf_segments_by_phase(
                gxg, rf, cruise_color=col, tf_lw=1.5, rf_lw=2.0,
                transform=ccrs.Geodetic(), zorder=8,
                transition_labels=(ri == 0), draw_rf_markers=False,
            )
            _plot_transition_phase_markers(
                gxg, rf, transform=ccrs.Geodetic(), zorder=11, labels=(ri == 0),
                include_stage1=False, general_output=True,
            )

            gxg.plot([], [], "-", color=col, linewidth=1.5, label=f"{lab} (TF)")
            gxg.plot([], [], "-", color=col, linewidth=2.5, label=f"{lab} (RF arc)")

        gxg.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, framealpha=0.9)
        gen_png = gen_snap_dir / f"gen_{gno:03d}_corridor.png"
        figg.savefig(gen_png, dpi=150, bbox_inches="tight")
        print(f"Saved {gen_png}")
        plt.close(figg)


def _compute_final_feasibility(
    pop,
    apply_rf_corridor_fn,
    eval_corridor_objectives_fn,
    airspace_center_lla,
    airspace_radius_m,
    airspace_alt_min_m,
    airspace_alt_max_m,
    min_corridor_distance_m,
    cruise_half_width_m,
    transition_corridor_cfg,
):
    """Evaluate final feasibility mask and RF no-clamp count for a population."""
    rf_no_clamp_count = 0
    feas_mask = []
    for i in range(len(pop)):
        rf = apply_rf_corridor_fn(pop[i])
        if not bool(rf.get("had_clamp", False)):
            rf_no_clamp_count += 1
        full_path = rf["path"]
        flight_phases = rf.get("flight_phases")
        _, feas = eval_corridor_objectives_fn(full_path, flight_phases)
        air_ok, _, _ = _is_path_inside_airspace_envelope_v1(
            full_path,
            flight_phases,
            airspace_center_lla[:2],
            airspace_radius_m,
            cruise_half_width_m=cruise_half_width_m,
            alt_min_m=airspace_alt_min_m,
            alt_max_m=airspace_alt_max_m,
            transition_corridor_cfg=transition_corridor_cfg,
        )
        dist_ok = True
        if min_corridor_distance_m > 0.0:
            dist_ok = _path_total_3d_distance_m(full_path) + 1e-6 >= min_corridor_distance_m
        feas = bool(feas and bool(rf.get("feasible", False)) and air_ok and dist_ok)
        feas_mask.append(bool(feas))

    feasible_count = sum(feas_mask)
    return feasible_count, rf_no_clamp_count, feas_mask


def _apply_rf_corridor_path(
    path_core,
    start_vertiport,
    end_vertiport,
    ground_speed_mps,
    bank_angle_deg,
    num_arc_points,
    look_ahead,
    look_ahead_threshold_m,
    look_ahead_min_scale,
    look_ahead_window,
    use_boundary_heading=False,
    rf_debug_level="off",
):
    """Apply RF turns on full corridor with fixed run configuration."""
    return apply_rf_turns_full_corridor(
        path_core,
        start_vertiport,
        end_vertiport,
        ground_speed_mps,
        bank_angle_deg,
        num_arc_points,
        look_ahead,
        look_ahead_threshold_m,
        look_ahead_min_scale,
        look_ahead_window,
        use_boundary_heading=use_boundary_heading,
        rf_debug_level=rf_debug_level,
    )


def _evaluate_corridor_objectives_path(path_points, flight_phases, eval_cfg):
    """Evaluate objectives/constraints for a corridor path using prebuilt config."""
    return evaluate_objectives_with_constraints_gp(
        path_points,
        flight_phases=flight_phases,
        **eval_cfg,
    )


def _make_initial_population(
    N_init,
    use_wp_skip_generator,
    init_pop_skip_mix_ratio,
    backbone,
    waypoints,
    takeoff_complete,
    landing_entry,
    wp_perturb_radius_m,
    min_extra_nodes_per_seg,
    max_extra_nodes_per_seg,
    safe_nodes_by_seg,
    emergency_points,
    emergency_strip_m,
    is_fixed,
    wp_perturb_steps,
    wp_skip_prob,
    min_seg_for_extra_nodes_m,
    airspace_center_lla,
    airspace_radius_m,
    airspace_alt_min_m,
    airspace_alt_max_m,
    enforce_mandatory_wp_order,
):
    """Create up to N_init airspace-filtered initial solutions."""
    pop = []
    max_draws = int(max(3 * N_init, 50))
    draws = 0
    can_use_wp_skip_generator = bool(use_wp_skip_generator) and waypoints is not None and np.size(waypoints) > 0
    skip_ratio = float(np.clip(init_pop_skip_mix_ratio, 0.0, 1.0)) if can_use_wp_skip_generator else 0.0
    n_all_wp = int(round(N_init * (1.0 - skip_ratio)))
    while len(pop) < N_init and draws < max_draws:
        i = len(pop)
        draws += 1
        if i < n_all_wp:
            sol = generate_single_initial_solution(
                backbone, wp_perturb_radius_m,
                min_extra_nodes_per_seg, max_extra_nodes_per_seg,
                safe_nodes_by_seg, emergency_points, emergency_strip_m,
                is_fixed,
                wp_perturb_steps=wp_perturb_steps,
                min_seg_for_extra_nodes_m=min_seg_for_extra_nodes_m,
            )
        else:
            if can_use_wp_skip_generator:
                sol = generate_single_initial_solution_with_skip(
                    waypoints, takeoff_complete, landing_entry,
                    wp_perturb_radius_m,
                    min_extra_nodes_per_seg, max_extra_nodes_per_seg,
                    safe_nodes_by_seg,
                    emergency_points, emergency_strip_m,
                    wp_perturb_steps=wp_perturb_steps,
                    wp_skip_prob=wp_skip_prob,
                    min_seg_for_extra_nodes_m=min_seg_for_extra_nodes_m,
                )
            else:
                sol = generate_single_initial_solution(
                    backbone, wp_perturb_radius_m,
                    min_extra_nodes_per_seg, max_extra_nodes_per_seg,
                    safe_nodes_by_seg, emergency_points, emergency_strip_m,
                    is_fixed,
                    wp_perturb_steps=wp_perturb_steps,
                    min_seg_for_extra_nodes_m=min_seg_for_extra_nodes_m,
                )
        if is_path_inside_airspace(
            sol,
            airspace_center_lla[:2],
            airspace_radius_m,
            alt_min_m=airspace_alt_min_m,
            alt_max_m=airspace_alt_max_m,
        ):
            if enforce_mandatory_wp_order:
                sol = _enforce_mandatory_wp_order(sol, backbone)
            pop.append(sol)
    return pop


def _evaluate_initial_candidates(
    candidate_pop,
    backbone,
    enforce_mandatory_wp_order,
    apply_rf_fn,
    eval_constraints_with_reason_fn,
    airspace_center_lla,
    airspace_radius_m,
    airspace_alt_min_m,
    airspace_alt_max_m,
    min_corridor_distance_m,
    cruise_half_width_m,
    transition_corridor_cfg,
):
    """Evaluate RF + constraints for initial candidates and return summary stats."""
    rf_ok_list = []
    for c in candidate_pop:
        c_eval = _enforce_mandatory_wp_order(c, backbone) if enforce_mandatory_wp_order else c
        rf = apply_rf_fn(c_eval)
        rf_ok_list.append((
            rf["feasible"],
            bool(rf.get("had_clamp", False)),
            rf,
            c_eval,
            (
                str(rf.get("transition_fail_reason", "transition_infeasible"))
                if not bool(rf.get("transition_feasible", True))
                else "rf_geometry_infeasible"
            ),
        ))

    rf_cnt = sum(1 for ok, _, _, _, _ in rf_ok_list if ok)
    rf_no_clamp_cnt = sum(1 for ok, had_clamp, _, _, _ in rf_ok_list if ok and (not had_clamp))

    both_cnt = 0
    cst_cnt = 0
    air_cnt = 0
    dist_cnt = 0
    reason_counts = {}
    feasible_init = []
    for rf_ok, _had_clamp, _rf, _c_eval, rf_reason in rf_ok_list:
        if not rf_ok:
            reason_counts[rf_reason] = int(reason_counts.get(rf_reason, 0) + 1)

    if rf_cnt > 0:
        for rf_ok, _had_clamp, rf, c_eval, rf_reason in rf_ok_list:
            if not rf_ok:
                continue
            full_path = rf["path"]
            flight_phases = rf.get("flight_phases")
            _, cst_ok, reason = eval_constraints_with_reason_fn(
                full_path, flight_phases=flight_phases
            )
            air_ok, air_reason, _ = _is_path_inside_airspace_envelope_v1(
                full_path,
                flight_phases,
                airspace_center_lla[:2],
                airspace_radius_m,
                cruise_half_width_m=cruise_half_width_m,
                alt_min_m=airspace_alt_min_m,
                alt_max_m=airspace_alt_max_m,
                transition_corridor_cfg=transition_corridor_cfg,
            )
            dist_ok = True
            if min_corridor_distance_m > 0.0:
                dist_ok = _path_total_3d_distance_m(full_path) + 1e-6 >= min_corridor_distance_m

            if cst_ok:
                cst_cnt += 1
            else:
                reason_counts[reason] = int(reason_counts.get(reason, 0) + 1)
            if air_ok:
                air_cnt += 1
            else:
                reason_counts[air_reason] = int(reason_counts.get(air_reason, 0) + 1)
            if dist_ok:
                dist_cnt += 1

            if cst_ok and air_ok and dist_ok:
                both_cnt += 1
                feasible_init.append(c_eval)

    return {
        "rf_cnt": rf_cnt,
        "rf_no_clamp_cnt": rf_no_clamp_cnt,
        "both_cnt": both_cnt,
        "cst_cnt": cst_cnt,
        "air_cnt": air_cnt,
        "dist_cnt": dist_cnt,
        "reason_counts": reason_counts,
        "feasible_init": feasible_init,
    }


def _export_route_outputs(
    rows,
    rf_best,
    rf_output,
    Norm_RT,
    AirRisk,
    altitude_levels,
    risk_altitude_levels,
    use_heading_map,
    air_thr_global,
    lat_lim,
    lon_lim,
    NoiseRisk,
    NoiseRiskDb,
    cell_size,
    refine_scales,
    min_corridor_distance_m,
    out_dir,
    params_dict,
    w_noise,
    noise_floor_db,
    evaluate_objectives_kwargs,
    start_vertiport,
    airspace_center_lla,
    airspace_radius_m,
    airspace_radius_km,
    airspace_alt_min_m,
    airspace_alt_max_m,
    forbidden_zones,
    use_takeoff_landing_transition,
    end_vertiport,
    takeoff_complete,
    landing_entry,
    corridor_lat_default,
    corridor_lon_default,
    waypoint_alt_fixed_m,
):
    """Build route DataFrames and export Excel/map artifacts for balanced corridor."""
    evaluation_flight_phases = np.asarray(
        rf_best.get("flight_phases", np.full(len(rf_best.get("path", [])), FLIGHT_PHASE_CRUISE)),
        dtype=object,
    ).reshape(-1)
    output_flight_phases = np.asarray(
        rf_output.get(
            "flight_phases",
            np.full(len(rf_output.get("path", [])), FLIGHT_PHASE_CRUISE),
        ),
        dtype=object,
    ).reshape(-1)
    balanced_phase_counts = {
        phase: int(np.count_nonzero(output_flight_phases == phase))
        for phase in (
            FLIGHT_PHASE_VERTIPORT,
            FLIGHT_PHASE_TAKEOFF_STAGE1,
            FLIGHT_PHASE_TAKEOFF_STAGE2,
            FLIGHT_PHASE_CRUISE,
            FLIGHT_PHASE_LANDING_STAGE2,
            FLIGHT_PHASE_LANDING_STAGE1,
        )
    }

    def _as_optional_point(value):
        if value is None:
            return None
        point = np.asarray(value, dtype=float).reshape(-1)
        if point.size < 3 or not np.all(np.isfinite(point[:3])):
            return None
        return point[:3].astype(float)

    def _point_json(point):
        if point is None:
            return None
        return {
            "lat": float(point[0]),
            "lon": float(point[1]),
            "alt_m": float(point[2]),
        }

    fixed_output_suppressed = bool(
        params_dict.get("fixed_transition_general_output_suppressed", False)
    )
    takeoff_optimized_transition_actual = bool(
        params_dict.get("takeoff_optimized_transition_actual", False)
    )
    landing_optimized_transition_actual = bool(
        params_dict.get("landing_optimized_transition_actual", False)
    )
    has_takeoff_stage1 = bool(
        not fixed_output_suppressed
        and takeoff_optimized_transition_actual
        and np.count_nonzero(
            evaluation_flight_phases == FLIGHT_PHASE_TAKEOFF_STAGE1
        ) > 0
    )
    has_landing_stage1 = bool(
        not fixed_output_suppressed
        and landing_optimized_transition_actual
        and np.count_nonzero(
            evaluation_flight_phases == FLIGHT_PHASE_LANDING_STAGE1
        ) > 0
    )
    takeoff_stage1_end = (
        _as_optional_point(rf_best.get("takeoff_stage1_end"))
        if has_takeoff_stage1 else None
    )
    landing_stage1_start = (
        _as_optional_point(rf_best.get("landing_stage1_start"))
        if has_landing_stage1 else None
    )
    takeoff_transition_end = (
        _as_optional_point(rf_best.get("takeoff_transition_end"))
        if bool(use_takeoff_landing_transition) else None
    )
    landing_transition_start = (
        _as_optional_point(rf_best.get("landing_transition_start"))
        if bool(use_takeoff_landing_transition) else None
    )
    takeoff_output_point = (
        takeoff_transition_end
        if takeoff_transition_end is not None
        else np.asarray(takeoff_complete, dtype=float).reshape(3)
    )
    landing_output_point = (
        landing_transition_start
        if landing_transition_start is not None
        else np.asarray(landing_entry, dtype=float).reshape(3)
    )
    transition_meta = rf_best.get("transition_meta", {})
    takeoff_cfg = params_dict.get("takeoff_transition_meta", {})
    landing_cfg = params_dict.get("landing_transition_meta", {})

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Point_No"] = np.arange(len(df), dtype=int)
    route_flight_phases = np.asarray(
        df.get("Flight_Phase", pd.Series(dtype=object)), dtype=object
    ).reshape(-1)
    phase_counts = {
        phase: int(np.count_nonzero(route_flight_phases == phase))
        for phase in (
            FLIGHT_PHASE_VERTIPORT,
            FLIGHT_PHASE_TAKEOFF_STAGE1,
            FLIGHT_PHASE_TAKEOFF_STAGE2,
            FLIGHT_PHASE_CRUISE,
            FLIGHT_PHASE_LANDING_STAGE2,
            FLIGHT_PHASE_LANDING_STAGE1,
        )
    }

    dist_prev_2d = [0.0]
    dist_prev_3d = [0.0]
    cum_2d = [0.0]
    cum_3d = [0.0]
    for i in range(1, len(df)):
        p_prev = np.array([
            float(df.loc[i - 1, "Lat"]), float(df.loc[i - 1, "Lon"]), float(df.loc[i - 1, "Altitude_MSL_m"])
        ], dtype=float)
        p_cur = np.array([
            float(df.loc[i, "Lat"]), float(df.loc[i, "Lon"]), float(df.loc[i, "Altitude_MSL_m"])
        ], dtype=float)
        d2 = _seg_dist_m(p_prev, p_cur)
        d3 = _seg_dist_3d_m(p_prev, p_cur)
        dist_prev_2d.append(d2)
        dist_prev_3d.append(d3)
        cum_2d.append(cum_2d[-1] + d2)
        cum_3d.append(cum_3d[-1] + d3)

    df["Dist_From_Prev_2D_km"] = np.asarray(dist_prev_2d, dtype=float) / 1000.0
    df["Dist_From_Prev_3D_km"] = np.asarray(dist_prev_3d, dtype=float) / 1000.0
    df["Cumulative_Dist_2D_km"] = np.asarray(cum_2d, dtype=float) / 1000.0
    df["Cumulative_Dist_3D_km"] = np.asarray(cum_3d, dtype=float) / 1000.0

    full_path = np.asarray(rf_best["path"], dtype=float)
    output_path = np.asarray(rf_output["path"], dtype=float).reshape(-1, 3)
    p2d_cum = [0.0]
    pcum = [0.0]
    for i in range(1, full_path.shape[0]):
        d2 = _seg_dist_m(full_path[i - 1], full_path[i])
        d3 = _seg_dist_3d_m(full_path[i - 1], full_path[i])
        p2d_cum.append(p2d_cum[-1] + d2)
        pcum.append(pcum[-1] + d3)

    route_points = df[["Lat", "Lon", "Altitude_MSL_m"]].to_numpy(dtype=float)
    pt_ground, pt_air, _ = _sample_point_risks(
        route_points, Norm_RT, AirRisk, risk_altitude_levels,
        use_heading_map, air_thr_global, lat_lim, lon_lim
    )
    pt_noise_norm = _sample_point_noise(route_points, NoiseRisk, risk_altitude_levels, lat_lim, lon_lim)
    pt_noise_db = _sample_point_noise(route_points, NoiseRiskDb, risk_altitude_levels, lat_lim, lon_lim)
    df["Combined_Risk"] = pt_ground + pt_air + pt_noise_norm
    df["Ground_Risk"] = pt_ground
    df["Air_Risk"] = pt_air
    df["Noise_Risk_Norm_0to1"] = pt_noise_norm
    df["Noise_Lden_dB_Above_Floor"] = pt_noise_db
    df["Corridor_Tag"] = "Balanced_Optimal"
    df["Ground_Speed_kmh"] = pd.to_numeric(df["Ground_Speed_mps"], errors="coerce") * 3.6

    total_ground_risk, total_air_risk, _ = _aggregate_path_risks(
        full_path, Norm_RT, AirRisk, risk_altitude_levels,
        use_heading_map, cell_size, refine_scales,
        air_thr_global, lat_lim, lon_lim
    )
    total_noise_risk_norm = _aggregate_path_noise(
        full_path, NoiseRisk, risk_altitude_levels, cell_size, refine_scales, lat_lim, lon_lim
    )
    total_noise_db_after_floor = _aggregate_path_noise(
        full_path, NoiseRiskDb, risk_altitude_levels, cell_size, refine_scales, lat_lim, lon_lim
    )
    total_combined_all_risk = float(total_ground_risk + total_air_risk + total_noise_risk_norm)
    total_corridor_dist_2d_km = (float(p2d_cum[-1]) if len(p2d_cum) > 0 else 0.0) / 1000.0
    total_corridor_dist_3d_m = float(pcum[-1]) if len(pcum) > 0 else 0.0
    total_corridor_dist_3d_km = total_corridor_dist_3d_m / 1000.0
    output_corridor_dist_2d_m = _polyline_cumulative_horizontal_m(output_path)
    output_corridor_dist_2d_km = (
        float(output_corridor_dist_2d_m[-1]) / 1000.0
        if output_corridor_dist_2d_m.size else 0.0
    )
    output_corridor_dist_3d_m = _path_total_3d_distance_m(output_path)
    output_corridor_dist_3d_km = output_corridor_dist_3d_m / 1000.0
    _ = (min_corridor_distance_m <= 0.0) or (total_corridor_dist_3d_m + 1e-6 >= min_corridor_distance_m)

    f_best, _ = evaluate_objectives_with_constraints_gp(
        full_path,
        flight_phases=evaluation_flight_phases,
        **evaluate_objectives_kwargs,
    )
    objective_weighting_result = dict(
        params_dict.get("objective_weighting_result", {})
    )
    objective_names_audit = [
        str(value)
        for value in objective_weighting_result.get("objective_names", [])
    ]
    objective_summary_rows = []
    for objective_name in objective_names_audit:
        metric_name = objective_name.replace(" ", "_")
        objective_summary_rows.extend([
            {
                "Metric": f"Objective_Weight_{metric_name}",
                "Value": objective_weighting_result.get(
                    "configured_weights", {}
                ).get(objective_name),
            },
            {
                "Metric": f"Objective_Normalized_Weight_{metric_name}",
                "Value": objective_weighting_result.get(
                    "normalized_weights", {}
                ).get(objective_name),
            },
            {
                "Metric": f"Objective_Normalization_Min_{metric_name}",
                "Value": objective_weighting_result.get(
                    "normalization_minimum", {}
                ).get(objective_name),
            },
            {
                "Metric": f"Objective_Normalization_Max_{metric_name}",
                "Value": objective_weighting_result.get(
                    "normalization_maximum", {}
                ).get(objective_name),
            },
            {
                "Metric": f"Balanced_Raw_Objective_{metric_name}",
                "Value": objective_weighting_result.get(
                    "selected_raw_objectives", {}
                ).get(objective_name),
            },
            {
                "Metric": f"Balanced_Normalized_Objective_{metric_name}",
                "Value": objective_weighting_result.get(
                    "selected_normalized_objectives", {}
                ).get(objective_name),
            },
            {
                "Metric": f"Balanced_Weighted_Contribution_{metric_name}",
                "Value": objective_weighting_result.get(
                    "selected_weighted_contributions", {}
                ).get(objective_name),
            },
        ])

    transition_3d_validation = dict(
        rf_best.get(
            "transition_3d_validation",
            params_dict.get("transition_3d_validation", {}),
        )
    )
    sector_selection = dict(params_dict.get("sector_selection_analysis", {}))
    sector_selected_pair = dict(sector_selection.get("selected_pair", {}))
    df_summary = pd.DataFrame([
        {"Metric": "Selected_Corridor", "Value": "Balanced_Optimal"},
        {"Metric": "Total_Corridor_Distance_2D_km", "Value": total_corridor_dist_2d_km},
        {"Metric": "Total_Corridor_Distance_3D_km", "Value": total_corridor_dist_3d_km},
        {"Metric": "Evaluation_Full_Path_Distance_2D_km", "Value": total_corridor_dist_2d_km},
        {"Metric": "Evaluation_Full_Path_Distance_3D_km", "Value": total_corridor_dist_3d_km},
        {"Metric": "Public_Corridor_Distance_2D_km", "Value": output_corridor_dist_2d_km},
        {"Metric": "Public_Corridor_Distance_3D_km", "Value": output_corridor_dist_3d_km},
        {"Metric": "Evaluation_Full_Path_Point_Count", "Value": int(full_path.shape[0])},
        {"Metric": "Public_Corridor_Point_Count", "Value": int(output_path.shape[0])},
        {"Metric": "Total_Ground_Risk", "Value": total_ground_risk},
        {"Metric": "Total_Air_Risk", "Value": total_air_risk},
        {"Metric": "Total_Noise_Risk", "Value": total_noise_risk_norm},
        {"Metric": "Total_Combined_Risk", "Value": total_combined_all_risk},
        {"Metric": "Objective_Values_Are_Raw", "Value": True},
        {"Metric": "Objective_Weighting_Formula", "Value": objective_weighting_result.get("formula")},
        {"Metric": "Objective_Normalization_Scope", "Value": objective_weighting_result.get("normalization_scope")},
        {"Metric": "Weighted_Normalized_Objective_Score", "Value": objective_weighting_result.get("weighted_normalized_objective_score")},
        {"Metric": "Transition_Enabled", "Value": bool(use_takeoff_landing_transition)},
        {"Metric": "Sector_Mode_Enabled", "Value": bool(params_dict.get("sector_mode_enabled", False))},
        {"Metric": "Sector_Selection_Mode", "Value": str(sector_selection.get("mode", "unknown"))},
        {"Metric": "Sector_Selection_Status", "Value": str(sector_selection.get("status", "unknown"))},
        {"Metric": "Sector_Selection_Priority", "Value": str(sector_selection.get("selection_priority", "unknown"))},
        {"Metric": "Sector_Wind_Period", "Value": str(sector_selection.get("season", params_dict.get("sector_season", "unknown")))},
        {"Metric": "Sector_Wind_Months", "Value": ",".join(str(v) for v in sector_selection.get("wind_months", []))},
        {"Metric": "Sector_MOC_Safe_Pair_Count", "Value": sector_selection.get("moc_safe_pair_count")},
        {"Metric": "Takeoff_Sector_User", "Value": int(params_dict.get("takeoff_sector_user", 0))},
        {"Metric": "Landing_Sector_User", "Value": int(params_dict.get("landing_sector_user", 0))},
        {"Metric": "Takeoff_Sector_Selected", "Value": sector_selected_pair.get("takeoff_sector", params_dict.get("takeoff_sector_selected"))},
        {"Metric": "Landing_Sector_Selected", "Value": sector_selected_pair.get("landing_sector", params_dict.get("landing_sector_selected"))},
        {"Metric": "Sector_Selected_Combined_Risk", "Value": sector_selected_pair.get("combined_risk_score")},
        {"Metric": "Sector_Selected_Wind_Risk", "Value": sector_selected_pair.get("wind_risk_score")},
        {"Metric": "Sector_Selected_Ground_Risk", "Value": sector_selected_pair.get("ground_risk_score")},
        {"Metric": "Sector_Selected_Air_Risk", "Value": sector_selected_pair.get("air_risk_score")},
        {"Metric": "Sector_Selected_MOC_Issue_Ratio", "Value": sector_selected_pair.get("moc_issue_ratio")},
        {"Metric": "Sector_Takeoff_MOC_Blocked_Cells", "Value": sector_selected_pair.get("takeoff_moc_blocked_cell_count")},
        {"Metric": "Sector_Takeoff_MOC_Tested_Cells", "Value": sector_selected_pair.get("takeoff_moc_tested_cell_count")},
        {"Metric": "Sector_Takeoff_MOC_OUT_OF_GRID", "Value": sector_selected_pair.get("takeoff_moc_out_of_grid_sample_count")},
        {"Metric": "Sector_Landing_MOC_Blocked_Cells", "Value": sector_selected_pair.get("landing_moc_blocked_cell_count")},
        {"Metric": "Sector_Landing_MOC_Tested_Cells", "Value": sector_selected_pair.get("landing_moc_tested_cell_count")},
        {"Metric": "Sector_Landing_MOC_OUT_OF_GRID", "Value": sector_selected_pair.get("landing_moc_out_of_grid_sample_count")},
        {"Metric": "Sector_Diagnostic_Figure", "Value": sector_selection.get("diagnostic_figure")},
        {"Metric": "Transition_Structure_Mode", "Value": str(params_dict.get("transition_structure_mode", "unknown"))},
        {"Metric": "Transition_Structure_Mode_Effective", "Value": str(params_dict.get("transition_structure_mode_effective", "off"))},
        {"Metric": "Transition_Geometry_Mode", "Value": str(params_dict.get("transition_mode", "unknown"))},
        {"Metric": "Two_Stage_Transition_Configured", "Value": bool(params_dict.get("use_two_stage_transition", False))},
        {"Metric": "Two_Stage_Transition_Enabled", "Value": bool(
            use_takeoff_landing_transition and params_dict.get("use_two_stage_transition", False)
        )},
        {"Metric": "Transition_Feasible", "Value": bool(rf_best.get("transition_feasible", True))},
        {"Metric": "Transition_Fail_Reason", "Value": str(rf_best.get("transition_fail_reason", "ok"))},
        {"Metric": "Fixed_Transition_General_Output_Suppressed", "Value": fixed_output_suppressed},
        {"Metric": "Fixed_Transition_Evaluated_But_Not_Exported", "Value": bool(params_dict.get("fixed_transition_evaluated_but_not_exported", False))},
        {"Metric": "MOC_Audit_Includes_Fixed_Transition", "Value": bool(params_dict.get("moc_audit_includes_fixed_transition", False))},
        {"Metric": "Takeoff_Optimized_Transition_Actual", "Value": bool(params_dict.get("takeoff_optimized_transition_actual", False))},
        {"Metric": "Landing_Optimized_Transition_Actual", "Value": bool(params_dict.get("landing_optimized_transition_actual", False))},
        {"Metric": "Takeoff_Stage2_Collapsed_At_Cruise", "Value": bool(params_dict.get("takeoff_stage2_collapsed_at_cruise", False))},
        {"Metric": "Landing_Stage2_Collapsed_At_Cruise", "Value": bool(params_dict.get("landing_stage2_collapsed_at_cruise", False))},
        {"Metric": "Transition_Corridor_Half_Width_m", "Value": float(params_dict.get("transition_corridor_half_width_m", 0.0))},
        {"Metric": "Transition_Corridor_Total_Width_m", "Value": float(2.0 * params_dict.get("transition_corridor_half_width_m", 0.0))},
        {"Metric": "Transition_Downward_Clearance_m", "Value": float(params_dict.get("transition_vertical_clearance_m", 0.0))},
        {"Metric": "Transition_3D_Validation_Status", "Value": str(transition_3d_validation.get("status", "NOT GENERATED"))},
        {"Metric": "Transition_3D_Validation_Reason", "Value": str(transition_3d_validation.get("reason", "unknown"))},
        {"Metric": "Takeoff_Transition_3D_Status", "Value": str(transition_3d_validation.get("directions", {}).get("takeoff", {}).get("status", "NOT GENERATED"))},
        {"Metric": "Landing_Transition_3D_Status", "Value": str(transition_3d_validation.get("directions", {}).get("landing", {}).get("status", "NOT GENERATED"))},
        {"Metric": "Transition_MOC_3D_Status", "Value": str(transition_3d_validation.get("moc_status", "NOT GENERATED"))},
        {"Metric": "Transition_NFZ_Envelope_Status", "Value": str(transition_3d_validation.get("nfz_status", "NOT GENERATED"))},
        {"Metric": "Transition_Airspace_Envelope_Status", "Value": str(transition_3d_validation.get("airspace_status", "NOT GENERATED"))},
        {"Metric": "Transition_Self_Overlap_Status", "Value": str(transition_3d_validation.get("self_overlap_status", "NOT GENERATED"))},
        {"Metric": "Takeoff_Stage1_Distance_m", "Value": float(takeoff_cfg.get("stage1_straight_distance_m", 0.0))},
        {"Metric": "Landing_Stage1_Distance_m", "Value": float(landing_cfg.get("stage1_straight_distance_m", 0.0))},
        {"Metric": "Takeoff_Fixed_Prefix_Requested_m", "Value": float(takeoff_cfg.get("stage1_requested_straight_distance_m", 0.0))},
        {"Metric": "Landing_Fixed_Prefix_Requested_m", "Value": float(landing_cfg.get("stage1_requested_straight_distance_m", 0.0))},
        {"Metric": "Takeoff_Total_Transition_Distance_Configured_m", "Value": params_dict.get("takeoff_total_transition_horizontal_distance_m")},
        {"Metric": "Landing_Total_Transition_Distance_Configured_m", "Value": params_dict.get("landing_total_transition_horizontal_distance_m")},
        {"Metric": "Takeoff_Transition_Distance_2D_m", "Value": float(transition_meta.get("takeoff_transition_total_horizontal_distance_m", 0.0))},
        {"Metric": "Landing_Transition_Distance_2D_m", "Value": float(transition_meta.get("landing_transition_total_horizontal_distance_m", 0.0))},
        {"Metric": "Takeoff_Climb_Angle_deg", "Value": params_dict.get("takeoff_climb_angle_deg", 0.0)},
        {"Metric": "Landing_Descent_Angle_deg", "Value": params_dict.get("landing_descent_angle_deg", 0.0)},
        {"Metric": "Takeoff_Actual_Climb_Angle_deg", "Value": float(params_dict.get("actual_takeoff_angle_deg", 0.0))},
        {"Metric": "Landing_Actual_Descent_Angle_deg", "Value": float(params_dict.get("actual_landing_angle_deg", 0.0))},
        {"Metric": "Takeoff_Stage1_End_Altitude_MSL_m", "Value": (None if takeoff_stage1_end is None else float(takeoff_stage1_end[2]))},
        {"Metric": "Takeoff_Transition_End_Altitude_MSL_m", "Value": (None if takeoff_transition_end is None else float(takeoff_transition_end[2]))},
        {"Metric": "Landing_Transition_Start_Altitude_MSL_m", "Value": (None if landing_transition_start is None else float(landing_transition_start[2]))},
        {"Metric": "Landing_Stage1_Start_Altitude_MSL_m", "Value": (None if landing_stage1_start is None else float(landing_stage1_start[2]))},
        {"Metric": "Phase_Count_Vertiport", "Value": phase_counts[FLIGHT_PHASE_VERTIPORT]},
        {"Metric": "Phase_Count_Takeoff_Stage1", "Value": phase_counts[FLIGHT_PHASE_TAKEOFF_STAGE1]},
        {"Metric": "Phase_Count_Takeoff_Stage2", "Value": phase_counts[FLIGHT_PHASE_TAKEOFF_STAGE2]},
        {"Metric": "Phase_Count_Cruise", "Value": phase_counts[FLIGHT_PHASE_CRUISE]},
        {"Metric": "Phase_Count_Landing_Stage2", "Value": phase_counts[FLIGHT_PHASE_LANDING_STAGE2]},
        {"Metric": "Phase_Count_Landing_Stage1", "Value": phase_counts[FLIGHT_PHASE_LANDING_STAGE1]},
    ] + objective_summary_rows)

    params_dict.update({
        "evaluation_full_path_distance_2d_m": float(total_corridor_dist_2d_km * 1000.0),
        "evaluation_full_path_distance_3d_m": float(total_corridor_dist_3d_m),
        "evaluation_full_path_point_count": int(full_path.shape[0]),
        "public_corridor_distance_2d_m": float(output_corridor_dist_2d_km * 1000.0),
        "public_corridor_distance_3d_m": float(output_corridor_dist_3d_m),
        "public_corridor_point_count": int(output_path.shape[0]),
        "noise_result_summary": {
            "total_noise_risk_normalized": float(total_noise_risk_norm),
            "total_noise_lden_db_after_floor": float(total_noise_db_after_floor),
            "objective_noise_risk_raw": (float(f_best[3]) if len(f_best) > 3 else None),
            "objective_noise_risk_weighted": (
                float(f_best[3]) * float(w_noise) if len(f_best) > 3 else None
            ),
            "w_noise": float(w_noise),
            "noise_floor_db": float(noise_floor_db),
        },
        # Preserve the legacy aliases, but keep their historical meaning as the
        # completed takeoff transition and the start of the landing transition.
        "takeoff_complete": _point_json(takeoff_output_point),
        "landing_entry": _point_json(landing_output_point),
        "balanced_transition_result": {
            "enabled": bool(use_takeoff_landing_transition),
            "transition_structure_mode": str(
                params_dict.get("transition_structure_mode", "unknown")
            ),
            "two_stage_enabled": bool(
                use_takeoff_landing_transition
                and params_dict.get("use_two_stage_transition", False)
            ),
            "feasible": bool(rf_best.get("transition_feasible", True)),
            "fail_reason": str(rf_best.get("transition_fail_reason", "ok")),
            "fail_reasons": [str(v) for v in rf_best.get("transition_fail_reasons", [])],
            "validation_checks": {
                str(k): bool(v)
                for k, v in transition_meta.get("validation_checks", {}).items()
            },
            "takeoff_stage1_end": _point_json(takeoff_stage1_end),
            "takeoff_transition_end": _point_json(takeoff_transition_end),
            "landing_transition_start": _point_json(landing_transition_start),
            "landing_stage1_start": _point_json(landing_stage1_start),
            "takeoff_stage1_distance_m": float(takeoff_cfg.get("stage1_straight_distance_m", 0.0)),
            "landing_stage1_distance_m": float(landing_cfg.get("stage1_straight_distance_m", 0.0)),
            "takeoff_fixed_prefix_requested_distance_m": float(
                takeoff_cfg.get("stage1_requested_straight_distance_m", 0.0)
            ),
            "landing_fixed_prefix_requested_distance_m": float(
                landing_cfg.get("stage1_requested_straight_distance_m", 0.0)
            ),
            "takeoff_transition_distance_2d_m": float(transition_meta.get("takeoff_transition_total_horizontal_distance_m", 0.0)),
            "landing_transition_distance_2d_m": float(transition_meta.get("landing_transition_total_horizontal_distance_m", 0.0)),
            "takeoff_optimized_transition_actual": bool(
                transition_meta.get("takeoff_optimized_transition_actual", False)
            ),
            "landing_optimized_transition_actual": bool(
                transition_meta.get("landing_optimized_transition_actual", False)
            ),
            "fixed_transition_general_output_suppressed": fixed_output_suppressed,
            "evaluation_full_path_distance_2d_m": float(total_corridor_dist_2d_km * 1000.0),
            "evaluation_full_path_distance_3d_m": float(total_corridor_dist_3d_m),
            "evaluation_full_path_point_count": int(full_path.shape[0]),
            "public_corridor_distance_2d_m": float(output_corridor_dist_2d_km * 1000.0),
            "public_corridor_distance_3d_m": float(output_corridor_dist_3d_m),
            "public_corridor_point_count": int(output_path.shape[0]),
            "takeoff_climb_angle_deg": params_dict.get("takeoff_climb_angle_deg", 0.0),
            "landing_descent_angle_deg": params_dict.get("landing_descent_angle_deg", 0.0),
            "takeoff_actual_climb_angle_deg": float(
                params_dict.get("actual_takeoff_angle_deg", 0.0)
            ),
            "landing_actual_descent_angle_deg": float(
                params_dict.get("actual_landing_angle_deg", 0.0)
            ),
            "takeoff_stage2_collapsed_at_cruise": bool(
                transition_meta.get("takeoff_stage2_collapsed_at_cruise", False)
            ),
            "landing_stage2_collapsed_at_cruise": bool(
                transition_meta.get("landing_stage2_collapsed_at_cruise", False)
            ),
            "flight_phase_counts": phase_counts,
            "balanced_path_flight_phase_counts": balanced_phase_counts,
            "transition_3d_validation": transition_3d_validation,
        },
    })
    with open(out_dir / "params.json", "w", encoding="utf-8") as _pf:
        json.dump(params_dict, _pf, indent=2, ensure_ascii=False)

    airspace_rows = [{
        "Type": "Center",
        "Zone_ID": 1,
        "Lat": float(airspace_center_lla[0]),
        "Lon": float(airspace_center_lla[1]),
        "Alt_m": float(airspace_center_lla[2]),
        "Radius_m": float(airspace_radius_m),
        "Radius_km": float(airspace_radius_km),
        "Alt_Min_m": float(airspace_alt_min_m),
        "Alt_Max_m": float(airspace_alt_max_m),
    }]
    for pi, p in enumerate(build_circle_lla(airspace_center_lla, airspace_radius_m, n_pts=180), start=1):
        airspace_rows.append({
            "Type": "Boundary",
            "Zone_ID": 1,
            "Point_No": pi,
            "Lat": float(p[0]),
            "Lon": float(p[1]),
            "Alt_m": float(p[2]),
            "Radius_m": float(airspace_radius_m),
            "Radius_km": float(airspace_radius_km),
            "Alt_Min_m": float(airspace_alt_min_m),
            "Alt_Max_m": float(airspace_alt_max_m),
        })
    df_airspace = pd.DataFrame(airspace_rows)

    nfz_rows = []
    for zi, z in enumerate(forbidden_zones, start=1):
        z = np.asarray(z, dtype=float)
        poly = bbox_to_polygon_lla(z, alt_m=0.0)
        for pi, p in enumerate(poly, start=1):
            nfz_rows.append({
                "Zone_ID": zi,
                "Lon_Min": float(z[0]),
                "Lon_Max": float(z[1]),
                "Lat_Min": float(z[2]),
                "Lat_Max": float(z[3]),
                "Point_No": pi,
                "Lat": float(p[0]),
                "Lon": float(p[1]),
                "Alt_m": float(p[2]),
            })
    if nfz_rows:
        df_nfz = pd.DataFrame(nfz_rows)
    else:
        df_nfz = pd.DataFrame(columns=[
            "Zone_ID", "Lon_Min", "Lon_Max", "Lat_Min", "Lat_Max",
            "Point_No", "Lat", "Lon", "Alt_m"
        ])

    df_excel = df.copy()
    for idx in range(len(df_excel) - 1):
        curr_type = str(df_excel.loc[idx, "Type"]).strip().lower()
        next_type = str(df_excel.loc[idx + 1, "Type"]).strip().lower()
        curr_tf_end = str(df_excel.loc[idx, "TF_End"]).strip().upper()
        next_rf_start = str(df_excel.loc[idx + 1, "RF_Start"]).strip().upper()
        if "tf_point" in curr_type and "rf_arc" in next_type:
            if curr_tf_end == "O" and next_rf_start == "O":
                df_excel.loc[idx + 1, "RF_Start"] = ""

    route_col_order = [
        "Point_No", "Type", "Segment", "Flight_Phase", "Lat", "Lon", "Altitude_MSL_m", "Altitude_AGL_m",
        "Combined_Risk", "Ground_Risk", "Air_Risk", "Noise_Risk_Norm_0to1", "Noise_Lden_dB_Above_Floor",
        "Dist_From_Prev_2D_km", "Dist_From_Prev_3D_km", "Cumulative_Dist_2D_km", "Cumulative_Dist_3D_km",
        "Turn_Radius_m", "Turn_Angle_deg", "LookAhead_Radius_Scale",
        "Ground_Speed_mps", "Ground_Speed_kmh", "Bank_Angle_deg",
        "TF_Start", "TF_End", "RF_Start", "RF_End",
        "Arc_Center_Lat", "Arc_Center_Lon", "Corridor_Tag", "CR_Name",
    ]
    if "CR_Name" not in df_excel.columns:
        df_excel["CR_Name"] = ""

    flag_cols = ["TF_Start", "TF_End", "RF_Start", "RF_End"]
    cr_mask = np.zeros(len(df_excel), dtype=bool)
    non_vertiport_mask = ~df_excel["Type"].astype(str).str.strip().str.lower().eq("vertiport").to_numpy(dtype=bool)
    for c in flag_cols:
        if c in df_excel.columns:
            cr_mask = cr_mask | (df_excel[c].astype(str).str.strip().str.upper().eq("O").to_numpy(dtype=bool))
    if "Flight_Phase" in df_excel.columns:
        stage1_mask = df_excel["Flight_Phase"].astype(str).isin([
            FLIGHT_PHASE_TAKEOFF_STAGE1,
            FLIGHT_PHASE_LANDING_STAGE1,
        ]).to_numpy(dtype=bool)
    else:
        stage1_mask = np.zeros(len(df_excel), dtype=bool)
    transition_boundary_mask = np.zeros(len(df_excel), dtype=bool)
    for boundary in (
        takeoff_stage1_end,
        landing_stage1_start,
        takeoff_transition_end,
        landing_transition_start,
    ):
        if boundary is None:
            continue
        for row_idx in range(len(df_excel)):
            row_point = np.array([
                float(df_excel.loc[row_idx, "Lat"]),
                float(df_excel.loc[row_idx, "Lon"]),
                float(df_excel.loc[row_idx, "Altitude_MSL_m"]),
            ], dtype=float)
            if _seg_dist_3d_m(row_point, boundary) <= 0.5:
                transition_boundary_mask[row_idx] = True
    cr_mask = cr_mask & non_vertiport_mask & ~stage1_mask & ~transition_boundary_mask
    cr_indices = np.flatnonzero(cr_mask)
    for k, ridx in enumerate(cr_indices, start=1):
        df_excel.loc[ridx, "CR_Name"] = f"CR{k:03d}"

    df_excel = df_excel[[c for c in route_col_order if c in df_excel.columns]]
    df_cr = df_excel.loc[cr_mask].copy()
    if not df_cr.empty:
        df_cr.reset_index(drop=True, inplace=True)
        cr_cols = list(df_cr.columns)
        if "Point_No" in cr_cols and "CR_Name" in cr_cols:
            cr_cols.remove("CR_Name")
            point_no_idx = cr_cols.index("Point_No")
            cr_cols.insert(point_no_idx + 1, "CR_Name")
            df_cr = df_cr[cr_cols]

    rf_centers = []
    if "Type" in df_excel.columns and {"Arc_Center_Lat", "Arc_Center_Lon"}.issubset(df_excel.columns):
        rf_rows = df_excel["Type"].astype(str).str.contains("RF_Arc", na=False).to_numpy(dtype=bool)
        ac_lat = pd.to_numeric(df_excel["Arc_Center_Lat"], errors="coerce").to_numpy(dtype=float)
        ac_lon = pd.to_numeric(df_excel["Arc_Center_Lon"], errors="coerce").to_numpy(dtype=float)
        for i in range(len(df_excel)):
            if not rf_rows[i] or not np.isfinite(ac_lat[i]) or not np.isfinite(ac_lon[i]):
                continue
            is_dup = False
            for p in rf_centers:
                if _seg_dist_m(np.array([ac_lat[i], ac_lon[i], altitude_levels[0]], dtype=float), p) <= 0.5:
                    is_dup = True
                    break
            if not is_dup:
                rf_centers.append(np.array([ac_lat[i], ac_lon[i], altitude_levels[0]], dtype=float))
    df_rfc = pd.DataFrame([
        {"RFC_Name": f"RFC{i+1:03d}", "Lat": float(p[0]), "Lon": float(p[1])}
        for i, p in enumerate(rf_centers)
    ])

    takeoff_transition_boundary_phase = (
        FLIGHT_PHASE_TAKEOFF_STAGE2
        if bool(transition_meta.get("takeoff_optimized_transition_actual", False))
        else FLIGHT_PHASE_CRUISE
    )
    input_rows = [
        {
            "Point_Name": "Start_Vertiport",
            "Flight_Phase": FLIGHT_PHASE_VERTIPORT,
            "Lat": float(start_vertiport[0]),
            "Lon": float(start_vertiport[1]),
            "Alt_m": float(start_vertiport[2]),
        },
        {
            "Point_Name": "End_Vertiport",
            "Flight_Phase": FLIGHT_PHASE_VERTIPORT,
            "Lat": float(end_vertiport[0]),
            "Lon": float(end_vertiport[1]),
            "Alt_m": float(end_vertiport[2]),
        },
    ]
    if not bool(use_takeoff_landing_transition):
        input_rows.extend([
            {
                "Point_Name": "Takeoff_Point",
                "Flight_Phase": takeoff_transition_boundary_phase,
                "Lat": float(takeoff_output_point[0]),
                "Lon": float(takeoff_output_point[1]),
                "Alt_m": float(takeoff_output_point[2]),
            },
            {
                "Point_Name": "Landing_Point",
                "Flight_Phase": FLIGHT_PHASE_CRUISE,
                "Lat": float(landing_output_point[0]),
                "Lon": float(landing_output_point[1]),
                "Alt_m": float(landing_output_point[2]),
            },
        ])
    for point_name, point, phase in (
        ("Takeoff_Stage1_End", takeoff_stage1_end, FLIGHT_PHASE_TAKEOFF_STAGE1),
        ("Landing_Stage1_Start", landing_stage1_start, FLIGHT_PHASE_LANDING_STAGE2),
        ("Takeoff_Transition_End", takeoff_transition_end, takeoff_transition_boundary_phase),
        ("Landing_Transition_Start", landing_transition_start, FLIGHT_PHASE_CRUISE),
    ):
        if point is not None:
            input_rows.append({
                "Point_Name": point_name,
                "Flight_Phase": phase,
                "Lat": float(point[0]),
                "Lon": float(point[1]),
                "Alt_m": float(point[2]),
            })
    waypoint_records = list(params_dict.get("backbone_waypoints") or [])
    if not waypoint_records:
        n_wp_default = min(
            int(np.size(corridor_lat_default)),
            int(np.size(corridor_lon_default)),
        )
        waypoint_records = [
            {
                "lat": float(corridor_lat_default[i_wp]),
                "lon": float(corridor_lon_default[i_wp]),
                "alt_m": float(waypoint_alt_fixed_m),
            }
            for i_wp in range(n_wp_default)
        ]
    waypoint_prefix = (
        "WP_Clicked"
        if str(params_dict.get("waypoint_source", "")).strip().lower() == "clicked_map"
        else "WP_Default"
    )
    for i_wp, waypoint in enumerate(waypoint_records):
        input_rows.append({
            "Point_Name": f"{waypoint_prefix}_{i_wp+1:03d}",
            "Flight_Phase": FLIGHT_PHASE_CRUISE,
            "Lat": float(waypoint["lat"]),
            "Lon": float(waypoint["lon"]),
            "Alt_m": float(waypoint.get("alt_m", waypoint_alt_fixed_m)),
        })
    df_input_points = pd.DataFrame(input_rows, columns=[
        "Point_Name", "Flight_Phase", "Lat", "Lon", "Alt_m"
    ])

    xlsx_name = out_dir / "route_data.xlsx"
    with pd.ExcelWriter(str(xlsx_name)) as writer:
        df_excel.to_excel(writer, index=False, sheet_name="Route_Data")
        df_cr.to_excel(writer, index=False, sheet_name="CR_Points")
        df_rfc.to_excel(writer, index=False, sheet_name="RF_Centers")
        df_input_points.to_excel(writer, index=False, sheet_name="Input_Points")
        df_summary.to_excel(writer, index=False, sheet_name="Summary")
        df_airspace.to_excel(writer, index=False, sheet_name="Airspace_Info")
        df_nfz.to_excel(writer, index=False, sheet_name="NFZ_Info")
    print(f"Saved {xlsx_name} (Full optimized corridor: {len(df_excel)} points)")

    excel_fig_name = out_dir / "fig_route_from_excel_map.png"
    excel_fig_title = _title_with_altitude(
        "Balanced Optimal Corridor by Phase (from route_data.xlsx)",
        altitude_levels,
        start_vertiport,
    )
    try:
        save_excel_route_map_figure(
            xlsx_name,
            out_png_path=excel_fig_name,
            figure_title=excel_fig_title,
            use_takeoff_landing_transition=bool(use_takeoff_landing_transition),
        )
    except Exception as e:
        print(f"Excel-based route figure generation failed: {e}")


def _plot_representative_corridor_figures(
    reps,
    apply_rf_corridor_fn,
    output_rf_view_fn,
    eval_corridor_objectives_fn,
    objective_names,
    altitude_levels,
    start_vertiport,
    W_half,
    transition_corridor_cfg,
    out_dir,
    request,
    map_extent,
    airspace_center_lla,
    airspace_radius_m,
    forbidden_zones,
    moc_plot_2d,
    lat_lim,
    lon_lim,
    bb_full,
    waypoints,
    end_vertiport,
    takeoff_complete,
    landing_entry,
    init_rep_objectives,
    f_initial_backbone,
    use_takeoff_landing_transition,
    transition_structure_mode,
    setup_corridor_axes_fn,
    plot_standard_key_markers_fn,
):
    """Render representative corridor figures (Fig4/Fig5/Fig5B)."""

    def _add_rf_legend_handles(ax, color, name, tf_lw, rf_lw, include_transition=False, transition_lw=1.2):
        ax.plot([], [], "-", color=color, linewidth=tf_lw, label=f"{name} (TF)")
        ax.plot([], [], "-", color=color, linewidth=rf_lw, label=f"{name} (RF arc)")
        if include_transition:
            ax.plot([], [], "-", color=TAKEOFF_TRANSITION_COLOR, linewidth=transition_lw, label="Takeoff Transition")
            ax.plot([], [], "-", color=LANDING_TRANSITION_COLOR, linewidth=transition_lw, label="Landing Transition")

    def _add_arc_marker_legend_handles(ax):
        ax.scatter([], [], s=40, c="yellow", marker=">", edgecolors="k", label="Arc Start")
        ax.scatter([], [], s=40, c="yellow", marker="s", edgecolors="k", label="Arc End")
        ax.scatter([], [], s=50, c="white", marker="x", linewidths=1.5, label="Arc Center")

    def _add_cr_rfc_legend_handles(ax):
        ax.scatter([], [], s=36, facecolors="none", edgecolors="red", linewidths=1.1, label="CR Point")
        ax.scatter([], [], s=50, c="white", marker="x", linewidths=1.5, label="RFC (Arc Center)")

    def _collect_cr_points_from_segments(segments, excluded_points=None, tol_m=0.5):
        segments = [
            seg for seg in segments
            if not bool(seg.get("is_fixed_transition_stage1", False))
        ]
        excluded_source = [] if excluded_points is None else excluded_points
        excluded = [
            np.asarray(p, dtype=float).reshape(3)
            for p in excluded_source
            if p is not None
        ]
        cr_pts = []
        for si, seg in enumerate(segments):
            pts = np.asarray(seg["points"], dtype=float)
            if pts.size == 0:
                continue
            if si > 0:
                cr_pts.append(pts[0].copy())
            if si < (len(segments) - 1):
                cr_pts.append(pts[-1].copy())

        unique_pts = []
        labels = []
        for p in cr_pts:
            if any(_seg_dist_3d_m(p, ep) <= float(tol_m) for ep in excluded):
                continue
            is_dup = False
            for up in unique_pts:
                if _seg_dist_3d_m(p, up) <= float(tol_m):
                    is_dup = True
                    break
            if not is_dup:
                unique_pts.append(p.copy())
                labels.append(f"CR{len(unique_pts):03d}")
        return unique_pts, labels

    def _collect_rf_centers_from_segments(segments, tol_m=0.5):
        centers = []
        labels = []
        for seg in segments:
            if seg.get("type") != "RF":
                continue
            ac = np.asarray(seg.get("arc_center", np.array([])), dtype=float).reshape(-1)
            if ac.size < 2:
                continue
            p = np.array([float(ac[0]), float(ac[1]), float(altitude_levels[0])], dtype=float)
            is_dup = False
            for cp in centers:
                if _seg_dist_3d_m(p, cp) <= float(tol_m):
                    is_dup = True
                    break
            if not is_dup:
                centers.append(p.copy())
                labels.append(f"RFC{len(centers):03d}")
        return centers, labels

    fig4 = plt.figure("Figure 4: Optimal Corridor", figsize=(14, 10))
    gx4 = setup_corridor_axes_fn(fig4, "Optimal Corridor", with_moc=True)
    plot_standard_key_markers_fn(gx4, include_waypoints=True, include_backbone=True, zorder=7)

    rep_labels = objective_names + ["Balanced"]
    rep_colors = [plt.cm.tab10(i % 10) for i in range(len(objective_names))] + ["black"]
    for ri, rep in enumerate(reps):
        rf = output_rf_view_fn(apply_rf_corridor_fn(rep))
        rp = rf["path"]
        segs = rf["segments"]
        col = rep_colors[ri] if ri < len(rep_colors) else rep_colors[-1]
        lab = rep_labels[ri] if ri < len(rep_labels) else f"Rep{ri}"

        _plot_corridor_width_by_phase_v1(
            gx4, rf, W_half, transition_corridor_cfg, color=col, alpha=0.08
        )

        _plot_rf_segments_by_phase(
            gx4, rf, cruise_color=col, tf_lw=1.5, rf_lw=2.0,
            transform=ccrs.Geodetic(), zorder=8,
            transition_labels=(ri == 0), draw_rf_markers=True,
        )
        _plot_transition_phase_markers(
            gx4, rf, transform=ccrs.Geodetic(), zorder=12, labels=(ri == 0),
            include_stage1=False, general_output=True,
        )

        _add_rf_legend_handles(gx4, col, lab, tf_lw=1.5, rf_lw=2.5, include_transition=False)

    _add_arc_marker_legend_handles(gx4)

    gx4.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, framealpha=0.9)
    fig4.savefig(out_dir / "fig4_optimal_corridor.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_dir / 'fig4_optimal_corridor.png'}")
    plt.close(fig4)

    if len(reps) > 3:
        risk_configs = [
            {"idx": 1, "name": "Ground Risk", "filename": "fig4_ground_risk_corridor.png", "color": "orange"},
            {"idx": 2, "name": "Air Risk", "filename": "fig4_air_risk_corridor.png", "color": "cyan"},
            {"idx": 3, "name": "Noise Risk", "filename": "fig4_noise_risk_corridor.png", "color": "purple"},
        ]
        for config in risk_configs:
            risk_idx = config["idx"]
            if risk_idx >= len(reps):
                continue

            risk_rep = reps[risk_idx]
            rf_risk_full = apply_rf_corridor_fn(risk_rep)
            risk_full_path = rf_risk_full["path"]
            rf_risk = output_rf_view_fn(rf_risk_full)
            risk_transition_meta = rf_risk_full.get("transition_meta", {})

            fig_risk = plt.figure(f"Figure 4: {config['name']} Corridor", figsize=(14, 10))
            fig_risk.subplots_adjust(left=0.05, right=0.72)
            gx_risk = fig_risk.add_subplot(1, 1, 1, projection=request.crs)
            gx_risk.set_extent(map_extent)
            gx_risk.add_image(request, 13)
            f_risk_opt, _ = eval_corridor_objectives_fn(
                risk_full_path,
                rf_risk_full.get("flight_phases"),
            )
            if init_rep_objectives is not None and risk_idx < len(init_rep_objectives):
                risk_init_val = float(init_rep_objectives[risk_idx][risk_idx])
            else:
                risk_init_val = float(f_initial_backbone[risk_idx])
            risk_opt_val = float(f_risk_opt[risk_idx])
            risk_title = _title_with_altitude(
                (
                    f"{config['name']} Corridor\n"
                    f"Risk init->opt: {risk_init_val:.4f} -> {risk_opt_val:.4f}"
                ),
                altitude_levels,
                start_vertiport,
            )
            gx_risk.set_title(risk_title)
            draw_vertiport_radius_rings(gx_risk, airspace_center_lla, radii_m=(airspace_radius_m,))
            plot_forbidden_zones(gx_risk, forbidden_zones, face_alpha=0.10, edge_alpha=0.80)
            plot_moc_binary_overlay(gx_risk, moc_plot_2d, lat_lim, lon_lim,
                                    label="MOC=1 (Corridor-Prohibited)",
                                    fill_color="magenta", fill_alpha=0.18)

            gx_risk.plot(bb_full[:, 1], bb_full[:, 0], "r--", linewidth=1.5, transform=ccrs.Geodetic(),
                         label="Backbone", zorder=4)
            gx_risk.scatter(waypoints[:, 1], waypoints[:, 0], s=60, c="orange", edgecolors="k",
                            linewidths=0.5, marker="o", transform=ccrs.Geodetic(), label="Waypoints", zorder=6)
            gx_risk.scatter([start_vertiport[1]], [start_vertiport[0]], s=120, c="red", edgecolors="k",
                            marker="s", transform=ccrs.Geodetic(), label="Start Vertiport", zorder=7)
            gx_risk.scatter([end_vertiport[1]], [end_vertiport[0]], s=120, c="crimson", edgecolors="k",
                            marker="D", transform=ccrs.Geodetic(), label="End Vertiport", zorder=7)
            if not use_takeoff_landing_transition:
                gx_risk.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90,
                                c=TAKEOFF_TRANSITION_COLOR, marker="^", transform=ccrs.Geodetic(),
                                label="Takeoff_End", zorder=7)
                gx_risk.scatter([landing_entry[1]], [landing_entry[0]], s=90,
                                c=LANDING_TRANSITION_COLOR, marker="v", transform=ccrs.Geodetic(),
                                label="Landing_End", zorder=7)
            elif transition_structure_mode == TRANSITION_STRUCTURE_FIXED_ONLY:
                gx_risk.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90,
                                c=TAKEOFF_TRANSITION_COLOR, marker="^", transform=ccrs.Geodetic(),
                                label="Takeoff Transition End", zorder=7)
                gx_risk.scatter([landing_entry[1]], [landing_entry[0]], s=90,
                                c=LANDING_TRANSITION_COLOR, marker="v", transform=ccrs.Geodetic(),
                                label="Landing Transition Start", zorder=7)
            elif _seg_dist_m(start_vertiport, takeoff_complete) > 0.5:
                if bool(risk_transition_meta.get("takeoff_optimized_transition_actual", True)):
                    gx_risk.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90, facecolors="none",
                                    edgecolors=TAKEOFF_TRANSITION_COLOR, linewidths=1.4, marker="o",
                                    transform=ccrs.Geodetic(), label="Takeoff Stage1 End", zorder=7)
                else:
                    gx_risk.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90,
                                    c=TAKEOFF_TRANSITION_COLOR, edgecolors="k", linewidths=0.7, marker="^",
                                    transform=ccrs.Geodetic(), label="Takeoff Transition End", zorder=7)
            if (
                use_takeoff_landing_transition
                and transition_structure_mode != TRANSITION_STRUCTURE_FIXED_ONLY
                and _seg_dist_m(end_vertiport, landing_entry) > 0.5
            ):
                if bool(risk_transition_meta.get("landing_optimized_transition_actual", True)):
                    gx_risk.scatter([landing_entry[1]], [landing_entry[0]], s=90, facecolors="none",
                                    edgecolors=LANDING_TRANSITION_COLOR, linewidths=1.4, marker="o",
                                    transform=ccrs.Geodetic(), label="Landing Stage1 Start", zorder=7)
                else:
                    gx_risk.scatter([landing_entry[1]], [landing_entry[0]], s=90,
                                    c=LANDING_TRANSITION_COLOR, edgecolors="k", linewidths=0.7, marker="v",
                                    transform=ccrs.Geodetic(), label="Landing Transition Start", zorder=7)

            _plot_corridor_width_by_phase_v1(
                gx_risk,
                rf_risk,
                W_half,
                transition_corridor_cfg,
                color=config["color"],
                alpha=0.08,
            )

            _plot_rf_segments_by_phase(
                gx_risk, rf_risk, cruise_color=config["color"], tf_lw=1.5, rf_lw=2.0,
                transform=ccrs.Geodetic(), zorder=8,
                transition_labels=True, draw_rf_markers=True,
            )
            _plot_transition_phase_markers(
                gx_risk, rf_risk, transform=ccrs.Geodetic(), zorder=12, labels=True,
                include_stage1=False, general_output=True,
            )

            _add_rf_legend_handles(
                gx_risk, config["color"], config["name"], tf_lw=1.5, rf_lw=2.5,
                include_transition=False, transition_lw=1.2
            )
            _add_arc_marker_legend_handles(gx_risk)

            gx_risk.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, framealpha=0.9)
            fig_risk.savefig(out_dir / config["filename"], dpi=150, bbox_inches="tight")
            print(f"Saved {out_dir / config['filename']}")
            plt.close(fig_risk)

    if reps:
        balanced_rep = reps[-1]
        rf_bal_full = apply_rf_corridor_fn(balanced_rep)
        bal_full_path = rf_bal_full["path"]
        rf_bal = output_rf_view_fn(rf_bal_full)

        fig5 = plt.figure("Figure 5: Balanced Corridor Only", figsize=(14, 10))
        gx5 = setup_corridor_axes_fn(fig5, "Balanced Corridor Only (RF Turn)", with_moc=True)
        plot_standard_key_markers_fn(
            gx5, include_waypoints=True, include_backbone=True,
            takeoff_label="Takeoff_End", landing_label="Landing_End",
            backbone_lw=1.2, zorder=7
        )

        _plot_corridor_width_by_phase_v1(
            gx5, rf_bal, W_half, transition_corridor_cfg, color="black", alpha=0.14
        )
        _plot_rf_segments_by_phase(
            gx5, rf_bal, cruise_color="black", tf_lw=1.8, rf_lw=2.8,
            transform=ccrs.Geodetic(), zorder=8,
            transition_labels=True, draw_rf_markers=True,
        )
        _plot_transition_phase_markers(
            gx5, rf_bal, transform=ccrs.Geodetic(), zorder=12, labels=True,
            include_stage1=False, general_output=True,
        )

        _add_rf_legend_handles(
            gx5, "black", "Balanced", tf_lw=1.8, rf_lw=2.8,
            include_transition=False, transition_lw=1.5
        )
        _add_arc_marker_legend_handles(gx5)
        gx5.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, framealpha=0.9)
        fig5.savefig(out_dir / "fig5_balanced_only.png", dpi=150, bbox_inches="tight")
        print(f"Saved {out_dir / 'fig5_balanced_only.png'}")
        plt.close(fig5)

        fig5b = plt.figure("Figure 5B: Balanced Corridor (Fig4 Style)", figsize=(14, 10))
        gx5b = setup_corridor_axes_fn(fig5b, "Balanced Corridor Only (Fig4 Style)", with_moc=True)
        plot_standard_key_markers_fn(
            gx5b, include_waypoints=True, include_backbone=True,
            takeoff_label="Takeoff_End", landing_label="Landing_End",
            backbone_lw=1.2, zorder=10
        )

        _plot_corridor_width_by_phase_v1(
            gx5b, rf_bal, W_half, transition_corridor_cfg, color="black", alpha=0.14
        )
        _plot_rf_segments_by_phase(
            gx5b, rf_bal, cruise_color="black", tf_lw=1.8, rf_lw=2.8,
            transform=ccrs.Geodetic(), zorder=8,
            transition_labels=True, draw_rf_markers=True,
        )
        _plot_transition_phase_markers(
            gx5b, rf_bal, transform=ccrs.Geodetic(), zorder=12, labels=True,
            include_stage1=False, general_output=True,
        )

        _add_rf_legend_handles(
            gx5b, "black", "Balanced", tf_lw=1.8, rf_lw=2.8,
            include_transition=False, transition_lw=1.5
        )
        _add_arc_marker_legend_handles(gx5b)

        gx5b.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, framealpha=0.9)
        fig5b.savefig(out_dir / "fig5b_balanced_fig4_style.png", dpi=150, bbox_inches="tight")
        print(f"Saved {out_dir / 'fig5b_balanced_fig4_style.png'}")
        plt.close(fig5b)

        fig5c = plt.figure("Figure 5C: Balanced Corridor + CR/RFC Labels", figsize=(14, 10))
        gx5c = setup_corridor_axes_fn(fig5c, "Balanced Corridor with CR/RFC Labels", with_moc=True)
        plot_standard_key_markers_fn(
            gx5c, include_waypoints=True, include_backbone=True,
            takeoff_label="Takeoff_End",
            landing_label="Landing_End",
            backbone_lw=1.2, zorder=10
        )
        _plot_corridor_width_by_phase_v1(
            gx5c, rf_bal, W_half, transition_corridor_cfg, color="black", alpha=0.14
        )
        _plot_rf_segments_by_phase(
            gx5c, rf_bal, cruise_color="black", tf_lw=1.8, rf_lw=2.8,
            transform=ccrs.Geodetic(), zorder=8,
            transition_labels=True, draw_rf_markers=False,
        )
        _plot_transition_phase_markers(
            gx5c, rf_bal, transform=ccrs.Geodetic(), zorder=12, labels=True,
            include_stage1=False, general_output=True,
        )

        transition_meta_bal = rf_bal.get("transition_meta", {})
        cr_pts, cr_labels = _collect_cr_points_from_segments(
            rf_bal["segments"],
            excluded_points=[
                transition_meta_bal.get("takeoff_stage1_end"),
                transition_meta_bal.get("landing_stage1_start"),
                transition_meta_bal.get("takeoff_transition_end"),
                transition_meta_bal.get("landing_transition_end"),
            ],
        )
        for p, name in zip(cr_pts, cr_labels):
            gx5c.scatter(p[1], p[0], s=36, facecolors="none", edgecolors="red", linewidths=1.1,
                         transform=ccrs.Geodetic(), zorder=12)
            gx5c.text(p[1] + 0.00018, p[0] + 0.00012, name, fontsize=4.5, color="red",
                      transform=ccrs.Geodetic(), zorder=13)

        rfc_pts, rfc_labels = _collect_rf_centers_from_segments(rf_bal["segments"])
        for p, name in zip(rfc_pts, rfc_labels):
            gx5c.scatter(p[1], p[0], s=50, c="white", marker="x", linewidths=1.5,
                         transform=ccrs.Geodetic(), zorder=12)
            gx5c.text(p[1] + 0.00018, p[0] - 0.00014, name, fontsize=4.5, color="dodgerblue",
                      transform=ccrs.Geodetic(), zorder=13)

        _add_cr_rfc_legend_handles(gx5c)
        gx5c.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, framealpha=0.9)
        fig5c.savefig(out_dir / "fig5c_balanced_cr_rfc_labels.png", dpi=150, bbox_inches="tight")
        print(f"Saved {out_dir / 'fig5c_balanced_cr_rfc_labels.png'}")
        plt.close(fig5c)


# ======================================================================
# MAIN ENTRY: one complete optimization attempt
# ======================================================================
# Main optimization pipeline: one end-to-end run attempt
def attempt_run_once():
    """Run one complete standalone optimization attempt."""
    # ==================== Core Parameters ====================
    W_half = 296.0                   # 순항 회랑의 중심선 기준 좌·우 반폭(m); 총폭은 2*W_half
    transition_corridor_half_width_m = 100.0  # 전이구간 전용 좌·우 반폭(m)이자 최대 하방 MOC 이격거리(m)
    # RF turn radius model:
    #   V: m/s, g: m/s^2, theta: rad, R: m
    #   R = V^2 / (g * tan(theta))
    # Speed conversion: 300 km/h = 300/3.6 = 83.333... m/s
    speed_max_kmh = 300.0
    ground_speed_mps = speed_max_kmh / 3.6
    bank_angle_deg = 25.0            # max bank angle (deg)
    g_mps2 = 9.80665

    # Base RF turn radius before any look-ahead scaling.
    # With 300 km/h and 25 deg bank angle, this is about 1,518 m.
    rf_base_turn_radius_m = (ground_speed_mps ** 2) / (g_mps2 * np.tan(np.deg2rad(bank_angle_deg)))


    num_arc_points = 30          # number of points used to draw each RF arc

    check_corridor_nfz = True
    check_corridor_moc = True

    # True이면 전체 회랑의 자기겹침 검사를 적용하고, False이면 적용하지 않는다.
    check_corridor_self_overlap = False

    N_init = 1000    # target initial candidates before feasibility filtering
    min_feasible_init_solutions = 1  # minimum feasible candidates required to start evolution
    N_pop = 50      # population size
    Nmax = 10       # number of generations
    offspring_ratio = 0.6   # n_offspring = round(len(parents) * offspring_ratio)
    require_rf_for_parent_selection = True  # require constraint+RF feasibility for parent selection

    # Mutation controls
    mutation_rate = 0.20                  # mutation probability
    use_local_safe_resample = True        # sample replacement nodes from local safe-node pools
    local_resample_prob = 0.70            # probability of using local-safe resampling path
    local_strip_width_m = 500.0           # strip width around parent segment (m)
    local_radius_m = 500.0                # local neighborhood radius for candidate sampling (m)
    local_max_tries = 5                   # max local-resample attempts
    risk_weight_boost = True              # bias sampling toward lower-risk candidates
    risk_weight_strength = 2.0            # stronger value increases low-risk preference

    mutation_cfg = {
        "mutation_rate": float(mutation_rate),
        "use_local_safe_resample": bool(use_local_safe_resample),
        "local_resample_prob": float(local_resample_prob),
        "local_strip_width_m": float(local_strip_width_m),
        "local_radius_m": float(local_radius_m),
        "local_max_tries": int(local_max_tries),
        "risk_weight_boost": bool(risk_weight_boost),
        "risk_weight_strength": float(risk_weight_strength),
    }

    wp_perturb_radius_m = 100.0     # WP 교란 반경 (m)
    wp_perturb_steps = 10           # WP 교란 반복 횟수 (1이면 단일 교란)
    min_extra_nodes_per_seg = 0     # 세그먼트별 최소 extra node 수 (int 또는 list)
    max_extra_nodes_per_seg = 2     # 세그먼트별 최대 extra node 수 (int 또는 list)
    use_wp_skip_generator = False   # True: WP-skip 초기해 생성기 혼용
    init_pop_skip_mix_ratio = 0.5   # skip 생성기 혼용 비율(0~1)
    wp_skip_prob = 0.00             # 중간 WP skip 확률 (0~1)
    airspace_radius_km = 5.0         # 공역 반경 제한(km)
    min_corridor_distance_km = 0.0  # 전체 회랑 최소 거리 제한 (km), 0이면 비활성
    emergency_strip_m = 500.0       # emergency 포함 완화 strip 폭
    min_seg_for_extra_nodes_m = 1500.0  # 짧은 세그먼트 extra node 생성 억제 길이

    # RF look-ahead controls
    look_ahead = True
    look_ahead_threshold_m = 2000.0  # radius scaling starts below this segment-length scale

    # Physical interpretation of look_ahead_min_scale:
    #   R_scaled = scale * R_base
    #   Equivalent speed under same bank angle: V_scaled = V_base * sqrt(scale)
    #       scale=0.11 -> V_scaled ~= 27.66 m/s (99.6 km/h)
    #       scale=0.15 -> V_scaled ~= 37.5 m/s (135 km/h)
    #       scale=0.3 -> V_scaled ~= 45.64 m/s (164.3 km/h)
    #       scale=0.5 -> V_scaled ~= 58.93 m/s (212.1 km/h)
    #       scale=0.8 -> V_scaled ~= 74.54 m/s (268.3 km/h)
    #       scale=1.0 -> V_scaled = V_base = 83.33 m/s (300 km/h)
    look_ahead_min_scale = 0.11   # lower bound for RF radius scaling (0~1)
    look_ahead_window = 3          # number of neighbor segments per side for look-ahead
    rf_use_boundary_heading = False  # True: pass candidate first/last tangents into RF boundary corners
    rf_debug_level = "off"         # "off" | "summary" | "detail", RF-debug print control
    rf_allow_tangent_clamp = True  # allow geometric tangent clamping to fit short segments
    rf_corner_fit_margin = 0.95    # maximum usable fraction of adjacent segment lengths
    rf_corner_min_tangent_m = 1.0  # minimum tangent distance for corner construction
    rf_min_turn_angle_deg = 0.5    # angles below this are treated as straight
    max_init_retries = 300         # max retries for feasible initial-population search
    global RF_ALLOW_TANGENT_CLAMP, RF_CORNER_FIT_MARGIN, RF_CORNER_MIN_TANGENT_M, RF_MIN_TURN_ANGLE_DEG
    global MOC_REFERENCE_MSL_M
    RF_ALLOW_TANGENT_CLAMP = bool(rf_allow_tangent_clamp)
    RF_CORNER_FIT_MARGIN = float(rf_corner_fit_margin)
    RF_CORNER_MIN_TANGENT_M = float(rf_corner_min_tangent_m)
    RF_MIN_TURN_ANGLE_DEG = float(rf_min_turn_angle_deg)

    look_ahead_min_equiv_speed_mps = ground_speed_mps * np.sqrt(look_ahead_min_scale)
    look_ahead_min_equiv_speed_kmh = look_ahead_min_equiv_speed_mps * 3.6
    look_ahead_min_turn_radius_m = rf_base_turn_radius_m * look_ahead_min_scale

    # 아래 네 값은 후보별 원목적을 0~1로 정규화한 뒤 계산하는 Balanced 선호 가중치다.
    w_dist = 0.1    # 실제 3D 거리 목적의 상대 중요도
    w_ground = 0.3  # 지상위험 목적의 상대 중요도
    w_air = 0.5     # 공중위험 목적의 상대 중요도
    w_noise = 0.1   # 소음위험 목적의 상대 중요도
    altitude_levels = np.array([600.0], dtype=float)  # 순항 고도(MSL, m)
    risk_altitude_levels = np.arange(0.0, 1000.0, 100.0, dtype=float)  # 위험자료 MSL 층
    use_heading_map = True

    sector_mode_enabled = True  # True: MOC·계절바람·지상·공중위험 자동선정, False: 아래 사용자 섹터를 그대로 사용
    sector_season = "winter"     # 자동선정에 사용할 바람자료 월 범위: annual/spring/summer/autumn/winter
    takeoff_sector_user = 7      # 수동모드에서 사용할 이륙 섹터(1~12); 자동모드에서도 입력값 자체는 기록용으로 보존
    landing_sector_user = 5      # 수동모드에서 사용할 착륙 섹터(1~12); 자동모드에서도 입력값 자체는 기록용으로 보존
    sector_half_width_deg = 15.0 # 고정 직선이 없을 때 최적화 경로 heading에 허용할 섹터 중심 ±각도(deg)
    sector_analysis_sample_spacing_m = 80.0  # 자동선정 명목 전이회랑의 종방향 MOC·위험 검사 최대 간격(m)
    sector_wind_tail_weight = 0.5   # 자동선정 바람위험 안에서 순풍 위험이 차지하는 비율
    sector_wind_cross_weight = 0.5  # 자동선정 바람위험 안에서 절대 측풍 위험이 차지하는 비율
    sector_wind_risk_weight = 1.0 / 3.0    # 자동선정 최종점수의 바람위험 비율
    sector_ground_risk_weight = 1.0 / 3.0  # 자동선정 최종점수의 지상위험 비율
    sector_air_risk_weight = 1.0 / 3.0     # 자동선정 최종점수의 공중위험 비율
    sector_wind_data_dir = Path("wind_data")  # 월별 AirRisk_Data_1~12.mat 바람자료 폴더

    takeoff_sector_user = _validate_sector_1based_v1(
        takeoff_sector_user, label="takeoff_sector_user"
    )
    landing_sector_user = _validate_sector_1based_v1(
        landing_sector_user, label="landing_sector_user"
    )
    takeoff_sector_selected = int(takeoff_sector_user)
    landing_sector_selected = int(landing_sector_user)
    takeoff_heading_deg = float(np.rad2deg(_sector_angle(takeoff_sector_selected)))
    landing_heading_deg = float(np.rad2deg(_sector_angle(landing_sector_selected)))
    use_takeoff_landing_transition = True  # 전체 전이 기능 스위치; False면 아래 구조·기하 설정을 무시하고 수동 endpoint 사용
    transition_structure_mode = "fixed_straight_only"  # 전이 경로 구조를 아래 3개 값 중 하나로 선택
    # "fixed_straight_only": 순항고도까지 고정 직선, 최적화 전이 없음
    # "fixed_straight_plus_optimized": 고정 직선 prefix + 남은 최적화 전이
    # "optimized_only": 버티포트부터 최적화 전이만 사용
    transition_structure_mode = str(transition_structure_mode).strip().lower()
    if (
        bool(use_takeoff_landing_transition)
        and transition_structure_mode not in TRANSITION_STRUCTURE_MODES
    ):
        raise ValueError(
            "transition_structure_mode must be one of "
            f"{TRANSITION_STRUCTURE_MODES}, got {transition_structure_mode!r}."
        )
    # 호환용 파생 alias이며 사용자가 직접 설정하는 파라미터가 아니다.
    # 세 구조를 모두 표현할 수 없으므로 실제 설정은 transition_structure_mode만 변경한다.
    use_two_stage_transition = bool(
        transition_structure_mode
        == TRANSITION_STRUCTURE_FIXED_PLUS_OPTIMIZED
    )

    if bool(use_takeoff_landing_transition) and (
        not np.isfinite(transition_corridor_half_width_m)
        or float(transition_corridor_half_width_m) <= 0.0
    ):
        raise ValueError(
            "transition_corridor_half_width_m must be finite and > 0."
        )

    sector_season = str(sector_season).strip().lower()
    _sector_wind_months_v1(sector_season)

    takeoff_sector_heading_deg_compass = float((90.0 - takeoff_heading_deg) % 360.0)
    landing_sector_heading_deg_compass = float((90.0 - landing_heading_deg) % 360.0)

    # 고정 직선이 있는 구조에서 전체 버티포트→순항고도 기울기를 정한다.
    # "distance": 밑변을 직접 입력하고 실제 경사각을 자동 계산한다.
    # "angle": 경사각을 직접 입력하고 밑변을 자동 계산한다.
    transition_mode = "angle"  # 고정 직선 포함 구조의 전체 기울기 입력 방식: "angle" 또는 "distance"; optimized_only에서는 무시
    takeoff_total_transition_horizontal_distance_m = None  # distance 모드의 이륙 전체 밑변(m); angle/optimized_only에서는 무시
    landing_total_transition_horizontal_distance_m = None  # distance 모드의 착륙 전체 밑변(m); angle/optimized_only에서는 무시
    takeoff_stage1_straight_distance_m = 300.0  # 혼합 구조에서 최적화 전 이륙 고정 직선 prefix 거리(m); 다른 구조에서는 무시
    landing_stage1_straight_distance_m = 300.0  # 혼합 구조에서 최적화 후 착륙 고정 직선 prefix 거리(m); 다른 구조에서는 무시
    takeoff_climb_angle_deg = 6.0  # angle 모드의 이륙 권위 각도(deg)
    landing_descent_angle_deg = 6.0  # angle 모드의 착륙 권위 각도(deg)
    transition_mode = str(transition_mode).strip().lower()
    fixed_transition_geometry_active = bool(
        use_takeoff_landing_transition
        and transition_structure_mode in (
            TRANSITION_STRUCTURE_FIXED_ONLY,
            TRANSITION_STRUCTURE_FIXED_PLUS_OPTIMIZED,
        )
    )
    fixed_prefix_input_active = bool(
        use_takeoff_landing_transition
        and transition_structure_mode
        == TRANSITION_STRUCTURE_FIXED_PLUS_OPTIMIZED
    )
    angle_input_active = bool(
        use_takeoff_landing_transition
        and (
            transition_structure_mode == TRANSITION_STRUCTURE_OPTIMIZED_ONLY
            or (
                fixed_transition_geometry_active
                and transition_mode == "angle"
            )
        )
    )
    total_distance_input_active = bool(
        fixed_transition_geometry_active and transition_mode == "distance"
    )
    if fixed_transition_geometry_active and transition_mode not in (
        "angle", "distance"
    ):
        raise ValueError(
            "transition_mode must be 'angle' or 'distance' when the transition "
            "structure contains a fixed straight segment."
        )
    if angle_input_active:
        _validate_transition_angle(
            takeoff_climb_angle_deg, "takeoff_climb_angle_deg"
        )
        _validate_transition_angle(
            landing_descent_angle_deg, "landing_descent_angle_deg"
        )
    if total_distance_input_active:
        for value, label in (
            (
                takeoff_total_transition_horizontal_distance_m,
                "takeoff_total_transition_horizontal_distance_m",
            ),
            (
                landing_total_transition_horizontal_distance_m,
                "landing_total_transition_horizontal_distance_m",
            ),
        ):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{label} must be finite and > 0 in distance mode."
                ) from exc
            if not np.isfinite(numeric_value) or numeric_value <= 0.0:
                raise ValueError(f"{label} must be finite and > 0 in distance mode.")
    if fixed_prefix_input_active:
        for value, label in (
            (
                takeoff_stage1_straight_distance_m,
                "takeoff_stage1_straight_distance_m",
            ),
            (
                landing_stage1_straight_distance_m,
                "landing_stage1_straight_distance_m",
            ),
        ):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} must be finite and >= 0.") from exc
            if not np.isfinite(numeric_value) or numeric_value < 0.0:
                raise ValueError(f"{label} must be finite and >= 0.")
    configured_fixed_transition_for_moc_audit = bool(
        use_takeoff_landing_transition
        and (
            transition_structure_mode == TRANSITION_STRUCTURE_FIXED_ONLY
            or (
                transition_structure_mode
                == TRANSITION_STRUCTURE_FIXED_PLUS_OPTIMIZED
                and (
                    float(takeoff_stage1_straight_distance_m) > 0.0
                    or float(landing_stage1_straight_distance_m) > 0.0
                )
            )
        )
    )

    # 경로 형상 계산과 무관하며 엑셀의 이착륙 속도 기록에만 사용한다.
    transition_speed_takeoff_mps = 50.0  # Excel 전이행에 기록할 이륙 속도(m/s); 기하·최적화에는 미사용
    transition_speed_landing_mps = 50.0  # Excel 전이행에 기록할 착륙 속도(m/s); 기하·최적화에는 미사용

    # 전이 경로에 점을 생성하는 수평 간격(m).
    # 예: 밑변 2000 m, 간격 100 m이면 양 끝점을 포함하여 약 21개 점이 생성된다.
    transition_sample_spacing_m = 100.0  # 고정 직선 Stage1 표본 간격(m); MOC의 80m 검사 간격과는 별도

    use_clicked_waypoints = False  # True: 지도 클릭으로 중간 WP 입력
    enforce_mandatory_wp_order = True  # True: takeoff -> 입력 WP 순서 -> landing 강제
    
    min_clicked_waypoints = 0   # 클릭 입력 WP 최소 개수 (takeoff/landing 제외)
    clicked_wp_map_zoom = 13    # 클릭 입력용 지도 초기 줌 레벨
    clicked_wp_base_name = "clicked_waypoints"
    if use_clicked_waypoints and not USE_INTERACTIVE_BACKEND:
        print("Interactive backend is not available. Falling back to default predefined waypoints.")
        use_clicked_waypoints = False

    W_buf = 1250.0  # 회랑 버퍼 폭 (m)
    node_grid_resolution_m = 100.0 # 안전 노드 생성 격자 간격 (m)

    MIN_SAFE_NODES_TARGET = 200
    SAFE_NODE_AIRRISK_MAX_LIST = [0.1, 0.2, 0.3, 0.4, 0.5]
    USE_PERCENTILE_SAFE_NODE_FILTER = True  # True이면 SAFE_NODE_AIRRISK_MAX_LIST는 백분위수 리스트로 해석, False이면 절대 위험도 임계값 리스트로 해석

    cell_size = 100.0
    refine_scales = np.array([1.0, 0.5, 0.2, 0.1])  # RF look-ahead 보간 스케일 단계
    delta_z_max = max(100.0, float(np.max(np.abs(altitude_levels - 150.0))) + 5.0)
    flight_dist_limit = 100000.0 
    objective_names = ["Distance", "Ground Risk", "Air Risk", "Noise Risk"]
    objective_weights, objective_weights_normalized = _validate_objective_weights_v1(
        [w_dist, w_ground, w_air, w_noise], len(objective_names)
    )
    airspace_radius_m = float(airspace_radius_km) * 1000.0
    airspace_alt_min_m = 100.0  # 공역 최소 고도(MSL, m)
    airspace_alt_max_m = 1000.0  # 공역 최대 고도(MSL, m)
    min_corridor_distance_m = float(min_corridor_distance_km) * 1000.0

    noise_npy_path = Path("noise_data", "noise_lden_grid.npy")
    noise_floor_db = 0.0

    #

    ground_risk_path = Path("ground_risk_data", "Modified_high_res_affected_population_GRC.npy")

    bird_airrisk_path = Path("air_risk_data", "bird_riskmap_springfall_3d.npy")
    moc_airrisk_dir = Path("260608_MOC")
    lat_lim = [35.535, 35.652]
    lon_lim = [129.020, 129.150]

    import datetime as _dt
    _run_ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / _run_ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # start_vertiport_default = np.array([35.6033361, 129.0776917, 150.0], dtype=float) # 26년 5월 28일 변경전 버티포트 좌표
    # end_vertiport_default = np.array([35.6033361, 129.0776917, 150.0], dtype=float)   # 26년 5월 28일 변경전 버티포트 좌표
    start_vertiport_default = np.array([35.603386, 129.078025, 150.0], dtype=float)
    end_vertiport_default = np.array([35.603386, 129.078025, 150.0], dtype=float)
    takeoff_end_lla = np.array([35.59468397, 129.07515721, float(altitude_levels[0])], dtype=float) # 이륙 끝 지점, 고도를 순항 고도와 일치하도록 설정
    landing_end_lla = np.array([35.59701567, 129.08585995, float(altitude_levels[0])], dtype=float) #  착륙 끝 지점, 고도를 순항 고도와 일치하도록 설정
    start_ref_alt_m = float(start_vertiport_default[2])
    MOC_REFERENCE_MSL_M = float(start_ref_alt_m)
    transition_corridor_cfg = {
        "enabled": bool(use_takeoff_landing_transition),
        "half_width_m": float(transition_corridor_half_width_m),
        "downward_clearance_m": float(transition_corridor_half_width_m),
        "start_vertiport_msl_m": float(start_vertiport_default[2]),
        "end_vertiport_msl_m": float(end_vertiport_default[2]),
    }
    _validate_transition_corridor_cfg_v1(transition_corridor_cfg)
    transition_3d_corridor_policy = {
        "enabled": bool(use_takeoff_landing_transition),
        "horizontal_half_width_m": float(transition_corridor_half_width_m),
        "horizontal_total_width_m": float(2.0 * transition_corridor_half_width_m),
        "configured_downward_clearance_m": float(transition_corridor_half_width_m),
        "downward_clearance_source": "transition_corridor_half_width_m",
        "effective_clearance_formula": (
            "min(configured_clearance, max(0, center_MSL - direction_vertiport_MSL))"
        ),
        "lower_face_formula": "center_MSL - effective_clearance",
        "transition_phases": [str(value) for value in TRANSITION_PHASE_NAMES],
        "cruise_half_width_m": float(W_half),
        "applies_to": ["MOC", "NFZ", "airspace", "self_overlap"],
        "airspace_vertical_extent": "lower_face_to_centerline",
        "self_overlap_vertical_policy": "legacy_conservative_2d",
        "moc_vertical_safety_assumption": (
            "blocked cells at higher MOC layers are subsets of lower-layer blocked cells"
        ),
        "moc_all_clear_first_clear_policy": (
            "AGL100 is recorded as the first clear available layer; required-safe MSL stays null"
        ),
    }
    transition_3d_validation = {
        "status": (
            "NOT GENERATED" if use_takeoff_landing_transition else "NOT APPLICABLE"
        ),
        "reason": (
            "balanced_path_unavailable"
            if use_takeoff_landing_transition else "transition_disabled"
        ),
        "directions": {},
    }
    if not bool(use_takeoff_landing_transition):
        off_direction_status = {
            "status": "NOT APPLICABLE",
            "moc_status": "NOT APPLICABLE",
            "nfz_status": "NOT APPLICABLE",
            "airspace_status": "NOT APPLICABLE",
            "self_overlap_status": "NOT APPLICABLE",
            "fail_reasons": [],
        }
        transition_3d_validation.update({
            "moc_status": "NOT APPLICABLE",
            "nfz_status": "NOT APPLICABLE",
            "airspace_status": "NOT APPLICABLE",
            "self_overlap_status": "NOT APPLICABLE",
            "directions": {
                "takeoff": dict(off_direction_status),
                "landing": dict(off_direction_status),
            },
        })

    moc_transition_visualization = {
        "enabled": bool(use_takeoff_landing_transition),
        "generated": False,
        "transition_structure_mode": str(transition_structure_mode),
        "audit_only_fixed_transition_geometry": bool(
            use_takeoff_landing_transition
            and transition_structure_mode == TRANSITION_STRUCTURE_FIXED_ONLY
        ),
        "moc_audit_includes_fixed_transition": bool(
            configured_fixed_transition_for_moc_audit
        ),
        "output_policy_notice": (
            "audit-only fixed transition geometry; omitted from general corridor outputs"
            if use_takeoff_landing_transition
            and transition_structure_mode == TRANSITION_STRUCTURE_FIXED_ONLY
            else None
        ),
        "reason": (
            "balanced_path_unavailable"
            if use_takeoff_landing_transition else "transition_disabled"
        ),
        "folder": None,
        "files": [],
        "moc_enforced": bool(check_corridor_moc),
        "status": (
            "NOT GENERATED" if use_takeoff_landing_transition else "NOT APPLICABLE"
        ),
        "sample_count": 0,
        "tested_count": 0,
        "hit_count": 0,
        "out_of_grid_count": 0,
        "corridor_half_width_m": float(transition_corridor_half_width_m),
        "transition_corridor_half_width_m": float(
            transition_corridor_half_width_m
        ),
        "configured_downward_clearance_m": float(transition_corridor_half_width_m),
        "transition_vertical_clearance_m": float(
            transition_corridor_half_width_m
        ),
        "cruise_corridor_half_width_m": float(W_half),
        "transition_3d_status": str(transition_3d_validation["status"]),
        "transition_3d_corridor_policy": transition_3d_corridor_policy,
        "transition_3d_validation": transition_3d_validation,
        "directions": {},
    }
    sector_auto_selection_active = bool(
        sector_mode_enabled and use_takeoff_landing_transition
    )
    sector_selection_analysis = {
        "enabled": bool(sector_auto_selection_active),
        "mode": (
            "automatic" if sector_auto_selection_active else (
                "manual" if use_takeoff_landing_transition else "not_applicable"
            )
        ),
        "status": (
            "PENDING" if sector_auto_selection_active else (
                "MANUAL" if use_takeoff_landing_transition else "NOT_APPLICABLE"
            )
        ),
        "reason": (
            None if use_takeoff_landing_transition else "transition_disabled"
        ),
        "season": str(sector_season),
        "wind_months": (
            [int(value) for value in _sector_wind_months_v1(sector_season)]
            if sector_auto_selection_active else []
        ),
        "configured_user_takeoff_sector": int(takeoff_sector_user),
        "configured_user_landing_sector": int(landing_sector_user),
        "selected_pair": {
            "takeoff_sector": int(takeoff_sector_selected),
            "landing_sector": int(landing_sector_selected),
            "selection_status": (
                "PENDING" if sector_auto_selection_active else (
                    "MANUAL" if use_takeoff_landing_transition else "NOT_APPLICABLE"
                )
            ),
        },
        "diagnostic_figure": None,
    }
    params_dict = {
        "run_timestamp": _run_ts,
        "WP_CLICK_MODE_ENV": _CLICK_MODE_ENV,
        "USE_INTERACTIVE_BACKEND": bool(USE_INTERACTIVE_BACKEND),
        "W_half": W_half,
        "transition_corridor_half_width_m": float(transition_corridor_half_width_m),
        "transition_vertical_clearance_m": float(transition_corridor_half_width_m),
        "transition_3d_corridor_policy": transition_3d_corridor_policy,
        "transition_3d_validation": transition_3d_validation,
        "ground_speed_mps": ground_speed_mps,
        "ground_speed_kmh": speed_max_kmh,
        "bank_angle_deg": bank_angle_deg,
        "gravity_mps2": g_mps2,
        "rf_base_turn_radius_m": rf_base_turn_radius_m,
        "num_arc_points": num_arc_points,
        "N_init": N_init,
        "min_feasible_init_solutions": min_feasible_init_solutions,
        "N_pop": N_pop,
        "Nmax": Nmax,
        "offspring_ratio": offspring_ratio,
        "require_rf_for_parent_selection": bool(require_rf_for_parent_selection),
        "mutation_cfg": mutation_cfg,
        "mutation_rate": float(mutation_rate),
        "use_local_safe_resample": bool(use_local_safe_resample),
        "local_resample_prob": float(local_resample_prob),
        "local_strip_width_m": float(local_strip_width_m),
        "local_radius_m": float(local_radius_m),
        "local_max_tries": int(local_max_tries),
        "risk_weight_boost": bool(risk_weight_boost),
        "risk_weight_strength": float(risk_weight_strength),
        "wp_perturb_radius_m": wp_perturb_radius_m,
        "wp_perturb_steps": wp_perturb_steps,
        "min_extra_nodes_per_seg": min_extra_nodes_per_seg,
        "max_extra_nodes_per_seg": max_extra_nodes_per_seg,
        "use_wp_skip_generator": bool(use_wp_skip_generator),
        "init_pop_skip_mix_ratio": float(init_pop_skip_mix_ratio),
        "emergency_strip_m": emergency_strip_m,
        "min_seg_for_extra_nodes_m": min_seg_for_extra_nodes_m,
        "look_ahead": look_ahead,
        "look_ahead_threshold_m": look_ahead_threshold_m,
        "look_ahead_min_scale": look_ahead_min_scale,
        "look_ahead_min_turn_radius_m": look_ahead_min_turn_radius_m,
        "look_ahead_min_equiv_speed_mps": look_ahead_min_equiv_speed_mps,
        "look_ahead_min_equiv_speed_kmh": look_ahead_min_equiv_speed_kmh,
        "look_ahead_window": look_ahead_window,
        "rf_use_boundary_heading": bool(rf_use_boundary_heading),
        "rf_boundary_heading_policy": "preserve_candidate_tangency_and_validate_actual_sector_legs",
        "rf_debug_level": str(rf_debug_level),
        "rf_allow_tangent_clamp": bool(rf_allow_tangent_clamp),
        "rf_corner_fit_margin": float(rf_corner_fit_margin),
        "rf_corner_min_tangent_m": float(rf_corner_min_tangent_m),
        "rf_min_turn_angle_deg": float(rf_min_turn_angle_deg),
        "max_init_retries": max_init_retries,
        "w_dist": w_dist,
        "w_ground": w_ground,
        "w_air": w_air,
        "w_noise": w_noise,
        "objective_values_are_raw": True,
        "objective_weighting_formula": (
            "sum(normalized_weight_i * minmax_normalized_objective_i)"
        ),
        "objective_weighting_applies_to": [
            "truncated_generation_front", "final_balanced_selection"
        ],
        "objective_weights_normalized": {
            str(objective_names[idx]): float(objective_weights_normalized[idx])
            for idx in range(len(objective_names))
        },
        "sector_mode_enabled": bool(sector_mode_enabled),
        "sector_auto_selection_active": bool(sector_auto_selection_active),
        "sector_season": str(sector_season),
        "takeoff_sector_user": int(takeoff_sector_user),
        "landing_sector_user": int(landing_sector_user),
        "takeoff_sector_selected": int(takeoff_sector_selected),
        "landing_sector_selected": int(landing_sector_selected),
        "sector_half_width_deg": float(sector_half_width_deg),
        "sector_analysis_sample_spacing_m": float(sector_analysis_sample_spacing_m),
        "sector_wind_tail_weight": float(sector_wind_tail_weight),
        "sector_wind_cross_weight": float(sector_wind_cross_weight),
        "sector_wind_risk_weight": float(sector_wind_risk_weight),
        "sector_ground_risk_weight": float(sector_ground_risk_weight),
        "sector_air_risk_weight": float(sector_air_risk_weight),
        "sector_wind_data_dir": str(sector_wind_data_dir),
        "takeoff_heading_deg": float(takeoff_heading_deg),
        "landing_heading_deg": float(landing_heading_deg),
        "takeoff_compass_heading_deg": takeoff_sector_heading_deg_compass,
        "landing_compass_heading_deg": landing_sector_heading_deg_compass,
        "takeoff_sector_heading_deg_compass": takeoff_sector_heading_deg_compass,
        "landing_sector_heading_deg_compass": landing_sector_heading_deg_compass,
        "sector_selection_analysis": sector_selection_analysis,
        "use_takeoff_landing_transition": bool(use_takeoff_landing_transition),
        "transition_structure_mode": str(transition_structure_mode),
        "transition_structure_mode_effective": (
            str(transition_structure_mode)
            if use_takeoff_landing_transition else "off"
        ),
        "use_two_stage_transition": bool(use_two_stage_transition),
        "use_two_stage_transition_deprecated_alias_is_lossy": True,
        "effective_two_stage_transition": bool(
            use_takeoff_landing_transition and use_two_stage_transition
        ),
        "fixed_transition_general_output_suppressed": bool(
            use_takeoff_landing_transition
            and transition_structure_mode == TRANSITION_STRUCTURE_FIXED_ONLY
        ),
        "fixed_transition_evaluated_but_not_exported": bool(
            use_takeoff_landing_transition
            and transition_structure_mode == TRANSITION_STRUCTURE_FIXED_ONLY
        ),
        "moc_audit_includes_fixed_transition": bool(
            configured_fixed_transition_for_moc_audit
        ),
        "takeoff_end_lla": [float(v) for v in takeoff_end_lla.tolist()],
        "landing_end_lla": [float(v) for v in landing_end_lla.tolist()],
        "transition_mode": str(transition_mode),
        "transition_mode_effective": (
            (
                "ignored"
                if transition_structure_mode == TRANSITION_STRUCTURE_OPTIMIZED_ONLY
                else str(transition_mode)
            )
            if use_takeoff_landing_transition else "off"
        ),
        "takeoff_total_transition_horizontal_distance_m": (
            None
            if takeoff_total_transition_horizontal_distance_m is None
            else (
                float(takeoff_total_transition_horizontal_distance_m)
                if total_distance_input_active
                else takeoff_total_transition_horizontal_distance_m
            )
        ),
        "landing_total_transition_horizontal_distance_m": (
            None
            if landing_total_transition_horizontal_distance_m is None
            else (
                float(landing_total_transition_horizontal_distance_m)
                if total_distance_input_active
                else landing_total_transition_horizontal_distance_m
            )
        ),
        "takeoff_stage1_straight_distance_m": (
            float(takeoff_stage1_straight_distance_m)
            if fixed_prefix_input_active else takeoff_stage1_straight_distance_m
        ),
        "landing_stage1_straight_distance_m": (
            float(landing_stage1_straight_distance_m)
            if fixed_prefix_input_active else landing_stage1_straight_distance_m
        ),
        "takeoff_climb_angle_deg": (
            float(takeoff_climb_angle_deg)
            if angle_input_active else takeoff_climb_angle_deg
        ),
        "landing_descent_angle_deg": (
            float(landing_descent_angle_deg)
            if angle_input_active else landing_descent_angle_deg
        ),
        # Backward-compatible aliases; v1 semantics are documented by the keys above.
        "takeoff_distance_m": (
            float(takeoff_stage1_straight_distance_m)
            if fixed_prefix_input_active else takeoff_stage1_straight_distance_m
        ),
        "landing_distance_m": (
            float(landing_stage1_straight_distance_m)
            if fixed_prefix_input_active else landing_stage1_straight_distance_m
        ),
        "takeoff_angle_deg": (
            float(takeoff_climb_angle_deg)
            if angle_input_active else takeoff_climb_angle_deg
        ),
        "landing_angle_deg": (
            float(landing_descent_angle_deg)
            if angle_input_active else landing_descent_angle_deg
        ),
        "transition_altitude_rule": "cumulative_horizontal_ground_track",
        "transition_speed_takeoff_mps": float(transition_speed_takeoff_mps),
        "transition_speed_landing_mps": float(transition_speed_landing_mps),
        "transition_sample_spacing_m": float(transition_sample_spacing_m),
        "risk_altitude_levels_msl_m": [float(v) for v in risk_altitude_levels.tolist()],
        "moc_altitude_policy": {
            "reference_msl_m": float(start_ref_alt_m),
            "available_agl_m": [int(v) for v in MOC_AGL_LEVELS_M.tolist()],
            "selection": "floor_to_available_agl_with_agl100_below_minimum",
        },
        "use_heading_map": bool(use_heading_map),
        "altitude_reference": {
            "cruise_altitude_msl_m": float(altitude_levels[0]),
            "agl_reference_point": {
                "name": "Vertiport ground",
                "elevation_msl_m": float(start_ref_alt_m)
            },
            "cruise_altitude_agl_m": float(altitude_levels[0] - start_ref_alt_m)
        },
        "W_buf": W_buf,
        "node_grid_resolution_m": node_grid_resolution_m,
        "MIN_SAFE_NODES_TARGET": int(MIN_SAFE_NODES_TARGET),
        "SAFE_NODE_AIRRISK_MAX_LIST": [float(v) for v in SAFE_NODE_AIRRISK_MAX_LIST],
        "USE_PERCENTILE_SAFE_NODE_FILTER": bool(USE_PERCENTILE_SAFE_NODE_FILTER),
        "cell_size": cell_size,
        "refine_scales": refine_scales.tolist(),
        "delta_z_max": delta_z_max,
        "flight_dist_limit": flight_dist_limit,
        "check_corridor_nfz": check_corridor_nfz,
        "check_corridor_moc": check_corridor_moc,
        "check_corridor_self_overlap": check_corridor_self_overlap,
        "wp_skip_prob": wp_skip_prob,
        "airspace_radius_km": airspace_radius_km,
        "airspace_alt_min_m": airspace_alt_min_m,
        "airspace_alt_max_m": airspace_alt_max_m,
        "min_corridor_distance_km": min_corridor_distance_km,
        "min_corridor_distance_m": min_corridor_distance_m,
        "use_clicked_waypoints": bool(use_clicked_waypoints),
        "enforce_mandatory_wp_order": bool(enforce_mandatory_wp_order),
        "min_clicked_waypoints": int(min_clicked_waypoints),
        "clicked_wp_map_zoom": int(clicked_wp_map_zoom),
        "clicked_wp_base_name": clicked_wp_base_name,
        "noise_npy_path": str(noise_npy_path),
        "noise_floor_db": noise_floor_db,
        "ground_risk_path": str(ground_risk_path),
        "bird_airrisk_path": str(bird_airrisk_path),
        "moc_transition_visualization": moc_transition_visualization,
    }
    with open(out_dir / "params.json", "w", encoding="utf-8") as _pf:
        json.dump(params_dict, _pf, indent=2, ensure_ascii=False)
    print(f"Output folder : {out_dir}")

    pop_risk_raw = np.load(str(ground_risk_path), allow_pickle=True)
    selected = pop_risk_raw[:, :, 0, 3:]
    Ny, Nx, H_time = selected.shape

    A = len(risk_altitude_levels)
    RT = np.zeros((A, H_time, Ny, Nx), dtype=float)
    for ai in range(A):
        for hi in range(H_time):
            RT[ai, hi] = selected[:, :, hi]
    mn = float(np.min(RT))
    RT -= mn
    mx = float(np.max(RT))
    Norm_RT = RT / mx if mx > 0 else RT

    def _align_to_ny_nx_z(raw_3d, name):
        if raw_3d.shape[0] == Nx and raw_3d.shape[1] == Ny:
            return np.transpose(raw_3d, (1, 0, 2))
        if raw_3d.shape[0] == Ny and raw_3d.shape[1] == Nx:
            return raw_3d
        raise RuntimeError(f"{name} shape {raw_3d.shape} != ({Ny},{Nx},Nz) or ({Nx},{Ny},Nz)")

    bird_raw = np.load(str(bird_airrisk_path), allow_pickle=True).item()
    bird_z_vec = np.asarray(
        bird_raw["altitude_vec"] if "altitude_vec" in bird_raw else bird_raw["z_vec"],
        dtype=float,
    ).ravel()
    bird_3d = _align_to_ny_nx_z(np.asarray(bird_raw["Risk_3d"], dtype=float), "BirdRisk")

    AirRisk = np.zeros((Ny, Nx, len(risk_altitude_levels)), dtype=float)
    for i, alt in enumerate(risk_altitude_levels):  # alt: MSL
        src_idx = int(np.argmin(np.abs(bird_z_vec - float(alt))))  # bird_z_vec: MSL
        AirRisk[:, :, i] = bird_3d[:, :, src_idx]

    moc_altitude_levels_msl = float(start_ref_alt_m) + MOC_AGL_LEVELS_M
    MOCRisk, _moc_plot_all_levels, moc_meta = load_fixed_agl_moc_maps(
        moc_dir=moc_airrisk_dir,
        altitude_levels=moc_altitude_levels_msl,
        vertiport_elevation_msl_m=start_ref_alt_m,
        Ny=Ny,
        Nx=Nx,
        lat_lim=lat_lim,
        lon_lim=lon_lim,
    )
    cruise_moc_idx = _moc_floor_layer_index_v1(float(altitude_levels[0]), MOCRisk.shape[2])
    moc_plot_2d = np.asarray(MOCRisk[:, :, cruise_moc_idx], dtype=float)
    moc_meta["plot_layer_agl_m"] = int(MOC_AGL_LEVELS_M[cruise_moc_idx])
    moc_meta["low_altitude_policy"] = "AGL100 is used below 100m AGL"

    print(
        f"Loaded bird air risk map: shape={bird_3d.shape}, "
        f"source_altitudes={bird_z_vec.tolist()}"
    )
    print(
        f"Loaded MOC binary map: shape={MOCRisk.shape}, "
        f"ones_ratio={float(np.mean(MOCRisk)):.4f}"
    )

    params_dict.update({
        "bird_airrisk_meta": {
            "path": str(bird_airrisk_path),
            "source_altitudes_m": [float(v) for v in bird_z_vec.tolist()],
            "global_min": float(np.min(AirRisk)),
            "global_max": float(np.max(AirRisk)),
        },
        "moc_meta": moc_meta,
    })

    noise_3d_norm, noise_3d_db_after_floor, noise_meta = load_noise_risk_from_npy(
        npy_path=noise_npy_path,
        Ny=Ny,
        Nx=Nx,
        altitude_levels=risk_altitude_levels,
        noise_floor_db=noise_floor_db,
    )
    NoiseRisk = np.asarray(noise_3d_norm, dtype=float)
    NoiseRiskDb = np.asarray(noise_3d_db_after_floor, dtype=float)
    print(
        f"Loaded noise NPY: path={noise_npy_path}, "
        f"raw_shape={tuple(noise_meta['risk3d_shape_raw'])}, aligned_shape={tuple(noise_meta['risk3d_shape_aligned'])}, "
        f"max_after_floor={noise_meta['noise_max_db_after_floor']:.3f} dB, nan_ratio_raw={noise_meta['nan_ratio_raw']:.4f}"
    )
    params_dict.update({
        "noise_meta": noise_meta,
    })

    # main eval extent is fixed by v18 settings; keep NPY extents as diagnostics only.
    _lat_lim_meta = noise_meta.get("lat_lim_meta", None)
    _lon_lim_meta = noise_meta.get("lon_lim_meta", None)
    if isinstance(_lat_lim_meta, (list, tuple)) and isinstance(_lon_lim_meta, (list, tuple)) and len(_lat_lim_meta) == 2 and len(_lon_lim_meta) == 2:
        _lat_gap = float(max(abs(_lat_lim_meta[0] - 35.535), abs(_lat_lim_meta[1] - 35.652)))
        _lon_gap = float(max(abs(_lon_lim_meta[0] - 129.020), abs(_lon_lim_meta[1] - 129.150)))
        if _lat_gap > 1e-3 or _lon_gap > 1e-3:
            print(
                "Warning: noise NPY lat/lon extent differs from v18 evaluation extent. "
                f"npy_lat_lim={_lat_lim_meta}, npy_lon_lim={_lon_lim_meta}, "
                "v18_lat_lim=[35.535, 35.652], v18_lon_lim=[129.020, 129.150]"
            )
    # start_vertiport = np.array([35.6033361, 129.0776917, 150.0], dtype=float)
    # end_vertiport = np.array([35.6249109, 129.0586710, 150.0], dtype=float)
    # start_vertiport = np.array([35.6033361, 129.0776917, 150.0], dtype=float)
    # end_vertiport = np.array([35.5980918, 129.1098345, 150.0], dtype=float)
    start_vertiport = start_vertiport_default.copy()  # [lat, lon, alt_m] in MSL
    end_vertiport = end_vertiport_default.copy()  # [lat, lon, alt_m] in MSL
    if start_vertiport.size != 3 or end_vertiport.size != 3:
        raise ValueError("Both start_vertiport and end_vertiport must be [lat, lon, alt].")
    params_dict["altitude_reference"]["agl_reference_point"]["elevation_msl_m"] = float(start_vertiport[2])
    params_dict["altitude_reference"]["cruise_altitude_agl_m"] = float(altitude_levels[0] - start_vertiport[2])

    # airspace_center_lla = None
    airspace_center_lla = np.array([35.603386, 129.078025, 150.0], dtype=float)

    if airspace_center_lla is None:
        airspace_center_lla = np.array([
            0.5 * (float(start_vertiport[0]) + float(end_vertiport[0])),
            0.5 * (float(start_vertiport[1]) + float(end_vertiport[1])),
            0.5 * (float(start_vertiport[2]) + float(end_vertiport[2])),
        ], dtype=float)
    else:
        airspace_center_lla = np.asarray(airspace_center_lla, dtype=float)
        if airspace_center_lla.size != 3:
            raise ValueError("airspace_center_lla must be [lat, lon, alt].")

    if float(airspace_alt_max_m) <= float(airspace_alt_min_m):
        raise ValueError("airspace_alt_max_m must be greater than airspace_alt_min_m.")

    cruise_alt_min_m = float(np.min(altitude_levels))
    cruise_alt_max_m = float(np.max(altitude_levels))
    if cruise_alt_min_m < float(airspace_alt_min_m) or cruise_alt_max_m > float(airspace_alt_max_m):
        raise ValueError(
            "Cruise altitude is outside configured airspace altitude range. "
            f"cruise_altitude_levels_m={altitude_levels.tolist()}, "
            f"airspace_alt_range_m=[{float(airspace_alt_min_m):.1f}, {float(airspace_alt_max_m):.1f}]. "
            "Map visualization is 2D (horizontal) and does not show altitude violations."
        )
    print(
        f"Airspace check: radius={airspace_radius_km:.1f}km, "
        f"alt_range=[{float(airspace_alt_min_m):.1f}, {float(airspace_alt_max_m):.1f}]m, "
        f"cruise={float(altitude_levels[0]):.1f}m MSL"
    )

    #
    #
    # lat_lim = [35.5446, 35.6427]
    # lon_lim = [129.0514, 129.1436]
    request = cimgt.OSM()

    if sector_auto_selection_active:
        sector_diagnostic_path = out_dir / "sector_selection_diagnostics.png"
        sector_selection_analysis = _automatic_sector_selection_v1(
            start_vertiport=start_vertiport,
            end_vertiport=end_vertiport,
            target_altitude_msl=float(altitude_levels[0]),
            transition_structure_mode=transition_structure_mode,
            transition_mode=transition_mode,
            takeoff_total_distance_m=takeoff_total_transition_horizontal_distance_m,
            landing_total_distance_m=landing_total_transition_horizontal_distance_m,
            takeoff_angle_deg=takeoff_climb_angle_deg,
            landing_angle_deg=landing_descent_angle_deg,
            corridor_half_width_m=transition_corridor_half_width_m,
            sector_half_width_deg=sector_half_width_deg,
            along_track_step_m=sector_analysis_sample_spacing_m,
            sector_season=sector_season,
            wind_tail_weight=sector_wind_tail_weight,
            wind_cross_weight=sector_wind_cross_weight,
            wind_risk_weight=sector_wind_risk_weight,
            ground_risk_weight=sector_ground_risk_weight,
            air_risk_weight=sector_air_risk_weight,
            wind_data_dir=sector_wind_data_dir,
            Norm_RT=Norm_RT,
            AirRisk=AirRisk,
            risk_altitude_levels=risk_altitude_levels,
            MOCRisk=MOCRisk,
            lat_lim=lat_lim,
            lon_lim=lon_lim,
            use_heading_map=use_heading_map,
            transition_corridor_cfg=transition_corridor_cfg,
            output_png_path=sector_diagnostic_path,
            request=request,
        )
        sector_selection_analysis.update({
            "configured_user_takeoff_sector": int(takeoff_sector_user),
            "configured_user_landing_sector": int(landing_sector_user),
        })
        takeoff_sector_selected = int(
            sector_selection_analysis["selected_pair"]["takeoff_sector"]
        )
        landing_sector_selected = int(
            sector_selection_analysis["selected_pair"]["landing_sector"]
        )
        print(
            "Automatic sector selection: "
            f"takeoff=S{takeoff_sector_selected}, landing=S{landing_sector_selected}, "
            f"status={sector_selection_analysis['status']}"
        )

    takeoff_heading_deg = float(np.rad2deg(_sector_angle(takeoff_sector_selected)))
    landing_heading_deg = float(np.rad2deg(_sector_angle(landing_sector_selected)))
    takeoff_sector_heading_deg_compass = float((90.0 - takeoff_heading_deg) % 360.0)
    landing_sector_heading_deg_compass = float((90.0 - landing_heading_deg) % 360.0)
    params_dict.update({
        "takeoff_sector_selected": int(takeoff_sector_selected),
        "landing_sector_selected": int(landing_sector_selected),
        "takeoff_heading_deg": float(takeoff_heading_deg),
        "landing_heading_deg": float(landing_heading_deg),
        "takeoff_compass_heading_deg": float(takeoff_sector_heading_deg_compass),
        "landing_compass_heading_deg": float(landing_sector_heading_deg_compass),
        "takeoff_sector_heading_deg_compass": float(takeoff_sector_heading_deg_compass),
        "landing_sector_heading_deg_compass": float(landing_sector_heading_deg_compass),
        "sector_selection_analysis": sector_selection_analysis,
    })
    with open(out_dir / "params.json", "w", encoding="utf-8") as _pf:
        json.dump(params_dict, _pf, indent=2, ensure_ascii=False)

    # 클릭 입력을 끈 경우 사용할 중간 WP: 같은 인덱스의 위도/경도가 한 점을 이룬다.
    # 예: WP01=(corridor_lat_default[0], corridor_lon_default[0])
    corridor_lat_default = np.array([
        35.5612842, 35.5933937,
    ], dtype=float)
    corridor_lon_default = np.array([
        129.0884254, 129.1296023,
    ], dtype=float)



    waypoint_alt_fixed_m = float(altitude_levels[0])  # 중간 경유 WP 고정 고도(MSL, m)
    clicked_wp_json_path = None
    clicked_wp_csv_path = None

    corridor_lat = corridor_lat_default.copy()
    corridor_lon = corridor_lon_default.copy()

    if use_takeoff_landing_transition:
        preview_takeoff, takeoff_transition_profile, takeoff_transition_meta = build_stage1_transition_profile(
            start_vertiport,
            target_alt_m=waypoint_alt_fixed_m,
            heading_deg=takeoff_heading_deg,
            transition_structure_mode=transition_structure_mode,
            transition_mode=transition_mode,
            straight_distance_m=takeoff_stage1_straight_distance_m,
            total_transition_horizontal_distance_m=(
                takeoff_total_transition_horizontal_distance_m
            ),
            angle_deg=takeoff_climb_angle_deg,
            sample_spacing_m=transition_sample_spacing_m,
            mode_label="takeoff",
        )
        preview_landing, landing_transition_profile, landing_transition_meta = build_stage1_transition_profile(
            end_vertiport,
            target_alt_m=waypoint_alt_fixed_m,
            heading_deg=landing_heading_deg,
            transition_structure_mode=transition_structure_mode,
            transition_mode=transition_mode,
            straight_distance_m=landing_stage1_straight_distance_m,
            total_transition_horizontal_distance_m=(
                landing_total_transition_horizontal_distance_m
            ),
            angle_deg=landing_descent_angle_deg,
            sample_spacing_m=transition_sample_spacing_m,
            mode_label="landing",
        )
        if landing_transition_profile is None or np.size(landing_transition_profile) == 0:
            landing_transition_profile_desc = landing_transition_profile
        else:
            landing_transition_profile_desc = np.asarray(landing_transition_profile, dtype=float)[::-1].copy()
    else:
        preview_takeoff = np.asarray(takeoff_end_lla, dtype=float).reshape(3)
        preview_landing = np.asarray(landing_end_lla, dtype=float).reshape(3)
        takeoff_transition_profile = np.empty((0, 3), dtype=float)
        landing_transition_profile = np.empty((0, 3), dtype=float)
        landing_transition_profile_desc = np.empty((0, 3), dtype=float)
        takeoff_transition_meta = {
            "mode": "off",
            "height_m": 0.0,
            "distance_m": 0.0,
            "angle_deg": 0.0,
            "heading_deg": float("nan"),
            "sample_spacing_m": float(transition_sample_spacing_m),
            "transition_structure_mode": "off",
            "transition_mode": "off",
            "total_horizontal_distance_m": 0.0,
            "stage1_requested_straight_distance_m": 0.0,
            "stage1_straight_distance_m": 0.0,
            "stage2_horizontal_distance_m": 0.0,
            "optimized_transition_actual": False,
            "stage2_collapsed_at_cruise": False,
            "stage1_clamped_to_cruise": False,
        }
        landing_transition_meta = {
            "mode": "off",
            "height_m": 0.0,
            "distance_m": 0.0,
            "angle_deg": 0.0,
            "heading_deg": float("nan"),
            "sample_spacing_m": float(transition_sample_spacing_m),
            "transition_structure_mode": "off",
            "transition_mode": "off",
            "total_horizontal_distance_m": 0.0,
            "stage1_requested_straight_distance_m": 0.0,
            "stage1_straight_distance_m": 0.0,
            "stage2_horizontal_distance_m": 0.0,
            "optimized_transition_actual": False,
            "stage2_collapsed_at_cruise": False,
            "stage1_clamped_to_cruise": False,
        }

    print(
        f"Takeoff optimization boundary: lat={preview_takeoff[0]:.8f}, "
        f"lon={preview_takeoff[1]:.8f}, alt={preview_takeoff[2]:.1f}m"
    )
    print(
        f"Landing optimization boundary: lat={preview_landing[0]:.8f}, "
        f"lon={preview_landing[1]:.8f}, alt={preview_landing[2]:.1f}m"
    )

    global TAKEOFF_TRANSITION_PROFILE, LANDING_TRANSITION_PROFILE_DESC, TRANSITION_CONTEXT
    TAKEOFF_TRANSITION_PROFILE = np.asarray(takeoff_transition_profile, dtype=float).copy() if takeoff_transition_profile is not None else np.empty((0, 3), dtype=float)
    LANDING_TRANSITION_PROFILE_DESC = np.asarray(landing_transition_profile_desc, dtype=float).copy() if landing_transition_profile_desc is not None else np.empty((0, 3), dtype=float)
    if use_takeoff_landing_transition:
        takeoff_transition_meta["stage1_profile"] = np.asarray(takeoff_transition_profile, dtype=float).copy()
        landing_transition_meta["stage1_profile_desc"] = np.asarray(landing_transition_profile_desc, dtype=float).copy()
        TRANSITION_CONTEXT = {
            "enabled": True,
            "two_stage_enabled": bool(use_two_stage_transition),
            "transition_structure_mode": str(transition_structure_mode),
            "cruise_altitude_m": float(waypoint_alt_fixed_m),
            "takeoff": takeoff_transition_meta,
            "landing": landing_transition_meta,
            "takeoff_heading_deg": takeoff_sector_heading_deg_compass,
            "landing_heading_deg": landing_sector_heading_deg_compass,
            "sector_half_width_deg": float(sector_half_width_deg),
            "require_takeoff_sector_heading": bool(
                takeoff_transition_meta.get("optimized_transition_actual", False)
                and takeoff_transition_meta.get(
                    "stage1_straight_distance_m", 0.0
                ) <= 0.0
            ),
            "require_landing_sector_heading": bool(
                landing_transition_meta.get("optimized_transition_actual", False)
                and landing_transition_meta.get(
                    "stage1_straight_distance_m", 0.0
                ) <= 0.0
            ),
            "start_vertiport_lla": np.asarray(start_vertiport, dtype=float).copy(),
            "end_vertiport_lla": np.asarray(end_vertiport, dtype=float).copy(),
        }
        TRANSITION_CONTEXT["require_sector_heading"] = bool(
            TRANSITION_CONTEXT["require_takeoff_sector_heading"]
            and TRANSITION_CONTEXT["require_landing_sector_heading"]
        )
    else:
        TRANSITION_CONTEXT = {
            "enabled": False,
            "start_vertiport_lla": np.asarray(start_vertiport, dtype=float).copy(),
            "end_vertiport_lla": np.asarray(end_vertiport, dtype=float).copy(),
        }

    use_emergency_points = True
    emergency_points_input = np.array([
        [35.6201083, 129.1191806, waypoint_alt_fixed_m],  # MSL
        [35.5678222, 129.1067280, waypoint_alt_fixed_m],  # MSL
        [35.5919889, 129.0751972, waypoint_alt_fixed_m],  # MSL
    ], dtype=float)
    if (not use_emergency_points) or emergency_points_input is None or np.size(emergency_points_input) == 0:
        emergency_points_preview = np.empty((0, 3), dtype=float)
    else:
        emergency_points_preview = np.asarray(emergency_points_input, dtype=float).reshape(-1, 3)
        emergency_points_preview = filter_nodes_in_airspace(
            emergency_points_preview,
            airspace_center_lla[:2],
            airspace_radius_m,
            alt_min_m=airspace_alt_min_m,
            alt_max_m=airspace_alt_max_m,
        )

    use_forbidden_zones = True

    # forbidden_zones_input = np.array([
    #     [129.08, 129.10, 35.59, 35.61],
    #     [129.11, 129.118, 35.62, 35.63],
    #     [129.12, 129.13, 35.59, 35.60],
    # ], dtype=float)

    forbidden_zones_input = np.array([], dtype=float).reshape(0, 4)

    if (not use_forbidden_zones) or forbidden_zones_input is None or np.size(forbidden_zones_input) == 0:
        forbidden_zones = np.array([], dtype=float).reshape(0, 4)
    else:
        forbidden_zones = np.asarray(forbidden_zones_input, dtype=float).reshape(-1, 4)

    if use_clicked_waypoints:
        try:
            print("Waypoint click mode is ON.")
            print("  Left click: add WP, Right click/Delete: undo, Enter: finish")
            clicked_latlon = collect_waypoints_from_clicks(
                vertiport=airspace_center_lla,
                lat_lim=lat_lim,
                lon_lim=lon_lim,
                request=request,
                altitude_levels=altitude_levels,
                map_zoom=clicked_wp_map_zoom,
                start_vertiport=start_vertiport,
                end_vertiport=end_vertiport,
                takeoff_complete=preview_takeoff,
                landing_entry=preview_landing,
                use_takeoff_landing_transition=use_takeoff_landing_transition,
                use_two_stage_transition=use_two_stage_transition,
                transition_structure_mode=transition_structure_mode,
                takeoff_optimized_transition_actual=bool(
                    takeoff_transition_meta.get("optimized_transition_actual", False)
                ),
                landing_optimized_transition_actual=bool(
                    landing_transition_meta.get("optimized_transition_actual", False)
                ),
                takeoff_heading_deg=takeoff_heading_deg,
                landing_heading_deg=landing_heading_deg,
                takeoff_sector_user=takeoff_sector_selected,
                landing_sector_user=landing_sector_selected,
                sector_half_width_deg=sector_half_width_deg,
                emergency_points=emergency_points_preview,
                forbidden_zones=forbidden_zones,
                moc_binary_2d=moc_plot_2d,
                ring_radii_m=(airspace_radius_m,),
            )
            if clicked_latlon.shape[0] >= int(min_clicked_waypoints):
                corridor_lat = clicked_latlon[:, 0].astype(float)
                corridor_lon = clicked_latlon[:, 1].astype(float)
                clicked_wp_json_path, clicked_wp_csv_path = save_clicked_waypoints(
                    clicked_latlon,
                    waypoint_alt_fixed_m,
                    out_dir,
                    base_name=clicked_wp_base_name,
                )
            else:
                print(
                    f"Clicked waypoint count {clicked_latlon.shape[0]} is less than "
                    f"minimum {min_clicked_waypoints}. Using fallback corridor WPs."
                )
        except Exception as e:
            print(f"Click-based waypoint input failed; using fallback corridor WPs. Reason: {e}")

    if corridor_lat.shape[0] != corridor_lon.shape[0]:
        raise ValueError("corridor_lat_default and corridor_lon_default must have same length.")
    if not np.all(np.isfinite(corridor_lat)) or not np.all(np.isfinite(corridor_lon)):
        raise ValueError("Waypoint latitude/longitude values must all be finite.")
    if np.any((corridor_lat < -90.0) | (corridor_lat > 90.0)):
        raise ValueError(
            "Waypoint latitude is outside [-90, 90]. "
            "Check that corridor_lat_default contains only latitudes and "
            "corridor_lon_default contains only longitudes."
        )
    if np.any((corridor_lon < -180.0) | (corridor_lon > 180.0)):
        raise ValueError(
            "Waypoint longitude is outside [-180, 180]. "
            "Check that corridor_lat_default contains only latitudes and "
            "corridor_lon_default contains only longitudes."
        )

    waypoint_alts = np.full(corridor_lat.shape[0], waypoint_alt_fixed_m, dtype=float)  # MSL
    waypoints = np.column_stack([corridor_lat, corridor_lon, waypoint_alts]) if corridor_lat.size > 0 else np.empty((0, 3), dtype=float)

    if waypoints.shape[0] > 0:
        print("Selected waypoints in click/order sequence:")
        for i, wp in enumerate(waypoints, start=1):
            print(f"  WP{i:02d}: lat={wp[0]:.7f}, lon={wp[1]:.7f}, alt={wp[2]:.1f}m")
    else:
        print("No middle waypoints provided. Optimization will run with start/end vertiports only.")

    takeoff_target_alt = float(waypoint_alt_fixed_m)  # 이륙 전이 목표 고도(MSL, m)
    landing_target_alt = float(waypoint_alt_fixed_m)  # 착륙 전이 목표 고도(MSL, m)
    takeoff_complete = np.asarray(preview_takeoff, dtype=float)
    landing_entry = np.asarray(preview_landing, dtype=float)
    has_middle_waypoints = bool(waypoints.shape[0] > 0)

    if (not has_middle_waypoints) and (not use_takeoff_landing_transition):
        takeoff_complete = np.asarray(start_vertiport, dtype=float).reshape(3)
        landing_entry = np.asarray(end_vertiport, dtype=float).reshape(3)
        print(
            "No middle waypoints and takeoff/landing transition is OFF. "
            "Using start/end vertiports directly as the optimization backbone."
        )
    print(
        f"Active corridor endpoints: start=({takeoff_complete[0]:.8f}, "
        f"{takeoff_complete[1]:.8f}, {takeoff_complete[2]:.1f}m), "
        f"end=({landing_entry[0]:.8f}, {landing_entry[1]:.8f}, {landing_entry[2]:.1f}m)"
    )

    backbone = np.vstack([takeoff_complete, waypoints, landing_entry])
    if not is_path_inside_airspace(
        backbone,
        airspace_center_lla[:2],
        airspace_radius_m,
        alt_min_m=airspace_alt_min_m,
        alt_max_m=airspace_alt_max_m,
    ):
        _d_backbone = _dist_to_center_m(backbone[:, :2], airspace_center_lla[:2])
        raise ValueError(
            "Backbone waypoints are outside airspace constraints. "
            f"max_horizontal_dist_m={float(np.max(_d_backbone)):.1f} (radius={float(airspace_radius_m):.1f}), "
            f"backbone_alt_range_m=[{float(np.min(backbone[:, 2])):.1f}, {float(np.max(backbone[:, 2])):.1f}] "
            f"(allowed=[{float(airspace_alt_min_m):.1f}, {float(airspace_alt_max_m):.1f}])."
        )
    is_fixed = np.zeros(backbone.shape[0], dtype=bool)
    is_fixed[:] = True    # takeoff + 모든 입력 WP + landing 고정

    if use_emergency_points and emergency_points_input is not None and np.size(emergency_points_input) > 0:
        emergency_points = np.asarray(emergency_points_input, dtype=float).reshape(-1, 3)
        emergency_points = filter_nodes_in_airspace(
            emergency_points,
            airspace_center_lla[:2],
            airspace_radius_m,
            alt_min_m=airspace_alt_min_m,
            alt_max_m=airspace_alt_max_m,
        )
    else:
        emergency_points = np.empty((0, 3), dtype=float)

    params_dict.update({
        "waypoint_source": ("clicked_map" if clicked_wp_json_path is not None else "manual_or_empty_default"),
        "use_emergency_points": bool(use_emergency_points),
        "use_forbidden_zones": bool(use_forbidden_zones),
        "clicked_waypoints_json": (str(clicked_wp_json_path) if clicked_wp_json_path is not None else None),
        "clicked_waypoints_csv": (str(clicked_wp_csv_path) if clicked_wp_csv_path is not None else None),
        "waypoint_altitude_fixed_m": waypoint_alt_fixed_m,
        "start_vertiport": {
            "lat": float(start_vertiport[0]),
            "lon": float(start_vertiport[1]),
            "alt_m": float(start_vertiport[2]),
        },
        "end_vertiport": {
            "lat": float(end_vertiport[0]),
            "lon": float(end_vertiport[1]),
            "alt_m": float(end_vertiport[2]),
        },
        "vertiport": {
            "lat": float(start_vertiport[0]),
            "lon": float(start_vertiport[1]),
            "alt_m": float(start_vertiport[2]),
        },
        "takeoff_sector": int(takeoff_sector_selected),
        "landing_sector": int(landing_sector_selected),
        "takeoff_sector_selected": int(takeoff_sector_selected),
        "landing_sector_selected": int(landing_sector_selected),
        "sector_selection_analysis": sector_selection_analysis,
        "transition_structure_mode_effective": (
            str(transition_structure_mode)
            if bool(use_takeoff_landing_transition) else "off"
        ),
        "transition_mode_effective": (
            str(takeoff_transition_meta.get("transition_mode", "off"))
            if bool(use_takeoff_landing_transition) else "off"
        ),
        "actual_takeoff_angle_deg": float(takeoff_transition_meta["angle_deg"]),
        "actual_landing_angle_deg": float(landing_transition_meta["angle_deg"]),
        "takeoff_total_transition_horizontal_distance_actual_m": float(
            takeoff_transition_meta.get("total_horizontal_distance_m", 0.0)
        ),
        "landing_total_transition_horizontal_distance_actual_m": float(
            landing_transition_meta.get("total_horizontal_distance_m", 0.0)
        ),
        "takeoff_fixed_prefix_requested_distance_m": float(
            takeoff_transition_meta.get("stage1_requested_straight_distance_m", 0.0)
        ),
        "landing_fixed_prefix_requested_distance_m": float(
            landing_transition_meta.get("stage1_requested_straight_distance_m", 0.0)
        ),
        "takeoff_fixed_prefix_effective_distance_m": float(
            takeoff_transition_meta.get("stage1_straight_distance_m", 0.0)
        ),
        "landing_fixed_prefix_effective_distance_m": float(
            landing_transition_meta.get("stage1_straight_distance_m", 0.0)
        ),
        "takeoff_fixed_straight_actual": bool(
            takeoff_transition_meta.get("fixed_straight_actual", False)
        ),
        "landing_fixed_straight_actual": bool(
            landing_transition_meta.get("fixed_straight_actual", False)
        ),
        "moc_audit_includes_fixed_transition": bool(
            takeoff_transition_meta.get("fixed_straight_actual", False)
            or landing_transition_meta.get("fixed_straight_actual", False)
        ),
        "takeoff_optimized_transition_actual": bool(
            takeoff_transition_meta.get("optimized_transition_actual", False)
        ),
        "landing_optimized_transition_actual": bool(
            landing_transition_meta.get("optimized_transition_actual", False)
        ),
        "takeoff_stage2_collapsed_at_cruise": bool(
            takeoff_transition_meta.get("stage2_collapsed_at_cruise", False)
        ),
        "landing_stage2_collapsed_at_cruise": bool(
            landing_transition_meta.get("stage2_collapsed_at_cruise", False)
        ),
        "takeoff_fixed_prefix_clamped_to_cruise": bool(
            takeoff_transition_meta.get("stage1_clamped_to_cruise", False)
        ),
        "landing_fixed_prefix_clamped_to_cruise": bool(
            landing_transition_meta.get("stage1_clamped_to_cruise", False)
        ),
        "alt_delta_m": float(abs(takeoff_target_alt - start_vertiport[2])),
        "takeoff_complete": {
            "lat": float(takeoff_complete[0]),
            "lon": float(takeoff_complete[1]),
            "alt_m": float(takeoff_complete[2]),
        },
        "landing_entry": {
            "lat": float(landing_entry[0]),
            "lon": float(landing_entry[1]),
            "alt_m": float(landing_entry[2]),
        },
        "waypoint_altitudes_m": waypoint_alts.tolist(),
        "has_middle_waypoints": bool(has_middle_waypoints),
        "corridor_endpoint_source": (
            "transition_profile"
            if bool(use_takeoff_landing_transition)
            else ("manual_takeoff_landing_endpoints" if has_middle_waypoints else "start_end_vertiports")
        ),
        "backbone_waypoints": [
            {"index": i, "lat": float(r[0]), "lon": float(r[1]), "alt_m": float(r[2])}
            for i, r in enumerate(waypoints)
        ],
        "emergency_landing_sites": [
            {"index": i, "lat": float(r[0]), "lon": float(r[1]), "alt_m": float(r[2])}
            for i, r in enumerate(emergency_points)
        ],
        "forbidden_zones_bbox": [
            {"lon_min": float(z[0]), "lon_max": float(z[1]),
             "lat_min": float(z[2]), "lat_max": float(z[3])}
            for z in forbidden_zones
        ],
        "airspace_info": {
            "type": "circle",
            "center_lla": {
                "lat": float(airspace_center_lla[0]),
                "lon": float(airspace_center_lla[1]),
                "alt_m": float(airspace_center_lla[2]),
            },
            "radius_m": float(airspace_radius_m),
            "radius_km": float(airspace_radius_km),
            "alt_min_m": float(airspace_alt_min_m),
            "alt_max_m": float(airspace_alt_max_m),
            "boundary_lla": [
                {"lat": float(p[0]), "lon": float(p[1]), "alt_m": float(p[2])}
                for p in build_circle_lla(airspace_center_lla, airspace_radius_m, n_pts=180)
            ],
        },
        "no_fly_zones": [
            {
                "zone_id": int(zi + 1),
                "bbox": {
                    "lon_min": float(z[0]), "lon_max": float(z[1]),
                    "lat_min": float(z[2]), "lat_max": float(z[3]),
                },
                "polygon_lla": [
                    {"lat": float(p[0]), "lon": float(p[1]), "alt_m": float(p[2])}
                    for p in bbox_to_polygon_lla(z, alt_m=0.0)
                ],
            }
            for zi, z in enumerate(forbidden_zones)
        ],
        "map_boundary": {"lat_lim": lat_lim, "lon_lim": lon_lim},
        "takeoff_transition_meta": {
            "mode": str(takeoff_transition_meta["mode"]),
            "transition_structure_mode": str(
                takeoff_transition_meta.get("transition_structure_mode", "off")
            ),
            "transition_mode": str(
                takeoff_transition_meta.get("transition_mode", "off")
            ),
            "height_m": float(takeoff_transition_meta["height_m"]),
            "distance_m": float(takeoff_transition_meta["distance_m"]),
            "angle_deg": float(takeoff_transition_meta["angle_deg"]),
            "actual_angle_deg": float(takeoff_transition_meta["angle_deg"]),
            "heading_deg": (
                float(takeoff_transition_meta["heading_deg"])
                if np.isfinite(float(takeoff_transition_meta["heading_deg"]))
                else None
            ),
            "sample_spacing_m": float(takeoff_transition_meta["sample_spacing_m"]),
            "sample_count": int(takeoff_transition_profile.shape[0]),
            "total_horizontal_distance_m": float(takeoff_transition_meta.get("total_horizontal_distance_m", 0.0)),
            "configured_total_horizontal_distance_m": takeoff_transition_meta.get("configured_total_horizontal_distance_m"),
            "stage1_requested_straight_distance_m": float(takeoff_transition_meta.get("stage1_requested_straight_distance_m", 0.0)),
            "stage1_straight_distance_m": float(takeoff_transition_meta.get("stage1_straight_distance_m", 0.0)),
            "stage1_effective_straight_distance_m": float(takeoff_transition_meta.get("stage1_straight_distance_m", 0.0)),
            "stage2_horizontal_distance_m": float(takeoff_transition_meta.get("stage2_horizontal_distance_m", 0.0)),
            "optimized_transition_actual": bool(takeoff_transition_meta.get("optimized_transition_actual", False)),
            "stage2_collapsed_at_cruise": bool(takeoff_transition_meta.get("stage2_collapsed_at_cruise", False)),
            "stage1_clamped_to_cruise": bool(takeoff_transition_meta.get("stage1_clamped_to_cruise", False)),
            "stage1_end_lla": [
                float(v) for v in np.asarray(
                    takeoff_transition_meta.get("stage1_end_lla", preview_takeoff), dtype=float
                ).tolist()
            ],
        },
        "landing_transition_meta": {
            "mode": str(landing_transition_meta["mode"]),
            "transition_structure_mode": str(
                landing_transition_meta.get("transition_structure_mode", "off")
            ),
            "transition_mode": str(
                landing_transition_meta.get("transition_mode", "off")
            ),
            "height_m": float(landing_transition_meta["height_m"]),
            "distance_m": float(landing_transition_meta["distance_m"]),
            "angle_deg": float(landing_transition_meta["angle_deg"]),
            "actual_angle_deg": float(landing_transition_meta["angle_deg"]),
            "heading_deg": (
                float(landing_transition_meta["heading_deg"])
                if np.isfinite(float(landing_transition_meta["heading_deg"]))
                else None
            ),
            "sample_spacing_m": float(landing_transition_meta["sample_spacing_m"]),
            "sample_count": int(landing_transition_profile.shape[0]),
            "total_horizontal_distance_m": float(landing_transition_meta.get("total_horizontal_distance_m", 0.0)),
            "configured_total_horizontal_distance_m": landing_transition_meta.get("configured_total_horizontal_distance_m"),
            "stage1_requested_straight_distance_m": float(landing_transition_meta.get("stage1_requested_straight_distance_m", 0.0)),
            "stage1_straight_distance_m": float(landing_transition_meta.get("stage1_straight_distance_m", 0.0)),
            "stage1_effective_straight_distance_m": float(landing_transition_meta.get("stage1_straight_distance_m", 0.0)),
            "stage2_horizontal_distance_m": float(landing_transition_meta.get("stage2_horizontal_distance_m", 0.0)),
            "optimized_transition_actual": bool(landing_transition_meta.get("optimized_transition_actual", False)),
            "stage2_collapsed_at_cruise": bool(landing_transition_meta.get("stage2_collapsed_at_cruise", False)),
            "stage1_clamped_to_cruise": bool(landing_transition_meta.get("stage1_clamped_to_cruise", False)),
            "stage1_start_lla": [
                float(v) for v in np.asarray(
                    landing_transition_meta.get("stage1_end_lla", preview_landing), dtype=float
                ).tolist()
            ],
        },
    })
    moc_transition_visualization["moc_audit_includes_fixed_transition"] = bool(
        takeoff_transition_meta.get("fixed_straight_actual", False)
        or landing_transition_meta.get("fixed_straight_actual", False)
    )
    with open(out_dir / "params.json", "w", encoding="utf-8") as _pf:
        json.dump(params_dict, _pf, indent=2, ensure_ascii=False)
    print("params.json updated with spatial data.")

    def _build_safe_nodes(a, b):
        nodes_seg, all_grid = generate_nodes_3d_segment(
            a, b, W_buf, node_grid_resolution_m, lat_lim, lon_lim, Ny, Nx, forbidden_zones,
            altitude_levels=altitude_levels,
        )
        nodes_seg = filter_nodes_in_airspace(
            nodes_seg,
            airspace_center_lla[:2],
            airspace_radius_m,
            alt_min_m=airspace_alt_min_m,
            alt_max_m=airspace_alt_max_m,
        )
        if nodes_seg.size == 0:
            return np.empty((0, 3)), 0.0
        half_all = int(all_grid.shape[0] // 2)
        target = int(max(MIN_SAFE_NODES_TARGET, half_all))
        target = int(min(target, nodes_seg.shape[0]))

        I_n = np.clip(((nodes_seg[:, 1] - lon_lim[0]) / (lon_lim[1] - lon_lim[0]) * (Nx - 1)).astype(int), 0, Nx - 1)
        J_n = np.clip(((nodes_seg[:, 0] - lat_lim[0]) / (lat_lim[1] - lat_lim[0]) * (Ny - 1)).astype(int), 0, Ny - 1)
        ai = np.argmin(np.abs(nodes_seg[:, 2][:, None] - risk_altitude_levels[None, :]), axis=1)
        risks = AirRisk[J_n, I_n, ai]

        safe = nodes_seg
        thr = 0.0
        for thr_max in SAFE_NODE_AIRRISK_MAX_LIST:
            thr = float(thr_max)
            safe = nodes_seg[risks <= thr]
            if safe.shape[0] >= target:
                break
        if safe.shape[0] < target:
            order = np.argsort(risks)
            pick = order[:target]
            safe = nodes_seg[pick]
            thr = float(risks[pick[-1]]) if pick.size > 0 else float(np.max(risks))
        return safe, thr

    def _build_safe_nodes_percentile(a, b):
        """Percentile-threshold safe-node builder."""
        nodes_seg, all_grid = generate_nodes_3d_segment(
            a, b, W_buf, node_grid_resolution_m, lat_lim, lon_lim, Ny, Nx, forbidden_zones,
            altitude_levels=altitude_levels,
        )
        nodes_seg = filter_nodes_in_airspace(
            nodes_seg,
            airspace_center_lla[:2],
            airspace_radius_m,
            alt_min_m=airspace_alt_min_m,
            alt_max_m=airspace_alt_max_m,
        )
        if nodes_seg.size == 0:
            return np.empty((0, 3)), 0.0
        half_all = int(all_grid.shape[0] // 2)
        target = int(max(MIN_SAFE_NODES_TARGET, half_all))
        target = int(min(target, nodes_seg.shape[0]))

        I_n = np.clip(((nodes_seg[:, 1] - lon_lim[0]) / (lon_lim[1] - lon_lim[0]) * (Nx - 1)).astype(int), 0, Nx - 1)
        J_n = np.clip(((nodes_seg[:, 0] - lat_lim[0]) / (lat_lim[1] - lat_lim[0]) * (Ny - 1)).astype(int), 0, Ny - 1)
        ai = np.argmin(np.abs(nodes_seg[:, 2][:, None] - risk_altitude_levels[None, :]), axis=1)
        risks = AirRisk[J_n, I_n, ai]

        safe = nodes_seg
        thr = 0.0
        for v in SAFE_NODE_AIRRISK_MAX_LIST:
            p = float(v * 100.0) if float(v) <= 1.0 else float(v)
            p = float(np.clip(p, 0.0, 100.0))
            thr = float(np.percentile(risks, p))
            safe = nodes_seg[risks <= thr]
            if safe.shape[0] >= target:
                break
        if safe.shape[0] < target:
            order = np.argsort(risks)
            pick = order[:target]
            safe = nodes_seg[pick]
            thr = float(risks[pick[-1]]) if pick.size > 0 else float(np.max(risks))
        return safe, thr

    safe_nodes_by_seg = []
    safe_airrisk_by_seg = []
    thr_list = []
    seg_count = backbone.shape[0] - 1
    for k in range(seg_count):
        seg_m = _seg_dist_m(backbone[k], backbone[k + 1])
        if seg_m < min_seg_for_extra_nodes_m:
            safe_nodes_by_seg.append(np.empty((0, 3)))
            safe_airrisk_by_seg.append(np.empty((0,), dtype=float))
            thr_list.append(0.0)
            continue
        s, t = _build_safe_nodes(backbone[k], backbone[k + 1])
        end_buffer_ratio = _segment_strip_end_buffer_ratio(k, seg_count)
        s = filter_nodes_in_strip(backbone[k], backbone[k + 1], s, 10 * W_half, end_buffer_ratio=end_buffer_ratio)
        safe_nodes_by_seg.append(s)

        if s.size > 0:
            I_s = np.clip(((s[:, 1] - lon_lim[0]) / (lon_lim[1] - lon_lim[0]) * (Nx - 1)).astype(int), 0, Nx - 1)
            J_s = np.clip(((s[:, 0] - lat_lim[0]) / (lat_lim[1] - lat_lim[0]) * (Ny - 1)).astype(int), 0, Ny - 1)
            ai_s = np.argmin(np.abs(s[:, 2][:, None] - risk_altitude_levels[None, :]), axis=1)
            safe_airrisk_by_seg.append(AirRisk[J_s, I_s, ai_s].astype(float))
        else:
            safe_airrisk_by_seg.append(np.empty((0,), dtype=float))
        thr_list.append(t)

    safe_nodes_by_seg_pct = []
    safe_airrisk_by_seg_pct = []
    thr_list_pct = []
    for k in range(seg_count):
        seg_m = _seg_dist_m(backbone[k], backbone[k + 1])
        if seg_m < min_seg_for_extra_nodes_m:
            safe_nodes_by_seg_pct.append(np.empty((0, 3)))
            safe_airrisk_by_seg_pct.append(np.empty((0,), dtype=float))
            thr_list_pct.append(0.0)
            continue
        s_pct, t_pct = _build_safe_nodes_percentile(backbone[k], backbone[k + 1])
        end_buffer_ratio = _segment_strip_end_buffer_ratio(k, seg_count)
        s_pct = filter_nodes_in_strip(backbone[k], backbone[k + 1], s_pct, 10 * W_half, end_buffer_ratio=end_buffer_ratio)
        safe_nodes_by_seg_pct.append(s_pct)
        if s_pct.size > 0:
            I_s = np.clip(((s_pct[:, 1] - lon_lim[0]) / (lon_lim[1] - lon_lim[0]) * (Nx - 1)).astype(int), 0, Nx - 1)
            J_s = np.clip(((s_pct[:, 0] - lat_lim[0]) / (lat_lim[1] - lat_lim[0]) * (Ny - 1)).astype(int), 0, Ny - 1)
            ai_s = np.argmin(np.abs(s_pct[:, 2][:, None] - risk_altitude_levels[None, :]), axis=1)
            safe_airrisk_by_seg_pct.append(AirRisk[J_s, I_s, ai_s].astype(float))
        else:
            safe_airrisk_by_seg_pct.append(np.empty((0,), dtype=float))
        thr_list_pct.append(t_pct)

    if USE_PERCENTILE_SAFE_NODE_FILTER:
        safe_nodes_active = safe_nodes_by_seg_pct
        safe_airrisk_active = safe_airrisk_by_seg_pct
        thr_list_active = thr_list_pct
        safe_nodes_compare = safe_nodes_by_seg
        safe_airrisk_compare = safe_airrisk_by_seg
        active_mode_name = "Percentile"
        compare_mode_name = "Absolute"
    else:
        safe_nodes_active = safe_nodes_by_seg
        safe_airrisk_active = safe_airrisk_by_seg
        thr_list_active = thr_list
        safe_nodes_compare = safe_nodes_by_seg_pct
        safe_airrisk_compare = safe_airrisk_by_seg_pct
        active_mode_name = "Absolute"
        compare_mode_name = "Percentile"

    air_thr_global = float(np.max(thr_list_active)) if thr_list_active else 1.0
    nodes_pool = np.vstack([s for s in safe_nodes_active if s.size > 0]) \
                 if any(s.size > 0 for s in safe_nodes_active) else (
                     emergency_points if emergency_points.size > 0 else backbone.copy()
                 )
    node_risk_pool = np.concatenate([r for r in safe_airrisk_active if r.size > 0]) \
                     if any(r.size > 0 for r in safe_airrisk_active) else np.empty((0,), dtype=float)
    if node_risk_pool.shape[0] != nodes_pool.shape[0]:
        node_risk_pool = np.empty((0,), dtype=float)

    print(
        f"Searching for RF+constraint feasible initial pop "
        f"(max {max_init_retries} retries, N_init={N_init}, "
        f"min_feasible_init_solutions={min_feasible_init_solutions}) ..."
    )
    rf_corridor_start = np.asarray(takeoff_complete, dtype=float)
    rf_corridor_end = np.asarray(landing_entry, dtype=float)
    _apply_rf_for_init = partial(
        _apply_rf_corridor_path,
        start_vertiport=rf_corridor_start,
        end_vertiport=rf_corridor_end,
        ground_speed_mps=ground_speed_mps,
        bank_angle_deg=bank_angle_deg,
        num_arc_points=num_arc_points,
        look_ahead=look_ahead,
        look_ahead_threshold_m=look_ahead_threshold_m,
        look_ahead_min_scale=look_ahead_min_scale,
        look_ahead_window=look_ahead_window,
        use_boundary_heading=rf_use_boundary_heading,
        rf_debug_level=rf_debug_level,
    )
    _eval_with_reason_for_init = partial(
        evaluate_objectives_with_constraints_gp,
        Norm_RT=Norm_RT,
        AirRisk=AirRisk,
        use_heading_map=use_heading_map,
        flight_dist_limit=flight_dist_limit,
        forbidden_zones=forbidden_zones,
        delta_z_max=delta_z_max,
        altitude_levels=risk_altitude_levels,
        cell_size=cell_size,
        refine_scales=refine_scales,
        air_risk_threshold=air_thr_global,
        w_dist=w_dist,
        w_ground=w_ground,
        w_air=w_air,
        lat_lim=lat_lim,
        lon_lim=lon_lim,
        NoiseRisk=NoiseRisk,
        noise_floor_db=noise_floor_db,
        w_noise=w_noise,
        W_half=W_half,
        check_corridor_nfz=check_corridor_nfz,
        MOCRisk=MOCRisk,
        check_corridor_moc=check_corridor_moc,
        check_corridor_self_overlap=check_corridor_self_overlap,
        vertiport=None,
        landing_entry=None,
        takeoff_complete=None,
        return_reason=True,
        transition_corridor_cfg=transition_corridor_cfg,
    )

    init_pop = None
    _last_init_candidate_count = 0
    _last_init_rf_count = 0
    _last_init_feasible_count = 0
    _last_init_reason_counts = {}
    for _retry in range(1, max_init_retries + 1):
        _candidate = _make_initial_population(
            N_init=N_init,
            use_wp_skip_generator=use_wp_skip_generator,
            init_pop_skip_mix_ratio=init_pop_skip_mix_ratio,
            backbone=backbone,
            waypoints=waypoints,
            takeoff_complete=takeoff_complete,
            landing_entry=landing_entry,
            wp_perturb_radius_m=wp_perturb_radius_m,
            min_extra_nodes_per_seg=min_extra_nodes_per_seg,
            max_extra_nodes_per_seg=max_extra_nodes_per_seg,
            safe_nodes_by_seg=safe_nodes_by_seg,
            emergency_points=emergency_points,
            emergency_strip_m=emergency_strip_m,
            is_fixed=is_fixed,
            wp_perturb_steps=wp_perturb_steps,
            wp_skip_prob=wp_skip_prob,
            min_seg_for_extra_nodes_m=min_seg_for_extra_nodes_m,
            airspace_center_lla=airspace_center_lla,
            airspace_radius_m=airspace_radius_m,
            airspace_alt_min_m=airspace_alt_min_m,
            airspace_alt_max_m=airspace_alt_max_m,
            enforce_mandatory_wp_order=enforce_mandatory_wp_order,
        )
        if len(_candidate) < N_init:
            print(f"  [Init] Airspace-filtered initial pop: {len(_candidate)}/{N_init}")
        _cand_n = len(_candidate)
        _last_init_candidate_count = int(_cand_n)
        if not _candidate:
            if _retry % 50 == 0:
                print(f"  [Init retry {_retry}/{max_init_retries}] candidate_after_initial_airspace: 0/{N_init}")
            continue

        _init_eval = _evaluate_initial_candidates(
            candidate_pop=_candidate,
            backbone=backbone,
            enforce_mandatory_wp_order=enforce_mandatory_wp_order,
            apply_rf_fn=_apply_rf_for_init,
            eval_constraints_with_reason_fn=_eval_with_reason_for_init,
            airspace_center_lla=airspace_center_lla,
            airspace_radius_m=airspace_radius_m,
            airspace_alt_min_m=airspace_alt_min_m,
            airspace_alt_max_m=airspace_alt_max_m,
            min_corridor_distance_m=min_corridor_distance_m,
            cruise_half_width_m=W_half,
            transition_corridor_cfg=transition_corridor_cfg,
        )
        _rf_cnt = int(_init_eval["rf_cnt"])
        _rf_no_clamp_cnt = int(_init_eval["rf_no_clamp_cnt"])
        _both_cnt = int(_init_eval["both_cnt"])
        _cst_cnt = int(_init_eval["cst_cnt"])
        _air_cnt = int(_init_eval["air_cnt"])
        _dist_cnt = int(_init_eval["dist_cnt"])
        _reason_counts = dict(_init_eval["reason_counts"])
        _feasible_init = list(_init_eval["feasible_init"])
        _last_init_rf_count = int(_rf_cnt)
        _last_init_feasible_count = int(_both_cnt)
        _last_init_reason_counts = dict(_reason_counts)

        if _retry % 50 == 0 or _rf_cnt > 0:
            _reason_txt = "none"
            if _reason_counts:
                _parts = [f"{k}:{v}" for k, v in sorted(_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))]
                _reason_txt = ", ".join(_parts)
            print(
                f"  [Init retry {_retry}/{max_init_retries}] "
                f"candidate_after_initial_airspace: {_cand_n}/{N_init} | "
                f"rf_feasible: {_rf_cnt}/{_cand_n} | "
                f"rf_no_clamp: {_rf_no_clamp_cnt}/{_cand_n} | "
                f"constraint_ok(given RF): {_cst_cnt}/{_rf_cnt} | "
                f"airspace_ok(given RF): {_air_cnt}/{_rf_cnt} | "
                f"min_dist_ok(given RF): {_dist_cnt}/{_rf_cnt} | "
                f"both_feasible: {_both_cnt}/{_cand_n} | "
                f"constraint_fail_breakdown: {_reason_txt} | "
                f"target: {min_feasible_init_solutions}"
            )

        if _both_cnt >= min_feasible_init_solutions:
            init_pop = _feasible_init
            print(
                f"  -> {_both_cnt} RF+constraint feasible solution(s) found at retry {_retry} "
                f"(target {min_feasible_init_solutions}). Proceeding with feasible-only init pop."
            )
            break

    if init_pop is None:
        print(
            f"  Warning: feasible init pop < target ({min_feasible_init_solutions}) "
            f"after {max_init_retries} retries. Restarting run."
        )
        return False, 0
    print(f"  -> {len(init_pop)} feasible initial solutions ready.")

    extent_points = [
        start_vertiport[:2],
        end_vertiport[:2],
        airspace_center_lla[:2],
        takeoff_complete[:2],
        landing_entry[:2],
    ]
    if backbone is not None and backbone.size > 0:
        extent_points.extend(backbone[:, :2].tolist())
    if waypoints is not None and waypoints.size > 0:
        extent_points.extend(waypoints[:, :2].tolist())
    if emergency_points is not None and emergency_points.size > 0:
        extent_points.extend(emergency_points[:, :2].tolist())
    for s in safe_nodes_active:
        if s is not None and s.size > 0:
            extent_points.extend(s[:, :2].tolist())
    if init_pop is not None:
        for p in init_pop:
            if p is not None and p.size > 0:
                extent_points.extend(p[:, :2].tolist())

    map_extent = compute_centered_map_extent(np.array(extent_points, dtype=float), airspace_center_lla,
                                             ring_radii_m=(airspace_radius_m,), pad_ratio=0.10)

    fixed_transition_general_output_suppressed = bool(
        use_takeoff_landing_transition
        and transition_structure_mode == TRANSITION_STRUCTURE_FIXED_ONLY
    )
    bb_full = (
        np.vstack([start_vertiport, backbone, end_vertiport])
        if use_takeoff_landing_transition
        and not fixed_transition_general_output_suppressed
        else np.asarray(backbone, dtype=float)
    )
    total_safe_count = sum(s.shape[0] for s in safe_nodes_active if s.size > 0)
    total_safe_count_compare = sum(s.shape[0] for s in safe_nodes_compare if s.size > 0)

    def _plot_selected_sector_wedges(gx):
        wedge_radius_m = float(np.clip(airspace_radius_m * 0.12, 250.0, 700.0))
        to_lon, to_lat = _build_sector_wedge_lonlat(
            start_vertiport,
            takeoff_heading_deg,
            sector_half_width_deg,
            wedge_radius_m,
        )
        ld_lon, ld_lat = _build_sector_wedge_lonlat(
            end_vertiport,
            landing_heading_deg,
            sector_half_width_deg,
            wedge_radius_m,
        )
        gx.fill(
            to_lon, to_lat,
            color="royalblue", alpha=0.24,
            edgecolor="navy", linewidth=0.8,
            transform=ccrs.PlateCarree(), zorder=6,
            label="Takeoff Sector",
        )
        gx.fill(
            ld_lon, ld_lat,
            color="seagreen", alpha=0.24,
            edgecolor="darkgreen", linewidth=0.8,
            transform=ccrs.PlateCarree(), zorder=6,
            label="Landing Sector",
        )

    def _setup_corridor_axes(fig, title, with_moc=False, moc_label="MOC=1 (Corridor-Prohibited)", moc_alpha=0.18):
        fig.subplots_adjust(left=0.05, right=0.72)
        gx = fig.add_subplot(1, 1, 1, projection=request.crs)
        gx.set_extent(map_extent)
        gx.add_image(request, 13)
        gx.set_title(_title_with_altitude(title, altitude_levels, start_vertiport))
        draw_vertiport_radius_rings(gx, airspace_center_lla, radii_m=(airspace_radius_m,))
        plot_forbidden_zones(gx, forbidden_zones, face_alpha=0.10, edge_alpha=0.80)
        if with_moc:
            plot_moc_binary_overlay(
                gx, moc_plot_2d, lat_lim, lon_lim,
                label=moc_label, fill_color="magenta", fill_alpha=moc_alpha
            )
        _plot_selected_sector_wedges(gx)
        return gx

    def _plot_standard_key_markers(gx, include_waypoints=False, include_backbone=True,
                                   takeoff_label="Takeoff_End", landing_label="Landing_End",
                                   backbone_lw=1.5, point_size=120, zorder=7):
        if include_backbone:
            gx.plot(bb_full[:, 1], bb_full[:, 0], "r--", linewidth=backbone_lw, transform=ccrs.Geodetic(),
                    label="Backbone", zorder=4)
        if include_waypoints:
            gx.scatter(waypoints[:, 1], waypoints[:, 0], s=60, c="orange", edgecolors="k",
                       linewidths=0.5, marker="o", transform=ccrs.Geodetic(), label="Waypoints", zorder=6)
        gx.scatter([start_vertiport[1]], [start_vertiport[0]], s=point_size, c="red", edgecolors="k",
                   marker="s", transform=ccrs.Geodetic(), label="Start Vertiport", zorder=zorder)
        gx.scatter([end_vertiport[1]], [end_vertiport[0]], s=point_size, c="crimson", edgecolors="k",
                   marker="D", transform=ccrs.Geodetic(), label="End Vertiport", zorder=zorder)
        if not use_takeoff_landing_transition:
            gx.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90, c=TAKEOFF_TRANSITION_COLOR,
                       marker="^", transform=ccrs.Geodetic(), label=takeoff_label, zorder=zorder)
            gx.scatter([landing_entry[1]], [landing_entry[0]], s=90, c=LANDING_TRANSITION_COLOR,
                       marker="v", transform=ccrs.Geodetic(), label=landing_label, zorder=zorder)
        elif transition_structure_mode == TRANSITION_STRUCTURE_FIXED_ONLY:
            gx.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90,
                       c=TAKEOFF_TRANSITION_COLOR, marker="^", transform=ccrs.Geodetic(),
                       label="Takeoff Transition End", zorder=zorder)
            gx.scatter([landing_entry[1]], [landing_entry[0]], s=90,
                       c=LANDING_TRANSITION_COLOR, marker="v", transform=ccrs.Geodetic(),
                       label="Landing Transition Start", zorder=zorder)
        elif use_two_stage_transition:
            if _seg_dist_m(start_vertiport, takeoff_complete) > 0.5:
                if bool(takeoff_transition_meta.get("optimized_transition_actual", False)):
                    gx.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90, facecolors="none",
                               edgecolors=TAKEOFF_TRANSITION_COLOR, linewidths=1.4, marker="o",
                               transform=ccrs.Geodetic(), label="Takeoff Stage1 End", zorder=zorder)
                else:
                    gx.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90,
                               c=TAKEOFF_TRANSITION_COLOR, edgecolors="k", linewidths=0.7, marker="^",
                               transform=ccrs.Geodetic(), label="Takeoff Transition End", zorder=zorder)
            if _seg_dist_m(end_vertiport, landing_entry) > 0.5:
                if bool(landing_transition_meta.get("optimized_transition_actual", False)):
                    gx.scatter([landing_entry[1]], [landing_entry[0]], s=90, facecolors="none",
                               edgecolors=LANDING_TRANSITION_COLOR, linewidths=1.4, marker="o",
                               transform=ccrs.Geodetic(), label="Landing Stage1 Start", zorder=zorder)
                else:
                    gx.scatter([landing_entry[1]], [landing_entry[0]], s=90,
                               c=LANDING_TRANSITION_COLOR, edgecolors="k", linewidths=0.7, marker="v",
                               transform=ccrs.Geodetic(), label="Landing Transition Start", zorder=zorder)

    def _plot_safe_nodes_figure(
        fig_title,
        title_text,
        safe_nodes_set,
        safe_risk_set,
        filter_label,
        out_name,
    ):
        fig = plt.figure(fig_title, figsize=(14, 10))
        fig.subplots_adjust(left=0.08, right=0.78)
        gx = fig.add_subplot(1, 1, 1, projection=request.crs)
        gx.set_extent(map_extent)
        gx.add_image(request, 13)
        gx.set_title(_title_with_altitude(title_text, altitude_levels, start_vertiport))
        draw_vertiport_radius_rings(gx, airspace_center_lla, radii_m=(airspace_radius_m,))
        plot_forbidden_zones(gx, forbidden_zones, face_alpha=0.10, edge_alpha=0.80)
        plot_moc_binary_overlay(
            gx, moc_plot_2d, lat_lim, lon_lim,
            label="MOC=1 (Corridor-Prohibited)",
            fill_color="magenta", fill_alpha=0.18,
        )
        _plot_selected_sector_wedges(gx)

        gx.plot(bb_full[:, 1], bb_full[:, 0], "r--", linewidth=2, transform=ccrs.Geodetic(),
                label="Backbone", zorder=5)
        gx.scatter(waypoints[:, 1], waypoints[:, 0], s=60, c="orange", edgecolors="k",
                   linewidths=0.5, marker="o", transform=ccrs.Geodetic(), label="Waypoints (WP)", zorder=6)
        gx.scatter([start_vertiport[1]], [start_vertiport[0]], s=120, c="red", edgecolors="k",
                   marker="s", transform=ccrs.Geodetic(), label="Start Vertiport", zorder=7)
        gx.scatter([end_vertiport[1]], [end_vertiport[0]], s=120, c="crimson", edgecolors="k",
                   marker="D", transform=ccrs.Geodetic(), label="End Vertiport", zorder=7)
        if not use_takeoff_landing_transition:
            gx.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90, c=TAKEOFF_TRANSITION_COLOR,
                       marker="^", transform=ccrs.Geodetic(), label="Takeoff_End", zorder=7)
            gx.scatter([landing_entry[1]], [landing_entry[0]], s=90, c=LANDING_TRANSITION_COLOR,
                       marker="v", transform=ccrs.Geodetic(), label="Landing_End", zorder=7)
        elif transition_structure_mode == TRANSITION_STRUCTURE_FIXED_ONLY:
            gx.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90,
                       c=TAKEOFF_TRANSITION_COLOR, marker="^", transform=ccrs.Geodetic(),
                       label="Takeoff Transition End", zorder=7)
            gx.scatter([landing_entry[1]], [landing_entry[0]], s=90,
                       c=LANDING_TRANSITION_COLOR, marker="v", transform=ccrs.Geodetic(),
                       label="Landing Transition Start", zorder=7)
        elif use_two_stage_transition:
            if _seg_dist_m(start_vertiport, takeoff_complete) > 0.5:
                if bool(takeoff_transition_meta.get("optimized_transition_actual", False)):
                    gx.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90, facecolors="none",
                               edgecolors=TAKEOFF_TRANSITION_COLOR, linewidths=1.4, marker="o",
                               transform=ccrs.Geodetic(), label="Takeoff Stage1 End", zorder=7)
                else:
                    gx.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90,
                               c=TAKEOFF_TRANSITION_COLOR, edgecolors="k", linewidths=0.7, marker="^",
                               transform=ccrs.Geodetic(), label="Takeoff Transition End", zorder=7)
            if _seg_dist_m(end_vertiport, landing_entry) > 0.5:
                if bool(landing_transition_meta.get("optimized_transition_actual", False)):
                    gx.scatter([landing_entry[1]], [landing_entry[0]], s=90, facecolors="none",
                               edgecolors=LANDING_TRANSITION_COLOR, linewidths=1.4, marker="o",
                               transform=ccrs.Geodetic(), label="Landing Stage1 Start", zorder=7)
                else:
                    gx.scatter([landing_entry[1]], [landing_entry[0]], s=90,
                               c=LANDING_TRANSITION_COLOR, edgecolors="k", linewidths=0.7, marker="v",
                               transform=ccrs.Geodetic(), label="Landing Transition Start", zorder=7)

        first_scatter = None
        for ki, seg_nodes in enumerate(safe_nodes_set):
            if seg_nodes.size > 0:
                risks = safe_risk_set[ki] if ki < len(safe_risk_set) else np.zeros(seg_nodes.shape[0], dtype=float)
                lab = f"Seg {ki+1} nodes ({seg_nodes.shape[0]})" if ki == 0 else None
                sc = gx.scatter(seg_nodes[:, 1], seg_nodes[:, 0], s=7, c=risks, cmap="jet",
                                vmin=0.0, vmax=1.0, alpha=0.78,
                                transform=ccrs.Geodetic(), label=lab, zorder=3)
                if first_scatter is None:
                    first_scatter = sc
        if len(safe_nodes_set) > 3:
            gx.scatter([], [], s=4, c="gray", alpha=0.9, label=f"... +{len(safe_nodes_set)-3} more segs")
        if first_scatter is not None:
            cax = fig.add_axes([0.035, 0.16, 0.022, 0.70])
            cbar = fig.colorbar(first_scatter, cax=cax)
            cbar.set_label("Air Risk (absolute 0-1)")
        if emergency_points.size > 0:
            gx.scatter(emergency_points[:, 1], emergency_points[:, 0], s=80, c="lime",
                       edgecolors="k", marker="P", transform=ccrs.Geodetic(),
                       label="Emergency Landing", zorder=7)

        gx.plot([], [], linestyle="none", label=filter_label)
        gx.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, framealpha=0.9)
        fig.savefig(out_dir / out_name, dpi=150, bbox_inches="tight")
        print(f"Saved {out_dir / out_name}")
        plt.close(fig)

    # ==================== [Fig 1] Candidate Safe Nodes ====================
    _plot_safe_nodes_figure(
        fig_title="Figure 1: Candidate Safe Nodes",
        title_text=f"Candidate Safe Nodes [{active_mode_name}]  (total {total_safe_count} nodes,  grid {node_grid_resolution_m}m,  W_buf {W_buf}m)",
        safe_nodes_set=safe_nodes_active,
        safe_risk_set=safe_airrisk_active,
        filter_label=f"Filter: {active_mode_name}",
        out_name="fig1_safe_nodes.png",
    )

    # ==================== [Fig 1P] Safe-Node Diagnostic Comparison ====================
    _plot_safe_nodes_figure(
        fig_title="Figure 1P: Candidate Safe Nodes (Comparison Diagnostic)",
        title_text=f"Candidate Safe Nodes [{compare_mode_name} Diagnostic]  (total {total_safe_count_compare} nodes,  grid {node_grid_resolution_m}m,  W_buf {W_buf}m)",
        safe_nodes_set=safe_nodes_compare,
        safe_risk_set=safe_airrisk_compare,
        filter_label=f"Filter: {compare_mode_name} (diagnostic only)",
        out_name="fig1p_safe_nodes_compare_diag.png",
    )

    # ==================== [Fig 1B] MOC Binary Obstacle Risk ====================
    fig1b = plt.figure("Figure 1B: MOC Binary Obstacle Risk", figsize=(14, 10))
    gx1b = _setup_corridor_axes(
        fig1b, "MOC Binary Risk Map (1 = Corridor-Prohibited)",
        with_moc=True, moc_label="MOC=1 (Obstacle Risk)", moc_alpha=0.24
    )
    _plot_standard_key_markers(gx1b, include_waypoints=False, include_backbone=True, zorder=7)
    gx1b.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, framealpha=0.9)
    fig1b.savefig(out_dir / "fig1b_moc_binary.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_dir / 'fig1b_moc_binary.png'}")
    plt.close(fig1b)

    # ==================== [Fig 1C] Waypoints vs MOC ====================
    # Visual check of waypoint placement against MOC=1 prohibited cells.
    fig1c = plt.figure("Figure 1C: Waypoints vs MOC", figsize=(14, 10))
    gx1c = _setup_corridor_axes(
        fig1c, "Waypoint Safety Check on MOC Map",
        with_moc=True, moc_label="MOC=1 (Corridor-Prohibited)", moc_alpha=0.24
    )
    _plot_standard_key_markers(gx1c, include_waypoints=True, include_backbone=True, point_size=130, zorder=8)
    if emergency_points.size > 0:
        gx1c.scatter(emergency_points[:, 1], emergency_points[:, 0], s=80, c="lime",
                     edgecolors="k", marker="P", transform=ccrs.Geodetic(),
                     label="Emergency Landing", zorder=8)

    gx1c.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, framealpha=0.9)
    fig1c.savefig(out_dir / "fig1c_waypoint_moc_safety.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_dir / 'fig1c_waypoint_moc_safety.png'}")
    plt.close(fig1c)

    rf_apply_fn = partial(
        _apply_rf_corridor_path,
        start_vertiport=rf_corridor_start,
        end_vertiport=rf_corridor_end,
        ground_speed_mps=ground_speed_mps,
        bank_angle_deg=bank_angle_deg,
        num_arc_points=num_arc_points,
        look_ahead=look_ahead,
        look_ahead_threshold_m=look_ahead_threshold_m,
        look_ahead_min_scale=look_ahead_min_scale,
        look_ahead_window=look_ahead_window,
        use_boundary_heading=rf_use_boundary_heading,
        rf_debug_level=rf_debug_level,
    )
    output_rf_view_fn = partial(
        _build_output_rf_view_v1,
        transition_structure_mode=transition_structure_mode,
        transition_enabled=use_takeoff_landing_transition,
    )
    eval_cfg = dict(
        Norm_RT=Norm_RT,
        AirRisk=AirRisk,
        use_heading_map=use_heading_map,
        flight_dist_limit=flight_dist_limit,
        forbidden_zones=forbidden_zones,
        delta_z_max=delta_z_max,
        altitude_levels=risk_altitude_levels,
        cell_size=cell_size,
        refine_scales=refine_scales,
        air_risk_threshold=air_thr_global,
        w_dist=w_dist,
        w_ground=w_ground,
        w_air=w_air,
        lat_lim=lat_lim,
        lon_lim=lon_lim,
        NoiseRisk=NoiseRisk,
        noise_floor_db=noise_floor_db,
        w_noise=w_noise,
        W_half=W_half,
        check_corridor_nfz=check_corridor_nfz,
        MOCRisk=MOCRisk,
        check_corridor_moc=check_corridor_moc,
        check_corridor_self_overlap=check_corridor_self_overlap,
        vertiport=None,
        landing_entry=None,
        takeoff_complete=None,
        transition_corridor_cfg=transition_corridor_cfg,
    )
    eval_corridor_fn = partial(_evaluate_corridor_objectives_path, eval_cfg=eval_cfg)

    # ==================== [Fig 2] Sample Initial Solutions + RF Turn ====================

    n_sample = min(5, len(init_pop))
    fig2 = plt.figure("Figure 2: Sample Initial Solutions + RF Turn", figsize=(14, 10))
    gx2 = _setup_corridor_axes(
        fig2, f"Sample Initial Solutions ({n_sample}) with RF Turn", with_moc=True
    )
    _plot_standard_key_markers(gx2, include_waypoints=False, include_backbone=True, zorder=8)
    colors_sample = plt.cm.tab10(np.linspace(0, 1, n_sample))
    for si in range(n_sample):
        rf = output_rf_view_fn(rf_apply_fn(init_pop[si]))
        _plot_rf_segments_by_phase(
            gx2, rf, cruise_color=colors_sample[si], tf_lw=1.2, rf_lw=1.6,
            transform=ccrs.Geodetic(), zorder=5,
            transition_labels=(si == 0), draw_rf_markers=False,
        )
        _plot_transition_phase_markers(
            gx2, rf, transform=ccrs.Geodetic(), zorder=10, labels=(si == 0),
            include_stage1=False, general_output=True,
        )
        gx2.plot([], [], "-", color=colors_sample[si], linewidth=1.2, label=f"Sol {si+1}")
    gx2.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, framealpha=0.9)
    fig2.savefig(out_dir / "fig2_sample_init.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_dir / 'fig2_sample_init.png'}")
    plt.close(fig2)

    # ==================== [Fig 2B] Before vs After RF on Same Initial Paths ====================
    fig2b = plt.figure("Figure 2B: Initial Solutions Before vs After RF", figsize=(14, 10))
    gx2b = _setup_corridor_axes(
        fig2b,
        f"Same Initial Solutions: Before RF (dashed) vs After RF (solid), n={n_sample}",
        with_moc=True,
    )
    _plot_standard_key_markers(gx2b, include_waypoints=False, include_backbone=True, zorder=8)

    for si in range(n_sample):
        col = colors_sample[si]
        path_before = np.asarray(init_pop[si], dtype=float).reshape(-1, 3)
        rf_before = _profile_rf_segments_for_transition(
            {
                "path": path_before,
                "segments": [{"type": "TF", "points": path_before}],
                "feasible": True,
            },
            TRANSITION_CONTEXT if TRANSITION_CONTEXT is not None else {"enabled": False},
        )
        rf_before = output_rf_view_fn(rf_before)
        rf = output_rf_view_fn(rf_apply_fn(init_pop[si]))

        _plot_rf_segments_by_phase(
            gx2b, rf_before, cruise_color=col, tf_lw=1.1, rf_lw=1.1,
            transform=ccrs.Geodetic(), zorder=5,
            transition_labels=(si == 0), draw_rf_markers=False,
            linestyle="--", include_fixed_stage1=False,
        )

        _plot_rf_segments_by_phase(
            gx2b, rf, cruise_color=col, tf_lw=1.7, rf_lw=2.1,
            transform=ccrs.Geodetic(), zorder=7,
            transition_labels=False, draw_rf_markers=False,
        )
        _plot_transition_phase_markers(
            gx2b, rf, transform=ccrs.Geodetic(), zorder=10, labels=(si == 0),
            include_stage1=False, general_output=True,
        )

    gx2b.plot([], [], "--", color="black", linewidth=1.1, label="Before RF (all samples)")
    gx2b.plot([], [], "-", color="black", linewidth=1.7, label="After RF (all samples)")
    gx2b.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, framealpha=0.9)
    fig2b.savefig(out_dir / "fig2b_init_before_after_rf.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_dir / 'fig2b_init_before_after_rf.png'}")
    plt.close(fig2b)

    print("Running NSGA-III ...")
    pop, fvals, gen_history = run_nsga3(
        nodes_pool=nodes_pool,
        node_risk_pool=node_risk_pool,
        population=init_pop,
        N_pop=N_pop, Nmax=Nmax, ratio=offspring_ratio,
        mutation_cfg=mutation_cfg,
        require_rf_for_parent_selection=require_rf_for_parent_selection,
        mandatory_backbone=backbone,
        Norm_RT=Norm_RT, AirRisk=AirRisk, use_map=use_heading_map,
        f_limit=flight_dist_limit, f_zones=forbidden_zones,
        alt=risk_altitude_levels, cs=cell_size, scales=refine_scales,
        air_thr=air_thr_global, dz=delta_z_max,
        w_d=w_dist, w_g=w_ground, w_a=w_air,
        lat_lim=lat_lim, lon_lim=lon_lim,
        NoiseRisk=NoiseRisk, noise_floor_db=noise_floor_db, w_n=w_noise,
        objective_weights=objective_weights,
        ground_speed_mps=ground_speed_mps,
        bank_angle_deg=bank_angle_deg,
        num_arc_points=num_arc_points,
        look_ahead=look_ahead,
        look_ahead_threshold_m=look_ahead_threshold_m,
        look_ahead_min_scale=look_ahead_min_scale,
        look_ahead_window=look_ahead_window,
        use_boundary_heading=rf_use_boundary_heading,
        W_half=W_half, check_corridor_nfz=check_corridor_nfz, check_corridor_moc=check_corridor_moc,
        check_corridor_self_overlap=check_corridor_self_overlap,
        MOCRisk=MOCRisk,
        start_vertiport=rf_corridor_start,
        end_vertiport=rf_corridor_end,
        landing_entry=landing_entry,
        takeoff_complete=takeoff_complete,
        airspace_center_latlon=airspace_center_lla[:2],
        airspace_radius_m=airspace_radius_m,
        airspace_alt_min_m=airspace_alt_min_m,
        airspace_alt_max_m=airspace_alt_max_m,
        min_corridor_distance_m=min_corridor_distance_m,
        transition_corridor_cfg=transition_corridor_cfg,
    )

    _save_generation_snapshots(
        gen_history=gen_history,
        out_dir=out_dir,
        request=request,
        map_extent=map_extent,
        airspace_center_lla=airspace_center_lla,
        airspace_radius_m=airspace_radius_m,
        forbidden_zones=forbidden_zones,
        bb_full=bb_full,
        waypoints=waypoints,
        start_vertiport=start_vertiport,
        end_vertiport=end_vertiport,
        takeoff_complete=takeoff_complete,
        landing_entry=landing_entry,
        use_takeoff_landing_transition=use_takeoff_landing_transition,
        use_two_stage_transition=use_two_stage_transition,
        transition_structure_mode=transition_structure_mode,
        takeoff_optimized_transition_actual=bool(
            takeoff_transition_meta.get("optimized_transition_actual", False)
        ),
        landing_optimized_transition_actual=bool(
            landing_transition_meta.get("optimized_transition_actual", False)
        ),
        objective_names=objective_names,
        objective_weights=objective_weights,
        altitude_levels=altitude_levels,
        apply_rf_corridor_fn=rf_apply_fn,
        output_rf_view_fn=output_rf_view_fn,
        W_half=W_half,
        transition_corridor_cfg=transition_corridor_cfg,
        moc_plot_2d=moc_plot_2d,
        lat_lim=lat_lim,
        lon_lim=lon_lim,
    )

    feasible_count, rf_no_clamp_count, feas_mask = _compute_final_feasibility(
        pop=pop,
        apply_rf_corridor_fn=rf_apply_fn,
        eval_corridor_objectives_fn=eval_corridor_fn,
        airspace_center_lla=airspace_center_lla,
        airspace_radius_m=airspace_radius_m,
        airspace_alt_min_m=airspace_alt_min_m,
        airspace_alt_max_m=airspace_alt_max_m,
        min_corridor_distance_m=min_corridor_distance_m,
        cruise_half_width_m=W_half,
        transition_corridor_cfg=transition_corridor_cfg,
    )
    print(f"Final feasible (constraints): {feasible_count}/{len(pop)}")
    print(f"RF geometric no-clamp: {rf_no_clamp_count}/{len(pop)}")

    if feasible_count == 0:
        print("No feasible solution. Retrying ...")
        return False, 0

    representative_indices = (
        _representative_indices_v1(
            fvals, objective_weights, feasible=feas_mask
        )
        if pop and fvals.size > 0 else []
    )
    reps = [pop[index] for index in representative_indices]
    balanced_population_index = (
        int(representative_indices[-1]) if representative_indices else None
    )
    objective_weighting_result = _balanced_objective_audit_v1(
        fvals,
        objective_weights,
        objective_names,
        feas_mask,
        balanced_population_index,
    )
    params_dict["objective_weighting_result"] = objective_weighting_result

    obj_pairs = [(i, j) for i in range(len(objective_names)) for j in range(i + 1, len(objective_names))]
    n_pair = len(obj_pairs)
    n_cols = 3
    n_rows = int(np.ceil(n_pair / n_cols))
    fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    fig3.suptitle(
        _title_with_altitude(
            "Pareto Front  (blue=feasible, gray=infeasible)",
            altitude_levels,
            start_vertiport,
        ),
        fontsize=13,
    )
    axes_flat = np.atleast_1d(axes3).ravel()

    for ax_i, (oi, oj) in enumerate(obj_pairs):
        ax = axes_flat[ax_i]
        for k in range(len(feas_mask)):
            col = "royalblue" if feas_mask[k] else "lightgray"
            z = 5 if feas_mask[k] else 1
            ax.scatter(fvals[k, oi], fvals[k, oj], c=col, s=18, alpha=0.7, zorder=z,
                       edgecolors="k", linewidths=0.3)
        ax.set_xlabel(objective_names[oi], fontsize=10)
        ax.set_ylabel(objective_names[oj], fontsize=10)
        ax.set_title(f"{objective_names[oi]} vs {objective_names[oj]}", fontsize=10)
        ax.grid(True, alpha=0.3)
        if reps:
            for ri, rep in enumerate(reps):
                rf_rep = rf_apply_fn(rep)
                f_rep, _ = eval_corridor_fn(
                    rf_rep["path"], rf_rep.get("flight_phases")
                )
                rep_labels_3 = objective_names + ["Balanced"]
                lab = rep_labels_3[ri] if ri < len(rep_labels_3) else f"Rep{ri}"
                is_balanced = (ri == len(reps) - 1)
                rep_marker = "*" if is_balanced else "D"
                rep_color = "red" if is_balanced else plt.cm.tab10(ri % 10)
                ax.scatter(f_rep[oi], f_rep[oj], color=rep_color,
                           s=90 if is_balanced else 80, marker=rep_marker,
                           edgecolors="k", linewidths=1, zorder=10, label=lab if ax_i == 0 else None)

    for j in range(n_pair, len(axes_flat)):
        axes_flat[j].axis("off")

    if reps:
        axes_flat[0].legend(loc="upper right", fontsize=7)
    fig3.tight_layout(rect=[0, 0, 1, 0.93])
    fig3.savefig(out_dir / "fig3_pareto.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_dir / 'fig3_pareto.png'}")
    plt.close(fig3)

    rf_initial_backbone = rf_apply_fn(backbone)
    f_initial_backbone, _ = eval_corridor_fn(
        rf_initial_backbone["path"],
        rf_initial_backbone.get("flight_phases"),
    )

    init_rep_objectives = None
    if init_pop:
        init_fvals = np.zeros((len(init_pop), len(objective_names)), dtype=float)
        for i_init, p_init in enumerate(init_pop):
            rf_init = rf_apply_fn(p_init)
            f_init_vec, _ = eval_corridor_fn(
                rf_init["path"], rf_init.get("flight_phases")
            )
            init_fvals[i_init, :] = np.asarray(f_init_vec, dtype=float)

        init_rep_indices = _representative_indices_v1(
            init_fvals, objective_weights
        )
        init_rep_objectives = [
            np.asarray(init_fvals[index], dtype=float)
            for index in init_rep_indices
        ]

    _plot_representative_corridor_figures(
        reps=reps,
        apply_rf_corridor_fn=rf_apply_fn,
        output_rf_view_fn=output_rf_view_fn,
        eval_corridor_objectives_fn=eval_corridor_fn,
        objective_names=objective_names,
        altitude_levels=altitude_levels,
        start_vertiport=start_vertiport,
        W_half=W_half,
        transition_corridor_cfg=transition_corridor_cfg,
        out_dir=out_dir,
        request=request,
        map_extent=map_extent,
        airspace_center_lla=airspace_center_lla,
        airspace_radius_m=airspace_radius_m,
        forbidden_zones=forbidden_zones,
        moc_plot_2d=moc_plot_2d,
        lat_lim=lat_lim,
        lon_lim=lon_lim,
        bb_full=bb_full,
        waypoints=waypoints,
        end_vertiport=end_vertiport,
        takeoff_complete=takeoff_complete,
        landing_entry=landing_entry,
        init_rep_objectives=init_rep_objectives,
        f_initial_backbone=f_initial_backbone,
        use_takeoff_landing_transition=use_takeoff_landing_transition,
        transition_structure_mode=transition_structure_mode,
        setup_corridor_axes_fn=_setup_corridor_axes,
        plot_standard_key_markers_fn=_plot_standard_key_markers,
    )

    rf_best = None
    rf_best_output = None
    best_rep = reps[-1] if reps else (pop[0] if pop else None)
    if best_rep is not None:
        rf_best = rf_apply_fn(best_rep)
        rf_best_output = output_rf_view_fn(rf_best)
        R_turn = rf_best["turn_radius_m"]
        segs_best = rf_best_output["segments"]
        g_mps2_local = 9.80665
        phi_local = np.deg2rad(bank_angle_deg)

        rows = []
        point_idx = 0

        def _is_same_point(p1, p2, tol_m=0.05):
            return _seg_dist_3d_m(np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)) <= float(tol_m)

        last_point = None

        def _append_vertiport_row(point, segment_label):
            nonlocal point_idx, last_point
            point = np.asarray(point, dtype=float).reshape(3)
            if last_point is not None and _is_same_point(point, last_point):
                # The profiled RF segments already include transition-on
                # vertiports. Reclassify that endpoint instead of duplicating it.
                rows[-1]["Type"] = "Vertiport"
                rows[-1]["Segment"] = segment_label
                rows[-1]["Flight_Phase"] = FLIGHT_PHASE_VERTIPORT
                rows[-1]["Ground_Speed_mps"] = 0.0
                rows[-1]["Bank_Angle_deg"] = 0.0
                last_point = point
                return
            rows.append({
                "Point_No": point_idx, "Type": "Vertiport", "Segment": segment_label,
                "Flight_Phase": FLIGHT_PHASE_VERTIPORT,
                "Lat": float(point[0]), "Lon": float(point[1]), "Altitude_MSL_m": float(point[2]),
                "Altitude_AGL_m": float(point[2] - start_vertiport[2]),
                "TF_Start": "", "TF_End": "", "RF_Start": "", "RF_End": "",
                "Arc_Center_Lat": "", "Arc_Center_Lon": "",
                "Turn_Radius_m": "", "Turn_Angle_deg": "", "LookAhead_Radius_Scale": "",
                "Ground_Speed_mps": 0.0, "Bank_Angle_deg": 0.0,
            })
            point_idx += 1
            last_point = point

        if not fixed_transition_general_output_suppressed:
            _append_vertiport_row(start_vertiport, "Start")

        # rf_best already contains fixed stage-1 TF segments and the profiled
        # optimized TF/RF span, in full flight order.
        seg_counter = 0
        for seg in segs_best:
            seg_counter += 1
            pts = np.asarray(seg["points"], dtype=float).reshape(-1, 3)
            stype = seg["type"]
            point_phases = np.asarray(
                seg.get("point_phases", np.full(pts.shape[0], FLIGHT_PHASE_CRUISE)),
                dtype=object,
            ).reshape(-1)
            if point_phases.size != pts.shape[0]:
                raise RuntimeError(
                    f"RF segment phase count mismatch: points={pts.shape[0]}, "
                    f"phases={point_phases.size}."
                )

            if stype == "TF":
                for pi in range(pts.shape[0]):
                    cur_pt = np.asarray(pts[pi], dtype=float)
                    if pi == 0 and last_point is not None and _is_same_point(cur_pt, last_point):
                        if rows:
                            rows[-1]["TF_Start"] = "O"
                        continue
                    phase = str(point_phases[pi])
                    is_start = "O" if pi == 0 else ""
                    is_end = "O" if pi == pts.shape[0] - 1 else ""
                    if phase in (FLIGHT_PHASE_TAKEOFF_STAGE1, FLIGHT_PHASE_TAKEOFF_STAGE2):
                        speed_mps = float(transition_speed_takeoff_mps)
                    elif phase in (FLIGHT_PHASE_LANDING_STAGE2, FLIGHT_PHASE_LANDING_STAGE1):
                        speed_mps = float(transition_speed_landing_mps)
                    else:
                        speed_mps = float(ground_speed_mps)
                    rows.append({
                        "Point_No": point_idx, "Type": "TF_Point", "Segment": f"Seg{seg_counter}",
                        "Flight_Phase": phase,
                        "Lat": cur_pt[0], "Lon": cur_pt[1], "Altitude_MSL_m": cur_pt[2],
                        "Altitude_AGL_m": float(cur_pt[2] - start_vertiport[2]),
                        "TF_Start": is_start, "TF_End": is_end,
                        "RF_Start": "", "RF_End": "",
                        "Arc_Center_Lat": "", "Arc_Center_Lon": "",
                        "Turn_Radius_m": "", "Turn_Angle_deg": "",
                        "LookAhead_Radius_Scale": "", "Ground_Speed_mps": speed_mps, "Bank_Angle_deg": 0.0,
                    })
                    point_idx += 1
                    last_point = cur_pt
            elif stype == "RF":
                arc_center = seg["arc_center"]
                turn_angle_deg = float(np.rad2deg(seg["turn_angle"]))
                turn_radius_i = float(seg.get("turn_radius", R_turn))
                radius_scale_i = (turn_radius_i / R_turn) if R_turn > 1e-12 else 1.0
                speed_i = float(np.sqrt(max(0.0, turn_radius_i * g_mps2_local * np.tan(phi_local))))
                for pi in range(pts.shape[0]):
                    cur_pt = np.asarray(pts[pi], dtype=float)
                    if pi == 0 and last_point is not None and _is_same_point(cur_pt, last_point):
                        if rows:
                            rows[-1]["RF_Start"] = "O"
                        continue
                    phase = str(point_phases[pi])
                    is_start = "O" if pi == 0 else ""
                    is_end = "O" if pi == pts.shape[0] - 1 else ""
                    arc_label = f"Arc_{pi+1}/{pts.shape[0]}"
                    rows.append({
                        "Point_No": point_idx, "Type": f"RF_Arc ({arc_label})",
                        "Segment": f"Seg{seg_counter}",
                        "Flight_Phase": phase,
                        "Lat": cur_pt[0], "Lon": cur_pt[1], "Altitude_MSL_m": cur_pt[2],
                        "Altitude_AGL_m": float(cur_pt[2] - start_vertiport[2]),
                        "TF_Start": "", "TF_End": "",
                        "RF_Start": is_start, "RF_End": is_end,
                        "Arc_Center_Lat": arc_center[0], "Arc_Center_Lon": arc_center[1],
                        "Turn_Radius_m": turn_radius_i, "Turn_Angle_deg": turn_angle_deg,
                        "LookAhead_Radius_Scale": radius_scale_i,
                        "Ground_Speed_mps": speed_i, "Bank_Angle_deg": bank_angle_deg,
                    })
                    point_idx += 1
                    last_point = cur_pt

        if not fixed_transition_general_output_suppressed:
            _append_vertiport_row(end_vertiport, "End")

        if use_takeoff_landing_transition:
            row_path = np.asarray([
                [row["Lat"], row["Lon"], row["Altitude_MSL_m"]]
                for row in rows
            ], dtype=float)
            rf_path = np.asarray(rf_best_output["path"], dtype=float).reshape(-1, 3)
            row_phases = np.asarray([row["Flight_Phase"] for row in rows], dtype=object)
            rf_phases = np.asarray(
                rf_best_output.get("flight_phases", []), dtype=object
            ).reshape(-1)
            if row_path.shape != rf_path.shape or row_phases.shape != rf_phases.shape:
                raise RuntimeError(
                    "Exported route is not aligned with the profiled RF path: "
                    f"rows={row_path.shape[0]}, path={rf_path.shape[0]}, "
                    f"row_phases={row_phases.size}, path_phases={rf_phases.size}."
                )
            point_errors_m = np.asarray([
                _seg_dist_3d_m(row_path[i], rf_path[i])
                for i in range(rf_path.shape[0])
            ], dtype=float)
            if np.any(point_errors_m > 0.05) or not np.array_equal(row_phases, rf_phases):
                raise RuntimeError("Exported route points/phases differ from rf_best path/phases.")

            _, overall_constraint_ok, overall_constraint_reason = (
                evaluate_objectives_with_constraints_gp(
                    rf_best["path"],
                    return_reason=True,
                    flight_phases=rf_best.get("flight_phases"),
                    **eval_cfg,
                )
            )
            airspace_ok, airspace_reason, transition_airspace_audit = (
                _is_path_inside_airspace_envelope_v1(
                rf_best["path"],
                rf_best.get("flight_phases"),
                airspace_center_lla[:2],
                airspace_radius_m,
                cruise_half_width_m=W_half,
                alt_min_m=airspace_alt_min_m,
                alt_max_m=airspace_alt_max_m,
                transition_corridor_cfg=transition_corridor_cfg,
                )
            )
            rf_best["transition_airspace_validation"] = transition_airspace_audit
            transition_3d_validation = _build_transition_3d_validation_v1(
                rf=rf_best,
                cruise_half_width_m=W_half,
                transition_corridor_cfg=transition_corridor_cfg,
                moc_risk=MOCRisk,
                moc_enforced=check_corridor_moc,
                lat_lim=lat_lim,
                lon_lim=lon_lim,
                forbidden_zones=forbidden_zones,
                check_corridor_nfz=check_corridor_nfz,
                check_corridor_self_overlap=check_corridor_self_overlap,
                airspace_audit=transition_airspace_audit,
            )
            rf_best["transition_3d_validation"] = transition_3d_validation
            params_dict["transition_3d_validation"] = transition_3d_validation
            min_distance_ok = bool(
                min_corridor_distance_m <= 0.0
                or _path_total_3d_distance_m(rf_best["path"]) + 1e-6
                >= min_corridor_distance_m
            )
        _export_route_outputs(
            rows=rows,
            rf_best=rf_best,
            rf_output=rf_best_output,
            Norm_RT=Norm_RT,
            AirRisk=AirRisk,
            altitude_levels=altitude_levels,
            risk_altitude_levels=risk_altitude_levels,
            use_heading_map=use_heading_map,
            air_thr_global=air_thr_global,
            lat_lim=lat_lim,
            lon_lim=lon_lim,
            NoiseRisk=NoiseRisk,
            NoiseRiskDb=NoiseRiskDb,
            cell_size=cell_size,
            refine_scales=refine_scales,
            min_corridor_distance_m=min_corridor_distance_m,
            out_dir=out_dir,
            params_dict=params_dict,
            w_noise=w_noise,
            noise_floor_db=noise_floor_db,
            evaluate_objectives_kwargs=dict(
                Norm_RT=Norm_RT,
                AirRisk=AirRisk,
                use_heading_map=use_heading_map,
                flight_dist_limit=flight_dist_limit,
                forbidden_zones=forbidden_zones,
                delta_z_max=delta_z_max,
                altitude_levels=risk_altitude_levels,
                cell_size=cell_size,
                refine_scales=refine_scales,
                air_risk_threshold=air_thr_global,
                w_dist=w_dist,
                w_ground=w_ground,
                w_air=w_air,
                lat_lim=lat_lim,
                lon_lim=lon_lim,
                NoiseRisk=NoiseRisk,
                noise_floor_db=noise_floor_db,
                w_noise=w_noise,
                W_half=W_half,
                check_corridor_nfz=check_corridor_nfz,
                MOCRisk=MOCRisk,
                check_corridor_moc=check_corridor_moc,
                check_corridor_self_overlap=check_corridor_self_overlap,
                vertiport=None,
                landing_entry=None,
                takeoff_complete=None,
                transition_corridor_cfg=transition_corridor_cfg,
            ),
            start_vertiport=start_vertiport,
            airspace_center_lla=airspace_center_lla,
            airspace_radius_m=airspace_radius_m,
            airspace_radius_km=airspace_radius_km,
            airspace_alt_min_m=airspace_alt_min_m,
            airspace_alt_max_m=airspace_alt_max_m,
            forbidden_zones=forbidden_zones,
            use_takeoff_landing_transition=use_takeoff_landing_transition,
            end_vertiport=end_vertiport,
            takeoff_complete=takeoff_complete,
            landing_entry=landing_entry,
            corridor_lat_default=corridor_lat_default,
            corridor_lon_default=corridor_lon_default,
            waypoint_alt_fixed_m=waypoint_alt_fixed_m,
        )

        if use_takeoff_landing_transition:
            try:
                moc_transition_visualization = _save_moc_transition_snapshots_v1(
                    rf=rf_best,
                    out_dir=out_dir,
                    request=request,
                    map_extent=map_extent,
                    moc_risk=MOCRisk,
                    moc_enforced=check_corridor_moc,
                    half_width_m=transition_corridor_half_width_m,
                    cruise_half_width_m=W_half,
                    transition_corridor_cfg=transition_corridor_cfg,
                    lat_lim=lat_lim,
                    lon_lim=lon_lim,
                    airspace_center_lla=airspace_center_lla,
                    airspace_radius_m=airspace_radius_m,
                    forbidden_zones=forbidden_zones,
                    start_vertiport=start_vertiport,
                    end_vertiport=end_vertiport,
                    overall_constraint_ok=overall_constraint_ok,
                    overall_constraint_reason=overall_constraint_reason,
                    airspace_ok=airspace_ok,
                    check_corridor_nfz=check_corridor_nfz,
                    check_corridor_self_overlap=check_corridor_self_overlap,
                    min_distance_enforced=min_corridor_distance_m > 0.0,
                    min_distance_ok=min_distance_ok,
                    rf_min_allowed_radius_m=look_ahead_min_turn_radius_m,
                )
            except Exception as exc:
                moc_transition_visualization = {
                    "enabled": True,
                    "generated": False,
                    "transition_structure_mode": str(transition_structure_mode),
                    "audit_only_fixed_transition_geometry": bool(
                        transition_structure_mode == TRANSITION_STRUCTURE_FIXED_ONLY
                    ),
                    "moc_audit_includes_fixed_transition": bool(
                        any(
                            bool(seg.get("is_fixed_transition_stage1", False))
                            for seg in rf_best.get("segments", [])
                        )
                    ),
                    "output_policy_notice": (
                        "audit-only fixed transition geometry; omitted from general corridor outputs"
                        if transition_structure_mode == TRANSITION_STRUCTURE_FIXED_ONLY
                        else None
                    ),
                    "reason": "generation_error",
                    "folder": None,
                    "partial_folder": (
                        "_moc_transition_snapshots_incomplete"
                        if (
                            out_dir / "_moc_transition_snapshots_incomplete"
                        ).exists() else None
                    ),
                    "files": [],
                    "moc_enforced": bool(check_corridor_moc),
                    "status": "FAIL",
                    "sample_count": 0,
                    "tested_count": 0,
                    "hit_count": 0,
                    "out_of_grid_count": 0,
                    "corridor_half_width_m": float(transition_corridor_half_width_m),
                    "transition_corridor_half_width_m": float(
                        transition_corridor_half_width_m
                    ),
                    "configured_downward_clearance_m": float(
                        transition_corridor_half_width_m
                    ),
                    "transition_vertical_clearance_m": float(
                        transition_corridor_half_width_m
                    ),
                    "cruise_corridor_half_width_m": float(W_half),
                    "transition_3d_status": str(
                        transition_3d_validation.get("status", "FAIL")
                    ),
                    "transition_3d_corridor_policy": transition_3d_corridor_policy,
                    "transition_3d_validation": transition_3d_validation,
                    "directions": {},
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                print(
                    "Warning: MOC transition snapshots were not completed: "
                    f"{type(exc).__name__}: {exc}"
                )
            params_dict["moc_transition_visualization"] = (
                moc_transition_visualization
            )

    cruise_risk_idx = int(np.argmin(
        np.abs(np.asarray(risk_altitude_levels, dtype=float) - float(altitude_levels[0]))
    ))
    noise_cruise_db = np.asarray(
        noise_3d_db_after_floor[:, :, cruise_risk_idx:cruise_risk_idx + 1], dtype=float
    ).copy()
    noise_cruise_vmax = float(np.max(noise_cruise_db)) if noise_cruise_db.size else 0.0
    noise_cruise_norm = (
        noise_cruise_db / noise_cruise_vmax
        if noise_cruise_vmax > 1e-12 else np.zeros_like(noise_cruise_db)
    )
    moc_cruise = np.asarray(
        MOCRisk[:, :, cruise_moc_idx:cruise_moc_idx + 1]
    ).copy()
    noise_meta_cruise = dict(noise_meta)
    selected_noise_indices = list(noise_meta.get("selected_layer_idx", []))
    noise_meta_cruise["selected_layer_idx"] = (
        [int(selected_noise_indices[cruise_risk_idx])]
        if cruise_risk_idx < len(selected_noise_indices) else []
    )
    noise_meta_cruise["noise_max_db_after_floor"] = noise_cruise_vmax
    moc_meta_cruise = dict(moc_meta)
    moc_meta_cruise["requested_agl_m"] = [
        float(altitude_levels[0] - MOC_REFERENCE_MSL_M)
    ]
    moc_meta_cruise["selected_agl_m"] = [int(MOC_AGL_LEVELS_M[cruise_moc_idx])]
    moc_meta_cruise["selected_ones_ratio_on_evaluation_grid"] = float(np.mean(moc_cruise))

    balanced_path = np.empty((0, 3), dtype=float)
    balanced_flight_phases = np.empty((0,), dtype=object)
    route_data_path = np.empty((0, 3), dtype=float)
    route_data_flight_phases = np.empty((0,), dtype=object)
    balanced_transition_meta = {}
    takeoff_stage1_end_result = None
    landing_stage1_start_result = None
    takeoff_transition_end_result = None
    landing_transition_start_result = None
    if rf_best is not None:
        balanced_path = np.asarray(
            rf_best_output.get("path", np.empty((0, 3))), dtype=float
        ).reshape(-1, 3).copy()
        balanced_flight_phases = np.asarray(
            rf_best_output.get("flight_phases", np.empty((0,), dtype=object)), dtype=object
        ).reshape(-1).copy()
        route_data_path = np.asarray([
            [row["Lat"], row["Lon"], row["Altitude_MSL_m"]]
            for row in rows
        ], dtype=float).reshape(-1, 3)
        route_data_flight_phases = np.asarray([
            row["Flight_Phase"] for row in rows
        ], dtype=object).reshape(-1)
        balanced_transition_meta = dict(rf_best.get("transition_meta", {}))
        if (
            np.any(balanced_flight_phases == FLIGHT_PHASE_TAKEOFF_STAGE1)
            and rf_best.get("takeoff_stage1_end") is not None
        ):
            takeoff_stage1_end_result = np.asarray(
                rf_best["takeoff_stage1_end"], dtype=float
            ).copy()
        if (
            np.any(balanced_flight_phases == FLIGHT_PHASE_LANDING_STAGE1)
            and rf_best.get("landing_stage1_start") is not None
        ):
            landing_stage1_start_result = np.asarray(
                rf_best["landing_stage1_start"], dtype=float
            ).copy()
        if rf_best.get("takeoff_transition_end") is not None:
            takeoff_transition_end_result = np.asarray(
                rf_best["takeoff_transition_end"], dtype=float
            ).copy()
        if rf_best.get("landing_transition_start") is not None:
            landing_transition_start_result = np.asarray(
                rf_best["landing_transition_start"], dtype=float
            ).copy()

    result = {
        "objective_names": objective_names,
        "backbone": backbone,
        "waypoints": waypoints,
        "representative_paths": reps,
        "population": pop,
        "f_vals": fvals,
        "start_vertiport": start_vertiport,
        "end_vertiport": end_vertiport,
        "vertiport": start_vertiport,
        "takeoff_complete": (
            takeoff_transition_end_result
            if takeoff_transition_end_result is not None else takeoff_complete
        ),
        "landing_entry": (
            landing_transition_start_result
            if landing_transition_start_result is not None else landing_entry
        ),
        "forbidden_zones": forbidden_zones,
        "emergency_points": emergency_points,
        "airspace_center_lla": airspace_center_lla,
        "airspace_radius_m": airspace_radius_m,
        "airspace_alt_min_m": airspace_alt_min_m,
        "airspace_alt_max_m": airspace_alt_max_m,
        "lat_lim": lat_lim,
        "lon_lim": lon_lim,
        "W_half": W_half,
        "transition_corridor_half_width_m": float(transition_corridor_half_width_m),
        "transition_vertical_clearance_m": float(transition_corridor_half_width_m),
        "transition_3d_corridor_policy": transition_3d_corridor_policy,
        "transition_3d_validation": transition_3d_validation,
        "ground_speed_mps": ground_speed_mps,
        "bank_angle_deg": bank_angle_deg,
        "bird_airrisk_path": str(bird_airrisk_path),
        "moc_meta": moc_meta_cruise,
        "moc_meta_all_agl": moc_meta,
        "check_corridor_moc": bool(check_corridor_moc),
        "moc_transition_visualization": moc_transition_visualization,
        "check_corridor_self_overlap": bool(check_corridor_self_overlap),
        "objective_values_are_raw": True,
        "w_dist": float(w_dist),
        "w_ground": float(w_ground),
        "w_air": float(w_air),
        "noise_npy_path": str(noise_npy_path),
        "noise_floor_db": noise_floor_db,
        "w_noise": w_noise,
        "objective_weighting_result": objective_weighting_result,
        "noise_meta": noise_meta_cruise,
        "noise_meta_all_msl": noise_meta,
        # Preserve the legacy one-layer result shapes at cruise altitude.
        "noise_map_3d_normalized": noise_cruise_norm,
        "noise_map_3d_db_after_floor": noise_cruise_db,
        "MOCRisk": moc_cruise,
        # v1 altitude-aware stacks and their explicit vertical coordinates.
        "noise_map_3d_normalized_all_msl": np.asarray(noise_3d_norm, dtype=float).copy(),
        "noise_map_3d_db_after_floor_all_msl": np.asarray(
            noise_3d_db_after_floor, dtype=float
        ).copy(),
        "risk_altitude_levels_msl_m": np.asarray(risk_altitude_levels, dtype=float).copy(),
        "MOCRisk_all_agl": np.asarray(MOCRisk).copy(),
        "moc_agl_levels_m": np.asarray(MOC_AGL_LEVELS_M, dtype=float).copy(),
        "moc_altitude_levels_msl_m": np.asarray(moc_altitude_levels_msl, dtype=float).copy(),
        "balanced_path": balanced_path,
        "balanced_flight_phases": balanced_flight_phases,
        "route_data_path": route_data_path,
        "route_data_flight_phases": route_data_flight_phases,
        "balanced_path_flight_phase_counts": {
            phase: int(np.count_nonzero(balanced_flight_phases == phase))
            for phase in (
                FLIGHT_PHASE_VERTIPORT,
                FLIGHT_PHASE_TAKEOFF_STAGE1,
                FLIGHT_PHASE_TAKEOFF_STAGE2,
                FLIGHT_PHASE_CRUISE,
                FLIGHT_PHASE_LANDING_STAGE2,
                FLIGHT_PHASE_LANDING_STAGE1,
            )
        },
        "route_data_flight_phase_counts": {
            phase: int(np.count_nonzero(route_data_flight_phases == phase))
            for phase in (
                FLIGHT_PHASE_VERTIPORT,
                FLIGHT_PHASE_TAKEOFF_STAGE1,
                FLIGHT_PHASE_TAKEOFF_STAGE2,
                FLIGHT_PHASE_CRUISE,
                FLIGHT_PHASE_LANDING_STAGE2,
                FLIGHT_PHASE_LANDING_STAGE1,
            )
        },
        "balanced_transition_meta": balanced_transition_meta,
        "sector_mode_enabled": bool(sector_mode_enabled),
        "sector_auto_selection_active": bool(sector_auto_selection_active),
        "sector_season": str(sector_season),
        "takeoff_sector_user": int(takeoff_sector_user),
        "landing_sector_user": int(landing_sector_user),
        "takeoff_sector_selected": int(takeoff_sector_selected),
        "landing_sector_selected": int(landing_sector_selected),
        "takeoff_sector": int(takeoff_sector_selected),
        "landing_sector": int(landing_sector_selected),
        "sector_selection_analysis": sector_selection_analysis,
        "use_takeoff_landing_transition": bool(use_takeoff_landing_transition),
        "transition_structure_mode": str(transition_structure_mode),
        "transition_structure_mode_effective": (
            str(transition_structure_mode)
            if use_takeoff_landing_transition else "off"
        ),
        "use_two_stage_transition": bool(use_two_stage_transition),
        "use_two_stage_transition_deprecated_alias_is_lossy": True,
        "effective_two_stage_transition": bool(
            use_takeoff_landing_transition and use_two_stage_transition
        ),
        "fixed_transition_general_output_suppressed": bool(
            fixed_transition_general_output_suppressed
        ),
        "fixed_transition_evaluated_but_not_exported": bool(
            fixed_transition_general_output_suppressed
        ),
        "moc_audit_includes_fixed_transition": bool(
            takeoff_transition_meta.get("fixed_straight_actual", False)
            or landing_transition_meta.get("fixed_straight_actual", False)
        ),
        "transition_mode": str(transition_mode),
        "transition_mode_effective": (
            str(takeoff_transition_meta.get("transition_mode", "off"))
            if use_takeoff_landing_transition else "off"
        ),
        "takeoff_total_transition_horizontal_distance_m": (
            None
            if takeoff_total_transition_horizontal_distance_m is None
            else (
                float(takeoff_total_transition_horizontal_distance_m)
                if total_distance_input_active
                else takeoff_total_transition_horizontal_distance_m
            )
        ),
        "landing_total_transition_horizontal_distance_m": (
            None
            if landing_total_transition_horizontal_distance_m is None
            else (
                float(landing_total_transition_horizontal_distance_m)
                if total_distance_input_active
                else landing_total_transition_horizontal_distance_m
            )
        ),
        "takeoff_total_transition_horizontal_distance_actual_m": float(
            takeoff_transition_meta.get("total_horizontal_distance_m", 0.0)
        ),
        "landing_total_transition_horizontal_distance_actual_m": float(
            landing_transition_meta.get("total_horizontal_distance_m", 0.0)
        ),
        "takeoff_actual_climb_angle_deg": float(
            takeoff_transition_meta.get("angle_deg", 0.0)
        ),
        "landing_actual_descent_angle_deg": float(
            landing_transition_meta.get("angle_deg", 0.0)
        ),
        "takeoff_fixed_prefix_requested_distance_m": float(
            takeoff_transition_meta.get("stage1_requested_straight_distance_m", 0.0)
        ),
        "landing_fixed_prefix_requested_distance_m": float(
            landing_transition_meta.get("stage1_requested_straight_distance_m", 0.0)
        ),
        "takeoff_fixed_prefix_effective_distance_m": float(
            takeoff_transition_meta.get("stage1_straight_distance_m", 0.0)
        ),
        "landing_fixed_prefix_effective_distance_m": float(
            landing_transition_meta.get("stage1_straight_distance_m", 0.0)
        ),
        "takeoff_fixed_straight_actual": bool(
            takeoff_transition_meta.get("fixed_straight_actual", False)
        ),
        "landing_fixed_straight_actual": bool(
            landing_transition_meta.get("fixed_straight_actual", False)
        ),
        "takeoff_optimized_transition_actual": bool(
            takeoff_transition_meta.get("optimized_transition_actual", False)
        ),
        "landing_optimized_transition_actual": bool(
            landing_transition_meta.get("optimized_transition_actual", False)
        ),
        "takeoff_stage2_collapsed_at_cruise": bool(
            takeoff_transition_meta.get("stage2_collapsed_at_cruise", False)
        ),
        "landing_stage2_collapsed_at_cruise": bool(
            landing_transition_meta.get("stage2_collapsed_at_cruise", False)
        ),
        "takeoff_fixed_prefix_clamped_to_cruise": bool(
            takeoff_transition_meta.get("stage1_clamped_to_cruise", False)
        ),
        "landing_fixed_prefix_clamped_to_cruise": bool(
            landing_transition_meta.get("stage1_clamped_to_cruise", False)
        ),
        "evaluation_full_path_distance_2d_m": float(
            params_dict.get("evaluation_full_path_distance_2d_m", 0.0)
        ),
        "evaluation_full_path_distance_3d_m": float(
            params_dict.get("evaluation_full_path_distance_3d_m", 0.0)
        ),
        "evaluation_full_path_point_count": int(
            params_dict.get("evaluation_full_path_point_count", 0)
        ),
        "public_corridor_distance_2d_m": float(
            params_dict.get("public_corridor_distance_2d_m", 0.0)
        ),
        "public_corridor_distance_3d_m": float(
            params_dict.get("public_corridor_distance_3d_m", 0.0)
        ),
        "public_corridor_point_count": int(
            params_dict.get("public_corridor_point_count", 0)
        ),
        "takeoff_stage1_straight_distance_m": (
            float(takeoff_stage1_straight_distance_m)
            if fixed_prefix_input_active else takeoff_stage1_straight_distance_m
        ),
        "landing_stage1_straight_distance_m": (
            float(landing_stage1_straight_distance_m)
            if fixed_prefix_input_active else landing_stage1_straight_distance_m
        ),
        "takeoff_climb_angle_deg": (
            float(takeoff_climb_angle_deg)
            if angle_input_active else takeoff_climb_angle_deg
        ),
        "landing_descent_angle_deg": (
            float(landing_descent_angle_deg)
            if angle_input_active else landing_descent_angle_deg
        ),
        "transition_feasible": bool(
            rf_best.get("transition_feasible", True) if rf_best is not None else True
        ),
        "transition_fail_reason": str(
            rf_best.get("transition_fail_reason", "ok") if rf_best is not None else "ok"
        ),
        "transition_fail_reasons": (
            [str(v) for v in rf_best.get("transition_fail_reasons", [])]
            if rf_best is not None else []
        ),
        "takeoff_stage1_end": takeoff_stage1_end_result,
        "takeoff_transition_end": takeoff_transition_end_result,
        "landing_transition_start": landing_transition_start_result,
        "landing_transition_end": landing_transition_start_result,
        "landing_stage1_start": landing_stage1_start_result,
        "rf_use_boundary_heading": bool(rf_use_boundary_heading),
        "rf_debug_level": str(rf_debug_level),
    }

    params_dict["moc_transition_visualization"] = moc_transition_visualization
    with open(out_dir / "params.json", "w", encoding="utf-8") as _pf:
        json.dump(params_dict, _pf, indent=2, ensure_ascii=False)

    out = out_dir / "results.pkl"
    with open(out, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved {out}")

    return True, feasible_count


if __name__ == "__main__":
    import gc
    os.makedirs("runs", exist_ok=True)
    _normal_finish = False
    _hard_exit_on_success = os.environ.get("WP_HARD_EXIT_ON_SUCCESS", "1").strip().lower() in ("1", "true", "yes", "on")

    try:
        attempt = 1
        while True:
            ok, feas = attempt_run_once()
            if ok:
                print(f"Success on attempt {attempt}. Feasible: {feas}")
                break
            else:
                print(f"Attempt {attempt} -> 0 feasible. Retrying ...")
            attempt += 1
        _normal_finish = True
    finally:
        # Cleanup: fully release matplotlib/tk resources before interpreter shutdown.
        cleanup_matplotlib_tk()
        gc.collect()
        # On Windows + TkAgg, interpreter teardown may still emit tkinter __del__ thread errors.
        # If the run finished normally, force a clean process exit to suppress teardown noise.
        if _normal_finish and _hard_exit_on_success:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            os._exit(0)



