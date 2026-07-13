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
from matplotlib.patches import Polygon
from scipy.ndimage import map_coordinates
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

from crossover_GP import crossover_gp
from mutation_GP import mutation_gp
from fast_non_dominated_sort import fast_non_dominated_sort
from generate_initial_population_GP import generate_initial_population_gp
from generate_reference_points import generate_reference_points
from normalize_objectives import normalize_objectives
from niching_selection import niching_selection
from evaluate_objectives_with_constraints_GP import evaluate_objectives_with_constraints_gp
from rf_turn import apply_rf_turns
from takeoff_landing_sector import (
    get_season_masks,
    normalize_season,
    sector_allowed,
    validate_sector_1based,
)


TAKEOFF_TRANSITION_PROFILE = None
LANDING_TRANSITION_PROFILE_DESC = None
RF_ALLOW_TANGENT_CLAMP = True
RF_CORNER_FIT_MARGIN = 0.95
RF_CORNER_MIN_TANGENT_M = 1.0
RF_MIN_TURN_ANGLE_DEG = 0.5


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


def _size_transition_horizontal_distance(
    delta_h_m,
    max_angle_deg,
    max_vertical_rate_mps,
    horiz_speed_mps,
    mode_label="transition",
    user_distance_m=None,
    strict_distance_check=False,
):
    """
    Size horizontal transition distance from two beginner-friendly constraints.

    계산 개요:
    - L_A: 경사각 제한(max_angle)으로 계산한 최소 수평 거리
    - L_B: 수직속도 제한(max_vertical_rate)으로 계산한 최소 수평 거리
    - 최종 요구 거리 L_req = max(L_A, L_B)
    즉, 두 제한을 동시에 만족하려면 더 큰 거리 기준을 사용한다.
    """
    dh = float(abs(delta_h_m))
    if dh <= 1e-9:
        la = 0.0
        lb = 0.0
        l_req = 0.0
        basis = "NONE(delta_h=0)"
    else:
        max_angle_deg = float(max_angle_deg)
        max_vertical_rate_mps = float(max_vertical_rate_mps)
        horiz_speed_mps = float(horiz_speed_mps)
        if max_angle_deg <= 0.0:
            raise ValueError("max_angle_deg must be > 0 for transition sizing.")
        if max_vertical_rate_mps <= 0.0:
            raise ValueError("max_vertical_rate_mps must be > 0 for transition sizing.")
        if horiz_speed_mps <= 0.0:
            raise ValueError("horiz_speed_mps must be > 0 for transition sizing.")

        gamma = np.deg2rad(np.clip(max_angle_deg, 1e-6, 89.0))
        la = float(dh / np.tan(gamma))
        lb = float((horiz_speed_mps * dh) / max_vertical_rate_mps)
        l_req = float(max(la, lb))
        basis = "A(angle)" if la >= lb else "B(vertical-rate)"

    l_use = float(l_req)
    user_note = "(auto)"
    if user_distance_m is not None:
        u = float(user_distance_m)
        if u < 0.0:
            raise ValueError("user_distance_m must be >= 0.")
        if u + 1e-9 < l_req:
            if strict_distance_check:
                raise ValueError(
                    f"{mode_label}: user_distance_m={u:.1f}m is shorter than required "
                    f"L_req={l_req:.1f}m (A={la:.1f}m, B={lb:.1f}m)."
                )
            l_use = float(l_req)
            user_note = f"(user={u:.1f}m too short -> clamped to L_req)"
        else:
            l_use = float(u)
            user_note = f"(user={u:.1f}m accepted)"

    print(
        f"[{mode_label}] transition sizing | "
        f"delta_h={dh:.1f}m | "
        f"L_A(angle-base)={la:.1f}m, L_B(rate-base)={lb:.1f}m | "
        f"selected={basis}, L_req={l_req:.1f}m | "
        f"L_use={l_use:.1f}m {user_note}"
    )

    return {
        "delta_h_m": dh,
        "L_A_m": float(la),
        "L_B_m": float(lb),
        "L_required_m": float(l_req),
        "L_used_m": float(l_use),
        "selected_basis": str(basis),
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


def build_transition_profile_by_constraints(
    start_lla,
    target_alt_m,
    heading_deg=None,
    sector=None,
    max_angle_deg=10.0,
    max_vertical_rate_mps=3.0,
    horiz_speed_mps=50.0,
    user_distance_m=None,
    strict_distance_check=False,
    sample_spacing_m=50.0,
    mode_label="transition",
):
    """Build endpoint + linear 3D profile using L_A/L_B sizing constraints."""
    s = np.asarray(start_lla, dtype=float).ravel()
    if s.size != 3:
        raise ValueError("start_lla must be [lat, lon, alt].")
    alt1 = float(target_alt_m)
    delta_h = float(alt1 - s[2])

    sizing = _size_transition_horizontal_distance(
        delta_h_m=delta_h,
        max_angle_deg=max_angle_deg,
        max_vertical_rate_mps=max_vertical_rate_mps,
        horiz_speed_mps=horiz_speed_mps,
        mode_label=mode_label,
        user_distance_m=user_distance_m,
        strict_distance_check=strict_distance_check,
    )

    if heading_deg is None:
        if sector is None:
            raise ValueError("Either heading_deg or sector must be provided.")
        heading_rad = _sector_angle(sector)
        heading_deg_used = float(np.rad2deg(heading_rad))
    else:
        heading_deg_used = float(heading_deg)
        heading_rad = np.deg2rad(heading_deg_used)

    d_use = float(sizing["L_used_m"])
    lat1, lon1 = _move_latlon(float(s[0]), float(s[1]), heading_rad, d_use)
    end_lla = np.array([lat1, lon1, alt1], dtype=float)
    profile = build_transition_profile_linear(s, end_lla, sample_spacing_m=sample_spacing_m)

    print(
        f"[{mode_label}] heading={heading_deg_used:.2f} deg, "
        f"end_alt(cruise target)={alt1:.1f}m, samples={profile.shape[0]}, "
        f"sample_spacing={float(sample_spacing_m):.1f}m"
    )

    sizing.update({
        "heading_deg": float(heading_deg_used),
        "sample_spacing_m": float(sample_spacing_m),
        "end_lla": end_lla.astype(float),
    })
    return end_lla.astype(float), d_use, profile.astype(float), sizing


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
    allow_tangent_clamp=None,
    corner_fit_margin=None,
    corner_min_tangent_m=None,
    min_turn_angle_deg=None,
):
    """
    Apply RF turns only to the cruise span between Takeoff_End and Landing_Start.

    The transition profiles stay straight and are stitched back around the RF core
    for the returned full-corridor path.
    """
    core = np.asarray(path_core, dtype=float)
    if core.size == 0:
        core = np.empty((0, 3), dtype=float)
    else:
        core = core.reshape(-1, 3)

    takeoff_profile = globals().get("TAKEOFF_TRANSITION_PROFILE", None)
    landing_profile_desc = globals().get("LANDING_TRANSITION_PROFILE_DESC", None)
    takeoff_complete = np.asarray(start_vertiport, dtype=float).reshape(3)
    landing_entry = np.asarray(end_vertiport, dtype=float).reshape(3)

    if takeoff_profile is not None and np.size(takeoff_profile) > 0:
        takeoff_profile = np.asarray(takeoff_profile, dtype=float).reshape(-1, 3)
        takeoff_complete = takeoff_profile[-1].astype(float)
    if landing_profile_desc is not None and np.size(landing_profile_desc) > 0:
        landing_profile_desc = np.asarray(landing_profile_desc, dtype=float).reshape(-1, 3)
        landing_entry = landing_profile_desc[0].astype(float)

    entry_heading_deg = None
    exit_heading_deg = None
    if takeoff_profile is not None and np.size(takeoff_profile) >= 6:
        tp = np.asarray(takeoff_profile, dtype=float).reshape(-1, 3)
        entry_heading_deg = _heading_deg_from_segment(tp[-2], tp[-1])
    if landing_profile_desc is not None and np.size(landing_profile_desc) >= 6:
        lp = np.asarray(landing_profile_desc, dtype=float).reshape(-1, 3)
        exit_heading_deg = _heading_deg_from_segment(lp[0], lp[1])

    backbone = np.vstack([
        takeoff_complete,
        core,
        landing_entry,
    ]).astype(float)

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
        entry_heading_deg=entry_heading_deg,
        exit_heading_deg=exit_heading_deg,
        allow_tangent_clamp=allow_tangent_clamp,
        corner_fit_margin=corner_fit_margin,
        corner_min_tangent_m=corner_min_tangent_m,
        min_turn_angle_deg=min_turn_angle_deg,
    )

    core_path = np.asarray(rf["path"], dtype=float)
    if core_path.ndim != 2:
        core_path = core_path.reshape(-1, 3)

    if takeoff_profile is not None and np.size(takeoff_profile) > 0 and landing_profile_desc is not None and np.size(landing_profile_desc) > 0:
        full_path = _stitch_full_corridor_from_profiles(
            takeoff_profile,
            core_path,
            landing_profile_desc,
        )
    else:
        full_path = core_path

    rf["path"] = np.asarray(full_path, dtype=float)
    return rf


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
    map_zoom=13,
    takeoff_complete=None,
    landing_entry=None,
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
    ax = fig.add_subplot(1, 1, 1, projection=request.crs)

    # Ensure full airspace rings are visible with margin, plus key overlays.
    extent_points = [np.asarray(vertiport[:2], dtype=float)]
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
    ax.set_title("Click WPs in order | Left: add, Right/Delete: undo, Enter: finish")

    draw_vertiport_radius_rings(ax, vertiport, radii_m=ring_radii_m)

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
        ax.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90, c="blue",
                   marker="^", transform=ccrs.Geodetic(), zorder=10, label="Takeoff")
    if landing_entry is not None:
        ax.scatter([landing_entry[1]], [landing_entry[0]], s=90, c="green",
                   marker="v", transform=ccrs.Geodetic(), zorder=10, label="Landing")

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

    ax.scatter([vertiport[1]], [vertiport[0]], s=140, c="red", edgecolors="k",
               marker="s", transform=ccrs.Geodetic(), zorder=10, label="Vertiport")
    ax.legend(loc="upper right")

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
            mk = ax.scatter(arr[:, 1], arr[:, 0], s=55, c="yellow", edgecolors="k",
                            marker="o", transform=ccrs.Geodetic(), zorder=11)
            click_markers.append(mk)
            if arr.shape[0] >= 2:
                ln = ax.plot(arr[:, 1], arr[:, 0], "-", color="yellow", linewidth=1.1,
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


def selection_nsga3(population, f_vals, feasible, N, ref_points):
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
                nf = normalize_objectives(f_vals[lf])
                sel = niching_selection(nf, ref_points, rem)
                next_idx.extend(lf[sel].tolist())
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
    mean_lat = float(np.mean(path[:, 0]))

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

        alt_idx = int(np.argmin(np.abs(altitude_levels - p1[2])))
        dist_2d_m = float(np.linalg.norm(vec) * 111000.0 * np.cos(np.deg2rad(mean_lat)))
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

        Iq = (xq_lon - minLon) / dLon_deg
        Jq = (yq_lat - minLat) / dLat_deg
        coords = np.vstack((Jq, Iq))

        ground_map = Norm_RT[alt_idx, head_idx, :, :]
        air_map = AirRisk[:, :, alt_idx]
        interp_ground = map_coordinates(ground_map, coords, order=1, cval=0.0)
        interp_air = map_coordinates(air_map, coords, order=1, cval=0.0)
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
    mean_lat = float(np.mean(path[:, 0]))

    total_noise = 0.0
    for i in range(path.shape[0] - 1):
        p1 = path[i, :]
        p2 = path[i + 1, :]
        vec = p2[:2] - p1[:2]

        alt_idx = int(np.argmin(np.abs(altitude_levels - p1[2]))) if nm.shape[2] > 1 else 0
        dist_2d_m = float(np.linalg.norm(vec) * 111000.0 * np.cos(np.deg2rad(mean_lat)))
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
        Iq = (xq_lon - minLon) / dLon_deg
        Jq = (yq_lat - minLat) / dLat_deg
        coords = np.vstack((Jq, Iq))

        noise_map = nm[:, :, alt_idx]
        interp_noise = map_coordinates(noise_map, coords, order=1, cval=0.0)
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


def plot_corridor_width(gx, path, W_half, color="yellow", alpha=0.2):
    if path is None or path.shape[0] < 2:
        return
    m_lat = 111000.0
    m_lon = 111000.0 * np.cos(np.deg2rad(float(np.mean(path[:, 0]))))
    wlat = W_half / m_lat
    wlon = W_half / m_lon
    left, right = [], []
    for i in range(len(path)):
        if i == 0:
            vec = path[1, :2] - path[0, :2]
        elif i == len(path) - 1:
            vec = path[-1, :2] - path[-2, :2]
        else:
            vec = path[i + 1, :2] - path[i - 1, :2]
        n = np.linalg.norm(vec)
        if n < 1e-10:
            continue
        perp = np.array([-vec[1], vec[0]]) / n
        left.append([path[i, 0] + perp[0] * wlat, path[i, 1] + perp[1] * wlon])
        right.append([path[i, 0] - perp[0] * wlat, path[i, 1] - perp[1] * wlon])
    if len(left) < 2:
        return
    pts = np.array(left + right[::-1])
    poly = Polygon(pts[:, [1, 0]], closed=True, facecolor=color, edgecolor="none",
                   alpha=alpha, transform=ccrs.Geodetic(), zorder=2)
    gx.add_patch(poly)


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


def save_excel_route_map_figure(xlsx_path, out_png_path=None, figure_title=None, use_takeoff_landing_transition=True):
    """
    Route_Data 시트의 경로를 지도 위에 시각화해 PNG로 저장한다.
    표시 요소:
    - Cruise Section, Takeoff Section, Landing Section
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
    mask_takeoff = type_text.str.contains("takeoff_path", na=False).to_numpy(dtype=bool)
    mask_landing = type_text.str.contains("landing_path", na=False).to_numpy(dtype=bool)
    mask_vertiport = type_text.eq("vertiport").to_numpy(dtype=bool)
    mask_rf = type_text.str.contains("rf_arc", na=False).to_numpy(dtype=bool)
    mask_tf = (type_text.str.contains("tf_point|takeoff_path_point|landing_path_point", na=False, regex=True)).to_numpy(dtype=bool)
    mask_cruise = ~(mask_takeoff | mask_landing | mask_vertiport)

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

    
    ax.plot(lons, lats, "-", color="gray", linewidth=2.5, alpha=0.4, zorder=3, label="Route Outline")
    
    _plot_masked_chunks(ax, lons, lats, mask_cruise, color="black", linewidth=2.8, alpha=1.0,
                        zorder=5, label="Cruise Section")
    
    _plot_masked_chunks(ax, lons, lats, mask_takeoff, color="royalblue", linewidth=3.2, alpha=0.95,
                        zorder=6, label="Takeoff Section")
    
    _plot_masked_chunks(ax, lons, lats, mask_landing, color="seagreen", linewidth=3.2, alpha=0.95,
                        zorder=6, label="Landing Section")
    
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

    if bool(use_takeoff_landing_transition):
        ax.scatter(lons[0], lats[0], c="limegreen", s=180, marker="*", edgecolors="darkgreen",
                   linewidths=1.2, zorder=15, label="Start (Vertiport)")
        ax.scatter(lons[-1], lats[-1], c="red", s=180, marker="X", edgecolors="darkred",
                   linewidths=1.2, zorder=15, label="End (Vertiport)")
    else:
        ax.scatter(lons[0], lats[0], c="limegreen", s=180, marker="*", edgecolors="darkgreen",
                   linewidths=1.2, zorder=15, label="Start (Takeoff Point)")
        ax.scatter(lons[-1], lats[-1], c="red", s=180, marker="X", edgecolors="darkred",
                   linewidths=1.2, zorder=15, label="End (Landing Point)")

    if figure_title is None:
        figure_title = "Balanced Optimal Corridor Replotted from Excel Route_Data"
    ax.set_title(figure_title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.grid(True, alpha=0.35, linestyle="--")

    mean_lat = float(np.mean(lats))
    ax.set_aspect(1.0 / max(1e-8, np.cos(np.deg2rad(mean_lat))))
    
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95, ncol=2, frameon=True, 
              fancybox=True, shadow=True)

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
    ground_speed_mps, bank_angle_deg, num_arc_points,
    look_ahead, look_ahead_threshold_m, look_ahead_min_scale, look_ahead_window,
    W_half, check_corridor_nfz, check_corridor_moc,
    MOCRisk,
    start_vertiport, end_vertiport, landing_entry, takeoff_complete,
    airspace_center_latlon, airspace_radius_m,
    airspace_alt_min_m, airspace_alt_max_m,
    min_corridor_distance_m,
):
    """NSGA-III with RF-turn preprocessing."""

    dummy = np.vstack([population[0][0], population[0][-1]])
    temp_f, _ = evaluate_objectives_with_constraints_gp(
        dummy, Norm_RT, AirRisk, use_map, f_limit, f_zones, dz, alt, cs, scales,
        air_thr, w_d, w_g, w_a, lat_lim, lon_lim,
        NoiseRisk=NoiseRisk, noise_floor_db=noise_floor_db, w_noise=w_n,
        W_half=W_half, check_corridor_nfz=check_corridor_nfz,
        MOCRisk=MOCRisk, check_corridor_moc=check_corridor_moc,
        vertiport=None, landing_entry=None, takeoff_complete=None,
    )
    num_obj = len(temp_f)
    H = num_obj + 1
    ref_points = generate_reference_points(num_obj, H)

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
        )
        full_path = rf["path"]
        if not is_path_inside_airspace(
            full_path,
            airspace_center_latlon,
            airspace_radius_m,
            alt_min_m=airspace_alt_min_m,
            alt_max_m=airspace_alt_max_m,
        ):
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
            vertiport=None, landing_entry=None, takeoff_complete=None,
        )
        if not rf["feasible"]:
            f_val = np.asarray(f_val, dtype=float) + 1e6
        return f_val, feas

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

        new_pop = selection_nsga3(pop, f_vals, selection_mask, N_pop, ref_points)
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


def pick_representatives(population, f_vals):
    if not population or f_vals.size == 0:
        return []
    n_obj = f_vals.shape[1]
    reps = []
    for i in range(n_obj):
        reps.append(population[int(np.argmin(f_vals[:, i]))])
    fronts = fast_non_dominated_sort(f_vals)
    if fronts and fronts[0]:
        f1 = np.array(fronts[0], dtype=int)
        nf = normalize_objectives(f_vals[f1])
        bal = int(np.argmin(np.linalg.norm(nf, axis=1)))
        reps.append(population[int(f1[bal])])
    else:
        reps.append(reps[0])
    return reps


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
    objective_names,
    apply_rf_corridor_fn,
    W_half,
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

        greps = pick_representatives(gpop, gf)

        figg = plt.figure(f"Generation {gno}: Evolved Corridor", figsize=(14, 10))
        figg.subplots_adjust(left=0.05, right=0.72)
        gxg = figg.add_subplot(1, 1, 1, projection=request.crs)
        gxg.set_extent(map_extent)
        gxg.add_image(request, 13)
        gxg.set_title(f"Generation {gno} Corridor (RF Turn applied)")
        draw_vertiport_radius_rings(gxg, airspace_center_lla, radii_m=(airspace_radius_m,))
        plot_forbidden_zones(gxg, forbidden_zones, face_alpha=0.10, edge_alpha=0.80)

        gxg.plot(bb_full[:, 1], bb_full[:, 0], "r--", linewidth=1.5, transform=ccrs.Geodetic(),
                 label="Backbone", zorder=4)
        gxg.scatter(waypoints[:, 1], waypoints[:, 0], s=60, c="orange", edgecolors="k",
                    linewidths=0.5, marker="o", transform=ccrs.Geodetic(), label="Waypoints", zorder=6)
        gxg.scatter([start_vertiport[1]], [start_vertiport[0]], s=120, c="red", edgecolors="k",
                    marker="s", transform=ccrs.Geodetic(), label="Start Vertiport", zorder=7)
        gxg.scatter([end_vertiport[1]], [end_vertiport[0]], s=120, c="crimson", edgecolors="k",
                    marker="D", transform=ccrs.Geodetic(), label="End Vertiport", zorder=7)
        gxg.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90, c="blue",
                    marker="^", transform=ccrs.Geodetic(), label="Takeoff", zorder=7)
        gxg.scatter([landing_entry[1]], [landing_entry[0]], s=90, c="green",
                    marker="v", transform=ccrs.Geodetic(), label="Landing", zorder=7)

        rep_labels = objective_names + ["Balanced"]
        rep_colors = [plt.cm.tab10(i % 10) for i in range(len(objective_names))] + ["black"]
        for ri, rep in enumerate(greps):
            rf = apply_rf_corridor_fn(rep)
            rp = rf["path"]
            segs = rf["segments"]
            col = rep_colors[ri] if ri < len(rep_colors) else rep_colors[-1]
            lab = rep_labels[ri] if ri < len(rep_labels) else f"Rep{ri}"

            plot_corridor_width(gxg, rp, W_half, color=col, alpha=0.08)

            for seg in segs:
                pts = seg["points"]
                if seg["type"] == "TF":
                    gxg.plot(pts[:, 1], pts[:, 0], "-", color=col, linewidth=1.5,
                             transform=ccrs.Geodetic(), zorder=8)
                elif seg["type"] == "RF":
                    gxg.plot(pts[:, 1], pts[:, 0], "-", color=col, linewidth=2.0,
                             transform=ccrs.Geodetic(), zorder=8)

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
):
    """Evaluate final feasibility mask and RF no-clamp count for a population."""
    rf_no_clamp_count = 0
    feas_mask = []
    for i in range(len(pop)):
        rf = apply_rf_corridor_fn(pop[i])
        if not bool(rf.get("had_clamp", False)):
            rf_no_clamp_count += 1
        full_path = rf["path"]
        _, feas = eval_corridor_objectives_fn(full_path)
        air_ok = is_path_inside_airspace(
            full_path,
            airspace_center_lla[:2],
            airspace_radius_m,
            alt_min_m=airspace_alt_min_m,
            alt_max_m=airspace_alt_max_m,
        )
        dist_ok = True
        if min_corridor_distance_m > 0.0:
            dist_ok = _path_total_3d_distance_m(full_path) + 1e-6 >= min_corridor_distance_m
        feas = bool(feas and air_ok and dist_ok)
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
    )


def _evaluate_corridor_objectives_path(path_points, eval_cfg):
    """Evaluate objectives/constraints for a corridor path using prebuilt config."""
    return evaluate_objectives_with_constraints_gp(path_points, **eval_cfg)


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
    skip_ratio = float(np.clip(init_pop_skip_mix_ratio, 0.0, 1.0)) if use_wp_skip_generator else 0.0
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
            if use_wp_skip_generator:
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
):
    """Evaluate RF + constraints for initial candidates and return summary stats."""
    rf_ok_list = []
    for c in candidate_pop:
        c_eval = _enforce_mandatory_wp_order(c, backbone) if enforce_mandatory_wp_order else c
        rf = apply_rf_fn(c_eval)
        rf_ok_list.append((rf["feasible"], bool(rf.get("had_clamp", False)), rf["path"], c_eval))

    rf_cnt = sum(1 for ok, _, _, _ in rf_ok_list if ok)
    rf_no_clamp_cnt = sum(1 for ok, had_clamp, _, _ in rf_ok_list if ok and (not had_clamp))

    both_cnt = 0
    cst_cnt = 0
    air_cnt = 0
    dist_cnt = 0
    reason_counts = {}
    feasible_init = []

    if rf_cnt > 0:
        for rf_ok, _had_clamp, rf_path, c_eval in rf_ok_list:
            if not rf_ok:
                continue
            full_path = rf_path
            _, cst_ok, reason = eval_constraints_with_reason_fn(full_path)
            air_ok = is_path_inside_airspace(
                full_path,
                airspace_center_lla[:2],
                airspace_radius_m,
                alt_min_m=airspace_alt_min_m,
                alt_max_m=airspace_alt_max_m,
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
    Norm_RT,
    AirRisk,
    altitude_levels,
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
    df = pd.DataFrame(rows)
    if not df.empty:
        df["Point_No"] = np.arange(len(df), dtype=int)

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
    p2d_cum = [0.0]
    pcum = [0.0]
    for i in range(1, full_path.shape[0]):
        d2 = _seg_dist_m(full_path[i - 1], full_path[i])
        d3 = _seg_dist_3d_m(full_path[i - 1], full_path[i])
        p2d_cum.append(p2d_cum[-1] + d2)
        pcum.append(pcum[-1] + d3)

    route_points = df[["Lat", "Lon", "Altitude_MSL_m"]].to_numpy(dtype=float)
    pt_ground, pt_air, _ = _sample_point_risks(
        route_points, Norm_RT, AirRisk, altitude_levels,
        use_heading_map, air_thr_global, lat_lim, lon_lim
    )
    pt_noise_norm = _sample_point_noise(route_points, NoiseRisk, altitude_levels, lat_lim, lon_lim)
    pt_noise_db = _sample_point_noise(route_points, NoiseRiskDb, altitude_levels, lat_lim, lon_lim)
    df["Combined_Risk"] = pt_ground + pt_air + pt_noise_norm
    df["Ground_Risk"] = pt_ground
    df["Air_Risk"] = pt_air
    df["Noise_Risk_Norm_0to1"] = pt_noise_norm
    df["Noise_Lden_dB_Above_Floor"] = pt_noise_db
    df["Corridor_Tag"] = "Balanced_Optimal"
    df["Ground_Speed_kmh"] = pd.to_numeric(df["Ground_Speed_mps"], errors="coerce") * 3.6

    total_ground_risk, total_air_risk, _ = _aggregate_path_risks(
        full_path, Norm_RT, AirRisk, altitude_levels,
        use_heading_map, cell_size, refine_scales,
        air_thr_global, lat_lim, lon_lim
    )
    total_noise_risk_norm = _aggregate_path_noise(
        full_path, NoiseRisk, altitude_levels, cell_size, refine_scales, lat_lim, lon_lim
    )
    total_noise_db_after_floor = _aggregate_path_noise(
        full_path, NoiseRiskDb, altitude_levels, cell_size, refine_scales, lat_lim, lon_lim
    )
    total_combined_all_risk = float(total_ground_risk + total_air_risk + total_noise_risk_norm)
    total_corridor_dist_2d_km = (float(p2d_cum[-1]) if len(p2d_cum) > 0 else 0.0) / 1000.0
    total_corridor_dist_3d_m = float(pcum[-1]) if len(pcum) > 0 else 0.0
    total_corridor_dist_3d_km = total_corridor_dist_3d_m / 1000.0
    _ = (min_corridor_distance_m <= 0.0) or (total_corridor_dist_3d_m + 1e-6 >= min_corridor_distance_m)

    f_best, _ = evaluate_objectives_with_constraints_gp(full_path, **evaluate_objectives_kwargs)

    df_summary = pd.DataFrame([
        {"Metric": "Selected_Corridor", "Value": "Balanced_Optimal"},
        {"Metric": "Total_Corridor_Distance_2D_km", "Value": total_corridor_dist_2d_km},
        {"Metric": "Total_Corridor_Distance_3D_km", "Value": total_corridor_dist_3d_km},
        {"Metric": "Total_Ground_Risk", "Value": total_ground_risk},
        {"Metric": "Total_Air_Risk", "Value": total_air_risk},
        {"Metric": "Total_Noise_Risk", "Value": total_noise_risk_norm},
        {"Metric": "Total_Combined_Risk", "Value": total_combined_all_risk},
    ])

    params_dict.update({
        "noise_result_summary": {
            "total_noise_risk_normalized": float(total_noise_risk_norm),
            "total_noise_lden_db_after_floor": float(total_noise_db_after_floor),
            "objective_noise_risk_weighted": (float(f_best[3]) if len(f_best) > 3 else None),
            "w_noise": float(w_noise),
            "noise_floor_db": float(noise_floor_db),
        }
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
        "Point_No", "Type", "Segment", "Lat", "Lon", "Altitude_MSL_m", "Altitude_AGL_m",
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
    cr_mask = cr_mask & non_vertiport_mask
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

    input_rows = [
        {
            "Point_Name": "Start_Vertiport",
            "Lat": float(start_vertiport[0]),
            "Lon": float(start_vertiport[1]),
            "Alt_m": float(start_vertiport[2]),
        },
        {
            "Point_Name": "End_Vertiport",
            "Lat": float(end_vertiport[0]),
            "Lon": float(end_vertiport[1]),
            "Alt_m": float(end_vertiport[2]),
        },
        {
            "Point_Name": "Takeoff_Point",
            "Lat": float(takeoff_complete[0]),
            "Lon": float(takeoff_complete[1]),
            "Alt_m": float(takeoff_complete[2]),
        },
        {
            "Point_Name": "Landing_Point",
            "Lat": float(landing_entry[0]),
            "Lon": float(landing_entry[1]),
            "Alt_m": float(landing_entry[2]),
        },
    ]
    n_wp_default = min(int(np.size(corridor_lat_default)), int(np.size(corridor_lon_default)))
    for i_wp in range(n_wp_default):
        input_rows.append({
            "Point_Name": f"WP_Default_{i_wp+1:03d}",
            "Lat": float(corridor_lat_default[i_wp]),
            "Lon": float(corridor_lon_default[i_wp]),
            "Alt_m": float(waypoint_alt_fixed_m),
        })
    df_input_points = pd.DataFrame(input_rows, columns=[
        "Point_Name", "Lat", "Lon", "Alt_m"
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
    print(f"Saved {xlsx_name} (Optimized Corridor: {len(df_excel)} points, Transitions excluded)")

    excel_fig_name = out_dir / "fig_route_from_excel_map.png"
    excel_agl_value = float(altitude_levels[0] - start_vertiport[2])
    excel_fig_title = (
        "Balanced Optimal Corridor by Phase (from route_data.xlsx)\n"
        f"Altitude: {float(altitude_levels[0]):.1f}m MSL (= {excel_agl_value:.1f}m AGL above vertiport)"
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
    eval_corridor_objectives_fn,
    objective_names,
    altitude_levels,
    start_vertiport,
    W_half,
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
    takeoff_transition_profile,
    landing_transition_profile_desc,
    use_takeoff_landing_transition,
    setup_corridor_axes_fn,
    plot_standard_key_markers_fn,
):
    """Render representative corridor figures (Fig4/Fig5/Fig5B)."""

    def _add_rf_legend_handles(ax, color, name, tf_lw, rf_lw, include_transition=False, transition_lw=1.2):
        ax.plot([], [], "-", color=color, linewidth=tf_lw, label=f"{name} (TF)")
        ax.plot([], [], "-", color=color, linewidth=rf_lw, label=f"{name} (RF arc)")
        if include_transition:
            ax.plot([], [], "-", color="dodgerblue", linewidth=transition_lw, label="Takeoff Transition")
            ax.plot([], [], "-", color="seagreen", linewidth=transition_lw, label="Landing Transition")

    def _add_arc_marker_legend_handles(ax):
        ax.scatter([], [], s=40, c="yellow", marker=">", edgecolors="k", label="Arc Start")
        ax.scatter([], [], s=40, c="yellow", marker="s", edgecolors="k", label="Arc End")
        ax.scatter([], [], s=50, c="white", marker="x", linewidths=1.5, label="Arc Center")

    def _add_cr_rfc_legend_handles(ax):
        ax.scatter([], [], s=36, facecolors="none", edgecolors="red", linewidths=1.1, label="CR Point")
        ax.scatter([], [], s=50, c="white", marker="x", linewidths=1.5, label="RFC (Arc Center)")

    def _collect_cr_points_from_segments(segments, tol_m=0.5):
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
    agl_value = float(altitude_levels[0] - start_vertiport[2])
    title_str = f"Optimal Corridor\nAltitude: {float(altitude_levels[0]):.1f}m MSL (= {agl_value:.1f}m AGL above vertiport)"
    gx4 = setup_corridor_axes_fn(fig4, title_str, with_moc=True)
    plot_standard_key_markers_fn(gx4, include_waypoints=True, include_backbone=True, zorder=7)

    rep_labels = objective_names + ["Balanced"]
    rep_colors = [plt.cm.tab10(i % 10) for i in range(len(objective_names))] + ["black"]
    for ri, rep in enumerate(reps):
        rf = apply_rf_corridor_fn(rep)
        rp = rf["path"]
        segs = rf["segments"]
        col = rep_colors[ri] if ri < len(rep_colors) else rep_colors[-1]
        lab = rep_labels[ri] if ri < len(rep_labels) else f"Rep{ri}"

        plot_corridor_width(gx4, rp, W_half, color=col, alpha=0.08)

        for seg in segs:
            pts = seg["points"]
            if seg["type"] == "TF":
                gx4.plot(pts[:, 1], pts[:, 0], "-", color=col, linewidth=1.5,
                         transform=ccrs.Geodetic(), zorder=8)
                gx4.scatter(pts[0, 1], pts[0, 0], s=25, color=col, marker="o",
                            edgecolors="k", linewidths=0.4, transform=ccrs.Geodetic(), zorder=9)
                gx4.scatter(pts[-1, 1], pts[-1, 0], s=25, color=col, marker="o",
                            edgecolors="k", linewidths=0.4, transform=ccrs.Geodetic(), zorder=9)
            elif seg["type"] == "RF":
                gx4.plot(pts[:, 1], pts[:, 0], "-", color=col, linewidth=2.0,
                         transform=ccrs.Geodetic(), zorder=8)
                gx4.scatter(pts[0, 1], pts[0, 0], s=40, c="yellow", marker=">",
                            edgecolors="k", linewidths=0.6, transform=ccrs.Geodetic(), zorder=10)
                gx4.scatter(pts[-1, 1], pts[-1, 0], s=40, c="yellow", marker="s",
                            edgecolors="k", linewidths=0.6, transform=ccrs.Geodetic(), zorder=10)
                ac = seg["arc_center"]
                gx4.scatter(ac[1], ac[0], s=50, c="white", marker="x",
                            linewidths=1.5, transform=ccrs.Geodetic(), zorder=10)

        _add_rf_legend_handles(gx4, col, lab, tf_lw=1.5, rf_lw=2.5, include_transition=False)

    _add_arc_marker_legend_handles(gx4)

    gx4.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, framealpha=0.9)
    fig4.savefig(out_dir / "fig4_optimal_corridor.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_dir / 'fig4_optimal_corridor.png'}")
    plt.close(fig4)

    if len(reps) > 3:
        risk_configs = [
            {"idx": 1, "name": "Ground Risk", "filename": "fig4_ground_risk_corridor.png", "color": "orange"},
            {"idx": 2, "name": "Air Risk", "filename": "fig4_air_risk_corridor.png", "color": "blue"},
            {"idx": 3, "name": "Noise Risk", "filename": "fig4_noise_risk_corridor.png", "color": "purple"},
        ]
        for config in risk_configs:
            risk_idx = config["idx"]
            if risk_idx >= len(reps):
                continue

            risk_rep = reps[risk_idx]
            rf_risk = apply_rf_corridor_fn(risk_rep)
            risk_full_path = rf_risk["path"]

            fig_risk = plt.figure(f"Figure 4: {config['name']} Corridor", figsize=(14, 10))
            fig_risk.subplots_adjust(left=0.05, right=0.72)
            gx_risk = fig_risk.add_subplot(1, 1, 1, projection=request.crs)
            gx_risk.set_extent(map_extent)
            gx_risk.add_image(request, 13)
            f_risk_opt, _ = eval_corridor_objectives_fn(risk_full_path)
            agl_risk = float(altitude_levels[0] - start_vertiport[2])
            if init_rep_objectives is not None and risk_idx < len(init_rep_objectives):
                risk_init_val = float(init_rep_objectives[risk_idx][risk_idx])
            else:
                risk_init_val = float(f_initial_backbone[risk_idx])
            risk_opt_val = float(f_risk_opt[risk_idx])
            risk_title = (
                f"{config['name']} Corridor | Alt: {float(altitude_levels[0]):.1f}m MSL ({agl_risk:.1f}m AGL)\n"
                f"Risk init->opt: {risk_init_val:.4f} -> {risk_opt_val:.4f}"
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
            gx_risk.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90, c="blue",
                            marker="^", transform=ccrs.Geodetic(), label="Takeoff_End", zorder=7)
            gx_risk.scatter([landing_entry[1]], [landing_entry[0]], s=90, c="green",
                            marker="v", transform=ccrs.Geodetic(), label="Landing_Start", zorder=7)

            plot_corridor_width(gx_risk, risk_full_path, W_half, color=config["color"], alpha=0.08)

            if takeoff_transition_profile is not None and np.size(takeoff_transition_profile) > 0:
                tp = np.asarray(takeoff_transition_profile, dtype=float)
                gx_risk.plot(tp[:, 1], tp[:, 0], "-", color="dodgerblue", linewidth=1.2,
                             transform=ccrs.Geodetic(), zorder=7)
            if landing_transition_profile_desc is not None and np.size(landing_transition_profile_desc) > 0:
                lp = np.asarray(landing_transition_profile_desc, dtype=float)
                gx_risk.plot(lp[:, 1], lp[:, 0], "-", color="seagreen", linewidth=1.2,
                             transform=ccrs.Geodetic(), zorder=7)

            for seg in rf_risk["segments"]:
                pts = seg["points"]
                if seg["type"] == "TF":
                    gx_risk.plot(pts[:, 1], pts[:, 0], "-", color=config["color"], linewidth=1.5,
                                 transform=ccrs.Geodetic(), zorder=8)
                    gx_risk.scatter(pts[0, 1], pts[0, 0], s=25, color=config["color"], marker="o",
                                    edgecolors="k", linewidths=0.4, transform=ccrs.Geodetic(), zorder=9)
                    gx_risk.scatter(pts[-1, 1], pts[-1, 0], s=25, color=config["color"], marker="o",
                                    edgecolors="k", linewidths=0.4, transform=ccrs.Geodetic(), zorder=9)
                elif seg["type"] == "RF":
                    gx_risk.plot(pts[:, 1], pts[:, 0], "-", color=config["color"], linewidth=2.0,
                                 transform=ccrs.Geodetic(), zorder=8)
                    gx_risk.scatter(pts[0, 1], pts[0, 0], s=40, c="yellow", marker=">",
                                    edgecolors="k", linewidths=0.6, transform=ccrs.Geodetic(), zorder=10)
                    gx_risk.scatter(pts[-1, 1], pts[-1, 0], s=40, c="yellow", marker="s",
                                    edgecolors="k", linewidths=0.6, transform=ccrs.Geodetic(), zorder=10)
                    ac = seg["arc_center"]
                    gx_risk.scatter(ac[1], ac[0], s=50, c="white", marker="x",
                                    linewidths=1.5, transform=ccrs.Geodetic(), zorder=10)

            _add_rf_legend_handles(
                gx_risk, config["color"], config["name"], tf_lw=1.5, rf_lw=2.5,
                include_transition=True, transition_lw=1.2
            )
            _add_arc_marker_legend_handles(gx_risk)

            gx_risk.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, framealpha=0.9)
            fig_risk.savefig(out_dir / config["filename"], dpi=150, bbox_inches="tight")
            print(f"Saved {out_dir / config['filename']}")
            plt.close(fig_risk)

    if reps:
        balanced_rep = reps[-1]
        rf_bal = apply_rf_corridor_fn(balanced_rep)
        bal_full_path = rf_bal["path"]

        fig5 = plt.figure("Figure 5: Balanced Corridor Only", figsize=(14, 10))
        agl_value = float(altitude_levels[0] - start_vertiport[2])
        title_str = f"Balanced Corridor Only (RF Turn)\nAltitude: {float(altitude_levels[0]):.1f}m MSL (= {agl_value:.1f}m AGL)"
        gx5 = setup_corridor_axes_fn(fig5, title_str, with_moc=True)
        plot_standard_key_markers_fn(
            gx5, include_waypoints=True, include_backbone=True,
            takeoff_label="Takeoff_End", landing_label="Landing_Start",
            backbone_lw=1.2, zorder=7
        )

        plot_corridor_width(gx5, bal_full_path, W_half, color="black", alpha=0.14)
        if takeoff_transition_profile is not None and np.size(takeoff_transition_profile) > 0:
            tp = np.asarray(takeoff_transition_profile, dtype=float)
            gx5.plot(tp[:, 1], tp[:, 0], "-", color="dodgerblue", linewidth=1.5,
                     transform=ccrs.Geodetic(), zorder=7)
        if landing_transition_profile_desc is not None and np.size(landing_transition_profile_desc) > 0:
            lp = np.asarray(landing_transition_profile_desc, dtype=float)
            gx5.plot(lp[:, 1], lp[:, 0], "-", color="seagreen", linewidth=1.5,
                     transform=ccrs.Geodetic(), zorder=7)
        for seg in rf_bal["segments"]:
            pts = seg["points"]
            if seg["type"] == "TF":
                gx5.plot(pts[:, 1], pts[:, 0], "-", color="black", linewidth=1.8,
                         transform=ccrs.Geodetic(), zorder=8)
            else:
                gx5.plot(pts[:, 1], pts[:, 0], "-", color="black", linewidth=2.8,
                         transform=ccrs.Geodetic(), zorder=9)
                gx5.scatter(pts[0, 1], pts[0, 0], s=40, c="yellow", marker=">",
                            edgecolors="k", linewidths=0.6, transform=ccrs.Geodetic(), zorder=10)
                gx5.scatter(pts[-1, 1], pts[-1, 0], s=40, c="yellow", marker="s",
                            edgecolors="k", linewidths=0.6, transform=ccrs.Geodetic(), zorder=10)
                ac = seg["arc_center"]
                gx5.scatter(ac[1], ac[0], s=50, c="white", marker="x",
                            linewidths=1.5, transform=ccrs.Geodetic(), zorder=10)

        _add_rf_legend_handles(
            gx5, "black", "Balanced", tf_lw=1.8, rf_lw=2.8,
            include_transition=True, transition_lw=1.5
        )
        _add_arc_marker_legend_handles(gx5)
        gx5.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, framealpha=0.9)
        fig5.savefig(out_dir / "fig5_balanced_only.png", dpi=150, bbox_inches="tight")
        print(f"Saved {out_dir / 'fig5_balanced_only.png'}")
        plt.close(fig5)

        fig5b = plt.figure("Figure 5B: Balanced Corridor (Fig4 Style)", figsize=(14, 10))
        agl_value = float(altitude_levels[0] - start_vertiport[2])
        title_str = f"Balanced Corridor Only (Fig4 Style)\nAltitude: {float(altitude_levels[0]):.1f}m MSL (= {agl_value:.1f}m AGL)"
        gx5b = setup_corridor_axes_fn(fig5b, title_str)
        plot_standard_key_markers_fn(
            gx5b, include_waypoints=True, include_backbone=True,
            takeoff_label="Takeoff_End", landing_label="Landing_Start",
            backbone_lw=1.2, zorder=10
        )

        plot_corridor_width(gx5b, bal_full_path, W_half, color="black", alpha=0.14)
        if takeoff_transition_profile is not None and np.size(takeoff_transition_profile) > 0:
            tp = np.asarray(takeoff_transition_profile, dtype=float)
            gx5b.plot(tp[:, 1], tp[:, 0], "-", color="dodgerblue", linewidth=1.5,
                      transform=ccrs.Geodetic(), zorder=7)
        if landing_transition_profile_desc is not None and np.size(landing_transition_profile_desc) > 0:
            lp = np.asarray(landing_transition_profile_desc, dtype=float)
            gx5b.plot(lp[:, 1], lp[:, 0], "-", color="seagreen", linewidth=1.5,
                      transform=ccrs.Geodetic(), zorder=7)
        for seg in rf_bal["segments"]:
            pts = seg["points"]
            if seg["type"] == "TF":
                gx5b.plot(pts[:, 1], pts[:, 0], "-", color="black", linewidth=1.8,
                          transform=ccrs.Geodetic(), zorder=8)
            else:
                gx5b.plot(pts[:, 1], pts[:, 0], "-", color="black", linewidth=2.8,
                          transform=ccrs.Geodetic(), zorder=9)
                gx5b.scatter(pts[0, 1], pts[0, 0], s=40, c="yellow", marker=">",
                             edgecolors="k", linewidths=0.6, transform=ccrs.Geodetic(), zorder=10)
                gx5b.scatter(pts[-1, 1], pts[-1, 0], s=40, c="yellow", marker="s",
                             edgecolors="k", linewidths=0.6, transform=ccrs.Geodetic(), zorder=10)
                ac = seg["arc_center"]
                gx5b.scatter(ac[1], ac[0], s=50, c="white", marker="x",
                             linewidths=1.5, transform=ccrs.Geodetic(), zorder=10)

        _add_rf_legend_handles(
            gx5b, "black", "Balanced", tf_lw=1.8, rf_lw=2.8,
            include_transition=True, transition_lw=1.5
        )
        _add_arc_marker_legend_handles(gx5b)

        gx5b.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, framealpha=0.9)
        fig5b.savefig(out_dir / "fig5b_balanced_fig4_style.png", dpi=150, bbox_inches="tight")
        print(f"Saved {out_dir / 'fig5b_balanced_fig4_style.png'}")
        plt.close(fig5b)

        fig5c = plt.figure("Figure 5C: Balanced Corridor + CR/RFC Labels", figsize=(14, 10))
        agl_value = float(altitude_levels[0] - start_vertiport[2])
        title_str = f"Balanced Corridor with CR/RFC Labels\nAltitude: {float(altitude_levels[0]):.1f}m MSL (= {agl_value:.1f}m AGL)"
        gx5c = setup_corridor_axes_fn(fig5c, title_str, with_moc=True)
        plot_standard_key_markers_fn(
            gx5c, include_waypoints=True, include_backbone=True,
            takeoff_label=("Takeoff_End" if use_takeoff_landing_transition else "OffMode_Start"),
            landing_label=("Landing_Start" if use_takeoff_landing_transition else "OffMode_End"),
            backbone_lw=1.2, zorder=10
        )
        plot_corridor_width(gx5c, bal_full_path, W_half, color="black", alpha=0.14)
        for seg in rf_bal["segments"]:
            pts = np.asarray(seg["points"], dtype=float)
            if pts.size == 0:
                continue
            if seg["type"] == "TF":
                gx5c.plot(pts[:, 1], pts[:, 0], "-", color="black", linewidth=1.8,
                          transform=ccrs.Geodetic(), zorder=8)
            else:
                gx5c.plot(pts[:, 1], pts[:, 0], "-", color="black", linewidth=2.8,
                          transform=ccrs.Geodetic(), zorder=9)

        cr_pts, cr_labels = _collect_cr_points_from_segments(rf_bal["segments"])
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
    # ==================== Core Parameters ====================
    W_half = 296.0                   # TSE=148 (m), W_half = TSE*2 (m)
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

    N_init = 1000    # target initial candidates before feasibility filtering
    min_feasible_init_solutions = 1  # minimum feasible candidates required to start evolution
    N_pop = 50      # population size
    Nmax = 2       # number of generations
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
    max_extra_nodes_per_seg = 1     # 세그먼트별 최대 extra node 수 (int 또는 list)
    use_wp_skip_generator = False   # True: WP-skip 초기해 생성기 혼용
    init_pop_skip_mix_ratio = 0.5   # skip 생성기 혼용 비율(0~1)
    wp_skip_prob = 0.00             # 중간 WP skip 확률 (0~1)
    airspace_radius_km = 5.0        # 공역 반경 제한 (km)
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
    look_ahead_window = 2          # number of neighbor segments per side for look-ahead
    rf_allow_tangent_clamp = True  # allow geometric tangent clamping to fit short segments
    rf_corner_fit_margin = 0.95    # maximum usable fraction of adjacent segment lengths
    rf_corner_min_tangent_m = 1.0  # minimum tangent distance for corner construction
    rf_min_turn_angle_deg = 0.5    # angles below this are treated as straight
    max_init_retries = 300         # max retries for feasible initial-population search
    global RF_ALLOW_TANGENT_CLAMP, RF_CORNER_FIT_MARGIN, RF_CORNER_MIN_TANGENT_M, RF_MIN_TURN_ANGLE_DEG
    RF_ALLOW_TANGENT_CLAMP = bool(rf_allow_tangent_clamp)
    RF_CORNER_FIT_MARGIN = float(rf_corner_fit_margin)
    RF_CORNER_MIN_TANGENT_M = float(rf_corner_min_tangent_m)
    RF_MIN_TURN_ANGLE_DEG = float(rf_min_turn_angle_deg)

    look_ahead_min_equiv_speed_mps = ground_speed_mps * np.sqrt(look_ahead_min_scale)
    look_ahead_min_equiv_speed_kmh = look_ahead_min_equiv_speed_mps * 3.6
    look_ahead_min_turn_radius_m = rf_base_turn_radius_m * look_ahead_min_scale

    w_dist, w_ground, w_air, w_noise = 0.1, 1.0, 2.0, 0.1
    altitude_levels = np.array([750.0], dtype=float)  # 순항 고도(MSL, m)
    use_heading_map = True

    sector_mode_enabled = False  # True: season 마스크로 사용자 섹터 허용 여부 검사, False: 검사 없이 사용자 섹터 그대로 사용
    sector_season = "annual"    # 시즌 키 (annual, spring, summer, autumn, winter). True일 때 허용 섹터 판정에 사용
    takeoff_sector_user = 7     # 이륙 섹터 번호 (1~12, 1=북쪽 시작, 시계방향)
    landing_sector_user = 11     # 착륙 섹터 번호 (1~12, 1=북쪽 시작, 시계방향)
    sector_half_width_deg = 15.0    # 플롯 wedge 반폭(deg): 섹터 중심 기준 ±각도

    takeoff_heading_deg = float(np.rad2deg(_sector_angle(int(takeoff_sector_user))))
    landing_heading_deg = float(np.rad2deg(_sector_angle(int(landing_sector_user))))
    use_takeoff_landing_transition = True   # 이착륙 전환 경로 사용 여부,  False이면 사용자가 지정한 takeoff_end_lla를 그대로 사용, 

    if bool(sector_mode_enabled):
        sector_season = normalize_season(sector_season)
        takeoff_sector_user = validate_sector_1based(takeoff_sector_user, label="takeoff_sector_user")
        landing_sector_user = validate_sector_1based(landing_sector_user, label="landing_sector_user")
        season_takeoff_mask, season_landing_mask = get_season_masks(sector_season)

        if not sector_allowed(season_takeoff_mask, takeoff_sector_user):
            raise ValueError(
                f"Takeoff sector {takeoff_sector_user} is not allowed for season '{sector_season}'. "
                f"Allowed takeoff sectors={np.where(season_takeoff_mask)[0] + 1}"
            )
        if not sector_allowed(season_landing_mask, landing_sector_user):
            raise ValueError(
                f"Landing sector {landing_sector_user} is not allowed for season '{sector_season}'. "
                f"Allowed landing sectors={np.where(season_landing_mask)[0] + 1}"
            )

        takeoff_heading_deg = float(np.rad2deg(_sector_angle(takeoff_sector_user)))
        landing_heading_deg = float(np.rad2deg(_sector_angle(landing_sector_user)))
    else:
        sector_season = normalize_season(sector_season)
        season_takeoff_mask, season_landing_mask = get_season_masks(sector_season)

    transition_max_climb_angle_deg = 15.4
    transition_max_descent_angle_deg = 15.4
    transition_max_climb_rate_mps = 13.75
    transition_max_descent_rate_mps = 13.75
    transition_horiz_speed_takeoff_mps = 50.0
    transition_horiz_speed_landing_mps = 50.0

    transition_takeoff_distance_m = None
    transition_landing_distance_m = None
    transition_strict_distance_check = False

    transition_sample_spacing_m = 50.0

    use_clicked_waypoints = True     # 클릭 기반 WP 입력 ON/OFF
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
    airspace_radius_m = float(airspace_radius_km) * 1000.0
    airspace_alt_min_m = 100.0  # 공역 최소 고도(MSL, m)
    airspace_alt_max_m = 1000.0  # 공역 최대 고도(MSL, m)
    min_corridor_distance_m = float(min_corridor_distance_km) * 1000.0

    noise_npy_path = Path("noise_data") / "noise_lden_grid.npy"
    noise_floor_db = 0.0

    #

    ground_risk_path = Path("Modified_high_res_affected_population_GRC.npy")

    bird_airrisk_path = Path("air_risk_data") / "bird_riskmap_springfall_3d.npy"
    moc_airrisk_path = Path("air_risk_data") / "UAM_MOC_3D_Risk_Map.npy"

    import datetime as _dt
    _run_ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / _run_ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # start_vertiport_default = np.array([35.6033361, 129.0776917, 150.0], dtype=float) # 26년 5월 28일 변경전 버티포트 좌표
    # end_vertiport_default = np.array([35.6033361, 129.0776917, 150.0], dtype=float)   # 26년 5월 28일 변경전 버티포트 좌표
    start_vertiport_default = np.array([35.603386, 129.078025, 150.0], dtype=float) # 변경후 버티포트 좌표
    end_vertiport_default = np.array([35.603386, 129.078025, 150.0], dtype=float)   # 변경후 버티포트 좌표
    takeoff_end_lla = np.array([35.59389287, 129.07489650, float(altitude_levels[0])], dtype=float) # 이륙 끝 지점, 고도를 순항 고도와 일치하도록 설정
    landing_end_lla = np.array([35.61033545, 129.06947779, float(altitude_levels[0])], dtype=float) #  착륙 끝 지점, 고도를 순항 고도와 일치하도록 설정
    start_ref_alt_m = float(start_vertiport_default[2])

    params_dict = {
        "run_timestamp": _run_ts,
        "WP_CLICK_MODE_ENV": _CLICK_MODE_ENV,
        "USE_INTERACTIVE_BACKEND": bool(USE_INTERACTIVE_BACKEND),
        "W_half": W_half,
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
        "rf_allow_tangent_clamp": bool(rf_allow_tangent_clamp),
        "rf_corner_fit_margin": float(rf_corner_fit_margin),
        "rf_corner_min_tangent_m": float(rf_corner_min_tangent_m),
        "rf_min_turn_angle_deg": float(rf_min_turn_angle_deg),
        "max_init_retries": max_init_retries,
        "w_dist": w_dist,
        "w_ground": w_ground,
        "w_air": w_air,
        "w_noise": w_noise,
        "sector_mode_enabled": bool(sector_mode_enabled),
        "sector_season": str(sector_season),
        "takeoff_sector_user": int(takeoff_sector_user),
        "landing_sector_user": int(landing_sector_user),
        "sector_half_width_deg": float(sector_half_width_deg),
        "season_takeoff_mask_12": [int(v) for v in season_takeoff_mask.astype(int).tolist()],
        "season_landing_mask_12": [int(v) for v in season_landing_mask.astype(int).tolist()],
        "takeoff_heading_deg": float(takeoff_heading_deg),
        "landing_heading_deg": float(landing_heading_deg),
        "use_takeoff_landing_transition": bool(use_takeoff_landing_transition),
        "takeoff_end_lla": [float(v) for v in takeoff_end_lla.tolist()],
        "landing_end_lla": [float(v) for v in landing_end_lla.tolist()],
        "transition_max_climb_angle_deg": float(transition_max_climb_angle_deg),
        "transition_max_descent_angle_deg": float(transition_max_descent_angle_deg),
        "transition_max_climb_rate_mps": float(transition_max_climb_rate_mps),
        "transition_max_descent_rate_mps": float(transition_max_descent_rate_mps),
        "transition_horiz_speed_takeoff_mps": float(transition_horiz_speed_takeoff_mps),
        "transition_horiz_speed_landing_mps": float(transition_horiz_speed_landing_mps),
        "transition_takeoff_distance_m": (None if transition_takeoff_distance_m is None else float(transition_takeoff_distance_m)),
        "transition_landing_distance_m": (None if transition_landing_distance_m is None else float(transition_landing_distance_m)),
        "transition_strict_distance_check": bool(transition_strict_distance_check),
        "transition_sample_spacing_m": float(transition_sample_spacing_m),
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
        "moc_airrisk_path": str(moc_airrisk_path),
    }
    with open(out_dir / "params.json", "w", encoding="utf-8") as _pf:
        json.dump(params_dict, _pf, indent=2, ensure_ascii=False)
    print(f"Output folder : {out_dir}")

    pop_risk_raw = np.load(str(ground_risk_path), allow_pickle=True)
    selected = pop_risk_raw[:, :, 0, 3:]
    Ny, Nx, H_time = selected.shape

    A = len(altitude_levels)
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

    AirRisk = np.zeros((Ny, Nx, len(altitude_levels)), dtype=float)
    for i, alt in enumerate(altitude_levels):  # alt: MSL
        src_idx = int(np.argmin(np.abs(bird_z_vec - float(alt))))  # bird_z_vec: MSL
        AirRisk[:, :, i] = bird_3d[:, :, src_idx]

    moc_raw = np.load(str(moc_airrisk_path), allow_pickle=True).item()
    moc_z_vec = np.asarray(
        moc_raw["z_vec"] if "z_vec" in moc_raw else moc_raw["altitude_vec"],
        dtype=float,
    ).ravel()
    moc_3d = _align_to_ny_nx_z(np.asarray(moc_raw["Risk_3d"], dtype=float), "MOCRisk")
    MOCRisk = np.zeros((Ny, Nx, len(altitude_levels)), dtype=np.uint8)
    for i, alt in enumerate(altitude_levels):  # alt: MSL
        src_idx = int(np.argmin(np.abs(moc_z_vec - float(alt))))  # moc_z_vec: MSL
        MOCRisk[:, :, i] = (moc_3d[:, :, src_idx] >= 0.5).astype(np.uint8)
    moc_plot_2d = np.max(MOCRisk, axis=2).astype(float)

    print(
        f"Loaded bird air risk map: shape={bird_3d.shape}, "
        f"source_altitudes={bird_z_vec.tolist()}"
    )
    print(
        f"Loaded MOC binary map: shape={moc_3d.shape}, "
        f"ones_ratio={float(np.mean(MOCRisk)):.4f}"
    )

    params_dict.update({
        "bird_airrisk_meta": {
            "path": str(bird_airrisk_path),
            "source_altitudes_m": [float(v) for v in bird_z_vec.tolist()],
            "global_min": float(np.min(AirRisk)),
            "global_max": float(np.max(AirRisk)),
        },
        "moc_meta": {
            "path": str(moc_airrisk_path),
            "source_altitudes_m": [float(v) for v in moc_z_vec.tolist()],
            "is_binary": True,
            "ones_ratio_selected_altitudes": float(np.mean(MOCRisk)),
        },
    })

    noise_3d_norm, noise_3d_db_after_floor, noise_meta = load_noise_risk_from_npy(
        npy_path=noise_npy_path,
        Ny=Ny,
        Nx=Nx,
        altitude_levels=altitude_levels,
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
    airspace_center_lla = np.array([35.6033361, 129.0776917, 150.0], dtype=float)

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
    lat_lim = [35.535, 35.652]
    lon_lim = [129.020, 129.150]
    request = cimgt.OSM()

    # corridor_lat_default = np.array([], dtype=float)
    # corridor_lon_default = np.array([], dtype=float)

    # msl 750일때, agl = 600 일때 위경도값, 이거 사용시 고도를 750으로 고정해야 함
    # corridor_lat_default = np.array([
    #     35.62436036,
    #     35.60999583,
    #     35.58276455,
    #     35.56839255,
    #     35.56962453,
    #     35.59453397,
    #     35.60000754,
    #     35.57797417,
    #     35.58974430,
    #     35.61806764,
    #     35.63420883,
    # ], dtype=float)

    # msl 750일때, agl = 600 일때 위경도값, 이거 사용시 고도를 750으로 고정해야 함
    # corridor_lon_default = np.array([
    #     129.11000284,
    #     129.12767299,
    #     129.12228780,
    #     129.10074704,
    #     129.08324518,
    #     129.11740747,
    #     129.10663709,
    #     129.07836485,
    #     129.06523845,
    #     129.04689515,
    #     129.05564608,
    # ], dtype=float)



    # WP set (alt=450m), msl 450일때, agl = 300 일때 위경도값, 이거 사용시 고도를 450으로 고정해야 함
    # corridor_lat_default = np.array([
    #     35.60387148, 35.62078195, 35.61077414, 35.58264205, 35.56883127,
    #     35.57090304, 35.59386156, 35.59972959, 35.58039796, 35.59092738,
    #     35.59834892, 35.62285237, 35.61888401,
    # ], dtype=float)
    # corridor_lon_default = np.array([
    #     129.09769647, 129.11467664, 129.12719952, 129.12168097, 129.10109250,
    #     129.08432458, 129.11807268, 129.10724782, 129.08007954, 129.07307521,
    #     129.06458513, 129.06373612, 129.07328747,
    # ], dtype=float)

    # WP set (alt=600m), msl 600일때, agl = 450 일때 위경도값, 이거 사용시 고도를 600으로 고정해야 함
    corridor_lat_default = np.array([
        35.5802140, 35.5657046, 35.5902048, 35.6016994, 35.5941733, 35.5714539,
        35.5889731, 35.6231791, 35.6011521, 35.6215376, 35.6268725, 35.6183911,
    ], dtype=float)
    corridor_lon_default = np.array([
        129.0763421, 129.0861028, 129.0950220, 129.1076436, 129.1150482, 129.1015852,
        129.1263234, 129.1197602, 129.0928343, 129.0756690, 129.0551379, 129.0440310,
    ], dtype=float)

    waypoint_alt_fixed_m = float(altitude_levels[0])  # 중간 경유 WP 고정 고도(MSL, m)
    clicked_wp_json_path = None
    clicked_wp_csv_path = None

    corridor_lat = corridor_lat_default.copy()
    corridor_lon = corridor_lon_default.copy()

    if use_takeoff_landing_transition:
        preview_takeoff, _, takeoff_transition_profile, takeoff_transition_meta = build_transition_profile_by_constraints(
            start_vertiport,
            target_alt_m=waypoint_alt_fixed_m,
            heading_deg=takeoff_heading_deg,
            max_angle_deg=transition_max_climb_angle_deg,
            max_vertical_rate_mps=transition_max_climb_rate_mps,
            horiz_speed_mps=transition_horiz_speed_takeoff_mps,
            user_distance_m=transition_takeoff_distance_m,
            strict_distance_check=transition_strict_distance_check,
            sample_spacing_m=transition_sample_spacing_m,
            mode_label="takeoff",
        )
        preview_landing, _, landing_transition_profile, landing_transition_meta = build_transition_profile_by_constraints(
            end_vertiport,
            target_alt_m=waypoint_alt_fixed_m,
            heading_deg=landing_heading_deg,
            max_angle_deg=transition_max_descent_angle_deg,
            max_vertical_rate_mps=transition_max_descent_rate_mps,
            horiz_speed_mps=transition_horiz_speed_landing_mps,
            user_distance_m=transition_landing_distance_m,
            strict_distance_check=transition_strict_distance_check,
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
            "heading_deg": float("nan"),
            "L_A_m": 0.0,
            "L_B_m": 0.0,
            "L_required_m": 0.0,
            "L_used_m": 0.0,
            "selected_basis": "off_mode_direct_start_end",
            "sample_spacing_m": float(transition_sample_spacing_m),
        }
        landing_transition_meta = {
            "heading_deg": float("nan"),
            "L_A_m": 0.0,
            "L_B_m": 0.0,
            "L_required_m": 0.0,
            "L_used_m": 0.0,
            "selected_basis": "off_mode_direct_start_end",
            "sample_spacing_m": float(transition_sample_spacing_m),
        }

    print(
        f"Applied takeoff endpoint: lat={preview_takeoff[0]:.8f}, "
        f"lon={preview_takeoff[1]:.8f}, alt={preview_takeoff[2]:.1f}m"
    )
    print(
        f"Applied landing endpoint: lat={preview_landing[0]:.8f}, "
        f"lon={preview_landing[1]:.8f}, alt={preview_landing[2]:.1f}m"
    )

    global TAKEOFF_TRANSITION_PROFILE, LANDING_TRANSITION_PROFILE_DESC
    TAKEOFF_TRANSITION_PROFILE = np.asarray(takeoff_transition_profile, dtype=float).copy() if takeoff_transition_profile is not None else np.empty((0, 3), dtype=float)
    LANDING_TRANSITION_PROFILE_DESC = np.asarray(landing_transition_profile_desc, dtype=float).copy() if landing_transition_profile_desc is not None else np.empty((0, 3), dtype=float)

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

    forbidden_zones_input = np.array([
        # [129.12, 129.13, 35.59, 35.60],
    ], dtype=float)

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
                map_zoom=clicked_wp_map_zoom,
                takeoff_complete=preview_takeoff,
                landing_entry=preview_landing,
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
        "takeoff_sector": 6,
        "landing_sector": 11,
        "takeoff_angle_deg": float(transition_max_climb_angle_deg),
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
            "heading_deg": float(takeoff_transition_meta["heading_deg"]),
            "L_A_m": float(takeoff_transition_meta["L_A_m"]),
            "L_B_m": float(takeoff_transition_meta["L_B_m"]),
            "L_required_m": float(takeoff_transition_meta["L_required_m"]),
            "L_used_m": float(takeoff_transition_meta["L_used_m"]),
            "selected_basis": str(takeoff_transition_meta["selected_basis"]),
            "sample_spacing_m": float(takeoff_transition_meta["sample_spacing_m"]),
            "sample_count": int(takeoff_transition_profile.shape[0]),
        },
        "landing_transition_meta": {
            "heading_deg": float(landing_transition_meta["heading_deg"]),
            "L_A_m": float(landing_transition_meta["L_A_m"]),
            "L_B_m": float(landing_transition_meta["L_B_m"]),
            "L_required_m": float(landing_transition_meta["L_required_m"]),
            "L_used_m": float(landing_transition_meta["L_used_m"]),
            "selected_basis": str(landing_transition_meta["selected_basis"]),
            "sample_spacing_m": float(landing_transition_meta["sample_spacing_m"]),
            "sample_count": int(landing_transition_profile.shape[0]),
        },
    })
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
        ai = np.argmin(np.abs(nodes_seg[:, 2][:, None] - altitude_levels[None, :]), axis=1)
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
        ai = np.argmin(np.abs(nodes_seg[:, 2][:, None] - altitude_levels[None, :]), axis=1)
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
            ai_s = np.argmin(np.abs(s[:, 2][:, None] - altitude_levels[None, :]), axis=1)
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
            ai_s = np.argmin(np.abs(s_pct[:, 2][:, None] - altitude_levels[None, :]), axis=1)
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
    rf_corridor_start = np.asarray(takeoff_complete if not use_takeoff_landing_transition else start_vertiport, dtype=float)
    rf_corridor_end = np.asarray(landing_entry if not use_takeoff_landing_transition else end_vertiport, dtype=float)
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
    )
    _eval_with_reason_for_init = partial(
        evaluate_objectives_with_constraints_gp,
        Norm_RT=Norm_RT,
        AirRisk=AirRisk,
        use_heading_map=use_heading_map,
        flight_dist_limit=flight_dist_limit,
        forbidden_zones=forbidden_zones,
        delta_z_max=delta_z_max,
        altitude_levels=altitude_levels,
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
        vertiport=None,
        landing_entry=None,
        takeoff_complete=None,
        return_reason=True,
    )

    init_pop = None
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
        )
        _rf_cnt = int(_init_eval["rf_cnt"])
        _rf_no_clamp_cnt = int(_init_eval["rf_no_clamp_cnt"])
        _both_cnt = int(_init_eval["both_cnt"])
        _cst_cnt = int(_init_eval["cst_cnt"])
        _air_cnt = int(_init_eval["air_cnt"])
        _dist_cnt = int(_init_eval["dist_cnt"])
        _reason_counts = dict(_init_eval["reason_counts"])
        _feasible_init = list(_init_eval["feasible_init"])

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

    bb_full = np.vstack([start_vertiport, backbone, end_vertiport]) if use_takeoff_landing_transition else np.asarray(backbone, dtype=float)
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
        gx.set_title(title)
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
                                   takeoff_label="Takeoff", landing_label="Landing",
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
        _takeoff_label = takeoff_label if use_takeoff_landing_transition else "OffMode_Start"
        _landing_label = landing_label if use_takeoff_landing_transition else "OffMode_End"
        gx.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90, c="blue",
                   marker="^", transform=ccrs.Geodetic(), label=_takeoff_label, zorder=zorder)
        gx.scatter([landing_entry[1]], [landing_entry[0]], s=90, c="green",
                   marker="v", transform=ccrs.Geodetic(), label=_landing_label, zorder=zorder)

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
        gx.set_title(title_text)
        draw_vertiport_radius_rings(gx, airspace_center_lla, radii_m=(airspace_radius_m,))
        plot_forbidden_zones(gx, forbidden_zones, face_alpha=0.10, edge_alpha=0.80)
        _plot_selected_sector_wedges(gx)

        gx.plot(bb_full[:, 1], bb_full[:, 0], "r--", linewidth=2, transform=ccrs.Geodetic(),
                label="Backbone", zorder=5)
        gx.scatter(waypoints[:, 1], waypoints[:, 0], s=60, c="orange", edgecolors="k",
                   linewidths=0.5, marker="o", transform=ccrs.Geodetic(), label="Waypoints (WP)", zorder=6)
        gx.scatter([start_vertiport[1]], [start_vertiport[0]], s=120, c="red", edgecolors="k",
                   marker="s", transform=ccrs.Geodetic(), label="Start Vertiport", zorder=7)
        gx.scatter([end_vertiport[1]], [end_vertiport[0]], s=120, c="crimson", edgecolors="k",
                   marker="D", transform=ccrs.Geodetic(), label="End Vertiport", zorder=7)
        gx.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90, c="blue",
                   marker="^", transform=ccrs.Geodetic(), label="Takeoff", zorder=7)
        gx.scatter([landing_entry[1]], [landing_entry[0]], s=90, c="green",
                   marker="v", transform=ccrs.Geodetic(), label="Landing", zorder=7)

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
    )
    eval_cfg = dict(
        Norm_RT=Norm_RT,
        AirRisk=AirRisk,
        use_heading_map=use_heading_map,
        flight_dist_limit=flight_dist_limit,
        forbidden_zones=forbidden_zones,
        delta_z_max=delta_z_max,
        altitude_levels=altitude_levels,
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
        vertiport=None,
        landing_entry=None,
        takeoff_complete=None,
    )
    eval_corridor_fn = partial(_evaluate_corridor_objectives_path, eval_cfg=eval_cfg)

    # ==================== [Fig 2] Sample Initial Solutions + RF Turn ====================

    n_sample = min(5, len(init_pop))
    fig2 = plt.figure("Figure 2: Sample Initial Solutions + RF Turn", figsize=(14, 10))
    gx2 = _setup_corridor_axes(fig2, f"Sample Initial Solutions ({n_sample}) with RF Turn")
    _plot_standard_key_markers(gx2, include_waypoints=False, include_backbone=True, zorder=8)
    colors_sample = plt.cm.tab10(np.linspace(0, 1, n_sample))
    for si in range(n_sample):
        rf = rf_apply_fn(init_pop[si])
        rf_path = rf["path"]
        gx2.plot(rf_path[:, 1], rf_path[:, 0], "-", color=colors_sample[si], linewidth=1.2,
                 transform=ccrs.Geodetic(), label=f"Sol {si+1}", zorder=5)
    gx2.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, framealpha=0.9)
    fig2.savefig(out_dir / "fig2_sample_init.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_dir / 'fig2_sample_init.png'}")
    plt.close(fig2)

    # ==================== [Fig 2B] Before vs After RF on Same Initial Paths ====================
    fig2b = plt.figure("Figure 2B: Initial Solutions Before vs After RF", figsize=(14, 10))
    gx2b = _setup_corridor_axes(fig2b, f"Same Initial Solutions: Before RF (dashed) vs After RF (solid), n={n_sample}")
    _plot_standard_key_markers(gx2b, include_waypoints=False, include_backbone=True, zorder=8)

    for si in range(n_sample):
        col = colors_sample[si]
        if use_takeoff_landing_transition:
            path_before = np.vstack([start_vertiport, init_pop[si], end_vertiport]).astype(float)
        else:
            path_before = np.vstack([takeoff_complete, init_pop[si], landing_entry]).astype(float)
        rf = rf_apply_fn(init_pop[si])
        path_after = rf["path"]

        gx2b.plot(
            path_before[:, 1], path_before[:, 0], "--", color=col, linewidth=1.1,
            transform=ccrs.Geodetic(), zorder=5,
            label=(f"Sol {si+1} Before RF" if si == 0 else None),
        )
        gx2b.scatter(
            path_before[:, 1], path_before[:, 0], s=18, color=col, marker="o",
            edgecolors="k", linewidths=0.3, transform=ccrs.Geodetic(), zorder=6,
        )

        gx2b.plot(
            path_after[:, 1], path_after[:, 0], "-", color=col, linewidth=1.7,
            transform=ccrs.Geodetic(), zorder=7,
            label=(f"Sol {si+1} After RF" if si == 0 else None),
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
        alt=altitude_levels, cs=cell_size, scales=refine_scales,
        air_thr=air_thr_global, dz=delta_z_max,
        w_d=w_dist, w_g=w_ground, w_a=w_air,
        lat_lim=lat_lim, lon_lim=lon_lim,
        NoiseRisk=NoiseRisk, noise_floor_db=noise_floor_db, w_n=w_noise,
        ground_speed_mps=ground_speed_mps,
        bank_angle_deg=bank_angle_deg,
        num_arc_points=num_arc_points,
        look_ahead=look_ahead,
        look_ahead_threshold_m=look_ahead_threshold_m,
        look_ahead_min_scale=look_ahead_min_scale,
        look_ahead_window=look_ahead_window,
        W_half=W_half, check_corridor_nfz=check_corridor_nfz, check_corridor_moc=check_corridor_moc,
        MOCRisk=MOCRisk,
        start_vertiport=start_vertiport,
        end_vertiport=end_vertiport,
        landing_entry=landing_entry,
        takeoff_complete=takeoff_complete,
        airspace_center_latlon=airspace_center_lla[:2],
        airspace_radius_m=airspace_radius_m,
        airspace_alt_min_m=airspace_alt_min_m,
        airspace_alt_max_m=airspace_alt_max_m,
        min_corridor_distance_m=min_corridor_distance_m,
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
        objective_names=objective_names,
        apply_rf_corridor_fn=rf_apply_fn,
        W_half=W_half,
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
    )
    print(f"Final feasible (constraints): {feasible_count}/{len(pop)}")
    print(f"RF geometric no-clamp: {rf_no_clamp_count}/{len(pop)}")

    if feasible_count == 0:
        print("No feasible solution. Retrying ...")
        return False, 0

    reps = pick_representatives(pop, fvals) if pop and fvals.size > 0 else []

    obj_pairs = [(i, j) for i in range(len(objective_names)) for j in range(i + 1, len(objective_names))]
    n_pair = len(obj_pairs)
    n_cols = 3
    n_rows = int(np.ceil(n_pair / n_cols))
    fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    fig3.suptitle("Pareto Front  (blue=feasible, gray=infeasible)", fontsize=13)
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
                f_rep, _ = eval_corridor_fn(rf_rep["path"])
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

    f_initial_backbone, _ = eval_corridor_fn(bb_full)

    init_rep_objectives = None
    if init_pop:
        init_fvals = np.zeros((len(init_pop), len(objective_names)), dtype=float)
        for i_init, p_init in enumerate(init_pop):
            rf_init = rf_apply_fn(p_init)
            f_init_vec, _ = eval_corridor_fn(rf_init["path"])
            init_fvals[i_init, :] = np.asarray(f_init_vec, dtype=float)

        init_rep_objectives = [
            np.asarray(init_fvals[int(np.argmin(init_fvals[:, oi]))], dtype=float)
            for oi in range(init_fvals.shape[1])
        ]
        fronts_init = fast_non_dominated_sort(init_fvals)
        if fronts_init and fronts_init[0]:
            f1_init = np.array(fronts_init[0], dtype=int)
            nf_init = normalize_objectives(init_fvals[f1_init])
            bal_init = int(np.argmin(np.linalg.norm(nf_init, axis=1)))
            init_rep_objectives.append(np.asarray(init_fvals[int(f1_init[bal_init])], dtype=float))
        elif init_rep_objectives:
            init_rep_objectives.append(np.asarray(init_rep_objectives[0], dtype=float))

    _plot_representative_corridor_figures(
        reps=reps,
        apply_rf_corridor_fn=rf_apply_fn,
        eval_corridor_objectives_fn=eval_corridor_fn,
        objective_names=objective_names,
        altitude_levels=altitude_levels,
        start_vertiport=start_vertiport,
        W_half=W_half,
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
        takeoff_transition_profile=takeoff_transition_profile,
        landing_transition_profile_desc=landing_transition_profile_desc,
        use_takeoff_landing_transition=use_takeoff_landing_transition,
        setup_corridor_axes_fn=_setup_corridor_axes,
        plot_standard_key_markers_fn=_plot_standard_key_markers,
    )

    best_rep = reps[-1] if reps else (pop[0] if pop else None)
    if best_rep is not None:
        rf_best = rf_apply_fn(best_rep)
        R_turn = rf_best["turn_radius_m"]
        segs_best = rf_best["segments"]
        g_mps2_local = 9.80665
        phi_local = np.deg2rad(bank_angle_deg)

        rows = []
        point_idx = 0

        def _is_same_point(p1, p2, tol_m=0.5):
            return _seg_dist_3d_m(np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)) <= float(tol_m)

        last_point = None

        def _append_vertiport_row(point, segment_label):
            nonlocal point_idx, last_point
            rows.append({
                "Point_No": point_idx, "Type": "Vertiport", "Segment": segment_label,
                "Lat": float(point[0]), "Lon": float(point[1]), "Altitude_MSL_m": float(point[2]),
                "Altitude_AGL_m": float(point[2] - start_vertiport[2]),
                "TF_Start": "", "TF_End": "", "RF_Start": "", "RF_End": "",
                "Arc_Center_Lat": "", "Arc_Center_Lon": "",
                "Turn_Radius_m": "", "Turn_Angle_deg": "", "LookAhead_Radius_Scale": "",
                "Ground_Speed_mps": 0.0, "Bank_Angle_deg": 0.0,
            })
            point_idx += 1
            last_point = np.asarray(point, dtype=float)

        def _append_transition(profile, segment_label, type_label, speed_mps, skip_first=False, skip_last=False):
            nonlocal point_idx, last_point
            prof = np.asarray(profile, dtype=float)
            if prof.ndim == 1:
                prof = prof.reshape(1, -1)
            if prof.size == 0:
                return
            start_idx = 1 if skip_first else 0
            end_idx = prof.shape[0] - 1 if skip_last else prof.shape[0]
            if end_idx <= start_idx:
                return
            for pi in range(start_idx, end_idx):
                cur_pt = np.asarray(prof[pi], dtype=float)
                if last_point is not None and _is_same_point(cur_pt, last_point, tol_m=0.5):
                    if rows and pi == start_idx:
                        rows[-1]["TF_Start"] = "O"
                    continue
                is_start = "O" if pi == start_idx else ""
                is_end = "O" if pi == end_idx - 1 else ""
                rows.append({
                    "Point_No": point_idx, "Type": type_label, "Segment": segment_label,
                    "Lat": cur_pt[0], "Lon": cur_pt[1], "Altitude_MSL_m": cur_pt[2],
                    "Altitude_AGL_m": float(cur_pt[2] - start_vertiport[2]),
                    "TF_Start": is_start, "TF_End": is_end,
                    "RF_Start": "", "RF_End": "",
                    "Arc_Center_Lat": "", "Arc_Center_Lon": "",
                    "Turn_Radius_m": "", "Turn_Angle_deg": "",
                    "LookAhead_Radius_Scale": "", "Ground_Speed_mps": float(speed_mps),
                    "Bank_Angle_deg": 0.0,
                })
                point_idx += 1
                last_point = cur_pt

        _append_vertiport_row(start_vertiport, "Start")
        if use_takeoff_landing_transition:
            _append_transition(
                takeoff_transition_profile,
                "Takeoff_Transition",
                "Takeoff_TF_Point",
                transition_horiz_speed_takeoff_mps,
                skip_first=True,
                skip_last=False,
            )

        # Segments (RF only on core)
        seg_counter = 0
        for seg in segs_best:
            seg_counter += 1
            pts = seg["points"]
            stype = seg["type"]

            if stype == "TF":
                for pi in range(pts.shape[0]):
                    cur_pt = np.asarray(pts[pi], dtype=float)
                    if last_point is not None and _is_same_point(cur_pt, last_point, tol_m=0.5):
                        if rows and pi == 0:
                            rows[-1]["TF_Start"] = "O"
                        continue
                    is_start = "O" if pi == 0 else ""
                    is_end = "O" if pi == pts.shape[0] - 1 else ""
                    rows.append({
                        "Point_No": point_idx, "Type": "TF_Point", "Segment": f"Seg{seg_counter}",
                        "Lat": cur_pt[0], "Lon": cur_pt[1], "Altitude_MSL_m": cur_pt[2],
                        "Altitude_AGL_m": float(cur_pt[2] - start_vertiport[2]),
                        "TF_Start": is_start, "TF_End": is_end,
                        "RF_Start": "", "RF_End": "",
                        "Arc_Center_Lat": "", "Arc_Center_Lon": "",
                        "Turn_Radius_m": "", "Turn_Angle_deg": "",
                        "LookAhead_Radius_Scale": "", "Ground_Speed_mps": ground_speed_mps, "Bank_Angle_deg": 0.0,
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
                    if last_point is not None and _is_same_point(cur_pt, last_point, tol_m=0.5):
                        if rows and pi == 0:
                            rows[-1]["RF_Start"] = "O"
                        continue
                    is_start = "O" if pi == 0 else ""
                    is_end = "O" if pi == pts.shape[0] - 1 else ""
                    arc_label = f"Arc_{pi+1}/{pts.shape[0]}"
                    rows.append({
                        "Point_No": point_idx, "Type": f"RF_Arc ({arc_label})",
                        "Segment": f"Seg{seg_counter}",
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

        if use_takeoff_landing_transition:
            _append_transition(
                landing_transition_profile_desc,
                "Landing_Transition",
                "Landing_TF_Point",
                transition_horiz_speed_landing_mps,
                skip_first=False,
                skip_last=True,
            )
        _append_vertiport_row(end_vertiport, "End")

        _export_route_outputs(
            rows=rows,
            rf_best=rf_best,
            Norm_RT=Norm_RT,
            AirRisk=AirRisk,
            altitude_levels=altitude_levels,
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
                altitude_levels=altitude_levels,
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
                vertiport=None,
                landing_entry=None,
                takeoff_complete=None,
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
        "takeoff_complete": takeoff_complete,
        "landing_entry": landing_entry,
        "forbidden_zones": forbidden_zones,
        "emergency_points": emergency_points,
        "airspace_center_lla": airspace_center_lla,
        "airspace_radius_m": airspace_radius_m,
        "airspace_alt_min_m": airspace_alt_min_m,
        "airspace_alt_max_m": airspace_alt_max_m,
        "lat_lim": lat_lim,
        "lon_lim": lon_lim,
        "W_half": W_half,
        "ground_speed_mps": ground_speed_mps,
        "bank_angle_deg": bank_angle_deg,
        "bird_airrisk_path": str(bird_airrisk_path),
        "moc_airrisk_path": str(moc_airrisk_path),
        "check_corridor_moc": bool(check_corridor_moc),
        "noise_npy_path": str(noise_npy_path),
        "noise_floor_db": noise_floor_db,
        "w_noise": w_noise,
        "noise_meta": noise_meta,
        "noise_map_3d_normalized": noise_3d_norm,
        "noise_map_3d_db_after_floor": noise_3d_db_after_floor,
        "MOCRisk": MOCRisk,
    }

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



