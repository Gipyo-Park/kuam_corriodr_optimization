import sys
import os
import csv
import json
from pathlib import Path
import numpy as np

import numpy as np
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

# ──────────────────────────────────────────────────────────────────
# Utility: 경로 세그먼트 주변 strip 안의 노드 필터
# ──────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────
# Utility: 3D 그리드 노드 생성 (세그먼트 주변)
# ──────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────
# Utility: 이착륙 구간 생성
# ──────────────────────────────────────────────────────────────────
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


def build_full_corridor_path(start_vertiport, takeoff_complete, path_core, landing_entry, end_vertiport):
    return np.vstack([start_vertiport, takeoff_complete, path_core, landing_entry, end_vertiport]).astype(float)


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


# ──────────────────────────────────────────────────────────────────
# NSGA-III 선택 / 변이
# ──────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────
# 초기 해 생성 (v3 방식)
#   backbone WP 를 중심으로:
#   1) WP 를 반경 내 교란 (vertiport 제외)
#   2) 세그먼트 사이에 extra node 를 추가
#   3) emergency landing 이 세그먼트 근처이면 자동 삽입
#   4) t-projection 으로 순서 보장
# ──────────────────────────────────────────────────────────────────
def _seg_dist_m(a, b):
    """두 lat/lon 점 사이 수평 거리 (미터)"""
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
    backbone,                 # (K, 3) 고정 뼈대 WP
    wp_perturb_radius_m,      # WP 교란 반경  (m)
    min_extra_nodes_per_seg,  # int | list[int]  세그먼트별 최소 추가 노드 수
    max_extra_nodes_per_seg,  # int | list[int]  세그먼트별 최대 추가 노드 수
    safe_nodes_by_seg,        # list of ndarray — 세그먼트별 후보 노드
    emergency_points,         # (E, 3) 비상착륙장
    emergency_strip_m,        # emergency 포함 판별 strip 폭 (m)
    is_fixed,                 # (K,) bool  — True 이면 교란 안 함
    wp_perturb_steps=1,       # WP 교란 횟수 (반복 적용)
    min_seg_for_extra_nodes_m=2000.0,  # 이보다 짧은 세그먼트에는 extra node 추가 안 함
):
    K = backbone.shape[0]
    m_lat = 111000.0
    m_lon = 111000.0 * np.cos(np.deg2rad(float(np.mean(backbone[:, 0]))))
    r_lat = wp_perturb_radius_m / m_lat
    r_lon = wp_perturb_radius_m / m_lon
    n_perturb = max(1, int(wp_perturb_steps))
    step_r_lat = r_lat / n_perturb
    step_r_lon = r_lon / n_perturb

    # 1) WP 교란
    perturbed = backbone.copy()
    for i in range(K):
        if is_fixed[i]:
            continue
        for _ in range(n_perturb):
            ang = np.random.uniform(0, 2 * np.pi)
            d = np.sqrt(np.random.uniform()) * 1.0   # within unit circle
            perturbed[i, 0] += d * step_r_lat * np.sin(ang)
            perturbed[i, 1] += d * step_r_lon * np.cos(ang)

    # min/max → list
    if isinstance(min_extra_nodes_per_seg, int):
        mn_list = [min_extra_nodes_per_seg] * (K - 1)
    else:
        mn_list = list(min_extra_nodes_per_seg)
    if isinstance(max_extra_nodes_per_seg, int):
        mx_list = [max_extra_nodes_per_seg] * (K - 1)
    else:
        mx_list = list(max_extra_nodes_per_seg)

    # 2-3) 세그먼트별 extra node + emergency 삽입
    path_pts = [perturbed[0]]
    for k in range(K - 1):
        a = perturbed[k]
        b = perturbed[k + 1]
        inserts = []

        seg_m = _seg_dist_m(a, b)
        seg_long_enough = seg_m >= min_seg_for_extra_nodes_m

        # extra nodes: 세그먼트가 충분히 길어야만 추가
        if seg_long_enough:
            lo = mn_list[k] if k < len(mn_list) else 0
            hi = mx_list[k] if k < len(mx_list) else 0
            m = int(np.random.randint(lo, hi + 1)) if hi >= lo else 0
            cand = safe_nodes_by_seg[k] if k < len(safe_nodes_by_seg) else np.empty((0, 3))
            if m > 0 and cand.size > 0:
                idx = np.random.choice(cand.shape[0], size=min(m, cand.shape[0]), replace=False)
                inserts.extend(cand[idx].tolist())

        # emergency landing 자동 삽입 (거리 무관)
        if emergency_points is not None and emergency_points.size > 0:
            em_in_strip = filter_nodes_in_strip(a, b, emergency_points, emergency_strip_m,
                                                 end_buffer_ratio=-0.05)
            if em_in_strip.size > 0:
                inserts.extend(em_in_strip.tolist())

        # t-projection 순서 정렬
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


# ──────────────────────────────────────────────────────────────────
# 초기 해 생성 (WP 스킵 버전)
#   일부 WP를 건너뛸 수 있음 → RF turn 가능성 높임
# ──────────────────────────────────────────────────────────────────
def generate_single_initial_solution_with_skip(
    full_waypoints,           # (M, 3) 모든 WP
    takeoff_wp,               # (3,) 필수 시작점 (takeoff_complete)
    landing_wp,               # (3,) 필수 끝점 (landing_entry)
    wp_perturb_radius_m,      # WP 교란 반경
    min_extra_nodes_per_seg,  # int | list[int]
    max_extra_nodes_per_seg,  # int | list[int]
    safe_nodes_by_seg_full,   # list — full_waypoints 기준 safe nodes
    emergency_points,         # (E, 3)
    emergency_strip_m,        # float
    wp_perturb_steps=1,       # WP 교란 횟수 (반복 적용)
    wp_skip_prob=0.25,        # 각 중간 WP를 건너뛸 확률 (0~1)
    min_seg_for_extra_nodes_m=2000.0,  # 이보다 짧은 세그먼트에는 extra node 추가 안 함
):
    """
    WP 스킵 기능: 모든 WP를 거치는 대신, 각 WP마다 skip_prob 확률로 건너뜀.
    필수: takeoff_wp, landing_wp
    선택: full_waypoints[1:-1]
    """
    M = full_waypoints.shape[0]
    m_lat = 111000.0
    m_lon = 111000.0 * np.cos(np.deg2rad(float(np.mean(full_waypoints[:, 0]))))
    r_lat = wp_perturb_radius_m / m_lat
    r_lon = wp_perturb_radius_m / m_lon
    n_perturb = max(1, int(wp_perturb_steps))
    step_r_lat = r_lat / n_perturb
    step_r_lon = r_lon / n_perturb

    # WP 선택: full_waypoints의 첫/마지막은 항상 유지 + 중간 WP만 선택적으로 skip
    selected_indices = [0]
    for i in range(1, M - 1):
        if np.random.uniform() > wp_skip_prob:
            selected_indices.append(i)
    selected_indices.append(M - 1)

    # 최종 경로: takeoff -> (선택된 기존 WPs 전체, 첫/끝 포함) -> landing
    selected_waypoints = full_waypoints[selected_indices]
    selected_wps = np.vstack([takeoff_wp, selected_waypoints, landing_wp])
    K = selected_wps.shape[0]

    # 1) WP 교란 (단, 첫/마지막은 고정)
    perturbed = selected_wps.copy()
    for i in range(1, K - 1):
        for _ in range(n_perturb):
            ang = np.random.uniform(0, 2 * np.pi)
            d = np.sqrt(np.random.uniform())
            perturbed[i, 0] += d * step_r_lat * np.sin(ang)
            perturbed[i, 1] += d * step_r_lon * np.cos(ang)

    # min/max → list
    if isinstance(min_extra_nodes_per_seg, int):
        mn_list = [min_extra_nodes_per_seg] * (K - 1)
    else:
        mn_list = list(min_extra_nodes_per_seg)
    if isinstance(max_extra_nodes_per_seg, int):
        mx_list = [max_extra_nodes_per_seg] * (K - 1)
    else:
        mx_list = list(max_extra_nodes_per_seg)

    # 2-3) 세그먼트별 extra node + emergency
    path_pts = [perturbed[0]]
    for k in range(K - 1):
        a = perturbed[k]
        b = perturbed[k + 1]
        inserts = []

        seg_m = _seg_dist_m(a, b)
        seg_long_enough = seg_m >= min_seg_for_extra_nodes_m

        # extra nodes: 세그먼트가 충분히 길어야만 추가
        if seg_long_enough:
            lo = mn_list[k] if k < len(mn_list) else 0
            hi = mx_list[k] if k < len(mx_list) else 0
            m = int(np.random.randint(lo, hi + 1)) if hi >= lo else 0
            cand = safe_nodes_by_seg_full[k] if k < len(safe_nodes_by_seg_full) else np.empty((0, 3))
            if m > 0 and cand.size > 0:
                idx = np.random.choice(cand.shape[0], size=min(m, cand.shape[0]), replace=False)
                inserts.extend(cand[idx].tolist())

        # emergency (거리 무관)
        if emergency_points is not None and emergency_points.size > 0:
            em_in_strip = filter_nodes_in_strip(a, b, emergency_points, emergency_strip_m,
                                                 end_buffer_ratio=-0.05)
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


# ──────────────────────────────────────────────────────────────────
# Corridor Width 시각화 (Cartopy 용)
# ──────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────
# NSGA-III 실행  (v3 – RF turn 적용 후 evaluation)
# ──────────────────────────────────────────────────────────────────
def run_nsga3(
    nodes_pool, node_risk_pool, population, N_pop, Nmax, ratio,
    mutation_cfg,
    require_rf_for_parent_selection,
    mandatory_backbone,
    # evaluation 공통 인자
    Norm_RT, AirRisk, use_map, f_limit, f_zones,
    alt, cs, scales, air_thr, dz,
    w_d, w_g, w_a, lat_lim, lon_lim,
    # RF turn 파라미터
    ground_speed_mps, bank_angle_deg, num_arc_points,
    look_ahead, look_ahead_threshold_m, look_ahead_min_scale, look_ahead_window,
    # 회랑폭 + vertiport
    W_half, check_corridor_nfz,
    start_vertiport, end_vertiport, landing_entry, takeoff_complete,
    airspace_center_latlon, airspace_radius_m,
    airspace_alt_min_m, airspace_alt_max_m,
    min_corridor_distance_m,
):
    """NSGA-III with RF-turn preprocessing."""

    # --- 목적함수 수 파악 ---
    dummy = np.vstack([population[0][0], population[0][-1]])
    temp_f, _ = evaluate_objectives_with_constraints_gp(
        dummy, Norm_RT, AirRisk, use_map, f_limit, f_zones, dz, alt, cs, scales,
        air_thr, w_d, w_g, w_a, lat_lim, lon_lim,
        W_half=W_half, check_corridor_nfz=check_corridor_nfz,
        vertiport=None, landing_entry=None, takeoff_complete=None,
    )
    num_obj = len(temp_f)
    H = num_obj + 1
    ref_points = generate_reference_points(num_obj, H)

    def _evaluate_one(chromo):
        chromo = _enforce_mandatory_wp_order(chromo, mandatory_backbone)
        # RF turn 적용
        rf = apply_rf_turns(chromo, ground_speed_mps, bank_angle_deg, num_arc_points,
                            look_ahead=look_ahead,
                            look_ahead_threshold_m=look_ahead_threshold_m,
                            look_ahead_min_scale=look_ahead_min_scale,
                            look_ahead_window=look_ahead_window)
        rf_path = rf["path"]
        full_path = build_full_corridor_path(start_vertiport, takeoff_complete, rf_path, landing_entry, end_vertiport)
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
            W_half=W_half, check_corridor_nfz=check_corridor_nfz,
            vertiport=None, landing_entry=None, takeoff_complete=None,
        )
        if not rf["feasible"]:
            # RF clamping이 필요한 해는 불리하게 두되, 완전 배제하지는 않음
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
        rf_mask = (f_vals[:, 0] < 1e6)  # +1e6 패널티 없는 = RF 기하학적 feasible
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
            # 현재 세대가 전부 실패면 이전 세대의 성공 해(동일 selection 기준)를 전달
            new_pop = list(last_success_pop)
            carry_over_used = True
            print(f"[Gen {gen}] no parent-selectable solution in current generation; carrying over {len(new_pop)} previous feasible parent(s).")

        if not new_pop:
            # 시작 시점부터 성공해가 전혀 없는 비정상 케이스만 실패 처리
            return [], np.empty((0, num_obj)), gen_history

        # 세대별 스냅샷 저장용: 선택/전달된 부모 집단만 재평가 기록
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

    # 최종 평가
    if not pop:
        return [], np.empty((0, num_obj)), gen_history

    f_final = np.zeros((len(pop), num_obj), dtype=float)
    for i in range(len(pop)):
        f_final[i], _ = _evaluate_one(pop[i])

    return pop, f_final, gen_history


# ──────────────────────────────────────────────────────────────────
# 대표 해 선택
# ──────────────────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def attempt_run_once():
    # ==================== 파라미터 ====================
    W_half = 150.0                   # TSE (m)
    # RF turn 단위 정리:
    #   V: m/s, g: m/s^2, theta: rad, R: m
    #   R = V^2 / (g * tan(theta))
    # 속도 변환: 300 km/h = 300/3.6 = 83.333... m/s
    speed_max_kmh = 300.0
    ground_speed_mps = speed_max_kmh / 3.6
    bank_angle_deg = 25.0            # 최대 bank angle (deg)
    g_mps2 = 9.80665

    # 기준 RF turn 반경(look-ahead 스케일 적용 전)
    # 이 설정(300 km/h, 25 deg)에서는 약 1,518 m
    rf_base_turn_radius_m = (ground_speed_mps ** 2) / (g_mps2 * np.tan(np.deg2rad(bank_angle_deg)))


    num_arc_points = 30          # RF turn 시 원호를 몇 점으로 표현할지 (너무 많으면 느려짐)

    check_corridor_nfz = True

    N_init = 1000    # 초기해 생성 목표 개수(후속 필터를 통과한 해들 중 일부만 실제 init_pop으로 사용될 수 있음)
    min_feasible_init_solutions = 1  # 진화 시작 전 확보할 최소 통과 초기해 개수(예: 20으로 올리면 초기 다양성 증가)
    N_pop = 100      # 세대별 유지할 부모 해 집단 크기 (population size)
    Nmax = 10       # 최대 세대 수
    offspring_ratio = 0.6   # 자식 해 수 비율: n_off = round(len(parents) * offspring_ratio)
                            # 예) parents=1, ratio=0.5 -> n_off=0 / parents=2, ratio=0.5 -> n_off=1
    require_rf_for_parent_selection = True  # True: 부모 선택은 constraint AND RF 동시 만족만 허용

    # ── Mutation 튜닝 파라미터 ──
    mutation_rate = 0.20                  # 변이 적용 확률
    use_local_safe_resample = True        # True면 부모 세그먼트 주변 safe node 재샘플링 시도, False면 단순 WP 교란으로 변이 대체
    local_resample_prob = 0.70            # 변이가 발생했을 때 로컬 재샘플링 분기 확률, 로컬 재샘플링이란 부모 세그먼트 주변의 safe node 후보에서 새 WP를 뽑는 것을 말함 (0~1)
    local_strip_width_m = 350.0           # 세그먼트 주변 strip 폭(m)
    local_radius_m = 100.0                # 현재 변이 지점 주변 반경(m)
    local_max_tries = 5                   # 로컬 후보 재샘플링 최대 시도 횟수
    risk_weight_boost = True              # True면 low-risk 후보를 가중치로 더 자주 선택
    risk_weight_strength = 1.0            # low-risk 선호 강도(클수록 저위험 편향 증가)

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
    wp_perturb_steps = 10           # WP 교란 반복 횟수 (1=기존과 동일)
    min_extra_nodes_per_seg = 0     # 세그먼트당 최소 추가 노드 수 (int 또는 list)
    max_extra_nodes_per_seg = 2     # 세그먼트당 최대 추가 노드 수 (int 또는 list)
    use_wp_skip_generator = False    # True: 초기해 생성 시 WP-skip 생성기 사용, False: all-WP 생성기만 사용
    init_pop_skip_mix_ratio = 0.5   # use_wp_skip_generator=True일 때 skip 생성기 비율(0~1)
    wp_skip_prob = 0.00             # 중간 WP 스킵 확률 (0~1)
    airspace_radius_km = 5.0       # 공역 반경 제약 (km) — 회랑은 이 반경 안에 있어야 함
    min_corridor_distance_km = 0.0  # 전체 회랑 최소 거리 제약 (km), 0이면 비활성화
    emergency_strip_m = 500.0       # emergency 포함 판별 strip 폭
    min_seg_for_extra_nodes_m = 1500.0  # 이보다 짧은 WP간 세그먼트에는 extra node 생성 안 함

    # ── RF Look-ahead 튜닝 파라미터 ──
    look_ahead = True            # True: 짧은 세그먼트 코너에서 turn radius 자동 축소
    look_ahead_threshold_m = 2000.0  # 이보다 짧은 세그먼트 → R 스케일 다운 시작 (m)

    # look_ahead_min_scale의 물리적 의미:
    #   코너 반경이 R_scaled = scale * R_base 까지 축소 가능
    #   같은 bank angle 기준으로 동등 속도는 V_scaled = V_base * sqrt(scale)
    #   예) 
    #       scale=0.11 -> V_scaled ~= 27.66 m/s (99.6 km/h)
    #       scale=0.15 -> V_scaled ~= 37.5 m/s (135 km/h)
    #       scale=0.3 -> V_scaled ~= 45.64 m/s (164.3 km/h)
    #       scale=0.5 -> V_scaled ~= 58.93 m/s (212.1 km/h)
    #       scale=0.8 -> V_scaled ~= 74.54 m/s (268.3 km/h)
    #       scale=1.0 -> V_scaled = V_base = 83.33 m/s (300 km/h)
    look_ahead_min_scale = 0.11   # 최소 R 스케일(0~1), 0.5면 R을 50%까지 축소 허용
    look_ahead_window = 2        # 코너 전후 몇 개 세그먼트까지 볼지 (1=바로 앞뒤만, 2=2개씩 ...)
    max_init_retries = 300       # Gen 1에서 RF+constraint feasible 해 없으면 최대 이 횟수만큼 초기 해 재생성

    # 현재 look_ahead_min_scale에서의 동등 속도/반경(참고값)
    look_ahead_min_equiv_speed_mps = ground_speed_mps * np.sqrt(look_ahead_min_scale)
    look_ahead_min_equiv_speed_kmh = look_ahead_min_equiv_speed_mps * 3.6
    look_ahead_min_turn_radius_m = rf_base_turn_radius_m * look_ahead_min_scale

    w_dist, w_ground, w_air = 0.1, 2.0, 2.0
    altitude_levels = np.array([300.0], dtype=float)
    use_heading_map = True

    # ════════════════════════════════════════════════════════════
    # WP 입력 모드 ON/OFF 스위치 (튜닝용)
    # True:  클릭으로 WP 입력 (튜닝 시작)
    # False: 기본 predefined waypoint 사용 (WP 고정 후)
    # ════════════════════════════════════════════════════════════
    use_clicked_waypoints = True  # ← 여기서 ON/OFF 조절
    enforce_mandatory_wp_order = True  # True: takeoff -> 입력 WP 순서 -> landing 순서를 항상 강제
    
    min_clicked_waypoints = 0   # 클릭으로 입력할 WP의 최소 개수 (takeoff/landing 제외)
    clicked_wp_map_zoom = 13    # 클릭으로 WP 입력 시 사용할 지도 초기 줌 레벨 (예: 13은 도시 지역에서 적당한 수준)
    clicked_wp_base_name = "clicked_waypoints"
    if use_clicked_waypoints and not USE_INTERACTIVE_BACKEND:
        print("Interactive backend is not available. Falling back to default predefined waypoints.")
        use_clicked_waypoints = False

    W_buf = 1250.0  # 회랑폭 버퍼 (m) — 리스크맵에서 회랑폭보다 이만큼 더 넓게 low-risk 영역으로 간주
    node_grid_resolution_m = 100.0 # 후보 노드 생성 시 격자 간격 (m) — 너무 촘촘하면 노드 수 폭발, 너무 넓으면 해 품질 저하

    MIN_SAFE_NODES_TARGET = 200
    SAFE_NODE_PERCENTILE_LIST = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]

    cell_size = 100.0
    refine_scales = np.array([1.0, 0.5, 0.2, 0.1])  # RF turn look-ahead 시 여러 scale 단계 적용 (큰 scale부터 순차적으로 축소 시도)
    # vertiport(지상) ↔ takeoff/landing(순항) 전환을 허용하도록 동적으로 완화
    delta_z_max = max(100.0, float(np.max(np.abs(altitude_levels - 150.0))) + 5.0)
    flight_dist_limit = 100000.0 
    objective_names = ["Distance", "Ground Risk", "Air Risk"]
    airspace_radius_m = float(airspace_radius_km) * 1000.0
    airspace_alt_min_m = 100.0
    airspace_alt_max_m = 700.0
    min_corridor_distance_m = float(min_corridor_distance_km) * 1000.0

    # ==================== 출력 폴더 생성 ====================
    import datetime as _dt
    _run_ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / _run_ts
    out_dir.mkdir(parents=True, exist_ok=True)

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
        "max_init_retries": max_init_retries,
        "w_dist": w_dist,
        "w_ground": w_ground,
        "w_air": w_air,
        "use_heading_map": bool(use_heading_map),
        "altitude_levels_m": altitude_levels.tolist(),
        "W_buf": W_buf,
        "node_grid_resolution_m": node_grid_resolution_m,
        "MIN_SAFE_NODES_TARGET": int(MIN_SAFE_NODES_TARGET),
        "SAFE_NODE_PERCENTILE_LIST": [float(v) for v in SAFE_NODE_PERCENTILE_LIST],
        "cell_size": cell_size,
        "refine_scales": refine_scales.tolist(),
        "delta_z_max": delta_z_max,
        "flight_dist_limit": flight_dist_limit,
        "check_corridor_nfz": check_corridor_nfz,
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
    }
    with open(out_dir / "params.json", "w", encoding="utf-8") as _pf:
        json.dump(params_dict, _pf, indent=2, ensure_ascii=False)
    print(f"Output folder : {out_dir}")

    # ==================== 리스크맵 로드 ====================
    pop_risk_raw = np.load("high_res_affected_population_GRC.npy", allow_pickle=True)
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

    air_raw = np.load("AirRisk_combined_max_risk_map.npy", allow_pickle=True).item()
    z_vec = air_raw["z_vec"]
    ar_3d = air_raw["Risk_3d"]

    AirRisk = np.zeros((Ny, Nx, len(altitude_levels)), dtype=float)
    if ar_3d.shape[0] == Nx and ar_3d.shape[1] == Ny:
        raw_a = np.transpose(ar_3d, (1, 0, 2))
    elif ar_3d.shape[0] == Ny and ar_3d.shape[1] == Nx:
        raw_a = ar_3d
    else:
        raise RuntimeError(f"AirRisk shape {ar_3d.shape} != ({Ny},{Nx})")

    for i, alt in enumerate(altitude_levels):
        AirRisk[:, :, i] = raw_a[:, :, int(np.argmin(np.abs(z_vec - alt)))]

    # ==================== 경로 정의 ====================
    # 시작/도착 버티포트는 필수이며, 두 점이 같아도 허용됩니다.
    # start_vertiport = np.array([35.6033361, 129.0776917, 150.0], dtype=float)
    # end_vertiport = np.array([35.6249109, 129.0586710, 150.0], dtype=float)
    start_vertiport = np.array([35.6033361, 129.0776917, 150.0], dtype=float)
    end_vertiport = np.array([35.6033361, 129.0776917, 150.0], dtype=float)
    if start_vertiport.size != 3 or end_vertiport.size != 3:
        raise ValueError("Both start_vertiport and end_vertiport must be [lat, lon, alt].")

    # 공역 입력: 중심 LLA + 반경 + 최소/최대 고도
    # None이면 시작/도착의 중점을 중심으로 자동 설정.
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

    # lat이 세로축 -> + 위쪽, - 아래쪽 
    # lon이 가로축 -> + 오른쪽, - 왼쪽
    lat_lim = [35.5446, 35.6427]
    lon_lim = [129.0514, 129.1436]
    request = cimgt.OSM()

    # 기본 중간 WP: 비워두면 start/end만으로 최적화
    corridor_lat_default = np.array([], dtype=float)
    # corridor_lat_default = np.array([ 35.5944587, 35.6195580,
    #     35.6218142,
    #     35.5625903, 35.5671301,
    #     35.5887324, 35.5931814,
    #     35.6185184,
    #     35.6249109], dtype=float)
    
    corridor_lon_default = np.array([], dtype=float)
    # corridor_lon_default = np.array([129.0977958, 129.1153758,
    #     129.1266116,
    #     129.0949662, 129.0776521,
    #     129.0691071, 129.0665565,
    #     129.0512209,
    #     129.0586710], dtype=float)

    waypoint_alt_fixed_m = float(altitude_levels[0])
    clicked_wp_json_path = None
    clicked_wp_csv_path = None

    corridor_lat = corridor_lat_default.copy()
    corridor_lon = corridor_lon_default.copy()

    # 클릭 입력 UI 표시용 오버레이(최종 경로 계산과 무관한 안내 정보)
    preview_takeoff, _ = build_transition_point(
        start_vertiport,
        angle_deg=25.0,
        alt_delta_m=350.0,
        sector=6,
        target_alt_m=waypoint_alt_fixed_m,
    )
    preview_landing, _ = build_transition_point(
        end_vertiport,
        angle_deg=25.0,
        alt_delta_m=350.0,
        sector=11,
        target_alt_m=waypoint_alt_fixed_m,
    )

    # 선택 입력: 비상착륙지점
    use_emergency_points = True
    emergency_points_input = np.array([
        # [35.6201083, 129.1191806, waypoint_alt_fixed_m],
        # [35.5678222, 129.1067280, waypoint_alt_fixed_m],
        # [35.5919889, 129.0751972, waypoint_alt_fixed_m],
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

    # 선택 입력: 금지구역 (bbox: [lon_min, lon_max, lat_min, lat_max])
    use_forbidden_zones = False

    # forbidden_zones_input = np.array([], dtype=float)

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

    waypoint_alts = np.full(corridor_lat.shape[0], waypoint_alt_fixed_m, dtype=float)
    waypoints = np.column_stack([corridor_lat, corridor_lon, waypoint_alts]) if corridor_lat.size > 0 else np.empty((0, 3), dtype=float)

    if waypoints.shape[0] > 0:
        print("Selected waypoints in click/order sequence:")
        for i, wp in enumerate(waypoints, start=1):
            print(f"  WP{i:02d}: lat={wp[0]:.7f}, lon={wp[1]:.7f}, alt={wp[2]:.1f}m")
    else:
        print("No middle waypoints provided. Optimization will run with start/end vertiports only.")

    takeoff_target_alt = float(waypoint_alt_fixed_m)
    landing_target_alt = float(waypoint_alt_fixed_m)

    takeoff_complete, _ = build_transition_point(
        start_vertiport,
        angle_deg=25.0,
        alt_delta_m=350.0,
        sector=6,
        target_alt_m=takeoff_target_alt,
    )
    landing_entry, _ = build_transition_point(
        end_vertiport,
        angle_deg=25.0,
        alt_delta_m=350.0,
        sector=11,
        target_alt_m=landing_target_alt,
    )

    # backbone: takeoff_complete(SSE) → WPs(원래 순서, CW) → landing_entry(NW)
    # sector 6 = SSE (이륙), sector 11 = NW (착륙)
    # WP 순서대로 이동하면 CW 회로가 됨
    backbone = np.vstack([takeoff_complete, waypoints, landing_entry])
    is_fixed = np.zeros(backbone.shape[0], dtype=bool)
    is_fixed[:] = True    # takeoff + 모든 입력 WP + landing 모두 고정

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

    # ==================== params.json 좌표/지역 정보 추가 저장 ====================
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
        "takeoff_angle_deg": 25.0,
        "alt_delta_m": 350.0,
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
    })
    with open(out_dir / "params.json", "w", encoding="utf-8") as _pf:
        json.dump(params_dict, _pf, indent=2, ensure_ascii=False)
    print("params.json updated with spatial data.")

    # ==================== 세그먼트별 후보 노드 ====================
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
        for perc in SAFE_NODE_PERCENTILE_LIST:
            thr = float(np.percentile(risks, perc))
            safe = nodes_seg[risks <= thr]
            if safe.shape[0] >= target:
                break
        if safe.size == 0:
            safe = nodes_seg
        return safe, thr

    safe_nodes_by_seg = []
    safe_airrisk_by_seg = []
    thr_list = []
    for k in range(backbone.shape[0] - 1):
        seg_m = _seg_dist_m(backbone[k], backbone[k + 1])
        if seg_m < min_seg_for_extra_nodes_m:
            # 짧은 세그먼트: 노드 생성 자체를 건너뜀
            safe_nodes_by_seg.append(np.empty((0, 3)))
            safe_airrisk_by_seg.append(np.empty((0,), dtype=float))
            thr_list.append(0.0)
            continue
        s, t = _build_safe_nodes(backbone[k], backbone[k + 1])
        # strip 필터
        s = filter_nodes_in_strip(backbone[k], backbone[k + 1], s, 10 * W_half)
        safe_nodes_by_seg.append(s)

        if s.size > 0:
            I_s = np.clip(((s[:, 1] - lon_lim[0]) / (lon_lim[1] - lon_lim[0]) * (Nx - 1)).astype(int), 0, Nx - 1)
            J_s = np.clip(((s[:, 0] - lat_lim[0]) / (lat_lim[1] - lat_lim[0]) * (Ny - 1)).astype(int), 0, Ny - 1)
            ai_s = np.argmin(np.abs(s[:, 2][:, None] - altitude_levels[None, :]), axis=1)
            safe_airrisk_by_seg.append(AirRisk[J_s, I_s, ai_s].astype(float))
        else:
            safe_airrisk_by_seg.append(np.empty((0,), dtype=float))
        thr_list.append(t)

    air_thr_global = float(np.max(thr_list)) if thr_list else 1.0
    nodes_pool = np.vstack([s for s in safe_nodes_by_seg if s.size > 0]) \
                 if any(s.size > 0 for s in safe_nodes_by_seg) else (
                     emergency_points if emergency_points.size > 0 else backbone.copy()
                 )
    node_risk_pool = np.concatenate([r for r in safe_airrisk_by_seg if r.size > 0]) \
                     if any(r.size > 0 for r in safe_airrisk_by_seg) else np.empty((0,), dtype=float)
    if node_risk_pool.shape[0] != nodes_pool.shape[0]:
        node_risk_pool = np.empty((0,), dtype=float)

    # ==================== 초기 해 생성 (RF+constraint feasible 해 확인될 때까지 재시도) ====================
    print(
        f"Searching for RF+constraint feasible initial pop "
        f"(max {max_init_retries} retries, N_init={N_init}, "
        f"min_feasible_init_solutions={min_feasible_init_solutions}) ..."
    )

    def _make_init_pop():
        """N_init 개의 초기 해를 새로 생성 (all-WP + optional WP-skip 혼합)"""
        _pop = []
        max_draws = int(max(3 * N_init, 50))
        draws = 0
        skip_ratio = float(np.clip(init_pop_skip_mix_ratio, 0.0, 1.0)) if use_wp_skip_generator else 0.0
        n_all_wp = int(round(N_init * (1.0 - skip_ratio)))
        while len(_pop) < N_init and draws < max_draws:
            _i = len(_pop)
            draws += 1
            if _i < n_all_wp:
                _sol = generate_single_initial_solution(
                    backbone, wp_perturb_radius_m,
                    min_extra_nodes_per_seg, max_extra_nodes_per_seg,
                    safe_nodes_by_seg, emergency_points, emergency_strip_m,
                    is_fixed,
                    wp_perturb_steps=wp_perturb_steps,
                    min_seg_for_extra_nodes_m=min_seg_for_extra_nodes_m,
                )
            else:
                if use_wp_skip_generator:
                    _sol = generate_single_initial_solution_with_skip(
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
                    _sol = generate_single_initial_solution(
                        backbone, wp_perturb_radius_m,
                        min_extra_nodes_per_seg, max_extra_nodes_per_seg,
                        safe_nodes_by_seg, emergency_points, emergency_strip_m,
                        is_fixed,
                        wp_perturb_steps=wp_perturb_steps,
                        min_seg_for_extra_nodes_m=min_seg_for_extra_nodes_m,
                    )
            if is_path_inside_airspace(
                _sol,
                airspace_center_lla[:2],
                airspace_radius_m,
                alt_min_m=airspace_alt_min_m,
                alt_max_m=airspace_alt_max_m,
            ):
                if enforce_mandatory_wp_order:
                    _sol = _enforce_mandatory_wp_order(_sol, backbone)
                _pop.append(_sol)

        if len(_pop) < N_init:
            print(f"  [Init] Airspace-filtered initial pop: {len(_pop)}/{N_init}")
        return _pop

    init_pop = None
    for _retry in range(1, max_init_retries + 1):
        _candidate = _make_init_pop()
        _cand_n = len(_candidate)
        if not _candidate:
            if _retry % 50 == 0:
                print(f"  [Init retry {_retry}/{max_init_retries}] candidate_after_initial_airspace: 0/{N_init}")
            continue

        # 1단계: RF 기하학적 feasibility 체크 (cheap — risk map 조회 없음)
        _rf_ok_list = []
        for _c in _candidate:
            _c_eval = _enforce_mandatory_wp_order(_c, backbone) if enforce_mandatory_wp_order else _c
            _rf = apply_rf_turns(_c_eval, ground_speed_mps, bank_angle_deg, num_arc_points,
                                 look_ahead=look_ahead,
                                 look_ahead_threshold_m=look_ahead_threshold_m,
                                 look_ahead_min_scale=look_ahead_min_scale,
                                 look_ahead_window=look_ahead_window)
            _rf_ok_list.append((_rf["feasible"], _rf["path"], _c_eval))
        _rf_cnt = sum(1 for ok, _, _ in _rf_ok_list if ok)

        # 2단계: RF feasible 해에 대해서만 constraint 체크 (expensive)
        _both_cnt = 0
        _cst_cnt = 0
        _air_cnt = 0
        _dist_cnt = 0
        _feasible_init = []
        if _rf_cnt > 0:
            for _idx, (_rf_ok, _rf_path, _c_eval) in enumerate(_rf_ok_list):
                if not _rf_ok:
                    continue
                _, _cst_ok = evaluate_objectives_with_constraints_gp(
                    build_full_corridor_path(start_vertiport, takeoff_complete, _rf_path, landing_entry, end_vertiport), Norm_RT, AirRisk, use_heading_map,
                    flight_dist_limit, forbidden_zones, delta_z_max,
                    altitude_levels, cell_size, refine_scales, air_thr_global,
                    w_dist, w_ground, w_air, lat_lim, lon_lim,
                    W_half=W_half, check_corridor_nfz=check_corridor_nfz,
                    vertiport=None, landing_entry=None,
                    takeoff_complete=None,
                )
                full_path = build_full_corridor_path(start_vertiport, takeoff_complete, _rf_path, landing_entry, end_vertiport)
                _air_ok = is_path_inside_airspace(
                    full_path,
                    airspace_center_lla[:2],
                    airspace_radius_m,
                    alt_min_m=airspace_alt_min_m,
                    alt_max_m=airspace_alt_max_m,
                )
                _dist_ok = True
                if min_corridor_distance_m > 0.0:
                    _dist_ok = _path_total_3d_distance_m(full_path) + 1e-6 >= min_corridor_distance_m

                if _cst_ok:
                    _cst_cnt += 1
                if _air_ok:
                    _air_cnt += 1
                if _dist_ok:
                    _dist_cnt += 1

                if _cst_ok and _air_ok and _dist_ok:
                    _both_cnt += 1
                    _feasible_init.append(_c_eval)

        # 진행 상황 출력 (50회마다 또는 RF feasible 발견 시)
        if _retry % 50 == 0 or _rf_cnt > 0:
            print(
                f"  [Init retry {_retry}/{max_init_retries}] "
                f"candidate_after_initial_airspace: {_cand_n}/{N_init} | "
                f"rf_feasible: {_rf_cnt}/{_cand_n} | "
                f"constraint_ok(given RF): {_cst_cnt}/{_rf_cnt} | "
                f"airspace_ok(given RF): {_air_cnt}/{_rf_cnt} | "
                f"min_dist_ok(given RF): {_dist_cnt}/{_rf_cnt} | "
                f"both_feasible: {_both_cnt}/{_cand_n} | "
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

    # ==================== 지도 공통 설정 ====================
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
    for s in safe_nodes_by_seg:
        if s is not None and s.size > 0:
            extent_points.extend(s[:, :2].tolist())
    if init_pop is not None:
        for p in init_pop:
            if p is not None and p.size > 0:
                extent_points.extend(p[:, :2].tolist())

    map_extent = compute_centered_map_extent(np.array(extent_points, dtype=float), airspace_center_lla,
                                             ring_radii_m=(airspace_radius_m,), pad_ratio=0.10)

    bb_full = np.vstack([start_vertiport, backbone, end_vertiport])
    total_safe_count = sum(s.shape[0] for s in safe_nodes_by_seg if s.size > 0)

    # ==================== [Fig 1] 후보 노드(Safe Node) 분포 ====================
    fig1 = plt.figure("Figure 1: Candidate Safe Nodes", figsize=(14, 10))
    fig1.subplots_adjust(left=0.08, right=0.78)
    gx1 = fig1.add_subplot(1, 1, 1, projection=request.crs)
    gx1.set_extent(map_extent)
    gx1.add_image(request, 13)
    gx1.set_title(f"Candidate Safe Nodes  (total {total_safe_count} nodes,  grid {node_grid_resolution_m}m,  W_buf {W_buf}m)")
    draw_vertiport_radius_rings(gx1, airspace_center_lla, radii_m=(airspace_radius_m,))
    plot_forbidden_zones(gx1, forbidden_zones, face_alpha=0.10, edge_alpha=0.80)

    # backbone
    gx1.plot(bb_full[:, 1], bb_full[:, 0], "r--", linewidth=2, transform=ccrs.Geodetic(),
             label="Backbone", zorder=5)
    gx1.scatter(waypoints[:, 1], waypoints[:, 0], s=60, c="orange", edgecolors="k",
                linewidths=0.5, marker="o", transform=ccrs.Geodetic(), label="Waypoints (WP)", zorder=6)
    gx1.scatter([start_vertiport[1]], [start_vertiport[0]], s=120, c="red", edgecolors="k",
                marker="s", transform=ccrs.Geodetic(), label="Start Vertiport", zorder=7)
    gx1.scatter([end_vertiport[1]], [end_vertiport[0]], s=120, c="crimson", edgecolors="k",
                marker="D", transform=ccrs.Geodetic(), label="End Vertiport", zorder=7)
    gx1.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90, c="blue",
                marker="^", transform=ccrs.Geodetic(), label="Takeoff", zorder=7)
    gx1.scatter([landing_entry[1]], [landing_entry[0]], s=90, c="green",
                marker="v", transform=ccrs.Geodetic(), label="Landing", zorder=7)
    # 세그먼트별 후보 노드 (공중위험도 0~1 컬러맵)
    all_safe_risks = np.concatenate([r for r in safe_airrisk_by_seg if r.size > 0]) \
        if any(r.size > 0 for r in safe_airrisk_by_seg) else np.empty((0,), dtype=float)
    r_min = float(np.min(all_safe_risks)) if all_safe_risks.size > 0 else 0.0
    r_max = float(np.max(all_safe_risks)) if all_safe_risks.size > 0 else 1.0
    denom = (r_max - r_min) if (r_max - r_min) > 1e-12 else 1.0
    first_scatter = None

    for ki, seg_nodes in enumerate(safe_nodes_by_seg):
        if seg_nodes.size > 0:
            risks = safe_airrisk_by_seg[ki] if ki < len(safe_airrisk_by_seg) else np.zeros(seg_nodes.shape[0], dtype=float)
            risks_01 = np.clip((risks - r_min) / denom, 0.0, 1.0)
            lab = f"Seg {ki+1} nodes ({seg_nodes.shape[0]})" if ki == 0 else None
            sc = gx1.scatter(seg_nodes[:, 1], seg_nodes[:, 0], s=7, c=risks_01, cmap="jet",
                             vmin=0.0, vmax=1.0, alpha=0.78,
                             transform=ccrs.Geodetic(), label=lab, zorder=3)
            if first_scatter is None:
                first_scatter = sc
    if len(safe_nodes_by_seg) > 3:
        gx1.scatter([], [], s=4, c="gray", alpha=0.9, label=f"... +{len(safe_nodes_by_seg)-3} more segs")
    if first_scatter is not None:
        cax = fig1.add_axes([0.035, 0.16, 0.022, 0.70])
        cbar = fig1.colorbar(first_scatter, cax=cax)
        cbar.set_label("Air Risk (normalized 0-1)")
    # emergency
    if emergency_points.size > 0:
        gx1.scatter(emergency_points[:, 1], emergency_points[:, 0], s=80, c="lime",
                    edgecolors="k", marker="P", transform=ccrs.Geodetic(),
                    label="Emergency Landing", zorder=7)
    gx1.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, framealpha=0.9)
    fig1.savefig(out_dir / "fig1_safe_nodes.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_dir / 'fig1_safe_nodes.png'}")
    plt.close(fig1)

    # ==================== [Fig 2] 초기 해 샘플 + RF turn ====================
    n_sample = min(5, len(init_pop))
    fig2 = plt.figure("Figure 2: Sample Initial Solutions + RF Turn", figsize=(14, 10))
    fig2.subplots_adjust(left=0.05, right=0.78)
    gx2 = fig2.add_subplot(1, 1, 1, projection=request.crs)
    gx2.set_extent(map_extent)
    gx2.add_image(request, 13)
    gx2.set_title(f"Sample Initial Solutions ({n_sample}) with RF Turn")
    draw_vertiport_radius_rings(gx2, airspace_center_lla, radii_m=(airspace_radius_m,))
    plot_forbidden_zones(gx2, forbidden_zones, face_alpha=0.10, edge_alpha=0.80)

    gx2.plot(bb_full[:, 1], bb_full[:, 0], "r--", linewidth=1.5, transform=ccrs.Geodetic(),
             label="Backbone", zorder=4)
    gx2.scatter([start_vertiport[1]], [start_vertiport[0]], s=120, c="red", edgecolors="k",
                marker="s", transform=ccrs.Geodetic(), label="Start Vertiport", zorder=8)
    gx2.scatter([end_vertiport[1]], [end_vertiport[0]], s=120, c="crimson", edgecolors="k",
                marker="D", transform=ccrs.Geodetic(), label="End Vertiport", zorder=8)
    gx2.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90, c="blue",
                marker="^", transform=ccrs.Geodetic(), label="Takeoff", zorder=8)
    gx2.scatter([landing_entry[1]], [landing_entry[0]], s=90, c="green",
                marker="v", transform=ccrs.Geodetic(), label="Landing", zorder=8)
    colors_sample = plt.cm.tab10(np.linspace(0, 1, n_sample))
    for si in range(n_sample):
        rf = apply_rf_turns(init_pop[si], ground_speed_mps, bank_angle_deg, num_arc_points,
                            look_ahead=look_ahead,
                            look_ahead_threshold_m=look_ahead_threshold_m,
                            look_ahead_min_scale=look_ahead_min_scale)
        rf_path = rf["path"]
        gx2.plot(rf_path[:, 1], rf_path[:, 0], "-", color=colors_sample[si], linewidth=1.2,
                 transform=ccrs.Geodetic(), label=f"Sol {si+1}", zorder=5)
    gx2.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, framealpha=0.9)
    fig2.savefig(out_dir / "fig2_sample_init.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_dir / 'fig2_sample_init.png'}")
    plt.close(fig2)

    # ==================== [Fig 2B] 동일 초기해 RF 적용 전/후 비교 ====================
    fig2b = plt.figure("Figure 2B: Initial Solutions Before vs After RF", figsize=(14, 10))
    fig2b.subplots_adjust(left=0.05, right=0.78)
    gx2b = fig2b.add_subplot(1, 1, 1, projection=request.crs)
    gx2b.set_extent(map_extent)
    gx2b.add_image(request, 13)
    gx2b.set_title(f"Same Initial Solutions: Before RF (dashed) vs After RF (solid), n={n_sample}")
    draw_vertiport_radius_rings(gx2b, airspace_center_lla, radii_m=(airspace_radius_m,))
    plot_forbidden_zones(gx2b, forbidden_zones, face_alpha=0.10, edge_alpha=0.80)

    gx2b.plot(bb_full[:, 1], bb_full[:, 0], "r--", linewidth=1.5, transform=ccrs.Geodetic(),
              label="Backbone", zorder=4)
    gx2b.scatter([start_vertiport[1]], [start_vertiport[0]], s=120, c="red", edgecolors="k",
                 marker="s", transform=ccrs.Geodetic(), label="Start Vertiport", zorder=8)
    gx2b.scatter([end_vertiport[1]], [end_vertiport[0]], s=120, c="crimson", edgecolors="k",
                 marker="D", transform=ccrs.Geodetic(), label="End Vertiport", zorder=8)
    gx2b.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90, c="blue",
                 marker="^", transform=ccrs.Geodetic(), label="Takeoff", zorder=8)
    gx2b.scatter([landing_entry[1]], [landing_entry[0]], s=90, c="green",
                 marker="v", transform=ccrs.Geodetic(), label="Landing", zorder=8)

    for si in range(n_sample):
        col = colors_sample[si]
        path_before = init_pop[si]
        rf = apply_rf_turns(
            path_before,
            ground_speed_mps,
            bank_angle_deg,
            num_arc_points,
            look_ahead=look_ahead,
            look_ahead_threshold_m=look_ahead_threshold_m,
            look_ahead_min_scale=look_ahead_min_scale,
            look_ahead_window=look_ahead_window,
        )
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

    # ==================== NSGA-III 최적화 ====================
    print("Running NSGA-III …")
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
        ground_speed_mps=ground_speed_mps,
        bank_angle_deg=bank_angle_deg,
        num_arc_points=num_arc_points,
        look_ahead=look_ahead,
        look_ahead_threshold_m=look_ahead_threshold_m,
        look_ahead_min_scale=look_ahead_min_scale,
        look_ahead_window=look_ahead_window,
        W_half=W_half, check_corridor_nfz=check_corridor_nfz,
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

    # ==================== 세대별 진화 스냅샷 저장 ====================
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
        rep_colors = ["cyan", "lime", "magenta", "black"]
        for ri, rep in enumerate(greps):
            rf = apply_rf_turns(rep, ground_speed_mps, bank_angle_deg, num_arc_points,
                                look_ahead=look_ahead,
                                look_ahead_threshold_m=look_ahead_threshold_m,
                                look_ahead_min_scale=look_ahead_min_scale,
                                look_ahead_window=look_ahead_window)
            rp = rf["path"]
            segs = rf["segments"]
            col = rep_colors[ri % len(rep_colors)]
            lab = rep_labels[ri] if ri < len(rep_labels) else f"Rep{ri}"

            full_path = build_full_corridor_path(start_vertiport, takeoff_complete, rp, landing_entry, end_vertiport)
            plot_corridor_width(gxg, full_path, W_half, color=col, alpha=0.08)

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

    # feasibility 체크
    feasible_count = 0
    rf_feasible_count = 0
    feas_mask = []
    for i in range(len(pop)):
        rf = apply_rf_turns(pop[i], ground_speed_mps, bank_angle_deg, num_arc_points,
                            look_ahead=look_ahead,
                            look_ahead_threshold_m=look_ahead_threshold_m,
                            look_ahead_min_scale=look_ahead_min_scale,
                            look_ahead_window=look_ahead_window)
        if rf["feasible"]:
            rf_feasible_count += 1
        full_path = build_full_corridor_path(start_vertiport, takeoff_complete, rf["path"], landing_entry, end_vertiport)
        _, feas = evaluate_objectives_with_constraints_gp(
            full_path, Norm_RT, AirRisk, use_heading_map,
            flight_dist_limit, forbidden_zones, delta_z_max,
            altitude_levels, cell_size, refine_scales,
            air_thr_global, w_dist, w_ground, w_air, lat_lim, lon_lim,
            W_half=W_half, check_corridor_nfz=check_corridor_nfz,
            vertiport=None, landing_entry=None,
            takeoff_complete=None,
        )
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
    print(f"Final feasible (constraints): {feasible_count}/{len(pop)}")
    print(f"RF geometric feasible (no clamp): {rf_feasible_count}/{len(pop)}")

    if feasible_count == 0:
        print("No feasible solution. Retrying …")
        return False, 0

    # ==================== 대표 해 ====================
    reps = pick_representatives(pop, fvals) if pop and fvals.size > 0 else []

    # ==================== [Fig 3] Pareto 결과 (3 objective pairs) ====================
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
    fig3.suptitle("Pareto Front  (blue=feasible, gray=infeasible)", fontsize=13)
    obj_pairs = [(0, 1), (0, 2), (1, 2)]
    for ax_i, (oi, oj) in enumerate(obj_pairs):
        ax = axes3[ax_i]
        for k in range(len(feas_mask)):
            col = "royalblue" if feas_mask[k] else "lightgray"
            z = 5 if feas_mask[k] else 1
            ax.scatter(fvals[k, oi], fvals[k, oj], c=col, s=18, alpha=0.7, zorder=z,
                       edgecolors="k", linewidths=0.3)
        ax.set_xlabel(objective_names[oi], fontsize=10)
        ax.set_ylabel(objective_names[oj], fontsize=10)
        ax.set_title(f"{objective_names[oi]} vs {objective_names[oj]}", fontsize=10)
        ax.grid(True, alpha=0.3)
        # 대표 해 표시
        if reps:
            for ri, rep in enumerate(reps):
                rf_rep = apply_rf_turns(rep, ground_speed_mps, bank_angle_deg, num_arc_points,
                                        look_ahead=look_ahead,
                                        look_ahead_threshold_m=look_ahead_threshold_m,
                                        look_ahead_min_scale=look_ahead_min_scale,
                                        look_ahead_window=look_ahead_window)
                f_rep, _ = evaluate_objectives_with_constraints_gp(
                    build_full_corridor_path(start_vertiport, takeoff_complete, rf_rep["path"], landing_entry, end_vertiport), Norm_RT, AirRisk, use_heading_map,
                    flight_dist_limit, forbidden_zones, delta_z_max,
                    altitude_levels, cell_size, refine_scales,
                    air_thr_global, w_dist, w_ground, w_air, lat_lim, lon_lim,
                    W_half=W_half, check_corridor_nfz=check_corridor_nfz,
                    vertiport=None, landing_entry=None,
                    takeoff_complete=None,
                )
                rep_labels_3 = objective_names + ["Balanced"]
                rep_markers = ["D", "D", "D", "*"]
                rep_cols = ["cyan", "lime", "magenta", "red"]
                lab = rep_labels_3[ri] if ri < len(rep_labels_3) else f"Rep{ri}"
                ax.scatter(f_rep[oi], f_rep[oj], c=rep_cols[ri % len(rep_cols)],
                           s=80, marker=rep_markers[ri % len(rep_markers)],
                           edgecolors="k", linewidths=1, zorder=10, label=lab if ax_i == 0 else None)
    if reps:
        axes3[0].legend(loc="upper right", fontsize=7)
    fig3.tight_layout(rect=[0, 0, 1, 0.93])
    fig3.savefig(out_dir / "fig3_pareto.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_dir / 'fig3_pareto.png'}")
    plt.close(fig3)

    # ==================== [Fig 4] 최종 최적 회랑 + RF turn 시각화 ====================
    fig4 = plt.figure("Figure 4: Optimal Corridor", figsize=(14, 10))
    fig4.subplots_adjust(left=0.05, right=0.72)
    gx4 = fig4.add_subplot(1, 1, 1, projection=request.crs)
    gx4.set_extent(map_extent)
    gx4.add_image(request, 13)
    gx4.set_title("Optimal Corridor (RF Turn applied)")
    draw_vertiport_radius_rings(gx4, airspace_center_lla, radii_m=(airspace_radius_m,))
    plot_forbidden_zones(gx4, forbidden_zones, face_alpha=0.10, edge_alpha=0.80)

    # backbone
    gx4.plot(bb_full[:, 1], bb_full[:, 0], "r--", linewidth=1.5, transform=ccrs.Geodetic(),
             label="Backbone", zorder=4)
    gx4.scatter(waypoints[:, 1], waypoints[:, 0], s=60, c="orange", edgecolors="k",
                linewidths=0.5, marker="o", transform=ccrs.Geodetic(), label="Waypoints", zorder=6)
    gx4.scatter([start_vertiport[1]], [start_vertiport[0]], s=120, c="red", edgecolors="k",
                marker="s", transform=ccrs.Geodetic(), label="Start Vertiport", zorder=7)
    gx4.scatter([end_vertiport[1]], [end_vertiport[0]], s=120, c="crimson", edgecolors="k",
                marker="D", transform=ccrs.Geodetic(), label="End Vertiport", zorder=7)
    gx4.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=90, c="blue",
                marker="^", transform=ccrs.Geodetic(), label="Takeoff", zorder=7)
    gx4.scatter([landing_entry[1]], [landing_entry[0]], s=90, c="green",
                marker="v", transform=ccrs.Geodetic(), label="Landing", zorder=7)

    # 대표 해 (balanced = last in reps)
    rep_labels = objective_names + ["Balanced"]
    rep_colors = ["cyan", "lime", "magenta", "black"]
    for ri, rep in enumerate(reps):
        rf = apply_rf_turns(rep, ground_speed_mps, bank_angle_deg, num_arc_points,
                            look_ahead=look_ahead,
                            look_ahead_threshold_m=look_ahead_threshold_m,
                            look_ahead_min_scale=look_ahead_min_scale,
                            look_ahead_window=look_ahead_window)
        rp = rf["path"]
        segs = rf["segments"]
        col = rep_colors[ri % len(rep_colors)]
        lab = rep_labels[ri] if ri < len(rep_labels) else f"Rep{ri}"

        # corridor width: start vertiport → takeoff(SSE) → path → landing(NW) → end vertiport
        full_path = build_full_corridor_path(start_vertiport, takeoff_complete, rp, landing_entry, end_vertiport)
        plot_corridor_width(gx4, full_path, W_half, color=col, alpha=0.08)

        # 구간별 시각화
        for seg in segs:
            pts = seg["points"]
            if seg["type"] == "TF":
                gx4.plot(pts[:, 1], pts[:, 0], "-", color=col, linewidth=1.5,
                         transform=ccrs.Geodetic(), zorder=8)
                gx4.scatter(pts[0, 1], pts[0, 0], s=25, c=col, marker="o",
                            edgecolors="k", linewidths=0.4, transform=ccrs.Geodetic(), zorder=9)
                gx4.scatter(pts[-1, 1], pts[-1, 0], s=25, c=col, marker="o",
                            edgecolors="k", linewidths=0.4, transform=ccrs.Geodetic(), zorder=9)
            elif seg["type"] == "RF":
                gx4.plot(pts[:, 1], pts[:, 0], "-", color=col, linewidth=2.0,
                         transform=ccrs.Geodetic(), zorder=8)
                gx4.scatter(pts[0, 1], pts[0, 0], s=40, c="yellow", marker=">",
                            edgecolors="k", linewidths=0.6, transform=ccrs.Geodetic(), zorder=10)
                gx4.scatter(pts[-1, 1], pts[-1, 0], s=40, c="yellow", marker="s",
                            edgecolors="k", linewidths=0.6, transform=ccrs.Geodetic(), zorder=10)
                # arc center
                ac = seg["arc_center"]
                gx4.scatter(ac[1], ac[0], s=50, c="white", marker="x",
                            linewidths=1.5, transform=ccrs.Geodetic(), zorder=10)

        # 레이블용 빈 플롯
        gx4.plot([], [], "-", color=col, linewidth=1.5, label=f"{lab} (TF)")
        gx4.plot([], [], "-", color=col, linewidth=2.5, label=f"{lab} (RF arc)")

    # legend 마커 설명
    gx4.scatter([], [], s=40, c="yellow", marker=">", edgecolors="k", label="Arc Start")
    gx4.scatter([], [], s=40, c="yellow", marker="s", edgecolors="k", label="Arc End")
    gx4.scatter([], [], s=50, c="white", marker="x", linewidths=1.5, label="Arc Center")

    gx4.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, framealpha=0.9)
    fig4.savefig(out_dir / "fig4_optimal_corridor.png", dpi=150, bbox_inches="tight")
    print(f"Saved {out_dir / 'fig4_optimal_corridor.png'}")
    plt.close(fig4)

    # ==================== Excel 출력 (최종 Balanced 해) ====================
    import pandas as pd

    best_rep = reps[-1] if reps else (pop[0] if pop else None)
    if best_rep is not None:
        rf_best = apply_rf_turns(best_rep, ground_speed_mps, bank_angle_deg, num_arc_points,
                                  look_ahead=look_ahead,
                                  look_ahead_threshold_m=look_ahead_threshold_m,
                                  look_ahead_min_scale=look_ahead_min_scale,
                                  look_ahead_window=look_ahead_window)
        R_turn = rf_best["turn_radius_m"]
        segs_best = rf_best["segments"]

        rows = []
        point_idx = 0

        # Vertiport (출발)
        rows.append({
            "Point_No": point_idx, "Type": "Vertiport", "Segment": "Start",
            "Lat": start_vertiport[0], "Lon": start_vertiport[1], "Alt_m": start_vertiport[2],
            "Seg_Type": "-", "TF_Start": "", "TF_End": "", "RF_Start": "", "RF_End": "",
            "Arc_Center_Lat": "", "Arc_Center_Lon": "",
            "Turn_Radius_m": "", "Turn_Angle_deg": "",
            "Ground_Speed_mps": ground_speed_mps, "Bank_Angle_deg": bank_angle_deg,
        })
        point_idx += 1

        # Takeoff complete (SSE, sector 6)
        rows.append({
            "Point_No": point_idx, "Type": "Takeoff_Complete", "Segment": "Transition",
            "Lat": takeoff_complete[0], "Lon": takeoff_complete[1], "Alt_m": takeoff_complete[2],
            "Seg_Type": "-", "TF_Start": "", "TF_End": "", "RF_Start": "", "RF_End": "",
            "Arc_Center_Lat": "", "Arc_Center_Lon": "",
            "Turn_Radius_m": "", "Turn_Angle_deg": "",
            "Ground_Speed_mps": ground_speed_mps, "Bank_Angle_deg": bank_angle_deg,
        })
        point_idx += 1

        # Segments
        seg_counter = 0
        for seg in segs_best:
            seg_counter += 1
            pts = seg["points"]
            stype = seg["type"]

            if stype == "TF":
                for pi in range(pts.shape[0]):
                    is_start = "O" if pi == 0 else ""
                    is_end = "O" if pi == pts.shape[0] - 1 else ""
                    rows.append({
                        "Point_No": point_idx, "Type": "TF_Point", "Segment": f"Seg{seg_counter}",
                        "Lat": pts[pi, 0], "Lon": pts[pi, 1], "Alt_m": pts[pi, 2],
                        "Seg_Type": "TF", "TF_Start": is_start, "TF_End": is_end,
                        "RF_Start": "", "RF_End": "",
                        "Arc_Center_Lat": "", "Arc_Center_Lon": "",
                        "Turn_Radius_m": "", "Turn_Angle_deg": "",
                        "Ground_Speed_mps": ground_speed_mps, "Bank_Angle_deg": bank_angle_deg,
                    })
                    point_idx += 1
            elif stype == "RF":
                arc_center = seg["arc_center"]
                turn_angle_deg = float(np.rad2deg(seg["turn_angle"]))
                for pi in range(pts.shape[0]):
                    is_start = "O" if pi == 0 else ""
                    is_end = "O" if pi == pts.shape[0] - 1 else ""
                    arc_label = f"Arc_{pi+1}/{pts.shape[0]}"
                    rows.append({
                        "Point_No": point_idx, "Type": f"RF_Arc ({arc_label})",
                        "Segment": f"Seg{seg_counter}",
                        "Lat": pts[pi, 0], "Lon": pts[pi, 1], "Alt_m": pts[pi, 2],
                        "Seg_Type": "RF", "TF_Start": "", "TF_End": "",
                        "RF_Start": is_start, "RF_End": is_end,
                        "Arc_Center_Lat": arc_center[0], "Arc_Center_Lon": arc_center[1],
                        "Turn_Radius_m": R_turn, "Turn_Angle_deg": turn_angle_deg,
                        "Ground_Speed_mps": ground_speed_mps, "Bank_Angle_deg": bank_angle_deg,
                    })
                    point_idx += 1

        # Landing entry (NW, sector 11)
        rows.append({
            "Point_No": point_idx, "Type": "Landing_Entry", "Segment": "Transition",
            "Lat": landing_entry[0], "Lon": landing_entry[1], "Alt_m": landing_entry[2],
            "Seg_Type": "-", "TF_Start": "", "TF_End": "", "RF_Start": "", "RF_End": "",
            "Arc_Center_Lat": "", "Arc_Center_Lon": "",
            "Turn_Radius_m": "", "Turn_Angle_deg": "",
            "Ground_Speed_mps": ground_speed_mps, "Bank_Angle_deg": bank_angle_deg,
        })
        point_idx += 1

        # Vertiport (도착)
        rows.append({
            "Point_No": point_idx, "Type": "Vertiport", "Segment": "End",
            "Lat": end_vertiport[0], "Lon": end_vertiport[1], "Alt_m": end_vertiport[2],
            "Seg_Type": "-", "TF_Start": "", "TF_End": "", "RF_Start": "", "RF_End": "",
            "Arc_Center_Lat": "", "Arc_Center_Lon": "",
            "Turn_Radius_m": "", "Turn_Angle_deg": "",
            "Ground_Speed_mps": ground_speed_mps, "Bank_Angle_deg": bank_angle_deg,
        })

        df = pd.DataFrame(rows)

        # 노드 간 간격/누적거리(2D,3D) 계산
        dist_prev_2d = [0.0]
        dist_prev_3d = [0.0]
        cum_3d = [0.0]
        for i in range(1, len(df)):
            p_prev = np.array([
                float(df.loc[i - 1, "Lat"]), float(df.loc[i - 1, "Lon"]), float(df.loc[i - 1, "Alt_m"])
            ], dtype=float)
            p_cur = np.array([
                float(df.loc[i, "Lat"]), float(df.loc[i, "Lon"]), float(df.loc[i, "Alt_m"])
            ], dtype=float)
            d2 = _seg_dist_m(p_prev, p_cur)
            d3 = _seg_dist_3d_m(p_prev, p_cur)
            dist_prev_2d.append(d2)
            dist_prev_3d.append(d3)
            cum_3d.append(cum_3d[-1] + d3)

        df["Dist_From_Prev_2D_m"] = dist_prev_2d
        df["Dist_From_Prev_3D_m"] = dist_prev_3d
        df["Cumulative_Dist_3D_m"] = cum_3d

        # RF 최종 경로 전체 점(고도 포함) 시트
        full_path = build_full_corridor_path(start_vertiport, takeoff_complete, rf_best["path"], landing_entry, end_vertiport).astype(float)
        p2d = [0.0]
        p3d = [0.0]
        pcum = [0.0]
        for i in range(1, full_path.shape[0]):
            d2 = _seg_dist_m(full_path[i - 1], full_path[i])
            d3 = _seg_dist_3d_m(full_path[i - 1], full_path[i])
            p2d.append(d2)
            p3d.append(d3)
            pcum.append(pcum[-1] + d3)
        df_path = pd.DataFrame({
            "Point_No": np.arange(full_path.shape[0]),
            "Lat": full_path[:, 0],
            "Lon": full_path[:, 1],
            "Alt_m": full_path[:, 2],
            "Dist_From_Prev_2D_m": p2d,
            "Dist_From_Prev_3D_m": p3d,
            "Cumulative_Dist_3D_m": pcum,
        })

        pt_ground, pt_air, pt_combined = _sample_point_risks(
            full_path, Norm_RT, AirRisk, altitude_levels,
            use_heading_map, air_thr_global, lat_lim, lon_lim
        )
        df_path["Ground_Risk"] = pt_ground
        df_path["Air_Risk"] = pt_air
        df_path["Combined_Risk"] = pt_combined

        total_ground_risk, total_air_risk, total_combined_risk = _aggregate_path_risks(
            full_path, Norm_RT, AirRisk, altitude_levels,
            use_heading_map, cell_size, refine_scales,
            air_thr_global, lat_lim, lon_lim
        )
        total_corridor_dist_3d_m = float(pcum[-1]) if len(pcum) > 0 else 0.0
        total_corridor_dist_3d_km = total_corridor_dist_3d_m / 1000.0
        min_dist_ok = (min_corridor_distance_m <= 0.0) or (total_corridor_dist_3d_m + 1e-6 >= min_corridor_distance_m)

        df_summary = pd.DataFrame([
            {"Metric": "Total_Corridor_Distance_3D_m", "Value": total_corridor_dist_3d_m},
            {"Metric": "Total_Corridor_Distance_3D_km", "Value": total_corridor_dist_3d_km},
            {"Metric": "Total_Ground_Risk", "Value": total_ground_risk},
            {"Metric": "Total_Air_Risk", "Value": total_air_risk},
            {"Metric": "Total_Combined_Risk", "Value": total_combined_risk},
            {"Metric": "Min_Corridor_Distance_km_Param", "Value": min_corridor_distance_km},
            {"Metric": "Min_Corridor_Distance_Constraint_Active", "Value": int(min_corridor_distance_km > 0.0)},
            {"Metric": "Meets_Min_Corridor_Distance", "Value": int(min_dist_ok)},
        ])

        # 세그먼트별 후보 노드(고도 포함) 시트
        cand_rows = []
        for si, seg_nodes in enumerate(safe_nodes_by_seg, start=1):
            if seg_nodes.size == 0:
                continue
            for ni in range(seg_nodes.shape[0]):
                cand_rows.append({
                    "Segment": si,
                    "Node_No": ni,
                    "Lat": float(seg_nodes[ni, 0]),
                    "Lon": float(seg_nodes[ni, 1]),
                    "Alt_m": float(seg_nodes[ni, 2]),
                })
        df_nodes = pd.DataFrame(cand_rows)

        # 공역 정보 시트 (center + boundary LLA)
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

        # NFZ 정보 시트 (bbox + polygon LLA)
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

        xlsx_name = out_dir / "route_data.xlsx"
        with pd.ExcelWriter(str(xlsx_name)) as writer:
            df.to_excel(writer, index=False, sheet_name="Routes Data")
            df_path.to_excel(writer, index=False, sheet_name="Full_Path_Points")
            df_summary.to_excel(writer, index=False, sheet_name="Summary")
            df_airspace.to_excel(writer, index=False, sheet_name="Airspace_Info")
            df_nfz.to_excel(writer, index=False, sheet_name="NFZ_Info")
            if not df_nodes.empty:
                df_nodes.to_excel(writer, index=False, sheet_name="Candidate_Nodes")
        print(f"Saved {xlsx_name}")

    # ==================== 결과 저장 ====================
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
    }

    out = out_dir / "results.pkl"
    with open(out, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved {out}")

    return True, feasible_count


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
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
                print(f"Attempt {attempt} → 0 feasible. Retrying …")
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

