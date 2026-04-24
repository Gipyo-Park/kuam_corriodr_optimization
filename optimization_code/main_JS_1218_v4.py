import sys
import numpy as np

import numpy as np
import pickle
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
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


def draw_vertiport_radius_rings(gx, vertiport, radii_m=(4500.0, 5000.0, 5500.0), n_pts=240):
    if gx is None or vertiport is None:
        return
    lat0 = float(vertiport[0])
    lon0 = float(vertiport[1])
    m_lat = 111000.0
    m_lon = 111000.0 * np.cos(np.deg2rad(lat0))
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_pts), endpoint=True)

    styles = [
        (4500.0, "4.5 km Radius", "deepskyblue", 1.2, 0.55, 1),
        (5000.0, "5.0 km Radius", "navy", 1.0, 0.35, 1),
        (5500.0, "5.5 km Radius", "slateblue", 1.0, 0.28, 1),
    ]
    for rad_m, label, col, lw, alpha, zorder in styles:
        lat_ring = lat0 + (rad_m * np.sin(theta)) / m_lat
        lon_ring = lon0 + (rad_m * np.cos(theta)) / m_lon
        gx.plot(lon_ring, lat_ring, "-", color=col, linewidth=lw,
                alpha=alpha, transform=ccrs.Geodetic(), zorder=zorder, label=label)


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


def variation_nsga3(pop, nodes, ratio):
    if not pop:
        return []
    n_off = int(round(len(pop) * ratio))
    n = len(pop)
    offspring = []
    for _ in range(n_off):
        i1, i2 = (np.random.choice(n, 2, replace=False) if n >= 2 else (0, 0))
        child = crossover_gp(pop[i1], pop[i2])
        child = mutation_gp(child, nodes)
        offspring.append(child)
    return offspring


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


def filter_nodes_in_airspace(cand, center_latlon, radius_m):
    if cand is None or cand.size == 0:
        return cand
    d = _dist_to_center_m(cand[:, :2], center_latlon)
    return cand[d <= float(radius_m)]


def is_path_inside_airspace(path, center_latlon, radius_m):
    if path is None or path.size == 0:
        return False
    d = _dist_to_center_m(path[:, :2], center_latlon)
    return bool(np.all(d <= float(radius_m)))


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

    # WP 선택: takeoff + (선택적 중간 WPs) + landing
    selected_indices = [0]  # takeoff_wp (index 0 in our mapping)
    for i in range(1, M - 1):  # 중간 WPs
        if np.random.uniform() > wp_skip_prob:  # 스킵하지 않을 확률
            selected_indices.append(i)
    selected_indices.append(M - 1)  # landing_wp (index M-1)

    # 선택된 WP들로 backbone 구성
    selected_wps = np.vstack([takeoff_wp] + [full_waypoints[i] if i < M else landing_wp
                                              for i in selected_indices[1:-1]] + [landing_wp])
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


# ──────────────────────────────────────────────────────────────────
# NSGA-III 실행  (v3 – RF turn 적용 후 evaluation)
# ──────────────────────────────────────────────────────────────────
def run_nsga3(
    nodes_pool, population, N_pop, Nmax, ratio,
    # evaluation 공통 인자
    Norm_RT, AirRisk, use_map, f_limit, f_zones,
    alt, cs, scales, air_thr, dz,
    w_d, w_g, w_a, lat_lim, lon_lim,
    # RF turn 파라미터
    ground_speed_mps, bank_angle_deg, num_arc_points,
    look_ahead, look_ahead_threshold_m, look_ahead_min_scale, look_ahead_window,
    # 회랑폭 + vertiport
    W_half, check_corridor_nfz,
    vertiport, landing_entry, takeoff_complete,
    airspace_center_latlon, airspace_radius_m,
):
    """NSGA-III with RF-turn preprocessing."""

    # --- 목적함수 수 파악 ---
    dummy = np.vstack([population[0][0], population[0][-1]])
    temp_f, _ = evaluate_objectives_with_constraints_gp(
        dummy, Norm_RT, AirRisk, use_map, f_limit, f_zones, dz, alt, cs, scales,
        air_thr, w_d, w_g, w_a, lat_lim, lon_lim,
        W_half=W_half, check_corridor_nfz=check_corridor_nfz,
        vertiport=vertiport, landing_entry=landing_entry, takeoff_complete=takeoff_complete,
    )
    num_obj = len(temp_f)
    H = num_obj + 1
    ref_points = generate_reference_points(num_obj, H)

    def _evaluate_one(chromo):
        # RF turn 적용
        rf = apply_rf_turns(chromo, ground_speed_mps, bank_angle_deg, num_arc_points,
                            look_ahead=look_ahead,
                            look_ahead_threshold_m=look_ahead_threshold_m,
                            look_ahead_min_scale=look_ahead_min_scale,
                            look_ahead_window=look_ahead_window)
        rf_path = rf["path"]
        full_path = np.vstack([vertiport, takeoff_complete, rf_path, landing_entry, vertiport])
        if not is_path_inside_airspace(full_path, airspace_center_latlon, airspace_radius_m):
            f_pen = np.asarray(temp_f, dtype=float) + 1e6
            return f_pen, False

        f_val, feas = evaluate_objectives_with_constraints_gp(
            rf_path, Norm_RT, AirRisk, use_map, f_limit, f_zones, dz, alt, cs, scales,
            air_thr, w_d, w_g, w_a, lat_lim, lon_lim,
            W_half=W_half, check_corridor_nfz=check_corridor_nfz,
            vertiport=vertiport, landing_entry=landing_entry, takeoff_complete=takeoff_complete,
        )
        if not rf["feasible"]:
            # RF clamping이 필요한 해는 불리하게 두되, 완전 배제하지는 않음
            f_val = np.asarray(f_val, dtype=float) + 1e6
        return f_val, feas

    pop = list(population[:N_pop]) if len(population) > N_pop else list(population)

    for gen in range(1, Nmax + 1):
        Np = len(pop)
        f_vals = np.zeros((Np, num_obj), dtype=float)
        feasible = np.zeros(Np, dtype=bool)

        for i in range(Np):
            f_vals[i], feasible[i] = _evaluate_one(pop[i])

        num_feas = int(np.sum(feasible))
        rf_feas = int(np.sum(f_vals[:, 0] < 1e6))  # +1e6 패널티 없는 = RF 기하학적 feasible
        print(f"[Gen {gen}] pop {Np}  |  constraint_feasible: {num_feas}/{Np}  |  RF_feasible: {rf_feas}/{Np}")

        new_pop = selection_nsga3(pop, f_vals, feasible, N_pop, ref_points)
        if not new_pop:
            break

        if gen < Nmax:
            offspring = variation_nsga3(new_pop, nodes_pool, ratio)
            pop = new_pop + offspring
        else:
            pop = new_pop

    # 최종 평가
    if not pop:
        return [], np.empty((0, num_obj))

    f_final = np.zeros((len(pop), num_obj), dtype=float)
    for i in range(len(pop)):
        f_final[i], _ = _evaluate_one(pop[i])

    return pop, f_final


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
    ground_speed_mps = 83.34        # 최대 300 km/h → 83.34m/s RF turn 시 속도 (m/s)
    bank_angle_deg = 25.0     # RF turn 시 최대 bank angle (degree) → turn radius 결정
    num_arc_points = 30          # RF turn 시 원호를 몇 점으로 표현할지 (너무 많으면 느려짐)

    check_corridor_nfz = True

    N_init = 100    # 초기해 개수 
    N_pop = 10      # 세대별 유지할 해 개수 (인구 수)
    Nmax = 10       # 최대 세대 수
    offspring_ratio = 0.5

    wp_perturb_radius_m = 100.0     # WP 교란 반경 (m)
    wp_perturb_steps = 10           # WP 교란 반복 횟수 (1=기존과 동일)
    min_extra_nodes_per_seg = 0     # 세그먼트당 최소 추가 노드 수 (int 또는 list)
    max_extra_nodes_per_seg = 2     # 세그먼트당 최대 추가 노드 수 (int 또는 list)
    wp_skip_prob = 0.00             # 중간 WP 스킵 확률 (0~1)
    airspace_radius_km = 5.0       # 공역 반경 제약 (km) — 회랑은 이 반경 안에 있어야 함
    emergency_strip_m = 500.0       # emergency 포함 판별 strip 폭
    min_seg_for_extra_nodes_m = 2000.0  # 이보다 짧은 WP간 세그먼트에는 extra node 생성 안 함

    # ── RF Look-ahead 튜닝 파라미터 ──
    look_ahead = True            # True: 짧은 세그먼트 코너에서 turn radius 자동 축소
    look_ahead_threshold_m = 2000.0  # 이보다 짧은 세그먼트 → R 스케일 다운 시작 (m)
    look_ahead_min_scale = 0.3   # R 축소 비율 (0.3 → 최대 70% 축소)
    look_ahead_window = 3        # 코너 전후 몇 개 세그먼트까지 볼지 (1=바로 앞뒤만, 2=2개씩 ...)
    max_init_retries = 300       # Gen 1에서 RF+constraint feasible 해 없으면 최대 이 횟수만큼 초기 해 재생성

    w_dist, w_ground, w_air = 0.5, 2.0, 2.0
    altitude_levels = np.array([600.0], dtype=float)
    use_heading_map = True
    W_buf = 1250.0
    node_grid_resolution_m = 100.0

    MIN_SAFE_NODES_TARGET = 200
    SAFE_NODE_PERCENTILE_LIST = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]

    cell_size = 100.0
    refine_scales = np.array([1.0, 0.5, 0.2, 0.1])
    # vertiport(지상) ↔ takeoff/landing(순항) 전환을 허용하도록 동적으로 완화
    delta_z_max = max(100.0, float(np.max(np.abs(altitude_levels - 150.0))) + 5.0)
    flight_dist_limit = 100000.0
    objective_names = ["Distance", "Ground Risk", "Air Risk"]
    airspace_radius_m = float(airspace_radius_km) * 1000.0

    # ==================== 출력 폴더 생성 ====================
    import pathlib, json
    import datetime as _dt
    _run_ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path("runs") / _run_ts
    out_dir.mkdir(parents=True, exist_ok=True)

    params_dict = {
        "run_timestamp": _run_ts,
        "W_half": W_half,
        "ground_speed_mps": ground_speed_mps,
        "bank_angle_deg": bank_angle_deg,
        "num_arc_points": num_arc_points,
        "N_init": N_init,
        "N_pop": N_pop,
        "Nmax": Nmax,
        "offspring_ratio": offspring_ratio,
        "wp_perturb_radius_m": wp_perturb_radius_m,
        "wp_perturb_steps": wp_perturb_steps,
        "min_extra_nodes_per_seg": min_extra_nodes_per_seg,
        "max_extra_nodes_per_seg": max_extra_nodes_per_seg,
        "emergency_strip_m": emergency_strip_m,
        "min_seg_for_extra_nodes_m": min_seg_for_extra_nodes_m,
        "look_ahead": look_ahead,
        "look_ahead_threshold_m": look_ahead_threshold_m,
        "look_ahead_min_scale": look_ahead_min_scale,
        "look_ahead_window": look_ahead_window,
        "max_init_retries": max_init_retries,
        "w_dist": w_dist,
        "w_ground": w_ground,
        "w_air": w_air,
        "altitude_levels_m": altitude_levels.tolist(),
        "W_buf": W_buf,
        "node_grid_resolution_m": node_grid_resolution_m,
        "cell_size": cell_size,
        "delta_z_max": delta_z_max,
        "flight_dist_limit": flight_dist_limit,
        "check_corridor_nfz": check_corridor_nfz,
        "wp_skip_prob": wp_skip_prob,
        "airspace_radius_km": airspace_radius_km,
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
    # lat이 세로축 -> + 위쪽, - 아래쪽 
    # lon이 가로축 -> + 오른쪽, - 왼쪽
    # vertiport = np.array([35.6061306, 129.0759861, 150.0])
    vertiport = np.array([35.6033361, 129.0776917, 150.0])

    # corridor_lat = np.array([35.5845917, 35.6026528, 35.6326806, 35.6249583, 35.6034750,
    #                         35.5845361, 35.5692361, 35.5546444, 35.5586722, 35.5784750,
    #                         35.5843722, 35.6163861, 35.6109972])
    
    # Updated waypoints (28 points)
    # corridor_lat = np.array([
    #     35.5966656, 35.5944587, 35.5987011, 35.6086321, 35.6195580,
    #     35.6218142, 35.5994190, 35.5920136, 35.5867976, 35.5793805,
    #     35.5724395, 35.5653508, 35.5595903, 35.5671301, 35.5767878,
    #     35.5820196, 35.5852538, 35.5881860, 35.5887324, 35.5931814,
    #     35.5995350, 35.6029908, 35.6140262, 35.6168624, 35.6185184,
    #     35.6249109, 35.6231928, 35.6204656
    # ])
    # corridor_lat = np.array([
    #     35.5944587, 35.6195580,
    #     35.6218142,
    #     35.5595903, 35.5671301,
    #     35.5887324, 35.5931814,
    #     35.6185184,
    #     35.6249109
    # ])
    corridor_lat = np.array([
        35.5944587, 35.6195580,
        35.6218142,
        35.5625903, 35.5671301,
        35.5887324, 35.5931814,
        35.6185184,
        35.6249109
    ])

    # corridor_lon = np.array([129.0936472, 129.1130667, 129.1238583, 129.1335528, 129.1268194,
    #                         129.1076472, 129.1085306, 129.0936972, 129.0816611, 129.0916889,
    #                         129.0770000, 129.0613944, 129.0711889])

    # Updated waypoints (28 points)
    # corridor_lon = np.array([
    #     129.0860053, 129.0977958, 129.1070368, 129.1146351, 129.1153758,
    #     129.1266116, 129.1286799, 129.1233059, 129.1092159, 129.1031010,
    #     129.1027601, 129.0975048, 129.0849662, 129.0776521, 129.0899711,
    #     129.0904238, 129.0876678, 129.0812627, 129.0691071, 129.0665565,
    #     129.0723625, 129.0725487, 129.0631380, 129.0583642, 129.0512209,
    #     129.0536710, 129.0610089, 129.0657129
    # ])
    # corridor_lon = np.array([
    #     129.0977958, 129.1153758,
    #     129.1266116,
    #     129.0849662, 129.0776521,
    #     129.0691071, 129.0665565,
    #     129.0512209,
    #     129.0536710
    # ])
    corridor_lon = np.array([
        129.0977958, 129.1153758,
        129.1266116,
        129.0949662, 129.0776521,
        129.0691071, 129.0665565,
        129.0512209,
        129.0586710
    ])

    if altitude_levels.size == 1:
        waypoint_alts = np.full(corridor_lat.shape[0], float(altitude_levels[0]), dtype=float)
    else:
        waypoint_alts = np.random.choice(altitude_levels, size=corridor_lat.shape[0]).astype(float)

    waypoints = np.column_stack([corridor_lat, corridor_lon, waypoint_alts])

    takeoff_target_alt = float(waypoints[0, 2])
    landing_target_alt = float(waypoints[-1, 2])

    takeoff_complete, landing_entry, _ = build_takeoff_landing(
        vertiport, angle_deg=25.0, alt_delta_m=350.0,
        takeoff_sector=6, landing_sector=11,
        takeoff_target_alt_m=takeoff_target_alt,
        landing_target_alt_m=landing_target_alt,
    )

    # backbone: takeoff_complete(SSE) → WPs(원래 순서, CW) → landing_entry(NW)
    # sector 6 = SSE (이륙), sector 11 = NW (착륙)
    # WP 순서대로 이동하면 CW 회로가 됨
    backbone = np.vstack([takeoff_complete, waypoints, landing_entry])
    is_fixed = np.zeros(backbone.shape[0], dtype=bool)
    is_fixed[0]  = True   # takeoff_complete 고정
    is_fixed[-1] = True   # landing_entry 고정

    emergency_lat = np.array([35.6201083, 35.5678222, 35.5919889])
    emergency_lon = np.array([129.1191806, 129.106728, 129.0751972])
    if altitude_levels.size == 1:
        emergency_alts = np.full(emergency_lat.shape[0], float(altitude_levels[0]), dtype=float)
    else:
        emergency_alts = np.random.choice(altitude_levels, size=emergency_lat.shape[0]).astype(float)
    emergency_points = np.column_stack([emergency_lat, emergency_lon, emergency_alts])
    emergency_points = filter_nodes_in_airspace(emergency_points, vertiport[:2], airspace_radius_m)

    forbidden_zones = np.array([], dtype=float).reshape(0, 4)
    # forbidden_zones = np.array([
    #     [129.08, 129.10, 35.59, 35.61],
    #     [129.11, 129.118, 35.62, 35.63],
    #     [129.12, 129.13, 35.59, 35.60]
    # ], dtype=float)

    lat_lim = [35.5446, 35.6427]
    lon_lim = [129.0514, 129.1436]

    # ==================== params.json 좌표/지역 정보 추가 저장 ====================
    params_dict.update({
        "vertiport": {
            "lat": float(vertiport[0]),
            "lon": float(vertiport[1]),
            "alt_m": float(vertiport[2]),
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
        nodes_seg = filter_nodes_in_airspace(nodes_seg, vertiport[:2], airspace_radius_m)
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
                 if any(s.size > 0 for s in safe_nodes_by_seg) else emergency_points

    # ==================== 초기 해 생성 (RF+constraint feasible 해 확인될 때까지 재시도) ====================
    print(f"Searching for RF+constraint feasible initial pop (max {max_init_retries} retries, N_init={N_init}) ...")

    def _make_init_pop():
        """N_init 개의 초기 해를 새로 생성 (절반 all-WP, 절반 WP-skip)"""
        _pop = []
        max_draws = int(max(3 * N_init, 50))
        draws = 0
        while len(_pop) < N_init and draws < max_draws:
            _i = len(_pop)
            draws += 1
            if _i < N_init // 2:
                _sol = generate_single_initial_solution(
                    backbone, wp_perturb_radius_m,
                    min_extra_nodes_per_seg, max_extra_nodes_per_seg,
                    safe_nodes_by_seg, emergency_points, emergency_strip_m,
                    is_fixed,
                    wp_perturb_steps=wp_perturb_steps,
                    min_seg_for_extra_nodes_m=min_seg_for_extra_nodes_m,
                )
            else:
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
            if is_path_inside_airspace(_sol, vertiport[:2], airspace_radius_m):
                _pop.append(_sol)

        if len(_pop) < N_init:
            print(f"  [Init] Airspace-filtered initial pop: {len(_pop)}/{N_init}")
        return _pop

    init_pop = None
    _last_candidate = None
    for _retry in range(1, max_init_retries + 1):
        _candidate = _make_init_pop()
        _last_candidate = _candidate

        # 1단계: RF 기하학적 feasibility 체크 (cheap — risk map 조회 없음)
        _rf_ok_list = []
        for _c in _candidate:
            _rf = apply_rf_turns(_c, ground_speed_mps, bank_angle_deg, num_arc_points,
                                 look_ahead=look_ahead,
                                 look_ahead_threshold_m=look_ahead_threshold_m,
                                 look_ahead_min_scale=look_ahead_min_scale,
                                 look_ahead_window=look_ahead_window)
            _rf_ok_list.append((_rf["feasible"], _rf["path"]))
        _rf_cnt = sum(1 for ok, _ in _rf_ok_list if ok)

        # 2단계: RF feasible 해에 대해서만 constraint 체크 (expensive)
        _both_cnt = 0
        if _rf_cnt > 0:
            for _rf_ok, _rf_path in _rf_ok_list:
                if not _rf_ok:
                    continue
                _, _cst_ok = evaluate_objectives_with_constraints_gp(
                    _rf_path, Norm_RT, AirRisk, use_heading_map,
                    flight_dist_limit, forbidden_zones, delta_z_max,
                    altitude_levels, cell_size, refine_scales, air_thr_global,
                    w_dist, w_ground, w_air, lat_lim, lon_lim,
                    W_half=W_half, check_corridor_nfz=check_corridor_nfz,
                    vertiport=vertiport, landing_entry=landing_entry,
                    takeoff_complete=takeoff_complete,
                )
                full_path = np.vstack([vertiport, takeoff_complete, _rf_path, landing_entry, vertiport])
                _air_ok = is_path_inside_airspace(full_path, vertiport[:2], airspace_radius_m)
                if _cst_ok and _air_ok:
                    _both_cnt += 1

        # 진행 상황 출력 (50회마다 또는 RF feasible 발견 시)
        if _retry % 50 == 0 or _rf_cnt > 0:
            print(f"  [Init retry {_retry}/{max_init_retries}]  RF_feasible: {_rf_cnt}/{N_init},  both_feasible: {_both_cnt}/{N_init}")

        if _both_cnt > 0:
            init_pop = _candidate
            print(f"  -> {_both_cnt} RF+constraint feasible solution(s) found at retry {_retry}. Proceeding to NSGA-III.")
            break

    if init_pop is None:
        print(f"  Warning: No RF+constraint feasible found after {max_init_retries} retries. Using last candidate.")
        init_pop = _last_candidate
    print(f"  -> {len(init_pop)} initial solutions ready.")

    # ==================== 지도 공통 설정 ====================
    request = cimgt.OSM()
    extent_points = [vertiport[:2], takeoff_complete[:2], landing_entry[:2]]
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

    map_extent = compute_centered_map_extent(np.array(extent_points, dtype=float), vertiport,
                                             ring_radii_m=(4500.0, 5000.0, 5500.0), pad_ratio=0.10)

    bb_full = np.vstack([vertiport, backbone, vertiport])
    total_safe_count = sum(s.shape[0] for s in safe_nodes_by_seg if s.size > 0)

    # ==================== [Fig 1] 후보 노드(Safe Node) 분포 ====================
    fig1 = plt.figure("Figure 1: Candidate Safe Nodes", figsize=(14, 10))
    fig1.subplots_adjust(left=0.08, right=0.78)
    gx1 = fig1.add_subplot(1, 1, 1, projection=request.crs)
    gx1.set_extent(map_extent)
    gx1.add_image(request, 13)
    gx1.set_title(f"Candidate Safe Nodes  (total {total_safe_count} nodes,  grid {node_grid_resolution_m}m,  W_buf {W_buf}m)")
    draw_vertiport_radius_rings(gx1, vertiport)

    # backbone
    gx1.plot(bb_full[:, 1], bb_full[:, 0], "r--", linewidth=2, transform=ccrs.Geodetic(),
             label="Backbone", zorder=5)
    gx1.scatter(waypoints[:, 1], waypoints[:, 0], s=60, c="orange", edgecolors="k",
                linewidths=0.5, marker="o", transform=ccrs.Geodetic(), label="Waypoints (WP)", zorder=6)
    gx1.scatter([vertiport[1]], [vertiport[0]], s=120, c="red", edgecolors="k",
                marker="s", transform=ccrs.Geodetic(), label="Vertiport", zorder=7)
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
    draw_vertiport_radius_rings(gx2, vertiport)

    gx2.plot(bb_full[:, 1], bb_full[:, 0], "r--", linewidth=1.5, transform=ccrs.Geodetic(),
             label="Backbone", zorder=4)
    gx2.scatter([vertiport[1]], [vertiport[0]], s=120, c="red", edgecolors="k",
                marker="s", transform=ccrs.Geodetic(), label="Vertiport", zorder=8)
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

    # ==================== NSGA-III 최적화 ====================
    print("Running NSGA-III …")
    pop, fvals = run_nsga3(
        nodes_pool=nodes_pool,
        population=init_pop,
        N_pop=N_pop, Nmax=Nmax, ratio=offspring_ratio,
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
        vertiport=vertiport, landing_entry=landing_entry,
        takeoff_complete=takeoff_complete,
        airspace_center_latlon=vertiport[:2],
        airspace_radius_m=airspace_radius_m,
    )

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
        _, feas = evaluate_objectives_with_constraints_gp(
            rf["path"], Norm_RT, AirRisk, use_heading_map,
            flight_dist_limit, forbidden_zones, delta_z_max,
            altitude_levels, cell_size, refine_scales,
            air_thr_global, w_dist, w_ground, w_air, lat_lim, lon_lim,
            W_half=W_half, check_corridor_nfz=check_corridor_nfz,
            vertiport=vertiport, landing_entry=landing_entry,
            takeoff_complete=takeoff_complete,
        )
        full_path = np.vstack([vertiport, takeoff_complete, rf["path"], landing_entry, vertiport])
        air_ok = is_path_inside_airspace(full_path, vertiport[:2], airspace_radius_m)
        feas = bool(feas and air_ok)
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
                    rf_rep["path"], Norm_RT, AirRisk, use_heading_map,
                    flight_dist_limit, forbidden_zones, delta_z_max,
                    altitude_levels, cell_size, refine_scales,
                    air_thr_global, w_dist, w_ground, w_air, lat_lim, lon_lim,
                    W_half=W_half, check_corridor_nfz=check_corridor_nfz,
                    vertiport=vertiport, landing_entry=landing_entry,
                    takeoff_complete=takeoff_complete,
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
    draw_vertiport_radius_rings(gx4, vertiport)

    # backbone
    gx4.plot(bb_full[:, 1], bb_full[:, 0], "r--", linewidth=1.5, transform=ccrs.Geodetic(),
             label="Backbone", zorder=4)
    gx4.scatter(waypoints[:, 1], waypoints[:, 0], s=60, c="orange", edgecolors="k",
                linewidths=0.5, marker="o", transform=ccrs.Geodetic(), label="Waypoints", zorder=6)
    gx4.scatter([vertiport[1]], [vertiport[0]], s=120, c="red", edgecolors="k",
                marker="s", transform=ccrs.Geodetic(), label="Vertiport", zorder=7)
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

        # corridor width: vertiport → takeoff(SSE) → path → landing(NW) → vertiport
        full_path = np.vstack([vertiport, takeoff_complete, rp, landing_entry, vertiport])
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
            "Lat": vertiport[0], "Lon": vertiport[1], "Alt_m": vertiport[2],
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
            "Lat": vertiport[0], "Lon": vertiport[1], "Alt_m": vertiport[2],
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
        full_path = np.vstack([vertiport, takeoff_complete, rf_best["path"], landing_entry, vertiport]).astype(float)
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

        xlsx_name = out_dir / "route_data.xlsx"
        with pd.ExcelWriter(str(xlsx_name)) as writer:
            df.to_excel(writer, index=False, sheet_name="Routes Data")
            df_path.to_excel(writer, index=False, sheet_name="Full_Path_Points")
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
        "vertiport": vertiport,
        "takeoff_complete": takeoff_complete,
        "landing_entry": landing_entry,
        "forbidden_zones": forbidden_zones,
        "emergency_points": emergency_points,
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
    os.makedirs("runs", exist_ok=True)

    attempt = 1
    while True:
        ok, feas = attempt_run_once()
        if ok:
            print(f"Success on attempt {attempt}. Feasible: {feas}")
            break
        else:
            print(f"Attempt {attempt} → 0 feasible. Retrying …")
        attempt += 1
