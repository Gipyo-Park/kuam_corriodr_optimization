import numpy as np


def _latlon_to_local_m(points, ref_lat_deg, ref_lon_deg):
    """lat/lon/alt -> local x(east), y(north), z(up)  (metres)"""
    m_per_lat = 111000.0
    m_per_lon = 111000.0 * np.cos(np.deg2rad(ref_lat_deg))
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]
    xy = np.empty((pts.shape[0], 3), dtype=float)
    xy[:, 0] = (pts[:, 1] - ref_lon_deg) * m_per_lon   # x = east
    xy[:, 1] = (pts[:, 0] - ref_lat_deg) * m_per_lat   # y = north
    xy[:, 2] = pts[:, 2]                                 # z = alt
    return xy


def _local_m_to_latlon(xy, ref_lat_deg, ref_lon_deg):
    """local x(east), y(north), z(up) -> lat/lon/alt"""
    m_per_lat = 111000.0
    m_per_lon = 111000.0 * np.cos(np.deg2rad(ref_lat_deg))
    pts = np.asarray(xy, dtype=float)
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]
    ll = np.empty_like(pts)
    ll[:, 0] = ref_lat_deg + pts[:, 1] / m_per_lat
    ll[:, 1] = ref_lon_deg + pts[:, 0] / m_per_lon
    ll[:, 2] = pts[:, 2]
    return ll


def apply_rf_turns(waypoints, ground_speed_mps, bank_angle_deg, num_arc_points=30,
                   look_ahead=True, look_ahead_threshold_m=1500.0,
                   look_ahead_min_scale=0.4, look_ahead_window=1):
    """
    RF (Radius-to-Fix) 선회를 WP 배열에 적용하여
    직선 구간(TF)과 호 구간(RF)이 연속적으로 이어진 경로를 반환한다.
    Dubins path 처럼 각 WP 에서 부드럽게 선회한다.

    look_ahead : bool
        True이면 인접 세그먼트 길이를 보고 turn radius를 동적으로 스케일
        (긴 세그먼트 양쪽 = R 유지, 짧은 세그먼트 = R 감소)
    look_ahead_threshold_m : float
        이 값보다 짧은 세그먼트가 있으면 turn radius 스케일 다운
    look_ahead_window : int
        각 코너에서 전후 몇 개 세그먼트까지 볼지 (기본 1 = 바로 앞뒤 1개씩)

    Returns
    -------
    dict with keys:
        path         : (M, 3) 최종 연속 경로 [lat, lon, alt]
        segments     : list[dict]  — type='TF'|'RF', points, (arc_center, turn_radius, turn_angle)
        turn_radius_m: float
        ground_speed_mps: float
        feasible     : bool — 모든 RF 선회가 기하학적으로 실현 가능한지
                              (clamping 없이 완벽히 맞으면 True)
    """
    wps = np.asarray(waypoints, dtype=float)
    N = wps.shape[0]

    g = 9.80665
    phi = np.deg2rad(bank_angle_deg)
    R = (ground_speed_mps ** 2) / (g * np.tan(phi))

    if N < 2:
        return {"path": wps.copy(), "segments": [],
                "turn_radius_m": R, "ground_speed_mps": ground_speed_mps,
                "feasible": True}
    if N == 2:
        seg = {"type": "TF", "points": wps.copy()}
        return {"path": wps.copy(), "segments": [seg],
                "turn_radius_m": R, "ground_speed_mps": ground_speed_mps,
                "feasible": True}

    # 좌표 기준점
    ref_lat = float(np.mean(wps[:, 0]))
    ref_lon = float(np.mean(wps[:, 1]))
    local = _latlon_to_local_m(wps, ref_lat, ref_lon)  # (N, 3)

    # 세그먼트 길이
    seg_lengths = np.array([
        np.linalg.norm(local[i + 1, :2] - local[i, :2]) for i in range(N - 1)
    ])

    # ── Look-ahead: 인접 세그먼트 길이 보고 각 코너별 turn radius 스케일 ──
    # look_ahead_window: 코너 i 에서 전/후 각각 최대 window 개 세그먼트를 참조
    radius_scales = np.ones(N, dtype=float)  # [1,1,...,1]
    if look_ahead and N >= 3:
        w = max(1, int(look_ahead_window))
        for i in range(1, N - 1):
            # seg_lengths[i-1]: WP[i-1]→WP[i], seg_lengths[i]: WP[i]→WP[i+1]
            before_segs = seg_lengths[max(0, i - w): i]          # 앞쪽 w개
            after_segs  = seg_lengths[i: min(N - 1, i + w)]      # 뒤쪽 w개
            all_segs    = np.concatenate([before_segs, after_segs])
            min_seg     = float(np.min(all_segs)) if len(all_segs) > 0 else float('inf')

            if min_seg < look_ahead_threshold_m:
                scale = max(look_ahead_min_scale, min_seg / look_ahead_threshold_m)
                radius_scales[i] = scale

    # 각 코너별로 동적 R 계산
    R_by_corner = R * radius_scales  # (N,) 배열


    # 각 중간 WP 에서 heading 변화량 · tangent 거리
    tangent_dists = np.zeros(N, dtype=float)
    turn_angles   = np.zeros(N, dtype=float)
    turn_dirs     = np.zeros(N, dtype=float)   # +1=좌(CCW), -1=우(CW)

    for i in range(1, N - 1):
        v_in  = local[i, :2] - local[i - 1, :2]
        v_out = local[i + 1, :2] - local[i, :2]
        n_in  = np.linalg.norm(v_in)
        n_out = np.linalg.norm(v_out)
        if n_in < 1e-6 or n_out < 1e-6:
            continue

        cos_inner = np.clip(np.dot(v_in, v_out) / (n_in * n_out), -1.0, 1.0)
        delta = np.arccos(cos_inner)   # heading change (0 = straight, π = U-turn)

        if abs(delta) < np.deg2rad(0.5):
            continue

        cross = float(v_in[0] * v_out[1] - v_in[1] * v_out[0])
        turn_dirs[i] = 1.0 if cross >= 0 else -1.0

        # ─── 동적 turn radius 사용 ───
        R_i = R_by_corner[i]  # look-ahead 기반 스케일된 R
        td = R_i * np.tan(abs(delta) / 2.0)
        tangent_dists[i] = td
        turn_angles[i] = delta

    # ── tangent distance clamping ──
    # 세그먼트 j: tangent_dists[j] + tangent_dists[j+1] ≤ seg_lengths[j]
    # 초과하면 비율로 줄여서 항상 연속 경로를 만든다.
    feasible = True
    available = seg_lengths.copy()  # 각 세그먼트에서 남은 길이
    for i in range(1, N - 1):
        if tangent_dists[i] < 1e-6:
            continue
        max_td = min(available[i - 1], available[i]) * 0.95
        if max_td < 1.0:
            max_td = 1.0
        if tangent_dists[i] > max_td:
            tangent_dists[i] = max_td
            feasible = False
        available[i - 1] -= tangent_dists[i]
        available[i]     -= tangent_dists[i]

    # ── 연속 경로 조립: TF → RF → TF → RF → … → TF ──
    segments = []
    path_points = []

    for i in range(N - 1):
        #  TF 시작점
        if tangent_dists[i] > 1e-6:
            u_out = local[i + 1, :2] - local[i, :2]
            u_out = u_out / np.linalg.norm(u_out)
            tf_start_2d = local[i, :2] + u_out * tangent_dists[i]
        else:
            tf_start_2d = local[i, :2].copy()

        #  TF 끝점
        if tangent_dists[i + 1] > 1e-6:
            u_in = local[i + 1, :2] - local[i, :2]
            u_in = u_in / np.linalg.norm(u_in)
            tf_end_2d = local[i + 1, :2] - u_in * tangent_dists[i + 1]
        else:
            tf_end_2d = local[i + 1, :2].copy()

        tf_start_3d = np.array([tf_start_2d[0], tf_start_2d[1], local[i, 2]])
        tf_end_3d   = np.array([tf_end_2d[0], tf_end_2d[1], local[i + 1, 2]])
        tf_pts_ll = _local_m_to_latlon(np.vstack([tf_start_3d, tf_end_3d]), ref_lat, ref_lon)

        segments.append({"type": "TF", "points": tf_pts_ll})
        path_points.append(tf_pts_ll[0])

        #  RF 호 (WP[i+1] 에서 선회가 필요하면)
        if tangent_dists[i + 1] > 1e-6 and (i + 1) < N - 1:
            wp_2d = local[i + 1, :2]
            v_in_vec  = wp_2d - local[i, :2]
            v_out_vec = local[i + 2, :2] - wp_2d
            u_in_n  = v_in_vec / np.linalg.norm(v_in_vec)
            u_out_n = v_out_vec / np.linalg.norm(v_out_vec)

            td = tangent_dists[i + 1]
            arc_start_2d = wp_2d - u_in_n * td
            arc_end_2d   = wp_2d + u_out_n * td

            direction = turn_dirs[i + 1]
            # ─── look-ahead 기반 스케일 R 사용 ───
            R_arc = R_by_corner[i + 1]  # 동적 turn radius
            if direction >= 0:
                perp = np.array([-u_in_n[1], u_in_n[0]])
            else:
                perp = np.array([ u_in_n[1], -u_in_n[0]])

            center_2d = arc_start_2d + perp * R_arc

            ang_start = np.arctan2(arc_start_2d[1] - center_2d[1],
                                   arc_start_2d[0] - center_2d[0])
            ang_end   = np.arctan2(arc_end_2d[1] - center_2d[1],
                                   arc_end_2d[0] - center_2d[0])

            if direction >= 0:
                if ang_end <= ang_start:
                    ang_end += 2.0 * np.pi
            else:
                if ang_end >= ang_start:
                    ang_end -= 2.0 * np.pi

            thetas = np.linspace(ang_start, ang_end, num_arc_points)
            arc_local = np.zeros((num_arc_points, 3), dtype=float)
            arc_local[:, 0] = center_2d[0] + R_arc * np.cos(thetas)
            arc_local[:, 1] = center_2d[1] + R_arc * np.sin(thetas)
            arc_local[:, 2] = np.linspace(local[i + 1, 2], local[i + 1, 2], num_arc_points)

            arc_ll = _local_m_to_latlon(arc_local, ref_lat, ref_lon)

            center_3d = np.array([center_2d[0], center_2d[1], local[i + 1, 2]])
            center_ll = _local_m_to_latlon(center_3d, ref_lat, ref_lon).ravel()

            segments.append({
                "type": "RF",
                "points": arc_ll,
                "turn_radius": R_arc,
                "arc_center": center_ll,
                "turn_angle": turn_angles[i + 1],
            })

            for k in range(num_arc_points):
                path_points.append(arc_ll[k])

    # 마지막 TF end 추가
    if segments and segments[-1]["type"] == "TF":
        path_points.append(segments[-1]["points"][-1])

    path_array = np.array(path_points, dtype=float)

    return {
        "path": path_array,
        "segments": segments,
        "turn_radius_m": R,
        "ground_speed_mps": ground_speed_mps,
        "feasible": feasible,
    }
