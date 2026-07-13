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


def _heading_deg_to_unit_xy(heading_deg):
    heading_rad = np.deg2rad(float(heading_deg))
    return np.array([np.sin(heading_rad), np.cos(heading_rad)], dtype=float)


def apply_rf_turns(waypoints, ground_speed_mps, bank_angle_deg, num_arc_points=30,
                   look_ahead=True, look_ahead_threshold_m=1500.0,
                   look_ahead_min_scale=0.4, look_ahead_window=1,
                   entry_heading_deg=None, exit_heading_deg=None,
                   allow_tangent_clamp=True,
                   corner_fit_margin=0.95,
                   corner_min_tangent_m=1.0,
                   min_turn_angle_deg=0.5,
                   rf_debug_level="off"):
    """
    Apply RF (Radius-to-Fix) turns to waypoint sequence and return a continuous path.

    Boundary headings (entry/exit) are applied directly to the first/last cruise
    corner geometry without synthetic anchor points.
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

    ref_lat = float(np.mean(wps[:, 0]))
    ref_lon = float(np.mean(wps[:, 1]))
    local = _latlon_to_local_m(wps, ref_lat, ref_lon)

    seg_lengths = np.array([
        np.linalg.norm(local[i + 1, :2] - local[i, :2]) for i in range(N - 1)
    ], dtype=float)

    # Per-corner look-ahead scaling for corners at WP[1]..WP[N-2]
    radius_scales = np.ones(N, dtype=float)
    if look_ahead and N >= 3:
        w = max(1, int(look_ahead_window))
        for i in range(1, N - 1):
            before_segs = seg_lengths[max(0, i - w): i]
            after_segs = seg_lengths[i: min(N - 1, i + w)]
            all_segs = np.concatenate([before_segs, after_segs])
            min_seg = float(np.min(all_segs)) if all_segs.size > 0 else float("inf")
            if min_seg < look_ahead_threshold_m:
                scale = max(look_ahead_min_scale, min_seg / look_ahead_threshold_m)
                radius_scales[i] = float(scale)

    R_by_corner = R * radius_scales

    tangent_dists = np.zeros(N, dtype=float)
    turn_angles = np.zeros(N, dtype=float)
    turn_dirs = np.zeros(N, dtype=float)  # +1=CCW, -1=CW
    u_in_by_corner = np.zeros((N, 2), dtype=float)
    u_out_by_corner = np.zeros((N, 2), dtype=float)

    # Build corner geometry at waypoint i in [1, N-2]
    for i in range(1, N - 1):
        if i == 1 and entry_heading_deg is not None:
            u_in = _heading_deg_to_unit_xy(entry_heading_deg)
        else:
            v_in = local[i, :2] - local[i - 1, :2]
            n_in = np.linalg.norm(v_in)
            if n_in < 1e-6:
                continue
            u_in = v_in / n_in

        if i == N - 2 and exit_heading_deg is not None:
            u_out = _heading_deg_to_unit_xy(exit_heading_deg)
        else:
            v_out = local[i + 1, :2] - local[i, :2]
            n_out = np.linalg.norm(v_out)
            if n_out < 1e-6:
                continue
            u_out = v_out / n_out

        cos_inner = np.clip(np.dot(u_in, u_out), -1.0, 1.0)
        delta = float(np.arccos(cos_inner))
        if abs(delta) < np.deg2rad(float(min_turn_angle_deg)):
            continue

        cross = float(u_in[0] * u_out[1] - u_in[1] * u_out[0])
        direction = 1.0 if cross >= 0 else -1.0

        R_i = float(R_by_corner[i])
        td = float(R_i * np.tan(abs(delta) / 2.0))

        tangent_dists[i] = td
        turn_angles[i] = delta
        turn_dirs[i] = direction
        u_in_by_corner[i, :] = u_in
        u_out_by_corner[i, :] = u_out

    # Clamp corner tangent distances by available adjacent segment lengths.
    # If a corner needs scale < look_ahead_min_scale to fit, mark infeasible.
    feasible = True
    had_clamp = False
    fail_reason_counts = {
        "td_exceeds_available": 0,
        "below_min_scale_requirement": 0,
        "invalid_turn_geometry": 0,
        "strict_no_clamp": 0,
    }
    fail_details = []
    available = seg_lengths.copy()
    for i in range(1, N - 1):
        if tangent_dists[i] < 1e-6:
            continue
        req_td = float(tangent_dists[i])
        max_td = min(available[i - 1], available[i]) * float(corner_fit_margin)
        if max_td < float(corner_min_tangent_m):
            max_td = float(corner_min_tangent_m)
        use_td = req_td
        if req_td > max_td:
            fail_reason_counts["td_exceeds_available"] += 1
            delta_i = float(turn_angles[i])
            tan_half = float(np.tan(abs(delta_i) / 2.0))
            if tan_half > 1e-9:
                td_min_scale = float(R * float(look_ahead_min_scale) * tan_half)
                if max_td + 1e-9 < td_min_scale:
                    feasible = False
                    fail_reason_counts["below_min_scale_requirement"] += 1
            else:
                feasible = False
                td_min_scale = float("nan")
                fail_reason_counts["invalid_turn_geometry"] += 1

            fail_details.append({
                "req_td_m": req_td,
                "max_td_m": max_td,
                "delta_deg": float(np.rad2deg(delta_i)),
                "td_min_scale_m": float(td_min_scale),
                "corner_idx": int(i),
                "seg_prev_idx": int(i - 1),
                "seg_next_idx": int(i),
                "is_budget_floored": bool(max_td <= (float(corner_min_tangent_m) + 1e-9)),
            })

            if allow_tangent_clamp:
                use_td = float(max_td)
                had_clamp = True
            else:
                # strict mode also keeps geometry bounded for downstream path build
                feasible = False
                fail_reason_counts["strict_no_clamp"] += 1
                use_td = float(max_td)

        tangent_dists[i] = use_td
        available[i - 1] -= use_td
        available[i] -= use_td

    arc_start_by_corner = np.zeros((N, 2), dtype=float)
    arc_end_by_corner = np.zeros((N, 2), dtype=float)
    arc_center_by_corner = np.zeros((N, 2), dtype=float)
    corner_active = np.zeros(N, dtype=bool)

    for i in range(1, N - 1):
        td = tangent_dists[i]
        if td < 1e-6:
            continue
        u_in = u_in_by_corner[i, :]
        u_out = u_out_by_corner[i, :]
        if np.linalg.norm(u_in) < 1e-9 or np.linalg.norm(u_out) < 1e-9:
            continue

        wp = local[i, :2]
        arc_start = wp - u_in * td
        arc_end = wp + u_out * td
        direction = turn_dirs[i]
        # Keep arc geometry consistent after tangent clamping:
        # td = R * tan(delta/2)  =>  R = td / tan(delta/2)
        delta_i = float(turn_angles[i])
        tan_half = float(np.tan(abs(delta_i) / 2.0))
        if tan_half < 1e-9:
            continue
        R_arc = float(td / tan_half)

        if direction >= 0:
            perp = np.array([-u_in[1], u_in[0]])
        else:
            perp = np.array([u_in[1], -u_in[0]])
        center = arc_start + perp * R_arc

        arc_start_by_corner[i, :] = arc_start
        arc_end_by_corner[i, :] = arc_end
        arc_center_by_corner[i, :] = center
        corner_active[i] = True

    segments = []
    path_points = []

    for i in range(N - 1):
        if i >= 1 and corner_active[i]:
            tf_start_2d = arc_end_by_corner[i, :].copy()
        else:
            tf_start_2d = local[i, :2].copy()

        if (i + 1) <= (N - 2) and corner_active[i + 1]:
            tf_end_2d = arc_start_by_corner[i + 1, :].copy()
        else:
            tf_end_2d = local[i + 1, :2].copy()

        tf_start_3d = np.array([tf_start_2d[0], tf_start_2d[1], local[i, 2]])
        tf_end_3d = np.array([tf_end_2d[0], tf_end_2d[1], local[i + 1, 2]])
        tf_pts_ll = _local_m_to_latlon(np.vstack([tf_start_3d, tf_end_3d]), ref_lat, ref_lon)

        segments.append({"type": "TF", "points": tf_pts_ll})
        path_points.append(tf_pts_ll[0])

        if (i + 1) <= (N - 2) and corner_active[i + 1]:
            corner_idx = i + 1
            arc_start_2d = arc_start_by_corner[corner_idx, :]
            arc_end_2d = arc_end_by_corner[corner_idx, :]
            center_2d = arc_center_by_corner[corner_idx, :]
            direction = turn_dirs[corner_idx]
            delta_i = float(turn_angles[corner_idx])
            tan_half = float(np.tan(abs(delta_i) / 2.0))
            if tan_half < 1e-9:
                continue
            R_arc = float(tangent_dists[corner_idx] / tan_half)

            ang_start = np.arctan2(arc_start_2d[1] - center_2d[1], arc_start_2d[0] - center_2d[0])
            ang_end = np.arctan2(arc_end_2d[1] - center_2d[1], arc_end_2d[0] - center_2d[0])

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
            arc_local[:, 2] = np.linspace(local[corner_idx, 2], local[corner_idx, 2], num_arc_points)

            arc_ll = _local_m_to_latlon(arc_local, ref_lat, ref_lon)
            center_3d = np.array([center_2d[0], center_2d[1], local[corner_idx, 2]])
            center_ll = _local_m_to_latlon(center_3d, ref_lat, ref_lon).ravel()

            segments.append({
                "type": "RF",
                "points": arc_ll,
                "turn_radius": R_arc,
                "arc_center": center_ll,
                "turn_angle": turn_angles[corner_idx],
            })

            for k in range(num_arc_points):
                path_points.append(arc_ll[k])

    if segments and segments[-1]["type"] == "TF":
        path_points.append(segments[-1]["points"][-1])

    path_array = np.array(path_points, dtype=float)

    debug_level = str(rf_debug_level).strip().lower()
    if debug_level not in ("off", "summary", "detail"):
        debug_level = "off"

    if (not feasible) and debug_level != "off":
        top_detail = None
        if fail_details:
            top_detail = max(fail_details, key=lambda d: float(d["req_td_m"] - d["max_td_m"]))
        if debug_level in ("summary", "detail"):
            print(
                "[RF-DEBUG] RF turn marked infeasible. Reason summary: "
                f"requested tangent distance larger than local segment budget={fail_reason_counts['td_exceeds_available']}, "
                f"cannot satisfy minimum look-ahead radius scale={fail_reason_counts['below_min_scale_requirement']}, "
                f"invalid turn geometry (near-degenerate angle)={fail_reason_counts['invalid_turn_geometry']}, "
                f"clamp disabled in strict mode={fail_reason_counts['strict_no_clamp']}, "
                f"any clamping applied={had_clamp}"
            )
        if debug_level == "detail" and top_detail is not None:
            floor_tag = ""
            if bool(top_detail.get("is_budget_floored", False)):
                floor_tag = " (segment budget floored to corner_min_tangent_m)"
            print(
                "[RF-DEBUG] Strongest failed corner detail: "
                f"corner_idx={top_detail['corner_idx']}, "
                f"seg_prev={top_detail['seg_prev_idx']}->{top_detail['corner_idx']}, "
                f"seg_next={top_detail['corner_idx']}->{top_detail['corner_idx'] + 1}, "
                f"req_td={top_detail['req_td_m']:.2f}m (required tangent distance), "
                f"max_td={top_detail['max_td_m']:.2f}m (available tangent distance), "
                f"delta={top_detail['delta_deg']:.2f}deg (turn angle), "
                f"td_min_scale={top_detail['td_min_scale_m']:.2f}m (minimum tangent distance allowed by look-ahead min scale). "
                f"If req_td > max_td, the corner is too tight for the current local geometry.{floor_tag}"
            )

    return {
        "path": path_array,
        "segments": segments,
        "turn_radius_m": R,
        "ground_speed_mps": ground_speed_mps,
        "feasible": feasible,
        "had_clamp": had_clamp,
        "fail_reason_counts": fail_reason_counts,
        "fail_details": fail_details,
    }
