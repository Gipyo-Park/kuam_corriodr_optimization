"""
main_cdr.py  –  Corridor Design & Refinement 파이프라인 시각화
=============================================================
main_JS_1218_v15_JS.py 와 동일한 최적화를 수행한 뒤,
파이프라인 전체를 한 장의 스토리보드로 출력합니다.

  [Backbone] → [초기 후보군] → [Gen 1] → … → [최종 회랑]
                              + 하단 수렴 그래프

실행: python main_cdr.py
"""

import sys, os, json, gc, datetime as _dt
from pathlib import Path

import numpy as np
import matplotlib
# Agg는 나중에 그림 저장 직전에 전환 — 클릭 입력 시 TkAgg 필요
_CLICK_MODE_ENV = os.environ.get("WP_CLICK_MODE", "1").strip().lower()
USE_INTERACTIVE_BACKEND = _CLICK_MODE_ENV in ("1", "true", "yes", "on")
matplotlib.use("Agg")          # 기본은 Agg — 클릭 시 TkAgg로 전환됨
if USE_INTERACTIVE_BACKEND:
    try:
        import tkinter  # noqa: F401
    except Exception:
        USE_INTERACTIVE_BACKEND = False
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

# ── main_JS 모듈에서 함수 전부 가져오기 ──
from main_JS_1218_v15_JS import (
    evaluate_objectives_with_constraints_gp,
    apply_rf_turns_full_corridor,
    build_transition_point,
    build_full_corridor_path,
    run_nsga3,
    pick_representatives,
    plot_corridor_width,
    plot_forbidden_zones,
    plot_moc_binary_overlay,
    compute_centered_map_extent,
    draw_vertiport_radius_rings,
    generate_nodes_3d_segment,
    filter_nodes_in_airspace,
    filter_nodes_in_strip,
    is_path_inside_airspace,
    generate_single_initial_solution,
    _enforce_mandatory_wp_order,
    _seg_dist_m,
    _path_total_3d_distance_m,
    load_noise_risk_from_csv,
    collect_waypoints_from_clicks,
    save_clicked_waypoints,
    cleanup_matplotlib_tk,
    fast_non_dominated_sort,
    normalize_objectives,
)


# ══════════════════════════════════════════════════════════════════
# 파이프라인 실행 + 그림 생성
# ══════════════════════════════════════════════════════════════════
def run_and_plot():
    # ==================== 파라미터 (main_JS 와 동일) ====================
    W_half = 150.0
    speed_max_kmh = 300.0
    ground_speed_mps = speed_max_kmh / 3.6
    bank_angle_deg = 25.0
    g_mps2 = 9.80665
    rf_base_turn_radius_m = (ground_speed_mps ** 2) / (g_mps2 * np.tan(np.deg2rad(bank_angle_deg)))
    num_arc_points = 30
    check_corridor_nfz = True
    check_corridor_moc = True

    N_init = 1000
    min_feasible_init_solutions = 1
    N_pop = 50
    Nmax = 50
    offspring_ratio = 1.0
    require_rf_for_parent_selection = True

    mutation_cfg = {
        "mutation_rate": 0.40,
        "use_local_safe_resample": True,
        "local_resample_prob": 0.70,
        "local_strip_width_m": 1000.0,
        "local_radius_m": 1000.0,
        "local_max_tries": 5,
        "risk_weight_boost": True,
        "risk_weight_strength": 2.0,
    }

    wp_perturb_radius_m = 300.0
    wp_perturb_steps = 10
    min_extra_nodes_per_seg = 0
    max_extra_nodes_per_seg = 4
    enforce_mandatory_wp_order = True
    airspace_radius_km = 5.0
    min_corridor_distance_km = 0.0
    emergency_strip_m = 500.0
    min_seg_for_extra_nodes_m = 1500.0

    look_ahead = True
    look_ahead_threshold_m = 2000.0
    look_ahead_min_scale = 0.11
    look_ahead_window = 2
    max_init_retries = 300

    w_dist, w_ground, w_air, w_noise = 0.1, 1.0, 2.0, 0.1
    altitude_levels = np.array([600.0], dtype=float)
    use_heading_map = True
    W_buf = 1250.0
    node_grid_resolution_m = 100.0
    MIN_SAFE_NODES_TARGET = 200
    SAFE_NODE_AIRRISK_MAX_LIST = [0.1, 0.2, 0.3, 0.4, 0.5]
    USE_PERCENTILE_SAFE_NODE_FILTER = True

    cell_size = 100.0
    refine_scales = np.array([1.0, 0.5, 0.2, 0.1])
    delta_z_max = max(100.0, float(np.max(np.abs(altitude_levels - 150.0))) + 5.0)
    flight_dist_limit = 100000.0
    objective_names = ["Distance", "Ground Risk", "Air Risk", "Noise Risk"]
    airspace_radius_m = float(airspace_radius_km) * 1000.0
    airspace_alt_min_m = 100.0
    airspace_alt_max_m = 700.0
    min_corridor_distance_m = float(min_corridor_distance_km) * 1000.0

    noise_csv_path = Path("noise_data") / "noise_output_lden.csv"
    noise_metric_col = "Lden_db"
    noise_floor_db = 0.0
    ground_risk_path = Path("Modified_high_res_affected_population_GRC.npy")
    bird_airrisk_path = Path("air_risk_data") / "bird_riskmap_springfall_3d.npy"
    moc_airrisk_path = Path("air_risk_data") / "UAM_MOC_3D_Risk_Map.npy"

    _run_ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"{_run_ts}_CDR"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir}")

    # ==================== 리스크맵 로드 ====================
    pop_risk_raw = np.load(str(ground_risk_path), allow_pickle=True)
    selected = pop_risk_raw[:, :, 0, 3:]
    Ny, Nx, H_time = selected.shape
    A = len(altitude_levels)
    RT = np.zeros((A, H_time, Ny, Nx), dtype=float)
    for ai in range(A):
        for hi in range(H_time):
            RT[ai, hi] = selected[:, :, hi]
    mn = float(np.min(RT)); RT -= mn
    mx = float(np.max(RT))
    Norm_RT = RT / mx if mx > 0 else RT

    def _align(raw_3d, name):
        if raw_3d.shape[0] == Nx and raw_3d.shape[1] == Ny:
            return np.transpose(raw_3d, (1, 0, 2))
        if raw_3d.shape[0] == Ny and raw_3d.shape[1] == Nx:
            return raw_3d
        raise RuntimeError(f"{name} shape {raw_3d.shape} mismatch")

    bird_raw = np.load(str(bird_airrisk_path), allow_pickle=True).item()
    bird_z_vec = np.asarray(bird_raw.get("altitude_vec", bird_raw.get("z_vec")), dtype=float).ravel()
    bird_3d = _align(np.asarray(bird_raw["Risk_3d"], dtype=float), "BirdRisk")
    AirRisk = np.zeros((Ny, Nx, len(altitude_levels)), dtype=float)
    for i, alt in enumerate(altitude_levels):
        AirRisk[:, :, i] = bird_3d[:, :, int(np.argmin(np.abs(bird_z_vec - float(alt))))]

    moc_raw = np.load(str(moc_airrisk_path), allow_pickle=True).item()
    moc_z_vec = np.asarray(moc_raw.get("z_vec", moc_raw.get("altitude_vec")), dtype=float).ravel()
    moc_3d = _align(np.asarray(moc_raw["Risk_3d"], dtype=float), "MOCRisk")
    MOCRisk = np.zeros((Ny, Nx, len(altitude_levels)), dtype=np.uint8)
    for i, alt in enumerate(altitude_levels):
        MOCRisk[:, :, i] = (moc_3d[:, :, int(np.argmin(np.abs(moc_z_vec - float(alt))))] >= 0.5).astype(np.uint8)
    moc_plot_2d = np.max(MOCRisk, axis=2).astype(float)

    noise_2d_norm, noise_2d_db, noise_meta = load_noise_risk_from_csv(
        csv_path=noise_csv_path, Ny=Ny, Nx=Nx, metric_col=noise_metric_col,
        receiver_id_base="auto", noise_floor_db=noise_floor_db,
    )
    NoiseRisk = np.repeat(noise_2d_norm[:, :, np.newaxis], len(altitude_levels), axis=2)

    # ==================== 경로 정의 ====================
    start_vertiport = np.array([35.6033361, 129.0776917, 150.0])
    end_vertiport   = np.array([35.6033361, 129.0776917, 150.0])
    airspace_center_lla = np.array([35.6033361, 129.0776917, 150.0])

    lat_lim = [35.535, 35.652]
    lon_lim = [129.020, 129.150]
    request = cimgt.OSM()

    waypoint_alt_fixed_m = float(altitude_levels[0])
    min_clicked_waypoints = 0
    clicked_wp_map_zoom = 13

    forbidden_zones = np.array([], dtype=float).reshape(0, 4)

    emergency_points_input = np.array([
        [35.6201083, 129.1191806, waypoint_alt_fixed_m],
        [35.5678222, 129.1067280, waypoint_alt_fixed_m],
        [35.5919889, 129.0751972, waypoint_alt_fixed_m],
    ])
    emergency_points_preview = filter_nodes_in_airspace(
        emergency_points_input.copy(), airspace_center_lla[:2], airspace_radius_m,
        alt_min_m=airspace_alt_min_m, alt_max_m=airspace_alt_max_m)

    # Takeoff/landing preview (클릭 UI 표시용)
    preview_takeoff, _ = build_transition_point(
        start_vertiport, angle_deg=25.0, alt_delta_m=350.0, sector=6, target_alt_m=waypoint_alt_fixed_m)
    preview_landing, _ = build_transition_point(
        end_vertiport, angle_deg=25.0, alt_delta_m=350.0, sector=11, target_alt_m=waypoint_alt_fixed_m)

    # ── 클릭으로 WP 입력 ──
    corridor_lat = np.array([], dtype=float)
    corridor_lon = np.array([], dtype=float)

    if USE_INTERACTIVE_BACKEND:
        try:
            print("Waypoint click mode is ON.")
            print("  Left click: add WP, Right click/Delete: undo, Enter: finish")
            clicked_latlon = collect_waypoints_from_clicks(
                vertiport=airspace_center_lla,
                lat_lim=lat_lim, lon_lim=lon_lim,
                request=request, map_zoom=clicked_wp_map_zoom,
                takeoff_complete=preview_takeoff,
                landing_entry=preview_landing,
                emergency_points=emergency_points_preview,
                forbidden_zones=forbidden_zones,
                moc_binary_2d=moc_plot_2d,
                ring_radii_m=(airspace_radius_m,),
            )
            if clicked_latlon.shape[0] >= min_clicked_waypoints:
                corridor_lat = clicked_latlon[:, 0].astype(float)
                corridor_lon = clicked_latlon[:, 1].astype(float)
                save_clicked_waypoints(clicked_latlon, waypoint_alt_fixed_m, out_dir)
                print(f"  -> {len(corridor_lat)} WPs clicked.")
            else:
                print(f"  Clicked {clicked_latlon.shape[0]} WPs (< min {min_clicked_waypoints}). Using fallback.")
        except Exception as e:
            print(f"  Click input failed: {e}. Using fallback.")
    else:
        print("Interactive backend unavailable. Using fallback default WPs.")

    # fallback: 하드코딩 WP (클릭 실패 시)
    if corridor_lat.size == 0:
        corridor_lat = np.array([
            35.59351636, 35.59662304, 35.61664094, 35.62302491,
            35.61163693, 35.58747526, 35.56779537, 35.57504642,
            35.59800374, 35.60732288, 35.60318118, 35.58177895,
            35.56537821, 35.56572352, 35.58540392, 35.6014554,
            35.6147429, 35.61198205,
        ])
        corridor_lon = np.array([
            129.0830511, 129.099819,  129.1063988, 129.1180727,
            129.1274118, 129.1265628, 129.1044885, 129.100668,
            129.1214687, 129.1189217, 129.1108561, 129.0953617,
            129.0938759, 129.077957,  129.0790183, 129.0635239,
            129.0654341, 129.0701037,
        ])

    # Agg로 전환 (이후 그림 저장용)
    try:
        plt.switch_backend("Agg")
    except Exception:
        pass

    waypoints = np.column_stack([corridor_lat, corridor_lon,
                                  np.full(len(corridor_lat), waypoint_alt_fixed_m)]) \
                if corridor_lat.size > 0 else np.empty((0, 3), dtype=float)

    if waypoints.shape[0] > 0:
        print("Selected waypoints:")
        for i, wp in enumerate(waypoints, 1):
            print(f"  WP{i:02d}: lat={wp[0]:.7f}, lon={wp[1]:.7f}")

    takeoff_complete, _ = build_transition_point(start_vertiport, angle_deg=25.0, alt_delta_m=350.0, sector=6, target_alt_m=waypoint_alt_fixed_m)
    landing_entry, _    = build_transition_point(end_vertiport,   angle_deg=25.0, alt_delta_m=350.0, sector=11, target_alt_m=waypoint_alt_fixed_m)
    backbone = np.vstack([takeoff_complete, waypoints, landing_entry])
    is_fixed = np.ones(backbone.shape[0], dtype=bool)
    bb_full  = np.vstack([start_vertiport, backbone, end_vertiport])

    emergency_points = filter_nodes_in_airspace(
        emergency_points_input.copy(), airspace_center_lla[:2], airspace_radius_m,
        alt_min_m=airspace_alt_min_m, alt_max_m=airspace_alt_max_m)

    # ==================== 후보 노드 생성 ====================
    def _build_safe_nodes_pct(a, b):
        nodes_seg, all_grid = generate_nodes_3d_segment(
            a, b, W_buf, node_grid_resolution_m, lat_lim, lon_lim, Ny, Nx, forbidden_zones,
            altitude_levels=altitude_levels)
        nodes_seg = filter_nodes_in_airspace(nodes_seg, airspace_center_lla[:2], airspace_radius_m,
                                              alt_min_m=airspace_alt_min_m, alt_max_m=airspace_alt_max_m)
        if nodes_seg.size == 0:
            return np.empty((0, 3)), 0.0
        half_all = int(all_grid.shape[0] // 2)
        target = int(min(max(MIN_SAFE_NODES_TARGET, half_all), nodes_seg.shape[0]))
        I_n = np.clip(((nodes_seg[:, 1] - lon_lim[0]) / (lon_lim[1] - lon_lim[0]) * (Nx - 1)).astype(int), 0, Nx - 1)
        J_n = np.clip(((nodes_seg[:, 0] - lat_lim[0]) / (lat_lim[1] - lat_lim[0]) * (Ny - 1)).astype(int), 0, Ny - 1)
        ai = np.argmin(np.abs(nodes_seg[:, 2][:, None] - altitude_levels[None, :]), axis=1)
        risks = AirRisk[J_n, I_n, ai]
        safe = nodes_seg; thr = 0.0
        for v in SAFE_NODE_AIRRISK_MAX_LIST:
            p = float(np.clip(v * 100.0 if v <= 1.0 else v, 0.0, 100.0))
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
    for k in range(backbone.shape[0] - 1):
        if _seg_dist_m(backbone[k], backbone[k + 1]) < min_seg_for_extra_nodes_m:
            safe_nodes_by_seg.append(np.empty((0, 3)))
            safe_airrisk_by_seg.append(np.empty((0,)))
            thr_list.append(0.0)
            continue
        s, t = _build_safe_nodes_pct(backbone[k], backbone[k + 1])
        s = filter_nodes_in_strip(backbone[k], backbone[k + 1], s, 10 * W_half)
        safe_nodes_by_seg.append(s)
        if s.size > 0:
            I_s = np.clip(((s[:, 1] - lon_lim[0]) / (lon_lim[1] - lon_lim[0]) * (Nx - 1)).astype(int), 0, Nx - 1)
            J_s = np.clip(((s[:, 0] - lat_lim[0]) / (lat_lim[1] - lat_lim[0]) * (Ny - 1)).astype(int), 0, Ny - 1)
            ai_s = np.argmin(np.abs(s[:, 2][:, None] - altitude_levels[None, :]), axis=1)
            safe_airrisk_by_seg.append(AirRisk[J_s, I_s, ai_s].astype(float))
        else:
            safe_airrisk_by_seg.append(np.empty((0,)))
        thr_list.append(t)

    air_thr_global = float(np.max(thr_list)) if thr_list else 1.0
    nodes_pool = np.vstack([s for s in safe_nodes_by_seg if s.size > 0]) \
                 if any(s.size > 0 for s in safe_nodes_by_seg) else backbone.copy()
    node_risk_pool = np.concatenate([r for r in safe_airrisk_by_seg if r.size > 0]) \
                     if any(r.size > 0 for r in safe_airrisk_by_seg) else np.empty((0,))
    if node_risk_pool.shape[0] != nodes_pool.shape[0]:
        node_risk_pool = np.empty((0,))

    # ==================== 초기 해 생성 ====================
    print("Generating initial population …")
    def _make_init_pop():
        _pop = []; draws = 0; max_draws = max(3 * N_init, 50)
        while len(_pop) < N_init and draws < max_draws:
            draws += 1
            _sol = generate_single_initial_solution(
                backbone, wp_perturb_radius_m,
                min_extra_nodes_per_seg, max_extra_nodes_per_seg,
                safe_nodes_by_seg, emergency_points, emergency_strip_m,
                is_fixed, wp_perturb_steps=wp_perturb_steps,
                min_seg_for_extra_nodes_m=min_seg_for_extra_nodes_m)
            if is_path_inside_airspace(_sol, airspace_center_lla[:2], airspace_radius_m,
                                       alt_min_m=airspace_alt_min_m, alt_max_m=airspace_alt_max_m):
                _sol = _enforce_mandatory_wp_order(_sol, backbone)
                _pop.append(_sol)
        return _pop

    init_pop = None
    for _retry in range(1, max_init_retries + 1):
        _candidate = _make_init_pop()
        if not _candidate:
            continue
        _feasible_init = []
        for _c in _candidate:
            _c_eval = _enforce_mandatory_wp_order(_c, backbone)
            _rf = apply_rf_turns_full_corridor(
                _c_eval, start_vertiport, end_vertiport,
                ground_speed_mps, bank_angle_deg, num_arc_points,
                look_ahead, look_ahead_threshold_m, look_ahead_min_scale, look_ahead_window)
            if not _rf["feasible"]:
                continue
            _, _cst_ok = evaluate_objectives_with_constraints_gp(
                _rf["path"], Norm_RT, AirRisk, use_heading_map,
                flight_dist_limit, forbidden_zones, delta_z_max,
                altitude_levels, cell_size, refine_scales, air_thr_global,
                w_dist, w_ground, w_air, lat_lim, lon_lim,
                NoiseRisk=NoiseRisk, noise_floor_db=noise_floor_db, w_noise=w_noise,
                W_half=W_half, check_corridor_nfz=check_corridor_nfz,
                MOCRisk=MOCRisk, check_corridor_moc=check_corridor_moc,
                vertiport=None, landing_entry=None, takeoff_complete=None)
            if not _cst_ok:
                continue
            if not is_path_inside_airspace(_rf["path"], airspace_center_lla[:2], airspace_radius_m,
                                            alt_min_m=airspace_alt_min_m, alt_max_m=airspace_alt_max_m):
                continue
            _feasible_init.append(_c_eval)
        if len(_feasible_init) >= min_feasible_init_solutions:
            init_pop = _feasible_init
            print(f"  -> {len(init_pop)} feasible init solutions found at retry {_retry}")
            break
        if _retry % 50 == 0:
            print(f"  [retry {_retry}/{max_init_retries}] feasible so far: {len(_feasible_init)}")

    if init_pop is None:
        print("Failed to find feasible init pop. Aborting.")
        return

    # ==================== NSGA-III 최적화 ====================
    print("Running NSGA-III …")
    pop, fvals, gen_history = run_nsga3(
        nodes_pool=nodes_pool, node_risk_pool=node_risk_pool,
        population=init_pop, N_pop=N_pop, Nmax=Nmax, ratio=offspring_ratio,
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
        ground_speed_mps=ground_speed_mps, bank_angle_deg=bank_angle_deg,
        num_arc_points=num_arc_points,
        look_ahead=look_ahead, look_ahead_threshold_m=look_ahead_threshold_m,
        look_ahead_min_scale=look_ahead_min_scale, look_ahead_window=look_ahead_window,
        W_half=W_half, check_corridor_nfz=check_corridor_nfz, check_corridor_moc=check_corridor_moc,
        MOCRisk=MOCRisk,
        start_vertiport=start_vertiport, end_vertiport=end_vertiport,
        landing_entry=landing_entry, takeoff_complete=takeoff_complete,
        airspace_center_latlon=airspace_center_lla[:2],
        airspace_radius_m=airspace_radius_m,
        airspace_alt_min_m=airspace_alt_min_m, airspace_alt_max_m=airspace_alt_max_m,
        min_corridor_distance_m=min_corridor_distance_m,
    )

    if not pop or fvals.size == 0:
        print("No feasible solution found.")
        return

    reps = pick_representatives(pop, fvals)
    print(f"Optimization done. pop={len(pop)}, reps={len(reps)}, gens recorded={len(gen_history)}")

    # ==================== RF turn 헬퍼 ====================
    def _rf(chromo):
        return apply_rf_turns_full_corridor(
            chromo, start_vertiport, end_vertiport,
            ground_speed_mps, bank_angle_deg, num_arc_points,
            look_ahead, look_ahead_threshold_m, look_ahead_min_scale, look_ahead_window)

    # ==================== 공통 지도 설정 ====================
    extent_pts = [start_vertiport[:2], end_vertiport[:2], airspace_center_lla[:2],
                  takeoff_complete[:2], landing_entry[:2]]
    extent_pts.extend(backbone[:, :2].tolist())
    map_extent = compute_centered_map_extent(np.array(extent_pts), airspace_center_lla,
                                              ring_radii_m=(airspace_radius_m,), pad_ratio=0.10)

    # ================================================================
    #  세대별 대표해(Balanced) 추출 — 수렴 그래프 + 중간 패널용
    # ================================================================
    gen_bal_fvals = []   # (gen_no, f_val_array)
    gen_bal_paths = []   # (gen_no, rf_path)
    for gh in gen_history:
        gno = int(gh["gen"])
        gpop = gh["population"]
        gf   = gh["f_vals"]
        if not gpop or gf.size == 0:
            continue
        g_reps = pick_representatives(gpop, gf)
        if not g_reps:
            continue
        bal = g_reps[-1]
        rf_res = _rf(bal)
        f_val, _ = evaluate_objectives_with_constraints_gp(
            rf_res["path"], Norm_RT, AirRisk, use_heading_map,
            flight_dist_limit, forbidden_zones, delta_z_max,
            altitude_levels, cell_size, refine_scales, air_thr_global,
            w_dist, w_ground, w_air, lat_lim, lon_lim,
            NoiseRisk=NoiseRisk, noise_floor_db=noise_floor_db, w_noise=w_noise,
            W_half=W_half, check_corridor_nfz=check_corridor_nfz,
            MOCRisk=MOCRisk, check_corridor_moc=check_corridor_moc,
            vertiport=None, landing_entry=None, takeoff_complete=None)
        gen_bal_fvals.append((gno, f_val))
        gen_bal_paths.append((gno, rf_res))

    # 최종 Balanced
    final_bal = reps[-1]
    rf_final = _rf(final_bal)
    f_final_val, _ = evaluate_objectives_with_constraints_gp(
        rf_final["path"], Norm_RT, AirRisk, use_heading_map,
        flight_dist_limit, forbidden_zones, delta_z_max,
        altitude_levels, cell_size, refine_scales, air_thr_global,
        w_dist, w_ground, w_air, lat_lim, lon_lim,
        NoiseRisk=NoiseRisk, noise_floor_db=noise_floor_db, w_noise=w_noise,
        W_half=W_half, check_corridor_nfz=check_corridor_nfz,
        MOCRisk=MOCRisk, check_corridor_moc=check_corridor_moc,
        vertiport=None, landing_entry=None, takeoff_complete=None)

    # ================================================================
    #  중간 세대 인덱스 결정 (최대 2개)
    # ================================================================
    n_gens = len(gen_bal_paths)
    mid_indices = []
    if n_gens >= 4:
        mid_indices = [n_gens // 3, 2 * n_gens // 3]
    elif n_gens >= 2:
        mid_indices = [n_gens // 2]

    # 패널 수: Backbone(1) + Candidates(1) + Gen1(1) + mid(0~2) + Final(1) = 4~6
    n_map_panels = 3 + len(mid_indices) + 1  # backbone, candidates, gen1, mids, final

    # ================================================================
    #  FIGURE 생성
    # ================================================================
    fig = plt.figure(figsize=(6.0 * n_map_panels, 12))
    gs = gridspec.GridSpec(2, n_map_panels, height_ratios=[3, 1.2],
                           hspace=0.22, wspace=0.06,
                           left=0.03, right=0.97, top=0.93, bottom=0.06)

    proj = request.crs

    # ----------- 공통 지도 기반 그리기 함수 -----------
    def _setup_map(ax):
        ax.set_extent(map_extent)
        ax.add_image(request, 13)
        draw_vertiport_radius_rings(ax, airspace_center_lla, radii_m=(airspace_radius_m,))
        plot_forbidden_zones(ax, forbidden_zones, face_alpha=0.08, edge_alpha=0.60)
        plot_moc_binary_overlay(ax, moc_plot_2d, lat_lim, lon_lim,
                                label=None, fill_color="magenta", fill_alpha=0.12)

    def _draw_vertiports(ax, show_wp=True):
        ax.scatter([start_vertiport[1]], [start_vertiport[0]], s=80, c="red", edgecolors="k",
                   marker="s", transform=ccrs.Geodetic(), zorder=10)
        ax.scatter([end_vertiport[1]], [end_vertiport[0]], s=80, c="crimson", edgecolors="k",
                   marker="D", transform=ccrs.Geodetic(), zorder=10)
        ax.scatter([takeoff_complete[1]], [takeoff_complete[0]], s=50, c="blue",
                   marker="^", transform=ccrs.Geodetic(), zorder=10)
        ax.scatter([landing_entry[1]], [landing_entry[0]], s=50, c="green",
                   marker="v", transform=ccrs.Geodetic(), zorder=10)
        if show_wp and waypoints.size > 0:
            ax.scatter(waypoints[:, 1], waypoints[:, 0], s=30, c="orange", edgecolors="k",
                       linewidths=0.4, marker="o", transform=ccrs.Geodetic(), zorder=9)

    def _draw_corridor(ax, rf_result, color="black", lw_tf=1.5, lw_rf=2.2, draw_width=True, alpha_w=0.10):
        path = rf_result["path"]
        if draw_width:
            plot_corridor_width(ax, path, W_half, color=color, alpha=alpha_w)
        for seg in rf_result["segments"]:
            pts = seg["points"]
            w = lw_rf if seg["type"] == "RF" else lw_tf
            ax.plot(pts[:, 1], pts[:, 0], "-", color=color, linewidth=w,
                    transform=ccrs.Geodetic(), zorder=8)

    panel_idx = 0

    # ─────────── Panel 1: Backbone (Ref 회랑) ───────────
    ax1 = fig.add_subplot(gs[0, panel_idx], projection=proj)
    _setup_map(ax1)
    _draw_vertiports(ax1)
    ax1.plot(bb_full[:, 1], bb_full[:, 0], "r-", linewidth=2.5, transform=ccrs.Geodetic(), zorder=7)
    ax1.plot(bb_full[:, 1], bb_full[:, 0], "r--", linewidth=1.0, transform=ccrs.Geodetic(), zorder=7)
    ax1.scatter(bb_full[:, 1], bb_full[:, 0], s=25, c="red", edgecolors="k", linewidths=0.4,
                transform=ccrs.Geodetic(), zorder=8)
    ax1.set_title("① Ref Corridor\n(Backbone)", fontsize=11, fontweight="bold", pad=8)
    panel_idx += 1

    # ─────────── Panel 2: 초기 후보 회랑들 ───────────
    ax2 = fig.add_subplot(gs[0, panel_idx], projection=proj)
    _setup_map(ax2)
    _draw_vertiports(ax2, show_wp=False)
    ax2.plot(bb_full[:, 1], bb_full[:, 0], "r--", linewidth=0.8, alpha=0.4,
             transform=ccrs.Geodetic(), zorder=4)
    n_show = min(30, len(init_pop))
    cmap_init = plt.cm.cool
    for si in range(n_show):
        rf_init = _rf(init_pop[si])
        c = cmap_init(si / max(n_show - 1, 1))
        ax2.plot(rf_init["path"][:, 1], rf_init["path"][:, 0], "-", color=c,
                 linewidth=0.7, alpha=0.55, transform=ccrs.Geodetic(), zorder=5)
    ax2.set_title(f"② Candidates\n({len(init_pop)} solutions)", fontsize=11, fontweight="bold", pad=8)
    panel_idx += 1

    # ─────────── Panel 3: Gen 1 ───────────
    if gen_bal_paths:
        ax3 = fig.add_subplot(gs[0, panel_idx], projection=proj)
        _setup_map(ax3)
        _draw_vertiports(ax3)
        ax3.plot(bb_full[:, 1], bb_full[:, 0], "r--", linewidth=0.6, alpha=0.3,
                 transform=ccrs.Geodetic(), zorder=4)
        g1_no, g1_rf = gen_bal_paths[0]
        _draw_corridor(ax3, g1_rf, color="#2196F3", lw_tf=1.8, lw_rf=2.5, alpha_w=0.12)
        ax3.set_title(f"③ Gen {g1_no}\n(Balanced)", fontsize=11, fontweight="bold", pad=8)
        panel_idx += 1

    # ─────────── Panel 4~5: 중간 세대 ───────────
    mid_colors = ["#FF9800", "#9C27B0"]
    for mi, midx in enumerate(mid_indices):
        ax_m = fig.add_subplot(gs[0, panel_idx], projection=proj)
        _setup_map(ax_m)
        _draw_vertiports(ax_m)
        ax_m.plot(bb_full[:, 1], bb_full[:, 0], "r--", linewidth=0.6, alpha=0.3,
                  transform=ccrs.Geodetic(), zorder=4)
        gm_no, gm_rf = gen_bal_paths[midx]
        _draw_corridor(ax_m, gm_rf, color=mid_colors[mi % 2], lw_tf=1.8, lw_rf=2.5, alpha_w=0.12)
        step_label = panel_idx + 1
        ax_m.set_title(f"④ Gen {gm_no}\n(Balanced)", fontsize=11, fontweight="bold", pad=8)
        panel_idx += 1

    # ─────────── Panel last: 최종 최적 회랑 ───────────
    ax_f = fig.add_subplot(gs[0, panel_idx], projection=proj)
    _setup_map(ax_f)
    _draw_vertiports(ax_f)
    ax_f.plot(bb_full[:, 1], bb_full[:, 0], "r--", linewidth=0.6, alpha=0.3,
              transform=ccrs.Geodetic(), zorder=4)
    _draw_corridor(ax_f, rf_final, color="#1B5E20", lw_tf=2.2, lw_rf=3.0, draw_width=True, alpha_w=0.15)
    step_no = panel_idx + 1
    final_val_str = "  ".join(f"{objective_names[j][:4]}={f_final_val[j]:.3f}" for j in range(len(objective_names)))
    ax_f.set_title(f"⑤ Final Corridor\n(Gen {Nmax}, Balanced)", fontsize=11, fontweight="bold", pad=8)

    # ─────────── 패널 간 화살표 (fig 좌표) ───────────
    for i in range(n_map_panels - 1):
        # 각 패널의 오른쪽 끝 → 다음 패널의 왼쪽 끝 사이에 화살표
        ax_left  = fig.axes[i]
        ax_right = fig.axes[i + 1]
        # 패널 bounding box (fig 좌표)
        bb_l = ax_left.get_position()
        bb_r = ax_right.get_position()
        mid_y = 0.5 * (bb_l.y0 + bb_l.y1)
        arrow = FancyArrowPatch(
            (bb_l.x1 + 0.003, mid_y), (bb_r.x0 - 0.003, mid_y),
            arrowstyle="->,head_width=6,head_length=4",
            color="#333333", linewidth=2.5,
            transform=fig.transFigure, clip_on=False,
        )
        fig.patches.append(arrow)

    # ================================================================
    #  하단: 수렴 그래프
    # ================================================================
    ax_conv = fig.add_subplot(gs[1, :])

    if gen_bal_fvals:
        gens = [g for g, _ in gen_bal_fvals]
        fmat = np.array([f for _, f in gen_bal_fvals])
        obj_colors = ["#1976D2", "#E64A19", "#388E3C", "#7B1FA2"]
        for oi in range(fmat.shape[1]):
            col = obj_colors[oi % len(obj_colors)]
            vals = fmat[:, oi]
            # 정규화 (0~1) for better comparison
            vmin, vmax = np.min(vals), np.max(vals)
            if vmax > vmin:
                vals_norm = (vals - vmin) / (vmax - vmin)
            else:
                vals_norm = np.zeros_like(vals)
            ax_conv.plot(gens, vals_norm, "-o", color=col, linewidth=2.0, markersize=4,
                         label=f"{objective_names[oi]}", alpha=0.85)
            # 실제 값 범위 표시
            ax_conv.annotate(f"{vals[0]:.2f}", (gens[0], vals_norm[0]),
                             fontsize=6, color=col, ha="right", va="bottom")
            ax_conv.annotate(f"{vals[-1]:.2f}", (gens[-1], vals_norm[-1]),
                             fontsize=6, color=col, ha="left", va="top")

        # 중간 세대 위치 세로선
        for midx in mid_indices:
            g_no = gen_bal_fvals[midx][0]
            ax_conv.axvline(g_no, color="gray", linestyle=":", alpha=0.5, linewidth=1)

    ax_conv.set_xlabel("Generation", fontsize=10)
    ax_conv.set_ylabel("Objective (normalized 0–1)", fontsize=10)
    ax_conv.set_title("Objective Convergence  (Balanced representative per generation)", fontsize=11, fontweight="bold")
    ax_conv.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
    ax_conv.grid(True, alpha=0.3)
    ax_conv.set_xlim(left=0)

    # ================================================================
    #  전체 타이틀
    # ================================================================
    fig.suptitle(
        "K-UAM Corridor Design & Refinement Pipeline",
        fontsize=15, fontweight="bold", y=0.98,
    )

    # ================================================================
    #  저장
    # ================================================================
    out_path = out_dir / "pipeline_overview.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nSaved pipeline figure → {out_path}")
    plt.close(fig)

    # ==================== 개별 세대 스냅샷 (보너스) ====================
    snap_dir = out_dir / "gen_snapshots"
    snap_dir.mkdir(exist_ok=True)
    for gno, grf in gen_bal_paths:
        fig_s = plt.figure(figsize=(10, 8))
        ax_s = fig_s.add_subplot(1, 1, 1, projection=proj)
        _setup_map(ax_s)
        _draw_vertiports(ax_s)
        ax_s.plot(bb_full[:, 1], bb_full[:, 0], "r--", linewidth=0.8, alpha=0.4,
                  transform=ccrs.Geodetic(), zorder=4)
        _draw_corridor(ax_s, grf, color="#1565C0", lw_tf=1.8, lw_rf=2.5, alpha_w=0.12)
        ax_s.set_title(f"Generation {gno} — Balanced Corridor", fontsize=11)
        fig_s.savefig(snap_dir / f"gen_{gno:03d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig_s)
    print(f"Saved {len(gen_bal_paths)} generation snapshots → {snap_dir}")


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs("runs", exist_ok=True)
    try:
        run_and_plot()
    finally:
        cleanup_matplotlib_tk()
        gc.collect()
