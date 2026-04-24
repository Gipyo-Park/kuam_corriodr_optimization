import sys
import numpy as np
import numpy.core as _core
import numpy.core.multiarray as _multiarray
import numpy.core.umath as _umath
import numpy.core._multiarray_umath as _mau

# pickle이 찾는 옛 경로들 전부 alias
sys.modules["numpy._core"] = _core
sys.modules["numpy._core.multiarray"] = _multiarray
sys.modules["numpy._core.umath"] = _umath
sys.modules["numpy._core._multiarray_umath"] = _mau



import numpy as np
import pickle
from tqdm import tqdm

from crossover_GP import crossover_gp
from mutation_GP import mutation_gp
from fast_non_dominated_sort import fast_non_dominated_sort
from generate_initial_population_GP import generate_initial_population_gp
from generate_reference_points import generate_reference_points
from normalize_objectives import normalize_objectives
from niching_selection import niching_selection
from evaluate_objectives_with_constraints_GP import evaluate_objectives_with_constraints_gp

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

    wx = cx - ax
    wy = cy - ay

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


##### NEW ######

def generate_valid_nodes_near_vertiport(vertiport, num_nodes, search_radius_m, lat_lim, lon_lim, forbidden_zones):
    mean_lat_rad = np.deg2rad(vertiport[0])
    meters_per_lat_deg = 111000.0
    meters_per_lon_deg = 111000.0 * np.cos(mean_lat_rad)

    search_radius_lat = search_radius_m / meters_per_lat_deg
    search_radius_lon = search_radius_m / meters_per_lon_deg

    valid_nodes = []
    max_attempts = int(num_nodes * 200)

    for _ in range(max_attempts):
        if len(valid_nodes) >= num_nodes:
            break

        angle = np.random.uniform(0.0, 2.0 * np.pi)
        distance_ratio = np.sqrt(np.random.uniform(0.0, 1.0))

        dlat = search_radius_lat * distance_ratio * np.sin(angle)
        dlon = search_radius_lon * distance_ratio * np.cos(angle)

        node_lat = vertiport[0] + dlat
        node_lon = vertiport[1] + dlon
        node_alt = vertiport[2]

        if not (lat_lim[0] <= node_lat <= lat_lim[1] and lon_lim[0] <= node_lon <= lon_lim[1]):
            continue

        is_in_nfz = False
        if forbidden_zones is not None and forbidden_zones.size > 0:
            for rect in forbidden_zones:
                min_lon, max_lon, min_lat, max_lat = rect
                if (min_lon <= node_lon <= max_lon) and (min_lat <= node_lat <= max_lat):
                    is_in_nfz = True
                    break

        if not is_in_nfz:
            valid_nodes.append([node_lat, node_lon, node_alt])

    if valid_nodes:
        return np.array(valid_nodes)
    return np.empty((0, 3))


def generate_nodes_3d_segment(p1, p2, W_buf, node_grid_resolution_m, lat_lim, lon_lim, Ny, Nx, forbidden_zones):
    minLat, maxLat = lat_lim
    minLon, maxLon = lon_lim
    dLat_deg = (maxLat - minLat) / (Ny - 1)
    dLon_deg = (maxLon - minLon) / (Nx - 1)

    j1, i1 = (p1[0] - minLat) / dLat_deg, (p1[1] - minLon) / dLon_deg
    j2, i2 = (p2[0] - minLat) / dLat_deg, (p2[1] - minLon) / dLon_deg
    p1_grid = np.array([i1, j1], dtype=float)
    p2_grid = np.array([i2, j2], dtype=float)

    vec = p2_grid - p1_grid
    len_val = float(np.linalg.norm(vec))
    if len_val < 1e-9:
        return np.empty((0, 3)), np.empty((0, 3))

    u_vec = vec / len_val
    v_vec = np.array([-u_vec[1], u_vec[0]], dtype=float)

    mean_lat_rad = np.deg2rad(float(np.mean([p1[0], p2[0]])))
    meters_per_lon_deg = 111000.0 * np.cos(mean_lat_rad)
    meters_per_lat_deg = 111000.0

    meters_per_unit_u = float(np.sqrt((u_vec[0] * dLon_deg * meters_per_lon_deg) ** 2 + (u_vec[1] * dLat_deg * meters_per_lat_deg) ** 2))
    meters_per_unit_v = float(np.sqrt((v_vec[0] * dLon_deg * meters_per_lon_deg) ** 2 + (v_vec[1] * dLat_deg * meters_per_lat_deg) ** 2))

    len_m = len_val * meters_per_unit_u
    n_s = max(2, int(round(len_m / node_grid_resolution_m)) + 1)
    s_vec_m = np.linspace(0.0, len_m, n_s)

    n_t = max(3, int(round(2.0 * W_buf / node_grid_resolution_m)) + 1)
    t_vec_m = np.linspace(-W_buf, W_buf, n_t)

    S_m, T_m = np.meshgrid(s_vec_m, t_vec_m)
    S_idx = S_m / meters_per_unit_u
    T_idx = T_m / meters_per_unit_v

    I = p1_grid[0] + S_idx * u_vec[0] + T_idx * v_vec[0]
    J = p1_grid[1] + S_idx * u_vec[1] + T_idx * v_vec[1]

    valid_mask = (I >= 0) & (I < Nx) & (J >= 0) & (J < Ny)
    Ii = I[valid_mask].ravel()
    Ji = J[valid_mask].ravel()

    all_nodes_lon = minLon + Ii * dLon_deg
    all_nodes_lat = minLat + Ji * dLat_deg
    all_nodes_alt = np.full_like(all_nodes_lat, p1[2], dtype=float)
    all_grid_nodes = np.vstack([all_nodes_lat, all_nodes_lon, all_nodes_alt]).T

    if forbidden_zones is not None and forbidden_zones.size > 0 and all_grid_nodes.size > 0:
        lats = all_grid_nodes[:, 0]
        lons = all_grid_nodes[:, 1]
        is_valid = np.ones(all_grid_nodes.shape[0], dtype=bool)
        for rect in forbidden_zones:
            min_lon, max_lon, min_lat, max_lat = rect
            in_rect = (lons >= min_lon) & (lons <= max_lon) & (lats >= min_lat) & (lats <= max_lat)
            is_valid[in_rect] = False
        nodes = all_grid_nodes[is_valid]
    else:
        nodes = all_grid_nodes

    return nodes, all_grid_nodes


def selection_nsga3(population, f_vals, feasible, N, ref_points):
    fronts = fast_non_dominated_sort(f_vals)
    next_indices = []

    for front in fronts:
        valid = [idx for idx in front if feasible[idx]]
        if not valid:
            continue

        if len(next_indices) + len(valid) <= N:
            next_indices.extend(valid)
        else:
            remaining = N - len(next_indices)
            last_front = np.array(valid, dtype=int)
            if last_front.size > 0 and remaining > 0:
                last_f = f_vals[last_front, :]
                norm_f = normalize_objectives(last_f)
                selected_local = niching_selection(norm_f, ref_points, remaining)
                next_indices.extend(last_front[selected_local].tolist())
            break

    next_indices = next_indices[:N]
    return [population[i] for i in next_indices]


def variation_nsga3(pop, nodes, ratio):
    if not pop:
        return []
    offspring_num = int(round(len(pop) * ratio))
    offspring = []

    n = len(pop)
    for _ in range(offspring_num):
        if n >= 2:
            p1_idx, p2_idx = np.random.choice(n, 2, replace=False)
        else:
            p1_idx = 0
            p2_idx = 0

        child = crossover_gp(pop[p1_idx], pop[p2_idx])
        child = mutation_gp(child, nodes)
        offspring.append(child)

    return offspring

def run_nsga3_segment(
    nodes, p1, p2,
    Norm_RT, AirRisk, use_map, f_limit, f_zones,
    Nmax, N_pop, ratio, alt, cs, scales,
    air_risk_threshold, dz, w_d, w_g, w_a,
    lat_lim, lon_lim, MAX_INIT_ATTEMPTS,
    min_inter_nodes, max_inter_nodes,
    initial_population=None,
    W_half=None, ground_speed_mps=None, bank_angle_deg=25.0, min_turn_radius_m=296.0,
    check_corridor_nfz=False, check_turn_radius=False, check_heading_continuity=False,
    prev_segment_heading=None,
    vertiport=None, landing_entry=None, takeoff_complete=None  # 추가
):
    dummy_path = np.vstack([p1, p2])
    temp = evaluate_objectives_with_constraints_gp(
        dummy_path, Norm_RT, AirRisk, use_map, f_limit, f_zones, dz, alt, cs, scales,
        air_risk_threshold, w_d, w_g, w_a, lat_lim, lon_lim,
        W_half=W_half, ground_speed_mps=ground_speed_mps, bank_angle_deg=bank_angle_deg,
        min_turn_radius_m=min_turn_radius_m, check_corridor_nfz=check_corridor_nfz,
        check_turn_radius=check_turn_radius, check_heading_continuity=check_heading_continuity,
        prev_segment_heading=prev_segment_heading,
        vertiport=vertiport, landing_entry=landing_entry, takeoff_complete=takeoff_complete
    )
    if len(temp) == 3:
        temp_f, _, _ = temp
    else:
        temp_f, _ = temp
    num_objectives = int(len(temp_f))

    H = num_objectives + 1
    ref_points = generate_reference_points(num_objectives, H)

    population = []
    if initial_population:
        population = list(initial_population)

    for _ in range(MAX_INIT_ATTEMPTS):
    #     need = 10
    #     if need > 0:
    #         # init용 cap
    #         max_trials = 10
    #         trials = 0

    #         while trials < max_trials and need > 0:
    #             trials += 1

    #             cand = generate_initial_population_gp(need, nodes, p1, p2, min_inter_nodes, max_inter_nodes)

    #             if len(cand)
    #             cost, ok, _ = evaluate_objectives_with_constraints_gp(cand)
    #             if ok:
    #                 population.append(cand)
    #                 need -= 1

    #     temp_full = population

        need = N_pop - len(population)
        random_pop = []
        if need > 0:
            random_pop = generate_initial_population_gp(need, nodes, p1, p2, min_inter_nodes, max_inter_nodes)
        temp_full = population + (random_pop if random_pop else [])

        if not temp_full:
            continue

        any_feasible = False
        for path in temp_full:
            ev = evaluate_objectives_with_constraints_gp(
                path, Norm_RT, AirRisk, use_map, f_limit, f_zones, dz, alt, cs, scales,
                air_risk_threshold, w_d, w_g, w_a, lat_lim, lon_lim,
                W_half=W_half, ground_speed_mps=ground_speed_mps, bank_angle_deg=bank_angle_deg,
                min_turn_radius_m=min_turn_radius_m, check_corridor_nfz=check_corridor_nfz,
                check_turn_radius=check_turn_radius, check_heading_continuity=check_heading_continuity,
                prev_segment_heading=prev_segment_heading,
                vertiport=vertiport, landing_entry=landing_entry, takeoff_complete=takeoff_complete
            )
            if len(ev) == 3:
                _, feas, _ = ev
            else:
                _, feas = ev
            if feas:
                any_feasible = True
                break

        if any_feasible:
            population = temp_full[:N_pop]
            break

    if not population:
        return [], np.empty((0, num_objectives)), None

    for gen in range(1, Nmax + 1):
        Np = len(population)
        f_vals = np.zeros((Np, num_objectives), dtype=float)
        feasible = np.zeros(Np, dtype=bool)

        for i in range(Np):
            ev = evaluate_objectives_with_constraints_gp(
                population[i], Norm_RT, AirRisk, use_map, f_limit, f_zones, dz, alt, cs, scales,
                air_risk_threshold, w_d, w_g, w_a, lat_lim, lon_lim,
                W_half=W_half, ground_speed_mps=ground_speed_mps, bank_angle_deg=bank_angle_deg,
                min_turn_radius_m=min_turn_radius_m, check_corridor_nfz=check_corridor_nfz,
                check_turn_radius=check_turn_radius, check_heading_continuity=check_heading_continuity,
                prev_segment_heading=prev_segment_heading,
                vertiport=vertiport, landing_entry=landing_entry, takeoff_complete=takeoff_complete
            )
            if len(ev) == 3:
                f_vals[i, :], feasible[i], _ = ev
            else:
                f_vals[i, :], feasible[i] = ev
                
        num_feasible = int(np.sum(feasible))
        print(f"[Gen {gen}] population {Np}, feasible {num_feasible}")


        new_pop = selection_nsga3(population, f_vals, feasible, N_pop, ref_points)
        if not new_pop:
            break

        if gen < Nmax:
            offspring = variation_nsga3(new_pop, nodes, ratio)
            population = new_pop + offspring
        else:
            population = new_pop

    if not population:
        return [], np.empty((0, num_objectives)), None

    f_final = np.zeros((len(population), num_objectives), dtype=float)
    headings = []

    for i in range(len(population)):
        ev = evaluate_objectives_with_constraints_gp(
            population[i], Norm_RT, AirRisk, use_map, f_limit, f_zones, dz, alt, cs, scales,
            air_risk_threshold, w_d, w_g, w_a, lat_lim, lon_lim,
            W_half=W_half, ground_speed_mps=ground_speed_mps, bank_angle_deg=bank_angle_deg,
            min_turn_radius_m=min_turn_radius_m, check_corridor_nfz=check_corridor_nfz,
            check_turn_radius=check_turn_radius, check_heading_continuity=check_heading_continuity,
            prev_segment_heading=prev_segment_heading,
            vertiport=vertiport, landing_entry=landing_entry, takeoff_complete=takeoff_complete
        )
        if len(ev) == 3:
            f_final[i, :], _, hd = ev
            headings.append(hd)
        else:
            f_final[i, :], _ = ev
            headings.append(None)

    final_heading = None
    for hd in headings:
        if hd is not None:
            final_heading = hd
            break

    return population, f_final, final_heading


def pick_representatives(population, f_vals):
    if (not population) or f_vals.size == 0:
        return []

    num_obj = f_vals.shape[1]
    rep = []

    for i in range(num_obj):
        rep.append(population[int(np.argmin(f_vals[:, i]))])

    fronts = fast_non_dominated_sort(f_vals)
    if fronts and fronts[0]:
        front1 = np.array(fronts[0], dtype=int)
        norm_f = normalize_objectives(f_vals[front1, :])
        balanced_local = int(np.argmin(np.linalg.norm(norm_f, axis=1)))
        rep.append(population[int(front1[balanced_local])])
    else:
        rep.append(rep[0])

    return rep

def attempt_run_once():
    W_half = 76.0
    ground_speed_mps = 70.65
    bank_angle_deg = 25.0
    min_turn_radius_m = 300

    check_corridor_nfz = True
    check_turn_radius = True
    check_heading_continuity = False

    N_init = 5000
    N_pop = 1000
    Nmax = 100
    offspring_ratio = 0.5

    w_dist, w_ground, w_air = 2.0, 0.5, 0.5

    altitude_levels = np.array([450.0])
    use_heading_map = True
    W_buf = 1250.0
    node_grid_resolution_m = 120.0

    MIN_INSERT_PER_SEG = 0
    MAX_INSERT_PER_SEG = 2

    MAX_INIT_ATTEMPTS = 1

    MIN_INTER_NODES_dummy = 0
    MAX_INTER_NODES_dummy = 0

    MIN_SAFE_NODES_TARGET = 200
    SAFE_NODE_PERCENTILE_LIST = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]

    cell_size = 100.0
    refine_scales = np.array([1.0, 0.5, 0.2, 0.1])
    delta_z_max = 100.0
    flight_dist_limit = 100000.0
    objective_names = ["Distance", "Ground Risk", "Air Risk"]

    grc_file_list = "high_res_affected_population_GRC.npy"
    air_risk_file = "AirRisk_combined_max_risk_map.npy"

    pop_risk_raw = np.load(grc_file_list, allow_pickle=True)
    selected_scenario = pop_risk_raw[:, :, 0, 3:]
    Ny, Nx, H_time = selected_scenario.shape[0], selected_scenario.shape[1], selected_scenario.shape[2]

    A = len(altitude_levels)
    RiskTensor = np.zeros((A, H_time, Ny, Nx), dtype=float)
    for ai in range(A):
        for hi in range(H_time):
            RiskTensor[ai, hi, :, :] = selected_scenario[:, :, hi]

    min_val = float(np.min(RiskTensor))
    RiskTensor_shifted = RiskTensor - min_val
    max_val = float(np.max(RiskTensor_shifted))
    Norm_RiskTensor = RiskTensor_shifted / max_val if max_val > 0 else RiskTensor_shifted

    air_risk_raw = np.load(air_risk_file, allow_pickle=True).item()
    z_vec_air = air_risk_raw["z_vec"]
    air_risk_3d_raw = air_risk_raw["Risk_3d"]

    AirRisk = np.zeros((Ny, Nx, len(altitude_levels)), dtype=float)
    if air_risk_3d_raw.shape[0] == Nx and air_risk_3d_raw.shape[1] == Ny:
        raw_air = np.transpose(air_risk_3d_raw, (1, 0, 2))
    elif air_risk_3d_raw.shape[0] == Ny and air_risk_3d_raw.shape[1] == Nx:
        raw_air = air_risk_3d_raw
    else:
        raise RuntimeError(f"AirRisk shape {air_risk_3d_raw.shape} incompatible with ({Ny}, {Nx}).")

    for i, alt in enumerate(altitude_levels):
        air_alt_idx = int(np.argmin(np.abs(z_vec_air - alt)))
        AirRisk[:, :, i] = raw_air[:, :, air_alt_idx]

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
    corridor_lat = np.array([
        35.5944587, 35.6195580,
        35.6218142,
        35.5595903, 35.5671301,
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
    corridor_lon = np.array([
        129.0977958, 129.1153758,
        129.1266116,
        129.0849662, 129.0776521,
        129.0691071, 129.0665565,
        129.0512209,
        129.0536710
    ])


    waypoints = np.column_stack([corridor_lat, corridor_lon, np.full_like(corridor_lat, vertiport[2])])
    
    ### SETTING ### HERE !
    takeoff_complete, landing_entry, _ = build_takeoff_landing_points_example(
        vertiport,
        angle_deg=25.0,
        alt_delta_m=350.0,
        takeoff_sector_1based=11,
        landing_sector_1based=6
    )
    
    print("verti:", vertiport)
    print("takeoff_complete:", takeoff_complete, "delta(m approx):",
        (takeoff_complete[0]-vertiport[0])*111000.0,
        (takeoff_complete[1]-vertiport[1])*(111000.0*np.cos(np.deg2rad(vertiport[0]))))
    print("landing_entry:", landing_entry, "delta(m approx):",
        (landing_entry[0]-vertiport[0])*111000.0,
        (landing_entry[1]-vertiport[1])*(111000.0*np.cos(np.deg2rad(vertiport[0]))))
    # raise KeyError
    
    # base_points = np.vstack([
    #     vertiport,
    #     takeoff_complete,
    #     waypoints,
    #     landing_entry,
    #     vertiport
    # ])
    base_points = np.vstack([
        vertiport,
        landing_entry,
        waypoints,
        takeoff_complete,
        vertiport
    ])
    
    base_points = np.vstack([waypoints])  # 고정 뼈대
    p_start = base_points[0, :]
    p_end = base_points[-1, :]

    forbidden_zones = np.array([
    ], dtype=float)
    
    # forbidden_zones = np.array([
    #     [129.08, 129.10, 35.59, 35.61],
    #     [129.11, 129.118, 35.62, 35.63],
    #     [129.12, 129.13, 35.59, 35.60]
    # ], dtype=float)

    emergency_lat = np.array([35.6201083, 35.5678222, 35.5919889])
    emergency_lon = np.array([129.1191806, 129.106728, 129.0751972])
    emergency_points = np.column_stack([emergency_lat, emergency_lon, np.full_like(emergency_lat, vertiport[2], dtype=float)])

    lat_lim = [35.5446, 35.6427]
    lon_lim = [129.0514, 129.1436]

    def build_safe_nodes_for_segment(a, b):
        nodes_seg, all_grid_nodes = generate_nodes_3d_segment(
            a, b, W_buf, node_grid_resolution_m, lat_lim, lon_lim, Ny, Nx, forbidden_zones
        )
        if nodes_seg.size == 0:
            return np.empty((0, 3)), 0.0

        half_all = int(all_grid_nodes.shape[0] // 2)
        base_target = int(max(MIN_SAFE_NODES_TARGET, half_all))
        current_target = int(min(base_target, nodes_seg.shape[0]))

        node_lons = nodes_seg[:, 1]
        node_lats = nodes_seg[:, 0]
        I_nodes = ((node_lons - lon_lim[0]) / (lon_lim[1] - lon_lim[0]) * (Nx - 1)).astype(int)
        J_nodes = ((node_lats - lat_lim[0]) / (lat_lim[1] - lat_lim[0]) * (Ny - 1)).astype(int)
        I_nodes = np.clip(I_nodes, 0, Nx - 1)
        J_nodes = np.clip(J_nodes, 0, Ny - 1)
        alt_idx = int(np.argmin(np.abs(altitude_levels - nodes_seg[0, 2])))

        air_slice = AirRisk[:, :, alt_idx]
        node_risks = air_slice[J_nodes, I_nodes]

        safe_nodes = nodes_seg
        current_threshold = 0.0
        for perc in SAFE_NODE_PERCENTILE_LIST:
            thr = float(np.percentile(node_risks, perc))
            current_threshold = thr
            safe_nodes = nodes_seg[node_risks <= thr]
            if safe_nodes.shape[0] >= current_target:
                break
        if safe_nodes.size == 0:
            safe_nodes = nodes_seg

        return safe_nodes, current_threshold

    def build_initial_population_whole(num_pop):
        safe_nodes_by_seg = []
        thr_by_seg = []

        for k in range(base_points.shape[0] - 1):
            s, thr = build_safe_nodes_for_segment(base_points[k, :], base_points[k + 1, :])
            safe_nodes_by_seg.append(s)
            thr_by_seg.append(thr)

        init_pop = []
        for _ in range(num_pop):
            path_pts = [base_points[0, :]]
            for k in range(base_points.shape[0] - 1):
                a = base_points[k, :]
                b = base_points[k + 1, :]
                cand = safe_nodes_by_seg[k]
                
                ### NEW ####
                cand = filter_nodes_in_strip(a, b, cand, 10 * W_half)
                m = 1
                ### NEW ####
                # here
                # m = int(np.random.randint(0, 1 + 1))
                # m = int(np.random.randint(0, 1))
                m = 0

                if cand.size == 0:
                    inserts = []
                else:
                    idx = np.random.choice(cand.shape[0], size=min(m, cand.shape[0]), replace=False)
                    chosen = cand[idx, :]

                    ab = b[:2] - a[:2]
                    denom = float(np.dot(ab, ab)) + 1e-12
                    t = ((chosen[:, :2] - a[:2]) @ ab) / denom
                    order = np.argsort(t)
                    inserts = chosen[order, :].tolist()

                path_pts.extend(inserts)
                path_pts.append(b)

            init_pop.append(np.array(path_pts, dtype=float))

        nodes_pool = np.vstack([s for s in safe_nodes_by_seg if s.size > 0]) if any(s.size > 0 for s in safe_nodes_by_seg) else emergency_points
        air_thr_global = float(np.max(thr_by_seg)) if len(thr_by_seg) > 0 else 1.0
        return init_pop, nodes_pool, air_thr_global

    initial_population, nodes_pool, air_thr_global = build_initial_population_whole(N_init)

    pop, fvals, _ = run_nsga3_segment(
        nodes=nodes_pool, p1=p_start, p2=p_end,
        Norm_RT=Norm_RiskTensor, AirRisk=AirRisk, use_map=use_heading_map,
        f_limit=flight_dist_limit, f_zones=forbidden_zones,
        Nmax=Nmax, N_pop=N_pop, ratio=offspring_ratio,
        alt=altitude_levels, cs=cell_size, scales=refine_scales,
        air_risk_threshold=air_thr_global,
        dz=delta_z_max, w_d=w_dist, w_g=w_ground, w_a=w_air,
        lat_lim=lat_lim, lon_lim=lon_lim,
        MAX_INIT_ATTEMPTS=MAX_INIT_ATTEMPTS,
        min_inter_nodes=MIN_INTER_NODES_dummy,
        max_inter_nodes=MAX_INTER_NODES_dummy,
        initial_population=initial_population,
        W_half=W_half, ground_speed_mps=ground_speed_mps,
        bank_angle_deg=bank_angle_deg, min_turn_radius_m=min_turn_radius_m,
        check_corridor_nfz=check_corridor_nfz, check_turn_radius=check_turn_radius,
        check_heading_continuity=check_heading_continuity,
        prev_segment_heading=None,
        vertiport=vertiport, landing_entry=landing_entry, takeoff_complete=takeoff_complete
    )

    # Compute feasible count for the final population
    feasible_count = 0
    if pop and len(pop) > 0:
        feas_list = []
        for i in range(len(pop)):
            ev = evaluate_objectives_with_constraints_gp(
                pop[i], Norm_RiskTensor, AirRisk, use_heading_map, flight_dist_limit, forbidden_zones,
                delta_z_max, altitude_levels, cell_size, refine_scales,
                air_thr_global, w_dist, w_ground, w_air, lat_lim, lon_lim,
                W_half=W_half, ground_speed_mps=ground_speed_mps, bank_angle_deg=bank_angle_deg,
                min_turn_radius_m=min_turn_radius_m, check_corridor_nfz=check_corridor_nfz,
                check_turn_radius=check_turn_radius, check_heading_continuity=check_heading_continuity,
                prev_segment_heading=None,
                vertiport=vertiport, landing_entry=landing_entry, takeoff_complete=takeoff_complete
            )
            if len(ev) == 3:
                _, feas, _ = ev
            else:
                _, feas = ev
            feas_list.append(bool(feas))
        feasible_count = int(np.sum(feas_list))

    # Only save results if we have at least one feasible solution
    if feasible_count > 0:
        reps = []
        if pop and fvals.size > 0:
            reps = pick_representatives(pop, fvals)

        result = {
            "objective_names": objective_names,
            "base_points": base_points,
            "representative_paths_final": reps,
            "population_final": pop,
            "f_final": fvals,
            "vertiport": vertiport,
            "forbidden_zones": forbidden_zones,
            "emergency_points": emergency_points,
            "lat_lim": lat_lim,
            "lon_lim": lon_lim,
            "W_half": W_half
        }

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Summary prints
        print("vertiport:", vertiport)
        print("takeoff:", takeoff_complete)
        print("landing:", landing_entry)
        if pop and len(pop) > 0:
            print("path:", pop[-1])

        # Export Excel of final path details
        import pandas as pd

        rows = []

        # vertiport
        rows.append({
            "type": "vertiport",
            "lat": vertiport[0],
            "lon": vertiport[1],
            "alt": 150.0
        })

        # takeoff
        rows.append({
            "type": "takeoff",
            "lat": takeoff_complete[0],
            "lon": takeoff_complete[1],
            "alt": 500.0
        })

        # landing
        rows.append({
            "type": "landing",
            "lat": landing_entry[0],
            "lon": landing_entry[1],
            "alt": 500.0
        })

        if pop and len(pop) > 0:
            tol = 1e-8
            for i, p in enumerate(pop[-1]):
                is_wp = np.any(
                    np.linalg.norm(base_points[:, :2] - p[:2], axis=1) < tol
                )

                t = "waypoint" if is_wp else "node"

                rows.append({
                    "type": t,
                    "lat": p[0],
                    "lon": p[1],
                    "alt": 500
                })

        import datetime
        import pandas as pd

        df = pd.DataFrame(rows)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"final_path_debug_{ts}.xlsx"

        df.to_excel(fname, index=False)

        # Plot final balanced solution
        if pop and len(pop) > 0:
            # core path
            path_core = pop[0]

            # final path (이착륙 붙인 것)
            path = np.vstack([
                vertiport,
                landing_entry,
                path_core,
                takeoff_complete,
                vertiport
            ])

            fig, ax = plt.subplots(figsize=(7, 7))

            # 전체 경로
            ax.plot(path[:,1], path[:,0], "-k", linewidth=1.5, label="Final path")

            # vertiport
            ax.scatter(
                [vertiport[1], vertiport[1]],
                [vertiport[0], vertiport[0]],
                c="red", s=80, marker="s", label="Vertiport"
            )

            # takeoff / landing
            ax.scatter(
                takeoff_complete[1], takeoff_complete[0],
                c="blue", s=80, marker="^", label="Takeoff complete"
            )
            ax.scatter(
                landing_entry[1], landing_entry[0],
                c="green", s=80, marker="v", label="Landing entry"
            )

            # corridor waypoints
            ax.scatter(
                waypoints[:,1], waypoints[:,0],
                c="orange", s=40, marker="o", label="Waypoints"
            )

            # GA-generated nodes (core path 중 waypoint 아닌 것)
            def is_waypoint(p):
                return np.any(np.all(np.isclose(waypoints, p, atol=1e-8), axis=1))

            gen_nodes = np.array([p for p in path_core if not is_waypoint(p)])
            if gen_nodes.size > 0:
                ax.scatter(
                    gen_nodes[:,1], gen_nodes[:,0],
                    c="purple", s=20, marker=".", label="Generated nodes"
                )

            # backbone (verti -> landing_entry -> waypoints -> takeoff_complete -> verti)
            backbone = np.vstack([vertiport, landing_entry, waypoints, takeoff_complete, vertiport])
            ax.plot(backbone[:,1], backbone[:,0], "r--", linewidth=2.0, label="Backbone")

            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("Final balanced solution check")
            ax.legend()
            ax.grid(True)
            
            plot_fname = f"final_path_solution_{ts}.png"
            plt.savefig(plot_fname)
            print(f"Saved plot to {plot_fname}")
            plt.close()

        # Save pickled results
        out_path = "uam_nsga3_results_whole.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(result, f)
        print(f"Saved results to {out_path}")

        out_path = "uam_nsga3_results.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(result, f)
        print(f"Saved results to {out_path}")

        return True, feasible_count
    else:
        print("Feasible count is 0. Skipping save and retrying...")
        return False, 0


def horiz_dist_for_alt_change(angle_deg, alt_change_m):
    ang = np.deg2rad(angle_deg)
    ang = np.clip(ang, 1e-6, np.pi/2 - 1e-6)
    return float(alt_change_m / np.tan(ang))

def sector_center_angle_rad(sector_idx_1based, num_sectors=12):
    # 1based: 1~12, 12시 시작, 시계방향
    i = int(sector_idx_1based) - 1
    w = 2.0 * np.pi / num_sectors
    return np.deg2rad(90.0) - (i + 0.5) * w

def move_latlon_by_heading_m(lat0, lon0, heading_rad, dist_m):
    mean_lat_rad = np.deg2rad(lat0)
    m_per_lat = 111000.0
    m_per_lon = 111000.0 * np.cos(mean_lat_rad)

    dlat = (dist_m * np.sin(heading_rad)) / m_per_lat
    dlon = (dist_m * np.cos(heading_rad)) / m_per_lon
    return float(lat0 + dlat), float(lon0 + dlon)

def build_takeoff_landing_points_example(vertiport, angle_deg=8.0, alt_delta_m=350.0,
                                         takeoff_sector_1based=12, landing_sector_1based=8):
    lat0, lon0, alt0 = float(vertiport[0]), float(vertiport[1]), float(vertiport[2])

    d_horiz = horiz_dist_for_alt_change(angle_deg, alt_delta_m)

    th_to = sector_center_angle_rad(takeoff_sector_1based, 12)
    to_lat, to_lon = move_latlon_by_heading_m(lat0, lon0, th_to, d_horiz)
    takeoff_complete = np.array([to_lat, to_lon, alt0], dtype=float)

    th_ld = sector_center_angle_rad(landing_sector_1based, 12)
    ld_lat, ld_lon = move_latlon_by_heading_m(lat0, lon0, th_ld, d_horiz)
    landing_entry = np.array([ld_lat, ld_lon, alt0], dtype=float)

    return takeoff_complete, landing_entry, d_horiz


if __name__ == "__main__":
    attempt_idx = 1
    while True:
        success, feas = attempt_run_once()
        if success:
            print(f"Success on attempt {attempt_idx}. Feasible solutions: {feas}")
            break
        else:
            print(f"Attempt {attempt_idx} produced 0 feasible solutions. Retrying...")
            attempt_idx += 1