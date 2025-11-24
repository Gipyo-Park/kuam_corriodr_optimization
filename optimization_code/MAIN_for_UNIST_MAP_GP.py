import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.ndimage import map_coordinates
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import time

# =============================================================================
# Helper function files that should be in the same directory
# =============================================================================
from crossover_GP import crossover_gp
from mutation_GP import mutation_gp
from fast_non_dominated_sort import fast_non_dominated_sort
from generate_initial_population_GP import generate_initial_population_gp
from generate_reference_points_2obj import generate_reference_points_2obj
from normalize_objectives import normalize_objectives
from niching_selection import niching_selection
from evaluate_objectives_with_constraints_GP import evaluate_objectives_with_constraints_gp


# =============================================================================
# Local functions for MAIN script
# =============================================================================

def generate_nodes_3d_segment(p1, p2, W_buf, node_grid_resolution_m, lat_lim, lon_lim, Ny, Nx):
    """
    두 점 사이에 W_buf 폭을 갖는 복도를 설정하고, 그 안에 일정한 간격의 격자 노드를 생성합니다.
    """
    # 1. 위경도 -> 그리드 인덱스 변환 준비
    minLat, maxLat = lat_lim
    minLon, maxLon = lon_lim
    dLat_deg = (maxLat - minLat) / (Ny - 1)
    dLon_deg = (maxLon - minLon) / (Nx - 1)

    # 시작점, 끝점의 그리드 인덱스 계산
    j1, i1 = (p1[0] - minLat) / dLat_deg, (p1[1] - minLon) / dLon_deg
    j2, i2 = (p2[0] - minLat) / dLat_deg, (p2[1] - minLon) / dLon_deg
    p1_grid = np.array([i1, j1])
    p2_grid = np.array([i2, j2])

    # 2. S/T 좌표계(경로 진행 방향 좌표계) 설정
    vec = p2_grid - p1_grid
    len_val = np.linalg.norm(vec)
    if len_val < 1e-6:
        return np.array([]), np.array([])
    
    u_vec = vec / len_val
    v_vec = np.array([-u_vec[1], u_vec[0]])

    # 3. 미터 단위를 그리드 인덱스 단위로 변환하기 위한 스케일 계산
    mean_lat_rad = np.deg2rad(np.mean([p1[0], p2[0]]))
    meters_per_lon_deg = 111000 * np.cos(mean_lat_rad)
    
    meters_per_unit_u = np.sqrt((u_vec[0] * dLon_deg * meters_per_lon_deg)**2 + (u_vec[1] * dLat_deg * 111000)**2)
    meters_per_unit_v = np.sqrt((v_vec[0] * dLon_deg * meters_per_lon_deg)**2 + (v_vec[1] * dLat_deg * 111000)**2)

    # 4. S/T 좌표계 상에서 격자 생성
    len_m = len_val * meters_per_unit_u
    
    n_s = max(1, round(len_m / node_grid_resolution_m))
    s_vec_m = np.linspace(0, len_m, n_s)
    
    n_t = max(1, round(2 * W_buf / node_grid_resolution_m))
    t_vec_m = np.linspace(-W_buf, W_buf, n_t)
    
    S_m, T_m = np.meshgrid(s_vec_m, t_vec_m)
    
    S_idx = S_m / meters_per_unit_u
    T_idx = T_m / meters_per_unit_v
    
    # 5. S/T 좌표를 지도 그리드 인덱스(I, J)로 변환
    I = p1_grid[0] + S_idx * u_vec[0] + T_idx * v_vec[0]
    J = p1_grid[1] + S_idx * u_vec[1] + T_idx * v_vec[1]
    
    valid_mask = (I >= 0) & (I < Nx) & (J >= 0) & (J < Ny)
    Ii = I[valid_mask].flatten()
    Ji = J[valid_mask].flatten()
    
    # 6. 최종 노드 좌표(위경도)로 변환
    all_nodes_lon = minLon + Ii * dLon_deg
    all_nodes_lat = minLat + Ji * dLat_deg
    
    all_nodes_alt = np.full_like(all_nodes_lat, p1[2])
    
    all_grid_nodes = np.vstack([all_nodes_lat, all_nodes_lon, all_nodes_alt]).T
    
    nodes = all_grid_nodes
    
    return nodes, all_grid_nodes


def plot_solutions(gx, solutions, styles, width, seg_num=None, labels=None):
    h = []
    if gx is None: return h
    if labels is None: labels = ['Risk', 'Dist', 'Pareto']
    data = list(solutions.values()) if isinstance(solutions, dict) else solutions
    for i, path_group in enumerate(data):
        paths = path_group if isinstance(path_group, list) else [path_group]
        for path_idx, path in enumerate(paths):
            if path is not None and path.shape[0] > 0:
                label = None
                if path_idx == 0 and seg_num is not None and i < len(labels):
                    label = f'Seg{seg_num} {labels[i]} Rep.'
                style = styles[i % len(styles)]
                line, = gx.plot(path[:, 1], path[:, 0], style, linewidth=width, label=label, transform=ccrs.Geodetic(), zorder=10)
                h.append(line)
    return h

def select_representative_paths(population, fvals, p1, p2):
    if not population or fvals.shape[0] == 0:
        fallback = np.vstack([p1, p2])
        return fallback, fallback, fallback
    rep_r, rep_d = population[np.argmin(fvals[:, 1])], population[np.argmin(fvals[:, 0])]
    F = fast_non_dominated_sort(fvals)
    if not F or not F[0]: rep_p = rep_r
    else:
        front1 = F[0]
        f_pareto = fvals[front1, :]
        if f_pareto.shape[0] > 1:
            norm_f = normalize_objectives(f_pareto)
            best_idx = front1[np.argmin(np.linalg.norm(norm_f, axis=1))]
        else: best_idx = front1[0]
        rep_p = population[best_idx]
    return rep_r, rep_d, rep_p

def selection_nsga3(population, f_vals, feasible, N, ref_points):
    Fronts = fast_non_dominated_sort(f_vals)
    next_pop_indices = []
    for front in Fronts:
        valid = [idx for idx in front if feasible[idx]]
        if not valid: continue
        if len(next_pop_indices) + len(valid) <= N:
            next_pop_indices.extend(valid)
        else:
            remaining = N - len(next_pop_indices)
            last_front = np.array(valid)
            if last_front.shape[0] > 0 and remaining > 0:
                last_front_fvals = f_vals[last_front, :]
                norm_f = normalize_objectives(last_front_fvals)
                selected = niching_selection(norm_f, ref_points, remaining)
                next_pop_indices.extend(last_front[selected])
            break
    return [population[i] for i in next_pop_indices[:N]]

def variation_nsga3(pop, nodes, ratio):
    if not pop: return []
    offspring_num = round(len(pop) * ratio)
    offspring = []
    for _ in range(offspring_num):
        p1_idx, p2_idx = np.random.choice(len(pop), 2)
        child = crossover_gp(pop[p1_idx], pop[p2_idx])
        offspring.append(mutation_gp(child, nodes))
    return offspring

def run_nsga3_segment(nodes, p1, p2, Norm_RT, AirRisk, use_map, f_limit, f_zones, Nmax, N_pop, ratio, H, gx, alt, cs, scales, dz, w_g, w_a, lat_lim, lon_lim):
    # nodes 인자에는 이제 필터링된 'safe_nodes'가 전달됨
    population = generate_initial_population_gp(N_pop, nodes, p1, p2)
    ref_points = generate_reference_points_2obj(H)
    h_paths = []
    for gen in range(1, Nmax + 1):
        print(f'  - Generation {gen}/{Nmax}')
        Np = len(population)
        f_vals, feasible = np.zeros((Np, 2)), np.zeros((Np), dtype=bool)
        for i in range(Np):
            f_vals[i, :], feasible[i] = evaluate_objectives_with_constraints_gp(population[i], Norm_RT, AirRisk, use_map, f_limit, f_zones, dz, alt, cs, scales, w_g, w_a, lat_lim, lon_lim)
        if gx:
            for line in h_paths: line.remove()
            h_paths.clear()
            for idx in np.random.permutation(Np)[:min(Np, 50)]:
                line, = gx.plot(population[idx][:, 1], population[idx][:, 0], '-', color=[0.5, 0.5, 0.5, 0.3], transform=ccrs.Geodetic())
                h_paths.append(line)
            plt.pause(0.01)
        new_pop = selection_nsga3(population, f_vals, feasible, N_pop, ref_points)
        if gen < Nmax:
            if not new_pop:
                print("Warning: New population is empty. Stopping early.")
                population = []
                break
            population = new_pop + variation_nsga3(new_pop, nodes, ratio)
        else:
            population = new_pop
            
    if not population: # 만약 new_pop이 비어서 population이 비게 되면 f_vals 계산을 건너뜀
        return [], np.array([])

    f_vals_final = np.zeros((len(population), 2))
    for i in range(len(population)):
        f_vals_final[i,:], _ = evaluate_objectives_with_constraints_gp(population[i], Norm_RT, AirRisk, use_map, f_limit, f_zones, dz, alt, cs, scales, w_g, w_a, lat_lim, lon_lim)
    if gx:
        for line in h_paths: line.remove()
    return population, f_vals_final

def select_final_solutions(pop, fv, pick, H):
    solutions = {'risk':[], 'dist':[], 'pareto':[]}
    if not pop or fv.shape[0] == 0: return solutions
    idxR = np.argsort(fv[:, 1])[:min(pick['risk'], fv.shape[0])]
    idxD = np.argsort(fv[:, 0])[:min(pick['dist'], fv.shape[0])]
    F = fast_non_dominated_sort(fv)
    selP = []
    if not F or not F[0]: selP = idxR[:min(pick['pareto'], len(idxR))]
    else:
        front1 = np.array(F[0])
        if len(front1) <= pick['pareto']: selP = front1
        else:
            ref_points = generate_reference_points_2obj(H)
            norm_f1 = normalize_objectives(fv[front1, :])
            selP = front1[niching_selection(norm_f1, ref_points, pick['pareto'])]
    solutions.update({'risk':[pop[i] for i in idxR], 'dist':[pop[i] for i in idxD], 'pareto':[pop[i] for i in selP]})
    return solutions

def main():
    # --- 0. Parameters ---
    w_ground, w_air = 0.5, 1.0
    # altitude_levels, use_heading_map = np.array([400,500,600,700]), True
    altitude_levels, use_heading_map = np.array([500]), True
    W_buf = 1000.0
    node_grid_resolution_m = 100.0
    
    # # <<< [수정 시작] 적응형 안전 노드 탐색을 위한 파라미터 >>>
    # # 기존 SAFE_NODE_PERCENTILE 변수를 아래 두 변수로 대체

    MIN_SAFE_NODES_TARGET = 50  # 최소 목표 안전 노드 개수
    SAFE_NODE_PERCENTILE_LIST = [20.0, 30.0, 40.0, 50.0]  # 탐색할 백분위수 목록
    # # <<< [수정 종료] >>>
    
    cell_size, refine_scales, delta_z_max = 100.0, np.array([1.0, 0.5, 0.2, 0.1]), 100.0
    final_pick = {'risk': 5, 'dist': 5, 'pareto': 5}
    Nmax, N_pop, offspring_ratio, H_ref_points = 50, 50, 2.0, 20
    flight_dist_limit = 100.0

    # --- 1. Load and Preprocess Risk Maps ---
    grc_file_list = 'high_res_affected_population_GRC.npy'
    air_risk_file = 'AirRisk_combined_max_risk_map.npy'
    
    print('Loading unified ground/population risk map...')
    try:
        pop_risk_raw = np.load(grc_file_list, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: {grc_file_list} not found.")
        return

    selected_scenario = pop_risk_raw[:, :, 0, 3:]
    Ny, Nx, H = selected_scenario.shape[0], selected_scenario.shape[1], selected_scenario.shape[2]
    A = len(altitude_levels)
    RiskTensor = np.zeros((A, H, Ny, Nx))

    for ai, alt in enumerate(altitude_levels):
        for hi in range(H):
            RiskTensor[ai, hi, :, :] = selected_scenario[:, :, hi]
    
    min_val = np.min(RiskTensor)
    RiskTensor_shifted = RiskTensor - min_val
    max_val = np.max(RiskTensor_shifted)
    Norm_RiskTensor = RiskTensor_shifted / max_val if max_val > 0 else RiskTensor_shifted
    print('Ground/Population risk map loaded and normalized.')

    print('Loading air risk map...')
    try:
        air_risk_raw = np.load(air_risk_file, allow_pickle=True).item()
        z_vec_air = air_risk_raw['z_vec']
        air_risk_3d_raw = air_risk_raw['Risk_3d']
    except FileNotFoundError:
        print(f"Error: {air_risk_file} not found.")
        return

    AirRisk = np.zeros((Ny, Nx, len(altitude_levels)))
    if air_risk_3d_raw.shape[0] == Nx and air_risk_3d_raw.shape[1] == Ny:
        raw_air_risk_tensor = np.transpose(air_risk_3d_raw, (1, 0, 2))
    elif air_risk_3d_raw.shape[0] == Ny and air_risk_3d_raw.shape[1] == Nx:
        raw_air_risk_tensor = air_risk_3d_raw
    else:
        print(f"FATAL ERROR: AirRisk shape {air_risk_3d_raw.shape} is incompatible with GRC shape ({Ny}, {Nx}).")
        return

    for i, alt in enumerate(altitude_levels):
        air_alt_idx = np.argmin(np.abs(z_vec_air - alt))
        air_slice = raw_air_risk_tensor[:, :, air_alt_idx]
        AirRisk[:, :, i] = air_slice
    print('Air risk map loaded and aligned.')

    # --- 2. Define Vertiport and Corridor Points ---
    # vertiport = np.array([35.6033361, 129.0776917, 500])
    vertiport = np.array([35.5845361, 129.1076472, 500])
    # corridor_lat = np.array([35.5845917, 35.6026528, 35.6326806, 35.6249583, 35.6034750, 35.5845361, 35.5692361, 35.5546444, 35.5586722, 35.5784750, 35.5843722, 35.6163861, 35.6212528, 35.6109972])
    # corridor_lon = np.array([129.0936472, 129.1130667, 129.1238583, 129.1335528, 129.1268194, 129.1076472, 129.1085306, 129.0936972, 129.0816611, 129.0916889, 129.0770000, 129.0613944, 129.0725444, 129.0711889])
    corridor_lat = np.array([35.6249583])
    corridor_lon = np.array([129.1335528])
    points = np.vstack([vertiport, np.column_stack([corridor_lat, corridor_lon, 500*np.ones_like(corridor_lat)]), vertiport])
    forbidden_zones = np.array([])
    emergency_lat = np.array([35.6201083, 35.5678222, 35.5919889]); emergency_lon = np.array([129.1191806, 129.106728, 129.0751972])
    
    # <<< [수정] 지도 범위를 동적으로 계산하는 대신, 최대 범위를 기준으로 고정합니다.
    # all_lat = np.concatenate([points[:, 0], emergency_lat]); all_lon = np.concatenate([points[:, 1], emergency_lon])
    # lat_lim = [np.min(all_lat)-0.01, np.max(all_lat)+0.01]; lon_lim = [np.min(all_lon)-0.01, np.max(all_lon)+0.01]
    lat_lim = [35.5446, 35.6427] # 15개 전체 구간을 포함하는 고정된 위도 범위
    lon_lim = [129.0514, 129.1436] # 15개 전체 구간을 포함하는 고정된 경도 범위
    
    # --- 3. Initialize Figure 1 ---
    ENABLE_VISUALIZATION = True
    fig1, gx1 = None, None
    cbar = None
    if ENABLE_VISUALIZATION:
        plt.ion(); fig1 = plt.figure('Figure 1: Comprehensive Process & Results', figsize=(12, 12))
        request = cimgt.OSM(); gx1 = fig1.add_subplot(1, 1, 1, projection=request.crs)
        gx1.set_extent([lon_lim[0], lon_lim[1], lat_lim[0], lat_lim[1]])
        try: gx1.add_image(request, 13)
        except Exception as e: print(f"Could not add map background image: {e}")
        gl = gx1.gridlines(draw_labels={"bottom": "x", "left": "y"}, dms=True); gl.top_labels = gl.right_labels = False
        gx1.set_title('UAM Pathfinding Initialized')
        gx1.plot(points[:, 1], points[:, 0], 'co-', linewidth=1.5, transform=ccrs.Geodetic(), label='Corridor', zorder=4)
        gx1.scatter(points[1:-1, 1], points[1:-1, 0], s=50, c='c', marker='o', transform=ccrs.Geodetic(), label='Corridor Points', zorder=5, edgecolors='k')
        gx1.plot(vertiport[1], vertiport[0], 'mp', markersize=15, transform=ccrs.Geodetic(), label='VertiPort', zorder=5)
        gx1.scatter(emergency_lon, emergency_lat, s=80, c='b', marker='^', transform=ccrs.Geodetic(), label='Emergency Landing', zorder=5)
        gx1.legend(loc='best'); plt.pause(0.1)

    # --- 4. Per-Segment NSGA-III Optimization ---
    num_segments = points.shape[0] - 1
    full_paths = [None] * num_segments; representative_paths = [[None]*3 for _ in range(num_segments)]
    h_nodes, cbar = None, None # 노드와 컬러바 핸들을 하나로 관리
    
    for k in tqdm(range(num_segments), desc="Total Segments"):
        p1, p2 = points[k, :], points[k + 1, :]
        if gx1: gx1.set_title(f'Processing Segment {k+1}/{num_segments}...', size=16)
        print(f'\nProcessing Segment {k+1}/{num_segments}...')
        
        # 1. 전체 후보 노드 생성
        nodes, _ = generate_nodes_3d_segment(p1, p2, W_buf, node_grid_resolution_m, lat_lim, lon_lim, Ny, Nx)
        
        # 2. 시각화: 전체 노드를 위험도에 따라 Heatmap처럼 표시
        if gx1:
            if h_nodes: h_nodes.remove()
            if nodes.shape[0] > 0:
                node_lons, node_lats, node_alts = nodes[:, 1], nodes[:, 0], nodes[:, 2]
                
                I_nodes = (node_lons - lon_lim[0]) / (lon_lim[1] - lon_lim[0]) * (Nx - 1)
                J_nodes = (node_lats - lat_lim[0]) / (lat_lim[1] - lat_lim[0]) * (Ny - 1)
                
                alt_idx = np.argmin(np.abs(altitude_levels - node_alts[0]))
                air_risk_slice = AirRisk[:, :, alt_idx]
                
                node_air_risks = map_coordinates(air_risk_slice, np.vstack((J_nodes, I_nodes)), order=1, cval=0.0)
                
                node_risks_min = np.min(node_air_risks)
                node_risks_max = np.max(node_air_risks)
                if node_risks_min == node_risks_max: node_risks_max += 1e-6

                # [코드 수정] jet 컬러맵을 사용하여 위험도 시각화
                scatter = gx1.scatter(nodes[:, 1], nodes[:, 0], s=25, c=node_air_risks, cmap='coolwarm',
                                        #  vmin=node_risks_min, vmax=node_risks_max,
                                          vmin=0.0, vmax=1.0,
                                          marker='.', transform=ccrs.Geodetic(), label='Candidate Nodes by Air Risk', zorder=6)
                h_nodes = scatter
                
                # <<< 변경점: 컬러바는 최초 1회 생성, 이후 update_normal로 갱신 >>>
                if cbar is None:
                    cbar = fig1.colorbar(scatter, ax=gx1, orientation='vertical', fraction=0.03, pad=0.04)
                    cbar.set_label('Air Risk Level')
                else:
                    cbar.update_normal(scatter)

            gx1.legend(loc='best'); plt.pause(5)

        # # <<< [수정 시작] 적응형 안전 노드 탐색 로직 >>>
        # 3. 필터링: 최적화에 사용할 '안전 노드' 추출
        print(f'Total {nodes.shape[0]} nodes generated. Filtering for optimization...')

        # 모든 노드에 대한 통합 위험도 미리 계산
        node_risks = np.zeros(nodes.shape[0])
        if nodes.shape[0] > 0:
            node_lons, node_lats, node_alts = nodes[:, 1], nodes[:, 0], nodes[:, 2]
            I_nodes = ((node_lons - lon_lim[0]) / (lon_lim[1] - lon_lim[0]) * (Nx - 1)).astype(int)
            J_nodes = ((node_lats - lat_lim[0]) / (lat_lim[1] - lat_lim[0]) * (Ny - 1)).astype(int)
            alt_idx = np.argmin(np.abs(altitude_levels - node_alts[0]))

            avg_ground_risk_map = np.mean(Norm_RiskTensor[alt_idx, :, :, :], axis=0)
            ground_risks_of_nodes = avg_ground_risk_map[J_nodes, I_nodes]
            
            air_risk_slice = AirRisk[:, :, alt_idx]
            air_risks_of_nodes = air_risk_slice[J_nodes, I_nodes]
            
            node_risks = w_ground * ground_risks_of_nodes + w_air * air_risks_of_nodes

        # 적응형 탐색 시작
        safe_nodes = np.array([])
        print("Starting adaptive search for safe nodes...")
        if nodes.shape[0] > 0:
            for percentile in SAFE_NODE_PERCENTILE_LIST:
                risk_threshold = np.percentile(node_risks, percentile)
                current_safe_nodes_mask = node_risks <= risk_threshold
                current_safe_nodes = nodes[current_safe_nodes_mask]
                
                print(f'  - At {percentile}% percentile, found {current_safe_nodes.shape[0]} nodes.')
                
                if current_safe_nodes.shape[0] >= MIN_SAFE_NODES_TARGET:
                    safe_nodes = current_safe_nodes
                    print(f'-> Target of {MIN_SAFE_NODES_TARGET} nodes met. Selecting {safe_nodes.shape[0]} nodes.')
                    break
            
            # 루프가 끝날 때까지 목표를 달성하지 못한 경우, 가장 마지막에 찾은 노드들을 사용
            if safe_nodes.shape[0] == 0 and 'current_safe_nodes' in locals():
                safe_nodes = current_safe_nodes
                print(f'-> Target not met. Using nodes from the last percentile ({percentile}%): {safe_nodes.shape[0]} nodes.')

        # 필터링 후에도 노드가 전혀 없는 경우에 대한 최종 안전장치
        if safe_nodes.shape[0] == 0:
            print("Warning: No safe nodes found even with adaptive search. Using all generated nodes.")
            safe_nodes = nodes
        # # <<< [수정 종료] >>>

        # 4. 최적화: '안전 노드'만을 대상으로 NSGA-III 실행
        print(f'Starting optimization with {safe_nodes.shape[0]} nodes...')
        population, f_vals = run_nsga3_segment(safe_nodes, p1, p2, Norm_RiskTensor, AirRisk, use_heading_map, flight_dist_limit, forbidden_zones, Nmax, N_pop, offspring_ratio, H_ref_points, gx1, altitude_levels, cell_size, refine_scales, delta_z_max, w_ground, w_air, lat_lim, lon_lim)
        
        solutions = select_final_solutions(population, f_vals, final_pick, H_ref_points)
        full_paths[k] = solutions
        rep_r, rep_d, rep_p = select_representative_paths(population, f_vals, p1, p2)
        representative_paths[k] = [rep_r, rep_d, rep_p]
        if gx1:
            h_temp = plot_solutions(gx1, solutions, ['r--', 'b:', 'g-.'], 0.5); plt.pause(5)
            for h in h_temp: h.remove()
            plot_solutions(gx1, [rep_r, rep_d, rep_p], ['r-', 'b-', 'g-'], 2.0, k+1)
            gx1.set_title(f'Segment {k+1} Complete.', size=16); gx1.legend(loc='best'); plt.pause(5)
        print(f'Segment {k+1} Complete.')

    # --- 5. Final Visualization ---
    if gx1:
        if h_nodes: h_nodes.remove()

        gx1.set_title('Final UAM Corridor Paths (All Candidates)', size=16); print('\nDisplaying all candidate paths on Figure 1...')
        for k in range(num_segments):
            if full_paths[k]: plot_solutions(gx1, full_paths[k], ['r:', 'b:', 'g:'], 0.2)
        handles, labels = gx1.get_legend_handles_labels()
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels[label] = handle
        gx1.legend(unique_labels.values(), unique_labels.keys(), loc='best')
        print('Figure 1: Comprehensive results displayed.')
    if ENABLE_VISUALIZATION:
        fig2 = plt.figure('Figure 2: Optimal End-to-End Routes', figsize=(12, 12))
        request = cimgt.OSM(); gx2 = fig2.add_subplot(1, 1, 1, projection=request.crs)
        gx2.set_extent([lon_lim[0], lon_lim[1], lat_lim[0], lat_lim[1]])
        try: gx2.add_image(request, 13)
        except Exception as e: print(f"Could not add map background image: {e}")
        gl2 = gx2.gridlines(draw_labels={"bottom": "x", "left": "y"}, dms=True); gl2.top_labels = gl2.right_labels = False
        gx2.set_title('Optimal End-to-End Routes', size=16)
        gx2.plot(points[:, 1], points[:, 0], 'co-', linewidth=1.5, transform=ccrs.Geodetic())
        gx2.scatter(points[1:-1, 1], points[1:-1, 0], s=50, c='c', marker='o', transform=ccrs.Geodetic(), zorder=5, edgecolors='k')
        gx2.plot(vertiport[1], vertiport[0], 'mp', markersize=15, transform=ccrs.Geodetic(), zorder=5)
        gx2.scatter(emergency_lon, emergency_lat, s=80, c='b', marker='^', transform=ccrs.Geodetic(), zorder=5)
        final_routes = [np.vstack([p[i] for p in representative_paths if p[i] is not None and len(p[i])>0]) for i in range(3)]
        if final_routes and final_routes[0].shape[0] > 0: gx2.plot(final_routes[0][:, 1], final_routes[0][:, 0], 'r-', linewidth=2.5, transform=ccrs.Geodetic(), label='Overall Minimum Risk Route')
        if final_routes and final_routes[1].shape[0] > 0: gx2.plot(final_routes[1][:, 1], final_routes[1][:, 0], 'b-', linewidth=2.5, transform=ccrs.Geodetic(), label='Overall Shortest Distance Route')
        if final_routes and final_routes[2].shape[0] > 0: gx2.plot(final_routes[2][:, 1], final_routes[2][:, 0], 'g-', linewidth=2.5, transform=ccrs.Geodetic(), label='Overall Balanced Route')
        gx2.legend(loc='best'); print('All tasks completed.'); plt.ioff(); plt.show()

if __name__ == '__main__':
    main()