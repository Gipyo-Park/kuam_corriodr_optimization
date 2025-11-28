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
# [설명] 유전 알고리즘(GA) 및 최적화에 필요한 보조 함수들을 임포트합니다.
from crossover_GP import crossover_gp # 교차 연산 (두 부모 해를 결합하여 자식 해 생성)
from mutation_GP import mutation_gp # 돌연변이 연산 (해의 다양성 유지를 위해 무작위 변형)
from fast_non_dominated_sort import fast_non_dominated_sort # 비지배 정렬 (파레토 최전선 탐색)
from generate_initial_population_GP import generate_initial_population_gp # 초기 해 집단 생성
from generate_reference_points import generate_reference_points # [수정] 통합된 함수를 임포트 # 참조점 생성 (NSGA-III에서 해의 다양성 유지를 위해 사용)
from normalize_objectives import normalize_objectives # 목적 함수 정규화 (서로 다른 스케일의 목표들을 비교 가능하게 함)
from niching_selection import niching_selection # 니칭 선택 (참조점에 가까운 해를 선택하여 다양성 확보)
from evaluate_objectives_with_constraints_GP import evaluate_objectives_with_constraints_gp # 목적 함수 및 제약 조건 평가


# =============================================================================
# Local functions for MAIN script
# =============================================================================

def generate_nodes_3d_segment(p1, p2, W_buf, node_grid_resolution_m, lat_lim, lon_lim, Ny, Nx, forbidden_zones):
    """
    두 점 사이에 W_buf 폭을 갖는 복도를 설정하고, 그 안에 일정한 간격의 격자 노드를 생성합니다.
    
    Args:
        p1, p2: 세그먼트의 시작점과 끝점 (위도, 경도, 고도)
        W_buf: 복도의 폭 (미터 단위)
        node_grid_resolution_m: 노드 간의 간격 (미터 단위)
        lat_lim, lon_lim: 지도의 위도/경도 범위
        Ny, Nx: 지도의 그리드 크기
        forbidden_zones: 금지 구역 정보 (비행 불가 영역)
    """
    # 1. 위경도 -> 그리드 인덱스 변환 준비
    # [설명] 위도/경도 좌표를 행렬 인덱스(Grid Index)로 변환하기 위한 비율을 계산합니다.
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
    # [설명] 경로의 진행 방향을 S축, 수직 방향을 T축으로 하는 로컬 좌표계를 정의합니다.
    vec = p2_grid - p1_grid
    len_val = np.linalg.norm(vec)
    if len_val < 1e-6:
        return np.array([]), np.array([])
    
    u_vec = vec / len_val # 진행 방향 단위 벡터
    v_vec = np.array([-u_vec[1], u_vec[0]]) # 수직 방향 단위 벡터

    # 3. 미터 단위를 그리드 인덱스 단위로 변환하기 위한 스케일 계산
    # [설명] 미터 단위의 거리(W_buf, resolution)를 위경도 그리드 인덱스 단위로 변환합니다.
    mean_lat_rad = np.deg2rad(np.mean([p1[0], p2[0]]))
    meters_per_lon_deg = 111000 * np.cos(mean_lat_rad) # 경도 1도당 미터 거리 (위도에 따라 다름)
    
    # S축과 T축 방향으로의 단위 길이당 미터 거리 계산
    meters_per_unit_u = np.sqrt((u_vec[0] * dLon_deg * meters_per_lon_deg)**2 + (u_vec[1] * dLat_deg * 111000)**2)
    meters_per_unit_v = np.sqrt((v_vec[0] * dLon_deg * meters_per_lon_deg)**2 + (v_vec[1] * dLat_deg * 111000)**2)

    # 4. S/T 좌표계 상에서 격자 생성
    # [설명] 복도 내부를 채울 격자 점들을 S/T 좌표계에서 생성합니다.
    len_m = len_val * meters_per_unit_u # 세그먼트의 실제 길이 (미터)
    
    n_s = max(1, round(len_m / node_grid_resolution_m)) # S축 방향 노드 개수
    s_vec_m = np.linspace(0, len_m, n_s)
    
    n_t = max(1, round(2 * W_buf / node_grid_resolution_m)) # T축 방향 노드 개수 (좌우 폭)
    t_vec_m = np.linspace(-W_buf, W_buf, n_t)
    
    S_m, T_m = np.meshgrid(s_vec_m, t_vec_m) # 격자 생성
    
    S_idx = S_m / meters_per_unit_u # 미터 -> 인덱스 변환
    T_idx = T_m / meters_per_unit_v
    
    # 5. S/T 좌표를 지도 그리드 인덱스(I, J)로 변환
    # [설명] 로컬 좌표계(S, T)를 다시 전역 지도 인덱스(I, J)로 변환합니다.
    I = p1_grid[0] + S_idx * u_vec[0] + T_idx * v_vec[0]
    J = p1_grid[1] + S_idx * u_vec[1] + T_idx * v_vec[1]
    
    # 지도 범위를 벗어나는 노드 제거
    valid_mask = (I >= 0) & (I < Nx) & (J >= 0) & (J < Ny)
    Ii = I[valid_mask].flatten()
    Ji = J[valid_mask].flatten()
    
    # 6. 최종 노드 좌표(위경도)로 변환
    # [설명] 인덱스를 다시 위도/경도 좌표로 변환하여 최종 노드 리스트를 만듭니다.
    all_nodes_lon = minLon + Ii * dLon_deg
    all_nodes_lat = minLat + Ji * dLat_deg
    
    all_nodes_alt = np.full_like(all_nodes_lat, p1[2]) # 고도는 시작점 고도로 고정
    
    all_grid_nodes = np.vstack([all_nodes_lat, all_nodes_lon, all_nodes_alt]).T

    # [수정] 금지 구역(NFZ)에 포함된 노드 필터링
    # [설명] 생성된 노드 중 금지 구역(Forbidden Zones) 내부에 있는 노드를 제거합니다.
    if forbidden_zones is not None and forbidden_zones.shape[0] > 0 and all_grid_nodes.shape[0] > 0:
        lats = all_grid_nodes[:, 0]
        lons = all_grid_nodes[:, 1]
        
        # 모든 노드에 대해 초기에 유효하다고 가정
        is_valid_node = np.ones(all_grid_nodes.shape[0], dtype=bool)
        
        for rect in forbidden_zones:
            min_lon, max_lon, min_lat, max_lat = rect
            
            # 현재 금지 구역 내에 있는 노드를 찾음
            in_rect = (lons >= min_lon) & (lons <= max_lon) & \
                      (lats >= min_lat) & (lats <= max_lat)
            
            # 금지 구역 내 노드를 유효하지 않음으로 표시
            is_valid_node[in_rect] = False
            
        # 유효한 노드만 선택
        nodes = all_grid_nodes[is_valid_node]
    else:
        nodes = all_grid_nodes
    

    
    return nodes, all_grid_nodes


def plot_solutions(gx, solutions, styles, width, seg_num=None, labels=None, custom_color=None):
    """
    최적화된 경로들을 지도 위에 시각화합니다.
    
    Args:
        gx: Matplotlib Axes 객체 (Cartopy projection 포함)
        solutions: 시각화할 경로들의 리스트 또는 딕셔너리
        styles: 선 스타일 리스트 (예: ['r-', 'b--'])
        width: 선 두께
        seg_num: 세그먼트 번호 (레이블용)
        labels: 범례 레이블 리스트
        custom_color: 사용자 지정 색상 (지정 시 styles의 색상 무시)
    """
    h = []
    if gx is None: return h
    if labels is None: labels = ['Path']
    data = list(solutions.values()) if isinstance(solutions, dict) else solutions
    for i, path_group in enumerate(data):
        paths = path_group if isinstance(path_group, list) else [path_group]
        for path_idx, path in enumerate(paths):
            if path is not None and path.shape[0] > 0:
                current_label = None
                # 레이블은 i(솔루션 그룹 인덱스)를 기반으로 할당
                if i < len(labels):
                    # 여러 경로가 같은 그룹에 속할 경우, 첫 번째 경로에만 레이블을 붙임
                    if path_idx == 0:
                        current_label = labels[i] if seg_num is None else f'Seg{seg_num} {labels[i]}'
                
                style = styles[i % len(styles)]
                kwargs = {'linewidth': width, 'label': current_label, 'transform': ccrs.Geodetic(), 'zorder': 10}
                if custom_color is not None:
                    kwargs['color'] = custom_color
                    style = '-' # 색상이 지정되면 스타일 문자열의 색상 코드는 무시하기 위해 실선으로 고정
                
                line, = gx.plot(path[:, 1], path[:, 0], style, **kwargs)
                h.append(line)
    return h

def selection_nsga3(population, f_vals, feasible, N, ref_points):
    """
    NSGA-III 알고리즘의 환경 선택(Environmental Selection) 단계입니다.
    다음 세대로 넘어갈 우수한 해들을 선택합니다.
    
    Args:
        population: 현재 세대의 해 집단
        f_vals: 각 해의 목적 함수 값들
        feasible: 각 해의 실행 가능 여부 (True/False)
        N: 다음 세대로 선택할 해의 개수
        ref_points: 참조점들 (다양성 유지를 위해 사용)
    """
    # 1. 비지배 정렬 (Non-dominated Sorting) 수행
    # [설명] 해들을 파레토 지배 관계에 따라 여러 프론트(Front)로 분류합니다.
    Fronts = fast_non_dominated_sort(f_vals)
    next_pop_indices = []
    
    # 2. 프론트 순서대로 해 선택
    for front in Fronts:
        # 실행 가능한 해만 필터링
        valid = [idx for idx in front if feasible[idx]]
        if not valid: continue
        
        # 현재 프론트를 모두 추가해도 N개를 넘지 않으면 모두 추가
        if len(next_pop_indices) + len(valid) <= N:
            next_pop_indices.extend(valid)
        else:
            # 3. 마지막 프론트에서 일부만 선택해야 할 경우 (Niching)
            # [설명] 남은 자리를 채우기 위해 참조점 기반의 니칭(Niching) 기법을 사용하여
            # 해 공간에 골고루 분포된 해들을 선택합니다.
            remaining = N - len(next_pop_indices)
            last_front = np.array(valid)
            if last_front.shape[0] > 0 and remaining > 0:
                last_front_fvals = f_vals[last_front, :]
                norm_f = normalize_objectives(last_front_fvals) # 목적 함수 정규화
                selected = niching_selection(norm_f, ref_points, remaining) # 니칭 선택 수행
                next_pop_indices.extend(last_front[selected])
            break
    return [population[i] for i in next_pop_indices[:N]]

def variation_nsga3(pop, nodes, ratio):
    """
    유전 알고리즘의 변이 연산(교차 및 돌연변이)을 수행하여 자식 해를 생성합니다.
    
    Args:
        pop: 부모 해 집단
        nodes: 사용 가능한 노드 리스트 (돌연변이 시 사용)
        ratio: 자식 해 생성 비율 (부모 집단 크기 대비)
    """
    if not pop: return []
    offspring_num = round(len(pop) * ratio)
    offspring = []
    for _ in range(offspring_num):
        # 1. 부모 선택 (무작위)
        p1_idx, p2_idx = np.random.choice(len(pop), 2)
        
        # 2. 교차 연산 (Crossover)
        # [설명] 두 부모 경로를 결합하여 새로운 경로를 생성합니다.
        child = crossover_gp(pop[p1_idx], pop[p2_idx])
        
        # 3. 돌연변이 연산 (Mutation)
        # [설명] 생성된 경로의 일부를 무작위로 변형하여 다양성을 확보합니다.
        offspring.append(mutation_gp(child, nodes))
    return offspring

def run_nsga3_segment(nodes, p1, p2, Norm_RT, AirRisk, use_map, f_limit, f_zones, Nmax, N_pop, ratio, H, gx, alt, cs, scales, air_risk_threshold, dz, w_d, w_g, w_a, lat_lim, lon_lim, MAX_INIT_ATTEMPTS, min_inter_nodes, max_inter_nodes, is_initial_stage=False, initial_population=None):
    """
    한 세그먼트(두 지점 사이)에 대해 NSGA-III 알고리즘을 실행하여 최적 경로를 탐색합니다.
    
    Args:
        nodes: 경로 생성에 사용할 수 있는 노드들 (Stage 1: 비상착륙장, Stage 2: 격자 노드)
        p1, p2: 시작점과 끝점
        Norm_RT, AirRisk: 지상 및 공중 위험도 맵 데이터
        use_map: 헤딩 맵 사용 여부
        f_limit: 비행 거리 제한
        f_zones: 금지 구역 정보
        Nmax: 최대 세대 수 (반복 횟수)
        N_pop: 인구 크기 (해 집단 크기)
        ratio: 자식 해 생성 비율
        H: 참조점 생성 파라미터 (목표 개수에 따라 결정됨)
        gx: 시각화용 Axes 객체
        alt: 고도 정보
        cs: 셀 크기
        scales: 보간 스케일
        air_risk_threshold: 공중 위험도 임계값 (필터링용)
        dz: 고도 변화 제한
        w_d, w_g, w_a: 거리, 지상 위험도, 공중 위험도 가중치
        lat_lim, lon_lim: 지도 범위
        MAX_INIT_ATTEMPTS: 초기 해 생성 최대 시도 횟수
        min_inter_nodes, max_inter_nodes: 경로 중간 노드 개수 범위
        is_initial_stage: 초기 단계(Stage 1) 여부
        initial_population: 초기 해 집단 (Stage 2에서 Stage 1의 결과를 사용할 때 입력)
    """
    
    # --- 1. Determine Number of Objectives and Reference Points ---
    # [설명] 목적 함수의 개수를 동적으로 파악하고, 이에 맞는 참조점(Reference Points)을 생성합니다.
    # 임시 경로로 목적 함수를 한번 호출하여 목표 개수를 동적으로 파악
    dummy_path = np.vstack([p1, p2])
    temp_f_val, _ = evaluate_objectives_with_constraints_gp(dummy_path, Norm_RT, AirRisk, use_map, f_limit, f_zones, dz, alt, cs, scales, air_risk_threshold, w_d, w_g, w_a, lat_lim, lon_lim)
    num_objectives = len(temp_f_val)
    
    # [수정] 사용자 요청: 목표 개수에 따라 H값 동적 설정 (H = 목표 개수 + 1)
    H = num_objectives + 1
    print(f"Running NSGA-III with {num_objectives} objectives. Setting H = {H}.")
    
    # 목표 개수에 맞는 참조점 생성
    ref_points = generate_reference_points(num_objectives, H)

    # --- 2. Initialize Population ---
    # [설명] 초기 해 집단을 생성합니다.
    population = []
    
    # Case A: 초기해가 주어진 경우 (Stage 2)
    # [설명] Stage 1에서 찾은 우수한 해들을 초기 해로 사용하여 수렴 속도를 높입니다.
    if initial_population:
        population = list(initial_population)
        print(f"  -> Using {len(population)} initial solutions from Stage 1.")

    # [수정] 초기해 생성 및 실행 가능성 검사 (Retry Logic)
    # Stage 1(초기 단계)이든 Stage 2(메인 단계)이든, 실행 가능한 해가 하나라도 포함되도록 보장합니다.
    
    print(f"  -> Generating/Filling population (Target: {N_pop})...")
    
    for attempt in range(MAX_INIT_ATTEMPTS):
        # 1. 부족한 개체수만큼 랜덤 생성
        current_needed = N_pop - len(population)
        random_pop = []
        
        if current_needed > 0:
            # Stage 1이면 Emergency nodes, Stage 2면 Grid nodes 사용
            random_pop = generate_initial_population_gp(current_needed, nodes, p1, p2, min_inter_nodes, max_inter_nodes)
        
        # 기존 해(seeds)와 랜덤 해 결합
        temp_full_pop = population + (random_pop if random_pop else [])
        
        if not temp_full_pop:
            continue

        # 2. 생성된 초기 집단이 실행 가능한 해를 포함하는지 즉시 검사
        # [설명] 유효성 검사(Feasibility Check):
        # 생성된 경로가 금지 구역(NFZ)을 침범하지 않는지, 회전 반경 등 제약 조건을 만족하는지 확인합니다.
        # evaluate_objectives_with_constraints_gp 함수는 (목적함수값, 유효성여부)를 반환합니다.
        # 여기서 feasible이 True여야 실제 비행 가능한 경로입니다.
        is_any_feasible = False
        for path in temp_full_pop:
            _, feasible = evaluate_objectives_with_constraints_gp(path, Norm_RT, AirRisk, use_map, f_limit, f_zones, dz, alt, cs, scales, air_risk_threshold, w_d, w_g, w_a, lat_lim, lon_lim)
            if feasible:
                is_any_feasible = True
                break
        
        # 3. 실행 가능한 해를 찾았으면, 이 집단으로 최적화 시작
        if is_any_feasible:
            print(f"    -> Found a feasible initial population on attempt {attempt + 1}.")
            population = temp_full_pop
            # 만약 랜덤 생성으로 인해 N_pop을 초과했다면 잘라냄 (seeds 우선)
            if len(population) > N_pop:
                population = population[:N_pop]
            break
        else:
            if attempt % 10 == 0:
                print(f"    -> Attempt {attempt + 1}: Initial population has no feasible solutions. Retrying...")
            # 실패 시, 랜덤 생성된 부분은 버리고 다시 시도 (seeds는 유지)
            # 만약 seeds 자체가 실행 불가능하다면? -> seeds는 유지하되 랜덤 부분에서 실행 가능한 해가 나오길 기대
            pass

    # 유효성 검사 및 재시도 (공통)
    if not population:
        print("Failed to generate or receive a valid initial population.")
        return [], np.array([])

    # --- 3. NSGA-III Main Loop ---
    # [설명] 정해진 세대 수(Nmax)만큼 진화를 반복합니다.
    h_paths = []
    for gen in range(1, Nmax + 1):
        if gen % 10 == 0 or gen == 1:
            print(f'  - Generation {gen}/{Nmax}')
            pass
        
        Np = len(population)
        f_vals = np.zeros((Np, num_objectives))
        feasible = np.zeros(Np, dtype=bool)
        
        # 현재 세대의 모든 해에 대해 목적 함수 평가
        for i in range(Np):
            f_vals[i, :], feasible[i] = evaluate_objectives_with_constraints_gp(population[i], Norm_RT, AirRisk, use_map, f_limit, f_zones, dz, alt, cs, scales, air_risk_threshold, w_d, w_g, w_a, lat_lim, lon_lim)
        
        # 시각화: Stage 2(Main)이거나, Stage 1이라도 가끔 업데이트
        if gx and (gen % 20 == 0 or gen == 1): 
            for line in h_paths: line.remove()
            h_paths.clear()
            
            # 모든 개체 그리기 (투명도 적용)
            for idx in range(Np):
                if feasible[idx]:
                    line, = gx.plot(population[idx][:, 1], population[idx][:, 0], '-', color=[0.5, 0.5, 0.5, 0.3], transform=ccrs.Geodetic())
                    h_paths.append(line)
            plt.pause(10) # 여기를 수정하면 plot이 천천히 된다
            
        # 환경 선택 (Selection): 우수한 해 선택
        new_pop = selection_nsga3(population, f_vals, feasible, N_pop, ref_points)
        
        if not new_pop:
            print("Warning: New population is empty after selection. Stopping early.")
            population = []
            break
            
        # 변이 연산 (Variation): 교차 및 돌연변이로 자식 해 생성
        if gen < Nmax:
            offspring = variation_nsga3(new_pop, nodes, ratio)
            population = new_pop + offspring
        else:
            population = new_pop
            
    if not population:
        return [], np.array([])

    # --- 4. Final Evaluation ---
    # [설명] 최종 세대의 해들에 대해 마지막으로 목적 함수 값을 계산하여 반환합니다.
    f_vals_final = np.zeros((len(population), num_objectives))
    for i in range(len(population)):
        f_vals_final[i,:], _ = evaluate_objectives_with_constraints_gp(population[i], Norm_RT, AirRisk, use_map, f_limit, f_zones, dz, alt, cs, scales, air_risk_threshold, w_d, w_g, w_a, lat_lim, lon_lim)
    
    if gx and h_paths:
        for line in h_paths:
            line.remove()
        
    return population, f_vals_final

# =============================================================================
# MAIN SCRIPT
# =============================================================================
def main():
    """
    메인 실행 함수입니다.
    데이터 로드, 파라미터 설정, 2단계 최적화(Stage 1 & Stage 2) 실행, 결과 시각화를 수행합니다.
    """
    
    # --- 0. Parameters ---
    # [설명] 최적화 및 시뮬레이션에 필요한 주요 파라미터들을 설정합니다.
    
    # [수정] 2단계 최적화 파라미터 분리
    # --- Stage 1: Initial Solution Finding ---
    # [설명] 1단계: 초기 해 탐색 (비상착륙장 활용)
    # 목적: 넓은 범위에서 대략적인 경로를 빠르게 찾고, 실행 가능한 초기 해를 확보합니다.
    N_pop_stage1 = 50          # 인구 크기 (해 집단 크기)
    Nmax_stage1 = 100          # 최대 세대 수 (반복 횟수)
    offspring_ratio_stage1 = 1.0 # 자식 해 생성 비율
    H_ref_points_stage1 = 4   # 참조점 생성 파라미터 (3개 목표에 대해)
    MIN_INTER_NODES_stage1 = 1 # 비상착륙장 중 최소 1개 경유
    MAX_INTER_NODES_stage1 = 3 # 비상착륙장 중 최대 3개 경유
    
    # --- Stage 2: Main Optimization ---
    # [설명] 2단계: 메인 최적화 (격자 노드 활용)
    # 목적: Stage 1의 결과를 바탕으로 경로를 미세 조정하여 최적의 해를 찾습니다.
    
    # [수정] 목적 함수 가중치 설정 (사용자 요청: 거리 가중치 추가)
    # w_dist: 거리 가중치 (높을수록 최단 거리 선호)
    # w_ground: 지상 위험도 가중치 (높을수록 인구 밀집 지역 회피)
    # w_air: 공중 위험도 가중치 (높을수록 다른 비행체와의 충돌 위험 회피)
    w_dist, w_ground, w_air = 1.0, 0.5, 1.0 
    
    # 고도 및 맵 설정
    # altitude_levels, use_heading_map = np.array([400,500,600,700]), True
    altitude_levels, use_heading_map = np.array([500]), True # 단일 고도 500m 사용
    W_buf = 1000.0            # 경로 탐색 복도 폭 (m)
    node_grid_resolution_m = 100.0 # 노드 간격 (m). 이 값이 크면 노드가 듬성듬성 생성됩니다.
    
    # 경로 노드 개수 제약
    MIN_INTER_NODES_stage2 = 5  # 최소 중간 노드 개수
    MAX_INTER_NODES_stage2 = 10 # 최대 중간 노드 개수
    
    # 안전 노드 필터링 파라미터
    MIN_SAFE_NODES_TARGET = 100 # 최소 확보해야 할 안전 노드 개수
    SAFE_NODE_PERCENTILE_LIST = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0] # 위험도 상위 퍼센타일 기준 리스트
    
    # 기타 최적화 파라미터
    MAX_INIT_ATTEMPTS_stage2 = 100 # 초기 해 생성 최대 시도 횟수
    cell_size = 100.0         # 격자 셀 크기 (m)
    refine_scales = np.array([1.0, 0.5, 0.2, 0.1]) # 보간 스케일
    delta_z_max = 100.0       # 고도 변화 허용 범위
    Nmax_stage2 = 100         # 최대 세대 수
    N_pop_stage2 = 50         # 인구 크기
    offspring_ratio_stage2 = 2.0 # 자식 해 생성 비율
    H_ref_points_stage2 = 4   # 참조점 생성 파라미터
    flight_dist_limit = 100.0 # (사용되지 않음) 비행 거리 제한
    objective_names = ["Distance", "Ground Risk", "Air Risk"] # [추가] 시각화를 위한 목표 이름

    # --- 1. Load and Preprocess Risk Maps ---
    # [설명] 지상 위험도(Ground Risk)와 공중 위험도(Air Risk) 데이터를 로드하고 전처리합니다.
    grc_file_list = 'high_res_affected_population_GRC.npy'
    air_risk_file = 'AirRisk_combined_max_risk_map.npy'
    
    print('Loading unified ground/population risk map...')
    try:
        pop_risk_raw = np.load(grc_file_list, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: {grc_file_list} not found.")
        return

    # 지상 위험도 데이터 구조: [Lat, Lon, Time, Alt] -> [Alt, Time, Lat, Lon] 변환 필요
    # 여기서는 특정 시나리오(Time)를 선택하여 처리
    selected_scenario = pop_risk_raw[:, :, 0, 3:]
    Ny, Nx, H = selected_scenario.shape[0], selected_scenario.shape[1], selected_scenario.shape[2]
    A = len(altitude_levels)
    RiskTensor = np.zeros((A, H, Ny, Nx))

    for ai, alt in enumerate(altitude_levels):
        for hi in range(H):
            RiskTensor[ai, hi, :, :] = selected_scenario[:, :, hi]
    
    # 정규화 (Normalization)
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

    # 공중 위험도 데이터 정렬 및 슬라이싱
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
    # [설명] 버티포트(이착륙장)와 회랑(Corridor)의 주요 지점들을 정의합니다.
    # vertiport = np.array([35.6033361, 129.0776917, 500])
    vertiport = np.array([35.5545361, 129.0876472, 500])
    # corridor_lat = np.array([35.5845917, 35.6026528, 35.6326806, 35.6249583, 35.6034750, 35.5845361, 35.5692361, 35.5546444, 35.5586722, 35.5784750, 35.5843722, 35.6163861, 35.6212528, 35.6109972])
    # corridor_lon = np.array([129.0936472, 129.1130667, 129.1238583, 129.1335528, 129.1268194, 129.1076472, 129.1085306, 129.0936972, 129.0816611, 129.0916889, 129.0770000, 129.0613944, 129.0725444, 129.0711889])
    corridor_lat = np.array([35.6249583])
    corridor_lon = np.array([129.1335528])
    
    # 전체 경로 포인트 구성 (Start -> Waypoints -> End)
    points = np.vstack([vertiport, np.column_stack([corridor_lat, corridor_lon, 500*np.ones_like(corridor_lat)]), vertiport])
    
    # 비행 금지 구역 (No-Fly Zones) 정의 [min_lon, max_lon, min_lat, max_lat]
    forbidden_zones = np.array([[129.08, 129.10, 35.59, 35.61], [129.11, 129.118, 35.62, 35.63], [129.12, 129.13, 35.59, 35.6]])
    # forbidden_zones = np.array([[129.08, 129.10, 35.59, 35.61], [129.11, 129.12, 35.59, 35.63], [129.12, 129.13, 35.62, 35.63]])
    # forbidden_zones = np.array([[129.11, 129.118, 35.62, 35.63]])


    # 비상착륙장 위치 정의
    # emergency_lat = np.array([35.6201083, 35.5678222, 35.5919889]); emergency_lon = np.array([129.1191806, 129.106728, 129.0751972])
    emergency_lat = np.array([35.6201083, 35.5678222, 35.5919889, 35.58, 35.625, 35.62, 35.624, 35.56]); emergency_lon = np.array([129.1191806, 129.106728, 129.0751972, 129.12, 129.115, 129.09, 129.065, 129.065])
    # emergency_lat = np.array([35.56]); emergency_lon = np.array([129.065])
    emergency_points = np.column_stack([emergency_lat, emergency_lon, np.full_like(emergency_lat, vertiport[2])])

    # <<< [수정] 지도 범위를 동적으로 계산하는 대신, 최대 범위를 기준으로 고정합니다.
    # all_lat = np.concatenate([points[:, 0], emergency_lat]); all_lon = np.concatenate([points[:, 1], emergency_lon])
    # lat_lim = [np.min(all_lat)-0.01, np.max(all_lat)+0.01]; lon_lim = [np.min(all_lon)-0.01, np.max(all_lon)+0.01]
    lat_lim = [35.5446, 35.6427] # 15개 전체 구간을 포함하는 고정된 위도 범위
    lon_lim = [129.0514, 129.1436] # 15개 전체 구간을 포함하는 고정된 경도 범위
    
    # --- 3. Initialize Figure 1 ---
    # [설명] 시각화를 위한 Figure 및 Axes 객체를 초기화합니다.
    ENABLE_VISUALIZATION = True
    fig1, gx1 = None, None
    fig2, gx2 = None, None
    cbar2 = None
    
    if ENABLE_VISUALIZATION:
        plt.ion() # 대화형 모드 켜기 (실시간 업데이트)
        
        # Figure 1: Stage 1 (Initial) - 초기 해 탐색 과정 시각화
        fig1 = plt.figure('Figure 1: Stage 1 - Initial Solution Finding', figsize=(12, 10))
        fig1.subplots_adjust(left=0.2, right=0.95) # [수정] 범례(좌측) 공간 확보 (Figure 2와 동일하게 left=0.2)
        request = cimgt.OSM() # OpenStreetMap 배경 사용
        gx1 = fig1.add_subplot(1, 1, 1, projection=request.crs)
        gx1.set_extent([lon_lim[0], lon_lim[1], lat_lim[0], lat_lim[1]])
        gx1.add_image(request, 13)
        gx1.set_title('Stage 1: Initial Solutions')
        
        # Figure 2: Stage 2 (Main) - 메인 최적화 과정 시각화
        fig2 = plt.figure('Figure 2: Stage 2 - Main Optimization', figsize=(12, 10))
        fig2.subplots_adjust(left=0.2, right=0.85) # 범례(좌측) 및 컬러바(우측) 공간 확보
        gx2 = fig2.add_subplot(1, 1, 1, projection=request.crs)
        gx2.set_extent([lon_lim[0], lon_lim[1], lat_lim[0], lat_lim[1]])
        gx2.add_image(request, 13)
        gx2.set_title('Stage 2: Main Optimization')
        
        # Common Plotting Elements (공통 요소: 회랑 포인트, 버티포트, 비상착륙장, 금지구역)
        for gx in [gx1, gx2]:
            gx.plot(points[:, 1], points[:, 0], 'co-', linewidth=1.5, transform=ccrs.Geodetic(), label='Corridor', zorder=4)
            gx.scatter(points[1:-1, 1], points[1:-1, 0], s=50, c='c', marker='o', transform=ccrs.Geodetic(), label='Corridor Points', zorder=5, edgecolors='k')
            gx.plot(vertiport[1], vertiport[0], 'mp', markersize=15, transform=ccrs.Geodetic(), label='VertiPort', zorder=5)
            gx.scatter(emergency_lon, emergency_lat, s=80, c='b', marker='^', transform=ccrs.Geodetic(), label='Emergency Landing', zorder=5)
            if forbidden_zones is not None:
                for i, rect in enumerate(forbidden_zones):
                    min_lon, max_lon, min_lat, max_lat = rect
                    lons = [min_lon, max_lon, max_lon, min_lon, min_lon]
                    lats = [min_lat, min_lat, max_lat, max_lat, min_lat]
                    gx.fill(lons, lats, color='red', alpha=0.3, transform=ccrs.Geodetic(), zorder=6)

        plt.pause(0.1)

    # --- 4. Main Loop (Segment by Segment) ---
    # [설명] 전체 경로를 세그먼트(구간) 단위로 나누어 순차적으로 최적화합니다.
    num_segments = points.shape[0] - 1
    representative_paths_final = []

    # [추가] 시각화 객체 관리를 위한 변수
    stage1_current_lines = []
    stage2_current_scatter = None
    all_stage1_solutions_history = [] 

    for k in tqdm(range(num_segments), desc="Stage 1: Finding Initial Solutions"):
        p1, p2 = points[k, :], points[k + 1, :]
        print(f'\n=== Processing Segment {k+1}/{num_segments} ===')

        # ---------------------------------------------------------
        # [Stage 1] Initial Solution Finding (Emergency Points)
        # ---------------------------------------------------------
        # [설명] 1단계: 비상착륙장을 노드로 사용하여 초기 해를 탐색합니다.
        print(f"--- Stage 1: Finding Initial Solutions (Segment {k+1}) ---")
        if gx1: 
            gx1.set_title(f'Stage 1: Segment {k+1} Initial Search', size=16)
            # [수정] 이전 세그먼트의 초기해 시각화 제거
            for line in stage1_current_lines:
                line.remove()
            stage1_current_lines = []
        
        nodes_stage1 = emergency_points
        
        population_stage1, f_vals_stage1 = run_nsga3_segment(
            nodes=nodes_stage1, p1=p1, p2=p2,
            Norm_RT=Norm_RiskTensor, AirRisk=AirRisk, use_map=use_heading_map,
            f_limit=flight_dist_limit, f_zones=forbidden_zones,
            Nmax=Nmax_stage1, N_pop=N_pop_stage1, ratio=offspring_ratio_stage1, H=H_ref_points_stage1,
            gx=gx1, alt=altitude_levels, cs=cell_size, scales=refine_scales,
            air_risk_threshold=1.0, # 1단계에서는 모든 비상착륙장을 고려하므로 임계값 0
            dz=delta_z_max, w_d=w_dist, w_g=w_ground, w_a=w_air,
            lat_lim=lat_lim, lon_lim=lon_lim,
            MAX_INIT_ATTEMPTS=5, 
            min_inter_nodes=MIN_INTER_NODES_stage1, 
            max_inter_nodes=MAX_INTER_NODES_stage1,
            is_initial_stage=True # 초기 단계임을 명시
        )

        # Select Initial Solutions
        # [설명] Stage 1에서 찾은 해들 중 우수한 해들을 선택하여 Stage 2의 초기 해로 사용합니다.
        initial_solutions = []
        if population_stage1 and f_vals_stage1.shape[0] > 0:
            num_obj = f_vals_stage1.shape[1]
            selected_indices = set()
            # 각 목표별 최적해 선택
            for i in range(num_obj):
                selected_indices.add(np.argmin(f_vals_stage1[:, i]))
            
            # 파레토 최전선에서 추가적인 해 선택 (필요 시)
            F = fast_non_dominated_sort(f_vals_stage1)
            if F and F[0]:
                front1 = F[0]
                remaining = [idx for idx in front1 if idx not in selected_indices]
                if remaining:
                    norm_f = normalize_objectives(f_vals_stage1[remaining, :])
                    best_bal = np.argmin(np.linalg.norm(norm_f, axis=1))
                    selected_indices.add(remaining[best_bal])
            
            initial_solutions = [population_stage1[i] for i in selected_indices]
            all_stage1_solutions_history.append(initial_solutions) # [추가] 이력 저장
            print(f"  -> Selected {len(initial_solutions)} initial solutions from Stage 1.")
            
            if gx1:
                # [수정] 핸들 저장 및 pause 시간 단축
                stage1_current_lines = plot_solutions(gx1, initial_solutions, ['m-'], 1.2, k+1, labels=["Initial Sol."])
                # [추가] 범례 표시 (Figure 2와 동일한 위치)
                gx1.legend(loc='upper right', bbox_to_anchor=(-0.1, 1), borderaxespad=0.)
                plt.pause(0.1)
        else:
            all_stage1_solutions_history.append([]) # [추가] 빈 리스트 저장
            print("  -> Stage 1 failed. Will use random initialization in Stage 2.")

        # ---------------------------------------------------------
        # [Stage 2] Main Optimization (Grid Nodes)
        # ---------------------------------------------------------
        # [설명] 2단계: 격자 노드를 사용하여 정밀한 최적화를 수행합니다.
        print(f"--- Stage 2: Main Optimization (Segment {k+1}) ---")
        if gx2: 
            gx2.set_title(f'Stage 2: Segment {k+1} Main Optimization', size=16)
            # [수정] 이전 세그먼트의 노드(scatter) 제거
            if stage2_current_scatter is not None:
                stage2_current_scatter.remove()
                stage2_current_scatter = None

        # 2-1. Generate Nodes
        # [설명] 세그먼트 주변에 격자 노드를 생성합니다.
        nodes_stage2, all_grid_nodes = generate_nodes_3d_segment(p1, p2, W_buf, node_grid_resolution_m, lat_lim, lon_lim, Ny, Nx, forbidden_zones)
        
        # 2-2. [복원] 적응형 안전 노드 필터링
        # [설명] 생성된 노드 중 위험도가 낮은 '안전 노드'만 선별합니다.
        # MIN_SAFE_NODES_TARGET 동적 조정
        # all_grid_nodes 개수의 절반과 사용자 설정값 중 더 큰 값을 이번 세그먼트의 목표로 설정
        half_all_nodes = all_grid_nodes.shape[0] // 2
        base_target = int(max(MIN_SAFE_NODES_TARGET, half_all_nodes))
        current_min_safe_nodes_target = min(base_target, nodes_stage2.shape[0])
        print(f"Dynamic MIN_SAFE_NODES_TARGET for this segment: {current_min_safe_nodes_target} (user_set: {int(MIN_SAFE_NODES_TARGET)}, half_all_nodes: {half_all_nodes}, available: {nodes_stage2.shape[0]})")

        # 3. 필터링: 최적화에 사용할 '안전 노드' 추출
        print(f'Total {nodes_stage2.shape[0]} nodes generated. Filtering for optimization...')

        # 모든 노드에 대한 통합 위험도 미리 계산
        node_risks = np.zeros(nodes_stage2.shape[0])
        if nodes_stage2.shape[0] > 0:
            node_lons, node_lats, node_alts = nodes_stage2[:, 1], nodes_stage2[:, 0], nodes_stage2[:, 2]
            I_nodes = ((node_lons - lon_lim[0]) / (lon_lim[1] - lon_lim[0]) * (Nx - 1)).astype(int)
            J_nodes = ((node_lats - lat_lim[0]) / (lat_lim[1] - lat_lim[0]) * (Ny - 1)).astype(int)
            alt_idx = np.argmin(np.abs(altitude_levels - node_alts[0]))

            avg_ground_risk_map = np.mean(Norm_RiskTensor[alt_idx, :, :, :], axis=0)
            ground_risks_of_nodes = avg_ground_risk_map[J_nodes, I_nodes]
            
            air_risk_slice = AirRisk[:, :, alt_idx]
            air_risks_of_nodes = air_risk_slice[J_nodes, I_nodes]
            
            # [수정] 사용자 요청: 안전 노드 필터링 시 공중 위험도(Air Risk)만 사용
            node_risks = air_risks_of_nodes 

        # 적응형 탐색 시작
        # [설명] 위험도 임계값을 점진적으로 높여가며 충분한 수의 안전 노드를 확보합니다.
        safe_nodes = np.array([])
        print("Starting adaptive search for safe nodes (Air Risk only)...")
        current_threshold = 0.0 # 초기화
        
        if nodes_stage2.shape[0] > 0:
            # 루프를 돌면서 safe_nodes를 계속 갱신하고, 목표 달성 시 중단
            for percentile in SAFE_NODE_PERCENTILE_LIST:
                risk_threshold = np.percentile(node_risks, percentile)
                current_threshold = risk_threshold # 임계값 업데이트
                current_safe_nodes_mask = node_risks <= risk_threshold
                safe_nodes = nodes_stage2[current_safe_nodes_mask] # 매번 safe_nodes를 갱신
                
                print(f'  - At {percentile}% percentile (Air Risk), found {safe_nodes.shape[0]} nodes.')
                
                if safe_nodes.shape[0] >= current_min_safe_nodes_target:
                    print(f'-> Target of {current_min_safe_nodes_target} nodes met. Selecting {safe_nodes.shape[0]} nodes.')
                    break
            
            # 루프가 break 없이 끝났을 경우 (목표 미달)
            else:
                print(f'-> Target not met. Using nodes from the last percentile ({percentile}%): {safe_nodes.shape[0]} nodes.')

        # 필터링 후에도 노드가 전혀 없는 경우에 대한 최종 안전장치
        if safe_nodes.shape[0] == 0:
            print("Warning: No safe nodes found even with adaptive search. Using all generated nodes.")
            safe_nodes = nodes_stage2
            
        if safe_nodes.shape[0] > 0:
            # [시각화 복원] Stage 2 노드 표시 (위험도에 따른 색상)
            if gx2:
                # 안전 노드에 대한 위험도 다시 계산 (색상 매핑용)
                # [중요] 필터링에 사용된 값과 동일한 방식으로 위험도 계산 (Air Risk Only)
                s_lons, s_lats = safe_nodes[:, 1], safe_nodes[:, 0]
                
                # 인덱스 계산 (필터링 로직과 동일하게 int 변환)
                s_I = ((s_lons - lon_lim[0]) / (lon_lim[1] - lon_lim[0]) * (Nx - 1)).astype(int)
                s_J = ((s_lats - lat_lim[0]) / (lat_lim[1] - lat_lim[0]) * (Ny - 1)).astype(int)
                
                # 범위 클리핑 (인덱스 초과 방지)
                s_I = np.clip(s_I, 0, Nx - 1)
                s_J = np.clip(s_J, 0, Ny - 1)
                
                # 위험도 값 추출
                s_g_risks = avg_ground_risk_map[s_J, s_I]
                s_a_risks = air_risk_slice[s_J, s_I]
                s_risks = s_a_risks # [수정] 시각화도 공중 위험도(Air Risk) 기준으로 변경
                
                # [수정] scatter 객체 저장
                stage2_current_scatter = gx2.scatter(s_lons, s_lats, c=s_risks, cmap='jet', vmin=0.0, vmax=1.0, s=10, alpha=0.5, transform=ccrs.Geodetic(), zorder=3)
                
                if cbar2 is None:
                    cbar2 = fig2.colorbar(stage2_current_scatter, ax=gx2, fraction=0.046, pad=0.04)
                    cbar2.set_label('Risk Level')
                
                plt.pause(0.1)

        else:
            print("  -> No grid nodes generated. Skipping Stage 2.")
            representative_paths_final.append([])
            continue

        # 2-3. Run Optimization
        # [설명] Stage 2 최적화 실행
        population_stage2, f_vals_stage2 = run_nsga3_segment(
            nodes=safe_nodes, p1=p1, p2=p2,
            Norm_RT=Norm_RiskTensor, AirRisk=AirRisk, use_map=use_heading_map,
            f_limit=flight_dist_limit, f_zones=forbidden_zones,
            Nmax=Nmax_stage2, N_pop=N_pop_stage2, ratio=offspring_ratio_stage2, H=H_ref_points_stage2,
            gx=gx2, alt=altitude_levels, cs=cell_size, scales=refine_scales,
            air_risk_threshold=current_threshold, # [수정] 계산된 임계값 사용
            dz=delta_z_max, w_d=w_dist, w_g=w_ground, w_a=w_air,
            lat_lim=lat_lim, lon_lim=lon_lim,
            MAX_INIT_ATTEMPTS=MAX_INIT_ATTEMPTS_stage2,
            min_inter_nodes=MIN_INTER_NODES_stage2,
            max_inter_nodes=MAX_INTER_NODES_stage2,
            is_initial_stage=False, # [중요] 메인 단계
            initial_population=initial_solutions # [중요] Stage 1 결과 주입
        )

        # 2-4. 최종 결과 저장 및 시각화
        # [설명] Stage 2 결과 중 대표 해(각 목표별 최적해 + 균형해)를 선택하여 저장합니다.
        if population_stage2 and f_vals_stage2.shape[0] > 0:
            num_obj = f_vals_stage2.shape[1]
            rep_paths = []
            # 각 목표에 대해 최적해 선택
            for i in range(num_obj):
                rep_paths.append(population_stage2[np.argmin(f_vals_stage2[:, i])])
            
            # 파레토 최전선에서 균형해 선택
            F = fast_non_dominated_sort(f_vals_stage2)
            if F and F[0]:
                front1 = F[0]
                norm_f = normalize_objectives(f_vals_stage2[front1, :])
                balanced_idx = front1[np.argmin(np.linalg.norm(norm_f, axis=1))]
                rep_paths.append(population_stage2[balanced_idx])
            else: # Fallback
                rep_paths.append(rep_paths[0])

            representative_paths_final.append(rep_paths)

            if gx2:
                styles = ['r-', 'b-', 'g-', 'm-']
                labels = [f'{name} Min' for name in objective_names[:num_obj]] + ['Balanced']
                plot_solutions(gx2, rep_paths, styles, 1.2, k+1, labels=labels)
                gx2.legend(loc='upper right', bbox_to_anchor=(-0.1, 1), borderaxespad=0.)
                plt.pause(1)
        else:
            representative_paths_final.append([])
            print("  -> Stage 2 failed.")

    # [수정] 마지막 세그먼트의 노드(scatter) 제거
    if stage2_current_scatter is not None:
        stage2_current_scatter.remove()
        stage2_current_scatter = None

    # --- 5. Final Visualization ---
    # [설명] 모든 세그먼트의 최적화가 완료된 후, 전체 경로를 종합하여 시각화합니다.
    
    # [추가] Stage 1 최종 결과 시각화
    if gx1:
        gx1.set_title('Final Stage 1 Initial Solutions', size=16)
        # 마지막에 남아있는 라인 제거
        for line in stage1_current_lines:
            line.remove()
        
        print('\nDisplaying all Stage 1 initial solutions on Figure 1...')
        
        # [수정] 세그먼트별 색상 생성 (jet colormap 사용)
        import matplotlib.cm as cm
        colors = cm.jet(np.linspace(0, 1, num_segments))
        
        for k, solutions in enumerate(all_stage1_solutions_history):
            if solutions:
                # 각 세그먼트별 고유 색상 적용
                seg_color = colors[k]
                plot_solutions(gx1, solutions, ['-'], 1.2, k+1, labels=["Init"], custom_color=seg_color)
                
        # 범례 추가
        handles, labels = gx1.get_legend_handles_labels()
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels[label] = handle
        # [수정] 범례 위치를 좌측으로 변경 (Figure 2와 동일)
        gx1.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', bbox_to_anchor=(-0.1, 1), borderaxespad=0.)
        
        plt.pause(0.1)

    if gx2:
        gx2.set_title('Final UAM Corridor Paths', size=16)
        print('\nDisplaying all final representative paths on Figure 2...')
        
        # 최종 End-to-End 경로 조합 및 시각화
        num_obj_final = len(objective_names)
        num_rep_path_types = num_obj_final + 1 # 각 목표 최적해 + 균형해
        
        final_routes = [np.vstack([p[i] for p in representative_paths_final if p and i < len(p) and p[i] is not None]) for i in range(num_rep_path_types)]
        
        styles = ['r-', 'b-', 'g-', 'm-']
        labels = [f'Overall {name} Min' for name in objective_names] + ['Overall Balanced']
        
        for i, route in enumerate(final_routes):
            if route.shape[0] > 0:
                gx2.plot(route[:, 1], route[:, 0], styles[i % len(styles)], linewidth=1.2, transform=ccrs.Geodetic(), label=labels[i], zorder=20)

        handles, labels = gx2.get_legend_handles_labels()
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels[label] = handle
        gx2.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', bbox_to_anchor=(-0.1, 1), borderaxespad=0.)
        print('Figure 2: Final results displayed.')
        plt.ioff()
        plt.show()

    print('All tasks completed.')


if __name__ == '__main__':
    main()