import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.ndimage import map_coordinates
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
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

def generate_valid_nodes_near_vertiport(vertiport, target_altitude, takeoff_landing_angle_deg, 
                                         sector_angle_deg, num_available_sectors, 
                                         lat_lim, lon_lim, Ny, Nx, forbidden_zones):
    """
    버티포트 중심으로 이착륙 각도 기반 섹터 방식으로 멀티스타트 노드를 생성합니다.
    
    로직:
    1. 이착륙 각도로부터 거리 계산: d = target_altitude / tan(angle)
    2. 360도를 sector_angle_deg 단위로 분할 (예: 30도 → 12개 섹터)
    3. 랜덤하게 num_available_sectors개 선택 (비행 가능 영역)
    4. 각 섹터의 중간 각도에서 노드 생성 (호의 중간점)
    5. NFZ 검사하여 금지구역 내 노드 제외
    
    Args:
        vertiport: 버티포트 위치 [lat, lon, alt]
        target_altitude: 목표 고도 (미터)
        takeoff_landing_angle_deg: 이착륙 각도 (도)
        sector_angle_deg: 섹터 분할 각도 (도) - 예: 30도면 12개 섹터
        num_available_sectors: 선택할 비행 가능 섹터 수
        lat_lim, lon_lim: 지도 범위
        Ny, Nx: 지도 그리드 크기
        forbidden_zones: 금지 구역 정보
    
    Returns:
        valid_nodes: 생성된 유효 노드들 [N x 3] (각 섹터의 중간점)
    """
    print(f"  -> Generating sector-based takeoff/landing nodes...")
    print(f"     Takeoff/Landing Angle: {takeoff_landing_angle_deg}°")
    print(f"     Target Altitude: {target_altitude}m")
    print(f"     Sector Division: {sector_angle_deg}° per sector")
    
    # 1. 이착륙 각도로부터 수평 거리 계산
    angle_rad = np.deg2rad(takeoff_landing_angle_deg)
    if np.abs(np.tan(angle_rad)) < 1e-6:
        print("  -> Warning: Takeoff angle too small, using default 500m radius")
        horizontal_distance_m = 500.0
    else:
        horizontal_distance_m = target_altitude / np.tan(angle_rad)
    
    print(f"     Horizontal Distance (d): {horizontal_distance_m:.1f}m")
    
    # 2. 위경도 -> 미터 변환 스케일
    mean_lat_rad = np.deg2rad(vertiport[0])
    meters_per_lat_deg = 111000
    meters_per_lon_deg = 111000 * np.cos(mean_lat_rad)
    
    # 3. 전체 섹터 개수 계산
    num_total_sectors = int(360 / sector_angle_deg)
    print(f"     Total Sectors: {num_total_sectors}")
    
    # 4. 랜덤하게 비행 가능 섹터 선택
    if num_available_sectors > num_total_sectors:
        num_available_sectors = num_total_sectors
        print(f"     Warning: num_available_sectors exceeds total sectors, using {num_total_sectors}")
    
    all_sector_indices = list(range(num_total_sectors))
    selected_sector_indices = np.random.choice(all_sector_indices, 
                                               size=num_available_sectors, 
                                               replace=False)
    
    print(f"     Selected {num_available_sectors} sectors (indices): {sorted(selected_sector_indices)}")
    
    # 5. 각 섹터의 중간점에서 노드 생성
    valid_nodes = []
    
    for sector_idx in selected_sector_indices:
        # 섹터의 시작 각도와 끝 각도
        start_angle_deg = sector_idx * sector_angle_deg
        end_angle_deg = start_angle_deg + sector_angle_deg
        
        # 섹터의 중간 각도 (호의 중간점)
        mid_angle_deg = (start_angle_deg + end_angle_deg) / 2.0
        mid_angle_rad = np.deg2rad(mid_angle_deg)
        
        # 극좌표 -> 직교좌표 변환 (북쪽이 0도, 시계방향)
        # 지리 좌표계: 북쪽 0도 기준, 시계방향
        offset_lat_deg = (horizontal_distance_m / meters_per_lat_deg) * np.cos(mid_angle_rad)
        offset_lon_deg = (horizontal_distance_m / meters_per_lon_deg) * np.sin(mid_angle_rad)
        
        candidate_lat = vertiport[0] + offset_lat_deg
        candidate_lon = vertiport[1] + offset_lon_deg
        candidate_alt = target_altitude  # 목표 고도로 설정
        
        # 지도 범위 내부 검사
        if not (lat_lim[0] <= candidate_lat <= lat_lim[1] and 
                lon_lim[0] <= candidate_lon <= lon_lim[1]):
            print(f"       Sector {sector_idx} ({start_angle_deg}°-{end_angle_deg}°): Out of bounds, skipped")
            continue
        
        # NFZ 검사
        is_in_nfz = False
        if forbidden_zones is not None and forbidden_zones.shape[0] > 0:
            for fz in forbidden_zones:
                min_lon, max_lon, min_lat, max_lat = fz
                if (min_lon <= candidate_lon <= max_lon and 
                    min_lat <= candidate_lat <= max_lat):
                    is_in_nfz = True
                    break
        
        if is_in_nfz:
            print(f"       Sector {sector_idx} ({start_angle_deg}°-{end_angle_deg}°): In NFZ, skipped")
            continue
        
        # 유효 노드 추가
        valid_nodes.append([candidate_lat, candidate_lon, candidate_alt])
        print(f"       Sector {sector_idx} ({start_angle_deg}°-{end_angle_deg}°, mid={mid_angle_deg:.1f}°): Node added at ({candidate_lat:.6f}, {candidate_lon:.6f}, {candidate_alt}m)")
    
    result = np.array(valid_nodes) if valid_nodes else np.empty((0, 3))
    print(f"  -> Generated {result.shape[0]} valid sector-based nodes (out of {num_available_sectors} selected sectors)")
    return result


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


def plot_corridor_width(gx, path, W_half, color='yellow', alpha=0.2, lat_lim=None, lon_lim=None):
    """
    경로의 회랑폭을 시각화합니다.
    
    Args:
        gx: Matplotlib Axes 객체
        path: 경로 [N x 3] (lat, lon, alt)
        W_half: 회랑 반폭 (미터)
        color: 색상
        alpha: 투명도
        lat_lim, lon_lim: 좌표계 변환을 위한 지도 범위
    """
    if path is None or path.shape[0] < 2:
        return None
    
    # 미터 -> 위경도 변환
    mean_lat_rad = np.deg2rad(np.mean(path[:, 0]))
    meters_per_lat_deg = 111000
    meters_per_lon_deg = 111000 * np.cos(mean_lat_rad)
    
    W_half_lat = W_half / meters_per_lat_deg
    W_half_lon = W_half / meters_per_lon_deg
    
    # 경로 양측으로 폭 생성
    left_points = []
    right_points = []
    
    for i in range(len(path)):
        if i == 0:
            # 첫 점: 다음 점과의 방향 사용
            vec = path[i+1][:2] - path[i][:2]
        elif i == len(path) - 1:
            # 마지막 점: 이전 점과의 방향 사용
            vec = path[i][:2] - path[i-1][:2]
        else:
            # 중간 점: 평균 방향 사용
            vec = path[i+1][:2] - path[i-1][:2]
        
        # 수직 벡터 계산
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            continue
        
        perp = np.array([-vec[1], vec[0]]) / norm
        
        # 좌우 점 생성 (위경도 단위)
        offset_lat = perp[0] * W_half_lat
        offset_lon = perp[1] * W_half_lon
        
        left_points.append([path[i, 0] + offset_lat, path[i, 1] + offset_lon])
        right_points.append([path[i, 0] - offset_lat, path[i, 1] - offset_lon])
    
    if len(left_points) < 2:
        return None
    
    # Polygon 생성 (left -> right 역순)
    polygon_points = left_points + right_points[::-1]
    polygon_array = np.array(polygon_points)
    
    # Cartopy 좌표계로 변환하여 그리기
    poly = Polygon(polygon_array[:, [1, 0]], closed=True, 
                   facecolor=color, edgecolor='none', alpha=alpha,
                   transform=ccrs.Geodetic(), zorder=2)
    gx.add_patch(poly)
    
    return poly


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

def run_nsga3_segment(nodes, p1, p2, Norm_RT, AirRisk, use_map, f_limit, f_zones, Nmax, N_pop, ratio, H, gx, alt, cs, scales, air_risk_threshold, dz, w_d, w_g, w_a, lat_lim, lon_lim, MAX_INIT_ATTEMPTS, min_inter_nodes, max_inter_nodes, is_initial_stage=False, initial_population=None, W_half=None, ground_speed_mps=None, bank_angle_deg=25.0, min_turn_radius_m=296.0, check_corridor_nfz=False, check_turn_radius=False, check_heading_continuity=False, prev_segment_heading=None, draw_final_corridor_width=True):
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
        W_half: 회랑 반폭 (미터)
        ground_speed_mps: 지상 속도 (m/s)
        bank_angle_deg: 뱅크각 (도)
        min_turn_radius_m: 최소 회전반경 (미터)
        check_corridor_nfz: 회랑폭 NFZ 검사 활성화
        check_turn_radius: 회전반경 검사 활성화
        prev_segment_heading: 이전 세그먼트 종료 헤딩 (라디안)
    """
    
    # --- 1. Determine Number of Objectives and Reference Points ---
    # [설명] 목적 함수의 개수를 동적으로 파악하고, 이에 맞는 참조점(Reference Points)을 생성합니다.
    # 임시 경로로 목적 함수를 한번 호출하여 목표 개수를 동적으로 파악
    dummy_path = np.vstack([p1, p2])
    temp_result = evaluate_objectives_with_constraints_gp(
        dummy_path, Norm_RT, AirRisk, use_map, f_limit, f_zones, dz, alt, cs, scales, 
        air_risk_threshold, w_d, w_g, w_a, lat_lim, lon_lim,
        W_half=W_half, ground_speed_mps=ground_speed_mps, bank_angle_deg=bank_angle_deg,
        min_turn_radius_m=min_turn_radius_m, check_corridor_nfz=check_corridor_nfz,
        check_turn_radius=check_turn_radius, prev_segment_heading=prev_segment_heading
    )
    
    # 반환값 처리 (하위 호환성)
    if len(temp_result) == 3:
        temp_f_val, _, _ = temp_result
    else:
        temp_f_val, _ = temp_result
    
    num_objectives = len(temp_f_val)
    
    # [수정] 사용자 요청: 목표 개수에 따라 H값 동적 설정 (H = 목표 개수 + 1)
    # H = num_objectives + 1
    # print(f"Running NSGA-III with {num_objectives} objectives. Setting H = {H}.")
    
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
        is_any_feasible = False
        for path in temp_full_pop:
            eval_result = evaluate_objectives_with_constraints_gp(
                path, Norm_RT, AirRisk, use_map, f_limit, f_zones, dz, alt, cs, scales,
                air_risk_threshold, w_d, w_g, w_a, lat_lim, lon_lim,
                W_half=W_half, ground_speed_mps=ground_speed_mps, bank_angle_deg=bank_angle_deg,
                min_turn_radius_m=min_turn_radius_m, check_corridor_nfz=check_corridor_nfz,
                check_turn_radius=check_turn_radius, check_heading_continuity=check_heading_continuity,
                prev_segment_heading=prev_segment_heading
            )
            
            if len(eval_result) == 3:
                _, feasible, _ = eval_result
            else:
                _, feasible = eval_result
            
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
            pass

    # 유효성 검사 및 재시도 (공통)
    if not population:
        print("Failed to generate or receive a valid initial population.")
        return [], np.array([]), None

    # --- 3. NSGA-III Main Loop ---
    # [설명] 정해진 세대 수(Nmax)만큼 진화를 반복합니다.
    h_paths = []
    final_heading = None
    
    for gen in range(1, Nmax + 1):
        if gen % 10 == 0 or gen == 1:
            print(f'  - Generation {gen}/{Nmax}')
            pass
        
        Np = len(population)
        f_vals = np.zeros((Np, num_objectives))
        feasible = np.zeros(Np, dtype=bool)
        headings = [None] * Np
        
        # 현재 세대의 모든 해에 대해 목적 함수 평가
        for i in range(Np):
            eval_result = evaluate_objectives_with_constraints_gp(
                population[i], Norm_RT, AirRisk, use_map, f_limit, f_zones, dz, alt, cs, scales,
                air_risk_threshold, w_d, w_g, w_a, lat_lim, lon_lim,
                W_half=W_half, ground_speed_mps=ground_speed_mps, bank_angle_deg=bank_angle_deg,
                min_turn_radius_m=min_turn_radius_m, check_corridor_nfz=check_corridor_nfz,
                check_turn_radius=check_turn_radius, check_heading_continuity=check_heading_continuity,
                prev_segment_heading=prev_segment_heading
            )
            
            if len(eval_result) == 3:
                f_vals[i, :], feasible[i], headings[i] = eval_result
            else:
                f_vals[i, :], feasible[i] = eval_result
                headings[i] = None
        
        # 시각화: Stage 2(Main)이거나, Stage 1이라도 가끔 업데이트
        if gx and (gen % 20 == 0 or gen == 1): 
            for line in h_paths: line.remove()
            h_paths.clear()
            
            # 모든 개체 그리기 (투명도 적용)
            for idx in range(Np):
                if feasible[idx]:
                    line, = gx.plot(population[idx][:, 1], population[idx][:, 0], '-', color=[0.5, 0.5, 0.5, 0.3], transform=ccrs.Geodetic())
                    h_paths.append(line)
            
            # [추가] Stage 2에서만 파레토 해들의 회랑폭 시각화 (얇은 점선)
            if not is_initial_stage and W_half is not None and W_half > 0:
                # 파레토 최전선 찾기
                feasible_indices = [i for i in range(Np) if feasible[i]]
                if feasible_indices:
                    feasible_f_vals = f_vals[feasible_indices, :]
                    F_vis = fast_non_dominated_sort(feasible_f_vals)
                    
                    if F_vis and F_vis[0]:
                        # 파레토 최전선의 해들에 대해 회랑폭 표시
                        front1_local = F_vis[0]
                        front1_global = [feasible_indices[i] for i in front1_local]
                        
                        for idx in front1_global:
                            path = population[idx]
                            if path.shape[0] >= 2:
                                # 회랑폭을 얇은 점선으로 표시
                                mean_lat_rad = np.deg2rad(np.mean(path[:, 0]))
                                meters_per_lat_deg = 111000
                                meters_per_lon_deg = 111000 * np.cos(mean_lat_rad)
                                W_half_lat = W_half / meters_per_lat_deg
                                W_half_lon = W_half / meters_per_lon_deg
                                
                                left_points = []
                                right_points = []
                                
                                for i in range(len(path)):
                                    if i == 0:
                                        vec = np.array([path[1, 0] - path[0, 0], path[1, 1] - path[0, 1]])
                                    elif i == len(path) - 1:
                                        vec = np.array([path[-1, 0] - path[-2, 0], path[-1, 1] - path[-2, 1]])
                                    else:
                                        vec = np.array([path[i+1, 0] - path[i-1, 0], path[i+1, 1] - path[i-1, 1]])
                                    
                                    norm = np.linalg.norm(vec)
                                    if norm < 1e-10:
                                        continue
                                    
                                    perp = np.array([-vec[1], vec[0]]) / norm
                                    offset_lat = perp[0] * W_half_lat
                                    offset_lon = perp[1] * W_half_lon
                                    
                                    left_points.append([path[i, 1] + offset_lon, path[i, 0] + offset_lat])
                                    right_points.append([path[i, 1] - offset_lon, path[i, 0] - offset_lat])
                                
                                if len(left_points) >= 2:
                                    left_arr = np.array(left_points)
                                    right_arr = np.array(right_points)
                                    
                                    # 얇은 점선으로 회랑폭 경계 표시
                                    line_l, = gx.plot(left_arr[:, 0], left_arr[:, 1], ':', 
                                                     color=[0.8, 0.6, 0.0, 0.4], linewidth=0.8, 
                                                     transform=ccrs.Geodetic(), zorder=4)
                                    line_r, = gx.plot(right_arr[:, 0], right_arr[:, 1], ':', 
                                                     color=[0.8, 0.6, 0.0, 0.4], linewidth=0.8, 
                                                     transform=ccrs.Geodetic(), zorder=4)
                                    h_paths.append(line_l)
                                    h_paths.append(line_r)
            
            plt.pause(1) # 여기를 수정하면 plot이 천천히 된다
            
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
        return [], np.array([]), None

    # --- 4. Final Evaluation ---
    # [설명] 최종 세대의 해들에 대해 마지막으로 목적 함수 값을 계산하여 반환합니다.
    f_vals_final = np.zeros((len(population), num_objectives))
    final_headings = []
    
    for i in range(len(population)):
        eval_result = evaluate_objectives_with_constraints_gp(
            population[i], Norm_RT, AirRisk, use_map, f_limit, f_zones, dz, alt, cs, scales,
            air_risk_threshold, w_d, w_g, w_a, lat_lim, lon_lim,
            W_half=W_half, ground_speed_mps=ground_speed_mps, bank_angle_deg=bank_angle_deg,
            min_turn_radius_m=min_turn_radius_m, check_corridor_nfz=check_corridor_nfz,
            check_turn_radius=check_turn_radius, check_heading_continuity=check_heading_continuity,
            prev_segment_heading=prev_segment_heading
        )
        
        if len(eval_result) == 3:
            f_vals_final[i,:], _, heading = eval_result
            final_headings.append(heading)
        else:
            f_vals_final[i,:], _ = eval_result
            final_headings.append(None)
    
    # 가장 우수한 해의 헤딩 반환
    if final_headings and any(h is not None for h in final_headings):
        valid_headings = [h for h in final_headings if h is not None]
        final_heading = valid_headings[0] if valid_headings else None
    
    # [수정] 최적화 중 임시 경로선만 제거하고, 최종 파레토 해의 회랑폭은 다시 그림
    if gx and h_paths:
        for line in h_paths:
            line.remove()
        h_paths.clear()
    
    # [수정] Stage 2에서만 Balanced 해의 회랑폭 시각화 유지 (플래그로 제어)
    if gx and not is_initial_stage and W_half is not None and W_half > 0 and draw_final_corridor_width:
        # 실행 가능한 해들 중 파레토 최전선 찾기
        feasible_indices = [i for i in range(len(population)) if np.all(f_vals_final[i, :] < 1e6)]
        if feasible_indices:
            feasible_f_vals = f_vals_final[feasible_indices, :]
            F_final = fast_non_dominated_sort(feasible_f_vals)
            
            if F_final and F_final[0]:
                # 파레토 최전선에서 Balanced 해 찾기
                front1_local = F_final[0]
                front1_fvals = feasible_f_vals[front1_local, :]
                
                # 정규화된 거리로 균형해 선택
                norm_f = normalize_objectives(front1_fvals)
                balanced_local_idx = np.argmin(np.linalg.norm(norm_f, axis=1))
                balanced_global_idx = feasible_indices[front1_local[balanced_local_idx]]
                
                print(f"  -> Drawing corridor width for Balanced solution only...")
                
                # Balanced 해의 회랑폭만 표시
                path = population[balanced_global_idx]
                if path.shape[0] >= 2:
                    # 회랑폭 계산
                    mean_lat_rad = np.deg2rad(np.mean(path[:, 0]))
                    meters_per_lat_deg = 111000
                    meters_per_lon_deg = 111000 * np.cos(mean_lat_rad)
                    W_half_lat = W_half / meters_per_lat_deg
                    W_half_lon = W_half / meters_per_lon_deg
                    
                    left_points = []
                    right_points = []
                    
                    for i in range(len(path)):
                        if i == 0:
                            vec = np.array([path[1, 0] - path[0, 0], path[1, 1] - path[0, 1]])
                        elif i == len(path) - 1:
                            vec = np.array([path[-1, 0] - path[-2, 0], path[-1, 1] - path[-2, 1]])
                        else:
                            vec = np.array([path[i+1, 0] - path[i-1, 0], path[i+1, 1] - path[i-1, 1]])
                        
                        norm = np.linalg.norm(vec)
                        if norm < 1e-10:
                            continue
                        
                        perp = np.array([-vec[1], vec[0]]) / norm
                        offset_lat = perp[0] * W_half_lat
                        offset_lon = perp[1] * W_half_lon
                        
                        left_points.append([path[i, 1] + offset_lon, path[i, 0] + offset_lat])
                        right_points.append([path[i, 1] - offset_lon, path[i, 0] - offset_lat])
                    
                    if len(left_points) >= 2:
                        left_arr = np.array(left_points)
                        right_arr = np.array(right_points)
                        
                        # Balanced 해의 회랑폭 - 진한 점선으로 유지
                        gx.plot(left_arr[:, 0], left_arr[:, 1], ':', 
                               color=[0.8, 0.6, 0.0, 0.7], linewidth=1.0, 
                               transform=ccrs.Geodetic(), zorder=15, label='_nolegend_')
                        gx.plot(right_arr[:, 0], right_arr[:, 1], ':', 
                               color=[0.8, 0.6, 0.0, 0.7], linewidth=1.0, 
                               transform=ccrs.Geodetic(), zorder=15, label='_nolegend_')
        
    return population, f_vals_final, final_heading

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
    
    # ===== [추가] 회랑 및 항공기 동역학 파라미터 =====
    W_half = 148.0  # 회랑 반폭 (미터) - TSE 148m 기준
    ground_speed_mps = 70.65  # 지상 속도 (m/s) - 약 148kt 기준
    bank_angle_deg = 25.0  # 뱅크각 (도)
    min_turn_radius_m = 296.0  # 최소 회전반경 (미터) - 계산식: R = V²/(g×tan(φ))
    
    # ===== 제약 조건 활성화/비활성화 설정 =====
    check_corridor_nfz = True  # 회랑폭 NFZ 검사 활성화 (True=활성화, False=비활성화)
    check_turn_radius = False  # 회전반경 검사 활성화 (True=활성화, False=비활성화)
    check_heading_continuity = False  # 헤딩 연속성 검사 활성화 (True=활성화, False=비활성화)
    max_heading_diff_deg = 10.0  # 세그먼트 연결점 헤딩 최대 차이 (도)
    
    # ===== 버티포트 이착륙 섹터 기반 멀티스타트 파라미터 =====
    ENABLE_VERTIPORT_MULTISTART = True  # 버티포트 멀티스타트 최적화 활성화 (첫 세그먼트에만 적용)
    TAKEOFF_LANDING_ANGLE_DEG = 25.0  # 이착륙 각도 (도) - 기본값 8도
    SECTOR_ANGLE_DEG = 30.0  # 섹터 분할 각도 (도) - 예: 30도면 12개 섹터, 45도면 8개 섹터
    NUM_AVAILABLE_SECTORS = 1  # 선택할 비행 가능 섹터 수 (랜덤 선택)
    TARGET_ALTITUDE = 500  # 목표 고도 (m) - 버티포트 이착륙 시나리오용
    
    
    # [수정] 2단계 최적화 파라미터 분리
    # --- Stage 1: Initial Solution Finding ---
    # [설명] 1단계: 초기 해 탐색 (비상착륙장 활용)
    # 목적: 넓은 범위에서 대략적인 경로를 빠르게 찾고, 실행 가능한 초기 해를 확보합니다.
    N_pop_stage1 = 50          # 인구 크기 (해 집단 크기)
    Nmax_stage1 = 100          # 최대 세대 수 (반복 횟수)
    offspring_ratio_stage1 = 1.0 # 자식 해 생성 비율
    H_ref_points_stage1 = 10   # 참조점 생성 파라미터 (3개 목표에 대해)
    MIN_INTER_NODES_stage1 = 1 # 비상착륙장 중 최소 1개 경유
    MAX_INTER_NODES_stage1 = 5 # 비상착륙장 중 최대 3개 경유
    MAX_INIT_ATTEMPTS_stage1 = 100 # 초기 해 생성 최대 시도 횟수 (비상착륙장 랜덤 샘플링)
    
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
    W_buf = 1250.0            # 경로 탐색 복도 폭 (m) - 노드 생성용
    node_grid_resolution_m = 100.0 # 노드 간격 (m). 이 값이 크면 노드가 듬성듬성 생성됩니다.
    
    # 경로 노드 개수 제약
    MIN_INTER_NODES_stage2 = 3  # 최소 중간 노드 개수
    MAX_INTER_NODES_stage2 = 50 # 최대 중간 노드 개수
    
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
    H_ref_points_stage2 = 10   # 참조점 생성 파라미터
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
    emergency_lat = np.array([35.6201083, 35.5678222, 35.5919889, 35.58, 35.625, 35.62, 35.624, 35.56, 35.620]); 
    emergency_lon = np.array([129.1191806, 129.106728, 129.0751972, 129.12, 129.115, 129.09, 129.065, 129.065, 129.125])
    # emergency_lat = np.array([35.620]); emergency_lon = np.array([129.125])
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
    
    # [추가] 헤딩 연속성 추적 변수
    segment_headings = []  # 각 세그먼트의 최종 헤딩 저장
    prev_heading = None  # 이전 세그먼트의 헤딩
    
    # [추가] 버티포트 멀티스타트 결과 저장
    vertiport_multistart_results = []  # 각 valid node별 최적화 결과
    best_vertiport_node = None  # 선택된 최적 valid node

    for k in tqdm(range(num_segments), desc="Optimizing Segments"):
        p1, p2 = points[k, :], points[k + 1, :]
        print(f'\n=== Processing Segment {k+1}/{num_segments} ===')
        
        # [추가] 버티포트 멀티스타트 로직 (첫 번째 세그먼트에만 적용)
        if k == 0 and ENABLE_VERTIPORT_MULTISTART:
            print(f"\n*** VERTIPORT MULTI-START MODE (Segment 1) ***")
            print(f"Using sector-based takeoff/landing node generation...")
            
            # 1. 버티포트 주변 섹터 기반 Valid Nodes 생성
            valid_nodes = generate_valid_nodes_near_vertiport(
                vertiport, TARGET_ALTITUDE, TAKEOFF_LANDING_ANGLE_DEG,
                SECTOR_ANGLE_DEG, NUM_AVAILABLE_SECTORS,
                lat_lim, lon_lim, Ny, Nx, forbidden_zones
            )
            
            if valid_nodes.shape[0] == 0:
                print("  WARNING: No valid nodes generated. Using vertiport as single start point.")
                valid_nodes = np.array([vertiport])
            
            # 시각화: Valid nodes 표시
            if gx1:
                gx1.scatter(valid_nodes[:, 1], valid_nodes[:, 0], s=100, c='magenta', 
                           marker='*', transform=ccrs.Geodetic(), label='Valid Nodes', zorder=10)
                gx1.legend(loc='upper right', bbox_to_anchor=(-0.1, 1), borderaxespad=0.)
                plt.pause(0.1)
            
            # 2. 각 Valid Node마다 Stage 1 & Stage 2 최적화 수행
            for vn_idx, valid_node in enumerate(valid_nodes):
                print(f"\n--- Optimizing from Valid Node {vn_idx+1}/{valid_nodes.shape[0]} ---")
                print(f"  Node position: {valid_node}")
                
                # 이 valid node를 시작점으로 사용
                p1_vn = valid_node
                
                # ---------------------------------------------------------
                # [Stage 1] Initial Solution Finding (Emergency Points)
                # ---------------------------------------------------------
                print(f"  -> Stage 1: Finding Initial Solutions")
                if gx1:
                    gx1.set_title(f'Stage 1: Valid Node {vn_idx+1} Initial Search', size=16)
                    for line in stage1_current_lines:
                        line.remove()
                    stage1_current_lines = []
                
                nodes_stage1 = emergency_points
                
                result_stage1 = run_nsga3_segment(
                    nodes=nodes_stage1, p1=p1_vn, p2=p2,
                    Norm_RT=Norm_RiskTensor, AirRisk=AirRisk, use_map=use_heading_map,
                    f_limit=flight_dist_limit, f_zones=forbidden_zones,
                    Nmax=Nmax_stage1, N_pop=N_pop_stage1, ratio=offspring_ratio_stage1, H=H_ref_points_stage1,
                    gx=gx1, alt=altitude_levels, cs=cell_size, scales=refine_scales,
                    air_risk_threshold=1.0,
                    dz=delta_z_max, w_d=w_dist, w_g=w_ground, w_a=w_air,
                    lat_lim=lat_lim, lon_lim=lon_lim,
                    MAX_INIT_ATTEMPTS=MAX_INIT_ATTEMPTS_stage1,
                    min_inter_nodes=MIN_INTER_NODES_stage1,
                    max_inter_nodes=MAX_INTER_NODES_stage1,
                    is_initial_stage=True,
                    W_half=W_half, ground_speed_mps=ground_speed_mps,
                    bank_angle_deg=bank_angle_deg, min_turn_radius_m=min_turn_radius_m,
                    check_corridor_nfz=check_corridor_nfz, check_turn_radius=check_turn_radius,
                    check_heading_continuity=check_heading_continuity,
                    prev_segment_heading=None  # 첫 세그먼트이므로 None
                )
                
                # 반환값 처리 (하위 호환성)
                if len(result_stage1) == 3:
                    population_stage1, f_vals_stage1, heading_stage1 = result_stage1
                else:
                    population_stage1, f_vals_stage1 = result_stage1
                    heading_stage1 = None
                
                # Select Initial Solutions
                initial_solutions = []
                if population_stage1 and f_vals_stage1.shape[0] > 0:
                    num_obj = f_vals_stage1.shape[1]
                    selected_indices = set()
                    for i in range(num_obj):
                        selected_indices.add(np.argmin(f_vals_stage1[:, i]))
                    
                    F = fast_non_dominated_sort(f_vals_stage1)
                    if F and F[0]:
                        front1 = F[0]
                        remaining = [idx for idx in front1 if idx not in selected_indices]
                        if remaining:
                            norm_f = normalize_objectives(f_vals_stage1[remaining, :])
                            best_bal = np.argmin(np.linalg.norm(norm_f, axis=1))
                            selected_indices.add(remaining[best_bal])
                    
                    initial_solutions = [population_stage1[i] for i in selected_indices]
                    print(f"  -> Selected {len(initial_solutions)} initial solutions from Stage 1.")
                    
                    if gx1:
                        stage1_current_lines = plot_solutions(gx1, initial_solutions, ['m-'], 1.2, k+1, labels=["Init"])
                        plt.pause(0.1)
                
                # ---------------------------------------------------------
                # [Stage 2] Main Optimization (Grid Nodes)
                # ---------------------------------------------------------
                print(f"  -> Stage 2: Main Optimization")
                if gx2:
                    gx2.set_title(f'Stage 2: Valid Node {vn_idx+1} Optimization', size=16)
                    if stage2_current_scatter is not None:
                        stage2_current_scatter.remove()
                        stage2_current_scatter = None
                
                # Generate nodes from valid node to p2
                nodes_stage2, all_grid_nodes = generate_nodes_3d_segment(
                    p1_vn, p2, W_buf, node_grid_resolution_m, lat_lim, lon_lim, Ny, Nx, forbidden_zones
                )
                
                # 안전 노드 필터링 (기존 로직)
                half_all_nodes = all_grid_nodes.shape[0] // 2
                base_target = int(max(MIN_SAFE_NODES_TARGET, half_all_nodes))
                current_min_safe_nodes_target = min(base_target, nodes_stage2.shape[0])
                
                safe_nodes = nodes_stage2
                if nodes_stage2.shape[0] > 0:
                    node_lons, node_lats, node_alts = nodes_stage2[:, 1], nodes_stage2[:, 0], nodes_stage2[:, 2]
                    I_nodes = ((node_lons - lon_lim[0]) / (lon_lim[1] - lon_lim[0]) * (Nx - 1)).astype(int)
                    J_nodes = ((node_lats - lat_lim[0]) / (lat_lim[1] - lat_lim[0]) * (Ny - 1)).astype(int)
                    alt_idx = np.argmin(np.abs(altitude_levels - node_alts[0]))
                    
                    avg_ground_risk_map = np.mean(Norm_RiskTensor[alt_idx, :, :, :], axis=0)
                    air_risk_slice = AirRisk[:, :, alt_idx]
                    air_risks_of_nodes = air_risk_slice[J_nodes, I_nodes]
                    node_risks = air_risks_of_nodes
                    
                    current_threshold = 0.0
                    for percentile in SAFE_NODE_PERCENTILE_LIST:
                        risk_threshold = np.percentile(node_risks, percentile)
                        current_threshold = risk_threshold
                        safe_nodes = nodes_stage2[node_risks <= risk_threshold]
                        if safe_nodes.shape[0] >= current_min_safe_nodes_target:
                            break
                    
                    if safe_nodes.shape[0] == 0:
                        safe_nodes = nodes_stage2
                    
                    # [추가] Stage 2 안전 노드 시각화
                    if gx2 and safe_nodes.shape[0] > 0:
                        s_lons, s_lats = safe_nodes[:, 1], safe_nodes[:, 0]
                        s_I = ((s_lons - lon_lim[0]) / (lon_lim[1] - lon_lim[0]) * (Nx - 1)).astype(int)
                        s_J = ((s_lats - lat_lim[0]) / (lat_lim[1] - lat_lim[0]) * (Ny - 1)).astype(int)
                        s_I = np.clip(s_I, 0, Nx - 1)
                        s_J = np.clip(s_J, 0, Ny - 1)
                        s_risks = air_risk_slice[s_J, s_I]
                        
                        stage2_current_scatter = gx2.scatter(s_lons, s_lats, c=s_risks, cmap='jet', 
                                                            vmin=0.0, vmax=1.0, s=10, alpha=0.5, 
                                                            transform=ccrs.Geodetic(), zorder=3)
                        
                        if cbar2 is None:
                            cbar2 = fig2.colorbar(stage2_current_scatter, ax=gx2, fraction=0.046, pad=0.04)
                            cbar2.set_label('Risk Level')
                        
                        plt.pause(0.1)
                
                # Stage 2 최적화 수행 (회랑폭 그리기 비활성화 - 나중에 최종 선택된 것만 그림)
                result_stage2 = run_nsga3_segment(
                    nodes=safe_nodes, p1=p1_vn, p2=p2,
                    Norm_RT=Norm_RiskTensor, AirRisk=AirRisk, use_map=use_heading_map,
                    f_limit=flight_dist_limit, f_zones=forbidden_zones,
                    Nmax=Nmax_stage2, N_pop=N_pop_stage2, ratio=offspring_ratio_stage2, H=H_ref_points_stage2,
                    gx=gx2, alt=altitude_levels, cs=cell_size, scales=refine_scales,
                    air_risk_threshold=current_threshold if 'current_threshold' in locals() else 0.0,
                    dz=delta_z_max, w_d=w_dist, w_g=w_ground, w_a=w_air,
                    lat_lim=lat_lim, lon_lim=lon_lim,
                    MAX_INIT_ATTEMPTS=MAX_INIT_ATTEMPTS_stage2,
                    min_inter_nodes=MIN_INTER_NODES_stage2,
                    max_inter_nodes=MAX_INTER_NODES_stage2,
                    is_initial_stage=False,
                    initial_population=initial_solutions,
                    W_half=W_half, ground_speed_mps=ground_speed_mps,
                    bank_angle_deg=bank_angle_deg, min_turn_radius_m=min_turn_radius_m,
                    check_corridor_nfz=check_corridor_nfz, check_turn_radius=check_turn_radius,
                    check_heading_continuity=check_heading_continuity,
                    prev_segment_heading=None,
                    draw_final_corridor_width=False  # 버티포트 멀티스타트에서는 회랑폭 그리지 않음
                )
                
                if len(result_stage2) == 3:
                    population_stage2, f_vals_stage2, heading_stage2 = result_stage2
                else:
                    population_stage2, f_vals_stage2 = result_stage2
                    heading_stage2 = None
                
                # 결과 저장 (Stage 1 초기해도 함께 저장)
                vertiport_multistart_results.append({
                    'valid_node': valid_node,
                    'valid_node_idx': vn_idx,
                    'population': population_stage2,
                    'f_vals': f_vals_stage2,
                    'final_heading': heading_stage2,
                    'stage1_initial_solutions': initial_solutions  # Stage 1 초기해 저장
                })
                
                print(f"  -> Completed optimization for Valid Node {vn_idx+1}")
            
            # 3. 모든 Valid Node 결과 중 가장 좋은 것 선택
            print(f"\n--- Selecting Best Valid Node ---")
            best_result = None
            best_score = float('inf')
            
            for res in vertiport_multistart_results:
                if res['population'] and res['f_vals'].shape[0] > 0:
                    # 파레토 최전선 찾기
                    F = fast_non_dominated_sort(res['f_vals'])
                    if F and F[0]:
                        front1_fvals = res['f_vals'][F[0], :]
                        # 정규화된 거리로 균형해 찾기
                        norm_f = normalize_objectives(front1_fvals)
                        balanced_idx = F[0][np.argmin(np.linalg.norm(norm_f, axis=1))]
                        score = np.linalg.norm(norm_f[np.argmin(np.linalg.norm(norm_f, axis=1)), :])
                        
                        if score < best_score:
                            best_score = score
                            best_result = res
                            best_vertiport_node = res['valid_node']
            
            if best_result is None:
                print("  ERROR: All valid nodes failed optimization. Falling back to vertiport.")
                best_vertiport_node = vertiport
                representative_paths_final.append([])
                continue
            
            print(f"  -> Best Valid Node: Index {best_result['valid_node_idx']}, Position: {best_vertiport_node}")
            
            # 4. 선택된 결과에서 대표 경로 추출 및 저장
            population_stage2 = best_result['population']
            f_vals_stage2 = best_result['f_vals']
            final_heading = best_result['final_heading']
            
            # 대표 경로 저장
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
                else:
                    rep_paths.append(rep_paths[0])
                
                representative_paths_final.append(rep_paths)
                
                # 헤딩 업데이트
                if final_heading is not None:
                    segment_headings.append(final_heading)
                    prev_heading = final_heading
                
                # 시각화
                if gx2:
                    styles = ['r-', 'b-', 'g-', 'm-']
                    labels = [f'{name} Min' for name in objective_names[:num_obj]] + ['Balanced']
                    plot_solutions(gx2, rep_paths, styles, 1.2, k+1, labels=labels)
                    
                    # [추가] Balanced 해에만 회랑폭 표시
                    balanced_path = rep_paths[-1]  # 마지막이 Balanced 해
                    if balanced_path.shape[0] >= 2:
                        mean_lat_rad = np.deg2rad(np.mean(balanced_path[:, 0]))
                        meters_per_lat_deg = 111000
                        meters_per_lon_deg = 111000 * np.cos(mean_lat_rad)
                        W_half_lat = W_half / meters_per_lat_deg
                        W_half_lon = W_half / meters_per_lon_deg
                        
                        left_points = []
                        right_points = []
                        
                        for i in range(len(balanced_path)):
                            if i == 0:
                                vec = np.array([balanced_path[1, 0] - balanced_path[0, 0], balanced_path[1, 1] - balanced_path[0, 1]])
                            elif i == len(balanced_path) - 1:
                                vec = np.array([balanced_path[-1, 0] - balanced_path[-2, 0], balanced_path[-1, 1] - balanced_path[-2, 1]])
                            else:
                                vec = np.array([balanced_path[i+1, 0] - balanced_path[i-1, 0], balanced_path[i+1, 1] - balanced_path[i-1, 1]])
                            
                            norm = np.linalg.norm(vec)
                            if norm < 1e-10:
                                continue
                            
                            perp = np.array([-vec[1], vec[0]]) / norm
                            offset_lat = perp[0] * W_half_lat
                            offset_lon = perp[1] * W_half_lon
                            
                            left_points.append([balanced_path[i, 1] + offset_lon, balanced_path[i, 0] + offset_lat])
                            right_points.append([balanced_path[i, 1] - offset_lon, balanced_path[i, 0] - offset_lat])
                        
                        if len(left_points) >= 2:
                            left_arr = np.array(left_points)
                            right_arr = np.array(right_points)
                            
                            # 점선으로 회랑폭 경계 표시
                            gx2.plot(left_arr[:, 0], left_arr[:, 1], ':', 
                                   color=[0.8, 0.6, 0.0, 0.7], linewidth=1.0, 
                                   transform=ccrs.Geodetic(), zorder=15, label='_nolegend_')
                            gx2.plot(right_arr[:, 0], right_arr[:, 1], ':', 
                                   color=[0.8, 0.6, 0.0, 0.7], linewidth=1.0, 
                                   transform=ccrs.Geodetic(), zorder=15, label='_nolegend_')
                    
                    gx2.legend(loc='upper right', bbox_to_anchor=(-0.1, 1), borderaxespad=0.)
                    plt.pause(1)
            else:
                representative_paths_final.append([])
            
            # p1을 선택된 valid node로 업데이트 (다음 세그먼트를 위해)
            p1 = best_vertiport_node
            
            # [수정] 선택된 Valid Node의 Stage 1 초기해를 이력에 저장
            if 'stage1_initial_solutions' in best_result and best_result['stage1_initial_solutions']:
                all_stage1_solutions_history.append(best_result['stage1_initial_solutions'])
                print(f"  -> Saved {len(best_result['stage1_initial_solutions'])} Stage 1 solutions for final visualization.")
            else:
                all_stage1_solutions_history.append([])
            
            # 다음 세그먼트로 이동 (일반 Stage 1/2 스킵)
            continue
            
        else:
            # [기존 로직] 버티포트가 아닌 일반 세그먼트 또는 멀티스타트 비활성화
            pass

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
        
        result_stage1 = run_nsga3_segment(
            nodes=nodes_stage1, p1=p1, p2=p2,
            Norm_RT=Norm_RiskTensor, AirRisk=AirRisk, use_map=use_heading_map,
            f_limit=flight_dist_limit, f_zones=forbidden_zones,
            Nmax=Nmax_stage1, N_pop=N_pop_stage1, ratio=offspring_ratio_stage1, H=H_ref_points_stage1,
            gx=gx1, alt=altitude_levels, cs=cell_size, scales=refine_scales,
            air_risk_threshold=1.0, # 1단계에서는 모든 비상착륙장을 고려하므로 임계값 0
            dz=delta_z_max, w_d=w_dist, w_g=w_ground, w_a=w_air,
            lat_lim=lat_lim, lon_lim=lon_lim,
            MAX_INIT_ATTEMPTS=MAX_INIT_ATTEMPTS_stage1, 
            min_inter_nodes=MIN_INTER_NODES_stage1, 
            max_inter_nodes=MAX_INTER_NODES_stage1,
            is_initial_stage=True, # 초기 단계임을 명시
            W_half=W_half, ground_speed_mps=ground_speed_mps,
            bank_angle_deg=bank_angle_deg, min_turn_radius_m=min_turn_radius_m,
            check_corridor_nfz=check_corridor_nfz, check_turn_radius=check_turn_radius,
            check_heading_continuity=check_heading_continuity,
            prev_segment_heading=prev_heading
        )
        
        # 반환값 처리 (하위 호환성)
        if len(result_stage1) == 3:
            population_stage1, f_vals_stage1, heading_stage1 = result_stage1
        else:
            population_stage1, f_vals_stage1 = result_stage1
            heading_stage1 = None

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
        result_stage2 = run_nsga3_segment(
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
            initial_population=initial_solutions, # [중요] Stage 1 결과 주입
            W_half=W_half, ground_speed_mps=ground_speed_mps,
            bank_angle_deg=bank_angle_deg, min_turn_radius_m=min_turn_radius_m,
            check_corridor_nfz=check_corridor_nfz, check_turn_radius=check_turn_radius,
            check_heading_continuity=check_heading_continuity,
            prev_segment_heading=prev_heading
        )
        
        # 반환값 처리 (하위 호환성)
        if len(result_stage2) == 3:
            population_stage2, f_vals_stage2, heading_stage2 = result_stage2
        else:
            population_stage2, f_vals_stage2 = result_stage2
            heading_stage2 = None

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
            
            # [추가] 헤딩 업데이트
            if heading_stage2 is not None:
                segment_headings.append(heading_stage2)
                prev_heading = heading_stage2

            if gx2:
                styles = ['r-', 'b-', 'g-', 'm-']
                labels = [f'{name} Min' for name in objective_names[:num_obj]] + ['Balanced']
                plot_solutions(gx2, rep_paths, styles, 1.2, k+1, labels=labels)
                
                # [추가] Balanced 해에만 회랑폭 표시
                balanced_path = rep_paths[-1]  # 마지막이 Balanced 해
                if balanced_path.shape[0] >= 2:
                    mean_lat_rad = np.deg2rad(np.mean(balanced_path[:, 0]))
                    meters_per_lat_deg = 111000
                    meters_per_lon_deg = 111000 * np.cos(mean_lat_rad)
                    W_half_lat = W_half / meters_per_lat_deg
                    W_half_lon = W_half / meters_per_lon_deg
                    
                    left_points = []
                    right_points = []
                    
                    for i in range(len(balanced_path)):
                        if i == 0:
                            vec = np.array([balanced_path[1, 0] - balanced_path[0, 0], balanced_path[1, 1] - balanced_path[0, 1]])
                        elif i == len(balanced_path) - 1:
                            vec = np.array([balanced_path[-1, 0] - balanced_path[-2, 0], balanced_path[-1, 1] - balanced_path[-2, 1]])
                        else:
                            vec = np.array([balanced_path[i+1, 0] - balanced_path[i-1, 0], balanced_path[i+1, 1] - balanced_path[i-1, 1]])
                        
                        norm = np.linalg.norm(vec)
                        if norm < 1e-10:
                            continue
                        
                        perp = np.array([-vec[1], vec[0]]) / norm
                        offset_lat = perp[0] * W_half_lat
                        offset_lon = perp[1] * W_half_lon
                        
                        left_points.append([balanced_path[i, 1] + offset_lon, balanced_path[i, 0] + offset_lat])
                        right_points.append([balanced_path[i, 1] - offset_lon, balanced_path[i, 0] - offset_lat])
                    
                    if len(left_points) >= 2:
                        left_arr = np.array(left_points)
                        right_arr = np.array(right_points)
                        
                        # 점선으로 회랑폭 경계 표시
                        gx2.plot(left_arr[:, 0], left_arr[:, 1], ':', 
                               color=[0.8, 0.6, 0.0, 0.7], linewidth=1.0, 
                               transform=ccrs.Geodetic(), zorder=15, label='_nolegend_')
                        gx2.plot(right_arr[:, 0], right_arr[:, 1], ':', 
                               color=[0.8, 0.6, 0.0, 0.7], linewidth=1.0, 
                               transform=ccrs.Geodetic(), zorder=15, label='_nolegend_')
                
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
        
        # [추가] 회랑폭 시각화를 위한 리스트
        corridor_polygons = []
        
        for i, route in enumerate(final_routes):
            if route.shape[0] > 0:
                # 경로 그리기
                gx2.plot(route[:, 1], route[:, 0], styles[i % len(styles)], linewidth=1.2, transform=ccrs.Geodetic(), label=labels[i], zorder=20)
                
                # [추가] 회랑폭 시각화 (파레토 최전선 경로에만 적용)
                # 여기서는 균형해 (마지막 경로)에 회랑폭 시각화
                if i == len(final_routes) - 1:  # 균형해 (Balanced solution)
                    poly = plot_corridor_width(gx2, route, W_half, color='yellow', alpha=0.15, lat_lim=lat_lim, lon_lim=lon_lim)
                    if poly is not None:
                        corridor_polygons.append(poly)

        handles, labels = gx2.get_legend_handles_labels()
        unique_labels = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels[label] = handle
        gx2.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', bbox_to_anchor=(-0.1, 1), borderaxespad=0.)
        print('Figure 2: Final results displayed.')
        
        # [추가] Waypoint 표시 (균형해 기준)
        if len(final_routes) > 0 and final_routes[-1].shape[0] > 0:
            balanced_route = final_routes[-1]
            # Waypoint를 다이아몬드 마커로 표시
            gx2.scatter(balanced_route[:, 1], balanced_route[:, 0], s=10, c='orange', 
                       marker='D', edgecolors='black', linewidths=1.5, 
                       transform=ccrs.Geodetic(), label='Waypoints', zorder=25)
    
    # =============================================================================
    # [추가] Waypoint 정보 저장 (CSV)
    # =============================================================================
    print('\n========================================')
    print('SAVING OPTIMAL WAYPOINTS TO CSV')
    print('========================================\n')
    
    import csv
    from datetime import datetime
    
    # CSV 파일 이름
    csv_filename = 'optimal_waypoints.csv'
    
    # 3가지 대표 솔루션 정의
    solution_types = ['Balanced', 'Distance-Optimal', 'Ground-Risk-Optimal', 'Air-Risk-Optimal']
    
    # CSV 헤더
    csv_header = [
        'SolutionType', 'SegmentID', 'WaypointID', 
        'Latitude', 'Longitude', 'Altitude_m',
        'SegmentDistance_m', 'CumulativeDist_km',
        'LocalGroundRisk', 'LocalAirRisk',
        'CumulGroundRisk', 'CumulAirRisk'
    ]
    
    # CSV 데이터 저장용 리스트
    csv_rows = []
    
    # 콘솔 출력 시작
    print('[OPTIMAL WAYPOINTS]\n')
    
    # 각 솔루션별로 처리
    for sol_idx, route in enumerate(final_routes):
        if route.shape[0] == 0:
            continue
        
        solution_name = solution_types[sol_idx] if sol_idx < len(solution_types) else f'Solution-{sol_idx+1}'
        
        print(f'--- {solution_name} ---')
        
        # 누적 거리 및 위험도 계산
        cumulative_dist_km = 0.0
        cumul_ground_risk = 0.0
        cumul_air_risk = 0.0
        
        # 각 waypoint별로 처리
        global_waypoint_id = 1
        
        # 세그먼트 구분을 위해 representative_paths_final에서 정보 가져오기
        segment_boundaries = []
        cumulative_waypoints = 0
        for seg_paths in representative_paths_final:
            if seg_paths and sol_idx < len(seg_paths) and seg_paths[sol_idx] is not None:
                num_waypoints = seg_paths[sol_idx].shape[0]
                segment_boundaries.append((cumulative_waypoints, cumulative_waypoints + num_waypoints))
                cumulative_waypoints += num_waypoints
        
        for wp_idx in range(route.shape[0]):
            lat, lon, alt = route[wp_idx]
            
            # 현재 waypoint가 속한 세그먼트 찾기
            current_segment = 1
            for seg_idx, (start, end) in enumerate(segment_boundaries):
                if start <= wp_idx < end:
                    current_segment = seg_idx + 1
                    break
            
            # 세그먼트 거리 계산 (이전 waypoint로부터)
            if wp_idx == 0:
                segment_distance_m = 0.0
            else:
                prev_lat, prev_lon, prev_alt = route[wp_idx - 1]
                # 3D 거리 계산
                dlat = (lat - prev_lat) * 111000  # 위도 차이 (m)
                dlon = (lon - prev_lon) * 111000 * np.cos(np.deg2rad((lat + prev_lat) / 2))  # 경도 차이 (m)
                dalt = alt - prev_alt  # 고도 차이 (m)
                segment_distance_m = np.sqrt(dlat**2 + dlon**2 + dalt**2)
                cumulative_dist_km += segment_distance_m / 1000.0
            
            # 지상/공중 위험도 (간단히 위치에서 보간)
            # 실제 값은 evaluate_objectives_gp에서 계산된 값 사용 가능
            # 여기서는 근사값으로 맵에서 직접 추출
            try:
                minLat, maxLat = lat_lim
                minLon, maxLon = lon_lim
                Ny, Nx = Norm_RiskTensor.shape[2], Norm_RiskTensor.shape[3]
                
                # 위경도 -> 그리드 인덱스
                i_grid = int((lon - minLon) / (maxLon - minLon) * (Nx - 1))
                j_grid = int((lat - minLat) / (maxLat - minLat) * (Ny - 1))
                
                # 범위 체크
                i_grid = max(0, min(Nx - 1, i_grid))
                j_grid = max(0, min(Ny - 1, j_grid))
                
                # 고도 인덱스
                alt_idx = np.argmin(np.abs(altitude_levels - alt))
                
                # 지상 위험도 (헤딩 인덱스 0 사용)
                local_ground_risk = float(Norm_RiskTensor[alt_idx, 0, j_grid, i_grid])
                
                # 공중 위험도
                local_air_risk = float(AirRisk[j_grid, i_grid, alt_idx])
                
                # 누적 위험도 (간단히 평균으로 근사)
                if wp_idx > 0:
                    cumul_ground_risk += local_ground_risk
                    cumul_air_risk += local_air_risk
                
            except Exception as e:
                local_ground_risk = 0.0
                local_air_risk = 0.0
            
            # CSV 행 추가
            csv_rows.append([
                solution_name,
                current_segment,
                global_waypoint_id,
                f'{lat:.6f}',
                f'{lon:.6f}',
                f'{alt:.1f}',
                f'{segment_distance_m:.2f}',
                f'{cumulative_dist_km:.4f}',
                f'{local_ground_risk:.6f}',
                f'{local_air_risk:.6f}',
                f'{cumul_ground_risk:.6f}',
                f'{cumul_air_risk:.6f}'
            ])
            
            # 콘솔 출력 (처음 3개와 마지막만)
            if wp_idx < 3 or wp_idx == route.shape[0] - 1:
                print(f'  Waypoint {global_waypoint_id:3d} [Seg {current_segment}]: '
                      f'Lat={lat:10.6f}, Lon={lon:10.6f}, Alt={alt:6.1f}m | '
                      f'Dist={segment_distance_m:7.1f}m | CumDist={cumulative_dist_km:6.2f}km')
            elif wp_idx == 3:
                print(f'  ... ({route.shape[0] - 4} more waypoints) ...')
            
            global_waypoint_id += 1
        
        # 솔루션 총합 출력
        print(f'  Total: {route.shape[0]} waypoints | '
              f'Distance: {cumulative_dist_km:.2f} km | '
              f'Ground Risk: {cumul_ground_risk:.4f} | '
              f'Air Risk: {cumul_air_risk:.4f}\n')
    
    # CSV 파일 저장
    try:
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)
            writer.writerows(csv_rows)
        
        print(f'✓ Waypoints saved to: {csv_filename}')
        print(f'  Total rows: {len(csv_rows)} waypoints')
        print('========================================\n')
    except Exception as e:
        print(f'✗ Error saving CSV: {e}\n')

    # =============================================================================
    # Plot 표시 (CSV 저장 후)
    # =============================================================================
    if gx2:
        plt.ioff()
        plt.show()

    print('All tasks completed.')


if __name__ == '__main__':
    main()