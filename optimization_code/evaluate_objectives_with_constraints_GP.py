import numpy as np
from evaluate_objectives_GP import evaluate_objectives_gp

# --- 금지 구역 검사를 위한 내부 헬퍼 함수들 ---
def _direction(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if abs(val) < 1e-10: return 0
    return 1 if val > 0 else 2

def _on_segment(p, q, r):
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

def _segments_intersect(p1, q1, p2, q2):
    o1 = _direction(p1, q1, p2); o2 = _direction(p1, q1, q2)
    o3 = _direction(p2, q2, p1); o4 = _direction(p2, q2, q1)
    if o1 != o2 and o3 != o4: return True
    if o1 == 0 and _on_segment(p1, p2, q1): return True
    if o2 == 0 and _on_segment(p1, q2, q1): return True
    if o3 == 0 and _on_segment(p2, p1, q2): return True
    if o4 == 0 and _on_segment(p2, q1, q2): return True
    return False

def _is_inside_rect_latlon(point, rect):
    lon, lat = point[1], point[0]
    min_lon, max_lon, min_lat, max_lat = rect
    return (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)

def _violates_forbidden_zone_latlon(p1, p2, rect):
    if _is_inside_rect_latlon(p1, rect) or _is_inside_rect_latlon(p2, rect):
        return True
    min_lon, max_lon, min_lat, max_lat = rect
    seg_p1, seg_p2 = (p1[1], p1[0]), (p2[1], p2[0])
    edges = [((min_lon, min_lat), (max_lon, min_lat)), ((max_lon, min_lat), (max_lon, max_lat)),
             ((max_lon, max_lat), (min_lon, max_lat)), ((min_lon, max_lat), (min_lon, min_lat))]
    for edge_p1, edge_p2 in edges:
        if _segments_intersect(seg_p1, seg_p2, edge_p1, edge_p2):
            return True
    return False


# --- 메인 함수 (최종 수정본) ---
def evaluate_objectives_with_constraints_gp(path, 
                                             # 위험도 데이터
                                             Norm_RT, 
                                             AirRisk, 
                                             # 제약 및 평가 파라미터
                                             use_heading_map, 
                                             flight_dist_limit, 
                                             forbidden_zones, 
                                             delta_z_max, 
                                             altitude_levels, 
                                             cell_size, 
                                             refine_scales, 
                                             air_risk_threshold, 
                                             w_dist, # [추가] 거리 가중치
                                             w_ground, 
                                             w_air, 
                                             lat_lim, 
                                             lon_lim):
    """
    경로의 제약 조건(거리, 고도 변화, 금지구역)을 평가하고, 만족 시 목적 함수를 계산합니다.
    (main 스크립트의 호출에 맞게 인자 개수를 수정한 최종 버전)

    Args:
        path (np.ndarray): N x 3 형태의 경로 [lat, lon, alt].
        Norm_RT (np.ndarray): 정규화된 지상/인구 위험도 텐서.
        AirRisk (np.ndarray): 정규화된 공중 위험도 텐서.
        use_heading_map (bool): 헤딩맵 사용 여부.
        flight_dist_limit (float): 노드 간 최대 2D 비행 거리 (원본 MATLAB 로직에 따라 위경도 차이값 기준).
        forbidden_zones (np.ndarray): 금지 구역 목록.
        delta_z_max (float): 인접 노드 간 최대 허용 고도 변화량.
        altitude_levels, cell_size, refine_scales: evaluate_objectives_gp에 필요한 파라미터.
        w_dist, w_ground, w_air: 거리/지상/공중 위험도 가중치.
        lat_lim, lon_lim: 좌표계 변환을 위한 위경도 범위.

    Returns:
        tuple: (f_val, feasible)
            f_val (np.ndarray): [거리, 통합 위험도] 값. 제약 위반 시 페널티.
            feasible (bool): 제약 조건 만족 여부.
    """
    penalty_cost = 1e6
    
    # 먼저 목적 함수를 호출하여 반환될 값의 개수를 확인합니다.
    # 이 값은 아직 제약조건이 검사되지 않았으므로 최종 결과가 아닙니다.
    temp_f_val = evaluate_objectives_gp(path, Norm_RT, AirRisk, use_heading_map,
                                        altitude_levels, cell_size, refine_scales,
                                        air_risk_threshold, w_dist, w_ground, w_air, lat_lim, lon_lim)
    
    # 제약 위반 시 반환될 페널티 배열을 동적으로 생성합니다.
    num_objectives = len(temp_f_val)
    penalty_cost_array = np.full(num_objectives, penalty_cost)

    # --- 이제 제약 조건을 검사합니다. ---
    if path.shape[0] < 2:
        return penalty_cost_array, False

    diffs = np.diff(path, axis=0)

    # 1) 노드 간 2D 거리 체크
    dists_2d_deg = np.linalg.norm(diffs[:, :2], axis=1)
    if np.any(dists_2d_deg > flight_dist_limit):
        return penalty_cost_array, False

    # 2) 노드 간 고도 변화 체크
    if np.any(np.abs(diffs[:, 2]) > delta_z_max):
        return penalty_cost_array, False

    # 3) 금지 구역 체크
    if forbidden_zones is not None and forbidden_zones.shape[0] > 0:
        for i in range(path.shape[0] - 1):
            p1 = path[i, :]
            p2 = path[i+1, :]
            for rect in forbidden_zones:
                if _violates_forbidden_zone_latlon(p1, p2, rect):
                    return penalty_cost_array, False

    # 4) 모든 제약을 만족했으므로, 아까 계산해 둔 목적 함수 값을 반환합니다.
    return temp_f_val, True