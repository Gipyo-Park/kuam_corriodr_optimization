# import numpy as np
# from evaluate_objectives_GP import evaluate_objectives_gp

# # --- 금지 구역 검사를 위한 내부 헬퍼 함수들 ---
# def _direction(p, q, r):
#     val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
#     if abs(val) < 1e-10: return 0
#     return 1 if val > 0 else 2

# def _on_segment(p, q, r):
#     return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
#             q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

# def _segments_intersect(p1, q1, p2, q2):
#     o1 = _direction(p1, q1, p2); o2 = _direction(p1, q1, q2)
#     o3 = _direction(p2, q2, p1); o4 = _direction(p2, q2, q1)
#     if o1 != o2 and o3 != o4: return True
#     if o1 == 0 and _on_segment(p1, p2, q1): return True
#     if o2 == 0 and _on_segment(p1, q2, q1): return True
#     if o3 == 0 and _on_segment(p2, p1, q2): return True
#     if o4 == 0 and _on_segment(p2, q1, q2): return True
#     return False

# def _is_inside_rect_latlon(point, rect):
#     lon, lat = point[1], point[0]
#     min_lon, max_lon, min_lat, max_lat = rect
#     return (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)

# def _violates_forbidden_zone_latlon(p1, p2, rect):
#     if _is_inside_rect_latlon(p1, rect) or _is_inside_rect_latlon(p2, rect):
#         return True
#     min_lon, max_lon, min_lat, max_lat = rect
#     seg_p1, seg_p2 = (p1[1], p1[0]), (p2[1], p2[0])
#     edges = [((min_lon, min_lat), (max_lon, min_lat)), ((max_lon, min_lat), (max_lon, max_lat)),
#              ((max_lon, max_lat), (min_lon, max_lat)), ((min_lon, max_lat), (min_lon, min_lat))]
#     for edge_p1, edge_p2 in edges:
#         if _segments_intersect(seg_p1, seg_p2, edge_p1, edge_p2):
#             return True
#     return False

# def _corridor_violates_nfz_with_width(p1, p2, W_half, rect, lat_lim, lon_lim):
#     """
#     회랑폭(W_half)을 고려하여 경로 세그먼트가 NFZ를 침범하는지 검사합니다.
    
#     Args:
#         p1, p2: 경로 세그먼트의 시작점과 끝점 [lat, lon, alt]
#         W_half: 회랑 반폭 (미터)
#         rect: NFZ 사각형 [min_lon, max_lon, min_lat, max_lat]
#         lat_lim, lon_lim: 좌표계 변환을 위한 지도 범위
    
#     Returns:
#         bool: 침범 여부 (True = 침범)
#     """
#     # 미터 -> 위경도 변환
#     mean_lat_rad = np.deg2rad((p1[0] + p2[0]) / 2)
#     meters_per_lat_deg = 111000
#     meters_per_lon_deg = 111000 * np.cos(mean_lat_rad)
    
#     W_half_lat = W_half / meters_per_lat_deg
#     W_half_lon = W_half / meters_per_lon_deg
    
#     # 경로 방향 벡터
#     vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])
#     norm = np.linalg.norm(vec)
    
#     if norm < 1e-10:
#         # 점 하나만 확인
#         expanded_rect = [
#             rect[0] - W_half_lon, rect[1] + W_half_lon,
#             rect[2] - W_half_lat, rect[3] + W_half_lat
#         ]
#         return _is_inside_rect_latlon(p1, expanded_rect)
    
#     # 수직 벡터 (좌우 폭)
#     perp = np.array([-vec[1], vec[0]]) / norm
#     offset_lat = perp[0] * W_half_lat
#     offset_lon = perp[1] * W_half_lon
    
#     # 회랑의 4개 꼭지점 생성
#     corridor_corners = [
#         [p1[0] + offset_lat, p1[1] + offset_lon],  # p1 left
#         [p1[0] - offset_lat, p1[1] - offset_lon],  # p1 right
#         [p2[0] + offset_lat, p2[1] + offset_lon],  # p2 left
#         [p2[0] - offset_lat, p2[1] - offset_lon]   # p2 right
#     ]
    
#     # NFZ 사각형의 4개 꼭지점
#     min_lon, max_lon, min_lat, max_lat = rect
#     nfz_corners = [
#         (min_lon, min_lat), (max_lon, min_lat),
#         (max_lon, max_lat), (min_lon, max_lat)
#     ]
    
#     # 1. 회랑 꼭지점이 NFZ 안에 있는지 검사
#     for corner in corridor_corners:
#         if _is_inside_rect_latlon(np.array([corner[0], corner[1], 0]), rect):
#             return True
    
#     # 2. NFZ 꼭지점이 회랑 안에 있는지 검사 (간단히 중심선과의 거리로 판단)
#     for nfz_corner in nfz_corners:
#         # NFZ 꼭지점이 p1-p2 중심선으로부터 W_half 이내인지 확인
#         nfz_pt = np.array([nfz_corner[1], nfz_corner[0]])  # [lat, lon]
        
#         # p1-p2 선분과 nfz_pt 사이의 최단거리 계산
#         # 벡터 투영 방식
#         v = np.array([p2[0] - p1[0], p2[1] - p1[1]])
#         w = np.array([nfz_pt[0] - p1[0], nfz_pt[1] - p1[1]])
        
#         if norm < 1e-10:
#             dist_to_line = np.linalg.norm(w)
#         else:
#             t = max(0, min(1, np.dot(w, v) / (norm**2)))
#             projection = np.array([p1[0], p1[1]]) + t * v
#             diff = nfz_pt - projection
            
#             # 미터 단위로 변환하여 거리 계산
#             dist_lat_m = diff[0] * meters_per_lat_deg
#             dist_lon_m = diff[1] * meters_per_lon_deg
#             dist_to_line = np.sqrt(dist_lat_m**2 + dist_lon_m**2)
        
#         if dist_to_line <= W_half:
#             return True
    
#     # 3. 회랑 경계선과 NFZ 경계선의 교차 검사
#     corridor_edges = [
#         (corridor_corners[0], corridor_corners[2]),  # left edge
#         (corridor_corners[1], corridor_corners[3]),  # right edge
#         (corridor_corners[0], corridor_corners[1]),  # front edge
#         (corridor_corners[2], corridor_corners[3])   # back edge
#     ]
    
#     nfz_edges = [
#         (nfz_corners[0], nfz_corners[1]),
#         (nfz_corners[1], nfz_corners[2]),
#         (nfz_corners[2], nfz_corners[3]),
#         (nfz_corners[3], nfz_corners[0])
#     ]
    
#     for c_edge in corridor_edges:
#         c_p1 = (c_edge[0][1], c_edge[0][0])  # (lon, lat)
#         c_p2 = (c_edge[1][1], c_edge[1][0])
        
#         for nfz_edge in nfz_edges:
#             if _segments_intersect(c_p1, c_p2, nfz_edge[0], nfz_edge[1]):
#                 return True
    
#     return False


# # --- 메인 함수 (최종 수정본) ---
# def evaluate_objectives_with_constraints_gp(path, 
#                                              # 위험도 데이터
#                                              Norm_RT, 
#                                              AirRisk, 
#                                              # 제약 및 평가 파라미터
#                                              use_heading_map, 
#                                              flight_dist_limit, 
#                                              forbidden_zones, 
#                                              delta_z_max, 
#                                              altitude_levels, 
#                                              cell_size, 
#                                              refine_scales, 
#                                              air_risk_threshold, 
#                                              w_dist, # [추가] 거리 가중치
#                                              w_ground, 
#                                              w_air, 
#                                              lat_lim, 
#                                              lon_lim,
#                                              # [추가] 회랑폭 및 곱률 제약 파라미터
#                                              W_half=None,  # 회랑 반폭 (미터)
#                                              ground_speed_mps=None,  # 지상 속도 (m/s)
#                                              bank_angle_deg=25.0,  # 뱅크각 (도)
#                                              min_turn_radius_m=296.0,  # 최소 회전반경 (미터)
#                                              check_corridor_nfz=False,  # 회랑폭 NFZ 검사 활성화 (기본값 False)
#                                              check_turn_radius=False,  # 회전반경 검사 활성화 (기본값 False)
#                                              check_heading_continuity=False,  # 헤딩 연속성 검사 활성화 (기본값 False)
#                                              prev_segment_heading=None):  # 이전 세그먼트 종료 헤딩
#     """
#     경로의 제약 조건(거리, 고도 변화, 금지구역, 회랑폭, 곡률)을 평가하고, 만족 시 목적 함수를 계산합니다.

#     Args:
#         path (np.ndarray): N x 3 형태의 경로 [lat, lon, alt].
#         Norm_RT (np.ndarray): 정규화된 지상/인구 위험도 텐서.
#         AirRisk (np.ndarray): 정규화된 공중 위험도 텐서.
#         use_heading_map (bool): 헤딩맵 사용 여부.
#         flight_dist_limit (float): 노드 간 최대 2D 비행 거리.
#         forbidden_zones (np.ndarray): 금지 구역 목록.
#         delta_z_max (float): 인접 노드 간 최대 허용 고도 변화량.
#         altitude_levels, cell_size, refine_scales: evaluate_objectives_gp에 필요한 파라미터.
#         w_dist, w_ground, w_air: 거리/지상/공중 위험도 가중치.
#         lat_lim, lon_lim: 좌표계 변환을 위한 위경도 범위.
#         W_half (float): 회랑 반폭 (미터). None이면 회랑폭 NFZ 검사 비활성화.
#         ground_speed_mps (float): 지상 속도 (m/s). 회전반경 계산용.
#         bank_angle_deg (float): 뱅크각 (도).
#         min_turn_radius_m (float): 최소 회전반경 (미터).
#         check_corridor_nfz (bool): 회랑폭 기반 NFZ 검사 활성화 여부.
#         check_turn_radius (bool): 회전반경 검사 활성화 여부.
#         prev_segment_heading (float): 이전 세그먼트 종료 헤딩 (라디안). None이면 검사 안 함.

#     Returns:
#         tuple: (f_val, feasible, final_heading)
#             f_val (np.ndarray): [거리, 통합 위험도] 값. 제약 위반 시 페널티.
#             feasible (bool): 제약 조건 만족 여부.
#             final_heading (float): 경로의 마지막 세그먼트 헤딩 (라디안).
#     """
#     penalty_cost = 1e6
    
#     # 먼저 목적 함수를 호출하여 반환될 값의 개수를 확인합니다.
#     temp_f_val = evaluate_objectives_gp(path, Norm_RT, AirRisk, use_heading_map,
#                                         altitude_levels, cell_size, refine_scales,
#                                         air_risk_threshold, w_dist, w_ground, w_air, lat_lim, lon_lim)
    
#     # 제약 위반 시 반환될 페널티 배열을 동적으로 생성합니다.
#     num_objectives = len(temp_f_val)
#     penalty_cost_array = np.full(num_objectives, penalty_cost)
#     final_heading = 0.0  # 기본값

#     # --- 이제 제약 조건을 검사합니다. ---
#     if path.shape[0] < 2:
#         return penalty_cost_array, False, final_heading

#     diffs = np.diff(path, axis=0)

#     # 1) 노드 간 2D 거리 체크
#     dists_2d_deg = np.linalg.norm(diffs[:, :2], axis=1)
#     if np.any(dists_2d_deg > flight_dist_limit):
#         return penalty_cost_array, False, final_heading

#     # 2) 노드 간 고도 변화 체크
#     if np.any(np.abs(diffs[:, 2]) > delta_z_max):
#         return penalty_cost_array, False, final_heading

#     # 3) 금지 구역 체크 (중심선 기반 - 기존 방식)
#     if forbidden_zones is not None and forbidden_zones.shape[0] > 0:
#         for i in range(path.shape[0] - 1):
#             p1 = path[i, :]
#             p2 = path[i+1, :]
#             for rect in forbidden_zones:
#                 if _violates_forbidden_zone_latlon(p1, p2, rect):
#                     return penalty_cost_array, False, final_heading

#     # 4) [추가] 회랑폭 기반 NFZ 침범 검사
#     if check_corridor_nfz and W_half is not None and W_half > 0:
#         if forbidden_zones is not None and forbidden_zones.shape[0] > 0:
#             for i in range(path.shape[0] - 1):
#                 p1 = path[i, :]
#                 p2 = path[i+1, :]
#                 for rect in forbidden_zones:
#                     if _corridor_violates_nfz_with_width(p1, p2, W_half, rect, lat_lim, lon_lim):
#                         return penalty_cost_array, False, final_heading

#     # 5) [추가] 최소 회전반경 (곡률) 제약 검사
#     if check_turn_radius and path.shape[0] >= 3:
#         # 미터 변환 스케일
#         mean_lat_rad = np.deg2rad(np.mean(path[:, 0]))
#         meters_per_lat_deg = 111000
#         meters_per_lon_deg = 111000 * np.cos(mean_lat_rad)
        
#         for i in range(path.shape[0] - 2):
#             p1 = path[i, :]
#             p2 = path[i+1, :]
#             p3 = path[i+2, :]
            
#             # 세 점으로 곡률 반경 계산
#             # 벡터 v1 = p2 - p1, v2 = p3 - p2
#             v1_lat = (p2[0] - p1[0]) * meters_per_lat_deg
#             v1_lon = (p2[1] - p1[1]) * meters_per_lon_deg
#             v1 = np.array([v1_lat, v1_lon])
            
#             v2_lat = (p3[0] - p2[0]) * meters_per_lat_deg
#             v2_lon = (p3[1] - p2[1]) * meters_per_lon_deg
#             v2 = np.array([v2_lat, v2_lon])
            
#             # 회전각 계산
#             norm1 = np.linalg.norm(v1)
#             norm2 = np.linalg.norm(v2)
            
#             if norm1 < 1e-6 or norm2 < 1e-6:
#                 continue
            
#             cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
#             angle = np.arccos(cos_angle)
            
#             # 회전각이 너무 작으면 (거의 직선) 스킵
#             if angle < 1e-3:
#                 continue
            
#             # 곡률 반경 근사: R = d / (2 * sin(angle/2))
#             # 여기서 d는 p1-p3 사이의 거리
#             d_lat = (p3[0] - p1[0]) * meters_per_lat_deg
#             d_lon = (p3[1] - p1[1]) * meters_per_lon_deg
#             d = np.sqrt(d_lat**2 + d_lon**2)
            
#             if d < 1e-6:
#                 continue
            
#             sin_half_angle = np.sin(angle / 2)
#             if sin_half_angle < 1e-6:
#                 continue
            
#             radius = d / (2 * sin_half_angle)
            
#             # 최소 회전반경 검사
#             if radius < min_turn_radius_m:
#                 return penalty_cost_array, False, final_heading

#     # 6) [추가] 세그먼트 연결점 헤딩 연속성 검사
#     if check_heading_continuity and prev_segment_heading is not None:
#         # 현재 경로의 첫 세그먼트 헤딩 계산
#         if path.shape[0] >= 2:
#             p1 = path[0, :]
#             p2 = path[1, :]
#             vec = p2[:2] - p1[:2]
#             current_heading = np.arctan2(vec[1], vec[0])  # lon(x), lat(y)
            
#             # 헤딩 차이 계산 (라디안)
#             heading_diff = abs(current_heading - prev_segment_heading)
#             # 각도를 -π ~ π 범위로 정규화
#             heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi
#             heading_diff = abs(heading_diff)
            
#             # 허용 오차 (예: 10도 = 0.174 라디안)
#             max_heading_diff = np.deg2rad(10.0)
            
#             if heading_diff > max_heading_diff:
#                 return penalty_cost_array, False, final_heading

#     # 7) 최종 헤딩 계산 (다음 세그먼트로 전달용)
#     if path.shape[0] >= 2:
#         p_end1 = path[-2, :]
#         p_end2 = path[-1, :]
#         vec_end = p_end2[:2] - p_end1[:2]
#         final_heading = np.arctan2(vec_end[1], vec_end[0])

#     # 4) 모든 제약을 만족했으므로, 아까 계산해 둔 목적 함수 값을 반환합니다.
#     return temp_f_val, True, final_heading

import numpy as np
from evaluate_objectives_GP import evaluate_objectives_gp

def _direction(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if abs(val) < 1e-10:
        return 0
    return 1 if val > 0 else 2

def _on_segment(p, q, r):
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

def _segments_intersect(p1, q1, p2, q2):
    o1 = _direction(p1, q1, p2)
    o2 = _direction(p1, q1, q2)
    o3 = _direction(p2, q2, p1)
    o4 = _direction(p2, q2, q1)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p2, q1):
        return True
    if o2 == 0 and _on_segment(p1, q2, q1):
        return True
    if o3 == 0 and _on_segment(p2, p1, q2):
        return True
    if o4 == 0 and _on_segment(p2, q1, q2):
        return True
    return False

def _is_inside_rect_latlon(point, rect):
    lon, lat = float(point[1]), float(point[0])
    min_lon, max_lon, min_lat, max_lat = rect
    return (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)

def _violates_forbidden_zone_latlon(p1, p2, rect):
    if _is_inside_rect_latlon(p1, rect) or _is_inside_rect_latlon(p2, rect):
        return True
    min_lon, max_lon, min_lat, max_lat = rect
    seg_p1 = (float(p1[1]), float(p1[0]))  # (lon, lat)
    seg_p2 = (float(p2[1]), float(p2[0]))
    edges = [
        ((min_lon, min_lat), (max_lon, min_lat)),
        ((max_lon, min_lat), (max_lon, max_lat)),
        ((max_lon, max_lat), (min_lon, max_lat)),
        ((min_lon, max_lat), (min_lon, min_lat)),
    ]
    for e1, e2 in edges:
        if _segments_intersect(seg_p1, seg_p2, e1, e2):
            return True
    return False

def _corridor_violates_nfz_with_width(p1, p2, W_half, rect):
    mean_lat_rad = np.deg2rad((float(p1[0]) + float(p2[0])) / 2.0)
    meters_per_lat_deg = 111000.0
    meters_per_lon_deg = 111000.0 * np.cos(mean_lat_rad)

    W_half_lat = float(W_half) / meters_per_lat_deg
    W_half_lon = float(W_half) / meters_per_lon_deg

    vec = np.array([float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1])], dtype=float)  # [dlat, dlon]
    norm = float(np.linalg.norm(vec))

    if norm < 1e-12:
        expanded_rect = [rect[0] - W_half_lon, rect[1] + W_half_lon, rect[2] - W_half_lat, rect[3] + W_half_lat]
        return _is_inside_rect_latlon(p1, expanded_rect)

    perp = np.array([-vec[1], vec[0]], dtype=float) / norm  # perpendicular in (dlat, dlon) space
    offset_lat = perp[0] * W_half_lat
    offset_lon = perp[1] * W_half_lon

    corridor_corners = [
        [float(p1[0]) + offset_lat, float(p1[1]) + offset_lon],
        [float(p1[0]) - offset_lat, float(p1[1]) - offset_lon],
        [float(p2[0]) + offset_lat, float(p2[1]) + offset_lon],
        [float(p2[0]) - offset_lat, float(p2[1]) - offset_lon],
    ]

    min_lon, max_lon, min_lat, max_lat = rect
    nfz_corners = [(min_lon, min_lat), (max_lon, min_lat), (max_lon, max_lat), (min_lon, max_lat)]

    for corner in corridor_corners:
        if _is_inside_rect_latlon(np.array([corner[0], corner[1], 0.0]), rect):
            return True

    corridor_edges = [
        (corridor_corners[0], corridor_corners[2]),
        (corridor_corners[1], corridor_corners[3]),
        (corridor_corners[0], corridor_corners[1]),
        (corridor_corners[2], corridor_corners[3]),
    ]
    nfz_edges = [
        (nfz_corners[0], nfz_corners[1]),
        (nfz_corners[1], nfz_corners[2]),
        (nfz_corners[2], nfz_corners[3]),
        (nfz_corners[3], nfz_corners[0]),
    ]

    for c_edge in corridor_edges:
        c_p1 = (float(c_edge[0][1]), float(c_edge[0][0]))  # (lon, lat)
        c_p2 = (float(c_edge[1][1]), float(c_edge[1][0]))
        for nfz_edge in nfz_edges:
            if _segments_intersect(c_p1, c_p2, nfz_edge[0], nfz_edge[1]):
                return True

    return False


def _corridor_hits_binary_moc_with_width(
    p1,
    p2,
    W_half,
    MOCRisk,
    altitude_levels,
    lat_lim,
    lon_lim,
    along_step_m=80.0,
):
    """Return True if corridor footprint intersects any binary MOC=1 cell."""
    if MOCRisk is None or np.size(MOCRisk) == 0:
        return False

    moc = np.asarray(MOCRisk, dtype=float)
    if moc.ndim == 2:
        moc = moc[:, :, np.newaxis]

    Ny, Nx = moc.shape[0], moc.shape[1]
    minLat, maxLat = lat_lim
    minLon, maxLon = lon_lim
    dLat_deg = (maxLat - minLat) / (Ny - 1) if Ny > 1 else 1.0
    dLon_deg = (maxLon - minLon) / (Nx - 1) if Nx > 1 else 1.0

    mean_lat = 0.5 * (float(p1[0]) + float(p2[0]))
    m_lat = 111000.0
    m_lon = 111000.0 * np.cos(np.deg2rad(mean_lat))

    ax, ay = 0.0, 0.0
    bx = (float(p2[1]) - float(p1[1])) * m_lon
    by = (float(p2[0]) - float(p1[0])) * m_lat
    vx, vy = bx - ax, by - ay
    seg_len = float(np.hypot(vx, vy))
    if seg_len < 1e-9:
        return False

    ux, uy = vx / seg_len, vy / seg_len
    nx, ny = -uy, ux

    along_step = float(max(10.0, along_step_m))
    n_along = max(2, int(np.ceil(seg_len / along_step)) + 1)
    s_vals = np.linspace(0.0, seg_len, n_along)

    half_w = float(max(0.0, W_half if W_half is not None else 0.0))
    if half_w <= 1e-9:
        t_vals = np.array([0.0], dtype=float)
    else:
        cross_step = float(np.clip(half_w / 3.0, 20.0, 100.0))
        n_cross = max(3, int(np.ceil((2.0 * half_w) / cross_step)) + 1)
        t_vals = np.linspace(-half_w, half_w, n_cross)

    alts = np.asarray(altitude_levels, dtype=float).ravel()

    for s in s_vals:
        cx = ax + s * ux
        cy = ay + s * uy
        tau = s / seg_len
        alt_s = float(p1[2]) + tau * (float(p2[2]) - float(p1[2]))
        alt_idx = int(np.argmin(np.abs(alts - alt_s))) if alts.size > 0 else 0

        for t in t_vals:
            qx = cx + t * nx
            qy = cy + t * ny
            lon = float(p1[1]) + qx / m_lon
            lat = float(p1[0]) + qy / m_lat

            I = int(np.round((lon - minLon) / dLon_deg))
            J = int(np.round((lat - minLat) / dLat_deg))
            if I < 0 or I >= Nx or J < 0 or J >= Ny:
                continue
            if float(moc[J, I, alt_idx]) >= 0.5:
                return True

    return False


def _point_to_segment_distance_m(pt, a, b):
    """Minimum distance (m) from point pt to segment a-b in 2D metric plane."""
    ab = b - a
    ab2 = float(np.dot(ab, ab))
    if ab2 <= 1e-12:
        return float(np.linalg.norm(pt - a))
    t = float(np.dot(pt - a, ab) / ab2)
    t = float(np.clip(t, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(pt - proj))


def _segment_to_segment_min_distance_m(a1, a2, b1, b2):
    """Minimum distance (m) between two 2D line segments."""
    # If centerlines intersect, distance is zero.
    p1 = (float(a1[0]), float(a1[1]))
    q1 = (float(a2[0]), float(a2[1]))
    p2 = (float(b1[0]), float(b1[1]))
    q2 = (float(b2[0]), float(b2[1]))
    if _segments_intersect(p1, q1, p2, q2):
        return 0.0

    d1 = _point_to_segment_distance_m(a1, b1, b2)
    d2 = _point_to_segment_distance_m(a2, b1, b2)
    d3 = _point_to_segment_distance_m(b1, a1, a2)
    d4 = _point_to_segment_distance_m(b2, a1, a2)
    return float(min(d1, d2, d3, d4))


def _has_self_corridor_width_overlap(path, W_half, eps_m=1.0):
    """
    Return True if non-adjacent corridor strips (centerline buffered by W_half)
    overlap in interior.
    """
    if path is None or np.size(path) == 0:
        return False
    p = np.asarray(path, dtype=float).reshape(-1, 3)
    if p.shape[0] < 4:
        return False

    mean_lat = float(np.mean(p[:, 0]))
    m_lat = 111000.0
    m_lon = 111000.0 * np.cos(np.deg2rad(mean_lat))
    if abs(m_lon) < 1e-9:
        m_lon = 1e-9

    xy = np.column_stack([p[:, 1] * m_lon, p[:, 0] * m_lat])  # [x=lon_m, y=lat_m]
    n_seg = xy.shape[0] - 1
    if n_seg < 3:
        return False

    threshold = float(2.0 * float(W_half) - max(0.0, float(eps_m)))
    if threshold <= 0.0:
        return False

    for i in range(n_seg):
        a1 = xy[i]
        a2 = xy[i + 1]
        for j in range(i + 1, n_seg):
            # Adjacent segments share an endpoint by construction -> ignore.
            if j <= i + 1:
                continue
            b1 = xy[j]
            b2 = xy[j + 1]

            dmin = _segment_to_segment_min_distance_m(a1, a2, b1, b2)
            if dmin < threshold:
                return True
    return False


def _resample_path_by_step_m(path, step_m=400.0, min_step_m=300.0, max_step_m=500.0):
    if path is None or path.shape[0] < 2:
        return path

    ref_lat = float(np.mean(path[:, 0]))
    meters_per_lat_deg = 111000.0
    meters_per_lon_deg = 111000.0 * np.cos(np.deg2rad(ref_lat))

    lat = path[:, 0].astype(float)
    lon = path[:, 1].astype(float)
    alt = path[:, 2].astype(float)

    x = (lon - lon[0]) * meters_per_lon_deg
    y = (lat - lat[0]) * meters_per_lat_deg

    dx = np.diff(x)
    dy = np.diff(y)
    seg_len = np.sqrt(dx * dx + dy * dy)
    total = float(np.sum(seg_len))
    if total < 1e-6:
        return path

    step_m = float(np.clip(step_m, min_step_m, max_step_m))
    n = int(np.floor(total / step_m)) + 1
    n = max(n, 2)

    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    s_new = np.linspace(0.0, total, n)

    x_new = np.interp(s_new, s, x)
    y_new = np.interp(s_new, s, y)
    alt_new = np.interp(s_new, s, alt)

    lon_new = x_new / meters_per_lon_deg + lon[0]
    lat_new = y_new / meters_per_lat_deg + lat[0]
    return np.column_stack([lat_new, lon_new, alt_new])

def _circumradius_m(p_prev_xy, p_xy, p_next_xy):
    a = p_xy - p_prev_xy
    b = p_next_xy - p_xy
    c = p_next_xy - p_prev_xy

    la = float(np.linalg.norm(a))
    lb = float(np.linalg.norm(b))
    lc = float(np.linalg.norm(c))

    cross = float(abs(a[0] * b[1] - a[1] * b[0]))
    if cross < 1e-12:
        return np.inf

    return (la * lb * lc) / (2.0 * cross)

def evaluate_objectives_with_constraints_gp(
    path,
    Norm_RT,
    AirRisk,
    use_heading_map,
    flight_dist_limit,          # meters
    forbidden_zones,
    delta_z_max,
    altitude_levels,
    cell_size,
    refine_scales,
    air_risk_threshold,
    w_dist,
    w_ground,
    w_air,
    lat_lim,
    lon_lim,
    NoiseRisk=None,
    noise_floor_db=0.0,
    w_noise=1.0,
    W_half=None,
    check_corridor_nfz=False,
    MOCRisk=None,
    check_corridor_moc=False,
    check_corridor_self_overlap=True,
    vertiport=None, landing_entry=None, takeoff_complete=None,
    return_reason=False,
):
    penalty_cost = 1e6

    temp_f_val = evaluate_objectives_gp(
        path, Norm_RT, AirRisk, use_heading_map,
        altitude_levels, cell_size, refine_scales,
        air_risk_threshold, w_dist, w_ground, w_air,
        lat_lim, lon_lim,
        NoiseRisk=NoiseRisk,
        noise_floor_db=noise_floor_db,
        w_noise=w_noise,
    )

    num_objectives = int(len(temp_f_val))
    penalty_cost_array = np.full(num_objectives, penalty_cost, dtype=float)

    def _pack(ret_f, ret_ok, reason):
        if return_reason:
            return ret_f, ret_ok, str(reason)
        return ret_f, ret_ok

    if path is None or path.shape[0] < 2:
        return _pack(penalty_cost_array, False, "path_too_short")

    path = path.astype(float)

    if (vertiport is not None) and (landing_entry is not None) and (takeoff_complete is not None):
        path = np.vstack([vertiport, takeoff_complete, path, landing_entry, vertiport])

    diffs = np.diff(path, axis=0)

    # 1) 노드 간 2D 거리 체크 (meters)
    ref_lat = float(np.mean(path[:, 0]))
    meters_per_lat_deg = 111000.0
    meters_per_lon_deg = 111000.0 * np.cos(np.deg2rad(ref_lat))

    dlat_m = diffs[:, 0] * meters_per_lat_deg
    dlon_m = diffs[:, 1] * meters_per_lon_deg
    dists_2d_m = np.sqrt(dlat_m * dlat_m + dlon_m * dlon_m)

    if np.any(dists_2d_m > float(flight_dist_limit)):
        return _pack(penalty_cost_array, False, "segment_distance_exceeds_limit")

    # 2) 고도 변화
    if np.any(np.abs(diffs[:, 2]) > float(delta_z_max)):
        return _pack(penalty_cost_array, False, "segment_altitude_jump_exceeds_limit")

    # 3) 중심선 NFZ
    if forbidden_zones is not None and np.size(forbidden_zones) > 0:
        for i in range(path.shape[0] - 1):
            p1 = path[i, :]
            p2 = path[i + 1, :]
            for rect in forbidden_zones:
                if _violates_forbidden_zone_latlon(p1, p2, rect):
                    return _pack(penalty_cost_array, False, "nfz_centerline_intersection")

    # 4) 회랑폭 NFZ
    if check_corridor_nfz and W_half is not None and float(W_half) > 0.0:
        if forbidden_zones is not None and np.size(forbidden_zones) > 0:
            for i in range(path.shape[0] - 1):
                p1 = path[i, :]
                p2 = path[i + 1, :]
                for rect in forbidden_zones:
                    if _corridor_violates_nfz_with_width(p1, p2, float(W_half), rect):
                        return _pack(penalty_cost_array, False, "nfz_corridor_width_intersection")

    # 5) 회랑폭 MOC(장애물 위험) 하드 제약
    # MOC는 지상 장애물(산/건물/전봇대 등)로 인해 해당 고도에서 발생하는 공중 위험이며,
    # 바이너리(0/1) 지도에서 1 셀을 회랑폭이 닿거나 통과하면 infeasible 처리한다.
    if check_corridor_moc and W_half is not None and float(W_half) > 0.0:
        if MOCRisk is not None and np.size(MOCRisk) > 0:
            for i in range(path.shape[0] - 1):
                p1 = path[i, :]
                p2 = path[i + 1, :]
                if _corridor_hits_binary_moc_with_width(
                    p1,
                    p2,
                    float(W_half),
                    MOCRisk,
                    altitude_levels,
                    lat_lim,
                    lon_lim,
                ):
                    return _pack(penalty_cost_array, False, "moc_corridor_width_intersection : Overlapping of obstacle danger zone and corridor width")

    # 6) 회랑폭 자기겹침 하드 제약 (비인접 세그먼트끼리만 검사)
    if check_corridor_self_overlap and W_half is not None and float(W_half) > 0.0:
        self_overlap_eps_m = 1.0
        if _has_self_corridor_width_overlap(path, float(W_half), eps_m=self_overlap_eps_m):
            return _pack(penalty_cost_array, False, "self_corridor_width_overlap")

    return _pack(temp_f_val, True, "ok")

