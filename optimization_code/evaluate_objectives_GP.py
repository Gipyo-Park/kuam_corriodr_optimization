# evaluate_objectives_GP.py

import numpy as np
from scipy.ndimage import map_coordinates

def evaluate_objectives_gp(path, 
                           # Risk Data
                           Norm_RT, 
                           AirRisk, 
                           # Parameters for evaluation
                           use_heading_map, 
                           altitude_levels, 
                           cell_size, 
                           refine_scales, 
                           air_risk_threshold, # [수정] 비용 함수 임계값 인자 추가
                           w_dist, # [추가] 거리 가중치
                           w_ground, 
                           w_air, 
                           lat_lim, 
                           lon_lim):
    """
    하나의 경로에 대해 '총 3D 거리'와 '누적 통합 위험도'를 평가합니다.
    (통합 위험도 로직 및 올바른 인자 개수 반영)

    Args:
        path (np.ndarray): N x 3 형태의 경로 [lat, lon, alt].
        Norm_RT (np.ndarray): 정규화된 지상/인구 위험도 텐서.
        AirRisk (np.ndarray): 정규화된 공중 위험도 텐서.
        use_heading_map (bool): 헤딩맵 사용 여부.
        altitude_levels (np.ndarray): 비행 고도 레벨 목록.
        cell_size (float): 그리드 한 칸의 크기 (m).
        refine_scales (np.ndarray): 경로 보간 정밀도.
        w_dist (float): 거리 가중치.
        w_ground (float): 지상/인구 위험도 가중치.
        w_air (float): 공중 위험도 가중치.
        lat_lim (list): 지도의 위도 범위 [min, max].
        lon_lim (list): 지도의 경도 범위 [min, max].

    Returns:
        np.ndarray: [총 3D 거리, 누적 통합 위험도] 값을 담은 배열.
    """
    # 1) 총 3D 거리 계산
    total_dist = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)) * w_dist

    # 2) 누적 위험도 계산을 위한 변수 초기화
    cumulative_risk = 0.0
    cumulative_ground_risk = 0.0
    cumulative_air_risk = 0.0
    
    # 위경도 좌표를 그리드 인덱스로 변환하기 위한 준비
    _, _, Ny, Nx = Norm_RT.shape
    minLat, maxLat = lat_lim
    minLon, maxLon = lon_lim
    dLat_deg = (maxLat - minLat) / (Ny - 1) if Ny > 1 else 0
    dLon_deg = (maxLon - minLon) / (Nx - 1) if Nx > 1 else 0

    for i in range(path.shape[0] - 1):
        p1, p2 = path[i, :], path[i+1, :]
        
        # 헤딩 인덱스 결정
        vec = p2[:2] - p1[:2]
        if use_heading_map:
            # 경도(x)가 vec[1], 위도(y)가 vec[0]
            theta = np.rad2deg(np.arctan2(vec[1], vec[0])) 
            if theta < 0: theta += 360
            rounded = round(theta / 45.0)
            head_idx = int(rounded % 8) # 8 -> 0
        else:
            head_idx = 0

        # 고도 인덱스 결정
        alt_idx = np.argmin(np.abs(altitude_levels - p1[2]))

        # 경로 보간
        dist_2d_m = np.linalg.norm(vec) * 111000 * np.cos(np.deg2rad(np.mean(path[:,0])))
        if dist_2d_m < 1e-6: continue
        
        if dist_2d_m < 200: refine_scale = refine_scales[3]
        elif dist_2d_m < 500: refine_scale = refine_scales[2]
        elif dist_2d_m < 1000: refine_scale = refine_scales[1]
        else: refine_scale = refine_scales[0]
        
        num_samples = int(np.ceil(dist_2d_m / (cell_size * refine_scale)))
        if num_samples < 2: num_samples = 2
        
        yq_lat = np.linspace(p1[0], p2[0], num_samples)
        xq_lon = np.linspace(p1[1], p2[1], num_samples)

        # 보간점들을 그리드 인덱스로 변환
        Iq = (xq_lon - minLon) / dLon_deg
        Jq = (yq_lat - minLat) / dLat_deg
        coords = np.vstack((Jq, Iq))

        # 통합 위험도 계산
        # 1. 지상/인구 위험도 보간
        ground_risk_map = Norm_RT[alt_idx, head_idx, :, :]
        interp_ground_risks = map_coordinates(ground_risk_map, coords, order=1, cval=0.0)
        
        # 2. 공중 위험도 보간
        air_risk_map = AirRisk[:, :, alt_idx]
        interp_air_risks = map_coordinates(air_risk_map, coords, order=1, cval=0.0)

        # 3. 임계값을 기준으로 추가 공중 위험도 계산
        additive_air_risk = np.where(interp_air_risks > air_risk_threshold, interp_air_risks, 0)
        
        
        # [코드 수정] 위험도 계산 방식을 가중합에서 곱셈(Expected Risk)으로 변경
        # 기존: combined_interp_risks = (w_ground * interp_ground_risks) + (w_air * interp_air_risks)
        # 변경 후: 공중 위험도(확률) * 지상 위험도(피해)
        # combined_interp_risks = (w_ground * interp_ground_risks) + (w_air * interp_air_risks)
        # combined_interp_risks = interp_ground_risks * interp_air_risks
        combined_interp_risks = (interp_ground_risks * interp_air_risks) + additive_air_risk

        
        cumulative_risk += np.sum(combined_interp_risks)
        cumulative_ground_risk += np.sum(interp_ground_risks)
        cumulative_air_risk += np.sum(interp_air_risks)

    # --- 최종 반환 값 ---
    # 사용자가 원하는 목표 조합을 아래 배열에 담아 반환합니다.
    # 주석을 수정하여 원하는 목표를 활성화/비활성화할 수 있습니다.
    
    # 예시 1: [거리, 통합위험] (기존 2개 목표)
    # return np.array([total_dist, cumulative_risk])
    
    # 예시 2: [거리, 지상위험, 공중위험] (요청된 3개 목표)
    return np.array([total_dist, cumulative_ground_risk, cumulative_air_risk])
    
    # 예시 3: [거리, 통합위험, 지상위험, 공중위험] (모든 정보 반환)
    # return np.array([total_dist, cumulative_risk, cumulative_ground_risk, cumulative_air_risk])