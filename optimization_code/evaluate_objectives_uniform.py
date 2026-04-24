"""
evaluate_objectives_uniform.py

전체 경로를 시작~끝까지 일정 간격(sample_spacing_m)으로 리샘플링한 뒤
각 샘플 점에서 위험도를 평가하는 방식.
기존 세그먼트별 보간 방식(evaluate_objectives_GP.py) 대비
거리와 위험도 목적함수의 독립성을 확보.
"""

import numpy as np
from scipy.ndimage import map_coordinates


def _resample_path_uniform(path, spacing_m):
    """
    path (N x 3: lat, lon, alt)를 일정 거리 간격으로 리샘플링.
    반환: resampled (M x 3), headings (M-1,) in degrees [0,360)
    """
    ref_lat = float(np.mean(path[:, 0]))
    m_lat = 111000.0
    m_lon = 111000.0 * np.cos(np.deg2rad(ref_lat))

    # 누적 거리 계산
    diffs = np.diff(path, axis=0)
    seg_dists = np.sqrt(
        (diffs[:, 0] * m_lat) ** 2 +
        (diffs[:, 1] * m_lon) ** 2 +
        diffs[:, 2] ** 2
    )
    cum_dist = np.concatenate(([0.0], np.cumsum(seg_dists)))
    total_dist = cum_dist[-1]

    if total_dist < 1e-6:
        return path.copy(), np.array([0.0])

    # 일정 간격 샘플 위치
    n_samples = max(int(np.ceil(total_dist / spacing_m)) + 1, 2)
    sample_dists = np.linspace(0.0, total_dist, n_samples)

    # 각 샘플 위치에 대응하는 보간 좌표
    resampled = np.zeros((n_samples, 3), dtype=float)
    for dim in range(3):
        resampled[:, dim] = np.interp(sample_dists, cum_dist, path[:, dim])

    return resampled, sample_dists


def evaluate_objectives_uniform(
    path,
    Norm_RT,
    AirRisk,
    use_heading_map,
    altitude_levels,
    cell_size,
    air_risk_threshold,
    w_dist,
    w_ground,
    w_air,
    lat_lim,
    lon_lim,
    sample_spacing_m=50.0,
    NoiseRisk=None,
    noise_floor_db=0.0,
    w_noise=1.0,
):
    """
    전체 경로를 sample_spacing_m 간격으로 균일 리샘플링 후 위험도 평가.

    Returns:
        np.ndarray: [총 3D 거리, 누적 지상위험, 누적 공중위험(, 누적 소음위험)]
    """
    # 1) 총 3D 거리
    total_dist = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)) * w_dist

    # 2) 균일 리샘플링
    resampled, _ = _resample_path_uniform(path, sample_spacing_m)
    n_pts = resampled.shape[0]

    if n_pts < 2:
        if NoiseRisk is None or np.size(NoiseRisk) == 0:
            return np.array([total_dist, 0.0, 0.0])
        return np.array([total_dist, 0.0, 0.0, 0.0])

    # 그리드 변환 준비
    _, _, Ny, Nx = Norm_RT.shape
    minLat, maxLat = lat_lim
    minLon, maxLon = lon_lim
    dLat_deg = (maxLat - minLat) / (Ny - 1) if Ny > 1 else 1e-10
    dLon_deg = (maxLon - minLon) / (Nx - 1) if Nx > 1 else 1e-10

    cumulative_ground_risk = 0.0
    cumulative_air_risk = 0.0
    cumulative_noise_risk = 0.0
    cumulative_risk = 0.0

    # 3) 각 샘플 점에서 위험도 평가 (헤딩/고도는 인접 점 기준)
    for i in range(n_pts):
        pt = resampled[i]

        # 헤딩: 현재→다음 점 방향 (마지막 점은 이전→현재)
        if i < n_pts - 1:
            vec = resampled[i + 1, :2] - pt[:2]
        else:
            vec = pt[:2] - resampled[i - 1, :2]

        if use_heading_map:
            theta = np.rad2deg(np.arctan2(vec[1], vec[0]))
            if theta < 0:
                theta += 360
            head_idx = int(round(theta / 45.0)) % 8
        else:
            head_idx = 0

        alt_idx = int(np.argmin(np.abs(altitude_levels - pt[2])))

        # 그리드 좌표
        Iq = (pt[1] - minLon) / dLon_deg
        Jq = (pt[0] - minLat) / dLat_deg
        coords = np.array([[Jq], [Iq]])

        # 지상 위험도
        ground_val = map_coordinates(
            Norm_RT[alt_idx, head_idx, :, :], coords, order=1, cval=0.0
        ).item()
        # 공중 위험도
        air_val = map_coordinates(
            AirRisk[:, :, alt_idx], coords, order=1, cval=0.0
        ).item()
        additive_air = air_val if air_val > air_risk_threshold else 0.0

        cumulative_ground_risk += ground_val
        cumulative_air_risk += air_val
        cumulative_risk += (ground_val * air_val) + additive_air

        # 소음
        if NoiseRisk is not None and np.size(NoiseRisk) > 0:
            if np.ndim(NoiseRisk) == 3:
                noise_map = NoiseRisk[:, :, alt_idx]
            else:
                noise_map = NoiseRisk
            noise_val = map_coordinates(noise_map, coords, order=1, cval=0.0).item()
            if noise_val > noise_floor_db:
                cumulative_noise_risk += noise_val

    # 4) 반환
    if NoiseRisk is None or np.size(NoiseRisk) == 0:
        return np.array([total_dist, cumulative_ground_risk, cumulative_air_risk])

    return np.array([
        total_dist,
        cumulative_ground_risk,
        cumulative_air_risk,
        cumulative_noise_risk * float(w_noise),
    ])
