# import numpy as np

# def mutation_gp(path, nodes):
#     """
#     일정 확률(20%)로 경로의 중간 노드 하나를 다른 무작위 노드로 교체합니다.

#     Args:
#         path (np.ndarray): N x 3 형태의 원본 경로.
#         nodes (np.ndarray): M x 3 형태의 교체될 수 있는 전체 경유지 노드 목록.

#     Returns:
#         np.ndarray: 돌연변이가 적용되었거나, 원본과 동일한 경로.
#     """
#     # 경로에 중간 노드가 없거나(크기가 2 이하), 교체할 노드 목록이 비어있으면 돌연변이 수행 불가
#     has_intermediate_nodes = path.shape[0] > 2
#     nodes_are_available = nodes.shape[0] > 0
    
#     # 20%의 확률로 돌연변이 실행
#     if np.random.rand() < 0.2 and has_intermediate_nodes and nodes_are_available:
        
#         # 돌연변이를 적용할 중간 노드의 인덱스를 무작위로 선택
#         # path[0]은 시작점, path[-1]은 끝점이므로, 1부터 len(path)-2 사이에서 선택
#         # MATLAB의 randi([2, size-1])는 Python의 randint(1, size-1)과 동일한 범위
#         idx_to_mutate = np.random.randint(1, path.shape[0] - 1)
        
#         # 전체 노드 목록에서 새로운 노드를 무작위로 선택
#         new_node_idx = np.random.randint(0, nodes.shape[0])
#         new_node = nodes[new_node_idx, :]
        
#         # 경로의 해당 위치에 새로운 노드를 덮어씀
#         path[idx_to_mutate, :] = new_node

#     # 수정되었거나 원본 그대로의 경로를 반환
#     mutated_path = path
    
#     return mutated_path

import numpy as np


def _weighted_choice(indices, risks, risk_weight_strength):
    if indices.size == 0:
        return None
    if risks is None or risks.size != indices.size:
        return int(np.random.choice(indices))

    r = np.asarray(risks, dtype=float)
    finite = np.isfinite(r)
    if not np.any(finite):
        return int(np.random.choice(indices))

    r = np.where(finite, r, np.max(r[finite]))
    r_min = float(np.min(r))
    r_max = float(np.max(r))
    if r_max - r_min < 1e-12:
        return int(np.random.choice(indices))

    rn = (r - r_min) / (r_max - r_min + 1e-12)
    w = np.exp(-float(risk_weight_strength) * rn)
    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        return int(np.random.choice(indices))
    p = w / w_sum
    return int(np.random.choice(indices, p=p))


def _segment_local_candidates(nodes, p_prev, p_next, p_cur, strip_width_m, local_radius_m):
    if nodes is None or nodes.size == 0:
        return np.empty((0,), dtype=int)

    ref_lat = float(0.5 * (p_prev[0] + p_next[0]))
    mlat = 111000.0
    mlon = 111000.0 * np.cos(np.deg2rad(ref_lat))

    def ll_to_xy(latlon):
        return np.array([latlon[1] * mlon, latlon[0] * mlat], dtype=float)

    a = ll_to_xy(p_prev[:2])
    b = ll_to_xy(p_next[:2])
    c = ll_to_xy(p_cur[:2])

    pts = np.column_stack([nodes[:, 1] * mlon, nodes[:, 0] * mlat]).astype(float)

    v = b - a
    vv = float(np.dot(v, v))
    if vv < 1e-9:
        return np.empty((0,), dtype=int)

    w = pts - a[None, :]
    t = (w @ v) / vv
    t_clip = np.clip(t, 0.0, 1.0)
    proj = a[None, :] + t_clip[:, None] * v[None, :]
    d_seg = np.linalg.norm(pts - proj, axis=1)
    d_cur = np.linalg.norm(pts - c[None, :], axis=1)

    # 완전한 세그먼트 내부만 허용하지 않고 약간의 끝점 버퍼를 둠
    in_t = (t >= -0.10) & (t <= 1.10)
    mask = in_t & (d_seg <= float(strip_width_m)) & (d_cur <= float(local_radius_m))
    return np.where(mask)[0]


def _fallback_parallel_normal_mutation(mutated_path, i):
    ref_lat = float(np.mean(mutated_path[:, 0]))
    mlat = 111000.0
    mlon = 111000.0 * np.cos(np.deg2rad(ref_lat))

    def ll_to_xy(p):
        x = (p[1] - mutated_path[0, 1]) * mlon
        y = (p[0] - mutated_path[0, 0]) * mlat
        return np.array([x, y], dtype=float)

    def xy_to_ll(xy, alt):
        lon = xy[0] / mlon + mutated_path[0, 1]
        lat = xy[1] / mlat + mutated_path[0, 0]
        return np.array([lat, lon, alt], dtype=float)

    p_prev = mutated_path[i - 1]
    p = mutated_path[i]
    p_next = mutated_path[i + 1]

    prev_xy = ll_to_xy(p_prev)
    p_xy = ll_to_xy(p)
    next_xy = ll_to_xy(p_next)

    t = next_xy - prev_xy
    nt = float(np.linalg.norm(t))
    if nt < 1e-6:
        return mutated_path
    t = t / nt
    n = np.array([-t[1], t[0]], dtype=float)

    a_scale = np.random.choice([20.0, 50.0, 100.0], p=[0.5, 0.35, 0.15])
    b_scale = np.random.choice([30.0, 80.0, 150.0], p=[0.5, 0.35, 0.15])

    alpha = (2.0 * np.random.rand() - 1.0) * a_scale
    beta = (2.0 * np.random.rand() - 1.0) * b_scale
    new_xy = p_xy + alpha * t + beta * n

    mutated_path[i, :] = xy_to_ll(new_xy, p[2])
    return mutated_path


def mutation_gp(
    path,
    nodes,
    node_risks=None,
    mutation_rate=0.2,
    use_local_safe_resample=True,
    local_resample_prob=0.7,
    local_strip_width_m=350.0,
    local_radius_m=500.0,
    local_max_tries=4,
    risk_weight_boost=True,
    risk_weight_strength=2.0,
):
    # path: (N,3) [lat, lon, alt] 가정. alt는 유지
    if path is None or path.shape[0] < 3:
        return path

    mutated_path = path.copy()

    if np.random.rand() >= float(mutation_rate):
        return mutated_path

    i = np.random.randint(1, mutated_path.shape[0] - 1)

    nodes_arr = np.asarray(nodes, dtype=float) if nodes is not None else np.empty((0, 3), dtype=float)
    risks_arr = None
    if node_risks is not None:
        rr = np.asarray(node_risks, dtype=float).reshape(-1)
        if rr.size == nodes_arr.shape[0]:
            risks_arr = rr

    # 1) 부모 경로의 로컬 세그먼트 주변 safe node 재샘플링
    if use_local_safe_resample and nodes_arr.size > 0 and np.random.rand() < float(local_resample_prob):
        p_prev = mutated_path[i - 1]
        p_cur = mutated_path[i]
        p_next = mutated_path[i + 1]

        selected = None
        for _ in range(max(1, int(local_max_tries))):
            cand_idx = _segment_local_candidates(
                nodes_arr,
                p_prev,
                p_next,
                p_cur,
                strip_width_m=float(local_strip_width_m),
                local_radius_m=float(local_radius_m),
            )
            if cand_idx.size == 0:
                continue

            cand_risk = risks_arr[cand_idx] if (risk_weight_boost and risks_arr is not None) else None
            selected = _weighted_choice(cand_idx, cand_risk, risk_weight_strength)
            if selected is not None:
                break

        if selected is not None:
            mutated_path[i, :] = nodes_arr[selected, :]
            mutated_path[i, 2] = path[i, 2]
            return mutated_path

    # 2) fallback: 기존 평행/법선 국소 이동
    return _fallback_parallel_normal_mutation(mutated_path, i)
