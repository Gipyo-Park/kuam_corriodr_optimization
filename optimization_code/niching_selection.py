import numpy as np
from scipy.spatial.distance import cdist

def niching_selection(norm_f, ref_points, remaining, niche_count=None):
    """
    참조점과 정규화된 목적 함수 값들을 이용해 'remaining' 개수만큼 해를 선택합니다.
    (NSGA-III의 니칭(Niching) 메커니즘)

    Args:
        norm_f (np.ndarray): M x 2 형태의 정규화된 목적 함수 값 배열.
        ref_points (np.ndarray): K x 2 형태의 참조점 배열.
        remaining (int): 선택해야 할 해의 개수.
        niche_count (np.ndarray, optional): 외부에서 전달된 초기 niche_count. 
                                            None이면 0으로 시작.

    Returns:
        np.ndarray: 선택된 해의 인덱스(0-based)를 담은 1D 배열.
    """
    M = norm_f.shape[0]
    K = ref_points.shape[0]

    # 1. 각 해와 참조점 간의 유클리드 거리 계산
    distances = cdist(norm_f, ref_points)
    
    # 2. 각 해를 가장 가까운 참조점과 연결
    min_dists = np.min(distances, axis=1)
    assoc_rp = np.argmin(distances, axis=1)

    # 3. 각 참조점에 어떤 해들이 연결되었는지 그룹화
    RP_to_solutions = [[] for _ in range(K)]
    for i in range(M):
        RP_to_solutions[assoc_rp[i]].append(i)

    # 4. 각 참조점의 niche_count 초기화
    if niche_count is None:
        # --- 오류 수정 지점 ---
        # 데이터 타입을 int에서 float으로 변경하여 np.inf 값을 저장할 수 있도록 함
        niche_count = np.zeros(K, dtype=float) 
    
    selected_mask = np.zeros(M, dtype=bool)
    selected_idx = []

    # 5. 'remaining' 개수만큼 해를 선택할 때까지 반복
    while len(selected_idx) < remaining:
        available_rp_indices = np.where(niche_count < np.inf)[0]
        if len(available_rp_indices) == 0:
            break

        min_niche_rp_local_idx = np.argmin(niche_count[available_rp_indices])
        rp_idx = available_rp_indices[min_niche_rp_local_idx]

        candidates_original_indices = RP_to_solutions[rp_idx]
        
        candidates = [idx for idx in candidates_original_indices if not selected_mask[idx]]

        if not candidates:
            niche_count[rp_idx] = np.inf
            continue

        candidate_distances = min_dists[candidates]
        best_candidate_local_idx = np.argmin(candidate_distances)
        
        final_selected_idx = candidates[best_candidate_local_idx]
        
        selected_idx.append(final_selected_idx)
        selected_mask[final_selected_idx] = True
        
        niche_count[rp_idx] += 1

    return np.array(selected_idx, dtype=int)