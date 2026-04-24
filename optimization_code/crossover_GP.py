import numpy as np

def _setdiff_rows_stable(A, B):
    """
    MATLAB의 setdiff(A, B, 'rows', 'stable')과 동일하게 동작하는 도우미 함수.
    A 배열에서 B 배열에 포함된 행을 제외한 나머지를 순서대로 반환합니다.
    """
    # B의 각 행을 튜플로 변환하여 set에 저장 (빠른 조회를 위해)
    B_set = set(map(tuple, B))
    
    # A의 각 행을 순회하며 B_set에 없는 행만 리스트에 추가 (순서 유지)
    C_list = [row for row in A if tuple(row) not in B_set]
    
    # 결과가 비어있으면 빈 배열을, 아니면 numpy 배열로 변환하여 반환
    if not C_list:
        # A의 데이터 타입을 유지하면서 빈 배열 생성
        return np.empty((0, A.shape[1]), dtype=A.dtype)
    return np.array(C_list)


def crossover_gp(parent1, parent2):
    """
    두 부모 경로의 중간 노드를 조합하여 자식 경로를 생성합니다.
    
    Args:
        parent1 (np.ndarray): N x 3 형태의 첫 번째 부모 경로.
        parent2 (np.ndarray): M x 3 형태의 두 번째 부모 경로.
        
    Returns:
        np.ndarray: 생성된 자식 경로.
    """
    # 두 부모 경로에서 시작점, 끝점, 중간 노드를 분리
    start_point = parent1[0, :]
    end_point = parent1[-1, :]
    
    # MATLAB의 A(2:end-1, :)와 동일한 슬라이싱
    inter1 = parent1[1:-1, :]
    inter2 = parent2[1:-1, :]

    # 중간 노드가 없는 경우의 예외 처리
    if inter1.shape[0] == 0 or inter2.shape[0] == 0:
        if inter1.shape[0] == 0:
            return parent2
        else:
            return parent1

    # 자식의 중간 노드 개수는 항상 parent1을 따름
    len1 = inter1.shape[0]
    child_len = len1
    
    # parent1의 중간 노드에서 무작위로 두 지점을 선택하여 절단
    # (MATLAB의 randi는 1부터 시작, Python은 0부터 시작하므로 범위에 주의)
    cut1 = np.random.randint(0, len1)
    cut2 = np.random.randint(0, len1)
    
    if cut1 > cut2:
        cut1, cut2 = cut2, cut1 # 두 지점 정렬

    # parent1에서 잘라낸 유전자(경유지) 부분
    # (Python 슬라이싱은 끝점을 포함하지 않으므로 +1)
    segment = inter1[cut1:cut2 + 1, :]
    
    # parent2의 중간 노드 중 segment에 포함되지 않은 노드들을 순서대로 가져옴
    remaining_nodes = _setdiff_rows_stable(inter2, segment)
    
    # parent1의 segment와 parent2의 나머지 노드를 결합
    child_inter = np.vstack((segment, remaining_nodes))
    
    # 결합된 자식 경로의 길이를 child_len에 맞게 조정
    if child_inter.shape[0] > child_len:
        child_inter = child_inter[:child_len, :]
    elif child_inter.shape[0] < child_len:
        needed = child_len - child_inter.shape[0]
        # 부족한 노드는 child_inter에 없는 parent1의 노드로 채움
        additional_nodes = _setdiff_rows_stable(inter1, child_inter)
        
        # 추가할 노드가 부족할 수 있으므로, 실제 추가 가능한 만큼만 선택
        num_to_add = min(needed, additional_nodes.shape[0])
        if num_to_add > 0:
            child_inter = np.vstack((child_inter, additional_nodes[:num_to_add, :]))

    # 시작점, 최종 조합된 중간 노드, 끝점을 합쳐 완전한 자식 경로 생성
    # (start_point와 end_point를 2D 배열로 만들어 vstack)
    child = np.vstack((start_point.reshape(1, -1),
                       child_inter,
                       end_point.reshape(1, -1)))
    
    return child