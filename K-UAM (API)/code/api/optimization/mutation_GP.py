import numpy as np

def mutation_gp(path, nodes):
    """
    일정 확률(20%)로 경로의 중간 노드 하나를 다른 무작위 노드로 교체합니다.

    Args:
        path (np.ndarray): N x 3 형태의 원본 경로.
        nodes (np.ndarray): M x 3 형태의 교체될 수 있는 전체 경유지 노드 목록.

    Returns:
        np.ndarray: 돌연변이가 적용되었거나, 원본과 동일한 경로.
    """
    # 경로에 중간 노드가 없거나(크기가 2 이하), 교체할 노드 목록이 비어있으면 돌연변이 수행 불가
    has_intermediate_nodes = path.shape[0] > 2
    nodes_are_available = nodes.shape[0] > 0
    
    # 20%의 확률로 돌연변이 실행
    if np.random.rand() < 0.2 and has_intermediate_nodes and nodes_are_available:
        
        # 돌연변이를 적용할 중간 노드의 인덱스를 무작위로 선택
        # path[0]은 시작점, path[-1]은 끝점이므로, 1부터 len(path)-2 사이에서 선택
        # MATLAB의 randi([2, size-1])는 Python의 randint(1, size-1)과 동일한 범위
        idx_to_mutate = np.random.randint(1, path.shape[0] - 1)
        
        # 전체 노드 목록에서 새로운 노드를 무작위로 선택
        new_node_idx = np.random.randint(0, nodes.shape[0])
        new_node = nodes[new_node_idx, :]
        
        # 경로의 해당 위치에 새로운 노드를 덮어씀
        path[idx_to_mutate, :] = new_node

    # 수정되었거나 원본 그대로의 경로를 반환
    mutated_path = path
    
    return mutated_path