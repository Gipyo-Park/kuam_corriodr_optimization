import numpy as np

def generate_initial_population_gp(pop_size, nodes, start_point, end_point, min_inter_nodes, max_inter_nodes):
    """
    주어진 노드들을 이용해 유전 알고리즘의 초기 경로 집단을 생성합니다.

    Args:
        pop_size (int): 생성할 개체(경로)의 수.
        nodes (np.ndarray): M x 3 형태의 경유지 후보 노드 배열.
        start_point (np.ndarray): 1 x 3 형태의 시작점 좌표.
        end_point (np.ndarray): 1 x 3 형태의 끝점 좌표.
        min_inter_nodes (int): 경로에 포함될 중간 노드의 최소 개수.
        max_inter_nodes (int): 경로에 포함될 중간 노드의 최대 개수.

    Returns:
        list: 각 요소가 하나의 경로(np.ndarray)인 리스트.
    """
    M = nodes.shape[0]
    
    # 경로에 포함될 중간 노드의 최대/최소 개수 설정
    # (사용 가능한 노드 수를 고려하여 실제 적용될 값을 계산)
    actual_max_inter_nodes = min(max_inter_nodes, M)
    actual_min_inter_nodes = min(min_inter_nodes, actual_max_inter_nodes)
    
    population = []  # MATLAB의 cell array 대신 Python 리스트 사용

    # 사용 가능한 노드가 0개이면 빈 리스트 반환
    if M == 0:
        return population

    for _ in range(pop_size):
        # 경유 노드 개수를 무작위로 선택
        if actual_min_inter_nodes >= actual_max_inter_nodes:
            num_inter = actual_max_inter_nodes
        else:
            # np.random.randint는 상한을 포함하지 않으므로 +1
            num_inter = np.random.randint(actual_min_inter_nodes, actual_max_inter_nodes + 1)
        
        # M개의 노드 중에서 num_inter개를 중복 없이 무작위로 선택 (인덱스)
        # MATLAB의 randperm(M, num_inter)와 동일
        selected_idx = np.random.choice(M, size=num_inter, replace=False)
        
        # 선택된 노드들의 순서를 다시 무작위로 섞음
        # MATLAB의 selected_idx(randperm(num_inter))와 동일
        ordered_idx = np.random.permutation(selected_idx)
        
        # 시작점, 선택된 중간 노드들, 끝점을 순서대로 합쳐 하나의 완전한 경로 생성
        # start_point와 end_point를 vstack을 위해 2D 배열로 만듦
        path = np.vstack((
            start_point.reshape(1, -1),
            nodes[ordered_idx, :],
            end_point.reshape(1, -1)
        ))
        
        population.append(path)
        
    return population