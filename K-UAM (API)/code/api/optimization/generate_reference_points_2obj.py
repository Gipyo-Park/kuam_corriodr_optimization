import numpy as np

def generate_reference_points_2obj(H):
    """
    2-Objective 최적화 문제에 대한 참조점(Reference Points)을 생성합니다.
    생성된 점들은 w1 + w2 = 1 선상에 균일하게 분포합니다.

    Args:
        H (int): 목적 함수 공간의 분할 수. 
                 (H+1)개의 참조점이 생성됩니다.

    Returns:
        np.ndarray: (H+1) x 2 크기의 참조점 배열.
    """
    # 0부터 1까지 H개의 구간으로 나눈 점들을 생성
    # 예: H=10 이면 [0.0, 0.1, 0.2, ..., 1.0] 배열이 생성됨
    w1 = np.linspace(0, 1, H + 1)
    
    # w2 = 1 - w1
    w2 = 1 - w1
    
    # w1과 w2를 2개의 열을 가진 배열로 합침
    # np.stack을 사용하여 두 1D 배열을 열(column) 방향으로 결합
    ref_points = np.stack((w1, w2), axis=1)
    
    return ref_points