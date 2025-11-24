import numpy as np

def normalize_objectives(f_vals):
    """
    목적 함수 값들을 0과 1 사이로 정규화(Min-Max Scaling)합니다.

    Args:
        f_vals (np.ndarray): N x M 형태의 원본 목적 함수 값 배열.

    Returns:
        np.ndarray: 정규화된 목적 함수 값 배열.
    """
    # 각 열(목적 함수)의 최솟값과 최댓값을 찾음
    # MATLAB의 min(..., [], 1)은 numpy의 min(..., axis=0)과 동일
    f_min = np.min(f_vals, axis=0)
    f_max = np.max(f_vals, axis=0)
    
    # 각 목적 함수의 값 범위를 계산
    # (참고: range는 Python의 내장 함수이므로 변수명으로 range_를 사용)
    range_ = f_max - f_min
    
    # 분모가 0이 되는 것을 방지하기 위한 안정성 처리
    # range_의 값이 1e-10보다 작은 경우, 해당 값을 1로 설정
    range_[range_ < 1e-10] = 1
    
    # Min-Max 정규화 공식 적용
    # Numpy의 브로드캐스팅 기능으로 모든 행에 대해 연산이 수행됨
    normalized_f = (f_vals - f_min) / range_
    
    return normalized_f