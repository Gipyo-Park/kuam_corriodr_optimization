import numpy as np

def fast_non_dominated_sort(f_vals):
    """
    목적 함수 값(f_vals)을 받아 비지배 정렬을 수행하고, 
    Front들의 집합을 반환합니다.

    Args:
        f_vals (np.ndarray): N x M 형태의 목적 함수 값 배열.
                             (N: 해의 개수, M: 목적 함수의 개수)

    Returns:
        list of list: 각 Front에 속한 해의 인덱스(0-based)를 담은 리스트.
                      (e.g., [[0, 2, 5], [1, 3], [4]])
    """
    N = f_vals.shape[0]
    
    # S[p]: p가 지배하는 해들의 인덱스 리스트
    S = [[] for _ in range(N)]
    # n[p]: p를 지배하는 해들의 개수
    n = np.zeros(N, dtype=int)
    
    # 각 Front를 담을 리스트 초기화
    Fronts = [[]]
    
    # 1단계: 각 해에 대한 지배 관계 및 n 값 계산
    for p in range(N):
        for q in range(N):
            # p가 q를 지배하는지 검사
            # 조건 1: p의 모든 목적 함수 값이 q보다 작거나 같다.
            # 조건 2: p의 목적 함수 값 중 적어도 하나는 q보다 작다.
            if np.all(f_vals[p] <= f_vals[q]) and np.any(f_vals[p] < f_vals[q]):
                S[p].append(q)
            # q가 p를 지배하는지 검사
            elif np.all(f_vals[q] <= f_vals[p]) and np.any(f_vals[q] < f_vals[p]):
                n[p] += 1
        
        # 아무에게도 지배당하지 않으면(n[p] == 0), Front 1에 추가
        if n[p] == 0:
            Fronts[0].append(p)
            
    # 2단계: 다음 Front들을 순차적으로 구성
    i = 0
    while len(Fronts[i]) > 0:
        # 다음 Front를 저장할 리스트 생성
        Q = []
        # 현재 Front(Fronts[i])에 있는 모든 p에 대해
        for p in Fronts[i]:
            # p가 지배하는 모든 q에 대해
            for q in S[p]:
                # q를 지배하던 해의 수를 1 감소
                n[q] -= 1
                # 만약 n[q]가 0이 되면, q는 다음 Front에 속하게 됨
                if n[q] == 0:
                    Q.append(q)
        
        i += 1
        # 다음 Front가 비어있지 않으면 Fronts 리스트에 추가
        if len(Q) > 0:
            Fronts.append(Q)
        else:
            # 더 이상 생성될 Front가 없으면 종료
            break
            
    return Fronts