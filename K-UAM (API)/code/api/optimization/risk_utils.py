import numpy as np
import pandas as pd

def normalize_tensor_with_shifting(tensor):
    """
    모든 값을 0 이상으로 이동시킨 후 Min-Max 정규화를 수행합니다.
    음수 값을 포함한 지상 위험도 데이터를 0과 1 사이로 변환하여
    공중 위험도와 공정하게 비교할 수 있도록 만듭니다.

    Args:
        tensor (np.ndarray): 원본 위험도 텐서 (음수 값 포함 가능).

    Returns:
        np.ndarray: 0과 1 사이로 정규화된 텐서.
    """
    # 1단계: 값 이동 (Shifting)
    # 텐서 전체에서 최솟값을 찾아 모든 요소에서 빼줍니다.
    # 이를 통해 가장 작은 값은 0이 됩니다.
    min_val = np.min(tensor)
    shifted_tensor = tensor - min_val

    # 2단계: Min-Max 정규화 (Min-Max Scaling)
    # 이동된 텐서의 최댓값으로 모든 요소를 나누어 0~1 사이로 만듭니다.
    max_val_of_shifted = np.max(shifted_tensor)

    # 분모가 0이 되는 엣지 케이스(모든 값이 동일한 경우)를 방지합니다.
    if max_val_of_shifted > 1e-9:
        normalized_tensor = shifted_tensor / max_val_of_shifted
    else:
        # 모든 값이 같다면 정규화 결과는 0으로 채워진 텐서가 됩니다.
        normalized_tensor = np.zeros_like(shifted_tensor)

    return normalized_tensor


def load_air_risk_tensor(air_risk_file, grc_ref_file, Ny, Nx, altitude_levels):
    """
    'AirRisk_data.xlsx' 파일을 읽어 [고도, Y, X] 차원의 3D NumPy 텐서로 변환합니다.
    GRC 파일과의 좌표 동기화를 위해 참조 GRC 파일의 그리드 인덱스(i, j)를 사용합니다.

    Args:
        air_risk_file (str): 공중 위험도 데이터 파일 경로.
        grc_ref_file (str): 그리드 인덱스(i, j) 참조를 위한 GRC 파일 경로.
        Ny (int): 전체 그리드의 Y축 크기.
        Nx (int): 전체 그리드의 X축 크기.
        altitude_levels (np.ndarray): 고도 레벨 목록 (e.g., [50, 100, ...]).

    Returns:
        np.ndarray: [고도, Y, X] 차원의 공중 위험도 텐서.
    """
    print("Loading air risk data...")
    try:
        # 엑셀 파일 로딩
        air_df = pd.read_excel(air_risk_file)
        grc_df = pd.read_excel(grc_ref_file)
    except FileNotFoundError as e:
        print(f"Error: Cannot find data file - {e}")
        return None

    # 고도 레벨 수
    A = len(altitude_levels)
    # 반환할 빈 텐서 생성
    AirRiskTensor = np.zeros((A, Ny, Nx))

    # 데이터프레임의 각 행을 순회하며 텐서 채우기
    # tqdm은 진행률 표시줄을 위한 라이브러리 (필요 시 'pip install tqdm'으로 설치)
    from tqdm import tqdm
    for idx in tqdm(range(len(grc_df)), desc="  - Building Air Risk Tensor"):
        # GRC 파일에서 해당 행의 그리드 인덱스(i, j)를 가져옴
        i, j = grc_df.loc[idx, 'i'], grc_df.loc[idx, 'j']

        # 각 고도 레벨에 대해
        for alt_idx, alt_val in enumerate(altitude_levels):
            # 엑셀의 컬럼 이름 (e.g., 'Risk_at_100m')
            col_name = f'Risk_at_{int(alt_val)}m'
            if col_name in air_df.columns:
                # 해당 위치(i, j)와 고도(alt_idx)에 위험도 값 할당
                risk_val = air_df.loc[idx, col_name]
                # 텐서의 인덱스는 0부터 시작하므로 j, i 그대로 사용
                AirRiskTensor[alt_idx, j, i] = risk_val

    print("Air risk tensor loaded successfully.")
    return AirRiskTensor