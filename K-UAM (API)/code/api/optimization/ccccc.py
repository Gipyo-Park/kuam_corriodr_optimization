import numpy as np
import pandas as pd
import os

def convert_4d_npy_to_excel(npy_filepath, excel_filepath):
    """
    (Y, X, Condition, Level) 형태의 4차원 .npy 파일을 엑셀 파일로 변환합니다.
    """
    if not os.path.exists(npy_filepath):
        print(f"오류: '{npy_filepath}' 파일을 찾을 수 없습니다.")
        return

    try:
        # 1. .npy 파일 불러오기
        data = np.load(npy_filepath)
        print(f"'{npy_filepath}' 파일을 성공적으로 불러왔습니다.")
        print(f"원본 데이터 모양 (Shape): {data.shape}")

        # 2. 데이터 차원 정보 추출
        Ny, Nx, num_conditions, num_levels = data.shape

        # 3. Y, X 좌표 인덱스 그리드 생성
        # 'ij' 인덱싱은 행렬 기반이므로 (Y, X) 순서와 일치합니다.
        y_indices = np.arange(Ny)
        x_indices = np.arange(Nx)
        yy, xx = np.meshgrid(y_indices, x_indices, indexing='ij')

        # 4. Pandas DataFrame으로 변환할 데이터 준비
        # 기본이 되는 Y, X 인덱스 열을 먼저 추가합니다.
        df_data = {
            'Y_Index': yy.flatten(),
            'X_Index': xx.flatten()
        }

        print("데이터를 2차원 테이블 형태로 재구성하는 중입니다...")
        # 5. 4가지 조건과 11가지 레벨에 대한 값을 열로 추가
        for c in range(num_conditions):
            for l in range(num_levels):
                # 'Condition_0_Level_0' 과 같은 형식으로 열 이름 생성
                column_name = f'Condition_{c}_Level_{l}'
                
                # 해당 조건과 레벨에 맞는 (Ny, Nx) 데이터 슬라이스를 1차원으로 펼침
                value_slice = data[:, :, c, l].flatten()
                
                # 딕셔너리에 새로운 열 추가
                df_data[column_name] = value_slice
        
        # 6. 딕셔너리를 이용해 최종 DataFrame 생성
        df = pd.DataFrame(df_data)
        print(f"데이터프레임이 생성되었습니다. (총 {df.shape[0]}개 행, {df.shape[1]}개 열)")

        # 7. 엑셀 파일로 저장
        print(f"'{excel_filepath}' 파일로 저장하는 중입니다...")
        df.to_excel(excel_filepath, index=False, engine='openpyxl')
        
        print(f"\n성공적으로 '{excel_filepath}' 파일을 생성했습니다.")

    except Exception as e:
        print(f"파일 변환 중 오류가 발생했습니다: {e}")

# --- 코드 실행 ---
input_npy_file = 'high_res_affected_population_GRC.npy'
output_excel_file = 'high_res_affected_population.xlsx'

convert_4d_npy_to_excel(input_npy_file, output_excel_file)