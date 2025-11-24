import numpy as np
import matplotlib.pyplot as plt
import os
import math

def visualize_grc_heatmap_subplots(file_path):
    """
    GRC .npy 파일에 담긴 다차원 데이터를 시나리오별 2D 히트맵으로 시각화합니다.
    """
    # 1. 파일 존재 여부 확인
    if not os.path.exists(file_path):
        print(f"오류: 파일이 존재하지 않습니다 - {file_path}")
        return

    print(f"'{file_path}' 파일을 불러오는 중입니다...")
    try:
        # 2. NPY 데이터 로딩
        pop_risk_raw = np.load(file_path, allow_pickle=True)
        print("파일 로딩 완료.")

        # 3. 데이터 슬라이싱 및 구조 확인
        selected_scenario_data = pop_risk_raw[:, :, 0, 3:]
        
        if selected_scenario_data.ndim != 3:
            print(f"오류: 슬라이싱된 데이터가 3차원이 아닙니다 (현재: {selected_scenario_data.ndim}차원).")
            return

        Ny, Nx, num_scenarios = selected_scenario_data.shape
        print(f"데이터 격자 크기 (Ny, Nx): ({Ny}, {Nx})")
        print(f"시나리오(Heading) 개수: {num_scenarios}")

    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    # 4. 서브플롯 그리드 설정 및 Figure 생성
    rows = 2 
    cols = math.ceil(num_scenarios / rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 6),
                           squeeze=False, constrained_layout=True)
    axes = axes.flatten()

    # 5. 각 시나리오별로 히트맵을 해당하는 서브플롯에 생성
    im = None
    degrees_per_scenario = 360 / num_scenarios

    for i in range(num_scenarios):
        ax = axes[i] 
        current_heading = i * degrees_per_scenario
        print(f"Heading {current_heading:.0f} deg 히트맵을 생성 중... ({i+1}/{num_scenarios})")
        
        scenario_slice_2d = selected_scenario_data[:, :, i]

        # <<< [수정 1] 데이터를 반시계 방향으로 90도 회전시켜 표본과 모양을 맞춥니다.
        rotated_slice_2d = np.rot90(scenario_slice_2d, k=1)

        # <<< [수정 2] 컬러맵을 'jet'으로 변경하고, 회전된 데이터를 사용합니다.
        im = ax.imshow(rotated_slice_2d, cmap='jet', origin='lower')
        
        ax.set_title(f'Heading {current_heading:.0f} deg')
        # <<< [수정 3] 회전된 축에 맞게 라벨을 올바르게 설정합니다.
        ax.set_xlabel('Original Y-axis Grid Index')
        ax.set_ylabel('Original X-axis Grid Index')
    
    # 6. 남는 빈 서브플롯이 있다면 축을 보이지 않게 처리
    for j in range(num_scenarios, len(axes)):
        axes[j].axis('off')

    # 7. 전체 Figure에 대한 하나의 컬러바 추가
    if im: 
        fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label='Affected Population Level')
        
    fig.suptitle('Ground and Population Risk Map', fontsize=20)
    
    # 8. 최종 결과 화면에 표시
    plt.show()

if __name__ == '__main__':
    npy_file = "high_res_affected_population_GRC.npy"
    visualize_grc_heatmap_subplots(npy_file)