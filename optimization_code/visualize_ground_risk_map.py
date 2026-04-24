import numpy as np
import matplotlib.pyplot as plt
import os
import math


CODE_LAT_LIM = [35.5446, 35.6427]
CODE_LON_LIM = [129.0514, 129.1436]

def visualize_grc_heatmap_subplots(
    file_path,
    show_plot=True,
    save_path=None,
    compare_heading_idx=0,
    compare_save_path=None,
):
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

        # 원본 축 기준으로 그대로 표시합니다.
        im = ax.imshow(scenario_slice_2d, cmap='jet', origin='lower')
        
        ax.set_title(f'Heading {current_heading:.0f} deg')
        ax.set_xlabel('X-axis Grid Index')
        ax.set_ylabel('Y-axis Grid Index')
    
    # 6. 남는 빈 서브플롯이 있다면 축을 보이지 않게 처리
    for j in range(num_scenarios, len(axes)):
        axes[j].axis('off')

    # 7. 전체 Figure에 대한 하나의 컬러바 추가
    if im: 
        fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label='Affected Population Level')
        
    fig.suptitle('Ground and Population Risk Map', fontsize=20)
    
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"시각화 이미지를 저장했습니다: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    # 9. 정렬 비교용 Figure: index / ROI-georef(no-rotation) / ROI-georef(transpose)
    hi = int(np.clip(compare_heading_idx, 0, num_scenarios - 1))
    layer = selected_scenario_data[:, :, hi]

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    axes2[0].imshow(layer, cmap='jet', origin='lower', interpolation='nearest')
    axes2[0].set_title(f'Ground index view | heading {hi}')
    axes2[0].set_xlabel('X index')
    axes2[0].set_ylabel('Y index')

    im1 = axes2[1].imshow(
        layer,
        cmap='jet',
        origin='lower',
        interpolation='nearest',
        extent=[CODE_LON_LIM[0], CODE_LON_LIM[1], CODE_LAT_LIM[0], CODE_LAT_LIM[1]],
        aspect='auto',
    )
    axes2[1].set_title('Ground main-eval georef (fills ROI)')
    axes2[1].set_xlabel('Longitude')
    axes2[1].set_ylabel('Latitude')

    axes2[2].imshow(
        layer.T,
        cmap='jet',
        origin='lower',
        interpolation='nearest',
        extent=[CODE_LON_LIM[0], CODE_LON_LIM[1], CODE_LAT_LIM[0], CODE_LAT_LIM[1]],
        aspect='auto',
    )
    axes2[2].set_title('Ground transpose georef (orientation check)')
    axes2[2].set_xlabel('Longitude')
    axes2[2].set_ylabel('Latitude')

    fig2.colorbar(im1, ax=axes2.tolist(), shrink=0.9, label='Affected Population Level')
    fig2.suptitle('Ground Risk Alignment Quick Compare', fontsize=16)

    if compare_save_path is None:
        compare_save_path = os.path.join('figure', 'ground_risk_alignment_compare.png')

    cmp_dir = os.path.dirname(compare_save_path)
    if cmp_dir:
        os.makedirs(cmp_dir, exist_ok=True)
    fig2.savefig(compare_save_path, dpi=150)
    print(f"비교 이미지를 저장했습니다: {compare_save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig2)

if __name__ == '__main__':
    npy_file = "Modified_high_res_affected_population_GRC.npy"
    visualize_grc_heatmap_subplots(
        npy_file,
        show_plot=True,
        save_path=os.path.join('figure', 'Modified_ground_risk_heatmaps.png'),
        compare_heading_idx=0,
        compare_save_path=os.path.join('figure', 'Modified_ground_risk_alignment_compare.png'),
    )