import numpy as np
import matplotlib.pyplot as plt
import os
import math

def visualize_npy_heatmap_subplots(file_path):
    """
    .npy 파일에 담긴 3D 위험도 데이터를 하나의 Figure에 고도별 2D 히트맵 서브플롯으로 시각화합니다.
    """
    # 1. 파일 존재 여부 확인
    if not os.path.exists(file_path):
        print(f"오류: 파일이 존재하지 않습니다 - {file_path}")
        return

    print(f"'{file_path}' 파일을 불러오는 중입니다...")
    try:
        # 2. NPY 데이터 로딩
        data = np.load(file_path, allow_pickle=True).item()
        print("파일 로딩 완료.")

        # 3. 데이터 구조 확인 및 추출
        print(f"데이터에 포함된 키: {list(data.keys())}")

        if 'Risk_3d' not in data or 'z_vec' not in data:
            print("오류: .npy 파일에 'Risk_3d' 또는 'z_vec' 키가 없습니다.")
            return
            
        risk_data_3d = data['Risk_3d']
        z_vector = data['z_vec']
        
        if risk_data_3d.ndim != 3:
            print(f"오류: 3차원 데이터가 필요하지만, 현재 데이터는 {risk_data_3d.ndim}차원입니다.")
            return

        print(f"데이터 차원 (Nx, Ny, Nz): {risk_data_3d.shape}")
        print(f"고도 레벨 (z_vec): {z_vector}")

    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    # 4. 서브플롯 그리드 설정 및 Figure 생성
    num_altitudes = len(z_vector)
    
    # <<< [수정] 행 개수를 2로 고정하고, 열 개수는 이에 맞춰 자동 계산
    rows = 2 
    cols = math.ceil(num_altitudes / rows)
    
    # 모든 서브플롯을 담을 하나의 Figure를 생성합니다.
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), 
                           squeeze=False, constrained_layout=True)
    axes = axes.flatten()

    # 5. 각 고도별로 히트맵을 해당하는 서브플롯에 생성
    im = None
    for i in range(num_altitudes):
        altitude = z_vector[i]
        ax = axes[i] 
        
        print(f"{altitude}m 고도의 히트맵을 생성 중... ({i+1}/{num_altitudes})")
        
        risk_slice_2d = risk_data_3d[:, :, i].T

        im = ax.imshow(risk_slice_2d, cmap='hot', origin='lower')
        
        ax.set_title(f'Altitude: {altitude}m')
        ax.set_xlabel('X-axis Grid Index')
        ax.set_ylabel('Y-axis Grid Index')
    
    # 6. 남는 빈 서브플롯이 있다면 축을 보이지 않게 처리
    for j in range(num_altitudes, len(axes)):
        axes[j].axis('off')

    # 7. 전체 Figure에 대한 하나의 컬러바 추가
    if im: 
        fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label='Risk Level')
        
    fig.suptitle('Combined Air Risk Heatmaps by Altitude', fontsize=20)
    
    # 8. 최종 결과 화면에 표시
    plt.show()

if __name__ == '__main__':
    npy_file = "AirRisk_combined_max_risk_map.npy"
    visualize_npy_heatmap_subplots(npy_file)