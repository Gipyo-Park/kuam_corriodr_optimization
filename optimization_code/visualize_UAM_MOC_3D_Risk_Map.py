import numpy as np
import matplotlib.pyplot as plt
import os
import math
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle


CODE_LAT_LIM = [35.5446, 35.6427]
CODE_LON_LIM = [129.0514, 129.1436]


def analyze_risk_data(risk_data_3d, z_vector):
    """
    위험도 데이터의 분포를 분석해 시각화 전략(이진/연속)을 결정합니다.
    """
    finite_values = risk_data_3d[np.isfinite(risk_data_3d)]
    unique_values = np.unique(finite_values)

    is_binary = (
        unique_values.size <= 2
        and np.all(np.isin(unique_values, [0, 1]))
    )

    print("\n[데이터 분석 결과]")
    print(f"- dtype: {risk_data_3d.dtype}")
    print(f"- 전체 최소/최대: {finite_values.min()} / {finite_values.max()}")
    print(f"- 고유값 개수: {unique_values.size}")
    print(f"- 고유값(최대 10개): {unique_values[:10]}")
    print(f"- 데이터 유형 판정: {'이진(0/1)' if is_binary else '연속값'}")

    if not is_binary:
        p1, p99 = np.percentile(finite_values, [1, 99])
    else:
        p1, p99 = 0, 1

    for i, altitude in enumerate(z_vector):
        slice_vals = risk_data_3d[:, :, i]
        zero_ratio = np.mean(slice_vals == 0)
        one_ratio = np.mean(slice_vals == 1)
        print(
            f"  고도 {altitude:>4}m | min={slice_vals.min()} max={slice_vals.max()} "
            f"zero={zero_ratio:.3f} one={one_ratio:.3f}"
        )

    return {
        'is_binary': is_binary,
        'vmin': p1,
        'vmax': p99,
    }


def _to_latlon_from_5179(x_2d, y_2d):
    try:
        import pyproj
    except Exception as e:
        raise RuntimeError("pyproj is required. Install with: pip install pyproj") from e

    tr = pyproj.Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)
    lon_2d, lat_2d = tr.transform(x_2d, y_2d)
    return np.asarray(lat_2d, dtype=float), np.asarray(lon_2d, dtype=float)


def _draw_roi_box(ax):
    rect = Rectangle(
        (CODE_LON_LIM[0], CODE_LAT_LIM[0]),
        CODE_LON_LIM[1] - CODE_LON_LIM[0],
        CODE_LAT_LIM[1] - CODE_LAT_LIM[0],
        fill=False,
        edgecolor="cyan",
        linewidth=2.0,
        linestyle="--",
        label="Main code lat/lon extent",
    )
    ax.add_patch(rect)


def plot_georef_alignment_compare(data, target_alt_m=600.0, show_plot=True, save_path=None):
    if "X_2d" not in data or "Y_2d" not in data:
        print("X_2d/Y_2d 키가 없어 georef 비교 Figure는 생략합니다.")
        return

    risk_3d = np.asarray(data["Risk_3d"], dtype=float)
    z_vector = np.asarray(data["z_vec"], dtype=float)
    x_2d = np.asarray(data["X_2d"], dtype=float)
    y_2d = np.asarray(data["Y_2d"], dtype=float)

    ai = int(np.argmin(np.abs(z_vector - float(target_alt_m))))
    alt_sel = float(z_vector[ai])
    layer = risk_3d[:, :, ai]
    lat_2d, lon_2d = _to_latlon_from_5179(x_2d, y_2d)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    ax = axes[0]
    pcm = ax.pcolormesh(lon_2d, lat_2d, layer, shading="auto", cmap="tab10", vmin=0, vmax=1)
    _draw_roi_box(ax)
    ax.set_title(f"MOC source georef - Full extent | alt={int(alt_sel)}m")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right")
    fig.colorbar(pcm, ax=ax, shrink=0.9, label="MOC binary")

    ax = axes[1]
    pcm = ax.pcolormesh(lon_2d, lat_2d, layer, shading="auto", cmap="tab10", vmin=0, vmax=1)
    _draw_roi_box(ax)
    ax.set_xlim(CODE_LON_LIM)
    ax.set_ylim(CODE_LAT_LIM)
    ax.set_title("MOC source georef - ROI only")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right")
    fig.colorbar(pcm, ax=ax, shrink=0.9, label="MOC binary")

    ax = axes[2]
    im = ax.imshow(
        layer,
        origin="lower",
        extent=[CODE_LON_LIM[0], CODE_LON_LIM[1], CODE_LAT_LIM[0], CODE_LAT_LIM[1]],
        interpolation="nearest",
        aspect="auto",
        cmap="tab10",
        vmin=0,
        vmax=1,
    )
    ax.set_title("MOC main-eval georef (fills ROI grid)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(im, ax=ax, shrink=0.9, label="MOC binary")

    fig.suptitle("MOC Risk Alignment Quick Compare", fontsize=16)

    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"시각화 이미지를 저장했습니다: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def visualize_npy_heatmap_subplots(file_path, show_plot=True, save_path=None):
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

    analysis = analyze_risk_data(risk_data_3d, z_vector)

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

        if analysis['is_binary']:
            # 이진 데이터는 2색상 고정 맵으로 표시해 흑백 혼동을 줄입니다.
            cmap = ListedColormap(['#1f77b4', '#ff7f0e'])
            norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
            im = ax.imshow(risk_slice_2d, cmap=cmap, norm=norm, origin='lower', interpolation='nearest')
        else:
            im = ax.imshow(
                risk_slice_2d,
                cmap='inferno',
                origin='lower',
                vmin=analysis['vmin'],
                vmax=analysis['vmax'],
            )
        
        ax.set_title(f'Altitude: {altitude}m')
        ax.set_xlabel('X-axis Grid Index')
        ax.set_ylabel('Y-axis Grid Index')
    
    # 6. 남는 빈 서브플롯이 있다면 축을 보이지 않게 처리
    for j in range(num_altitudes, len(axes)):
        axes[j].axis('off')

    # 7. 전체 Figure에 대한 하나의 컬러바 추가
    if im: 
        cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.8)
        if analysis['is_binary']:
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['0', '1'])
            cbar.set_label('Binary Risk Class')
        else:
            cbar.set_label('Risk Level (1~99 percentile scaled)')
        
    fig.suptitle('Combined Air Risk Heatmaps by Altitude', fontsize=20)
    
    # 8. 결과 저장 및 화면 표시
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"시각화 이미지를 저장했습니다: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    compare_out = os.path.join(os.path.dirname(save_path), "moc_risk_alignment_compare.png") if save_path else os.path.join("figure", "moc_risk_alignment_compare.png")
    plot_georef_alignment_compare(
        data,
        target_alt_m=600.0,
        show_plot=show_plot,
        save_path=compare_out,
    )

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    npy_file = os.path.join(base_dir, "air_risk_data", "UAM_MOC_3D_Risk_Map.npy")
    output_img = os.path.join(base_dir, "figure", "air_risk_heatmaps.png")

    visualize_npy_heatmap_subplots(npy_file, show_plot=True, save_path=output_img)