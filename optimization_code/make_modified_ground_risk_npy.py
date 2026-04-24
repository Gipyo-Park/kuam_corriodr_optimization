import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

# 사용 목적:
# - Ground 위험도 원본의 공간축 방향이 main 평가 기준과 다를 때,
#   공간축(0,1)을 교환해 "Modified_high_res_affected_population_GRC.npy"를 생성합니다.
# - main_JS_1218_v9_1.py는 Ground를 별도로 회전/전치하지 않고 입력 파일을 그대로 사용하므로,
#   방향 보정은 이 스크립트에서 사전에 수행해야 합니다.
# - Bird/MOC는 main 내부에서 (Ny,Nx,Nz) 축 정렬 보정을 수행하지만,
#   Ground는 본 변환 파일을 사용하는 운영 규칙으로 맞춥니다.


def convert_ground_risk_orientation(src: np.ndarray) -> np.ndarray:
    """Main pipeline 기준으로 원하는 방향이 되도록 공간축(0,1)을 교환한다."""
    if src.ndim < 2:
        raise ValueError(f"Expected ndim >= 2, got {src.ndim}")
    return np.swapaxes(src, 0, 1)


def save_compare_figure(original: np.ndarray, converted: np.ndarray, out_path: str) -> None:
    """원본/원본전치/변환후를 한 화면에서 확인한다."""
    if original.ndim < 4 or converted.ndim < 4:
        print("[warn] compare figure skipped: expected 4D array like (Ny, Nx, A, H)")
        return

    layer_old = original[:, :, 0, 3]
    layer_new = converted[:, :, 0, 3]

    # main에서 자주 쓰는 슬라이스(selected = [:,:,0,3:]) 기준으로 비교합니다.
    # 즉, 오른쪽 패널(Modified main view)이 원하는 기준과 맞으면 변환이 정상입니다.

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    im = axes[0].imshow(layer_old, origin="lower", cmap="jet")
    axes[0].set_title("Original main view")
    axes[0].set_xlabel("X index")
    axes[0].set_ylabel("Y index")

    axes[1].imshow(layer_old.T, origin="lower", cmap="jet")
    axes[1].set_title("Original transpose target")
    axes[1].set_xlabel("X index")
    axes[1].set_ylabel("Y index")

    axes[2].imshow(layer_new, origin="lower", cmap="jet")
    axes[2].set_title("Modified main view")
    axes[2].set_xlabel("X index")
    axes[2].set_ylabel("Y index")

    fig.colorbar(im, ax=axes.tolist(), shrink=0.9, label="Ground risk")
    fig.suptitle("Ground risk orientation conversion check", fontsize=16)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(input_path: str, output_path: str, compare_fig_path: str | None) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    arr = np.load(input_path, allow_pickle=True)
    print(f"[info] input: {input_path}")
    print(f"[info] original shape: {arr.shape}, dtype: {arr.dtype}")

    converted = convert_ground_risk_orientation(arr)
    print(f"[info] converted shape: {converted.shape}, dtype: {converted.dtype}")

    if arr.ndim >= 4:
        check = np.max(np.abs(converted[:, :, 0, 3] - arr[:, :, 0, 3].T))
        print(f"[check] max|converted[:,:,0,3] - original[:,:,0,3].T| = {check:.6g}")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(output_path, converted)
    print(f"[done] saved: {output_path}")

    if compare_fig_path:
        save_compare_figure(arr, converted, compare_fig_path)
        print(f"[done] compare figure: {compare_fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create modified ground risk npy by swapping spatial axes (0<->1)."
    )
    parser.add_argument(
        "--input",
        default="high_res_affected_population_GRC.npy",
        help="Original ground risk npy path",
    )
    parser.add_argument(
        "--output",
        default="Modified_high_res_affected_population_GRC.npy",
        help="Modified output npy path",
    )
    parser.add_argument(
        "--compare-fig",
        default=os.path.join("figure", "Modified_high_res_affected_population_GRC.png"),
        help="Optional quick comparison figure path",
    )

    args = parser.parse_args()
    main(args.input, args.output, args.compare_fig)
