import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def transpose_ground_spatial_axes(arr: np.ndarray) -> np.ndarray:
    """Swap X/Y spatial axes (axis 0 <-> axis 1) for any ndarray with at least 2 dims."""
    if arr.ndim < 2:
        raise ValueError(f"Expected array with ndim >= 2, got ndim={arr.ndim}")

    axes = list(range(arr.ndim))
    axes[0], axes[1] = axes[1], axes[0]
    return np.transpose(arr, axes=axes)


def save_quick_compare_figure(
    original: np.ndarray,
    converted: np.ndarray,
    out_path: str,
    scenario_axis2: int = 0,
    heading_axis3: int = 3,
) -> None:
    """Create a quick visual check aligned with main pipeline slicing style.

    Main currently uses: selected = grc[:, :, 0, 3:]
    This helper visualizes one heading from original/converted for easy orientation check.
    """
    if original.ndim < 4 or converted.ndim < 4:
        print("[warn] Skip compare figure: expected 4D array shape like (Ny, Nx, A, H)")
        return

    old_layer = original[:, :, scenario_axis2, heading_axis3]
    new_layer = converted[:, :, scenario_axis2, heading_axis3]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    im0 = axes[0].imshow(old_layer, origin="lower", cmap="jet")
    axes[0].set_title("Original main-eval view")
    axes[0].set_xlabel("X index")
    axes[0].set_ylabel("Y index")

    axes[1].imshow(old_layer.T, origin="lower", cmap="jet")
    axes[1].set_title("Original transpose-check view")
    axes[1].set_xlabel("X index")
    axes[1].set_ylabel("Y index")

    axes[2].imshow(new_layer, origin="lower", cmap="jet")
    axes[2].set_title("Converted main-eval view")
    axes[2].set_xlabel("X index")
    axes[2].set_ylabel("Y index")

    fig.colorbar(im0, ax=axes.tolist(), shrink=0.9, label="Ground risk")
    fig.suptitle("Ground risk orientation conversion quick check", fontsize=16)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def convert_ground_risk_file(
    input_path: str,
    output_path: str,
    overwrite: bool = False,
    compare_figure_path: str | None = None,
) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    arr = np.load(input_path, allow_pickle=True)
    print(f"[info] Loaded: {input_path}")
    print(f"[info] Original shape: {arr.shape}, dtype: {arr.dtype}")

    converted = transpose_ground_spatial_axes(arr)
    print(f"[info] Converted shape: {converted.shape}, dtype: {converted.dtype}")

    if overwrite and os.path.abspath(input_path) == os.path.abspath(output_path):
        backup_path = f"{input_path}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak.npy"
        np.save(backup_path, arr)
        print(f"[info] Backup saved: {backup_path}")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(output_path, converted)
    print(f"[done] Converted file saved: {output_path}")

    if compare_figure_path:
        save_quick_compare_figure(arr, converted, compare_figure_path)
        print(f"[done] Compare figure saved: {compare_figure_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ground risk npy orientation by swapping spatial axes (0 <-> 1)."
    )
    parser.add_argument(
        "--input",
        default="high_res_affected_population_GRC.npy",
        help="Input npy path",
    )
    parser.add_argument(
        "--output",
        default="high_res_affected_population_GRC_transposed.npy",
        help="Output npy path",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite input path (creates timestamped backup first)",
    )
    parser.add_argument(
        "--compare-fig",
        default=os.path.join("figure", "ground_risk_conversion_compare.png"),
        help="Optional output path for quick compare figure",
    )

    args = parser.parse_args()

    output_path = args.input if args.inplace else args.output
    convert_ground_risk_file(
        input_path=args.input,
        output_path=output_path,
        overwrite=args.inplace,
        compare_figure_path=args.compare_fig,
    )
