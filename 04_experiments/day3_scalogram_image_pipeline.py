# -*- coding: utf-8 -*-
"""Day 3: CWT 系数到 CNN 单通道图像的底层流水线函数。

本脚本只使用 Day 1 已生成的模拟轴承冲击信号，不读取 CWRU 数据，也不依赖
PyTorch。正式模型输入使用无坐标轴、无标题、无颜色条的 8 位灰度 PNG；
Matplotlib 生成的伪彩色图仅用于学习和检查。本文件作为 Day 3 V1.1
教学实验复用的函数模块保留；旧版独立输出已移入项目外归档。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pywt
from PIL import Image


EXPERIMENT_DIR = Path(__file__).resolve().parent
INPUT_SIGNAL_PATH = EXPERIMENT_DIR / "signal2_bearing_impact_v2.npy"

FS = 10_000
SCALES = np.arange(1, 129)
WAVELET = "morl"
OUTPUT_96_PATH = EXPERIMENT_DIR / "day3_scalogram_96x96.png"
OUTPUT_32_PATH = EXPERIMENT_DIR / "day3_scalogram_32x32.png"
OVERVIEW_PATH = EXPERIMENT_DIR / "day3_pipeline_overview.png"
NORMALIZATION_PATH = EXPERIMENT_DIR / "day3_normalization_comparison.png"
RESOLUTION_PATH = EXPERIMENT_DIR / "day3_resolution_comparison.png"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def validate_signal(signal: np.ndarray, fs: float) -> np.ndarray:
    """校验并返回 float64 一维信号。"""
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"signal 必须是一维数组，实际 shape={array.shape}")
    if array.size == 0:
        raise ValueError("signal 不能为空")
    if not np.all(np.isfinite(array)):
        raise ValueError("signal 不能包含 NaN 或 Inf")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"fs 必须是正数，实际 fs={fs}")
    if np.ptp(array) <= np.finfo(np.float64).eps:
        raise ValueError("signal 不能是常量信号，否则无法进行 min-max 归一化")
    return array


def validate_scales(scales: Iterable[float]) -> np.ndarray:
    """校验 CWT scale，并返回 float64 一维数组。"""
    array = np.asarray(scales, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("scales 必须是非空一维数组")
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError("scales 必须全部为有限正数")
    return array


def compute_cwt_magnitude(
    signal: np.ndarray,
    fs: float,
    scales: Iterable[float] = SCALES,
    wavelet: str = WAVELET,
) -> tuple[np.ndarray, np.ndarray]:
    """计算 CWT，并返回系数绝对值矩阵和伪频率。"""
    checked_signal = validate_signal(signal, fs)
    checked_scales = validate_scales(scales)
    coefficients, frequencies = pywt.cwt(
        checked_signal,
        checked_scales,
        wavelet,
        sampling_period=1.0 / fs,
    )
    magnitude = np.abs(coefficients).astype(np.float64, copy=False)
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("CWT 结果包含 NaN 或 Inf")
    return magnitude, np.asarray(frequencies, dtype=np.float64)


def normalize_matrix(matrix: np.ndarray, method: str) -> np.ndarray:
    """返回原值、min-max 或 z-score 归一化结果。"""
    array = np.asarray(matrix, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("待归一化矩阵必须非空且全部为有限值")

    if method == "none":
        return array.copy()
    if method == "minmax":
        minimum = float(array.min())
        value_range = float(array.max() - minimum)
        if value_range <= np.finfo(np.float64).eps:
            raise ValueError("矩阵极差为 0，无法进行 min-max 归一化")
        return (array - minimum) / value_range
    if method == "zscore":
        mean = float(array.mean())
        std = float(array.std())
        if std <= np.finfo(np.float64).eps:
            raise ValueError("矩阵标准差为 0，无法进行 z-score 标准化")
        return (array - mean) / std
    raise ValueError(f"未知归一化方法: {method!r}")


def resize_unit_matrix(matrix: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """使用双线性插值把 [0, 1] 浮点矩阵缩放到 (height, width)。"""
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"matrix 必须是二维数组，实际 shape={array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix 不能包含 NaN 或 Inf")
    if float(array.min()) < 0.0 or float(array.max()) > 1.0:
        raise ValueError("resize_unit_matrix 的输入必须位于 [0, 1]")
    if len(output_size) != 2:
        raise ValueError("output_size 必须是 (height, width)")
    height, width = output_size
    if not isinstance(height, int) or not isinstance(width, int) or height <= 0 or width <= 0:
        raise ValueError("output_size 的高度和宽度必须是正整数")

    source = Image.fromarray(array, mode="F")
    resized = source.resize((width, height), resample=Image.Resampling.BILINEAR)
    result = np.asarray(resized, dtype=np.float32)
    return np.clip(result, 0.0, 1.0)


def unit_matrix_to_uint8(matrix: np.ndarray) -> np.ndarray:
    """把 [0, 1] 浮点强度转换为 8 位灰度。"""
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError("matrix 必须是有限的二维数组")
    if float(array.min()) < 0.0 or float(array.max()) > 1.0:
        raise ValueError("unit_matrix_to_uint8 的输入必须位于 [0, 1]")
    return np.rint(array * 255.0).astype(np.uint8)


def build_scalogram(
    signal: np.ndarray,
    fs: float,
    scales: Iterable[float] = SCALES,
    wavelet: str = WAVELET,
    output_size: tuple[int, int] = (96, 96),
    normalization: str = "minmax",
) -> np.ndarray:
    """把一维信号转换为供 CNN 使用的二维 uint8 scalogram。"""
    if normalization != "minmax":
        raise ValueError("当前正式流水线固定使用 normalization='minmax'")
    magnitude, _ = compute_cwt_magnitude(signal, fs, scales, wavelet)
    normalized = normalize_matrix(magnitude, method=normalization)
    resized = resize_unit_matrix(normalized, output_size=output_size)
    return unit_matrix_to_uint8(resized)


def save_grayscale_png(array: np.ndarray, path: Path) -> None:
    """直接保存 CNN 输入，不加入任何绘图装饰。"""
    image_array = np.asarray(array)
    if image_array.ndim != 2 or image_array.dtype != np.uint8:
        raise ValueError("灰度 PNG 输入必须是二维 uint8 数组")
    Image.fromarray(image_array, mode="L").save(path, format="PNG")


def add_scalogram_axes(
    ax: plt.Axes,
    matrix: np.ndarray,
    duration: float,
    title: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """绘制仅供人阅读的 scalogram。"""
    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        cmap=cmap,
        extent=[0.0, duration, SCALES[-1], SCALES[0]],
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("Scale（小 → 高频）")
    ax.set_title(title)
    ax.set_ylim(SCALES[-1], SCALES[0])
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def create_pipeline_overview(
    signal: np.ndarray,
    magnitude: np.ndarray,
    normalized: np.ndarray,
    image_96: np.ndarray,
    path: Path,
) -> None:
    """生成从一维信号到正式模型图像的四步总览。"""
    duration = signal.size / FS
    time = np.arange(signal.size) / FS
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(time, signal, color="#2455A4", linewidth=0.8)
    axes[0, 0].set_xlim(0.0, duration)
    axes[0, 0].set_xlabel("时间 (s)")
    axes[0, 0].set_ylabel("幅值")
    axes[0, 0].set_title(f"步骤 1：模拟轴承冲击信号（{FS} Hz，{signal.size} 点）")
    axes[0, 0].grid(alpha=0.25)

    add_scalogram_axes(
        axes[0, 1],
        magnitude,
        duration,
        f"步骤 2：CWT 系数绝对值，shape={magnitude.shape}",
    )
    add_scalogram_axes(
        axes[1, 0],
        normalized,
        duration,
        "步骤 3：逐样本 min-max，范围 [0, 1]",
        vmin=0.0,
        vmax=1.0,
    )

    axes[1, 1].imshow(image_96, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[1, 1].set_title("步骤 4：双线性缩放后的 96×96 单通道输入")
    axes[1, 1].axis("off")

    figure.suptitle("Day 3：CWT → CNN 图像完整流水线", fontsize=16)
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def create_normalization_comparison(magnitude: np.ndarray, path: Path) -> None:
    """对比原始幅值、min-max 和 z-score 的图像及直方图。"""
    raw = normalize_matrix(magnitude, "none")
    minmax = normalize_matrix(magnitude, "minmax")
    zscore = normalize_matrix(magnitude, "zscore")
    matrices = [raw, minmax, zscore]
    names = [
        f"原始幅值\n[{raw.min():.4g}, {raw.max():.4g}]",
        f"Min-max\n[{minmax.min():.1f}, {minmax.max():.1f}]",
        f"Z-score\n均值={zscore.mean():.2e}, 标准差={zscore.std():.2f}",
    ]

    figure, axes = plt.subplots(2, 3, figsize=(16, 8))
    for index, (matrix, name) in enumerate(zip(matrices, names)):
        axes[0, index].imshow(matrix, aspect="auto", origin="upper", cmap="viridis")
        axes[0, index].set_title(name)
        axes[0, index].set_xlabel("时间索引")
        axes[0, index].set_ylabel("Scale 索引")

        axes[1, index].hist(matrix.ravel(), bins=80, color="#3B7EA1", alpha=0.9)
        axes[1, index].set_yscale("log")
        axes[1, index].set_xlabel("数值")
        axes[1, index].set_ylabel("像素数（对数）")
        axes[1, index].grid(alpha=0.2)

    figure.suptitle("归一化改变数值尺度，不改变 CWT 的空间位置", fontsize=16)
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def resize_back_to_source(image: np.ndarray, source_shape: tuple[int, int]) -> np.ndarray:
    """把 uint8 小图放回原矩阵尺寸，仅用于量化信息损失。"""
    height, width = source_shape
    source = Image.fromarray(image, mode="L")
    restored = source.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(restored, dtype=np.float32) / 255.0


def create_resolution_comparison(
    normalized: np.ndarray,
    image_96: np.ndarray,
    image_32: np.ndarray,
    path: Path,
) -> tuple[float, float]:
    """生成全图与局部放大对比，并返回 96/32 的重建 MAE。"""
    restored_96 = resize_back_to_source(image_96, normalized.shape)
    restored_32 = resize_back_to_source(image_32, normalized.shape)
    mae_96 = float(np.mean(np.abs(normalized - restored_96)))
    mae_32 = float(np.mean(np.abs(normalized - restored_32)))

    matrices = [normalized, image_96 / 255.0, image_32 / 255.0]
    titles = [
        f"原始归一化矩阵\n{normalized.shape[0]}×{normalized.shape[1]}",
        f"96×96（双线性）\n回放 MAE={mae_96:.5f}",
        f"32×32（双线性）\n回放 MAE={mae_32:.5f}",
    ]
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    for index, (matrix, title) in enumerate(zip(matrices, titles)):
        axes[0, index].imshow(
            matrix,
            aspect="auto" if index == 0 else "equal",
            origin="upper",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axes[0, index].set_title(title)
        axes[0, index].set_xlabel("时间方向")
        axes[0, index].set_ylabel("Scale 方向")

    # 取中部约 30% 区域，统一放大比较条纹和衰减细节。
    source_height, source_width = normalized.shape
    crops = [
        normalized[int(0.15 * source_height) : int(0.65 * source_height), int(0.35 * source_width) : int(0.65 * source_width)],
        image_96[14:62, 34:63] / 255.0,
        image_32[5:21, 11:21] / 255.0,
    ]
    for index, crop in enumerate(crops):
        axes[1, index].imshow(
            crop,
            aspect="auto",
            origin="upper",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axes[1, index].set_title("局部像素放大")
        axes[1, index].axis("off")

    figure.suptitle("图像分辨率越低，周期性冲击的细节损失越明显", fontsize=16)
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return mae_96, mae_32


def file_sha256(path: Path) -> str:
    """计算文件摘要，用于确定性检查。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_saved_image(path: Path, expected_size: tuple[int, int], expected: np.ndarray) -> None:
    """重新读取正式 PNG，检查模式、尺寸和像素。"""
    with Image.open(path) as image:
        if image.mode != "L":
            raise AssertionError(f"{path.name} 应为 L 模式，实际为 {image.mode}")
        expected_height, expected_width = expected_size
        if image.size != (expected_width, expected_height):
            raise AssertionError(
                f"{path.name} 尺寸错误，期望 {(expected_width, expected_height)}，实际 {image.size}"
            )
        reloaded = np.asarray(image)
    if not np.array_equal(reloaded, expected):
        raise AssertionError(f"{path.name} 保存前后像素不一致")


def verify_error_paths() -> None:
    """验证关键失败场景会给出明确错误。"""
    invalid_signals = [
        np.array([], dtype=float),
        np.ones(16, dtype=float),
        np.array([0.0, np.nan, 1.0]),
    ]
    for invalid in invalid_signals:
        try:
            build_scalogram(invalid, FS)
        except ValueError:
            continue
        raise AssertionError("无效信号未触发 ValueError")


def main() -> None:
    print("=" * 72)
    print("Day 3: CWT 图像生成完整闭环")
    print("=" * 72)
    if not INPUT_SIGNAL_PATH.exists():
        raise FileNotFoundError(f"未找到输入信号: {INPUT_SIGNAL_PATH}")

    signal = np.load(INPUT_SIGNAL_PATH)
    signal = validate_signal(signal, FS)
    magnitude, frequencies = compute_cwt_magnitude(signal, FS, SCALES, WAVELET)
    normalized = normalize_matrix(magnitude, "minmax")

    assert signal.shape == (2000,), signal.shape
    assert magnitude.shape == (128, 2000), magnitude.shape
    assert normalized.shape == magnitude.shape
    assert np.isclose(normalized.min(), 0.0)
    assert np.isclose(normalized.max(), 1.0)
    assert np.all(np.isfinite(normalized))

    image_96 = build_scalogram(signal, FS, output_size=(96, 96))
    image_32 = build_scalogram(signal, FS, output_size=(32, 32))
    repeat_96 = build_scalogram(signal, FS, output_size=(96, 96))
    assert image_96.shape == (96, 96) and image_96.dtype == np.uint8
    assert image_32.shape == (32, 32) and image_32.dtype == np.uint8
    assert int(image_96.min()) >= 0 and int(image_96.max()) <= 255
    assert int(image_32.min()) >= 0 and int(image_32.max()) <= 255
    assert np.array_equal(image_96, repeat_96), "同一输入的两次输出不一致"

    save_grayscale_png(image_96, OUTPUT_96_PATH)
    save_grayscale_png(image_32, OUTPUT_32_PATH)
    verify_saved_image(OUTPUT_96_PATH, (96, 96), image_96)
    verify_saved_image(OUTPUT_32_PATH, (32, 32), image_32)

    create_pipeline_overview(signal, magnitude, normalized, image_96, OVERVIEW_PATH)
    create_normalization_comparison(magnitude, NORMALIZATION_PATH)
    mae_96, mae_32 = create_resolution_comparison(
        normalized, image_96, image_32, RESOLUTION_PATH
    )
    verify_error_paths()

    print(f"输入信号: {INPUT_SIGNAL_PATH.name}")
    print(f"  shape={signal.shape}, fs={FS} Hz, duration={signal.size / FS:.3f} s")
    print(f"CWT: wavelet={WAVELET}, scales={int(SCALES[0])}~{int(SCALES[-1])}")
    print(f"  magnitude.shape={magnitude.shape}")
    print(f"  pseudo-frequency={frequencies[-1]:.2f}~{frequencies[0]:.2f} Hz")
    print(f"min-max: min={normalized.min():.1f}, max={normalized.max():.1f}")
    print(f"96×96: range=[{image_96.min()}, {image_96.max()}], MAE={mae_96:.6f}")
    print(f"32×32: range=[{image_32.min()}, {image_32.max()}], MAE={mae_32:.6f}")
    print(f"96×96 SHA256: {file_sha256(OUTPUT_96_PATH)}")
    print("\n生成文件:")
    for output_path in [
        OUTPUT_96_PATH,
        OUTPUT_32_PATH,
        OVERVIEW_PATH,
        NORMALIZATION_PATH,
        RESOLUTION_PATH,
    ]:
        print(f"  - {output_path.name}")
    print("\n全部自动验证通过。")


if __name__ == "__main__":
    main()
