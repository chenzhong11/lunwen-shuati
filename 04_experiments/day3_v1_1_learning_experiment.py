# -*- coding: utf-8 -*-
"""Day 3 V1.1：按 A/B/C 三个单问题实验重做学习路径。

Day 3A：2000 与 2048 点到底改变什么，并生成不重叠的教学冲击信号。
Day 3B：CWT 系数怎样变成 [0, 1] 强度矩阵。
Day 3C：强度矩阵怎样变成无绘图装饰的 96×96 灰度 PNG。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from day3_scalogram_image_pipeline import (
    SCALES,
    WAVELET,
    build_scalogram,
    compute_cwt_magnitude,
    normalize_matrix,
    save_grayscale_png,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent

FS = 10_000
DURATION_2000 = 0.2
N_2000 = 2_000
N_2048 = 2_048
FAULT_FREQUENCY = 100
NATURAL_FREQUENCY = 2_000
DAMPING = 1_000
RESPONSE_DURATION = 0.005

SIGNAL_PATH = EXPERIMENT_DIR / "signal2_bearing_impact_V3.npy"
DAY3A_FIGURE_PATH = EXPERIMENT_DIR / "day3A_sampling_length_comparison_V1.1.png"
DAY3B_FIGURE_PATH = EXPERIMENT_DIR / "day3B_cwt_magnitude_minmax_V1.1.png"
DAY3C_FIGURE_PATH = EXPERIMENT_DIR / "day3C_image_generation_V1.1.png"
SCALOGRAM_PATH = EXPERIMENT_DIR / "day3_scalogram_96x96_V1.1.png"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def generate_separated_impact_signal(sample_count: int, fs: int = FS) -> tuple[np.ndarray, np.ndarray]:
    """生成响应段短于冲击周期的教学信号，确保相邻响应不重叠。"""
    if sample_count <= 0:
        raise ValueError("sample_count 必须为正整数")
    time = np.arange(sample_count, dtype=np.float64) / fs
    signal = np.zeros(sample_count, dtype=np.float64)
    period_samples = int(round(fs / FAULT_FREQUENCY))
    response_samples = int(round(fs * RESPONSE_DURATION))

    if response_samples >= period_samples:
        raise ValueError("响应段必须短于冲击周期，否则相邻响应会重叠")

    for start in range(0, sample_count, period_samples):
        end = min(start + response_samples, sample_count)
        local_time = np.arange(end - start, dtype=np.float64) / fs
        response = np.exp(-DAMPING * local_time) * np.sin(
            2 * np.pi * NATURAL_FREQUENCY * local_time
        )
        signal[start:end] = response
    return time, signal


def single_sided_spectrum(signal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """返回便于比较频谱泄漏的单边幅值谱。"""
    spectrum = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(signal.size, d=1 / fs)
    amplitude = 2.0 * np.abs(spectrum) / signal.size
    amplitude[0] /= 2.0
    if signal.size % 2 == 0:
        amplitude[-1] /= 2.0
    return frequencies, amplitude


def create_day3a_figure(
    time_2000: np.ndarray,
    signal_2000: np.ndarray,
    time_2048: np.ndarray,
    signal_2048: np.ndarray,
) -> None:
    """直观看清 N 改变的是时长，不是同一采样率下的采样密度。"""
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(time_2000, signal_2000, linewidth=0.8, label="N=2000")
    axes[0, 0].plot(time_2048, signal_2048, linewidth=0.7, alpha=0.75, label="N=2048")
    axes[0, 0].axvline(0.2, color="crimson", linestyle="--", label="0.2 s")
    axes[0, 0].set_title("同为 10 kHz：2048 点只是多记录 4.8 ms")
    axes[0, 0].set_xlabel("时间 (s)")
    axes[0, 0].set_ylabel("幅值")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.2)

    zoom_samples = int(0.003 * FS)
    axes[0, 1].plot(
        time_2000[:zoom_samples],
        signal_2000[:zoom_samples],
        "o-",
        markersize=3,
        linewidth=0.8,
        label="N=2000",
    )
    axes[0, 1].plot(
        time_2048[:zoom_samples],
        signal_2048[:zoom_samples],
        "x--",
        markersize=4,
        linewidth=0.8,
        label="N=2048",
    )
    axes[0, 1].set_title("前 3 ms 的采样点完全重合：不会更平滑")
    axes[0, 1].set_xlabel("时间 (s)")
    axes[0, 1].set_ylabel("幅值")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.2)

    period_samples = int(FS / FAULT_FREQUENCY)
    response_samples = int(FS * RESPONSE_DURATION)
    first_period = signal_2000[:period_samples]
    axes[1, 0].plot(
        np.arange(period_samples) / FS * 1_000,
        first_period,
        color="#2455A4",
        linewidth=1.0,
    )
    axes[1, 0].axvspan(
        RESPONSE_DURATION * 1_000,
        1_000 / FAULT_FREQUENCY,
        color="#4CAF50",
        alpha=0.18,
        label=f"无响应间隔：{(period_samples - response_samples) / FS * 1000:.1f} ms",
    )
    axes[1, 0].set_title("修正信号：5 ms 衰减响应 + 5 ms 间隔")
    axes[1, 0].set_xlabel("一个冲击周期内的时间 (ms)")
    axes[1, 0].set_ylabel("幅值")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.2)

    sine_2000 = np.sin(2 * np.pi * 100 * time_2000)
    sine_2048 = np.sin(2 * np.pi * 100 * time_2048)
    freq_2000, amp_2000 = single_sided_spectrum(sine_2000, FS)
    freq_2048, amp_2048 = single_sided_spectrum(sine_2048, FS)
    mask_2000 = (freq_2000 >= 70) & (freq_2000 <= 130)
    mask_2048 = (freq_2048 >= 70) & (freq_2048 <= 130)
    axes[1, 1].stem(
        freq_2000[mask_2000],
        amp_2000[mask_2000],
        linefmt="C0-",
        markerfmt="C0o",
        basefmt=" ",
        label="N=2000",
    )
    axes[1, 1].stem(
        freq_2048[mask_2048],
        amp_2048[mask_2048],
        linefmt="C1--",
        markerfmt="C1x",
        basefmt=" ",
        label="N=2048",
    )
    axes[1, 1].set_title("100 Hz FFT：2000 点恰落在频率栅格，2048 点反而泄漏")
    axes[1, 1].set_xlabel("频率 (Hz)")
    axes[1, 1].set_ylabel("单边幅值")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.2)

    figure.suptitle("Day 3A：采样点数为什么取 2000，而不是机械地取 2048？", fontsize=16)
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(DAY3A_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def add_cwt_axes(
    ax: plt.Axes,
    matrix: np.ndarray,
    duration: float,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="upper",
        cmap="viridis",
        extent=[0, duration, SCALES[-1], SCALES[0]],
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_ylim(SCALES[-1], SCALES[0])
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("Scale（小 → 高频）")
    ax.set_title(title)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def create_day3b_figure(
    time: np.ndarray,
    signal: np.ndarray,
    magnitude: np.ndarray,
    normalized: np.ndarray,
) -> None:
    """只解释 CWT → 绝对值 → min-max，不加入图像保存知识。"""
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    axes[0].plot(time, signal, color="#2455A4", linewidth=0.8)
    axes[0].set_title("输入：不重叠的周期冲击信号")
    axes[0].set_xlabel("时间 (s)")
    axes[0].set_ylabel("幅值")
    axes[0].grid(alpha=0.2)
    add_cwt_axes(axes[1], magnitude, time[-1] + 1 / FS, "取绝对值：非负匹配强度")
    add_cwt_axes(
        axes[2],
        normalized,
        time[-1] + 1 / FS,
        "Min-max：只把范围变为 [0, 1]",
        vmin=0,
        vmax=1,
    )
    figure.suptitle("Day 3B：CWT 系数怎样变成强度矩阵？", fontsize=16)
    figure.tight_layout(rect=[0, 0, 1, 0.94])
    figure.savefig(DAY3B_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def create_day3c_figure(normalized: np.ndarray, image_96: np.ndarray) -> None:
    """只解释双线性缩放和正式灰度 PNG。"""
    with Image.open(SCALOGRAM_PATH) as reloaded:
        reloaded_array = np.asarray(reloaded)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(normalized, aspect="auto", origin="upper", cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title(f"原强度矩阵\n{normalized.shape[0]}×{normalized.shape[1]}")
    axes[0].set_xlabel("时间索引")
    axes[0].set_ylabel("Scale 索引")

    axes[1].imshow(image_96, cmap="viridis", vmin=0, vmax=255, interpolation="nearest")
    axes[1].set_title("双线性缩放后的 96×96\n伪彩色只供人观察")
    axes[1].axis("off")

    axes[2].imshow(reloaded_array, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[2].set_title("磁盘中的正式 PNG\n单通道 L，无坐标轴和色条")
    axes[2].axis("off")

    figure.suptitle("Day 3C：强度矩阵怎样保存成 CNN 可读取的图片？", fontsize=16)
    figure.tight_layout(rect=[0, 0, 1, 0.92])
    figure.savefig(DAY3C_FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_outputs(signal: np.ndarray, magnitude: np.ndarray, normalized: np.ndarray, image_96: np.ndarray) -> None:
    """验证修正信号、矩阵和正式图像的关键约束。"""
    period_samples = int(round(FS / FAULT_FREQUENCY))
    response_samples = int(round(FS * RESPONSE_DURATION))
    gap_samples = period_samples - response_samples

    assert signal.shape == (N_2000,)
    assert period_samples == 100
    assert response_samples == 50
    assert gap_samples == 50
    for period_start in range(0, N_2000, period_samples):
        gap_start = min(period_start + response_samples, N_2000)
        gap_end = min(period_start + period_samples, N_2000)
        assert np.allclose(signal[gap_start:gap_end], 0.0)

    assert magnitude.shape == (128, 2000)
    assert normalized.shape == (128, 2000)
    assert np.isclose(normalized.min(), 0.0)
    assert np.isclose(normalized.max(), 1.0)
    assert image_96.shape == (96, 96)
    assert image_96.dtype == np.uint8
    assert np.array_equal(image_96, build_scalogram(signal, FS, output_size=(96, 96)))

    with Image.open(SCALOGRAM_PATH) as image:
        assert image.mode == "L"
        assert image.size == (96, 96)
        assert np.array_equal(np.asarray(image), image_96)


def main() -> None:
    print("=" * 72)
    print("Day 3 V1.1：降低跨度后的 A/B/C 学习实验")
    print("=" * 72)

    time_2000, signal_2000 = generate_separated_impact_signal(N_2000)
    time_2048, signal_2048 = generate_separated_impact_signal(N_2048)
    assert np.array_equal(time_2000, time_2048[:N_2000])
    assert np.array_equal(signal_2000, signal_2048[:N_2000])

    np.save(SIGNAL_PATH, signal_2000)
    create_day3a_figure(time_2000, signal_2000, time_2048, signal_2048)

    magnitude, frequencies = compute_cwt_magnitude(signal_2000, FS, SCALES, WAVELET)
    normalized = normalize_matrix(magnitude, "minmax")
    create_day3b_figure(time_2000, signal_2000, magnitude, normalized)

    image_96 = build_scalogram(signal_2000, FS, output_size=(96, 96))
    save_grayscale_png(image_96, SCALOGRAM_PATH)
    create_day3c_figure(normalized, image_96)
    verify_outputs(signal_2000, magnitude, normalized, image_96)

    frequency_step_2000 = FS / N_2000
    frequency_step_2048 = FS / N_2048
    print("Day 3A")
    print(f"  N=2000: T={N_2000 / FS:.4f} s, Δt={1 / FS * 1000:.4f} ms, Δf={frequency_step_2000:.4f} Hz")
    print(f"  N=2048: T={N_2048 / FS:.4f} s, Δt={1 / FS * 1000:.4f} ms, Δf={frequency_step_2048:.4f} Hz")
    print(f"  2000 Hz 每周期采样点数: {FS / NATURAL_FREQUENCY:.1f}")
    print("  修正信号: 5 ms 响应 + 5 ms 间隔，相邻响应不重叠")
    print("Day 3B")
    print(f"  CWT shape={magnitude.shape}, pseudo-frequency={frequencies[-1]:.2f}~{frequencies[0]:.2f} Hz")
    print(f"  min-max range=[{normalized.min():.1f}, {normalized.max():.1f}]")
    print("Day 3C")
    print(f"  PNG mode=L, size=96×96, range=[{image_96.min()}, {image_96.max()}]")
    print(f"  SHA256={file_sha256(SCALOGRAM_PATH)}")
    print("\n生成文件:")
    for path in [SIGNAL_PATH, DAY3A_FIGURE_PATH, DAY3B_FIGURE_PATH, DAY3C_FIGURE_PATH, SCALOGRAM_PATH]:
        print(f"  - {path.name}")
    print("\n全部验证通过。")


if __name__ == "__main__":
    main()
