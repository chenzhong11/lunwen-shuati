# -*- coding: utf-8 -*-
"""
Day2: 理解scale的最小实验
目标：直观看懂scale与frequency的关系
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("Day2: 理解scale的最小实验")
print("=" * 60)

# ============================================================
# 1. 生成三个简单信号
# ============================================================
fs = 1000  # 采样频率
t = np.linspace(0, 1, fs, endpoint=False)  # 1秒

# A: 100Hz正弦
signal_A = np.sin(2 * np.pi * 100 * t)

# B: 200Hz正弦
signal_B = np.sin(2 * np.pi * 200 * t)

# C: 100Hz + 200Hz
signal_C = signal_A + signal_B

print("\n信号信息:")
print(f"  采样频率: {fs} Hz")
print(f"  信号时长: 1 秒")
print(f"  数据点数: {len(t)}")

# ============================================================
# 2. 计算scale-frequency对应关系
# ============================================================
scales = np.arange(1, 129)  # 1~128

# 使用pywt.scale2frequency计算每个scale对应的频率
# 注意：scale2frequency返回归一化频率，需要乘以fs得到实际频率
frequencies = []
for s in scales:
    freq_normalized = pywt.scale2frequency('morl', s)
    freq_actual = freq_normalized * fs  # 转换为实际频率
    frequencies.append(freq_actual)
frequencies = np.array(frequencies)

print("\n" + "=" * 60)
print("Scale-Frequency对应关系（部分）")
print("=" * 60)
print(f"{'Scale':<8} {'Frequency (Hz)':<15}")
print("-" * 25)

# 找到100Hz和200Hz对应的scale
idx_100 = np.argmin(np.abs(frequencies - 100))
idx_200 = np.argmin(np.abs(frequencies - 200))

for s in [1, 2, 4, 8, 10, 15, 20, 30, 50, 75, 100, 128]:
    idx = s - 1
    marker = ""
    if s == scales[idx_100]:
        marker = " <-- 最接近100Hz"
    elif s == scales[idx_200]:
        marker = " <-- 最接近200Hz"
    print(f"{s:<8} {frequencies[idx]:<15.2f}{marker}")

print(f"\n100Hz 对应 scale ≈ {scales[idx_100]} (实际频率: {frequencies[idx_100]:.2f} Hz)")
print(f"200Hz 对应 scale ≈ {scales[idx_200]} (实际频率: {frequencies[idx_200]:.2f} Hz)")

# ============================================================
# 3. 对三个信号做CWT并可视化
# ============================================================
print("\n" + "=" * 60)
print("对三个信号做CWT")
print("=" * 60)

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

signals = [signal_A, signal_B, signal_C]
titles = ['A: 100Hz', 'B: 200Hz', 'C: 100Hz + 200Hz']

for i, (sig, title) in enumerate(zip(signals, titles)):
    # 做CWT
    coefficients, _ = pywt.cwt(sig, scales, 'morl', sampling_period=1/fs)
    magnitude = np.abs(coefficients)
    
    # 左图：时域信号
    ax1 = axes[i, 0]
    ax1.plot(t, sig, 'b-', linewidth=0.5)
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('幅值')
    ax1.set_title(f'{title} - 时域信号')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 0.1)  # 只显示前0.1秒，看得更清楚
    
    # 右图：CWT scalogram
    ax2 = axes[i, 1]
    im = ax2.imshow(magnitude, aspect='auto', cmap='jet',
                    extent=[0, 1, scales[-1], scales[0]])
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('Scale')
    ax2.set_title(f'{title} - CWT Scalogram')
    
    # 标注100Hz和200Hz对应的scale
    ax2.axhline(y=scales[idx_100], color='white', linestyle='--', linewidth=2, 
                label=f'100Hz (scale={scales[idx_100]})')
    ax2.axhline(y=scales[idx_200], color='cyan', linestyle='--', linewidth=2,
                label=f'200Hz (scale={scales[idx_200]})')
    ax2.legend(loc='upper right', fontsize=8)
    
    plt.colorbar(im, ax=ax2, label='系数幅值')

plt.tight_layout()
plt.savefig('E:\\硕士学习记录\\研零\\论文刷题\\paper\\04_experiments\\day2_scale_three_signals.png', 
            dpi=150, bbox_inches='tight')
print("图表已保存: day2_scale_three_signals.png")

# ============================================================
# 4. 测试不同scale范围
# ============================================================
print("\n" + "=" * 60)
print("测试不同scale范围")
print("=" * 60)

# 使用信号C（100Hz + 200Hz）
scale_ranges = [
    (1, 32, "1~32"),
    (1, 64, "1~64"),
    (1, 128, "1~128")
]

print(f"\n信号: 100Hz + 200Hz 正弦波")
print("-" * 70)
print(f"{'Scale范围':<12} {'shape':<15} {'频率范围':<25} {'覆盖100Hz':<12} {'覆盖200Hz':<12}")
print("-" * 70)

for scale_start, scale_end, label in scale_ranges:
    scales_test = np.arange(scale_start, scale_end + 1)
    
    # 计算频率范围
    freq_high = pywt.scale2frequency('morl', scale_start) * fs
    freq_low = pywt.scale2frequency('morl', scale_end) * fs
    
    # 做CWT
    coeffs, _ = pywt.cwt(signal_C, scales_test, 'morl', sampling_period=1/fs)
    
    # 检查是否覆盖100Hz和200Hz
    freqs_test = np.array([pywt.scale2frequency('morl', s) * fs for s in scales_test])
    cover_100 = "是" if np.min(np.abs(freqs_test - 100)) < 10 else "否"
    cover_200 = "是" if np.min(np.abs(freqs_test - 200)) < 10 else "否"
    
    print(f"{label:<12} {str(coeffs.shape):<15} {freq_low:.1f} ~ {freq_high:.1f} Hz{'':<5} {cover_100:<12} {cover_200:<12}")

# ============================================================
# 5. 矩阵解释实验
# ============================================================
print("\n" + "=" * 60)
print("矩阵解释实验")
print("=" * 60)

# 使用信号C，scale 1~128
coefficients_C, _ = pywt.cwt(signal_C, scales, 'morl', sampling_period=1/fs)

print(f"\ncoefficients.shape = {coefficients_C.shape}")
print(f"含义: {coefficients_C.shape[0]}个scale × {coefficients_C.shape[1]}个时间点")

print("\n具体元素解释:")
print("-" * 60)

# coefficients[0, 0]
print(f"coefficients[0, 0] = {coefficients_C[0, 0]:.4f}")
print(f"  位置: scale索引=0, 时间索引=0")
print(f"  含义: 第1个scale (scale={scales[0]}) 在第1个时间点 (t={t[0]:.3f}s) 的小波系数")
print(f"  对应频率: {frequencies[0]:.2f} Hz")
print(f"  物理意义: 信号在t=0时刻与scale={scales[0]}的小波的匹配程度")

print(f"\ncoefficients[20, 500] = {coefficients_C[20, 500]:.4f}")
print(f"  位置: scale索引=20, 时间索引=500")
print(f"  含义: 第21个scale (scale={scales[20]}) 在第501个时间点 (t={t[500]:.3f}s) 的小波系数")
print(f"  对应频率: {frequencies[20]:.2f} Hz")
print(f"  物理意义: 信号在t=0.5s时刻与scale={scales[20]}的小波的匹配程度")

print(f"\ncoefficients[80, 500] = {coefficients_C[80, 500]:.4f}")
print(f"  位置: scale索引=80, 时间索引=500")
print(f"  含义: 第81个scale (scale={scales[80]}) 在第501个时间点 (t={t[500]:.3f}s) 的小波系数")
print(f"  对应频率: {frequencies[80]:.2f} Hz")
print(f"  物理意义: 信号在t=0.5s时刻与scale={scales[80]}的小波的匹配程度")

# ============================================================
# 6. 可视化矩阵结构
# ============================================================
print("\n" + "=" * 60)
print("可视化矩阵结构")
print("=" * 60)

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

# 原始信号
ax1 = axes2[0, 0]
ax1.plot(t, signal_C, 'b-', linewidth=0.5)
ax1.set_xlabel('时间 (s)')
ax1.set_ylabel('幅值')
ax1.set_title('信号C: 100Hz + 200Hz')
ax1.grid(True, alpha=0.3)

# CWT矩阵（scale为纵轴）
ax2 = axes2[0, 1]
im2 = ax2.imshow(np.abs(coefficients_C), aspect='auto', cmap='jet',
                 extent=[0, 1, scales[-1], scales[0]])
ax2.set_xlabel('时间 (s)')
ax2.set_ylabel('Scale')
ax2.set_title('CWT Scalogram (Scale为纵轴)')
ax2.axhline(y=scales[idx_100], color='white', linestyle='--', linewidth=2)
ax2.axhline(y=scales[idx_200], color='cyan', linestyle='--', linewidth=2)
plt.colorbar(im2, ax=ax2)

# 频率为纵轴的scalogram
ax3 = axes2[1, 0]
im3 = ax3.imshow(np.abs(coefficients_C), aspect='auto', cmap='jet',
                 extent=[0, 1, frequencies[-1], frequencies[0]])
ax3.set_xlabel('时间 (s)')
ax3.set_ylabel('频率 (Hz)')
ax3.set_title('CWT Scalogram (频率为纵轴)')
ax3.axhline(y=100, color='white', linestyle='--', linewidth=2, label='100Hz')
ax3.axhline(y=200, color='cyan', linestyle='--', linewidth=2, label='200Hz')
ax3.legend()
plt.colorbar(im3, ax=ax3)

# 100Hz和200Hz对应的scale行
ax4 = axes2[1, 1]
ax4.plot(t, np.abs(coefficients_C[idx_100, :]), 'r-', linewidth=1, 
         label=f'scale={scales[idx_100]} (~100Hz)')
ax4.plot(t, np.abs(coefficients_C[idx_200, :]), 'b-', linewidth=1,
         label=f'scale={scales[idx_200]} (~200Hz)')
ax4.set_xlabel('时间 (s)')
ax4.set_ylabel('系数幅值')
ax4.set_title('100Hz和200Hz对应的系数随时间变化')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('E:\\硕士学习记录\\研零\\论文刷题\\paper\\04_experiments\\day2_scale_matrix_explain.png', 
            dpi=150, bbox_inches='tight')
print("图表已保存: day2_scale_matrix_explain.png")

print("\n" + "=" * 60)
print("实验完成!")
print("=" * 60)
