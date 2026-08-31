# -*- coding: utf-8 -*-
"""
Day 2: CWT Parameters and Output Structure Experiment
Goal: Understand CWT data structure and 96x96 image formation process
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set font for Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("Day 2: CWT Parameters and Output Structure Experiment")
print("=" * 60)

# ============================================================
# Part 1: CWT Minimum Experiment
# ============================================================
print("\n" + "=" * 60)
print("Part 1: CWT Minimum Experiment")
print("=" * 60)

# Create simple signal: 0~0.5s is 100 Hz sine; 0.5~1.0s is 200 Hz sine
fs = 1000  # Sampling frequency 1000 Hz
t = np.linspace(0, 1, fs, endpoint=False)  # 1 second, 1000 points

# Generate signal
signal = np.zeros_like(t)
signal[:500] = np.sin(2 * np.pi * 100 * t[:500])  # 0~0.5s: 100 Hz
signal[500:] = np.sin(2 * np.pi * 200 * t[500:])  # 0.5~1.0s: 200 Hz

print(f"\nOriginal signal information:")
print(f"  - signal.shape: {signal.shape}")
print(f"  - Sampling frequency: {fs} Hz")
print(f"  - Signal duration: 1 second")
print(f"  - Number of data points: {len(signal)}")

# Test different scale ranges
scale_ranges = [
    (1, 32, "1~32"),
    (1, 64, "1~64"),
    (1, 128, "1~128")
]

print("\n" + "-" * 60)
print("CWT Output Shape Verification:")
print("-" * 60)

cwt_results = {}
for scale_start, scale_end, label in scale_ranges:
    scales = np.arange(scale_start, scale_end + 1)
    
    # Execute CWT
    coefficients, frequencies = pywt.cwt(signal, scales, 'morl', sampling_period=1/fs)
    
    cwt_results[label] = {
        'scales': scales,
        'coefficients': coefficients,
        'frequencies': frequencies
    }
    
    print(f"\nScale range: {label}")
    print(f"  - Number of scales: {len(scales)}")
    print(f"  - scales: {scales[:5]}...{scales[-5:]}")
    print(f"  - coefficients.shape: {coefficients.shape}")
    print(f"  - frequencies.shape: {frequencies.shape}")
    print(f"  - frequencies range: {frequencies[-1]:.2f} Hz ~ {frequencies[0]:.2f} Hz")
    
    # Verify dimension meaning
    print(f"  - Dimension explanation:")
    print(f"    * First dimension (axis=0): scale dimension, size={coefficients.shape[0]}")
    print(f"    * Second dimension (axis=1): time dimension, size={coefficients.shape[1]}")

# ============================================================
# Part 2: Verify scale-frequency relationship
# ============================================================
print("\n" + "=" * 60)
print("Part 2: Verify scale-frequency relationship")
print("=" * 60)

# Use 1~128 scale range for detailed analysis
scales = np.arange(1, 129)
coefficients, frequencies = pywt.cwt(signal, scales, 'morl', sampling_period=1/fs)

# Generate scale-frequency correspondence table
print("\nScale-Frequency Correspondence Table (partial):")
print("-" * 40)
print(f"{'Scale':<10} {'Frequency (Hz)':<15}")
print("-" * 40)

# Select key scale values to display
key_scales = [1, 2, 5, 10, 15, 20, 30, 50, 75, 100, 128]
for s in key_scales:
    idx = s - 1  # Scale starts from 1, index starts from 0
    if idx < len(frequencies):
        print(f"{s:<10} {frequencies[idx]:<15.2f}")

# Analyze relationship
print("\nAnalysis:")
print("-" * 60)

# Check if linear
print("1. How frequency changes as scale increases:")
print(f"   - scale=1: {frequencies[0]:.2f} Hz")
print(f"   - scale=64: {frequencies[63]:.2f} Hz")
print(f"   - scale=128: {frequencies[127]:.2f} Hz")
print(f"   - Conclusion: As scale increases, frequency decreases (inverse relationship)")

# Check linear relationship
print("\n2. Is it strictly linear?")
print("   - Calculate frequency difference between adjacent scales:")
diffs = np.diff(frequencies[:10])
print(f"   - First 10 adjacent frequency differences: {diffs}")
print(f"   - Differences are not constant, so not strictly linear")
print(f"   - Actual relationship is inverse: f is proportional to 1/scale")

# Find scales corresponding to 100 Hz and 200 Hz
print("\n3. Which scales correspond to 100 Hz and 200 Hz at current sampling frequency?")
idx_100 = np.argmin(np.abs(frequencies - 100))
idx_200 = np.argmin(np.abs(frequencies - 200))
print(f"   - Near 100 Hz: scale={scales[idx_100]}, actual frequency={frequencies[idx_100]:.2f} Hz")
print(f"   - Near 200 Hz: scale={scales[idx_200]}, actual frequency={frequencies[idx_200]:.2f} Hz")

# ============================================================
# Part 3: Observe CWT matrix itself
# ============================================================
print("\n" + "=" * 60)
print("Part 3: Observe CWT matrix itself")
print("=" * 60)

# Use 128 scales result
coefficients_128 = cwt_results["1~128"]['coefficients']
frequencies_128 = cwt_results["1~128"]['frequencies']

print(f"\nCWT matrix information:")
print(f"  - coefficients.shape: {coefficients_128.shape}")
print(f"  - Matrix element count: {coefficients_128.size}")
print(f"  - Matrix data type: {coefficients_128.dtype}")
print(f"  - Matrix minimum value: {np.min(np.abs(coefficients_128)):.6f}")
print(f"  - Matrix maximum value: {np.max(np.abs(coefficients_128)):.6f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Original time-domain signal
ax1 = axes[0, 0]
ax1.plot(t, signal, 'b-', linewidth=0.5)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.set_title('Original Time-Domain Signal')
ax1.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Frequency jump point')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. CWT coefficient magnitude matrix
ax2 = axes[0, 1]
im2 = ax2.imshow(np.abs(coefficients_128), aspect='auto', cmap='jet',
                 extent=[0, 1, 128, 1])
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Scale')
ax2.set_title('CWT Coefficient Magnitude Matrix')
plt.colorbar(im2, ax=ax2, label='Coefficient magnitude')

# 3. Scalogram with scale as y-axis
ax3 = axes[1, 0]
im3 = ax3.imshow(np.abs(coefficients_128), aspect='auto', cmap='jet',
                 extent=[0, 1, 128, 1])
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Scale')
ax3.set_title('Scalogram (Scale as y-axis)')
plt.colorbar(im3, ax=ax3, label='Energy')

# 4. Scalogram with frequency as y-axis
ax4 = axes[1, 1]
# Create frequency axis
freq_axis = frequencies_128
im4 = ax4.imshow(np.abs(coefficients_128), aspect='auto', cmap='jet',
                 extent=[0, 1, freq_axis[-1], freq_axis[0]])
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Frequency (Hz)')
ax4.set_title('Scalogram (Frequency as y-axis)')
plt.colorbar(im4, ax=ax4, label='Energy')

plt.tight_layout()
plt.savefig('E:\\硕士学习记录\\研零\\论文刷题\\paper\\04_experiments\\day2_cwt_matrix_visualization.png', 
            dpi=150, bbox_inches='tight')
print("\nChart saved: day2_cwt_matrix_visualization.png")

# ============================================================
# Part 4: Verify "96x96 is not CWT natural output"
# ============================================================
print("\n" + "=" * 60)
print("Part 4: Verify '96x96 is not CWT natural output'")
print("=" * 60)

# Current CWT output
print(f"\nCurrent CWT parameters:")
print(f"  - Number of scales: 128")
print(f"  - Number of time points: {len(t)}")
print(f"  - CWT original output shape: {coefficients_128.shape}")
print(f"  - i.e.: {coefficients_128.shape[0]} x {coefficients_128.shape[1]}")

# Try resize to 96x96
print("\nTry resize to 96x96:")
print("-" * 60)

# Method 1: Simple interpolation resize
from scipy.ndimage import zoom

# Calculate scale factors
target_size = 96
scale_factor_y = target_size / coefficients_128.shape[0]  # scale dimension
scale_factor_x = target_size / coefficients_128.shape[1]  # time dimension

print(f"  - Target size: {target_size} x {target_size}")
print(f"  - Scale dimension scale factor: {scale_factor_y:.4f}")
print(f"  - Time dimension scale factor: {scale_factor_x:.4f}")

# Use bicubic interpolation for resize
coefficients_resized = zoom(np.abs(coefficients_128), (scale_factor_y, scale_factor_x), order=3)

print(f"  - Resized shape: {coefficients_resized.shape}")

# Visualization comparison
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

# Original CWT
ax1 = axes2[0]
im1 = ax1.imshow(np.abs(coefficients_128), aspect='auto', cmap='jet',
                 extent=[0, 1, 128, 1])
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Scale')
ax1.set_title(f'Original CWT\n{coefficients_128.shape[0]} x {coefficients_128.shape[1]}')
plt.colorbar(im1, ax=ax1)

# Resized
ax2 = axes2[1]
im2 = ax2.imshow(coefficients_resized, aspect='auto', cmap='jet',
                 extent=[0, 1, 96, 1])
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Scale (scaled)')
ax2.set_title(f'Resized\n{target_size} x {target_size}')
plt.colorbar(im2, ax=ax2)

# Difference map
ax3 = axes2[2]
# For comparison, resize original to same size
original_resized_for_compare = zoom(np.abs(coefficients_128), 
                                     (target_size/coefficients_128.shape[0], 
                                      target_size/coefficients_128.shape[1]), 
                                     order=3)
difference = np.abs(coefficients_resized - original_resized_for_compare)
im3 = ax3.imshow(difference, aspect='auto', cmap='hot',
                 extent=[0, 1, 96, 1])
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Scale (scaled)')
ax3.set_title('Difference Map (should be 0)')
plt.colorbar(im3, ax=ax3)

plt.tight_layout()
plt.savefig('E:\\硕士学习记录\\研零\\论文刷题\\paper\\04_experiments\\day2_resize_comparison.png', 
            dpi=150, bbox_inches='tight')
print("\nChart saved: day2_resize_comparison.png")

# Explain what resize changes
print("\nWhat resize changes:")
print("-" * 60)
print("1. Scale dimension: 128 -> 96")
print("   - Loses information from 32 scales")
print("   - Uses interpolation to 'guess' intermediate values")
print("   - Low frequency resolution decreases")
print("\n2. Time dimension: 1000 -> 96")
print("   - Loses information from 904 time points")
print("   - Uses interpolation to 'guess' intermediate values")
print("   - Time resolution decreases significantly")
print("\n3. Visual impact:")
print("   - Image becomes more blurred")
print("   - Detail information is lost")
print("   - But overall pattern is preserved")

# ============================================================
# Part 5: Prepare for paper reproduction
# ============================================================
print("\n" + "=" * 60)
print("Part 5: Prepare for paper reproduction")
print("=" * 60)

print("\nPaper parameter investigation results:")
print("-" * 60)

# Parameters extracted from paper analysis
paper_params = {
    "Scale range": "Paper does not explicitly state specific range, only mentions using Morlet wavelet",
    "Wavelet type": "Morlet wavelet (morl) - Paper explicitly states (Page 5)",
    "Sampling frequency": "CWR: 48,000 Hz (Page 11)",
    "CWT output method": "Generate scalogram image - Paper explicitly states (Page 5)",
    "96x96 image generation": "Paper does not specify resize method",
    "Cropping": "Paper does not specify",
    "Resize": "Paper does not specify method",
    "Normalization": "Paper does not specify",
    "Save image": "Paper does not specify format"
}

print("\nPaper explicit parameters:")
for key, value in paper_params.items():
    if "explicitly states" in value:
        print(f"  [OK] {key}: {value}")

print("\nPaper missing parameters:")
for key, value in paper_params.items():
    if "does not specify" in value or "does not explicitly" in value:
        print(f"  [X] {key}: {value}")

print("\nAuthor code supplementary information:")
print("-" * 60)
print("Repository: s-whynot/Bearing_Fault_MFPT")
print("URL: https://github.com/s-whynot/Bearing_Fault_MFPT")
print("\nSupplementary content (based on README):")
print("  - Uses STFT (not CWT)")
print("  - Image generation flow: STFT -> normalization -> save as image")
print("  - But this repository is for MFPT dataset, not CWR")

print("\n" + "=" * 60)
print("Experiment completed!")
print("=" * 60)
