"""Day 4 teaching experiment: what a 2-D CNN sees in a CWT-like image.

This script intentionally does not train a CNN or alter the formal WDCNN
pipeline.  It creates a small artificial 96x96 matrix and applies fixed,
human-readable kernels so that convolution, feature maps, ReLU, and pooling
can be inspected visually.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "04_experiments" / "day4"


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-12)


def make_simple_image(size: int = 96) -> np.ndarray:
    """A simple image containing vertical/horizontal lines and a bright spot."""
    image = np.zeros((size, size), dtype=float)
    image[18:80, 25:29] = 1.0
    image[61:65, 12:84] = 0.75
    image[39:45, 68:74] = 1.0
    image[40:44, 69:73] = 1.5
    # A small repeated texture, similar only in appearance to local CWT texture.
    for x in (42, 47, 52):
        image[28:52, x : x + 2] = 0.55
    return image


def make_teaching_cwt_like(size: int = 96) -> np.ndarray:
    """Make a clearly artificial scale x time energy matrix for the lesson."""
    rng = np.random.default_rng(4)
    time = np.arange(size)[None, :]
    scale = np.arange(size)[:, None]
    image = 0.025 * rng.random((size, size))

    # Localized high-frequency burst: short in time, concentrated at small scale.
    image += 1.1 * np.exp(-((time - 25) ** 2) / (2 * 3.0**2)) * np.exp(
        -((scale - 18) ** 2) / (2 * 7.0**2)
    )
    # A lower-frequency burst: wider in time and at a larger scale.
    image += 0.9 * np.exp(-((time - 66) ** 2) / (2 * 8.0**2)) * np.exp(
        -((scale - 62) ** 2) / (2 * 11.0**2)
    )
    # Repeated vertical energy traces, with a scale-dependent envelope.
    for center in (36, 44, 52, 60, 68):
        image += 0.22 * np.exp(-((time - center) ** 2) / (2 * 1.2**2)) * np.exp(
            -((scale - 39) ** 2) / (2 * 18.0**2)
        )
    return minmax(image)


def convolve_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Valid 2-D cross-correlation, matching the usual CNN convention."""
    kh, kw = kernel.shape
    oh, ow = image.shape[0] - kh + 1, image.shape[1] - kw + 1
    output = np.empty((oh, ow), dtype=float)
    for row in range(oh):
        for col in range(ow):
            output[row, col] = np.sum(image[row : row + kh, col : col + kw] * kernel)
    return output


def max_pool2d(image: np.ndarray, size: int = 2, stride: int = 2) -> np.ndarray:
    oh = (image.shape[0] - size) // stride + 1
    ow = (image.shape[1] - size) // stride + 1
    output = np.empty((oh, ow), dtype=float)
    for row in range(oh):
        for col in range(ow):
            block = image[row * stride : row * stride + size, col * stride : col * stride + size]
            output[row, col] = block.max()
    return output


def style_axis(ax, title: str, xlabel: str = "column / time", ylabel: str = "row / scale") -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    simple = make_simple_image()
    cwt_like = make_teaching_cwt_like()

    kernels = {
        "vertical edge": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float),
        "horizontal edge": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=float),
        "bright point": np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=float),
    }
    responses = {name: convolve_valid(simple, kernel) for name, kernel in kernels.items()}

    # 01: the input matrix and a clearly labelled artificial CWT-like matrix.
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].imshow(simple, cmap="gray", vmin=0, vmax=simple.max())
    style_axis(axes[0], "Simple teaching input (96×96)")
    axes[1].imshow(cwt_like, cmap="gray", aspect="auto", origin="upper")
    style_axis(axes[1], "Teaching CWT-like matrix (not project CWT)", ylabel="scale-like row")
    save(fig, "01_input_example.png")

    # 02: make the kernels themselves visible as small pattern detectors.
    fig, axes = plt.subplots(1, 4, figsize=(10, 3), constrained_layout=True)
    axes[0].imshow(simple, cmap="gray", vmin=0, vmax=simple.max())
    style_axis(axes[0], "Input")
    for ax, (name, kernel) in zip(axes[1:], kernels.items()):
        ax.imshow(kernel, cmap="coolwarm", vmin=-4, vmax=4)
        for (row, col), value in np.ndenumerate(kernel):
            ax.text(col, row, f"{value:g}", ha="center", va="center", fontsize=10)
        style_axis(ax, name, xlabel="kernel column", ylabel="kernel row")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
    save(fig, "02_kernel_demo.png")

    # 03: one kernel produces one response map; brightness is response magnitude.
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.2), constrained_layout=True)
    axes[0].imshow(simple, cmap="gray", vmin=0, vmax=simple.max())
    style_axis(axes[0], "Input")
    for ax, (name, response) in zip(axes[1:], responses.items()):
        ax.imshow(np.abs(response), cmap="magma")
        style_axis(ax, f"Feature map: {name}")
    save(fig, "03_feature_map_demo.png")

    # 04: pooling is shown after ReLU, as in a standard CNN block.
    response = responses["vertical edge"]
    relu = np.maximum(response, 0)
    pooled = max_pool2d(relu, size=2, stride=2)
    fig, axes = plt.subplots(2, 2, figsize=(7, 6), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(relu, cmap="magma")
    style_axis(axes[0], f"After ReLU ({relu.shape[0]}×{relu.shape[1]})")
    axes[1].imshow(pooled, cmap="magma")
    style_axis(axes[1], f"After 2×2 max pool ({pooled.shape[0]}×{pooled.shape[1]})")
    crop = relu[32:48, 62:78]
    axes[2].imshow(crop, cmap="magma", interpolation="nearest")
    style_axis(axes[2], "Same local region before pooling")
    pooled_crop = pooled[16:24, 31:39]
    axes[3].imshow(pooled_crop, cmap="magma", interpolation="nearest")
    style_axis(axes[3], "Same region after pooling")
    save(fig, "04_pooling_demo.png")

    # Keep the console output short but useful for checking the experiment.
    print(f"input_shape={simple.shape}")
    print(f"single_kernel_to_feature_map={responses['vertical edge'].shape}")
    print(f"relu_shape={relu.shape}; pooled_shape={pooled.shape}")
    print("note=teaching matrices only; no training and no formal WDCNN changes")
    for path in sorted(FIG_DIR.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
