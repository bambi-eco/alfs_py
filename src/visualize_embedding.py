"""
Embedding Light Field Visualizer

Visualizes a (H, W, embed_dim) .npy embedding file produced by embedded_field.py.

Usage:
    python src/visualize_embedding.py path/to/embedding.npy
    python src/visualize_embedding.py path/to/embedding.npy --method umap
    python src/visualize_embedding.py path/to/embedding.npy --method pca --components 1
    python src/visualize_embedding.py path/to/embedding.npy --side-by-side path/to/rgb.png
    python src/visualize_embedding.py path/to/embedding.npy --save output.png --no-show
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


# ── dimensionality reduction ──────────────────────────────────────────────────

def _normalize_channels(arr: NDArray) -> NDArray:
    """Normalize each channel of (H, W, C) independently to [0, 1]."""
    out = np.zeros_like(arr, dtype=np.float32)
    for c in range(arr.shape[2]):
        ch = arr[:, :, c]
        lo, hi = ch.min(), ch.max()
        if hi > lo:
            out[:, :, c] = (ch - lo) / (hi - lo)
    return out


def reduce_embedding(emb: NDArray, method: str, n_components: int,
                     max_pixels: int = 200_000) -> NDArray:
    """
    Reduces (H, W, D) → (H, W, n_components) using the chosen method.

    For large images a random spatial subsample is used for fitting; the full
    image is then transformed in one pass.

    :param emb: (H, W, D) float32 embedding array.
    :param method: 'pca', 'umap', or 'tsne'.
    :param n_components: Number of output dimensions (1, 2, or 3).
    :param max_pixels: Maximum pixels used for fitting (subsampled if larger).
    :return: (H, W, n_components) float32 array, values in [0, 1].
    """
    H, W, D = emb.shape
    flat = emb.reshape(-1, D)
    N = flat.shape[0]

    print(f"  Input: ({H}x{W}x{D}), fitting on {min(N, max_pixels):,} / {N:,} pixels ...")

    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)
        if N > max_pixels:
            idx = np.random.choice(N, max_pixels, replace=False)
            reducer.fit(flat[idx])
            reduced = reducer.transform(flat)
        else:
            reduced = reducer.fit_transform(flat)
        explained = reducer.explained_variance_ratio_
        print(f"  PCA explained variance: {[f'{v:.1%}' for v in explained]}")

    elif method == 'umap':
        from umap import UMAP
        reducer = UMAP(n_components=n_components)
        if N > max_pixels:
            idx = np.random.choice(N, max_pixels, replace=False)
            reducer.fit(flat[idx])
            reduced = reducer.transform(flat)
        else:
            reduced = reducer.fit_transform(flat)

    elif method == 'tsne':
        from sklearn.manifold import TSNE
        if N > max_pixels:
            flat = flat[np.random.choice(N, max_pixels, replace=False)]
            print(f"  t-SNE: subsampled to {max_pixels:,} pixels (no transform back)")
        reduced = TSNE(n_components=n_components).fit_transform(flat)

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'pca', 'umap', or 'tsne'.")

    reduced = reduced.astype(np.float32).reshape(-1, n_components)
    # pad back to H*W if t-SNE subsampled
    if reduced.shape[0] < H * W:
        full = np.zeros((H * W, n_components), dtype=np.float32)
        full[:reduced.shape[0]] = reduced
        reduced = full

    result = _normalize_channels(reduced.reshape(H, W, n_components))
    print(f"  Output: {result.shape}")
    return result


# ── plot helpers ──────────────────────────────────────────────────────────────

def _plot_rgb(ax, img: NDArray, title: str) -> None:
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis('off')


def _plot_heatmap(ax, channel: NDArray, title: str, cmap: str = 'viridis') -> None:
    im = ax.imshow(channel, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def visualize(
    npy_path: str,
    method: str = 'pca',
    n_components: int = 3,
    rgb_path: str = None,
    save_path: str = None,
    show: bool = True,
    max_pixels: int = 200_000,
) -> None:
    """
    Main visualization function.

    :param npy_path: Path to the (H, W, D) embedding .npy file.
    :param method: Dimensionality reduction method ('pca', 'umap', 'tsne').
    :param n_components: 1 → single heatmap; 2 → two heatmaps; 3 → RGB image.
    :param rgb_path: Optional path to an RGB reference image shown alongside.
    :param save_path: If given, save the figure to this path.
    :param show: Whether to call plt.show() (set False for headless runs).
    :param max_pixels: Max pixels used for DR fitting.
    """
    print(f"Loading {npy_path} ...")
    emb = np.load(npy_path)
    H, W, D = emb.shape
    print(f"  Shape: {H} x {W} x {D}")

    print(f"Applying {method.upper()} ({n_components} components) ...")
    reduced = reduce_embedding(emb, method, n_components, max_pixels)

    # ── load reference RGB if provided ───────────────────────────────────────
    ref_img = None
    if rgb_path is not None:
        ref_img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        ref_img = cv2.resize(ref_img, (W, H), interpolation=cv2.INTER_LINEAR)

    # ── build figure ──────────────────────────────────────────────────────────
    stem = Path(npy_path).stem
    method_label = f"{method.upper()} ({n_components}C)"

    if n_components == 3:
        n_panels = 2 + (1 if ref_img is not None else 0)  # RGB + variance bars + optional ref
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6),
                                 constrained_layout=True)
        if n_panels == 1:
            axes = [axes]

        panel = 0
        if ref_img is not None:
            _plot_rgb(axes[panel], ref_img, "RGB reference"); panel += 1

        _plot_rgb(axes[panel], reduced, f"{method_label} → RGB\n{stem}"); panel += 1

        # per-component heatmaps in a small sub-grid
        fig2, comp_axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        for c, cmap in enumerate(['Reds', 'Greens', 'Blues']):
            _plot_heatmap(comp_axes[c], reduced[:, :, c],
                          f"Component {c + 1}", cmap=cmap)
        fig2.suptitle(f"{method_label} components — {stem}", fontsize=11)
        if save_path:
            comp_path = save_path.replace('.png', '_components.png')
            fig2.savefig(comp_path, dpi=150, bbox_inches='tight')
            print(f"Saved components figure: {comp_path}")

    elif n_components == 2:
        n_panels = 2 + (1 if ref_img is not None else 0)
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6),
                                 constrained_layout=True)
        panel = 0
        if ref_img is not None:
            _plot_rgb(axes[panel], ref_img, "RGB reference"); panel += 1
        _plot_heatmap(axes[panel], reduced[:, :, 0], f"{method_label} — component 1"); panel += 1
        _plot_heatmap(axes[panel], reduced[:, :, 1], f"{method_label} — component 2"); panel += 1

    else:  # n_components == 1
        n_panels = 1 + (1 if ref_img is not None else 0)
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6),
                                 constrained_layout=True)
        if n_panels == 1:
            axes = [axes]
        panel = 0
        if ref_img is not None:
            _plot_rgb(axes[panel], ref_img, "RGB reference"); panel += 1
        _plot_heatmap(axes[panel], reduced[:, :, 0],
                      f"{method_label} — {stem}", cmap='inferno')

    fig.suptitle(f"Embedding visualisation — {stem}", fontsize=12)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close('all')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize an embedding light field .npy file."
    )
    parser.add_argument("--npy", default=r"Z:\Hugo\embedded_fields\embeddings\train\101_962.npy", help="Path to the (H, W, D) embedding .npy file")
    parser.add_argument(
        "--method", choices=["pca", "umap", "tsne"], default="pca",
        help="Dimensionality reduction method (default: pca)",
    )
    parser.add_argument(
        "--components", type=int, choices=[1, 2, 3], default=1,
        help="Number of output components: 1=heatmap, 3=RGB (default: 3)",
    )
    parser.add_argument(
        "--side-by-side", metavar="RGB_PATH", dest="rgb_path", default=r"Z:\Hugo\alfs\images\train\101_962.npy",
        help="Optional RGB reference image shown alongside the embedding",
    )
    parser.add_argument(
        "--save", metavar="OUTPUT_PNG", default=None,
        help="Save the figure to this path instead of (or in addition to) showing it",
    )
    parser.add_argument(
        "--no-show", dest="show", action="store_false",
        help="Do not open an interactive window (useful for headless runs)",
    )
    parser.add_argument(
        "--max-pixels", type=int, default=200_000, metavar="N",
        help="Max pixels used for DR fitting (default: 200000)",
    )
    args = parser.parse_args()

    # auto-generate save path next to the .npy if --save not specified but --no-show set
    save_path = args.save
    if save_path is None and not args.show:
        save_path = str(Path(args.npy).with_suffix('')) + f"_{args.method}{args.components}c.png"
        print(f"--no-show set without --save; will save to {save_path}")

    visualize(
        npy_path=args.npy,
        method=args.method,
        n_components=args.components,
        rgb_path=args.rgb_path,
        save_path=save_path,
        show=args.show,
        max_pixels=args.max_pixels,
    )


if __name__ == "__main__":
    main()
