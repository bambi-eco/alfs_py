"""
Embedding Cosine-Similarity Visualizer

For a (H, W, embed_dim) .npy embedding produced by embedded_field.py, pick one
or more reference pixels and display the cosine similarity between the feature
vector at each reference and every other pixel — the visualization used in the
DINOv3 demos (https://github.com/facebookresearch/dinov3).

Usage:
    # click on the RGB image to place reference points
    python src/visualize_similarity.py --npy emb.npy --rgb rgb.jpg

    # fixed reference at pixel (x=512, y=300)
    python src/visualize_similarity.py --npy emb.npy --ref 512 300

    # several references in one figure
    python src/visualize_similarity.py --npy emb.npy --ref 512 300 --ref 900 600

    # save without opening a window (requires --ref)
    python src/visualize_similarity.py --npy emb.npy --ref 512 300 --save out.png --no-show
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from numpy.typing import NDArray


# ── similarity computation ────────────────────────────────────────────────────

def l2_normalize(emb: NDArray, eps: float = 1e-8) -> NDArray:
    """Return (H, W, D) with every feature vector L2-normalized along D."""
    norm = np.linalg.norm(emb, axis=-1, keepdims=True)
    return emb / np.maximum(norm, eps)


def cosine_similarity_map(emb_norm: NDArray, ref_xy: Tuple[int, int]) -> NDArray:
    """
    :param emb_norm: (H, W, D) L2-normalized embedding.
    :param ref_xy: (x, y) pixel coordinate of the reference.
    :return: (H, W) float32 cosine similarity map in [-1, 1].
    """
    x, y = ref_xy
    ref = emb_norm[y, x]                       # (D,)
    return emb_norm @ ref                      # (H, W)


# ── plotting ──────────────────────────────────────────────────────────────────

def _draw_reference_marker(ax, ref_xy: Tuple[int, int]) -> None:
    x, y = ref_xy
    ax.scatter([x], [y], s=120, facecolors='none',
               edgecolors='white', linewidths=2.0, zorder=5)
    ax.scatter([x], [y], s=30, c='red', zorder=6)


def _plot_similarity(ax, sim: NDArray, ref_xy: Tuple[int, int],
                     cmap: str = 'turbo', symmetric: bool = True) -> None:
    if symmetric:
        vmax = float(np.nanmax(np.abs(sim)))
        norm = Normalize(vmin=-vmax, vmax=vmax)
    else:
        norm = Normalize(vmin=float(sim.min()), vmax=float(sim.max()))
    im = ax.imshow(sim, cmap=cmap, norm=norm)
    _draw_reference_marker(ax, ref_xy)
    ax.set_title(f"cosine sim @ ({ref_xy[0]}, {ref_xy[1]})", fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_rgb(ax, img: NDArray, title: str,
              refs: Optional[List[Tuple[int, int]]] = None) -> None:
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    if refs:
        for r in refs:
            _draw_reference_marker(ax, r)


def render_figure(emb_norm: NDArray,
                  refs: List[Tuple[int, int]],
                  ref_img: Optional[NDArray],
                  stem: str,
                  cmap: str,
                  symmetric: bool) -> plt.Figure:
    """One panel per reference, plus a leading RGB/reference panel."""
    n_refs = len(refs)
    n_panels = n_refs + 1
    ncols = min(n_panels, 4)
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 5.5 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    _plot_rgb(axes[0], ref_img if ref_img is not None else emb_norm[..., :3],
              "RGB reference" if ref_img is not None else "embedding[:,:,:3]",
              refs=refs)

    for i, ref in enumerate(refs, start=1):
        sim = cosine_similarity_map(emb_norm, ref)
        _plot_similarity(axes[i], sim, ref, cmap=cmap, symmetric=symmetric)

    for k in range(n_panels, len(axes)):
        axes[k].axis('off')

    fig.suptitle(f"cosine similarity — {stem}", fontsize=12)
    return fig


# ── interactive picker ───────────────────────────────────────────────────────

def pick_refs_interactively(emb_norm: NDArray,
                            ref_img: Optional[NDArray],
                            stem: str,
                            cmap: str,
                            symmetric: bool) -> List[Tuple[int, int]]:
    """
    Open a window showing the RGB (or the first 3 embedding channels); each
    left-click adds a reference point and refreshes the similarity panels.
    Right-click clears all references. Close the window to finish.
    """
    H, W, _ = emb_norm.shape
    display = ref_img if ref_img is not None else emb_norm[..., :3]
    refs: List[Tuple[int, int]] = []

    fig, ax_rgb = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)
    _plot_rgb(ax_rgb, display,
              "click to add reference · right-click to clear · close to finish")

    sim_fig: Optional[plt.Figure] = None

    def refresh():
        nonlocal sim_fig
        ax_rgb.clear()
        _plot_rgb(ax_rgb, display,
                  "click to add reference · right-click to clear · close to finish",
                  refs=refs)
        fig.canvas.draw_idle()

        if sim_fig is not None:
            plt.close(sim_fig)
            sim_fig = None
        if refs:
            sim_fig = render_figure(emb_norm, refs, ref_img, stem, cmap, symmetric)
            sim_fig.show()

    def on_click(event):
        if event.inaxes is not ax_rgb or event.xdata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        if not (0 <= x < W and 0 <= y < H):
            return
        if event.button == 1:      # left-click
            refs.append((x, y))
        elif event.button == 3:    # right-click
            refs.clear()
        else:
            return
        refresh()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    return refs


# ── orchestration ────────────────────────────────────────────────────────────

def _load_rgb(rgb_path: Optional[str], H: int, W: int) -> Optional[NDArray]:
    if rgb_path is None:
        return None
    img = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)


def visualize(npy_path: str,
              refs: List[Tuple[int, int]],
              rgb_path: Optional[str] = None,
              save_path: Optional[str] = None,
              show: bool = True,
              cmap: str = 'turbo',
              symmetric: bool = True,
              interactive: bool = False) -> None:
    print(f"Loading {npy_path} ...")
    emb = np.load(npy_path).astype(np.float32)
    H, W, D = emb.shape
    print(f"  Shape: {H} x {W} x {D}")

    print("Normalizing features ...")
    emb_norm = l2_normalize(emb)

    ref_img = _load_rgb(rgb_path, H, W)
    stem = Path(npy_path).stem

    if interactive:
        refs = pick_refs_interactively(emb_norm, ref_img, stem, cmap, symmetric)
        if not refs:
            print("No reference points selected — nothing to save.")
            return

    # clamp & dedupe
    cleaned: List[Tuple[int, int]] = []
    for x, y in refs:
        x = int(np.clip(x, 0, W - 1))
        y = int(np.clip(y, 0, H - 1))
        if (x, y) not in cleaned:
            cleaned.append((x, y))
    print(f"References: {cleaned}")

    fig = render_figure(emb_norm, cleaned, ref_img, stem, cmap, symmetric)

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
        description="Visualize cosine similarity maps of an embedded light field "
                    "against one or more reference pixels (DINOv3-style).",
    )
    parser.add_argument(
        "--npy",
        default=r"Z:\Hugo\embedded_fields_upa\embeddings\test\276_2605.npy",
        help="Path to the (H, W, D) embedding .npy file",
    )
    parser.add_argument(
        "--rgb", dest="rgb_path",
        default=r"Z:\Hugo\alfs\images\test\276_2605.jpg",
        help="Optional RGB reference image shown alongside the similarity maps",
    )
    parser.add_argument(
        "--ref", action='append', nargs=2, type=int, metavar=('X', 'Y'),
        default=[(768, 1034)],
        help="Reference pixel coordinate (x y); repeatable. "
             "If omitted, opens an interactive picker.",
    )
    parser.add_argument(
        "--cmap", default='turbo',
        help="Matplotlib colormap for similarity maps (default: turbo). "
             "Try 'RdBu_r' for signed similarity.",
    )
    parser.add_argument(
        "--asymmetric", action='store_true',
        help="Use the map's own [min, max] instead of a symmetric [-|max|, |max|] "
             "scale. Default is symmetric so 0 sits at the colormap center.",
    )
    parser.add_argument(
        "--save", metavar="OUTPUT_PNG", default=None,
        help="Save the figure to this path (in addition to showing it).",
    )
    parser.add_argument(
        "--no-show", dest="show", action='store_false',
        help="Do not open an interactive window (useful for headless runs). "
             "Requires --ref.",
    )
    args = parser.parse_args()

    refs = [tuple(r) for r in args.ref] if args.ref else []
    interactive = not refs

    if interactive and not args.show:
        parser.error("--no-show requires at least one --ref X Y (no interactive picker).")

    save_path = args.save
    if save_path is None and not args.show:
        save_path = str(Path(args.npy).with_suffix('')) + "_cossim.png"
        print(f"--no-show set without --save; will save to {save_path}")

    visualize(
        npy_path=args.npy,
        refs=refs,
        rgb_path=args.rgb_path,
        save_path=save_path,
        show=args.show,
        cmap=args.cmap,
        symmetric=not args.asymmetric,
        interactive=interactive,
    )


if __name__ == "__main__":
    main()
