from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray


def reduce_to_rgb(
    embedding_npy: str,
    method: str = 'pca',
    output_file: str = 'reduced.png',
    mask_npy: Optional[str] = None,
) -> NDArray:
    """
    Loads a (H, W, embed_dim) embedding light field and reduces it to a 3-channel
    RGB image via dimensionality reduction.

    Each of the 3 output channels is independently normalized to [0, 255].

    :param embedding_npy: Path to the ``.npy`` file containing the (H, W, D) array.
    :param method: Reduction method — ``'pca'`` (default) or ``'umap'``.
    :param output_file: Path to save the resulting PNG image.
    :param mask_npy: Optional path to a (H, W) boolean ``.npy`` mask. Only unmasked
        pixels participate in fitting; masked pixels are set to black.
    :return: (H, W, 3) uint8 RGB array.
    """
    emb = np.load(embedding_npy)
    H, W, D = emb.shape
    flat = emb.reshape(-1, D)  # (H*W, D)

    if mask_npy is not None:
        mask = np.load(mask_npy).reshape(-1).astype(bool)
        fit_data = flat[mask]
    else:
        mask = None
        fit_data = flat

    reduced_fit = _apply_reduction(fit_data, method, n_components=3)

    if mask is not None:
        reduced = np.zeros((H * W, 3), dtype=np.float32)
        reduced[mask] = reduced_fit
    else:
        reduced = reduced_fit

    reduced = reduced.reshape(H, W, 3)

    # Normalize each channel independently to [0, 255]
    out = np.zeros_like(reduced, dtype=np.uint8)
    for c in range(3):
        ch = reduced[:, :, c]
        lo, hi = ch.min(), ch.max()
        if hi > lo:
            out[:, :, c] = ((ch - lo) / (hi - lo) * 255).astype(np.uint8)

    cv2.imwrite(output_file, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return out


def reduce_to_2d(
    embedding_npy: str,
    method: str = 'pca',
    output_file: str = 'reduced_2d.npy',
) -> NDArray:
    """
    Reduces a (H, W, embed_dim) embedding light field to 2 components and saves as
    a (H, W, 2) float32 ``.npy`` file.

    :param embedding_npy: Path to the ``.npy`` source file.
    :param method: Reduction method — ``'pca'`` or ``'umap'``.
    :param output_file: Path to save the (H, W, 2) result.
    :return: (H, W, 2) float32 array.
    """
    emb = np.load(embedding_npy)
    H, W, D = emb.shape
    flat = emb.reshape(-1, D)

    reduced = _apply_reduction(flat, method, n_components=2).reshape(H, W, 2)
    np.save(output_file, reduced)
    return reduced


def _apply_reduction(data: NDArray, method: str, n_components: int) -> NDArray:
    """
    Applies the chosen dimensionality reduction to a (N, D) array.

    :param data: Input array of shape (N, D).
    :param method: ``'pca'``, ``'umap'``, or ``'tsne'``.
    :param n_components: Number of output dimensions.
    :return: (N, n_components) float32 array.
    """
    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)
    elif method == 'umap':
        from umap import UMAP
        reducer = UMAP(n_components=n_components)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components)
    else:
        raise ValueError(f"Unknown reduction method '{method}'. Use 'pca', 'umap', or 'tsne'.")

    return reducer.fit_transform(data).astype(np.float32)
