"""Reducing an N-channel field to something you can look at.

A 1280-dimensional field is not viewable, so it is projected down to three channels (an RGB
false-colour image) or two (a scatter embedding). PCA, UMAP and t-SNE are all available; PCA
is the default because it is the only one of the three that is cheap, deterministic, and has
an out-of-sample transform.

Two things here differ deliberately from the obvious implementation:

**Uncovered pixels are excluded from the fit.** A rendered field is only valid where shots
overlapped; everywhere else it is zero. Those zeros are not observations, so pass the
coverage mask and they are neither fitted on nor coloured -- they come back black, as absent
data rather than as a confident-looking false colour. (Measured on synthetic fields, PCA
finds the dominant direction with or without the mask, so this is about not inventing data
for the empty region rather than about rescuing the fit.)

**A fitted reducer can be reused.** Fitting per frame gives every frame its own basis, so the
same terrain changes colour from one frame to the next and a sequence flickers. Fit once on a
representative frame and reuse it -- which is why this is a class and not only a function.
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

__all__ = ['FieldReducer', 'reduce_to_rgb', 'reduce_to_2d', 'is_available']

_METHODS = ('pca', 'umap', 'tsne')

# t-SNE has no out-of-sample extension: there is no meaningful way to place a new point in an
# existing embedding, so scikit-learn's TSNE offers no `transform`.
_NO_TRANSFORM = ('tsne',)


def is_available(method: str = 'pca') -> bool:
    """
    :param method: The reduction method to check.
    :return: Whether its dependency is installed.
    """
    try:
        if method == 'umap':
            import umap  # noqa: F401
        else:
            import sklearn  # noqa: F401
    except ImportError:
        return False
    return True


def _make_reducer(method: str, n_components: int, **kwargs):
    method = method.lower()
    if method == 'pca':
        from sklearn.decomposition import PCA

        return PCA(n_components=n_components, **kwargs)
    if method == 'umap':
        from umap import UMAP

        return UMAP(n_components=n_components, **kwargs)
    if method == 'tsne':
        from sklearn.manifold import TSNE

        return TSNE(n_components=n_components, **kwargs)
    raise ValueError(
        f'Unknown reduction method {method!r}. Use one of: {", ".join(_METHODS)}.')


def _as_flat(field: NDArray) -> tuple:
    arr = np.asarray(field)
    if arr.ndim != 3:
        raise ValueError(f'Expected an (H, W, C) field, got shape {arr.shape}')
    height, width, channels = arr.shape
    return arr.reshape(-1, channels), height, width


def _as_flat_mask(mask: Optional[NDArray], count: int) -> Optional[NDArray]:
    if mask is None:
        return None
    flat = np.asarray(mask).reshape(-1).astype(bool)
    if flat.size != count:
        raise ValueError(f'Mask has {flat.size} entries but the field has {count} pixels')
    return flat


class FieldReducer:
    """
    A dimensionality reduction fitted once and applied to many fields.

    :cvar method: The reduction method.
    :cvar n_components: How many output channels.
    """

    def __init__(self, method: str = 'pca', n_components: int = 3, **kwargs):
        """
        :param method: ``"pca"``, ``"umap"`` or ``"tsne"``.
        :param n_components: The number of output components.
        :param kwargs: Forwarded to the underlying estimator.
        """
        self.method = method.lower()
        if self.method not in _METHODS:
            raise ValueError(
                f'Unknown reduction method {method!r}. Use one of: {", ".join(_METHODS)}.')
        self.n_components = n_components
        self._kwargs = kwargs
        self._reducer = None
        self._range: Optional[tuple] = None

    @property
    def fitted(self) -> bool:
        """:return: Whether this reducer has been fitted."""
        return self._reducer is not None

    @property
    def reusable(self) -> bool:
        """
        :return: Whether a fit can be applied to other fields. False for t-SNE, which has no
            out-of-sample transform.
        """
        return self.method not in _NO_TRANSFORM

    def fit(self, field: NDArray, mask: Optional[NDArray] = None) -> 'FieldReducer':
        """
        Fits on one field.

        :param field: An ``(H, W, C)`` field.
        :param mask: An ``(H, W)`` boolean mask of pixels to fit on (optional). Pass the
            coverage mask: uncovered pixels are zeros rather than observations, and fitting
            on them lets the empty region influence the basis and the output range.
        :return: This reducer.
        """
        flat, _, _ = _as_flat(field)
        selection = _as_flat_mask(mask, len(flat))
        data = flat[selection] if selection is not None else flat

        if len(data) == 0:
            raise ValueError('Nothing to fit on: the mask selected no pixels')

        self._reducer = _make_reducer(self.method, self.n_components, **self._kwargs)
        if self.reusable:
            self._reducer.fit(data)
            reduced = self._reducer.transform(data)
        else:
            reduced = self._reducer.fit_transform(data)

        self._range = (reduced.min(axis=0), reduced.max(axis=0))
        return self

    def transform(self, field: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Applies the fit to a field.

        :param field: An ``(H, W, C)`` field.
        :param mask: An ``(H, W)`` boolean mask (optional). Masked-out pixels come back zero
            rather than being fed through the estimator.
        :return: An ``(H, W, n_components)`` float32 array.
        :raises RuntimeError: If the reducer has not been fitted, or the method has no
            out-of-sample transform.
        """
        if not self.fitted:
            raise RuntimeError('Fit the reducer before transforming')
        if not self.reusable:
            raise RuntimeError(
                f'{self.method} has no out-of-sample transform, so a fit cannot be applied to '
                'another field. Use fit_transform, or choose pca or umap for a sequence.')

        flat, height, width = _as_flat(field)
        selection = _as_flat_mask(mask, len(flat))

        out = np.zeros((len(flat), self.n_components), dtype=np.float32)
        if selection is not None:
            if selection.any():
                out[selection] = self._reducer.transform(flat[selection]).astype(np.float32)
        else:
            out = self._reducer.transform(flat).astype(np.float32)

        return out.reshape(height, width, self.n_components)

    def fit_transform(self, field: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Fits on a field and reduces it in one step.

        :param field: An ``(H, W, C)`` field.
        :param mask: An ``(H, W)`` boolean mask (optional).
        :return: An ``(H, W, n_components)`` float32 array.
        """
        flat, height, width = _as_flat(field)
        selection = _as_flat_mask(mask, len(flat))
        data = flat[selection] if selection is not None else flat

        if len(data) == 0:
            raise ValueError('Nothing to fit on: the mask selected no pixels')

        self._reducer = _make_reducer(self.method, self.n_components, **self._kwargs)
        reduced = self._reducer.fit_transform(data).astype(np.float32)
        self._range = (reduced.min(axis=0), reduced.max(axis=0))

        out = np.zeros((len(flat), self.n_components), dtype=np.float32)
        if selection is not None:
            out[selection] = reduced
        else:
            out = reduced
        return out.reshape(height, width, self.n_components)

    def to_image(self, reduced: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        """
        Scales a reduced field into a viewable 8-bit image.

        The scaling uses the range recorded at fit time, not this field's own range, so every
        frame reduced by the same reducer shares one colour mapping. Rescaling per frame is
        the second reason sequences flicker, after refitting per frame.

        :param reduced: An ``(H, W, 3)`` reduced field.
        :param mask: An ``(H, W)`` boolean mask (optional). Masked-out pixels stay black.
        :return: An ``(H, W, 3)`` uint8 RGB image.
        """
        arr = np.asarray(reduced, dtype=np.float32)
        low, high = self._range if self._range is not None else (
            arr.min(axis=(0, 1)), arr.max(axis=(0, 1)))

        span = np.where(np.asarray(high) > np.asarray(low),
                        np.asarray(high) - np.asarray(low), 1.0)
        scaled = np.clip((arr - low) / span, 0.0, 1.0)

        out = (scaled * 255).astype(np.uint8)
        if mask is not None:
            out[~np.asarray(mask).astype(bool)] = 0
        return out


def reduce_to_rgb(field: NDArray, method: str = 'pca', mask: Optional[NDArray] = None,
                  reducer: Optional[FieldReducer] = None,
                  output_file: Optional[str] = None) -> NDArray:
    """
    Reduces an N-channel field to a viewable false-colour RGB image.

    :param field: An ``(H, W, C)`` field, or a path to a ``.npy`` holding one.
    :param method: ``"pca"``, ``"umap"`` or ``"tsne"``. Ignored when ``reducer`` is given.
    :param mask: An ``(H, W)`` boolean mask of valid pixels (optional). Pass the coverage
        mask: uncovered pixels are identical zeros and bias the fit.
    :param reducer: An already fitted reducer to reuse (optional). Pass one to give a whole
        sequence a single, stable colour mapping.
    :param output_file: Where to write a PNG (optional).
    :return: An ``(H, W, 3)`` uint8 RGB image.
    """
    if isinstance(field, str):
        field = np.load(field, mmap_mode='r')

    if reducer is None:
        reducer = FieldReducer(method=method, n_components=3)
        reduced = reducer.fit_transform(field, mask=mask)
    else:
        reduced = reducer.transform(field, mask=mask)

    image = reducer.to_image(reduced, mask=mask)

    if output_file is not None:
        import cv2

        cv2.imwrite(output_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return image


def reduce_to_2d(field: NDArray, method: str = 'pca', mask: Optional[NDArray] = None,
                 reducer: Optional[FieldReducer] = None,
                 output_file: Optional[str] = None) -> NDArray:
    """
    Reduces an N-channel field to two components.

    :param field: An ``(H, W, C)`` field, or a path to a ``.npy`` holding one.
    :param method: ``"pca"``, ``"umap"`` or ``"tsne"``. Ignored when ``reducer`` is given.
    :param mask: An ``(H, W)`` boolean mask of valid pixels (optional).
    :param reducer: An already fitted reducer to reuse (optional).
    :param output_file: Where to write the result as ``.npy`` (optional).
    :return: An ``(H, W, 2)`` float32 array.
    """
    if isinstance(field, str):
        field = np.load(field, mmap_mode='r')

    if reducer is None:
        reducer = FieldReducer(method=method, n_components=2)
        reduced = reducer.fit_transform(field, mask=mask)
    else:
        reduced = reducer.transform(field, mask=mask)

    if output_file is not None:
        np.save(output_file, reduced)
    return reduced
