from typing import Optional

import cv2
import moderngl as mgl
import numpy as np
from moderngl import Texture
from numpy.typing import NDArray
from pyrr import Matrix44

from alfspy.core.rendering.shot import CtxShot


class EmbeddingCtxShot:
    """
    Wraps a ``CtxShot`` and replaces its raw-image texture with a DINOv3 embedding
    slice served as a 4-channel float32 GPU texture.

    For each multi-pass rendering iteration, call ``set_channel_slice(start, end)``
    to select which 4 embedding dimensions should be bound as the active texture.
    The geometric properties (camera, projection, correction) are delegated to the
    underlying ``CtxShot``.
    """

    def __init__(self, wrapped: CtxShot, embedding: NDArray, ctx: mgl.Context,
                 render_width: int, render_height: int):
        """
        :param wrapped: The original ``CtxShot`` providing camera geometry.
        :param embedding: Dense patch embedding of shape (H_e, W_e, embed_dim) float32.
        :param ctx: The ModernGL context used to upload textures.
        :param render_width: Target texture width (render resolution width).
        :param render_height: Target texture height (render resolution height).
        """
        self._wrapped = wrapped
        self._embedding = embedding.astype(np.float16)  # (H_e, W_e, embed_dim) — float16 halves RAM
        self._ctx = ctx
        self._render_w = render_width
        self._render_h = render_height
        self._active_tex: Optional[Texture] = None

    # ------------------------------------------------------------------
    # Texture management
    # ------------------------------------------------------------------

    def set_channel_slice(self, start: int, end: int) -> None:
        """
        Prepares the GPU texture for the embedding channels [start:end].

        Up to 4 channels are placed in RGBA; if ``end - start < 4`` the remaining
        channels are zero-padded. The slice is bilinearly upsampled to the render
        resolution so the existing shader UV-sampling logic works unchanged.

        :param start: First embedding dimension index (inclusive).
        :param end: Last embedding dimension index (exclusive). ``end - start <= 4``.
        """
        if self._active_tex is not None:
            self._active_tex.release()
            self._active_tex = None

        n_channels = end - start
        slice_data = self._embedding[:, :, start:end]  # (H_e, W_e, n_channels)

        # Pad to exactly 4 channels
        if n_channels < 4:
            pad = np.zeros((*slice_data.shape[:2], 4 - n_channels), dtype=np.float32)
            slice_data = np.concatenate([slice_data, pad], axis=2)

        slice_data = slice_data.astype(np.float32)

        # Upsample to render resolution using bilinear interpolation
        if slice_data.shape[0] != self._render_h or slice_data.shape[1] != self._render_w:
            # cv2.resize works on (H, W, C); handles multi-channel fine
            slice_data = cv2.resize(slice_data, (self._render_w, self._render_h),
                                    interpolation=cv2.INTER_LINEAR)

        # Upload to GPU: shape is (H, W, 4) → bytes in row-major order
        tex_data = np.ascontiguousarray(slice_data)
        self._active_tex = self._ctx.texture(
            (self._render_w, self._render_h), 4,
            tex_data.tobytes(), dtype='f4'
        )

    def tex_use(self, location: int = 0) -> None:
        """Binds the currently active embedding-slice texture to the given texture unit."""
        if self._active_tex is None:
            raise RuntimeError('Call set_channel_slice() before tex_use()')
        self._active_tex.use(location)

    def release(self) -> None:
        """Releases the GPU texture and the underlying CtxShot."""
        if self._active_tex is not None:
            self._active_tex.release()
            self._active_tex = None
        self._wrapped.release()

    # ------------------------------------------------------------------
    # Geometry delegation (unchanged from CtxShot)
    # ------------------------------------------------------------------

    @property
    def camera(self):
        return self._wrapped.camera

    @property
    def correction(self):
        return self._wrapped.correction

    @correction.setter
    def correction(self, value):
        self._wrapped.correction = value

    def get_proj(self) -> Matrix44:
        return self._wrapped.get_proj()

    def get_view(self) -> Matrix44:
        return self._wrapped.get_view()

    def get_correction(self) -> Matrix44:
        return self._wrapped.get_correction()

    def get_mat(self) -> Matrix44:
        return self._wrapped.get_mat()
