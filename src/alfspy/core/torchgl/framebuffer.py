"""Render target replacement backed by a torch tensor.

Like the ModernGL framebuffer it replaces, pixel row ``0`` is the **bottom** of the image.
Every ``[::-1]`` flip in the surrounding codebase therefore stays correct and unchanged.

Unlike the ModernGL version there is no driver-owned surface: "clearing" is
:meth:`torch.Tensor.zero_` on memory this object exclusively owns, which is the direct fix
for the stale-buffer artifacts documented in ``docs/MIGRATION_REVIEW.md`` §3.
"""

from typing import Iterable, Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray

__all__ = ['TorchFramebuffer']

_DTYPE_ALIASES = {
    'f1': np.uint8, 'u1': np.uint8,
    'f2': np.float16,
    'f4': np.float32, 'f': np.float32,
    'f8': np.float64,
    'i4': np.int32, 'u4': np.uint32,
}


class TorchFramebuffer:
    """
    An RGBA render target held as an ``(H, W, 4)`` float tensor in GL bottom-up order.
    """

    __slots__ = ('_data', '_released', '_width', '_height', '_components')

    def __init__(self, size: tuple[int, int], device: torch.device,
                 components: int = 4, dtype: torch.dtype = torch.float32) -> None:
        """
        Initialises a new ``TorchFramebuffer``.

        :param size: The size as ``(width, height)``, matching ``Context.simple_framebuffer``.
        :param device: The device to allocate the target on.
        :param components: The number of colour components (defaults to ``4``).
        :param dtype: The floating point dtype of the target.
        """
        width, height = int(size[0]), int(size[1])
        if width <= 0 or height <= 0:
            raise ValueError(f'Framebuffer size must be positive but is {(width, height)}')

        self._width = width
        self._height = height
        self._components = int(components)
        self._data = torch.zeros((height, width, self._components), device=device, dtype=dtype)
        self._released = False

    # region properties

    @property
    def data(self) -> torch.Tensor:
        """
        :return: The underlying ``(H, W, C)`` tensor in GL bottom-up order.
        """
        if self._released:
            raise RuntimeError('Framebuffer has been released')
        return self._data

    @property
    def size(self) -> tuple[int, int]:
        """
        :return: The framebuffer size as ``(width, height)``, matching ``moderngl.Framebuffer.size``.
        """
        return self._width, self._height

    @property
    def components(self) -> int:
        """
        :return: The number of colour components per pixel.
        """
        return self._components

    @property
    def device(self) -> torch.device:
        """
        :return: The device this framebuffer lives on.
        """
        return self._data.device

    @property
    def dtype(self) -> torch.dtype:
        """
        :return: The dtype of the underlying tensor.
        """
        return self._data.dtype

    @property
    def released(self) -> bool:
        """
        :return: Whether this framebuffer has been released.
        """
        return self._released

    # endregion

    def use(self) -> None:
        """
        Binds this framebuffer as the active render target.

        Kept for API compatibility with ``moderngl.Framebuffer.use``. The torch backend has no
        implicit global binding -- render calls take their target explicitly -- so this only
        checks liveness. That absence of hidden global state is precisely what removes the
        Xvfb artifact class.
        """
        if self._released:
            raise RuntimeError('Framebuffer has been released')

    def clear(self, color: Optional[Union[float, Iterable[float]]] = None,
              *extra: float) -> None:
        """
        Clears the framebuffer.

        Accepts the same shapes as ``moderngl.Framebuffer.clear``: no argument, a single
        scalar, an iterable of components, or components as positional arguments.

        :param color: The clear colour (optional; defaults to fully transparent black).
        :param extra: Additional colour components when passed positionally.
        """
        data = self.data
        if color is None and not extra:
            data.zero_()
            return

        if isinstance(color, (int, float)):
            values = [float(color), *(float(v) for v in extra)]
        else:
            values = [float(v) for v in color]  # type: ignore[union-attr]
            values.extend(float(v) for v in extra)

        if len(values) == 1:
            data.fill_(values[0])
            return

        if len(values) < self._components:
            values = values + [0.0] * (self._components - len(values))
        data[...] = torch.tensor(values[:self._components], device=data.device, dtype=data.dtype)

    def read_array(self, dtype: Union[str, np.dtype] = 'f1', clamp: Optional[bool] = None) -> NDArray:
        """
        Reads the framebuffer contents as a numpy array in GL bottom-up order.

        :param dtype: The target dtype, accepting ModernGL-style codes (``'f1'``, ``'f4'``).
            ``'f1'`` reproduces GL's normalised-to-byte readback: values are clamped to
            ``[0, 1]`` and scaled by 255.
        :param clamp: Whether to clamp values to ``[0, 1]`` (optional). Defaults to ``True``
            for ``'f1'`` and ``False`` otherwise, mirroring ``moderngl.Framebuffer.read``.
        :return: An ``(H, W, C)`` numpy array.
        """
        np_dtype = np.dtype(_DTYPE_ALIASES.get(str(dtype), dtype))
        data = self.data

        byte_like = np_dtype == np.uint8
        if clamp is None:
            clamp = byte_like

        if clamp:
            data = data.clamp(0.0, 1.0)

        if byte_like:
            # GL rounds to nearest when converting normalised floats to bytes.
            data = torch.round(data * 255.0).clamp(0.0, 255.0)

        return data.detach().to('cpu').numpy().astype(np_dtype, copy=False)

    def read(self, components: int = 4, attachment: int = 0,
             dtype: Union[str, np.dtype] = 'f1', clamp: Optional[bool] = None) -> bytes:
        """
        Reads the framebuffer contents as raw bytes, matching ``moderngl.Framebuffer.read``.

        Provided for API compatibility. Internal code should prefer :meth:`read_array`, which
        avoids the byte round-trip.

        :param components: The number of components to read.
        :param attachment: The colour attachment index (only ``0`` exists here).
        :param dtype: The target dtype as a ModernGL-style code.
        :param clamp: Whether to clamp values to ``[0, 1]`` (optional).
        :return: The raw bytes of the framebuffer contents.
        """
        if attachment != 0:
            raise ValueError(f'TorchFramebuffer has a single attachment but {attachment} was requested')
        arr = self.read_array(dtype=dtype, clamp=clamp)
        return np.ascontiguousarray(arr[..., :components]).tobytes()

    def release(self) -> None:
        """
        Releases the memory held by this framebuffer.
        """
        if not self._released:
            self._data = torch.empty(0)
            self._released = True

    def __repr__(self) -> str:
        if self._released:
            return '<TorchFramebuffer released>'
        return (f'<TorchFramebuffer {self._width}x{self._height}x{self._components} '
                f'{self._data.dtype} on {self._data.device}>')
