"""GPU texture replacement backed by a torch tensor.

Storage convention matches what the ModernGL path uploaded: images are stored **bottom-up**
(row 0 is ``v = 0``) and normalised to ``[0, 1]``. Keeping that convention means UV
semantics are byte-for-byte identical to the GL version and every ``[::-1]`` flip elsewhere
in the codebase stays correct.
"""

from typing import Final, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from alfspy.core.torchgl.compat import as_tensor

__all__ = ['TorchTexture', 'sample_texture']

_FILTERS: Final[frozenset] = frozenset({'linear', 'nearest'})
_WRAPS: Final[frozenset] = frozenset({'repeat', 'clamp'})


def sample_texture(tex: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
                   filter_mode: str = 'linear', wrap: str = 'repeat') -> torch.Tensor:
    """
    Samples a texture at the given UV coordinates, reproducing GLSL ``texture()``.

    :param tex: The texture as an ``(H, W, C)`` tensor stored bottom-up.
    :param u: Flat tensor of U coordinates.
    :param v: Flat tensor of V coordinates (``0`` at the bottom, as in GL).
    :param filter_mode: ``'linear'`` (GL_LINEAR) or ``'nearest'`` (GL_NEAREST).
    :param wrap: ``'repeat'`` (GL_REPEAT) or ``'clamp'`` (GL_CLAMP_TO_EDGE).
    :return: An ``(N, C)`` tensor of sampled values.
    """
    if filter_mode not in _FILTERS:
        raise ValueError(f'Unknown filter mode "{filter_mode}"; expected one of {sorted(_FILTERS)}')
    if wrap not in _WRAPS:
        raise ValueError(f'Unknown wrap mode "{wrap}"; expected one of {sorted(_WRAPS)}')

    count = u.numel()
    if count == 0:
        return torch.zeros((0, tex.shape[-1]), device=tex.device, dtype=tex.dtype)

    if wrap == 'repeat':
        # GL_REPEAT: fract(uv). Exact, unlike any grid_sample padding_mode.
        u = u - torch.floor(u)
        v = v - torch.floor(v)
    else:
        u = u.clamp(0.0, 1.0)
        v = v.clamp(0.0, 1.0)

    # (H, W, C) -> (1, C, H, W) for grid_sample.
    nchw = tex.permute(2, 0, 1).unsqueeze(0)

    # align_corners=False maps grid -1 to the outer edge of the first texel, matching GL's
    # normalised texture coordinates exactly.
    grid = torch.stack((u * 2.0 - 1.0, v * 2.0 - 1.0), dim=-1).reshape(1, 1, count, 2)
    grid = grid.to(dtype=nchw.dtype)

    mode = 'bilinear' if filter_mode == 'linear' else 'nearest'
    sampled = F.grid_sample(nchw, grid, mode=mode, padding_mode='border', align_corners=False)
    return sampled.reshape(tex.shape[-1], count).transpose(0, 1)


class TorchTexture:
    """
    A texture living on a torch device.

    Mirrors the slice of ``moderngl.Texture`` this project uses: construction from raw image
    data, binding to a "texture unit" (bookkeeping only), and explicit release.
    """

    __slots__ = ('_data', '_released', 'filter_mode', 'wrap')

    def __init__(self, data: Union[NDArray, torch.Tensor], device: torch.device,
                 dtype: torch.dtype = torch.float32, flip: bool = True,
                 normalize: Optional[bool] = None,
                 filter_mode: str = 'linear', wrap: str = 'repeat') -> None:
        """
        Initialises a new ``TorchTexture``.

        :param data: The image as an ``(H, W)`` or ``(H, W, C)`` array, top-down.
        :param device: The device to place the texture on.
        :param dtype: The floating point dtype to store the texture with.
        :param flip: Whether to flip the image vertically into GL bottom-up order
            (defaults to ``True``, matching ``TextureData.to_bytes``).
        :param normalize: Whether to divide by 255 (optional). When ``None`` the texture is
            normalised only if its maximum exceeds ``1.0``, mirroring ``TextureData.to_bytes``.
        :param filter_mode: ``'linear'`` or ``'nearest'``.
        :param wrap: ``'repeat'`` or ``'clamp'``.
        """
        tensor = as_tensor(data, device=device, dtype=dtype)

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        elif tensor.ndim != 3:
            raise ValueError(f'Texture data must be 2D or 3D but has shape {tuple(tensor.shape)}')

        if normalize is None:
            normalize = bool(tensor.numel()) and float(tensor.max()) > 1.0
        if normalize:
            tensor = tensor / 255.0

        if flip:
            tensor = torch.flip(tensor, dims=(0,))

        self._data = tensor.contiguous()
        self._released = False
        self.filter_mode = filter_mode
        self.wrap = wrap

    # region properties

    @property
    def data(self) -> torch.Tensor:
        """
        :return: The underlying ``(H, W, C)`` tensor in bottom-up GL order.
        """
        if self._released:
            raise RuntimeError('Texture has been released')
        return self._data

    @property
    def size(self) -> tuple[int, int]:
        """
        :return: The texture size as ``(width, height)``.
        """
        return int(self._data.shape[1]), int(self._data.shape[0])

    @property
    def components(self) -> int:
        """
        :return: The number of colour components per texel.
        """
        return int(self._data.shape[2])

    @property
    def released(self) -> bool:
        """
        :return: Whether this texture has been released.
        """
        return self._released

    @property
    def device(self) -> torch.device:
        """
        :return: The device this texture lives on.
        """
        return self._data.device

    # endregion

    def use(self, location: int = 0) -> None:
        """
        Binds the texture to a texture unit.

        Kept for API compatibility with ``moderngl.Texture.use``; the torch backend passes
        textures explicitly, so this only validates that the texture is still alive.

        :param location: The texture unit index (unused).
        """
        if self._released:
            raise RuntimeError('Texture has been released')

    def sample(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Samples this texture, reproducing GLSL ``texture()``.

        :param u: Flat tensor of U coordinates.
        :param v: Flat tensor of V coordinates.
        :return: An ``(N, C)`` tensor of sampled values.
        """
        return sample_texture(self.data, u, v, self.filter_mode, self.wrap)

    def release(self) -> None:
        """
        Releases the memory held by this texture.
        """
        if not self._released:
            self._data = torch.empty(0)
            self._released = True

    def to_numpy(self, flip_back: bool = True) -> NDArray:
        """
        Returns the texture as a numpy array.

        :param flip_back: Whether to flip back into top-down image order (defaults to ``True``).
        :return: An ``(H, W, C)`` numpy array.
        """
        arr = self.data.detach().to('cpu').numpy()
        return np.flip(arr, axis=0).copy() if flip_back else arr.copy()

    def __repr__(self) -> str:
        if self._released:
            return '<TorchTexture released>'
        width, height = self.size
        return f'<TorchTexture {width}x{height}x{self.components} {self._data.dtype} on {self._data.device}>'
