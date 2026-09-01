from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any, Iterator

import cv2
import numpy as np
from numpy.typing import NDArray

from alfspy.core.geo import Transform

f4_type = np.dtype('f4')


@dataclass(frozen=True)
class Resolution:
    """
    Describes a 2D resolution using whole numbers for width and height.
    :cvar width: The width of the resolution.
    :cvar height: The height of the resolution.
    """
    width: int
    height: int

    def __iter__(self) -> Iterator[int]:
        return iter(self.as_tuple())

    def __getitem__(self, key: int) -> int:
        return self.as_tuple()[key]

    def as_tuple(self) -> tuple[int, int]:
        return self.width, self.height


@dataclass
class MeshData:
    """
    Class that represents the most basic information of a mesh for rendering.
    :cvar vertices: The vertices of the mesh as a numpy array.
    :cvar indices: The indices of the mesh as a numpy array (optional).
    :cvar uvs: The uvs coordinates of the vertices (optional).
    """
    vertices: NDArray
    indices: Optional[NDArray] = None
    uvs: Optional[NDArray] = None
    transform: Optional[Transform] = None


@dataclass
class TextureData:
    """
    Class that represents a texture.
    :cvar texture: The texture data as a BGR or BGRA numpy array.
    """
    texture: NDArray

    def to_bytes(self) -> bytes:
        """
        Returns a byte representation of the held texture. Ensures percentage channel values.
        :return: Bytes representing the texture.
        """
        img = self.texture
        if img.max(initial=0.0) > 1.0:
            img = self.texture / 255.0
        img = img[::-1, ...]  # flip image vertically into GL bottom-up order
        return img.astype('f4').tobytes()

    def to_tensor(self, device: Any = 'cpu', dtype: Any = None) -> 'torch.Tensor':  # noqa: F821
        """
        Returns the held texture as a tensor in the same layout ``to_bytes`` produces:
        vertically flipped into GL bottom-up order with channel values in ``[0, 1]``.

        torch is imported here rather than at module scope so this module -- which holds the
        backend-agnostic data classes every backend shares -- stays importable without torch
        installed. Only the torch backend calls this.

        :param device: The device to place the tensor on (defaults to the CPU).
        :param dtype: The floating point dtype to use (defaults to ``torch.float32``).
        :return: A ``(H, W, C)`` tensor representing the texture.
        """
        import torch

        from alfspy.core.torchgl import as_tensor

        if dtype is None:
            dtype = torch.float32
        tensor = as_tensor(self.texture, device=torch.device(device), dtype=dtype)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        if tensor.numel() and float(tensor.max()) > 1.0:
            tensor = tensor / 255.0
        return torch.flip(tensor, dims=(0,)).contiguous()

    @property
    def width(self) -> int:
        return self.texture.shape[1]

    @property
    def height(self) -> int:
        return self.texture.shape[0]

    def tex_gen_input(self) -> tuple[tuple[int, int], int, bytes]:
        """
        Returns a tuple representing the required input for creating a texture object.
        Retained from the ModernGL version because lazy shot loading caches this tuple
        ahead of upload; see ``AsyncShotLoader``.
        :return: Returns a tuple containing size, component count and a byte representation of the given texture.
        """
        return self.texture.shape[1::-1], self.texture.shape[2], self.to_bytes()

    def byte_size(self, dtype: Any = None) -> int:
        """
        Computes the size of the texture held by this object assuming each value will be encoded as the given dtype.
        :param dtype: The type the color values of the texture will be expressed with (defaults to texture type)
        :return: The byte size of the texture held by this object.
        """
        if dtype is None:
            dtype = self.texture.dtype
        w, h = self.texture.shape[1::-1]
        c = self.texture.shape[-1] if len(self.texture.shape) >= 3 else 1
        return w * h * c * np.dtype(dtype).itemsize

    def scale_to_fit(self, size: int, dtype: Any = None) -> None:
        """
        Scales the texture held by this object to fit the given size in bytes.
        This method only reduces the scale of textures.
        :param size: The amount of bytes to occupy.
        :param dtype: The dtype to be used for calculations (defaults to texture type).
        """
        if self.byte_size(dtype=dtype) < size:
            return

        if dtype is None:
            dtype = self.texture.dtype

        width, height = self.texture.shape[1::-1]
        channels = self.texture.shape[-1] if len(self.texture.shape) >= 3 else 1
        byte_depth = np.dtype(dtype).itemsize
        ratio = width/height
        n_height = np.sqrt(size/(ratio*channels*byte_depth))
        n_width = n_height * ratio
        self.texture = cv2.resize(self.texture, (int(n_height), int(n_width)))


class RenderResultMode(Enum):
    """
    Enumeration of render result format and content codes.
    :cvar Complete: The result includes the shot projection and the background object.
    :cvar ShotOnly: The result only shows the projected shot.
    """
    Complete = 0x00,
    ShotOnly = 0x01,

    def __str__(self):
        return self.name
