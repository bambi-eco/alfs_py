from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any, Iterator

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from pyrr import Matrix44

from alfspy.core.geo import Transform
from alfspy.core.torchgl import TorchContext, TorchTexture, as_tensor

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

    def to_tensor(self, device: Any = 'cpu', dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Returns the held texture as a tensor in the same layout ``to_bytes`` produces:
        vertically flipped into GL bottom-up order with channel values in ``[0, 1]``.
        :param device: The device to place the tensor on (defaults to the CPU).
        :param dtype: The floating point dtype to use (defaults to ``torch.float32``).
        :return: A ``(H, W, C)`` tensor representing the texture.
        """
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


@dataclass
class RenderObject:
    """
    Class that represents a mesh that has been uploaded to the render device.

    The ModernGL version wrapped a VAO plus its vertex/uv/index buffers. The torch backend
    has no vertex-array concept, so the same fields now hold plain tensors.

    :cvar vertices: ``(V, 3)`` vertex position tensor.
    :cvar indices: ``(T, 3)`` triangle index tensor (optional). When ``None`` vertices are
        consumed as consecutive triples, matching ``GL_TRIANGLES`` without an index buffer.
    :cvar uvs: ``(V, 2)`` vertex UV tensor (optional).
    :cvar tex: The associated texture (optional).
    :cvar transform: The model transform of this object (optional).
    """
    vertices: torch.Tensor
    indices: Optional[torch.Tensor] = None
    uvs: Optional[torch.Tensor] = None
    tex: Optional[TorchTexture] = None
    transform: Optional[Transform] = None

    @property
    def triangles(self) -> torch.Tensor:
        """
        :return: A ``(T, 3)`` index tensor, synthesised for non-indexed meshes.
        """
        if self.indices is not None:
            return self.indices
        count = self.vertices.shape[0] // 3
        return torch.arange(count * 3, device=self.vertices.device, dtype=torch.int64).reshape(count, 3)

    def tex_use(self, location: int = 0) -> None:
        """
        Binds the texture of this object to a texture unit.
        """
        if self.tex is not None:
            self.tex.use(location)

    def mat(self, dtype: Any = None) -> Matrix44:
        if self.transform is None:
            return Matrix44.identity(dtype='f4')
        else:
            return self.transform.mat(dtype)

    def release(self) -> None:
        """
        Releases all resources associated with this object.
        """
        if self.tex is not None:
            self.tex.release()
            self.tex = None

        self.indices = None
        self.uvs = None
        self.vertices = torch.empty(0)

    @staticmethod
    def from_mesh(ctx: TorchContext, mesh: MeshData,
                  texture: Optional[TextureData] = None) -> 'RenderObject':
        """
        Takes mesh data and uploads it to the render device.

        The ModernGL signature took a ``Program`` plus attribute names in order to build a
        VAO; those arguments have no meaning in the torch backend and were dropped. This is
        a backend-internal type -- ``Renderer`` is the public surface and is unchanged.

        :param ctx: The context whose device the mesh should be uploaded to.
        :param mesh: The mesh data of the object to convert.
        :param texture: The texture data of the object to convert (optional).
        :return: A ``RenderObject`` representing the given mesh data.
        """
        device = ctx.device
        dtype = ctx.dtype

        vertices = as_tensor(np.ascontiguousarray(mesh.vertices, dtype=f4_type), device, dtype)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f'Mesh vertices must have shape (V, 3) but have {tuple(vertices.shape)}')

        if mesh.uvs is not None:
            uvs = as_tensor(np.ascontiguousarray(mesh.uvs, dtype=f4_type), device, dtype)
        else:
            uvs = None

        if mesh.indices is not None:
            raw_indices = mesh.indices

            if not np.issubdtype(raw_indices.dtype, np.unsignedinteger):
                raise TypeError(f'Mesh indices must be unsigned integers but are {raw_indices.dtype.name}')

            index_element_size = raw_indices.dtype.itemsize
            if index_element_size not in (1, 2, 4):
                raise ValueError('Mesh indices must be either 1, 2, or 4 bytes in size.')

            indices = as_tensor(
                np.ascontiguousarray(raw_indices, dtype=np.int64).reshape(-1, 3), device, torch.int64
            )
        else:
            indices = None

        tex = TorchTexture(texture.texture, device=device, dtype=dtype) if texture is not None else None

        return RenderObject(vertices, indices, uvs, tex, mesh.transform)


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
