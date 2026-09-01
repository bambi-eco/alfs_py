"""Torch-resident mesh objects.

``RenderObject`` holds a mesh that has already been uploaded to the render device. It is
backend-specific by nature -- the ModernGL version wraps a VAO and GL buffers, this one wraps
tensors -- so it lives with its backend rather than beside the backend-agnostic
``MeshData``/``TextureData``/``Resolution`` in ``alfspy.core.rendering.data``.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from pyrr import Matrix44

from alfspy.core.geo import Transform
from alfspy.core.rendering.data import MeshData, TextureData, f4_type
from alfspy.core.torchgl import TorchContext, TorchTexture, as_tensor

__all__ = ['RenderObject']


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

