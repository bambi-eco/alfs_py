"""WebGPU-resident mesh objects."""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from pyrr import Matrix44

from alfspy.core.geo import Transform
from alfspy.core.rendering.data import MeshData, TextureData

__all__ = ['RenderObject']


@dataclass
class RenderObject:
    """
    A mesh uploaded to the GPU: vertex, uv and index buffers plus its texture.

    :cvar vertex_buffer: Vertex positions.
    :cvar uv_buffer: Texture coordinates.
    :cvar index_buffer: Triangle indices, always ``uint32``.
    :cvar index_count: How many indices to draw.
    :cvar texture: The mesh texture.
    :cvar transform: The model transform (optional).
    """
    vertex_buffer: Any
    uv_buffer: Any
    index_buffer: Any
    index_count: int
    texture: Any
    transform: Optional[Transform] = None

    def mat(self, dtype: Any = None) -> Matrix44:
        """
        :param dtype: The dtype for the matrix (optional).
        :return: The model matrix.
        """
        if self.transform is None:
            return Matrix44.identity(dtype='f4')
        return self.transform.mat(dtype)

    def release(self) -> None:
        """
        Drops the GPU resources. wgpu-py frees them when the last reference goes.
        """
        self.vertex_buffer = None
        self.uv_buffer = None
        self.index_buffer = None
        self.texture = None

    @staticmethod
    def from_mesh(ctx, mesh: MeshData, texture: Optional[TextureData],
                  texture_format: str) -> 'RenderObject':
        """
        Uploads mesh data.

        :param ctx: The WebGPU context.
        :param mesh: The mesh to upload.
        :param texture: Its texture (optional).
        :param texture_format: The texture format to store it in.
        :return: The uploaded object.
        """
        import wgpu

        device = ctx.device

        vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32).reshape(-1, 3)

        if mesh.uvs is not None:
            uvs = np.ascontiguousarray(mesh.uvs, dtype=np.float32).reshape(-1, 2)
        else:
            uvs = np.zeros((len(vertices), 2), dtype=np.float32)

        if mesh.indices is not None:
            indices = np.ascontiguousarray(mesh.indices, dtype=np.uint32).reshape(-1)
        else:
            indices = np.arange(len(vertices), dtype=np.uint32)

        def buffer(data, usage):
            buf = device.create_buffer(
                size=max(data.nbytes, 4), usage=usage | wgpu.BufferUsage.COPY_DST)
            device.queue.write_buffer(buf, 0, data.tobytes())
            return buf

        vertex_buffer = buffer(vertices, wgpu.BufferUsage.VERTEX)
        uv_buffer = buffer(uvs, wgpu.BufferUsage.VERTEX)
        index_buffer = buffer(indices, wgpu.BufferUsage.INDEX)

        if texture is None:
            image = np.ones((1, 1, 4), dtype=np.float32)
        else:
            image = np.asarray(texture.texture, dtype=np.float32)
            if image.ndim == 2:
                image = image[..., np.newaxis]
            if image.shape[2] < 4:
                pad = np.ones((*image.shape[:2], 4 - image.shape[2]), dtype=np.float32)
                image = np.concatenate([image, pad], axis=2)
            if texture.normalise and image.max(initial=0.0) > 1.0:
                image = image / 255.0
            # Uploaded bottom-up, matching TextureData.to_bytes; the shader samples 1 - v.
            image = image[::-1, ...]

        dtype = np.float32 if texture_format.endswith('32float') else np.float16
        payload = np.ascontiguousarray(image, dtype=dtype)
        height, width = payload.shape[0], payload.shape[1]

        gpu_texture = device.create_texture(
            size=(width, height, 1), format=texture_format,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        device.queue.write_texture(
            {'texture': gpu_texture},
            payload.tobytes(),
            {'bytes_per_row': width * 4 * payload.dtype.itemsize, 'rows_per_image': height},
            (width, height, 1),
        )

        return RenderObject(
            vertex_buffer=vertex_buffer,
            uv_buffer=uv_buffer,
            index_buffer=index_buffer,
            index_count=len(indices),
            texture=gpu_texture,
            transform=mesh.transform,
        )
