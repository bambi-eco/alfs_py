"""ModernGL-resident mesh objects.

``RenderObject`` was previously defined in ``alfspy.core.rendering.data`` alongside the
backend-agnostic ``MeshData``/``TextureData``/``Resolution``. It is the only class in that
module that was ever ModernGL-specific -- it holds a VAO, buffers and a texture -- so it
lives with the backend that owns those handles. The shared data classes stay where they were.
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from moderngl import Buffer, Program, Texture, VertexArray
from pyrr import Matrix44

from alfspy.core.geo import Transform
from alfspy.core.rendering.data import MeshData, TextureData

f4_type = np.dtype('f4')


@dataclass
class RenderObject:
    """
    Class that represents an object that has already been loaded into VRAM.
    :cvar vao: The associated vertex array.
    :cvar vao_content: A tuple describing the content of the VAO.
    :cvar vertex_buf: The associated buffer holding vertex positions.
    :cvar uv_buf: The associated buffer holding vertex uv coordinates.
    :cvar ibo: The associated buffer holding index data (optional).
    :cvar tex: The associated texture buffer (optional).
    """
    vao: VertexArray
    vao_content: list[tuple[Buffer, str, ...]]
    vertex_buf: Buffer
    uv_buf: Optional[Buffer] = None
    ibo: Optional[Buffer] = None
    tex: Optional[Texture] = None
    transform: Optional[Transform] = None

    def tex_use(self, location: int = 0) -> None:
        """
        Binds the texture of this object to a texture unit
        """
        if self.tex is not None:
            self.tex.use(location)

    def mat(self, dtype: Any = None) -> Matrix44:
        if self.transform is None:
            return Matrix44.identity(dtype='f4')
        else:
            return self.transform.mat(dtype)

    def render(self, mode: Optional[int] = None) -> None:
        """
        Renders everything contained within the vertex array.
        :param mode: The drawing mode to be used (defaults to mgl.TRIANGLES).
        """
        self.vao.render(mode)

    def release(self) -> None:
        """
        Releases all resources associated with this object.
        :return:
        """
        if self.tex is not None:
            self.tex.release()
            self.tex = None

        if self.ibo is not None:
            self.ibo.release()
            self.ibo = None

            self.vertex_buf.release()
            self.vao.release()

    @staticmethod
    def from_mesh(prog: Program, mesh: MeshData, texture: Optional[TextureData] = None,
                  vert_par: str = 'pos_in', uv_par: str = 'uv_in') -> 'RenderObject':
        """
        Takes mesh data and converts into a ``RenderObject`` using the provided shader and its context,
        loading all data into the buffers automatically.
        :param prog: The shader program to attach all buffers to.
        :param mesh: The mesh data of the object to convert.
        :param texture: The texture data of the object to convert (optional).
        :param vert_par: The name of the vertex position variable within the vertex shader (defaults to ``'pos_in'``).
        :param uv_par: The name of the vertex uv coordinate variable within the vertex shader (defaults to ``'uv_in'``).
        :return: A ``RenderObject`` representing the given mesh data.
        """
        ctx = prog.ctx
        vao_content = []

        vertices = mesh.vertices
        if vertices.dtype != f4_type:
            vertices = vertices.astype(f4_type)
        vertex_buf = ctx.buffer(vertices.tobytes())
        vao_content.append((vertex_buf, '3f4', vert_par))


        if mesh.uvs is not None:
            uvs = mesh.uvs
            if uvs.dtype != f4_type:
                uvs = uvs.astype(f4_type)
            uv_buf = ctx.buffer(uvs.tobytes())
            vao_content.append((uv_buf, '2f4', uv_par))
        else:
            uv_buf = None

        if mesh.indices is not None:
            indices = mesh.indices

            if not np.issubdtype(indices.dtype, np.unsignedinteger):
                raise TypeError(f'Mesh indices must be unsigned integers but are {indices.dtype.name}')

            index_element_size = indices.dtype.itemsize
            if index_element_size not in (1, 2, 4):
                raise ValueError('Mesh indices must be either 1, 2, or 4 bytes in size.')

            ibo = ctx.buffer(indices.tobytes())
            vao = ctx.vertex_array(prog, vao_content, index_buffer=ibo, index_element_size=4)
        else:
            ibo = None
            vao = ctx.vertex_array(prog, vao_content)

        if texture is not None:
            tex_input = texture.tex_gen_input()
            tex = ctx.texture(*tex_input, dtype='f4')  # TODO: throws exception when texture is too big -> cpp issue of moderngl
        else:
            tex = None

        obj = RenderObject(vao, vao_content, vertex_buf, uv_buf, ibo, tex)
        return obj

