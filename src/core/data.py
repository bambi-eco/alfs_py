from typing import Optional

import cv2
import numpy as np
from attr import define, dataclass
from moderngl import VertexArray, Buffer, Texture, Program
from numpy.typing import NDArray


@define
class MeshData:
    """
    Class that represents the most basic information of a mesh for rendering
    :cvar vertices: The vertices of the mesh as a numpy array
    :cvar indices: The indices of the mesh as a numpy array (optional)
    :cvar uvs: The uvs coordinates of the vertices (optional)
    """

    vertices: NDArray
    indices: Optional[NDArray]
    uvs: Optional[NDArray]


@define
class TextureData:
    """
    Class that represents a texture
    :cvar texture: The texture data as a BGR or BGRA numpy array
    Methods:
        to_tex_bytes (bytes): Returns a byte representation of the held texture
        text_gen_input (tuple[tuple[int, int], int, bytes]):
    """

    texture: NDArray

    def to_bytes(self) -> bytes:
        """
        Returns a byte representation of the held texture in RGB or RGBA
        :return: Bytes representing the texture
        """
        code = cv2.COLOR_BGR2RGB if self.texture.shape[2] <= 3 else cv2.COLOR_BGRA2RGBA
        img = cv2.cvtColor(self.texture, code) / 255.0
        return img.astype('f4').tobytes()

    def tex_gen_input(self) -> tuple[tuple[int, int], int, bytes]:
        """
        Returns a tuple representing the required input for creating a ModernGL texture object via ``Context.texture``
        :return: Returns a tuple containing size, component count and a byte representation of the given texture
        """
        return self.texture.shape[1::-1], self.texture.shape[2], self.to_bytes()


@dataclass
class RenderObject:
    """
    Class that represents an object that has already been loaded into VRAM
    :cvar vao: The associated vertex array
    :cvar vao_content: A tuple describing the content of the VAO
    :cvar vbo: The associated buffer holding vertex data (position, color, uv, etc.)
    :cvar ibo: The associated buffer holding index data (nullable)
    :cvar tex: The associated texture buffer (nullable)
    """
    vao: VertexArray
    vao_content: list[tuple[Buffer, str, ...]]
    vbo: Buffer
    ibo: Optional[Buffer] = None
    tex: Optional[Texture] = None

    def tex_use(self) -> None:
        """
        Binds the texture to a texture unit
        """
        if self.tex is not None:
            self.tex.use()

    def render(self, mode: Optional[int] = None) -> None:
        """
        Renders everything contained within the vertex array
        :param mode: The drawing mode to be used (defaults to mgl.TRIANGLES)
        """
        self.vao.render(mode)

    def release(self) -> None:
        """
        Releases all resources associated with this object
        :return:
        """
        if self.tex is not None:
            self.tex.release()
            self.tex = None

        if self.ibo is not None:
            self.ibo.release()
            self.ibo = None

            self.vbo.release()
            self.vao.release()

    def _vao_content_for_prog(self, prog: Program, in_count) -> list[tuple[Buffer, str, ...]]:
        tup = self.vao_content[0]
        buf = tup[0]
        types = ' '.join(tup[1].split(' ')[0:in_count])
        in_vars = tup[2:2 + in_count]
        return [(buf, types, *in_vars)]

    def copy_for_prog(self, prog: Program, in_count: int) -> 'RenderObject':
        """
        Creates a shallow copy of this object, changing its shader to the given shader.
        Otherwise, all buffers and data are reused. Type order of previous program will also be reused
        :param prog: The shader program to be applied to the shallow copy
        :param in_count: The amount of input variables in the vertex shader
        :return: A shallow copy of this object with the given shader applied
        """
        vao_content_copy = self._vao_content_for_prog(prog, in_count)

        if self.ibo is not None:
            vao = prog.ctx.vertex_array(prog, vao_content_copy, index_buffer=self.ibo, index_element_size=4)
        else:
            vao = prog.ctx.vertex_array(prog, vao_content_copy)

        return RenderObject(vao, vao_content_copy, self.vbo, self.ibo, self.tex)

    @staticmethod
    def from_mesh(prog: Program, mesh: MeshData, texture: Optional[TextureData] = None,
                  vert_par: str = 'pos_in', uv_par: str = 'uv_in') -> 'RenderObject':
        """
        Takes mesh data and converts into a ``RenderObject`` using the provided shader and its context,
        loading all data into the buffers automatically
        :param prog: The shader program to attach all buffers to
        :param mesh: The mesh data of the object to convert
        :param texture: The texture data of the object to convert (optional)
        :param vert_par: The name of the vertex position variable within the vertex shader (defaults to ``'pos_in'``)
        :param uv_par: The name of the vertex uv coordinate variable within the vertex shader (defaults to ``'uv_in'``)
        :return: A ``RenderObject`` representing the given mesh data
        """
        ctx = prog.ctx
        vao_content = []
        if MeshData.uvs is not None:
            x, y, z = mesh.vertices[..., 0], mesh.vertices[..., 1], mesh.vertices[..., 2]
            u, v = mesh.uvs[..., 0], mesh.uvs[..., 1]
            shader_data = np.dstack([x, y, z, u, v])
            vbo = ctx.buffer(shader_data.tobytes())
            vao_content.append((vbo, '3f4 2f4', vert_par, uv_par))
        else:
            vbo = ctx.buffer(mesh.vertices.tobytes())
            vao_content.append((vbo, '3f4', vert_par))

        if MeshData.indices is not None:
            ibo = ctx.buffer(mesh.indices.tobytes())
            vao = ctx.vertex_array(prog, vao_content, index_buffer=ibo, index_element_size=4)
        else:
            ibo = None
            vao = ctx.vertex_array(prog, vao_content)

        if texture is not None:
            tex = ctx.texture(*texture.tex_gen_input(), dtype='f4')
        else:
            tex = None

        obj = RenderObject(vao, vao_content, vbo, ibo, tex)
        return obj
