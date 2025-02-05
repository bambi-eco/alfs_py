from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any, Iterator

import cv2
import numpy as np
from moderngl import VertexArray, Buffer, Texture, Program
from numpy.typing import NDArray

from pyrr import Matrix44

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
        img = img[::-1, ...]  # flip image vertically for moderngl
        return img.astype('f4').tobytes()

    @property
    def width(self) -> int:
        return self.texture.shape[1]

    @property
    def height(self) -> int:
        return self.texture.shape[0]

    def tex_gen_input(self) -> tuple[tuple[int, int], int, bytes]:
        """
        Returns a tuple representing the required input for creating a ModernGL texture object via ``Context.texture``.
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
