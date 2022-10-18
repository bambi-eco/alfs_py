from typing import Optional

import cv2
from attr import define, dataclass
from moderngl import VertexArray, Buffer, Texture
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
    :cvar vbo: The associated buffer holding vertex data (position, color, uv, etc.)
    :cvar ibo: The associated buffer holding index data (nullable)
    :cvar tex: The associated texture buffer (nullable)
    """
    vao: VertexArray
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

