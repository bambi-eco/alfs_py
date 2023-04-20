from dataclasses import dataclass
from typing import Optional, Any

import cv2
import numpy as np
from moderngl import VertexArray, Buffer, Texture, Program
from numpy.typing import NDArray
from pyrr import Vector3

from src.core.geo.transform import Transform


@dataclass
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


@dataclass
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
        Returns a byte representation of the held texture. Ensures percentage channel values
        :return: Bytes representing the texture
        """
        img = self.texture
        if img.max(initial=0.0) > 1.0:
            img = self.texture / 255.0
        img = img[::-1, ...]  # flip image vertically for moderngl
        return img.astype('f4').tobytes()

    def tex_gen_input(self) -> tuple[tuple[int, int], int, bytes]:
        """
        Returns a tuple representing the required input for creating a ModernGL texture object via ``Context.texture``
        :return: Returns a tuple containing size, component count and a byte representation of the given texture
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
        :param dtype: The dtype to be used for calculations (defaults to texture type)
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
    Class that represents an object that has already been loaded into VRAM
    :cvar vao: The associated vertex array
    :cvar vao_content: A tuple describing the content of the VAO
    :cvar vertex_buf: The associated buffer holding vertex positions
    :cvar uv_buf: The associated buffer holding vertex uv coordinates
    :cvar ibo: The associated buffer holding index data (nullable)
    :cvar tex: The associated texture buffer (nullable)
    """
    vao: VertexArray
    vao_content: list[tuple[Buffer, str, ...]]
    vertex_buf: Buffer
    uv_buf: Optional[Buffer] = None
    ibo: Optional[Buffer] = None
    tex: Optional[Texture] = None

    def tex_use(self) -> None:
        """
        Binds the texture of this object to a texture unit
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

            self.vertex_buf.release()
            self.vao.release()

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

        vertex_buf = ctx.buffer(mesh.vertices.tobytes())
        vao_content.append((vertex_buf, '3f4', vert_par))

        if mesh.uvs is not None:
            uv_buf = ctx.buffer(mesh.uvs.tobytes())
            vao_content.append((uv_buf, '2f4', uv_par))
        else:
            uv_buf = None

        if mesh.indices is not None:
            ibo = ctx.buffer(mesh.indices.tobytes())
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


@dataclass
class AABB:
    """
    Represents a 3D axis aligned bounding box
    :cvar p_s: The start corner of the bounding box
    :cvar p_e: The end corner of the bounding box
    """
    p_s: Vector3
    p_e: Vector3

    @property
    def width(self) -> float:
        """
        :return: The width / length in X direction of the AABB
        """
        return abs(self.p_e.x - self.p_s.x)

    @property
    def height(self) -> float:
        """
        :return: The height / length in Y direction of the AABB
        """
        return abs(self.p_e.y - self.p_s.y)

    @property
    def depth(self) -> float:
        """
        :return: The depth / length in Z direction of the AABB
        """
        return abs(self.p_e.z - self.p_s.z)

    @property
    def corners(self) -> tuple[Vector3, Vector3, Vector3, Vector3, Vector3, Vector3, Vector3, Vector3]:
        """
        :return: The corners of the AABB from front to back, bottom to top, and left to right
        """
        min_x, min_y, min_z = self.p_s
        max_x, max_y, max_z = self.p_e
        return Vector3([min_x, min_y, min_z]), Vector3([min_x, min_y, max_z]), \
            Vector3([min_x, max_y, min_z]), Vector3([min_x, max_y, max_z]), \
            Vector3([max_x, min_y, min_z]), Vector3([max_x, min_y, max_z]), \
            Vector3([max_x, max_y, min_z]), Vector3([max_x, max_y, max_z])


@dataclass
class ProjectionSettings:
    """
    Class storing settings used by the projection process
    :cvar count: The max amount of shots to be used
    :cvar initial_skip: The number of shots read from the JSON file to be skipped from the beginning
    :cvar skip: The number of shots to be skipped after every shot projection
    :cvar lazy: Whether the shots should be loaded lazy
    :cvar shot_centered_camera: Whether the camera should be moved to look at the center of the shots
    :cvar release_shots: Whether the renderer should release shots as soon as he used them
    :cvar correction: The correction to be applied to every single shot
    :cvar show_integral: Whether to show the resulting integral in a photo viewer
    :cvar output_file: The path and name of the output to be generated
    """
    count: int = 1
    initial_skip: int = 0
    skip: int = 1
    lazy: bool = True
    shot_centered_camera: bool = False
    resolution: tuple[int, int] = 1024, 1024
    ortho_size: Optional[tuple[float, float]] = None
    release_shots: bool = True
    correction: Optional[Transform] = None
    show_integral: bool = False
    output_file: str = ''

@dataclass
class FocusAnimationSettings:
    """
    Class storing settings used by the focus animation process
    :cvar count: The max amount of shots to be used
    :cvar initial_skip: The number of shots read from the JSON file to be skipped from the beginning
    :cvar skip: The number of shots to be skipped after every shot projection
    :cvar lazy: Whether the shots should be loaded lazy
    :cvar shot_centered_camera: Whether the camera should be moved to look at the center of the shots
    :cvar release_shots: Whether the renderer should release shots as soon as he used them
    :cvar frame_dir: The directory to which all frames will be saved
    :cvar delete_frames: Whether the frames saved to the frame directory should be deleted after they were written to the video file
    :cvar correction: The correction to be applied to every single shot
    :cvar output_file: The path and name of the output to be generated
    """
    count: int = 1
    initial_skip: int = 0
    skip: int = 1
    lazy: bool = True
    shot_centered_camera: bool = False
    resolution: tuple[int, int] = 1024, 1024
    ortho_size: Optional[tuple[float, float]] = None
    release_shots: bool = True
    correction: Optional[Transform] = None
    frame_dir: str = './.frames'
    delete_frames: bool = True
    output_file: str = ''

