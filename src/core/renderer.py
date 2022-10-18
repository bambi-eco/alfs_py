from enum import Enum
from typing import Final, Optional, Iterable, Union

from moderngl import Context, Program, Framebuffer
from numpy.typing import NDArray
from pyrr import Matrix44

from src.core.camera import Camera
from src.core.data import MeshData, TextureData, RenderObject
from src.core.defs import TRANSPARENT
from src.core.shot import CtxShot
from src.core.utils import img_from_fbo, mesh_to_render_obj, overlay, crop_to_content


class ProjectMode(Enum):
    COMPLETE_VIEW = 0,
    SHOT_VIEW = 1


class Renderer:
    _released: bool
    _resolution: Final[tuple[int, int]]
    _ctx: Final[Context]
    _prog: Final[Program]
    _fbo: Final[Framebuffer]
    _obj: RenderObject
    camera: Camera

    def __init__(self, resolution: tuple[int, int], ctx: Context, camera: Camera, mesh: MeshData,
                 texture: Optional[TextureData] = None):
        self._released = False
        self._resolution = resolution
        self._ctx = ctx
        self._prog = ctx.program(vertex_shader=self._VERT_SHADER, fragment_shader=self._FRAG_SHADER)
        self._fbo = self._ctx.simple_framebuffer(resolution, components=4)
        self._obj = mesh_to_render_obj(self._prog, mesh, texture)

        self.camera = camera
        self.apply_camera()

    def apply_camera(self) -> None:
        """
        Applies the current camera values to the shader
        """
        self._prog["m_model"].write(Matrix44.identity(dtype='f4'))
        self._prog["m_cam"].write(self.camera.get_view())
        self._prog["m_proj"].write(self.camera.get_proj())

    def release(self) -> None:
        """
        Releases all objects associated with the given context
        """
        if not self._released:
            self._obj.release()
            self._fbo.release()
            self._prog.release()
            self._released = True

    def render_ground(self) -> NDArray:
        """
        Renders the ground object
        :return: The finished render result
        """
        self._fbo.clear(*TRANSPARENT)
        self._obj.render()
        return img_from_fbo(self._fbo)

    def project_shots(self, shots: Union[CtxShot, Iterable[CtxShot]], mode: ProjectMode) -> list[NDArray]:
        """
        Projects and renders all passed shots
        :param shots: A single or multiple shots to be projected
        :param mode: The projection mode to be used
        :return: A list of images representing all done projections
        """
        if not isinstance(shots, Iterable):
            shots = [shots]
        results = []

        for shot in shots:
            self._fbo.clear(*TRANSPARENT)
            self._use_shot(shot)
            self._obj.render()
            results.append(img_from_fbo(self._fbo))

        if mode is ProjectMode.COMPLETE_VIEW:
            background = self.render_ground()
            results = [overlay(background, result) for result in results]

        elif mode is ProjectMode.SHOT_VIEW:
            results = [crop_to_content(result) for result in results]

        return results

    def _use_shot(self, shot: CtxShot):
        shot.use()
        self._prog['m_shot_proj'].write(shot.get_proj())
        self._prog['m_shot_cam'].write(shot.get_view())

    _VERT_SHADER: Final[str] = """
        #version 330

        // model view projection matrices of the focus surface (virtual camera)
        uniform mat4 m_proj;
        uniform mat4 m_model;
        uniform mat4 m_cam;

        // view and camera/projection matrix for one shot:
        uniform mat4 m_shot_cam;
        uniform mat4 m_shot_proj;

        in vec3 in_position;
        out vec4 wpos;
        out vec4 shotUV;

        void main() {
            wpos = m_model * vec4(in_position, 1.0);
            gl_Position = m_proj * m_cam * wpos;

            shotUV = m_shot_proj * m_shot_cam * wpos;
        }
    """

    _FRAG_SHADER: Final[str] = """
        #version 330

        uniform sampler2D shotTexture;

        in vec4 wpos;
        in vec4 shotUV;
        out vec4 color;

        void main() {
            vec4 uv = shotUV;
            uv = vec4(uv.xyz / uv.w / 2.0 + .5, 1.0); // perspective division and conversion to [0,1] from NDC

            if(uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
                discard; // throw away the fragment 
                color = vec4(0.0, 0.0, 0.0, 0.0);
            } else {
                // DEBUG: color = vec4(1.0, 1.0, 0.0, 1.0);
                color = vec4(texture(shotTexture, uv.xy).rgb, 1.0);
            }
        }
    """