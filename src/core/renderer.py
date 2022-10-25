from enum import Enum
from typing import Final, Optional, Iterable, Union, Iterator

import cv2
from moderngl import Context, Program, Framebuffer
from numpy.typing import NDArray
from pyrr import Matrix44

from src.core.camera import Camera
from src.core.data import MeshData, TextureData, RenderObject
from src.core.defs import TRANSPARENT
from src.core.shot import CtxShot
from src.core.utils import img_from_fbo, overlay, crop_to_content


class ProjectMode(Enum):
    COMPLETE_VIEW = 0,
    SHOT_VIEW = 1


class Renderer:
    _released: bool
    _resolution: Final[tuple[int, int]]
    _ctx: Final[Context]
    _fbo: Final[Framebuffer]
    _obj_prog: Final[Program]
    _obj: RenderObject
    _shot_prog: Final[Program]
    camera: Camera

    def __init__(self, resolution: tuple[int, int], ctx: Context, camera: Camera, mesh: MeshData,
                 texture: Optional[TextureData] = None):
        self._released = False
        self._resolution = resolution
        self._ctx = ctx
        self._fbo = self._ctx.simple_framebuffer(resolution, components=4)
        self._fbo.use()

        self._obj_prog = ctx.program(vertex_shader=self._OBJ_VERT_SHADER, fragment_shader=self._OBJ_FRAG_SHADER)
        self._obj = RenderObject.from_mesh(self._obj_prog, mesh, texture, self._PAR_POS, self._PAR_UV)

        self._shot_prog = ctx.program(vertex_shader=self._SHOT_VERT_SHADER, fragment_shader=self._SHOT_FRAG_SHADER)
        self._shot = self._get_shot_render_object()

        self.camera = camera
        self.apply_camera()

    def _get_shot_render_object(self) -> RenderObject:
        vertex_buf = self._obj.vertex_buf
        vao_content = [(vertex_buf, '3f4', self._PAR_POS)]
        ibo = self._obj.ibo
        if ibo is not None:
            vao = self._ctx.vertex_array(self._shot_prog, vao_content, index_buffer=ibo, index_element_size=4)
        else:
            vao = self._ctx.vertex_array(self._shot_prog, vao_content)

        return RenderObject(vao, vao_content, vertex_buf, None, ibo, self._obj.tex)

    def apply_camera(self) -> None:
        """
        Applies the current camera values to the shader
        """
        for prog in (self._obj_prog, self._shot_prog):
            prog[self._PAR_MODEL].write(Matrix44.identity(dtype='f4'))
            prog[self._PAR_VIEW].write(self.camera.get_view())
            prog[self._PAR_PROJ].write(self.camera.get_proj())

    def release(self) -> None:
        """
        Releases all objects associated with the given context
        """
        if not self._released:
            self._obj.release()
            self._fbo.release()
            self._obj_prog.release()
            self._shot_prog.release()
            self._released = True

    def render_ground(self) -> NDArray:
        """
        Renders the ground object
        :return: The finished render result
        """
        self._fbo.clear(*TRANSPARENT)
        self._obj.tex_use()
        self._obj.render()
        return img_from_fbo(self._fbo)

    def _use_shot(self, shot: CtxShot):
        shot.use()
        self._shot_prog[self._PAR_SHOT_PROJ].write(shot.get_proj())
        self._shot_prog[self._PAR_SHOT_VIEW].write(shot.get_view())

    def project_shots(self, shots: Union[CtxShot, Iterable[CtxShot]], mode: ProjectMode,
                      save: bool = False, save_name_iter: Optional[Iterator[str]] = None) -> Optional[list[NDArray]]:
        """
        Projects and renders all passed shots
        :param shots: A single or multiple shots to be projected
        :param mode: The projection mode to be used
        :param save: Whether the images should be directly saved instead of being returned
        :param save_name_iter: An iterator iterating over file names to be used when the projections should be saved
        instead of being returned
        :return: A list of images representing all done projections
        """
        if not isinstance(shots, Iterable):
            shots = [shots]

        if mode is ProjectMode.COMPLETE_VIEW:
            background = self.render_ground()
            def process_proj(proj: NDArray) -> NDArray: return overlay(background, proj)
        else:
            def process_proj(proj: NDArray) -> NDArray: return crop_to_content(proj)

        if save:
            results = None
            def handle_result(res: NDArray) -> NDArray: return cv2.imwrite(next(save_name_iter), res)
        else:
            results = []
            def handle_result(res: NDArray) -> None: results.append(res)

        for shot in shots:
            self._fbo.clear(*TRANSPARENT)
            self._use_shot(shot)
            self._shot.render()
            result = process_proj(img_from_fbo(self._fbo))
            handle_result(result)

        return results

    # region Shader Constants

    _PAR_POS: Final[str] = 'v_in_v3_pos'
    _PAR_UV: Final[str] = 'v_in_v2_uv'
    _PAR_PROJ: Final[str] = 'u_m4_proj'
    _PAR_VIEW: Final[str] = 'u_m4_view'
    _PAR_MODEL: Final[str] = 'u_m4_model'
    _PAR_TEX: Final[str] = 'u_s2_tex'
    _PAR_SHOT_PROJ: Final[str] = 'u_m4_shot_proj'
    _PAR_SHOT_VIEW: Final[str] = 'u_m4_shot_cam'

    _OBJ_VERT_SHADER: Final[str] = f"""
    #version 330
    uniform mat4 {_PAR_PROJ};
    uniform mat4 {_PAR_VIEW};
    uniform mat4 {_PAR_MODEL};
    
    layout (location = 0) in vec3 {_PAR_POS};
    layout (location = 1) in vec2 {_PAR_UV};
    out vec2 v_out_v2_uv;
    
    void main() {{
        v_out_v2_uv = {_PAR_UV}.xy;
        gl_Position = {_PAR_PROJ} * {_PAR_VIEW} * {_PAR_MODEL} * vec4({_PAR_POS}.xyz, 1.0);
    }}
    """

    _OBJ_FRAG_SHADER: Final[str] = f"""
    #version 330

    uniform sampler2D {_PAR_TEX};
    
    in vec2 v_out_v2_uv;
    out vec4 f_out_v4_color;
    
    void main() {{
        f_out_v4_color = texture(u_s2_tex, v_out_v2_uv);
    }}
    """

    _SHOT_VERT_SHADER: Final[str] = f"""
    #version 330

    // model view projection matrices of the focus surface (virtual camera)
    uniform mat4 {_PAR_PROJ};
    uniform mat4 {_PAR_VIEW};
    uniform mat4 {_PAR_MODEL};

    // view and camera/projection matrix for one shot:
    uniform mat4 {_PAR_SHOT_PROJ};
    uniform mat4 {_PAR_SHOT_VIEW};

    layout (location = 0) in vec3 {_PAR_POS};
    out vec4 v_out_v4_shot_uv;

    void main() {{
        vec4 world_pos = {_PAR_MODEL} * vec4({_PAR_POS}.xyz, 1.0);
        gl_Position = {_PAR_PROJ} * {_PAR_VIEW} *  world_pos;
        v_out_v4_shot_uv = {_PAR_SHOT_PROJ} * {_PAR_SHOT_VIEW} * world_pos;
    }}
    """

    _SHOT_FRAG_SHADER: Final[str] = f"""
    #version 330

    uniform sampler2D {_PAR_TEX};

    in vec4 v_out_v4_shot_uv;
    out vec4 f_out_v4_color;

    void main() {{
        vec4 uv = v_out_v4_shot_uv;
        uv = vec4(uv.xyz / uv.w / 2.0 + .5, 1.0); // perspective division and conversion to [0,1] from NDC
        
        if(uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {{ // uv out of bounds
            discard; // throw away the fragment 
            f_out_v4_color = vec4(0.0, 0.0, 0.0, 0.0);
        }} else {{
            f_out_v4_color = vec4(texture({_PAR_TEX}, uv.xy).rgb, 1.0);
        }}
    }}
    """

    # endregion
