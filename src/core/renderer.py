from collections import defaultdict
from copy import deepcopy
from enum import Enum
from functools import cached_property
from typing import Final, Optional, Iterable, Union, Iterator, Callable

import cv2
import numpy as np
from moderngl import Context, Program, Framebuffer, Texture
from numpy.typing import NDArray
from pyrr import Matrix44

from src.core.camera import Camera
from src.core.data import MeshData, TextureData, RenderObject, AABB
from src.core.decorators import incomplete
from src.core.defs import TRANSPARENT, BLACK, MAGENTA, OUTPUT_DIR
from src.core.geo.frustum import Frustum
from src.core.shot import CtxShot
from src.core.util.basic import gen_checkerboard_tex, get_center, int_up, get_aabb
from src.core.util.image import overlay
from src.core.util.moderngl import img_from_fbo


class ProjectMode(Enum):
    COMPLETE_VIEW = 0,
    SHOT_VIEW_RELATIVE = 1,
    SHOT_VIEW_EXCLUSIVE = 2,

    def __str__(self):
        return self.name

class _IntegralSumCallback:
    def __init__(self, shape: Union[int, Iterable, tuple[int]], dtype: Optional[object] = np.uint64):
        self.sum = np.zeros(shape, dtype=dtype)

    def __call__(self, arr: NDArray) -> None:
        self.sum += arr
        # print(' | '.join([str(arr.shape), str(arr.max(initial=0)), str(arr.min(initial=255))]))
        del arr


class Renderer:
    _released: bool
    _resolution: Final[tuple[int, int]]
    _ctx: Final[Context]
    _fbo: Final[Framebuffer]
    _obj_prog: Final[Program]
    _obj: RenderObject
    _shot_prog: Final[Program]
    _mask_tex: Optional[Texture]
    mesh_aabb: Final[AABB]
    camera: Camera

    def __init__(self, resolution: tuple[int, int], ctx: Context, camera: Camera, mesh: MeshData,
                 texture: Optional[TextureData] = None):
        """
        Initializes a new ``Renderer`` object
        :param resolution: The resolution of the images to render
        :param ctx: The ModernGL context to be used by the renderer
        :param camera: The camera to be used by the renderer
        :param mesh: The mesh data of the main mesh the renderer should work with. It represents the canvas and or
        background of all done projections or renders
        :param texture: The texture data of the main mesh (optional). If no texture is given a single colored texture
        will be generated
        """
        self._released = False
        self._resolution = resolution
        self._ctx = ctx
        self._fbo = self._ctx.simple_framebuffer(resolution, components=4)
        self._fbo.use()

        if texture is None:
            texture = TextureData(gen_checkerboard_tex(10, 50, BLACK, MAGENTA, dtype='f4'))

        self._obj_prog = ctx.program(vertex_shader=self._OBJ_VERT_SHADER, fragment_shader=self._OBJ_FRAG_SHADER)
        self._obj = RenderObject.from_mesh(self._obj_prog, mesh, texture, self._PAR_POS, self._PAR_UV)
        self.mesh_aabb = get_aabb(mesh.vertices)

        self._shot_prog = ctx.program(vertex_shader=self._SHOT_VERT_SHADER, fragment_shader=self._SHOT_FRAG_SHADER)
        self._shot_prog[self._PAR_TEX] = self._S2_LOC_TEX
        self._shot_prog[self._PAR_MASK] = self._S2_LOC_MASK
        self._shot = self._get_shot_render_object()

        self._mask_tex = None

        self.camera = camera
        self.apply_camera()

    @property
    def render_shape(self) -> tuple[int, int, int]:
        return self._resolution[1], self._resolution[0], 4

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
            if self._mask_tex is not None:
                self._mask_tex.release()

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
        self._ctx.clear(*TRANSPARENT)
        self._obj.tex_use()
        self._obj.render()
        return img_from_fbo(self._fbo)

    def _use_mask(self, mask: Optional[TextureData]):
        if mask is not None:
            if self._mask_tex is not None:
                self._mask_tex.release()
                del self._mask_tex
            self._mask_tex = self._ctx.texture(*mask.tex_gen_input(), dtype='f4')
            self._mask_tex.use(self._S2_LOC_MASK)
            self._shot_prog[self._PAR_MASK_FLAG].value = self._VAL_TRUE
        else:
            self._shot_prog[self._PAR_MASK_FLAG].value = self._VAL_FALSE

    def _use_shot(self, shot: CtxShot):
        shot.tex_use(self._S2_LOC_TEX)
        self._shot_prog[self._PAR_SHOT_PROJ].write(shot.get_proj())
        self._shot_prog[self._PAR_SHOT_VIEW].write(shot.get_view())
        self._shot_prog[self._PAR_SHOT_CORRECTION].write(shot.get_correction())

    def _project_shot(self, shot: CtxShot) -> NDArray:
        self._ctx.clear(*TRANSPARENT)
        self._use_shot(shot)
        self._shot.render()
        result = img_from_fbo(self._fbo)
        return result

    def _psi_complete_view(self, shots: Iterable[CtxShot], release_shots: bool) -> Iterator[NDArray]:
        background = self.render_ground()
        for shot in shots:
            result = self._project_shot(shot)
            if release_shots:
                shot.release()
            yield overlay(background, result)

    def _psi_shot_view_relative(self, shots: Iterable[CtxShot], release_shots: bool) -> Iterator[NDArray]:
        for shot in shots:
            result = self._project_shot(shot)
            if release_shots:
                shot.release()
            yield result

    @incomplete('Method for projecting points not finished yet')
    def _psi_shot_view_exclusive(self, shots: Iterable[CtxShot], release_shots: bool) -> Iterator[NDArray]:
        camera_cache = deepcopy(self.camera)
        for shot in shots:
            # compute projected points

            # compute camera to capture all points

            camera = Camera()

            # apply camera
            self.camera = camera
            self.apply_camera()

            # project shot
            result = self._project_shot(shot)

            if release_shots:
                shot.release()
            yield result

        # reset camera
        self.camera = camera_cache

    @cached_property
    def _psi_look_up(self) -> dict[ProjectMode, Callable[[Iterable[CtxShot], bool], Iterator[NDArray]]]:
        # Maybe make this static somehow
        def default(_):
            raise NotImplementedError(f'Renderer is using invalid projection mode')

        result: dict[ProjectMode, Callable[[Iterable[CtxShot]], Iterator[NDArray]]] = defaultdict(default)
        result[ProjectMode.COMPLETE_VIEW] = self._psi_complete_view
        result[ProjectMode.SHOT_VIEW_RELATIVE] = self._psi_shot_view_relative
        result[ProjectMode.SHOT_VIEW_EXCLUSIVE] = self._psi_shot_view_exclusive
        return result

    def project_shots_iter(self, shots: Union[CtxShot, Iterable[CtxShot]], mode: ProjectMode,
                           release_shots: bool = False, mask: Optional[TextureData] = None) -> Iterator[NDArray]:
        """
        Projects and renders all passed shots
        :param shots: A single or multiple shots to be projected
        :param mode: The projection mode to be used
        :param release_shots: Whether shots should be released after projection (defaults to ``False``)
        :param mask: The mask to be applied to each shot texture (optional)
        :return: An iterator iterating over all performed projections
        """
        if not isinstance(shots, Iterable):
            shots = [shots]

        self._use_mask(mask)

        return self._psi_look_up[mode](shots, release_shots)

    def project_shots(self, shots: Union[CtxShot, Iterable[CtxShot]], mode: ProjectMode, release_shots: bool = False,
                      mask: Optional[TextureData] = None, integral: bool = False, save: bool = False,
                      save_name_iter: Optional[Iterator[str]] = None) -> Optional[Union[NDArray, list[NDArray]]]:
        """
        Projects and renders all passed shots
        :param shots: A single or multiple shots to be projected
        :param mode: The projection mode to be used
        :param release_shots: Whether shots should be released after projection (defaults to ``False``)
        :param mask: The mask to be applied to each shot texture (optional)
        :param integral: Whether the result should be the integral of all rendered shots instead of single shot renders
        (defaults to False)
        :param save: Whether the images should be directly saved instead of being returned (defaults to ``False``)
        :param save_name_iter: An iterator iterating over file names to be used when the projections should be saved
        instead of being returned
        :return: If save is ``True`` ``None``; otherwise a list of images representing all performed projections
        """

        if integral:
            handle_result = _IntegralSumCallback(self.render_shape, dtype=np.uint64)

            for result in self.project_shots_iter(shots, mode, release_shots, mask):
                handle_result(result)

            integral_arr = handle_result.sum

            alpha = integral_arr[:, :, -1][:, :, np.newaxis]
            alpha_mask = (alpha >= 1.0)
            out = np.divide(integral_arr, alpha, where=alpha_mask, dtype=np.float64)
            result = (out * 255).astype(np.uint8)

            del integral_arr
            del out

            if save:
                cv2.imwrite(next(save_name_iter), result)
            else:
                return result
        else:

            if save:
                results = None

                def handle_result(res: NDArray) -> None:
                    cv2.imwrite(next(save_name_iter), res)
                    del res
            else:
                results = []

                def handle_result(res: NDArray) -> None:
                    results.append(res)

            for result in self.project_shots_iter(shots, mode, release_shots, mask):
                handle_result(result)

            return results

    # region Shader Constants

    _PAR_POS: Final[str] = 'v_in_v3_pos'
    _PAR_UV: Final[str] = 'v_in_v2_uv'
    _PAR_PROJ: Final[str] = 'u_m4_proj'
    _PAR_VIEW: Final[str] = 'u_m4_view'
    _PAR_MODEL: Final[str] = 'u_m4_model'
    _PAR_TEX: Final[str] = 'u_s2_tex'
    _PAR_MASK: Final[str] = 'u_s2_mask'
    _PAR_MASK_FLAG: Final[str] = 'u_f_mask'
    _PAR_SHOT_PROJ: Final[str] = 'u_m4_shot_proj'
    _PAR_SHOT_VIEW: Final[str] = 'u_m4_shot_view'
    _PAR_SHOT_CORRECTION: Final[str] = 'u_m4_shot_correction'

    _VAL_TRUE: Final[float] = 1.0
    _VAL_FALSE: Final[float] = -1.0

    _S2_LOC_TEX: Final[int] = 0
    _S2_LOC_MASK: Final[int] = 1

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
    uniform mat4 {_PAR_SHOT_CORRECTION};

    layout (location = 0) in vec3 {_PAR_POS};
    out vec4 v_out_v4_shot_uv;

    void main() {{
        vec4 world_pos = {_PAR_MODEL} * vec4({_PAR_POS}.xyz, 1.0);
        gl_Position = {_PAR_PROJ} * {_PAR_VIEW} *  world_pos;
        v_out_v4_shot_uv = {_PAR_SHOT_PROJ} * {_PAR_SHOT_CORRECTION} * {_PAR_SHOT_VIEW}  * world_pos;
        // * {_PAR_SHOT_CORRECTION};
    }}
    """

    _SHOT_FRAG_SHADER: Final[str] = f"""
    #version 330

    uniform sampler2D {_PAR_TEX};
    uniform sampler2D {_PAR_MASK};
    uniform float {_PAR_MASK_FLAG};

    in vec4 v_out_v4_shot_uv;
    out vec4 f_out_v4_color;

    void main() {{
        vec4 uv = v_out_v4_shot_uv;
        uv = vec4(uv.xyz / uv.w / 2.0 + .5, 1.0); // perspective division and conversion to [0,1] from NDC
        
        if(uv.w <= 0.0 || uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {{ // uv out of bounds
            discard; // throw away the fragment 
            f_out_v4_color = vec4(0.0, 0.0, 0.0, 0.0);
        }} else {{
            f_out_v4_color = vec4(texture({_PAR_TEX}, uv.xy).rgba);
            if ({_PAR_MASK_FLAG} > 0.0) {{
                f_out_v4_color.a *= texture({_PAR_MASK}, uv.xy).r;
            }}
        }}
    }}
    """

    # endregion


class CenteredRenderer(Renderer):

    @incomplete(f'Class depends on {Frustum.fit_to_points.__name__} which is incomplete')
    def __init__(self, resolution: tuple[int, int], ctx: Context, camera: Camera, mesh: MeshData,
                 texture: Optional[TextureData] = None):
        """
        Initializes a new ``Renderer`` object
        :param resolution: The resolution of the images to render
        :param ctx: The ModernGL context to be used by the renderer
        :param camera: A camera object holding the desired properties the centered camera should have. The centered
        camera will adjust position and alter the far clipping plane distance if needed.
        :param mesh: The mesh data of the main mesh the renderer should work with. It represents the canvas and or
        background of all done projections or renders
        :param texture: The texture data of the main mesh (optional). If no texture is given a single colored texture
        will be generated
        """
        center, aabb = get_center(mesh.vertices)
        if camera.orthogonal:
            center.z = 1
            ortho_size = int_up(aabb.width), int_up(aabb.height)
            camera = Camera(orthogonal=True, orthogonal_size=ortho_size, position=center)
        else:
            camera_dir = (camera.transform.position - center).normalized
            corner_dist = (aabb.p_s - center).length

            if camera.aspect_ratio <= 1.0:
                # fit cameras distance based on the fovy
                pass
            else:
                # fit cameras distance based on the fovx
                fovx = camera.fovy * camera.aspect_ratio

            camera = Camera()

        super().__init__(resolution, ctx, camera, mesh, texture)
