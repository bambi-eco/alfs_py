from collections import defaultdict
from functools import cached_property
from typing import Final, Optional, Iterable, Union, Iterator, Callable, cast

import cv2
import numpy as np
import moderngl as mgl
from moderngl import Context, Program, Framebuffer, Texture
from numpy.typing import NDArray

from src.core.defs import TRANSPARENT, BLACK, MAGENTA, PATH_SEP
from src.core.geo.aabb import AABB
from src.core.rendering.camera import Camera
from src.core.rendering.data import MeshData, TextureData, RenderObject, Resolution, RenderResultMode
from src.core.rendering.shot import CtxShot
from src.core.util.basic import gen_checkerboard_tex, get_aabb
from src.core.util.image import overlay
from src.core.util.moderngl import img_from_fbo


class _IntegralSumCallback:
    def __init__(self, shape: Union[int, Iterable, tuple[int]], dtype: Optional[object] = np.uint64):
        self.sum = np.zeros(shape, dtype=dtype)

    def __call__(self, arr: NDArray) -> None:
        self.sum += arr
        # print(' | '.join([str(arr.shape), str(arr.max(initial=0)), str(arr.min(initial=255))]))
        del arr


class Renderer:
    _released: bool
    _resolution: Final[Resolution]
    _ctx: Final[Context]
    _fbo: Final[Framebuffer]
    _obj_prog: Final[Program]
    _obj: RenderObject
    _shot_prog: Final[Program]
    _mask_tex: Optional[Texture]
    mesh_aabb: Final[AABB]
    camera: Camera

    def __init__(self, resolution: Resolution, ctx: Context, camera: Camera, mesh: MeshData,
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
        self._fbo = self._ctx.simple_framebuffer(resolution.as_tuple(), components=4, dtype='f4')
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
        self.apply_matrices()

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

    def apply_matrices(self) -> None:
        """
        Applies the current camera and mesh matrix values to the shader
        """
        for prog in (self._obj_prog, self._shot_prog):
            prog[self._PAR_MODEL].write(self._obj.mat())
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

    def render_background(self) -> NDArray:
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

    def _project_shot(self, shot: CtxShot) -> None:
        self._use_shot(shot)
        self._shot.render()

    def _psi_complete_view(self, shots: Iterable[CtxShot], release_shots: bool) -> Iterator[NDArray]:
        background = self.render_background()
        for shot in shots:
            self._ctx.clear(color=TRANSPARENT)
            self._project_shot(shot)
            result = img_from_fbo(self._fbo)
            if release_shots:
                shot.release()
            yield overlay(background, result)

    def _psi_shot_view_relative(self, shots: Iterable[CtxShot], release_shots: bool) -> Iterator[NDArray]:
        for shot in shots:
            self._ctx.clear(color=TRANSPARENT)
            self._project_shot(shot)
            result = img_from_fbo(self._fbo)
            if release_shots:
                shot.release()
            yield result

    @cached_property
    def _psi_look_up(self) -> dict[RenderResultMode, Callable[[Iterable[CtxShot], bool], Iterator[NDArray]]]:
        # Maybe make this static somehow
        def default(_):
            raise NotImplementedError(f'Renderer is using invalid render mode')

        result: dict[RenderResultMode, Callable[[Iterable[CtxShot]], Iterator[NDArray]]] = defaultdict(default)
        result[RenderResultMode.complete] = self._psi_complete_view
        result[RenderResultMode.shot_only] = self._psi_shot_view_relative
        return result

    def project_shots_iter(self, shots: Union[CtxShot, Iterable[CtxShot]], mode: RenderResultMode,
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

    def project_shots(self, shots: Union[CtxShot, Iterable[CtxShot]], mode: RenderResultMode,
                      release_shots: bool = False,
                      mask: Optional[TextureData] = None, integral: bool = False, save: bool = False,
                      save_name_iter: Optional[Iterator[str]] = None) -> Optional[Union[NDArray, list[NDArray]]]:
        """
        Projects and renders all passed shots
        :param shots: A single or multiple shots to be projected
        :param mode: The projection mode to be used
        :param release_shots: Whether shots should be released after projection (defaults to ``False``)
        :param mask: The mask to be applied to each shot texture (optional)
        :param integral: Whether the result should be the integral of all rendered shots instead of single shot renders
        (defaults to False). This process utilizes the CPU for integration and is significantly slower than the
        ``render_integral`` method, which should be preferred.
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
            alpha_mask = (alpha > 0.0)
            out = np.divide(integral_arr, alpha, where=alpha_mask)
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

    def render_integral(self, shots: Union[CtxShot, Iterable[CtxShot]], release_shots: bool = False,
                        mask: Optional[TextureData] = None, save: bool = False,
                        save_name: Optional[Iterator[str]] = None) -> Optional[NDArray]:
        """
        Renders the integral of the given shots on GPU using additive blending. This process will overwrite the current
        blending function and disable the depth test.
        :param shots: The shots to be projected and integrated
        :param release_shots: Whether shots should be released after projection (defaults to ``False``)
        :param mask: The mask to be applied to each shot texture (optional)
        :param save: Whether the images should be directly saved instead of being returned (defaults to ``False``)
        :param save_name: The file name to be used when saving the result (optional)
        :return: If save is ``True`` ``None``; otherwise the integral of the projected shots
        """
        if not isinstance(shots, Iterable):
            shots = [shots]

        self._use_mask(mask)

        self._ctx.enable(cast(int, mgl.BLEND))
        self._ctx.disable(cast(int, mgl.DEPTH_TEST))
        self._ctx.blend_func = mgl.ADDITIVE_BLENDING

        self._fbo.clear(color=TRANSPARENT)
        for shot in shots:
            self._project_shot(shot)
            if release_shots:
                shot.release()

        integral_bytes = self._fbo.read(components=4, dtype='f4', clamp=False)
        integral_arr = np.frombuffer(integral_bytes, dtype=np.single).reshape((*self._fbo.size[1::-1], 4))
        alpha = integral_arr[:, :, -1][:, :, np.newaxis]
        alpha_mask = (alpha > 0.0)
        out = np.divide(integral_arr, alpha, where=alpha_mask)
        result = (out * 255).astype(np.uint8)[::-1, ...]

        self._ctx.disable(cast(int, mgl.BLEND))

        if save:
            if save_name is None:
                save_name = rf'.{PATH_SEP}integral'
            cv2.imwrite(save_name, result)
            return None
        else:
            return result


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
