"""The WebGPU/Vulkan renderer.

Same three operations as every other backend -- render the textured DEM, project a shot onto
it, integrate several shots -- and the same public method signatures, so it drops into the
registry and the existing call sites unchanged.

The one structural difference from OpenGL is that WebGPU has no global render state: blending
and depth testing are baked into a pipeline object rather than toggled on a context. The
pipelines this needs are therefore built once up front and selected per draw.

Two WebGPU restrictions shape the target formats. Blending into an ``rgba32float`` attachment
and bilinear sampling of a float32 texture are both forbidden by the base spec, and the ALFS
integral needs exactly those two things. Where the adapter exposes ``float32-blendable`` and
``float32-filterable`` they are used; where it does not, the backend falls back to 16-bit
float, which still represents the values and overlap counts involved but carries about three
decimal digits instead of seven. :attr:`Renderer.precision` reports which happened, because
silently halving precision on some machines and not others is exactly the kind of difference
that is impossible to debug from the output.
"""

from typing import Final, Iterable, Iterator, Optional, Union

import cv2
import numpy as np
from numpy.typing import NDArray

from alfspy.core.rendering.camera import Camera
from alfspy.core.rendering.data import (
    IntegralResult,
    MeshData,
    RenderResultMode,
    Resolution,
    TextureData,
)
from alfspy.core.util.basic import gen_checkerboard_tex
from alfspy.core.util.defs import BLACK, MAGENTA, PATH_SEP
from alfspy.core.util.geo import get_aabb
from alfspy.core.util.image import overlay

from .context import BLEND, WgpuContext
from .data import RenderObject
from .shaders import OBJECT_SHADER, SHOT_SHADER
from .shot import CtxShot

__all__ = ['Renderer']

_MAT4 = 16 * 4                    # bytes in a mat4x4<f32>
_CAMERA_BYTES: Final[int] = 3 * _MAT4
_SHOT_BYTES: Final[int] = 3 * _MAT4 + 16   # three matrices plus use_mask and its padding


def _additive():
    return {
        'color': {'src_factor': 'one', 'dst_factor': 'one', 'operation': 'add'},
        'alpha': {'src_factor': 'one', 'dst_factor': 'one', 'operation': 'add'},
    }


class Renderer:
    """
    Renders light-field projections of a mesh through WebGPU.
    """

    def __init__(self, resolution: Resolution, ctx: WgpuContext, camera: Camera,
                 mesh: MeshData, texture: Optional[TextureData] = None):
        """
        Initializes a new ``Renderer``.

        :param resolution: The resolution of the images to render.
        :param ctx: The WebGPU context to be used by the renderer.
        :param camera: The camera to be used by the renderer.
        :param mesh: The mesh data of the main mesh, which is the projection surface and the
            background of every render.
        :param texture: The texture data of the main mesh (optional).
        """
        import wgpu

        self._wgpu = wgpu
        self._released = False
        self._resolution = resolution
        self._ctx = ctx
        self.camera = camera

        blendable = ctx.float32_blendable
        self._colour_format = 'rgba32float' if blendable else 'rgba16float'
        self._coverage_format = 'r32float' if blendable else 'r16float'
        self._texture_format = 'rgba32float' if ctx.float32_filterable else 'rgba16float'
        self.precision = 'float32' if blendable and ctx.float32_filterable else 'float16'

        device = ctx.device
        width, height = resolution.as_tuple()

        usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC
        self._colour_tex = device.create_texture(
            size=(width, height, 1), format=self._colour_format, usage=usage)
        self._coverage_tex = device.create_texture(
            size=(width, height, 1), format=self._coverage_format, usage=usage)
        self._depth_tex = device.create_texture(
            size=(width, height, 1), format='depth32float',
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT)

        self._colour_view = self._colour_tex.create_view()
        self._coverage_view = self._coverage_tex.create_view()
        self._depth_view = self._depth_tex.create_view()

        if texture is None:
            texture = TextureData(gen_checkerboard_tex(10, 50, BLACK, MAGENTA, dtype='f4'))

        self._obj = RenderObject.from_mesh(ctx, mesh, texture, self._texture_format)
        self.mesh_aabb = get_aabb(mesh.vertices)

        self._sampler = device.create_sampler(
            mag_filter='linear', min_filter='linear',
            address_mode_u='repeat', address_mode_v='repeat')

        self._camera_buf = device.create_buffer(
            size=_CAMERA_BYTES, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self._shot_buf = device.create_buffer(
            size=_SHOT_BYTES, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        # WebGPU requires every declared binding to be bound, so an unmasked render still
        # needs a mask texture. A 1x1 white texel is the identity weight.
        self._null_mask = self._upload_texture(
            np.ones((1, 1, 4), dtype=np.float32), self._texture_format)
        self._mask_tex = None
        self._mask_view = self._null_mask.create_view()

        self._build_pipelines()
        self.apply_matrices()

    # region setup

    def _upload_texture(self, data: NDArray, fmt: str):
        """Uploads an (H, W, 4) float array as a sampled texture."""
        wgpu = self._wgpu
        device = self._ctx.device
        height, width = data.shape[0], data.shape[1]

        dtype = np.float32 if fmt.endswith('32float') else np.float16
        payload = np.ascontiguousarray(data, dtype=dtype)

        tex = device.create_texture(
            size=(width, height, 1), format=fmt,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        device.queue.write_texture(
            {'texture': tex},
            payload.tobytes(),
            {'bytes_per_row': width * 4 * payload.dtype.itemsize, 'rows_per_image': height},
            (width, height, 1),
        )
        return tex

    def _build_pipelines(self) -> None:
        wgpu = self._wgpu
        device = self._ctx.device

        obj_module = device.create_shader_module(code=OBJECT_SHADER)
        shot_module = device.create_shader_module(code=SHOT_SHADER)

        vis = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
        obj_layout = device.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': vis, 'buffer': {'type': 'uniform'}},
            {'binding': 1, 'visibility': wgpu.ShaderStage.FRAGMENT, 'texture': {}},
            {'binding': 2, 'visibility': wgpu.ShaderStage.FRAGMENT, 'sampler': {}},
        ])
        shot_layout = device.create_bind_group_layout(entries=[
            {'binding': 0, 'visibility': vis, 'buffer': {'type': 'uniform'}},
            {'binding': 1, 'visibility': wgpu.ShaderStage.FRAGMENT, 'texture': {}},
            {'binding': 2, 'visibility': wgpu.ShaderStage.FRAGMENT, 'sampler': {}},
            {'binding': 3, 'visibility': vis, 'buffer': {'type': 'uniform'}},
            {'binding': 4, 'visibility': wgpu.ShaderStage.FRAGMENT, 'texture': {}},
        ])
        self._obj_bg_layout = obj_layout
        self._shot_bg_layout = shot_layout

        depth_state = {
            'format': 'depth32float',
            'depth_write_enabled': True,
            'depth_compare': 'less',
        }
        no_depth = {
            'format': 'depth32float',
            'depth_write_enabled': False,
            'depth_compare': 'always',
        }

        def targets(blend):
            return [
                {'format': self._colour_format, 'blend': blend},
                {'format': self._coverage_format, 'blend': blend},
            ]

        def make(module, layout, blend, depth):
            return device.create_render_pipeline(
                layout=device.create_pipeline_layout(bind_group_layouts=[layout]),
                vertex={'module': module, 'entry_point': 'vs_main',
                        'buffers': self._vertex_layout(module is obj_module)},
                fragment={'module': module, 'entry_point': 'fs_main',
                          'targets': targets(blend)},
                primitive={'topology': 'triangle-list', 'cull_mode': 'back',
                           'front_face': 'ccw'},
                depth_stencil=depth,
            )

        self._obj_pipeline = make(obj_module, obj_layout, None, depth_state)
        self._shot_pipeline = make(shot_module, shot_layout, None, depth_state)
        self._shot_pipeline_additive = make(shot_module, shot_layout, _additive(), no_depth)

        self._obj_bind_group = device.create_bind_group(layout=obj_layout, entries=[
            {'binding': 0, 'resource': {'buffer': self._camera_buf, 'offset': 0,
                                        'size': _CAMERA_BYTES}},
            {'binding': 1, 'resource': self._obj.texture.create_view()},
            {'binding': 2, 'resource': self._sampler},
        ])
        self._shot_bind_group = None

    def _vertex_layout(self, with_uv: bool) -> list:
        layout = [{
            'array_stride': 3 * 4,
            'step_mode': 'vertex',
            'attributes': [{'format': 'float32x3', 'offset': 0, 'shader_location': 0}],
        }]
        if with_uv:
            layout.append({
                'array_stride': 2 * 4,
                'step_mode': 'vertex',
                'attributes': [{'format': 'float32x2', 'offset': 0, 'shader_location': 1}],
            })
        return layout

    # endregion

    @property
    def render_shape(self) -> tuple:
        """
        :return: The shape of the renders produced by this renderer.
        """
        return self._resolution[1], self._resolution[0], 4

    def apply_matrices(self) -> None:
        """
        Applies the current camera and mesh matrix values to the uniform buffer.
        """
        payload = np.concatenate([
            np.asarray(self.camera.get_proj(dtype='f4'), dtype=np.float32).reshape(-1),
            np.asarray(self.camera.get_view(dtype='f4'), dtype=np.float32).reshape(-1),
            np.asarray(self._obj.mat(dtype='f4'), dtype=np.float32).reshape(-1),
        ])
        self._ctx.device.queue.write_buffer(self._camera_buf, 0, payload.tobytes())

    def _write_shot_uniform(self, shot: CtxShot, use_mask: bool) -> None:
        payload = np.concatenate([
            np.asarray(shot.get_proj(), dtype=np.float32).reshape(-1),
            np.asarray(shot.get_view(), dtype=np.float32).reshape(-1),
            np.asarray(shot.get_correction(), dtype=np.float32).reshape(-1),
            np.array([1.0 if use_mask else -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ])
        self._ctx.device.queue.write_buffer(self._shot_buf, 0, payload.tobytes())

    def _use_mask(self, mask: Optional[TextureData]) -> None:
        if mask is None:
            self._mask_tex = None
            self._mask_view = self._null_mask.create_view()
            return
        data = mask.texture
        if data.ndim == 2:
            data = data[..., np.newaxis]
        if data.shape[2] < 4:
            data = np.concatenate(
                [data] + [data[..., :1]] * (4 - data.shape[2]), axis=2)
        if mask.normalise and data.max(initial=0.0) > 1.0:
            data = data / 255.0
        self._mask_tex = self._upload_texture(data, self._texture_format)
        self._mask_view = self._mask_tex.create_view()

    def _bind_shot(self, shot: CtxShot):
        device = self._ctx.device
        return device.create_bind_group(layout=self._shot_bg_layout, entries=[
            {'binding': 0, 'resource': {'buffer': self._camera_buf, 'offset': 0,
                                        'size': _CAMERA_BYTES}},
            {'binding': 1, 'resource': shot.get_texture(self._ctx).create_view()},
            {'binding': 2, 'resource': self._sampler},
            {'binding': 3, 'resource': {'buffer': self._shot_buf, 'offset': 0,
                                        'size': _SHOT_BYTES}},
            {'binding': 4, 'resource': self._mask_view},
        ])

    # region drawing

    def _begin(self, encoder, load: bool):
        op = 'load' if load else 'clear'
        return encoder.begin_render_pass(
            color_attachments=[
                {'view': self._colour_view, 'load_op': op, 'store_op': 'store',
                 'clear_value': (0.0, 0.0, 0.0, 0.0)},
                {'view': self._coverage_view, 'load_op': op, 'store_op': 'store',
                 'clear_value': (0.0, 0.0, 0.0, 0.0)},
            ],
            depth_stencil_attachment={
                'view': self._depth_view,
                'depth_load_op': 'clear' if not load else 'load',
                'depth_store_op': 'store',
                'depth_clear_value': 1.0,
            },
        )

    def _draw_object(self, encoder, load: bool = False) -> None:
        rp = self._begin(encoder, load=load)
        rp.set_pipeline(self._obj_pipeline)
        rp.set_bind_group(0, self._obj_bind_group)
        rp.set_vertex_buffer(0, self._obj.vertex_buffer)
        rp.set_vertex_buffer(1, self._obj.uv_buffer)
        rp.set_index_buffer(self._obj.index_buffer, 'uint32')
        rp.draw_indexed(self._obj.index_count, 1, 0, 0, 0)
        rp.end()

    def _draw_shot(self, encoder, shot: CtxShot, load: bool, additive: bool) -> None:
        rp = self._begin(encoder, load=load)
        rp.set_pipeline(self._shot_pipeline_additive if additive else self._shot_pipeline)
        rp.set_bind_group(0, self._bind_shot(shot))
        rp.set_vertex_buffer(0, self._obj.vertex_buffer)
        rp.set_index_buffer(self._obj.index_buffer, 'uint32')
        rp.draw_indexed(self._obj.index_count, 1, 0, 0, 0)
        rp.end()

    def _read_attachment(self, texture, components: int) -> NDArray:
        width, height = self._resolution.as_tuple()
        itemsize = 4 if texture.format.endswith('32float') else 2
        dtype = np.float32 if itemsize == 4 else np.float16

        raw = self._ctx.device.queue.read_texture(
            {'texture': texture},
            {'bytes_per_row': width * components * itemsize, 'rows_per_image': height},
            (width, height, 1),
        )
        arr = np.frombuffer(bytes(raw), dtype=dtype).reshape((height, width, components))
        return arr.astype(np.float32)

    def _read_targets(self) -> IntegralResult:
        """
        Reads both attachments.

        No vertical flip here, unlike the OpenGL backend: WebGPU's framebuffer row 0 is the
        top of the image, OpenGL's is the bottom.
        """
        colour = self._read_attachment(self._colour_tex, 4)
        coverage = self._read_attachment(self._coverage_tex, 1)[:, :, 0]
        return IntegralResult(accum=colour, coverage=coverage)

    # endregion

    def render_background(self) -> NDArray:
        """
        Renders the ground object.

        :return: The finished render result as a ``uint8`` RGBA image.
        """
        encoder = self._ctx.device.create_command_encoder()
        self._draw_object(encoder)
        self._ctx.device.queue.submit([encoder.finish()])

        result = self._read_targets()
        img = np.clip(result.accum, 0.0, 1.0)
        return (img * 255).astype(np.uint8)

    def project_shots_iter(self, shots: Union[CtxShot, Iterable[CtxShot]],
                           mode: RenderResultMode, release_shots: bool = False,
                           mask: Optional[TextureData] = None) -> Iterator[NDArray]:
        """
        Projects and renders all passed shots. Results are in RGBA format.

        :param shots: A single or multiple shots to be projected.
        :param mode: The projection mode to be used.
        :param release_shots: Whether shots should be released after projection.
        :param mask: The mask to be applied to each shot texture (optional).
        :return: An iterator over the performed projections.
        """
        if not isinstance(shots, Iterable):
            shots = [shots]

        self._use_mask(mask)
        background = self.render_background() if mode is RenderResultMode.Complete else None

        for shot in shots:
            self._write_shot_uniform(shot, mask is not None)
            encoder = self._ctx.device.create_command_encoder()
            self._draw_shot(encoder, shot, load=False, additive=False)
            self._ctx.device.queue.submit([encoder.finish()])

            result = self._read_targets()
            img = (np.clip(result.accum, 0.0, 1.0) * 255).astype(np.uint8)
            if release_shots:
                shot.release()

            yield overlay(background, img) if background is not None else img

    def project_shots(self, shots: Union[CtxShot, Iterable[CtxShot]], mode: RenderResultMode,
                      release_shots: bool = False, mask: Optional[TextureData] = None,
                      integral: bool = False, save: bool = False,
                      save_name_iter: Optional[Iterator[str]] = None):
        """
        Projects and renders all passed shots.

        :param shots: A single or multiple shots to be projected.
        :param mode: The projection mode to be used.
        :param release_shots: Whether shots should be released after projection.
        :param mask: The mask to be applied to each shot texture (optional).
        :param integral: Whether to return the CPU-side integral of all renders.
        :param save: Whether the images should be saved instead of returned.
        :param save_name_iter: File names to use when saving.
        :return: ``None`` when saving; otherwise the renders.
        """
        if integral:
            total = None
            for result in self.project_shots_iter(shots, mode, release_shots, mask):
                total = result.astype(np.uint64) if total is None else total + result
            alpha = total[:, :, -1][:, :, np.newaxis]
            out = np.divide(total, alpha, where=alpha > 0,
                            out=np.zeros_like(total, dtype=np.float64))
            result = (out * 255).astype(np.uint8)
            if save:
                cv2.imwrite(next(save_name_iter), cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA))
                return None
            return result

        results = None if save else []
        for item in self.project_shots_iter(shots, mode, release_shots, mask):
            if save:
                cv2.imwrite(next(save_name_iter), cv2.cvtColor(item, cv2.COLOR_RGBA2BGRA))
            else:
                results.append(item)
        return results

    def render_integral_raw(self, shots: Union[CtxShot, Iterable[CtxShot]],
                            release_shots: bool = False,
                            mask: Optional[TextureData] = None) -> IntegralResult:
        """
        Integrates shots and returns the raw accumulation, without normalising or quantising.

        :param shots: The shots to be projected and integrated.
        :param release_shots: Whether shots should be released after projection.
        :param mask: The mask to be applied to each shot texture (optional).
        :return: The accumulated samples and per-pixel coverage.
        """
        if not isinstance(shots, Iterable):
            shots = [shots]

        self._use_mask(mask)
        first = True
        for shot in shots:
            self._write_shot_uniform(shot, mask is not None)
            encoder = self._ctx.device.create_command_encoder()
            self._draw_shot(encoder, shot, load=not first, additive=True)
            self._ctx.device.queue.submit([encoder.finish()])
            first = False
            if release_shots:
                shot.release()

        if first:  # no shots drawn: clear so the result is defined
            encoder = self._ctx.device.create_command_encoder()
            rp = self._begin(encoder, load=False)
            rp.end()
            self._ctx.device.queue.submit([encoder.finish()])

        return self._read_targets()

    def render_integral(self, shots: Union[CtxShot, Iterable[CtxShot]],
                        release_shots: bool = False, mask: Optional[TextureData] = None,
                        save: bool = False, save_name: Optional[str] = None,
                        auto_contrast: bool = True,
                        alpha_threshold: float = 0.1) -> Optional[NDArray]:
        """
        Renders the integral of the given shots using additive blending.

        :param shots: The shots to be projected and integrated.
        :param release_shots: Whether shots should be released after projection.
        :param mask: The mask to be applied to each shot texture (optional).
        :param save: Whether the image should be saved instead of returned.
        :param save_name: The file name to use when saving.
        :param auto_contrast: Whether to stretch the result's contrast.
        :param alpha_threshold: The minimum number of overlapping shots for a pixel to count.
        :return: ``None`` when saving; otherwise the integral as a ``uint8`` RGBA image.
        """
        integral = self.render_integral_raw(shots, release_shots=release_shots, mask=mask)

        covered = integral.coverage > alpha_threshold
        out = integral.normalised(threshold=alpha_threshold)

        if auto_contrast and covered.any():
            samples = out[..., :3][covered]
            min_val, max_val = samples.min(), samples.max()
            if max_val > min_val:
                stretched = (out[..., :3] - min_val) / (max_val - min_val)
                out[..., :3] = np.where(covered[..., np.newaxis], stretched, out[..., :3])

        out[..., 3] = covered.astype(np.float32)
        result = (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)

        if save:
            if save_name is None:
                save_name = rf'.{PATH_SEP}integral'
            cv2.imwrite(save_name, cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA))
            return None
        return result

    def release(self) -> None:
        """
        Releases everything this renderer owns. Idempotent.
        """
        if self._released:
            return
        self._obj.release()
        self._released = True
