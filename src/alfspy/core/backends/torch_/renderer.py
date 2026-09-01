"""The ALFS renderer, backed by PyTorch instead of OpenGL.

The public surface is unchanged from the ModernGL version -- ``render_background``,
``project_shots``, ``project_shots_iter``, ``render_integral``, ``apply_matrices`` and
``release`` keep their signatures and return types. What changed is underneath: there is no
GL context, no bound framebuffer and no driver-owned surface, which is what removes the
stale-buffer artifacts documented in ``docs/MIGRATION_REVIEW.md`` section 3.

Matrices stay in pyrr's row-vector convention throughout; see ``docs/pyrr_conventions.md``.
"""

from collections import defaultdict
from functools import cached_property
from typing import Final, Optional, Iterable, Union, Iterator, Callable

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from alfspy.core.geo import AABB
from alfspy.core.rendering.camera import Camera
from alfspy.core.rendering.data import (
    IntegralResult,
    MeshData,
    RenderResultMode,
    Resolution,
    TextureData,
)

from .data import RenderObject
from .shot import CtxShot
from alfspy.core.torchgl import (
    ADDITIVE_BLENDING,
    BLEND,
    DEPTH_TEST,
    Fragments,
    TorchContext,
    TorchFramebuffer,
    TorchTexture,
    as_matrix,
    iter_fragments,
    resolve_depth,
    setup_triangles,
    shade_object,
    shade_shot_projected,
    shot_clip_coords,
    transform_vertices,
)
from alfspy.core.util.basic import gen_checkerboard_tex
from alfspy.core.util.defs import TRANSPARENT, BLACK, MAGENTA, PATH_SEP
from .framebuffer import img_from_fbo
from alfspy.core.util.geo import get_aabb
from alfspy.core.util.image import overlay


def _no_fragments(device: torch.device, dtype: torch.dtype) -> Fragments:
    """
    :return: An empty :class:`Fragments` batch on the given device.
    """
    empty = torch.zeros(0, device=device, dtype=torch.int64)
    return Fragments(empty, empty.clone(),
                     torch.zeros((0, 3), device=device, dtype=dtype),
                     torch.zeros((0, 3), device=device, dtype=dtype),
                     torch.zeros(0, device=device, dtype=dtype))


class _IntegralSumCallback:
    def __init__(self, shape: Union[int, Iterable, tuple[int]], dtype: Optional[object] = np.uint64):
        self.sum = np.zeros(shape, dtype=dtype)

    def __call__(self, arr: NDArray) -> None:
        self.sum += arr
        del arr


class Renderer:
    """
    Renders a background mesh and projects source frames onto it.
    """

    _released: bool
    _resolution: Final[Resolution]
    _ctx: Final[TorchContext]
    _fbo: Final[TorchFramebuffer]
    _obj: RenderObject
    _mask_tex: Optional[TorchTexture]
    mesh_aabb: Final[AABB]
    camera: Camera

    def __init__(self, resolution: Resolution, ctx: TorchContext, camera: Camera, mesh: MeshData,
                 texture: Optional[TextureData] = None):
        """
        Initializes a new ``Renderer`` object.
        :param resolution: The resolution of the images to render.
        :param ctx: The torch context to be used by the renderer.
        :param camera: The camera to be used by the renderer.
        :param mesh: The mesh data of the main mesh the renderer should work with. It represents the canvas and or
        background of all done projections or renders.
        :param texture: The texture data of the main mesh (optional). If no texture is given a single colored texture
        will be generated.
        """
        self._released = False
        self._resolution = resolution
        self._ctx = ctx
        self._fbo = ctx.simple_framebuffer(resolution.as_tuple(), components=4)
        # Coverage lives in its own accumulator rather than in the alpha channel. The
        # ModernGL backend uses a second colour attachment for the same reason: alpha as an
        # overlap counter costs a channel that an N-channel field needs for data.
        self._coverage = torch.zeros(
            (resolution.height, resolution.width), device=ctx.device, dtype=ctx.dtype)

        if texture is None:
            texture = TextureData(gen_checkerboard_tex(10, 50, BLACK, MAGENTA, dtype='f4'))

        self._obj = RenderObject.from_mesh(ctx, mesh, texture)
        self.mesh_aabb = get_aabb(mesh.vertices)

        self._mask_tex = None

        # Cached matrices, refreshed by ``apply_matrices``.
        self._model_mat: Optional[torch.Tensor] = None
        self._view_mat: Optional[torch.Tensor] = None
        self._proj_mat: Optional[torch.Tensor] = None
        self._world_positions: Optional[torch.Tensor] = None
        # (depth_test, cull_mode, fragments) for the current camera and model transform.
        self._raster_cache: Optional[tuple] = None
        # Per-fragment world positions matching ``_raster_cache``; see ``_fragment_world``.
        self._frag_world: Optional[torch.Tensor] = None

        self.camera = camera
        self.apply_matrices()

    # region properties

    @property
    def ctx(self) -> TorchContext:
        """
        :return: The context this renderer draws with.
        """
        return self._ctx

    @property
    def fbo(self) -> TorchFramebuffer:
        """
        :return: The render target this renderer draws into.
        """
        return self._fbo

    @property
    def render_shape(self) -> tuple[int, int, int]:
        """
        :return: The shape of the renders produced by this renderer.
        """
        return self._resolution[1], self._resolution[0], 4

    # endregion

    def apply_matrices(self) -> None:
        """
        Applies the current camera and mesh matrix values to the render state.

        Replaces the ModernGL version's uniform uploads. World-space vertex positions are
        cached here because both programs need them and the model matrix rarely changes.
        """
        device, dtype = self._ctx.device, self._ctx.dtype

        self._model_mat = as_matrix(self._obj.mat(), device, dtype)
        self._view_mat = as_matrix(self.camera.get_view(), device, dtype)
        self._proj_mat = as_matrix(self.camera.get_proj(), device, dtype)
        self._world_positions = transform_vertices(self._obj.vertices, self._model_mat)
        self._raster_cache = None
        self._frag_world = None

    def release(self) -> None:
        """
        Releases all objects associated with this renderer.
        """
        if not self._released:
            if self._mask_tex is not None:
                self._mask_tex.release()
                self._mask_tex = None

            self._obj.release()
            self._fbo.release()
            self._world_positions = None
            self._raster_cache = None
            self._frag_world = None
            self._released = True

    # region rasterisation

    def _rasterise(self, depth_test: Optional[bool] = None) -> Fragments:
        """
        Rasterises the background mesh from the current camera.

        The result is cached. Coverage depends only on the mesh, the camera and the render
        state -- none of which change while shots are being projected -- so integrating N
        shots rasterises once and shades N times instead of repeating identical geometry work
        N times. :meth:`apply_matrices` invalidates the cache.

        :param depth_test: Whether to depth-resolve (optional). Defaults to the context state.
        :return: The covered fragments. With depth testing that is one fragment per pixel;
            without it, every covered fragment, which is what ``disable(DEPTH_TEST)``
            produced in GL.
        """
        if depth_test is None:
            depth_test = self._ctx.depth_test

        cached = self._raster_cache
        if cached is not None and cached[0] == depth_test and cached[1] == self._ctx.culling:
            return cached[2]

        self._frag_world = None
        width, height = self._resolution.as_tuple()
        mvp = self._model_mat @ self._view_mat @ self._proj_mat
        clip = transform_vertices(self._obj.vertices, mvp)

        setup = setup_triangles(clip, self._obj.triangles, width, height,
                                cull_face=self._ctx.culling, front_face=self._ctx.front_face)

        if depth_test:
            fragments = resolve_depth(setup, width, height, self._ctx.sample_budget)
        else:
            chunks = list(iter_fragments(setup, width, height, self._ctx.sample_budget))
            fragments = Fragments.concat(chunks) if chunks else _no_fragments(
                self._ctx.device, self._ctx.dtype)

        self._raster_cache = (depth_test, self._ctx.culling, fragments)
        return fragments

    def _fragment_world(self, fragments: Fragments) -> torch.Tensor:
        """
        Returns the world-space position of every fragment, interpolating on first use.

        This is the per-shot loop's hot path. Projecting a shot needs its clip coordinates at
        each fragment, which the GL shader obtained by interpolating a per-vertex varying.
        Doing that per shot means an ``(N, 3, 4)`` gather over a million fragments for every
        one of them. Interpolating the *world* position once instead and multiplying it by
        each shot's matrix gives the identical result -- see
        :func:`~alfspy.core.torchgl.programs.shade_shot_projected` for why -- at the cost of
        one matmul per shot.

        The cache is tied to the rasterisation and dropped whenever that is.

        :param fragments: The fragments to interpolate at.
        :return: ``(N, 4)`` per-fragment world positions.
        """
        cached = self._frag_world
        if cached is not None and cached.shape[0] == len(fragments):
            return cached

        world = fragments.interpolate(self._world_positions, self._obj.triangles)
        self._frag_world = world
        return world

    def invalidate_raster_cache(self) -> None:
        """
        Drops the cached rasterisation, freeing its memory.

        Call this when a renderer is kept alive but idle. :meth:`apply_matrices` does it
        automatically whenever the camera or model transform changes.
        """
        self._raster_cache = None
        self._frag_world = None

    def _write(self, pixel_index: torch.Tensor, colour: torch.Tensor, additive: bool) -> None:
        """
        Writes shaded fragments into the framebuffer.

        :param pixel_index: ``(N,)`` flat pixel indices.
        :param colour: ``(N, 4)`` RGBA values aligned with ``pixel_index``.
        :param additive: Whether to accumulate (``ADDITIVE_BLENDING``) or overwrite.
        """
        if colour.numel() == 0:
            return

        flat = self._fbo.data.view(-1, self._fbo.components)
        if additive:
            flat.index_add_(0, pixel_index, colour.to(flat.dtype))
        else:
            flat[pixel_index] = colour.to(flat.dtype)

    # endregion

    def render_background(self) -> NDArray:
        """
        Renders the ground object.
        :return: The finished render result.
        """
        self._fbo.clear(TRANSPARENT)

        fragments = self._rasterise(depth_test=True)
        if len(fragments):
            uvs = self._obj.uvs
            if uvs is None:
                colour = torch.ones((len(fragments), 4), device=self._ctx.device, dtype=self._ctx.dtype)
            else:
                colour = shade_object(fragments, uvs, self._obj.triangles, self._obj.tex)
            self._write(fragments.pixel_index, colour, additive=False)

        return img_from_fbo(self._fbo)

    def _use_mask(self, mask: Optional[TextureData]) -> None:
        if self._mask_tex is not None:
            self._mask_tex.release()
            self._mask_tex = None

        if mask is not None:
            self._mask_tex = TorchTexture(mask.texture, device=self._ctx.device, dtype=self._ctx.dtype)

    def _shot_clip(self, shot: CtxShot, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param shot: The shot to compute clip coordinates for.
        :param positions: The world-space positions to transform (optional; defaults to the
            mesh vertices). The renderer passes per-*fragment* positions, which is valid
            because the transform is linear -- see
            :func:`~alfspy.core.torchgl.programs.shade_shot_projected`.
        :return: ``(N, 4)`` coordinates in the shot camera's clip space.
        """
        device, dtype = self._ctx.device, self._ctx.dtype
        if positions is None:
            positions = self._world_positions
        return shot_clip_coords(
            positions,
            as_matrix(shot.get_view(), device, dtype),
            as_matrix(shot.get_correction(), device, dtype),
            as_matrix(shot.get_proj(), device, dtype),
        )

    def _project_shot(self, shot: CtxShot, additive: bool, depth_test: Optional[bool] = None) -> None:
        """
        Projects a single shot onto the mesh and writes it into the framebuffer.

        :param shot: The shot to project.
        :param additive: Whether to accumulate instead of overwrite.
        :param depth_test: Whether to depth-resolve (optional; defaults to the context state).
        """
        fragments = self._rasterise(depth_test=depth_test)
        if not len(fragments):
            return

        clip = self._shot_clip(shot, self._fragment_world(fragments))
        texture = shot.get_texture(self._ctx)

        colour, keep, weight = shade_shot_projected(
            clip, texture, self._mask_tex,
            reject_behind_camera=self._ctx.reject_behind_camera, return_weight=True)
        if colour.numel() == 0:
            return

        pixel_index = fragments.pixel_index.index_select(0, keep)
        self._write(pixel_index, colour, additive=additive)

        if additive:
            self._coverage.view(-1).index_add_(
                0, pixel_index, weight.squeeze(-1).to(self._coverage.dtype))

    def _psi_complete_view(self, shots: Iterable[CtxShot], release_shots: bool) -> Iterator[NDArray]:
        background = self.render_background()
        for shot in shots:
            self._fbo.clear(TRANSPARENT)
            self._project_shot(shot, additive=False, depth_test=True)
            result = img_from_fbo(self._fbo)
            if release_shots:
                shot.release()
            yield overlay(background, result)

    def _psi_shot_view_relative(self, shots: Iterable[CtxShot], release_shots: bool) -> Iterator[NDArray]:
        for shot in shots:
            self._fbo.clear(TRANSPARENT)
            self._project_shot(shot, additive=False, depth_test=True)
            result = img_from_fbo(self._fbo)
            if release_shots:
                shot.release()
            yield result

    @cached_property
    def _psi_look_up(self) -> dict[RenderResultMode, Callable[[Iterable[CtxShot], bool], Iterator[NDArray]]]:
        def default(_):
            raise NotImplementedError(f'Renderer is using invalid render mode')

        result: dict[RenderResultMode, Callable[[Iterable[CtxShot], bool], Iterator[NDArray]]] = defaultdict(default)
        result[RenderResultMode.Complete] = self._psi_complete_view
        result[RenderResultMode.ShotOnly] = self._psi_shot_view_relative
        return result

    def project_shots_iter(self, shots: Union[CtxShot, Iterable[CtxShot]], mode: RenderResultMode,
                           release_shots: bool = False, mask: Optional[TextureData] = None) -> Iterator[NDArray]:
        """
        Projects and renders all passed shots. Results are in RGBA format.
        :param shots: A single or multiple shots to be projected.
        :param mode: The projection mode to be used.
        :param release_shots: Whether shots should be released after projection (defaults to ``False``).
        :param mask: The mask to be applied to each shot texture (optional).
        :return: An iterator iterating over all performed projections.
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
        Projects and renders all passed shots.
        :param shots: A single or multiple shots to be projected.
        :param mode: The projection mode to be used.
        :param release_shots: Whether shots should be released after projection (defaults to ``False``).
        :param mask: The mask to be applied to each shot texture (optional).
        :param integral: Whether the result should be the integral of all rendered shots instead of single shot renders
        (defaults to False). This process integrates on the CPU and is significantly slower than the
        ``render_integral`` method, which should be preferred.
        :param save: Whether the images should be directly saved instead of being returned (defaults to ``False``).
        :param save_name_iter: An iterator iterating over file names to be used when the projections should be saved
        instead of being returned.
        :return: If save is ``True`` ``None``; otherwise a list of images representing all performed projections.
        """

        if integral:
            handle_result = _IntegralSumCallback(self.render_shape, dtype=np.uint64)

            for result in self.project_shots_iter(shots, mode, release_shots, mask):
                handle_result(result)

            integral_arr = handle_result.sum

            alpha = integral_arr[:, :, -1][:, :, np.newaxis]
            alpha_mask = (alpha > 0.0)
            # `out=` is essential: np.divide with `where=` leaves the excluded entries at
            # whatever the freshly allocated buffer happened to contain. See render_integral.
            out = np.divide(integral_arr, alpha, where=alpha_mask,
                            out=np.zeros_like(integral_arr, dtype=np.float64))
            result = (out * 255).astype(np.uint8)

            del integral_arr
            del out

            if save:
                result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA)
                cv2.imwrite(next(save_name_iter), result)
            else:
                return result
        else:

            if save:
                results = None

                def handle_result(res: NDArray) -> None:
                    res = cv2.cvtColor(res, cv2.COLOR_RGBA2BGRA)
                    cv2.imwrite(next(save_name_iter), res)
                    del res
            else:
                results = []

                def handle_result(res: NDArray) -> None:
                    results.append(res)

            for result in self.project_shots_iter(shots, mode, release_shots, mask):
                handle_result(result)

            return results

    def render_integral_raw(self, shots: Union[CtxShot, Iterable[CtxShot]],
                            release_shots: bool = False,
                            mask: Optional[TextureData] = None) -> IntegralResult:
        """
        Integrates shots and returns the raw accumulation, without normalising or quantising.

        :param shots: The shots to be projected and integrated.
        :param release_shots: Whether shots should be released after projection.
        :param mask: The mask to be applied to each shot texture (optional).
        :return: The accumulated samples and per-pixel coverage, in image row order.
        """
        if not isinstance(shots, Iterable):
            shots = [shots]

        self._use_mask(mask)
        self._ctx.enable(BLEND)
        self._ctx.disable(DEPTH_TEST)
        self._ctx.blend_func = ADDITIVE_BLENDING
        depth_test = self._ctx.depth_test_in_integral

        self._fbo.clear(TRANSPARENT)
        self._coverage.zero_()
        for shot in shots:
            self._project_shot(shot, additive=True, depth_test=depth_test)
            if release_shots:
                shot.release()

        accum = self._fbo.read_array(dtype='f4', clamp=False)
        coverage = self._coverage.detach().cpu().numpy().astype(np.float32)

        self._ctx.disable(BLEND)
        self._ctx.enable(DEPTH_TEST)
        return IntegralResult(accum=accum[::-1], coverage=coverage[::-1])

    def render_integral(self, shots: Union[CtxShot, Iterable[CtxShot]], release_shots: bool = False,
                        mask: Optional[TextureData] = None, save: bool = False,
                        save_name: Optional[str] = None, auto_contrast: bool = True,
                        alpha_threshold: float = 0.1) -> Optional[NDArray]:
        """
        Renders the integral of the given shots using additive accumulation. The image returned will be in the
        RGBA format.

        The alpha channel doubles as an overlap counter: every contributing shot adds ``1.0``, so dividing by it
        normalises each pixel by how many shots actually saw it.

        :param shots: The shots to be projected and integrated.
        :param release_shots: Whether shots should be released after projection (defaults to ``False``).
        :param mask: The mask to be applied to each shot texture (optional).
        :param save: Whether the images should be directly saved instead of being returned (defaults to ``False``).
        :param save_name: The file name to be used when saving the result (optional).
        :param auto_contrast: Whether the result should be automatically contrast adjusted (defaults to ``True``).
        :param alpha_threshold: The minimum number of overlapping shots for a pixel to be used (defaults to 0.1).
        :return: If save is ``True`` ``None``; otherwise the integral of the projected shots.
        """
        if not isinstance(shots, Iterable):
            shots = [shots]

        self._use_mask(mask)

        # Mirrors the GL state the original set up. ``depth_test_in_integral`` keeps the
        # ModernGL behaviour (every fragment accumulates) unless explicitly enabled.
        self._ctx.enable(BLEND)
        self._ctx.disable(DEPTH_TEST)
        self._ctx.blend_func = ADDITIVE_BLENDING
        depth_test = self._ctx.depth_test_in_integral

        self._fbo.clear(TRANSPARENT)
        self._coverage.zero_()
        for shot in shots:
            self._project_shot(shot, additive=True, depth_test=depth_test)
            if release_shots:
                shot.release()

        integral = IntegralResult(
            accum=self._fbo.read_array(dtype='f4', clamp=False),
            coverage=self._coverage.detach().cpu().numpy().astype(np.float32),
        )

        self._ctx.disable(BLEND)
        self._ctx.enable(DEPTH_TEST)

        covered = integral.coverage > alpha_threshold
        out = integral.normalised(threshold=alpha_threshold)

        if auto_contrast and covered.any():
            # Stretch the samples only. The alpha channel is a constant 1 wherever a pixel is
            # covered, so including it would peg the range at [0, 1] and undo the stretch.
            samples = out[..., :3][covered]
            min_val, max_val = samples.min(), samples.max()
            if max_val > min_val:
                stretched = (out[..., :3] - min_val) / (max_val - min_val)
                out[..., :3] = np.where(covered[..., np.newaxis], stretched, out[..., :3])

        # Alpha in the returned image means opacity, as a PNG reader expects; the overlap
        # count is `integral.coverage`.
        out[..., 3] = covered.astype(np.float32)
        result = (out * 255).astype(np.uint8)[::-1, ...]

        if save:
            if save_name is None:
                save_name = rf'.{PATH_SEP}integral'
            result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(save_name, result)
            return None
        else:
            return result
