import os
from collections import defaultdict
from copy import copy
from functools import cache
from logging import getLogger, Logger
from typing import Optional, Iterable, Callable, Sequence, cast, Protocol, Union, TypeVar, Type, Final

import cv2
import moderngl as mgl
import numpy as np
from PIL import Image
from numpy.typing import NDArray
from pyrr import Quaternion, Vector3

from alfspy.core.geo.aabb import AABB
from alfspy.core.geo.transform import Transform
from alfspy.core.rendering.camera import Camera
from alfspy.core.rendering.data import TextureData, RenderResultMode, MeshData, Resolution
from alfspy.core.rendering.renderer import Renderer
from alfspy.core.rendering.shot import CtxShot
from alfspy.core.rendering.shot_loader import AsyncShotLoader
from alfspy.core.util.basic import gen_checkerboard_tex, nearest_int
from alfspy.core.util.defs import MAGENTA, BLACK, MAX_TEX_DIM, CPP_INT_MAX, PATH_SEP
from alfspy.core.util.geo import get_aabb
from alfspy.core.util.gltf import gltf_extract
from alfspy.core.util.image import overlay
from alfspy.core.util.io import delete_all
from alfspy.core.util.loggings import LoggerStep
from alfspy.core.util.video import video_from_images
from alfspy.render.data import CameraPositioningMode, BaseSettings, BaseAnimationSettings, IntegralSettings, \
    FocusAnimationSettings, ShutterAnimationSettings

class Releasable(Protocol):
    """
    Static duck typing class for anything that implements a method ``release() -> None``.
    """

    def release(self) -> None:
        pass

logger: Final[Logger] = getLogger(__name__)

# region Camera helpers

@cache
def _cpm_lookup() -> dict[CameraPositioningMode, Callable]:
    lookup: dict[CameraPositioningMode, Callable] = defaultdict(lambda: Vector3())

    def _cpm_bc(aabb: AABB, _):
        center = aabb.center
        center.z += aabb.depth // 2
        return center

    def _cpm_fs(_, shots: Sequence[CtxShot]):
        x, y, z = shots[0].camera.transform.position
        return Vector3([x, y, z], dtype='f4')

    def _cpm_cs(_, shots: Sequence[CtxShot]):
        count = len(shots)
        idx = count // 2
        x, y, z = shots[idx].camera.transform.position
        return Vector3([x, y, z], dtype='f4')

    def _cpm_ls(_, shots: Sequence[CtxShot]):
        x, y, z = shots[-1].camera.transform.position
        return Vector3([x, y, z], dtype='f4')

    def _cpm_as(_, shots: Sequence[CtxShot]):
        shot_positions = [shot.camera.transform.position for shot in shots]
        return Vector3(np.average(np.stack(shot_positions), axis=0))

    def _cpm_sc(_, shots: Sequence[CtxShot]):
        shot_positions = [shot.camera.transform.position for shot in shots]
        aabb = get_aabb(shot_positions)
        return aabb.center

    lookup[CameraPositioningMode.BackgroundCentered] = _cpm_bc
    lookup[CameraPositioningMode.FirstShot] = _cpm_fs
    lookup[CameraPositioningMode.CenterShot] = _cpm_cs
    lookup[CameraPositioningMode.LastShot] = _cpm_ls
    lookup[CameraPositioningMode.AverageShot] = _cpm_as
    lookup[CameraPositioningMode.ShotCentered] = _cpm_sc

    return lookup


def _get_camera_position(mode: CameraPositioningMode, camera_dist: float, background_aabb: AABB,
                         shots: Sequence[CtxShot]) -> Vector3:
    center = _cpm_lookup()[CameraPositioningMode(mode.value)](background_aabb, shots)
    center.z += camera_dist
    return center


# endregion

# region Common functionality


def make_shot_loader(shots: Iterable[CtxShot]) -> Iterable[CtxShot]:
    """
    Creates a shot loader. Ensures all examples use the same parameters when using a shot loader.
    :return: A shot loader instance ensuring the ``Iterable[CtxShot]`` type.
    """
    return AsyncShotLoader(shots, 96, 12)


def read_gltf(gltf_file: str) -> tuple[Optional[MeshData], Optional[TextureData]]:
    """
    Tries to extract mesh and texture data from the given gltf file.
    :param gltf_file: The path pointing at the GLTF file to process.
    :return: The extracted mesh and texture data.
    """
    return gltf_extract(gltf_file)


def process_render_data(mesh_data: Optional[MeshData],
                        texture_data: Optional[TextureData]) -> tuple[MeshData, TextureData]:
    """
    Transforms data extracted from a GLTF file into valid mesh and texture data that can be used by the renderer.
    :param mesh_data: The extracted mesh data.
    :param texture_data: The extracted texture data.
    :return: Processed mesh and texture data ready to be rendered.
    """

    if mesh_data is None:
        raise ValueError('Mesh data could not be extracted')

    if texture_data is None:
        texture_data = TextureData(gen_checkerboard_tex(8, 8, MAGENTA, BLACK, dtype='f4'))
        logger.info(f'No texture extracted: Default texture was generated')
    else:
        byte_size = texture_data.byte_size(dtype='f4')
        width, height = texture_data.texture.shape[1::-1]
        logger.info(f'Texture extracted: ({width}, {height}) x {texture_data.texture.shape[2]} [{byte_size} B]')

        if width > MAX_TEX_DIM or height > MAX_TEX_DIM:
            if width > height:
                fact = MAX_TEX_DIM / width
            else:
                fact = MAX_TEX_DIM / height
            texture_data.texture = cv2.resize(texture_data.texture, None, fx=fact, fy=fact)
            logger.info(f'Texture downscaled to {texture_data.texture.shape[1::-1]} [{texture_data.byte_size("f4")} B] '
                        f'to fit texture dimension restriction of {MAX_TEX_DIM}px')
            byte_size = texture_data.byte_size(dtype='f4')

        if byte_size > CPP_INT_MAX:
            texture_data.scale_to_fit(CPP_INT_MAX, dtype='f4')  # necessary since moderngl uses this data type
            logger.info(f'Texture downscaled to {texture_data.texture.shape[1::-1]} [{texture_data.byte_size("f4")} B] '
                        f'to fit size restriction of {CPP_INT_MAX} B')

    return mesh_data, texture_data


def make_mgl_context(standalone:bool = True) -> mgl.Context:
    """
    Creates a ModernGL context. Ensures all examples use the same context settings.
    :return: A ModernGL context instance.
    """
    ctx = mgl.create_context(standalone=standalone)
    ctx.enable(cast(int, mgl.DEPTH_TEST))
    ctx.enable(cast(int, mgl.CULL_FACE))
    ctx.cull_face = 'back'
    return ctx


def read_shots(json_file: str, ctx: mgl.Context, se: BaseSettings) -> list[CtxShot]:
    """
    Reads the shots from a JSON file and filters them according to the given settings.
    :param json_file: The shot JSON file.
    :param ctx: The context the shots should be associated with.
    :param se: The basic render settings to be used.
    :return: A list of context related shots.
    """
    shots = CtxShot.from_json(json_file, ctx, count=se.count + se.initial_skip, correction=se.correction, lazy=se.lazy)
    return shots[se.initial_skip::se.skip]


def make_camera(mesh_aabb: AABB, shots: Sequence[CtxShot], se: BaseSettings, rotation: Optional[Quaternion] = None) -> Camera:
    """
    Creates a camera. Ensures all render examples create the camera the same way.
    :param mesh_aabb: The AABB of the mesh to render.
    :param shots: The shots to render.
    :param se: The basic settings associated with the render process.
    :return: A ``Camera`` instance.
    """
    if se.orthogonal:
        # ensure valid ortho_size
        ortho_size = se.ortho_size if se.ortho_size is not None else \
                     (nearest_int(mesh_aabb.width), nearest_int(mesh_aabb.height))
    else:
        ortho_size = se.ortho_size

    camera_pos = _get_camera_position(se.camera_position_mode, se.camera_dist, mesh_aabb, shots)
    return Camera(fovy=se.fovy, aspect_ratio=se.aspect_ratio, orthogonal=se.orthogonal, orthogonal_size=ortho_size,
                  position=camera_pos, rotation=rotation, near=se.near_clipping, far=se.far_clipping)


def read_mask(mask_file: str) -> TextureData:
    """
    Reads a binary mask from the given file path and converts it into texture data.
    :param mask_file: The mask file to be used.
    :return: Texture data resembling the given mask.
    """
    mask_img = cv2.imread(mask_file)
    mask_img = mask_img[..., 0].astype('f4')
    mask_img = np.resize(mask_img, (*mask_img.shape, 1))
    mask_img = np.where(mask_img < 255.0, 0.0, 1.0)
    return TextureData(mask_img)


def release_all(*releasables: Union[Releasable, Iterable[Releasable]]) -> None:
    """
    Calls the release method for all given objects.
    :param releasables: All objects to release.
    """

    rels = []
    for releasable in releasables:
        if isinstance(releasable, Iterable):
            rels.extend(releasable)
        else:
            rels.append(releasable)

    for releasable in rels:
        releasable.release()


def create_video(frame_files: Sequence[str], se: BaseAnimationSettings) -> None:
    """
    Creates a video from a given sequence of frame files.
    :param frame_files: The locations of the frames that should be concatenated to a video.
    :param se: The basic animation settings.
    """
    video_from_images(frame_files, se.output_file, fps=se.fps, release_images=True,
                      first_frame_repetitions=se.first_frame_repetitions,
                      last_frame_repetitions=se.last_frame_repetitions)


# endregion

# region Common processes

S = TypeVar('S', bound=BaseSettings)


def _ensure_or_copy_settings(se: Optional[S], cls: Type[S]) -> S:
    if se is None:
        return cls()
    else:
        return copy(se)


def _base_steps(gltf_file: str, shot_json_file: str, mask_file: Optional[str],
                se: BaseSettings) -> tuple[mgl.Context, Camera, Renderer, list[CtxShot], Optional[TextureData]]:
    with LoggerStep(logger, f'Reading GLTF file from "{gltf_file}"'):
        mesh_data, texture_data = read_gltf(gltf_file)
        mesh_data, texture_data = process_render_data(mesh_data, texture_data)

    with LoggerStep(logger, 'Creating MGL context'):
        ctx = make_mgl_context()

    with LoggerStep(logger,f'Extracting shots from "{shot_json_file}" (Creating lazy shots: {se.lazy})'):
        shots = read_shots(shot_json_file, ctx, se)

    with LoggerStep(logger,f'Creating camera and renderer (camera position mode: {se.camera_position_mode.name})'):
        mesh_aabb = get_aabb(mesh_data.vertices)
        camera = make_camera(mesh_aabb, shots, se)
        logger.info(f'Computed camera position: {camera.transform.position}')
        renderer = Renderer(se.resolution, ctx, camera, mesh_data, texture_data)

    if mask_file is not None:
        with LoggerStep(logger,f'Reading mask from "{mask_file}"'):
            mask = read_mask(mask_file)
    else:
        mask = None

    return ctx, camera, renderer, shots, mask


def _integral_processing(renderer: Renderer, integral: NDArray, se: IntegralSettings) -> None:
    # region Adding Background

    if se.add_background:
        with LoggerStep(logger, 'Rendering background'):
            background = renderer.render_background()

        with LoggerStep(logger, 'Laying integral over background'):
            img = overlay(background, integral)
            im_pil = Image.fromarray(img)
    else:
        logger.info('Converting array to PIL image')
        im_pil = Image.fromarray(integral)

    # endregion

    # region Showing Integral

    if se.show_integral:
        with LoggerStep(logger, 'Showing integral'):
            im_pil.show('Integral')

    # endregion

    # region Saving Integral

    with LoggerStep(logger, f'Saving integral image to "{se.output_file}"'):
        if '.' not in se.output_file:
            se.output_file += '.png'
        im_pil.save(se.output_file)

    # endregion


def _frame_processing(frame_files: Sequence[str], se: BaseAnimationSettings) -> None:
    with LoggerStep(logger, 'Creating video file'):
        create_video(frame_files, se)

    if se.delete_frames:
        with LoggerStep(logger, 'Deleting frames'):
            delete_all(frame_files)


# endregion

def project_shots(gltf_file: str, shot_json_file: str, mask_file: Optional[str] = None,
                  settings: Optional[IntegralSettings] = None) -> None:
    with LoggerStep(logger, 'Start Project Shots', 'All done'):

        # region Initializing

        with LoggerStep(logger, 'Initializing'):
            se = _ensure_or_copy_settings(settings, IntegralSettings)

        # endregion

        ctx, _, renderer, shots, mask = _base_steps(gltf_file, shot_json_file, mask_file, se)

        # region Projecting Shots

        with LoggerStep(logger, f'Projecting shots (Releasing shots after projection: {se.release_shots})'):
            shot_loader = make_shot_loader(shots)
            result = renderer.project_shots(shot_loader, RenderResultMode.ShotOnly, mask=mask, integral=True, save=False,
                                            release_shots=se.release_shots)

        # endregion

        _integral_processing(renderer, result, se)

        with LoggerStep(logger, 'Release all resources'):
            release_all(ctx, renderer, shots)


def render_integral(gltf_file: str, shot_json_file: str, mask_file: Optional[str] = None,
                    settings: Optional[IntegralSettings] = None) -> Camera:
    with LoggerStep(logger, 'Start Render Integral', 'All done'):

        # region Initializing

        with LoggerStep(logger, 'Initializing'):
            se = _ensure_or_copy_settings(settings, IntegralSettings)

        # endregion

        ctx, camera, renderer, shots, mask = _base_steps(gltf_file, shot_json_file, mask_file, se)

        # region Rendering Integral

        with LoggerStep(logger, f'Projecting shots (Releasing shots after projection: {se.release_shots})'):
            shot_loader = make_shot_loader(shots)
            result = renderer.render_integral(shot_loader, mask=mask, save=False, release_shots=se.release_shots)

        # endregion

        _integral_processing(renderer, result, se)

        with LoggerStep(logger, 'Release all resources'):
            release_all(ctx, renderer, shots)

        return camera


def animate_focus(gltf_file: str, shot_json_file: str, mask_file: Optional[str] = None,
                  settings: Optional[FocusAnimationSettings] = None) -> None:
    with LoggerStep(logger, 'Start Animate Focus', 'All done'):

        # region Initializing

        with LoggerStep(logger, 'Initializing'):
            se = _ensure_or_copy_settings(settings, FocusAnimationSettings)

            if se.correction is None:
                se.correction = Transform()

            se.correction.position.z = se.start_focus

        # endregion

        ctx, camera, renderer, shots, mask = _base_steps(gltf_file, shot_json_file, mask_file, se)

        # region Frame Rendering

        with LoggerStep(logger, f'Creating Frames (Frames to be rendered: {se.frame_count}; Focus: {se.start_focus} -> {se.end_focus})'):

            if not se.frame_dir.endswith(PATH_SEP):
                se.frame_dir += PATH_SEP
            os.makedirs(se.frame_dir, exist_ok=True)

            range_focus = se.end_focus - se.start_focus
            focus_step = range_focus / se.frame_count
            logger.info(f'Using focus step: {focus_step}')
            frame_files = []

            if se.move_camera_with_focus:
                camera.transform.position.z -= se.start_focus
                renderer.apply_matrices()

            release_shots = se.release_shots
            add_background = se.add_background
            frame_dir = se.frame_dir
            move_camera_with_focus = se.move_camera_with_focus

            # background needs to be rendered only once when camera is not moving
            if add_background and not move_camera_with_focus:
                background = renderer.render_background()
            else:
                background = None

            for i in range(se.frame_count):
                print(f'Creating frame {i}')
                shots_copy = [shot.create_anew() for shot in shots]
                shot_loader = make_shot_loader(shots_copy)
                result = renderer.render_integral(shot_loader, mask=mask, save=False, release_shots=release_shots)

                if add_background:
                    if move_camera_with_focus:
                        background = renderer.render_background()
                    img = overlay(background, result)
                else:
                    img = result

                im_pil = Image.fromarray(img)
                frame_file = f'{frame_dir}{i}.png'
                im_pil.save(frame_file)
                frame_files.append(frame_file)

                del shot_loader
                del img
                del im_pil

                for shot in shots:
                    shot.correction.position.z += focus_step
                if move_camera_with_focus:
                    camera.transform.position.z -= focus_step
                    renderer.apply_matrices()

        # endregion

        _frame_processing(frame_files, se)

        with LoggerStep(logger, 'Release all resources'):
            release_all(ctx, renderer, shots)


def animate_shutter(gltf_file: str, shot_json_file: str, mask_file: Optional[str] = None,
                    settings: Optional[ShutterAnimationSettings] = None) -> None:
    with LoggerStep(logger, 'Start Animate Shutter', 'All done'):

        # region Initializing
        with LoggerStep(logger, 'Initializing'):
            se = _ensure_or_copy_settings(settings, ShutterAnimationSettings)

        # endregion

        ctx, camera, renderer, shots, mask = _base_steps(gltf_file, shot_json_file, mask_file, se)

        # region Render Background

        if se.add_background:
            with LoggerStep(logger, 'Rendering background'):
                background = renderer.render_background()
        else:
            background = None

        # endregion

        # region Frame Rendering

        with LoggerStep(logger, f'Creating Frames (Frames to be rendered: {se.frame_count})'):
            if not se.frame_dir.endswith(PATH_SEP):
                se.frame_dir += PATH_SEP
            os.makedirs(se.frame_dir, exist_ok=True)

            frame_files = []

            shot_count = len(shots)
            max_shot_count = shot_count - se.reference_index

            shots_grow_func = se.shots_grow_func
            grow_symmetrical = se.grow_symmetrical
            reference_index = se.reference_index
            release_shots = se.release_shots
            add_background = se.add_background
            frame_dir = se.frame_dir

            cur_frame = 0
            cur_grow_size = 0
            while cur_grow_size < max_shot_count:
                logger.info(f'Creating frame {cur_frame}')

                # recreate shots
                cur_grow_size += shots_grow_func(cur_frame)
                if grow_symmetrical:
                    first = max(reference_index - cur_grow_size, 0)
                else:
                    first = reference_index
                last = min(reference_index + cur_grow_size + 1, shot_count)

                shots_copy = [shot.create_anew() for shot in shots[first:last]]
                shot_loader = make_shot_loader(shots_copy)

                result = renderer.render_integral(shot_loader, mask=mask, save=False, release_shots=release_shots)
                if add_background:
                    result = overlay(background, result)
                im_pil = Image.fromarray(result)
                frame_file = f'{frame_dir}{cur_frame}.png'
                im_pil.save(frame_file)
                frame_files.append(frame_file)

                del shot_loader
                del result
                del im_pil

                cur_frame += 1

        # endregion

        _frame_processing(frame_files, se)

        with LoggerStep(logger, 'Release all resources'):
            release_all(ctx, renderer, shots)


if __name__ == '__main__':
    gltf_file = r"C:\Users\P41743\Desktop\New Folder\dem_mesh_r2.gltf"
    shot_json_file = r"C:\Users\P41743\Desktop\New folder\Frames_T\matched_poses.json"
    mask_file = r"C:\Users\P41743\Desktop\New folder\Frames_T\mask_T.png"
    settings = FocusAnimationSettings(initial_skip=False, add_background=True,
            camera_dist=30,
            camera_position_mode=CameraPositioningMode.CenterShot, fovy=50, aspect_ratio=1.0, orthogonal=True,
            ortho_size=(70, 70), correction=None, resolution=Resolution(2048, 2048))
    animate_focus(gltf_file, shot_json_file, mask_file)