import os
import time
from collections import defaultdict
from copy import copy
from functools import cache
from typing import Optional, Iterable, Callable, Sequence, cast, Protocol, Union, TypeVar, Type

import cv2
import moderngl as mgl
import numpy as np
from PIL import Image
from numpy import ndarray
from pyrr import Vector3

from src.core.defs import MAGENTA, BLACK, MAX_TEX_DIM, CPP_INT_MAX, PATH_SEP
from src.core.geo.aabb import AABB
from src.core.geo.transform import Transform
from src.core.rendering.camera import Camera
from src.core.rendering.data import TextureData, RenderResultMode, MeshData
from src.core.rendering.renderer import Renderer
from src.core.rendering.shot import CtxShot
from src.core.rendering.shot_loader import AsyncShotLoader
from src.core.util.basic import get_aabb, gen_checkerboard_tex, int_up, delete_all
from src.core.util.gltf import gltf_extract
from src.core.util.image import overlay
from src.core.util.video import video_from_images
from src.examples.rendering.data import CameraPositioningMode, BaseSettings, BaseAnimationSettings, IntegralSettings, \
    FocusAnimationSettings, ShutterAnimationSettings


class Releasable(Protocol):
    """
    Static duck typing class for anything that implements a method ``release() -> None``
    """

    def release(self) -> None:
        pass


class DoneCallback:
    """
    Logging callback that prints a message and the time that has passed since the last call
    """

    def __init__(self, indent: Optional[str] = None) -> None:
        self.indent = indent if indent is not None else ''
        self.start = time.time()
        self.last = self.start

    def _print(self, log_time: float, msg: str = 'Done', indent: bool = True) -> None:
        print(f'{self.indent if indent else ""}{msg + " "}[{log_time * 1000:.3f} ms]')

    def __call__(self, print_msg: bool = True, msg: str = 'Done', indent: bool = True) -> None:
        """
        Prints the time since the previous call and optionally a prefixed message
        :param print_msg: Whether a message should be printed
        :param msg: The message to be printed
        :param indent: Whether the message should be prefixed with the indent associated with this callback
        """
        current = time.time()
        if print_msg:
            self._print(current - self.last, msg, indent)
        self.last = current

    def total(self, msg: str = 'Done', indent: bool = True) -> None:
        current = time.time()
        self._print(current - self.start, msg, indent)


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

    lookup[CameraPositioningMode.background_centered] = _cpm_bc
    lookup[CameraPositioningMode.first_shot] = _cpm_fs
    lookup[CameraPositioningMode.center_shot] = _cpm_cs
    lookup[CameraPositioningMode.last_shot] = _cpm_ls
    lookup[CameraPositioningMode.average_shot] = _cpm_as
    lookup[CameraPositioningMode.shot_centered] = _cpm_sc

    return lookup


def _get_camera_position(mode: CameraPositioningMode, camera_dist: float, background_aabb: AABB,
                         shots: Sequence[CtxShot]) -> Vector3:
    center = _cpm_lookup()[mode](background_aabb, shots)
    center.z += camera_dist
    return center


# endregion

# region Common functionality

def make_done_callback() -> DoneCallback:
    """
    Creates a done callback. Ensures all examples use the same parameters when using a done callback.
    :return: A ``DoneCallback`` instance
    """
    return DoneCallback('    ')


def make_shot_loader(shots: Iterable[CtxShot]) -> Iterable[CtxShot]:
    """
    Creates a shot loader. Ensures all examples use the same parameters when using a shot loader.
    :return: A shot loader instance ensuring the ``Iterable[CtxShot]`` type.
    """
    return AsyncShotLoader(shots, 32, 8)


def read_gltf(gltf_file: str) -> tuple[Optional[MeshData], Optional[TextureData]]:
    """
    Tries to extract mesh and texture data from the given gltf file
    :param gltf_file: The path pointing at the GLTF file to process
    :return: The extracted mesh and texture data
    """
    return gltf_extract(gltf_file)


def process_render_data(mesh_data: Optional[MeshData], texture_data: Optional[TextureData]) -> tuple[MeshData, TextureData]:
    """
    Transforms data extracted from a GLTF file into valid mesh and texture data that can be used by the renderer
    :param mesh_data: The extracted mesh data
    :param texture_data: The extracted texture data
    :return: Processed mesh and texture data ready to be rendered
    """

    if mesh_data is None:
        raise ValueError('Mesh data could not be extracted')

    if texture_data is None:
        texture_data = TextureData(gen_checkerboard_tex(8, 8, MAGENTA, BLACK, dtype='f4'))
        print(f'    No texture extracted: Default texture was generated')
    else:
        byte_size = texture_data.byte_size(dtype='f4')
        width, height = texture_data.texture.shape[1::-1]
        print(f'    Texture extracted: ({width}, {height}) x {texture_data.texture.shape[2]} [{byte_size} B]')

        if width > MAX_TEX_DIM or height > MAX_TEX_DIM:
            if width > height:
                fact = MAX_TEX_DIM / width
            else:
                fact = MAX_TEX_DIM / height
            texture_data.texture = cv2.resize(texture_data.texture, None, fx=fact, fy=fact)
            print(f'    Texture downscaled to {texture_data.texture.shape[1::-1]} [{texture_data.byte_size("f4")} B] '
                  f'to fit texture dimension restriction of {MAX_TEX_DIM}px')
            byte_size = texture_data.byte_size(dtype='f4')

        if byte_size > CPP_INT_MAX:
            texture_data.scale_to_fit(CPP_INT_MAX, dtype='f4')  # necessary since moderngl uses this data type
            print(f'    Texture downscaled to {texture_data.texture.shape[1::-1]} [{texture_data.byte_size("f4")} B] '
                  f'to fit size restriction of {CPP_INT_MAX} B')

    return mesh_data, texture_data


def make_mgl_context() -> mgl.Context:
    """
    Creates a ModernGL context. Ensures all examples use the same context settings.
    :return: A ModernGL context instance.
    """
    ctx = mgl.create_context(standalone=True)
    ctx.enable(cast(int, mgl.DEPTH_TEST))
    ctx.enable(cast(int, mgl.CULL_FACE))
    ctx.cull_face = 'back'
    return ctx


def read_shots(json_file: str, ctx: mgl.Context, se: BaseSettings) -> list[CtxShot]:
    """
    Reads the shots from a JSON file and filters them according to the given settings
    :param json_file: The shot JSON file
    :param ctx: The context the shots should be associated with
    :param se: The basic render settings to be used
    :return: A list of context related shots
    """
    shots = CtxShot.from_json(json_file, ctx, count=se.count + se.initial_skip, correction=se.correction, lazy=se.lazy)
    return shots[se.initial_skip::se.skip]


def make_camera(mesh_aabb: AABB, shots: Sequence[CtxShot], se: BaseSettings) -> Camera:
    """
    Creates a camera. Ensures all render examples create the camera the same way.
    :param mesh_aabb: The AABB of the mesh to render
    :param shots: The shots to render
    :param se: The basic settings associated with the render process
    :return: A ``Camera`` instance
    """
    if se.orthogonal:
        # ensure valid ortho_size
        ortho_size = se.ortho_size if se.ortho_size is not None else (int_up(mesh_aabb.width), int_up(mesh_aabb.height))
    else:
        ortho_size = se.ortho_size

    camera_pos = _get_camera_position(se.camera_position_mode, se.camera_dist, mesh_aabb, shots)
    return Camera(fovy=se.fovy, aspect_ratio=se.aspect_ratio, orthogonal=se.orthogonal, orthogonal_size=ortho_size,
                  position=camera_pos, near=se.near_clipping, far=se.far_clipping)


def read_mask(mask_file: str) -> TextureData:
    """
    Reads a binary mask from the given file path and converts it into texture data
    :param mask_file: The mask file to be used
    :return: Texture data resembling the given mask
    """
    mask_img = cv2.imread(mask_file)
    mask_img = mask_img[..., 0].astype('f4')
    mask_img = np.resize(mask_img, (*mask_img.shape, 1))
    mask_img /= 255.0
    return TextureData(mask_img)


def release_all(*releasables: Union[Releasable, Iterable[Releasable]]) -> None:
    """
    Calls the release method for all given objects
    :param releasables: All objects to release
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
    :param frame_files: The locations of the frames that should be concatenated to a video
    :param se: The basic animation settings
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


def _base_steps(done: DoneCallback, gltf_file: str, shot_json_file: str, mask_file: Optional[str],
                se: BaseSettings) -> tuple[mgl.Context, Camera, Renderer, list[CtxShot], Optional[TextureData]]:
    print(f'  Reading GLTF file from "{gltf_file}"')
    mesh_data, texture_data = read_gltf(gltf_file)
    mesh_data, texture_data = process_render_data(mesh_data, texture_data)
    done()

    print('  Creating MGL context')
    ctx = make_mgl_context()
    done()

    print(f'  Extracting shots from "{shot_json_file}" (Creating lazy shots: {se.lazy})')
    shots = read_shots(shot_json_file, ctx, se)
    done()

    print(f'  Creating camera and renderer (camera position mode: {se.camera_position_mode.name})')
    mesh_aabb = get_aabb(mesh_data.vertices)
    camera = make_camera(mesh_aabb, shots, se)
    print(f'    Computed camera position: {camera.transform.position}')
    renderer = Renderer(se.resolution, ctx, camera, mesh_data, texture_data)
    done()

    if mask_file is not None:
        print(f'  Reading mask from "{mask_file}"')
        mask = read_mask(mask_file)
        done()
    else:
        mask = None

    return ctx, camera, renderer, shots, mask


def _integral_processing(done: DoneCallback, renderer: Renderer, integral: ndarray, se: IntegralSettings) -> None:
    # region Adding Background

    if se.add_background:
        print('  Rendering background')
        background = cv2.cvtColor(renderer.render_background(), cv2.COLOR_BGRA2RGBA)
        done()

        print('  Laying integral over background')
        img = cv2.cvtColor(integral, cv2.COLOR_BGRA2RGBA)
        img = overlay(background, img)
        im_pil = Image.fromarray(img)
        done()
    else:
        print(' Converting array to PIL image')
        img = cv2.cvtColor(integral, cv2.COLOR_BGRA2RGBA)
        im_pil = Image.fromarray(img)

    # endregion

    # region Showing Integral

    if se.show_integral:
        print('  Showing integral')
        im_pil.show('Integral')
        done()

    # endregion

    # region Saving Integral

    print(f'  Saving integral image to "{se.output_file}"')
    if '.' not in se.output_file:
        se.output_file += '.png'
    im_pil.save(se.output_file)
    done()

    # endregion


def _frame_processing(done: DoneCallback, frame_files: Sequence[str], se: BaseAnimationSettings) -> None:
    print('  Creating video file')
    create_video(frame_files, se)
    done()

    if se.delete_frames:
        print('  Deleting frames')
        delete_all(frame_files)
        done()


# endregion

def project_shots(gltf_file: str, shot_json_file: str, mask_file: Optional[str] = None,
                  settings: Optional[IntegralSettings] = None) -> None:
    done = make_done_callback()
    print('Start projection process')

    # region Initializing

    print('    Initializing')
    se = _ensure_or_copy_settings(settings, IntegralSettings)
    done()

    # endregion

    ctx, _, renderer, shots, mask = _base_steps(done, gltf_file, shot_json_file, mask_file, se)

    # region Projecting Shots

    print(f'  Projecting shots (Releasing shots after projection: {se.release_shots})')
    shot_loader = make_shot_loader(shots)
    result = renderer.project_shots(shot_loader, RenderResultMode.shot_only, mask=mask, integral=True, save=False,
                                    release_shots=se.release_shots)
    done()

    # endregion

    _integral_processing(done, renderer, result, se)

    print('  Release all resources')
    release_all(ctx, renderer, shots)
    done()

    done.total(msg='All done', indent=False)


def render_integral(gltf_file: str, shot_json_file: str, mask_file: Optional[str] = None,
                    settings: Optional[IntegralSettings] = None) -> None:
    done = make_done_callback()
    print('Start projection process')

    # region Initializing

    print('    Initializing')
    se = _ensure_or_copy_settings(settings, IntegralSettings)
    done()

    # endregion

    ctx, _, renderer, shots, mask = _base_steps(done, gltf_file, shot_json_file, mask_file, se)

    # region Rendering Integral

    print(f'  Projecting shots (Releasing shots after projection: {se.release_shots})')
    shot_loader = make_shot_loader(shots)
    result = renderer.render_integral(shot_loader, mask=mask, save=False, release_shots=se.release_shots)
    done()

    # endregion

    _integral_processing(done, renderer, result, se)

    print('  Release all resources')
    release_all(ctx, renderer, shots)
    done()

    done.total(msg='All done', indent=False)


def animate_focus(gltf_file: str, shot_json_file: str, mask_file: Optional[str] = None,
                  settings: Optional[FocusAnimationSettings] = None) -> None:
    done = make_done_callback()
    print('Start focus animation process')

    # region Initializing

    print('    Initializing')
    se = _ensure_or_copy_settings(settings, FocusAnimationSettings)

    if se.correction is None:
        se.correction = Transform()

    se.correction.position.z = se.start_focus

    done()

    # endregion

    ctx, camera, renderer, shots, mask = _base_steps(done, gltf_file, shot_json_file, mask_file, se)

    # region Frame Rendering

    print(f'  Creating Frames (Frames to be rendered: {se.frame_count}; Focus: {se.start_focus} -> {se.end_focus})')
    frame_done = DoneCallback('      ')

    if not se.frame_dir.endswith(PATH_SEP):
        se.frame_dir += PATH_SEP
    os.makedirs(se.frame_dir, exist_ok=True)

    range_focus = se.end_focus - se.start_focus
    focus_step = range_focus / se.frame_count
    print(f'    Focus step: {focus_step}')
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
        background = cv2.cvtColor(renderer.render_background(), cv2.COLOR_BGRA2RGBA)
    else:
        background = None

    for i in range(se.frame_count):
        print(f'    Creating frame {i}')
        shots_copy = [shot.create_anew() for shot in shots]
        shot_loader = make_shot_loader(shots_copy)
        result = renderer.render_integral(shot_loader, mask=mask, save=False, release_shots=release_shots)
        img = cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA)

        if add_background:
            if move_camera_with_focus:
                background = cv2.cvtColor(renderer.render_background(), cv2.COLOR_BGRA2RGBA)
            img = overlay(background, img)

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

        frame_done()
    done()

    # endregion

    _frame_processing(done, frame_files, se)

    print('  Release all resources')
    release_all(ctx, renderer, shots)
    done()

    done.total(msg='All done', indent=False)


def animate_shutter(gltf_file: str, shot_json_file: str, mask_file: Optional[str] = None,
                    settings: Optional[ShutterAnimationSettings] = None) -> None:
    done = make_done_callback()
    print('Start shutter animation process')

    # region Initializing
    print('    Initializing')
    se = _ensure_or_copy_settings(settings, ShutterAnimationSettings)
    done()

    # endregion

    ctx, camera, renderer, shots, mask = _base_steps(done, gltf_file, shot_json_file, mask_file, se)

    # region Render Background

    if se.add_background:
        print('  Rendering background')
        background = cv2.cvtColor(renderer.render_background(), cv2.COLOR_BGRA2RGBA)
        done()
    else:
        background = None

    # endregion

    # region Frame Rendering

    print(f'  Creating Frames (Frames to be rendered: {se.frame_count})')
    frame_done = DoneCallback('      ')

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
        print(f'    Creating frame {cur_frame}')

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
        img = cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA)
        if add_background:
            img = overlay(background, img)
        im_pil = Image.fromarray(img)
        frame_file = f'{frame_dir}{cur_frame}.png'
        im_pil.save(frame_file)
        frame_files.append(frame_file)

        del shot_loader
        del img
        del im_pil

        cur_frame += 1
        frame_done()
    done()

    # endregion

    _frame_processing(done, frame_files, se)

    print('  Release all resources')
    release_all(ctx, renderer, shots)
    done()

    done.total(msg='All done', indent=False)
