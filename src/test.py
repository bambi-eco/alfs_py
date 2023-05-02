import os
import time
from collections import defaultdict
from functools import cache
from typing import Final, Optional, Sequence, Callable, cast, Iterable

import cv2
import moderngl as mgl
import numpy as np
from PIL import Image
from pyrr import Matrix44, Vector3, Quaternion

from src.core.camera import Camera
from src.core.data import TextureData, ProjectionSettings, FocusAnimationSettings, CameraPositioningMode, AABB, \
    ShutterAnimationSettings, BaseSettings, BaseAnimationSettings
from src.core.defs import OUTPUT_DIR, INPUT_DIR, DEF_FRAG_SHADER_PATH, \
    DEF_VERT_SHADER_PATH, DEF_PASS_VERT_SHADER_PATH, DEF_PASS_FRAG_SHADER_PATH, CPP_INT_MAX, MAGENTA, BLACK, \
    MAX_TEX_DIM, PATH_SEP
from src.core.geo.transform import Transform
from src.core.iters import file_name_gen
from src.core.renderer import Renderer, ProjectMode
from src.core.shot import CtxShot
from src.core.shot_loader import AsyncShotLoader
from src.core.util.basic import get_center, int_up, make_quad, gen_checkerboard_tex, get_aabb
from src.core.util.gltf import gltf_extract
from src.core.util.image import crop_to_content, split_components, overlay
from src.core.util.moderngl import img_from_fbo
from src.core.util.video import video_from_images

_OUTPUT_RESOLUTION: Final[tuple[int, int]] = (1024 * 2, 1024 * 2)  # (1024 * 4, 1024 * 4)
_CLEAR_COLOR: Final[tuple[float, ...]] = (1.0, 0.0, 1.0, 0.1)

_FOV: Final[float] = 45.0
_NEAR_CLIP: Final[float] = 0.1
_FAR_CLIP: Final[float] = 10000.0


class DoneCallback:

    def __init__(self, indent: Optional[str] = None):
        self.indent = indent if indent is not None else ''
        self.start = time.time()
        self.last = self.start

    def __call__(self, print_msg: bool = True):
        current = time.time()
        if print_msg:
            print(f'{self.indent}Done [{(current - self.last) * 1000:.3f} ms]')
        self.last = current

    def all_done(self, print_msg: bool = True):
        current = time.time()
        if print_msg:
            print(f'All Done [{(current - self.start) * 1000:.3f} ms]')
        self.start = current

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

def _get_camera_position(mode: CameraPositioningMode, camera_dist: float, background_aabb: AABB, shots: Sequence[CtxShot]) -> Vector3:
    center = _cpm_lookup()[mode](background_aabb, shots)
    center.z += camera_dist
    return center

def _initial_steps(done: DoneCallback, gltf_file: str, shot_json_file: str, mask_file: Optional[str], se: BaseSettings)\
        -> tuple[mgl.Context, Camera, Renderer, list[CtxShot], Optional[TextureData]]:

    # region Reading GLTF

    print(f'  Reading GLTF file from "{gltf_file}"')
    mesh_data, texture_data = gltf_extract(gltf_file)

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
    done()

    # endregion

    # region Creating MGL context

    print('  Creating MGL context')
    ctx = mgl.create_context(standalone=True)
    ctx.enable(cast(int, mgl.DEPTH_TEST))
    ctx.enable(cast(int, mgl.CULL_FACE))
    ctx.cull_face = 'back'
    done()

    # endregion

    # region Reading Shots

    print(f'  Extracting shots from "{shot_json_file}" (Creating lazy shots: {se.lazy})')
    shots = CtxShot.from_json(shot_json_file, ctx, count=se.count + se.initial_skip, correction=se.correction, lazy=se.lazy)
    shots = shots[se.initial_skip::se.skip]
    done()

    # endregion

    # region Creating Renderer

    print(f'  Creating camera and renderer (camera position mode: {se.camera_position_mode.name})')
    aabb = get_aabb(mesh_data.vertices)
    ortho_size = se.ortho_size if se.ortho_size is not None else (int_up(aabb.width), int_up(aabb.height))
    camera_pos = _get_camera_position(se.camera_position_mode, se.camera_dist, aabb, shots)
    print(f'    Computed camera position: {camera_pos}')
    camera = Camera(fovy=se.fovy, aspect_ratio=se.aspect_ratio, orthogonal=se.orthogonal, orthogonal_size=ortho_size,
                    position=camera_pos, near=se.near_clipping, far=se.far_clipping)
    renderer = Renderer(se.resolution, ctx, camera, mesh_data, texture_data)
    done()

    # endregion

    # region Reading Mask

    if mask_file is not None:
        print(f'  Reading mask from "{mask_file}"')
        mask_img = cv2.imread(mask_file)
        mask_img = mask_img[..., 0].astype('f4')
        mask_img = np.resize(mask_img, (*mask_img.shape, 1))
        mask_img /= 255.0
        mask = TextureData(mask_img)
        done()
    else:
        mask = None

    # endregion

    return ctx, camera, renderer, shots, mask

def _release_all(done: DoneCallback, ctx: mgl.Context, renderer: Renderer, shots: Iterable[CtxShot]) -> None:
    print('  Release all resources')
    for shot in shots:
        shot.release()

    renderer.release()
    ctx.release()
    done()

def _create_video(done: DoneCallback, frame_files: Sequence[str], se: BaseAnimationSettings) -> None:

    # region Creating Video

    print('  Creating video file')
    video_from_images(frame_files, se.output_file, fps=se.fps, release_images=True,
                      first_frame_repetitions=se.first_frame_repetitions,
                      last_frame_repetitions=se.last_frame_repetitions)
    done()

    # endregion

    # region Deleting Frames

    if se.delete_frames:
        print('  Deleting frames')
        for frame_file in frame_files:
            os.remove(frame_file)
        done()

    # endregion

def test_projection(gltf_file: str, shot_json_file: str, mask_file: Optional[str] = None,
                    settings: Optional[ProjectionSettings] = None) -> None:
    done = DoneCallback('    ')
    print('Start projection process')

    # region Initializing

    print('    Initializing')
    if settings is None:
        settings = ProjectionSettings()
    add_background = settings.add_background
    release_shots = settings.release_shots
    show_integral = settings.show_integral
    output_file = settings.output_file
    done()

    # endregion

    ctx, _, renderer, shots, mask = _initial_steps(done, gltf_file, shot_json_file, mask_file, settings)

    # region Projecting Shots

    print(f'  Projecting shots (Releasing shots after projection: {release_shots})')
    shot_loader = AsyncShotLoader(shots, 15, 8)
    result = renderer.project_shots(shot_loader, ProjectMode.SHOT_VIEW_RELATIVE, mask=mask, integral=True, save=False,
                                     release_shots=release_shots)
    done()

    # endregion

    # region Adding Background

    if add_background:
        print('  Rendering background')
        background = cv2.cvtColor(renderer.render_ground(), cv2.COLOR_BGRA2RGBA)
        done()

        print('  Laying integral over background')
        img = cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA)
        img = overlay(background, img)
        im_pil = Image.fromarray(img)
        done()
    else:
        print(' Converting array to PIL image')
        img = cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA)
        im_pil = Image.fromarray(img)

    # endregion

    # region Showing Integral

    if show_integral:
        print('  Showing integral')
        im_pil.show('Integral')
        done()

    # endregion

    # region Saving Integral

    print(f'  Saving integral image to "{output_file}"')
    if '.' not in output_file:
        output_file += '.png'
    im_pil.save(output_file)
    done()

    # endregion

    _release_all(done, ctx, renderer, shots)

    done.all_done()


def test_focus_animation(gltf_file: str, shot_json_file: str, mask_file: Optional[str] = None,
                         settings: Optional[FocusAnimationSettings] = None) -> None:
    done = DoneCallback('    ')
    print('Start focus animation process')

    # region Initializing

    print('    Initializing')
    if settings is None:
        settings = ProjectionSettings()

    if settings.correction is None:
        settings.correction = Transform()

    start_focus = settings.start_focus
    end_focus = settings.end_focus
    frame_count = settings.frame_count
    add_background = settings.add_background
    move_camera_with_focus = settings.move_camera_with_focus
    release_shots = settings.release_shots
    frame_dir = settings.frame_dir

    settings.correction.position.z = start_focus

    done()

    # endregion

    ctx, camera, renderer, shots, mask = _initial_steps(done, gltf_file, shot_json_file, mask_file, settings)

    # region Frame Rendering

    print(f'  Creating Frames (Frames to be rendered: {frame_count}; Focus: {start_focus} -> {end_focus})')
    frame_done = DoneCallback('      ')

    if not frame_dir.endswith(PATH_SEP):
        frame_dir += PATH_SEP
    os.makedirs(frame_dir, exist_ok=True)

    range_focus = end_focus - start_focus
    focus_step = range_focus / frame_count
    print(f'    Focus step: {focus_step}')
    frame_files = []

    if move_camera_with_focus:
        camera.transform.position.z -= start_focus
        renderer.apply_matrices()

    for i in range(frame_count):
        print(f'    Creating frame {i}')
        shots_copy = [shot.create_anew() for shot in shots]
        shot_loader = AsyncShotLoader(shots_copy, 25, 8)
        result = renderer.project_shots(shot_loader, ProjectMode.SHOT_VIEW_RELATIVE, mask=mask, integral=True, save=False,
                                        release_shots=release_shots)
        img = cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA)
        if add_background:
            background = cv2.cvtColor(renderer.render_ground(), cv2.COLOR_BGRA2RGBA)
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

    _create_video(done, frame_files, settings)

    _release_all(done, ctx, renderer, shots)

    done.all_done()


def test_shutter_animation(gltf_file: str, shot_json_file: str, mask_file: Optional[str] = None,
                           settings: Optional[ShutterAnimationSettings] = None) -> None:
    done = DoneCallback('    ')
    print('Start shutter animation process')

    # region Initializing
    print('    Initializing')
    if settings is None:
        settings = ShutterAnimationSettings()

    if settings.correction is None:
        settings.correction = Transform()

    add_background = settings.add_background
    release_shots = settings.release_shots
    frame_count = settings.frame_count
    frame_dir = settings.frame_dir
    shots_grow_func = settings.shots_grow_func
    reference_index = settings.reference_index
    grow_symmetrical = settings.grow_symmetrical

    done()

    # endregion

    ctx, camera, renderer, shots, mask = _initial_steps(done, gltf_file, shot_json_file, mask_file, settings)

    # region Adding Background

    if add_background:
        print('  Rendering background')
        background = cv2.cvtColor(renderer.render_ground(), cv2.COLOR_BGRA2RGBA)
        done()
    else:
        background = None

    # endregion

    # region Frame Rendering

    print(f'  Creating Frames (Frames to be rendered: {frame_count})')
    frame_done = DoneCallback('      ')

    if not frame_dir.endswith(PATH_SEP):
        frame_dir += PATH_SEP
    os.makedirs(frame_dir, exist_ok=True)

    frame_files = []

    shot_count = len(shots)
    max_shot_count = shot_count - reference_index

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

        shots_copy =  [shot.create_anew() for shot in shots[first:last]]
        shot_loader = AsyncShotLoader(shots_copy, 25, 8)


        result = renderer.project_shots(shot_loader, ProjectMode.SHOT_VIEW_RELATIVE, mask=mask, integral=True, save=False,
                                        release_shots=release_shots)
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

    _create_video(done, frame_files, settings)

    _release_all(done, ctx, renderer, shots)

    done.all_done()

def test_deferred_shading() -> None:
    file = f'{INPUT_DIR}mesh.glb'

    mesh_data, tex_data = gltf_extract(file)

    vertices = mesh_data.vertices
    indices = mesh_data.indices

    center, aabb = get_center(mesh_data.vertices)
    center.z = 750
    ortho_size = int_up(aabb.width), int_up(aabb.height)

    camera = Camera(orthogonal=True, orthogonal_size=ortho_size, position=center)
    projection = camera.get_proj()
    view = camera.get_view()
    model = Matrix44.identity(dtype='f4')

    ctx = mgl.create_context(standalone=True)
    ctx.enable(cast(int, mgl.DEPTH_TEST))

    # first pass
    with open(DEF_PASS_VERT_SHADER_PATH) as file:
        vert_shader = file.read()
    with open(DEF_PASS_FRAG_SHADER_PATH) as file:
        frag_shader = file.read()
    prog = ctx.program(vertex_shader=vert_shader, fragment_shader=frag_shader,
                       fragment_outputs={'f_out_v4_dir': 0, 'f_out_v4_dist': 1})

    prog['u_m4_proj'].write(projection.tobytes())
    prog['u_m4_view'].write(view.tobytes())
    prog['u_m4_model'].write(model.tobytes())

    dir_rbo = ctx.renderbuffer(_OUTPUT_RESOLUTION, 4)
    dist_rbo = ctx.renderbuffer(_OUTPUT_RESOLUTION, 4)
    dbo = ctx.depth_renderbuffer(_OUTPUT_RESOLUTION)
    fbo = ctx.framebuffer([dir_rbo, dist_rbo], dbo)
    fbo.use()

    vbo = ctx.buffer(vertices.tobytes())
    ibo = ctx.buffer(indices.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, '3f4', 'v_in_v3_pos')], index_buffer=ibo, index_element_size=4)

    ctx.clear(*_CLEAR_COLOR)
    vao.render(cast(int, mgl.TRIANGLES))

    # copy color attachments of first pass into textures
    dir_tex = ctx.texture(_OUTPUT_RESOLUTION, 4)
    dist_tex = ctx.texture(_OUTPUT_RESOLUTION, 4)
    fbo_tex = ctx.framebuffer((dir_tex, dist_tex))
    ctx.copy_framebuffer(fbo_tex, fbo)

    for releasable in (fbo, dbo, dist_rbo, dir_rbo, vao, ibo, vbo, prog):
        releasable.release()

    # second Pass
    with open(DEF_VERT_SHADER_PATH) as file:
        vert_shader = file.read()
    with open(DEF_FRAG_SHADER_PATH) as file:
        frag_shader = file.read()
    prog = ctx.program(vertex_shader=vert_shader, fragment_shader=frag_shader)

    # bind results of previous pass
    dir_tex.use(0)
    dist_tex.use(1)

    quad = make_quad()
    x, y, z = split_components(quad.vertices)
    u, v = split_components(quad.uvs)
    shader_data = np.dstack([x, y, z, u, v])
    vbo = ctx.buffer(shader_data.tobytes())
    ibo = ctx.buffer(indices.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, '3f4 2f4', 'v_in_v3_pos', 'v_in_v2_uv')],
                           index_buffer=ibo, index_element_size=4)
    fbo = ctx.simple_framebuffer(_OUTPUT_RESOLUTION, components=4)
    fbo.use()
    fbo.clear()
    vao.render()
    render = img_from_fbo(fbo, attachment=0)

    cv2.imwrite(f'{OUTPUT_DIR}def_render.png', render)

    for releasable in (fbo, vao, ibo, vbo, dist_tex, dir_tex, fbo_tex, prog, ctx):
        releasable.release()

    # TODO: fix second pass not rendering anything


def test_fit_to_points(count: int = 1):
    ctx = mgl.create_context(standalone=True)
    ctx.enable(cast(int, mgl.DEPTH_TEST))

    gltf_file = f'{INPUT_DIR}mesh.glb'
    mesh_data, texture_data = gltf_extract(gltf_file)

    center, aabb = get_center(mesh_data.vertices)
    # center.z = 1
    # ortho_size = int_up(aabb.width), int_up(aabb.height)

    camera = Camera(orthogonal=False, position=Vector3([center.x, center.y, 100.0]))
    camera.look_at(center)
    corners = camera._frustum.corners
    for i, corner in enumerate(corners):
        print(f'Corner {chr(65 + i)} = ({corner.x:.9f}, {corner.y:.9f}, {corner.z:.9f})')

    # camera.transform.position.z = 600

    # print(camera.transform.position)
    # camera.fit_to_points(aabb.corners, 0.0)
    # print(camera.transform.position)
    renderer = Renderer(_OUTPUT_RESOLUTION, ctx, camera, mesh_data, texture_data)

    json_file = f'{INPUT_DIR}data\\1\\poses.json'
    shots = CtxShot.from_json(json_file, ctx, count=count)

    file_name_iter = file_name_gen('.png', f'{OUTPUT_DIR}proj')
    results = renderer.project_shots(shots, ProjectMode.COMPLETE_VIEW, save=False, save_name_iter=file_name_iter)

    res_center = Vector3([_OUTPUT_RESOLUTION[0] / 2.0, _OUTPUT_RESOLUTION[1] / 2.0, 0.0])
    res_center_tup = int_up(res_center[0]), int_up(res_center[1])
    for result in results:
        crop, delta = crop_to_content(result, return_delta=True)
        h, w = crop.shape[0:2]
        c_d = Vector3([w / 2.0, h / 2.0, 0])
        print(f'tl: {res_center + delta - c_d} | br: {res_center + delta + c_d}')

        res_delta = (int_up(res_center[0] + delta[0]), int_up(res_center[1] + delta[1]))
        cv2.line(result, res_center_tup, res_delta, (255, 0, 0), 5)
        result = cv2.resize(result, None, None, 0.5, 0.5)
        cv2.imshow('delta', result)
        cv2.waitKey()
        cv2.destroyAllWindows()

    for shot in shots:
        shot.release()

    renderer.release()
    ctx.release()


def main() -> None:
    # bambi_data_dir = 'D:\\BambiData\\'
    frame_type = 't'
    gltf_file = rf'D:\BambiData\DEM\Hagenberg\dem_mesh_r2.glb'
    shot_json_file = rf'D:\BambiData\Processed\Hagenberg\KFV-hgb-Enew\Frames_{frame_type}\poses.json'
    mask_file = rf'D:\BambiData\Processed\Hagenberg\KFV-hgb-Enew\Frames_{frame_type}\mask_{frame_type}.png'
    count = 200
    if frame_type == 't':
        center = 35380
    elif frame_type == 'w':
        center = 35820
    else:
        center = count // 2

    first = center - count // 2

    # correction = Transform()
    # correction.position.z = 2.0
    # correction.rotation = Quaternion.from_z_rotation(-np.deg2rad(1.0), dtype='f4')
    # output_file = rf'{OUTPUT_DIR}integral'
    # settings = ProjectionSettings(count=count, initial_skip=first, camera_dist=0.0,
    #                               camera_position_mode=CameraPositioningMode.shot_centered,
    #                               correction=correction, resolution=(1024, 1024), orthogonal=False, show_integral=True,
    #                               output_file=output_file)
    # test_projection(gltf_file, shot_json_file, mask_file, settings)

    # fps = 3
    # duration = 2
    # start_focus = 22
    # end_focus = 2
    # correction = Transform()
    # correction.rotation = Quaternion.from_z_rotation(-np.deg2rad(1.0), dtype='f4')
    # output_file = rf'{OUTPUT_DIR}focus_anim'
    # settings = FocusAnimationSettings(
    #     start_focus=start_focus, end_focus=end_focus, frame_count=duration * fps, fps=fps,
    #     count=count, initial_skip=first, add_background=False, fovy=50.0, camera_dist=0.0, #-17.5,
    #     camera_position_mode=CameraPositioningMode.center_shot, move_camera_with_focus=True,
    #     resolution=(1024*2, 1024*2), aspect_ratio=1.0, orthogonal=False, ortho_size=(65, 65),
    #     delete_frames=False, first_frame_repetitions=fps, last_frame_repetitions=fps, output_file=output_file)
    # test_focus_animation(gltf_file, shot_json_file, mask_file, settings)

    shots_grow_func = lambda x: int(np.ceil(np.exp(x * 0.2 - 0.8)))
    correction = Transform()
    correction.position.z = 22
    correction.rotation = Quaternion.from_z_rotation(-np.deg2rad(1.0), dtype='f4')
    output_file = rf'{OUTPUT_DIR}shutter_anim_{frame_type}_22'
    settings = ShutterAnimationSettings(
        shots_grow_func=shots_grow_func, reference_index=center - first, grow_symmetrical=True,
        count=count, initial_skip=first, add_background=False, fovy=50.0, camera_dist=-22.0,
        camera_position_mode=CameraPositioningMode.center_shot,
        frame_dir='./.frames/22',
        resolution=(1024*2, 1024*2), aspect_ratio=1.0, orthogonal=False, ortho_size=(65, 65),
        correction=correction,
        delete_frames=False, output_file=output_file
    )
    test_shutter_animation(gltf_file, shot_json_file, mask_file, settings)

    settings.correction.position.z = 2.0
    settings.camera_dist = -2.0
    settings.output_file = rf'{OUTPUT_DIR}shutter_anim_{frame_type}_2'
    settings.frame_dir = './.frames/2'

    test_shutter_animation(gltf_file, shot_json_file, mask_file, settings)

if __name__ == '__main__':
    main()

# TODO: Animation of increasing shot count with amount of used shots growing exponentially ( ceil(exp(x * 0.2 - 0.8)) )
# TODO: Test direct sharepoint loading
# TODO: Maybe try shifting integral calculations onto GPU using additional shader
# TODO: Fix shot reloading for future dgx use