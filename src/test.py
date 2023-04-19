import time
from typing import Final, Optional

import cv2
import moderngl as mgl
import numpy as np
from PIL import Image
from pyrr import Matrix44, Vector3, Quaternion

from src.core.camera import Camera
from src.core.data import TextureData, ProjectionSettings
from src.core.defs import OUTPUT_DIR, INPUT_DIR, DEF_FRAG_SHADER_PATH, \
    DEF_VERT_SHADER_PATH, DEF_PASS_VERT_SHADER_PATH, DEF_PASS_FRAG_SHADER_PATH, CPP_INT_MAX, MAGENTA, BLACK, MAX_TEX_DIM
from src.core.geo.transform import Transform
from src.core.iters import file_name_gen
from src.core.renderer import Renderer, ProjectMode
from src.core.shot import CtxShot
from src.core.shot_loader import AsyncShotLoader
from src.core.util.basic import get_center, int_up, make_quad, gen_checkerboard_tex, get_vector_center
from src.core.util.gltf import gltf_extract
from src.core.util.image import crop_to_content, split_components, integral, overlay, replace_color, laplacian_variance
from src.core.util.moderngl import img_from_fbo

_OUTPUT_RESOLUTION: Final[tuple[int, int]] = (2109, 4096)  # (1024 * 4, 1024 * 4)
_CLEAR_COLOR: Final[tuple[float, ...]] = (1.0, 0.0, 1.0, 0.1)

_FOV: Final[float] = 45.0
_NEAR_CLIP: Final[float] = 0.1
_FAR_CLIP: Final[float] = 10000.0


class DoneCallback:

    def __init__(self):
        self.start = time.time()
        self.last = self.start

    def __call__(self, print_msg: bool = True):
        current = time.time()
        if print_msg:
            print(f'    Done [{(current - self.last) * 1000:.3f} ms]')
        self.last = current


def test_projection(gltf_file: str, json_file: str, mask_file: Optional[str] = None, settings: Optional[ProjectionSettings] = None):
    done = DoneCallback()
    print('Start projection process')

    count = settings.count
    initial_skip = settings.initial_skip
    skip = settings.skip
    lazy = settings.lazy
    release_shots = settings.release_shots
    correction = settings.correction
    output_file = settings.output_file

    print('    Creating MGL context')
    ctx = mgl.create_context(standalone=True)
    ctx.enable(mgl.DEPTH_TEST)
    ctx.enable(mgl.CULL_FACE)
    ctx.cull_face = 'back'
    done()

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

    center, aabb = get_center(mesh_data.vertices)
    center.z += aabb.depth + 100
    ortho_size = int_up(aabb.width), int_up(aabb.height)
    done()

    print(f'  Creating context and renderer')
    camera = Camera(orthogonal=True, orthogonal_size=ortho_size, position=center, far=100000)
    renderer = Renderer(_OUTPUT_RESOLUTION, ctx, camera, mesh_data, texture_data)
    done()

    print(f'  Extracting shots from "{json_file}" (Creating lazy shots: {lazy})')
    shots = CtxShot.from_json(json_file, ctx, count=count + initial_skip, correction=correction, lazy=lazy)
    shots = shots[initial_skip::skip]
    # shot_loader = SyncShotLoader(shots)
    shot_loader = AsyncShotLoader(shots, 15, 8)
    done()

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

    print('  Rendering background')
    background = cv2.cvtColor(renderer.render_ground(), cv2.COLOR_BGRA2RGBA)
    cv2.imwrite(f'{OUTPUT_DIR}back.png', cv2.cvtColor(background, cv2.COLOR_BGRA2RGBA))
    done()

    print(f'  Projecting shots (Releasing shots after projection: {release_shots})')
    results = renderer.project_shots(shot_loader, ProjectMode.SHOT_VIEW_RELATIVE, mask=mask, integral=True, save=False,
                                     release_shots=True)
    done()

    print('  Showing integral')
    result = results
    img = cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA)
    img = overlay(background, img)
    im_pil = Image.fromarray(img)
    im_pil.show('Integral')
    done()

    print(f'  Saving integral image to "{output_file}"')
    im_pil.save(output_file)
    done()

    print('  Release all resources')
    for shot in shots:
        shot.release()

    renderer.release()
    ctx.release()
    done()


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
    ctx.enable(mgl.DEPTH_TEST)

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
    vao.render(mgl.TRIANGLES)

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
    ctx.enable(mgl.DEPTH_TEST)

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
    bambi_data_dir = 'D:\\BambiData\\'
    gltf_file = rf'{bambi_data_dir}DEM\Hagenberg\dem_mesh_r2.glb'
    json_file = rf'{bambi_data_dir}Processed\Hagenberg\NeRF Grid\Frames_T\poses.json'
    mask_file = rf'{bambi_data_dir}Processed\Hagenberg\NeRF Grid\Frames_T\mask_T.png'

    correction = Transform()
    correction.position.z = 2
    correction.rotation = Quaternion.from_z_rotation(np.deg2rad(1.0), dtype='f4')

    output_file = f'{OUTPUT_DIR}integral.png'

    settings = ProjectionSettings(count=2, initial_skip=0, correction=correction, output_file=output_file)

    test_projection(gltf_file, json_file, mask_file, settings)

    # vals = np.arange(-1, 1.05, 0.1) * 0.08726646
    # for val in vals:
    # correction_rot = Quaternion.from_z_rotation(0.08726646, dtype='f4')
    # correction.rotation = correction_rot
    # test_projection(100, 0, ProjectMode.SHOT_VIEW_RELATIVE, correction, f'{0.08726646*1000:.0f}')
    # test_load_all_images(f'{INPUT_DIR}data\\haag\\')


if __name__ == '__main__':
    main()
