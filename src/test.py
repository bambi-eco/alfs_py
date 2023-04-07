import glob
import time
from typing import Final, Optional

import cv2
import moderngl as mgl
import numpy as np
from PIL import Image
from pyrr import Matrix44, Vector3, Quaternion

from src.core.camera import Camera
from src.core.data import TextureData
from src.core.defs import OUTPUT_DIR, INPUT_DIR, COL_VERT_SHADER_PATH, COL_FRAG_SHADER_PATH, \
    TEX_VERT_SHADER_PATH, TEX_FRAG_SHADER_PATH, DEF_FRAG_SHADER_PATH, \
    DEF_VERT_SHADER_PATH, DEF_PASS_VERT_SHADER_PATH, DEF_PASS_FRAG_SHADER_PATH, CPP_INT_MAX, MAGENTA, BLACK, MAX_TEX_DIM
from src.core.geo.transform import Transform
from src.core.iters import file_name_gen
from src.core.renderer import Renderer, ProjectMode
from src.core.shot import CtxShot
from src.core.util.basic import get_center, int_up, make_quad, gen_checkerboard_tex
from src.core.util.gltf import gltf_extract
from src.core.util.image import crop_to_content, split_components, integral, overlay, replace_color, laplacian_variance
from src.core.util.moderngl import img_from_fbo

_OUTPUT_RESOLUTION: Final[tuple[int, int]] = (1024 * 4, 1024 * 4)
_CLEAR_COLOR: Final[tuple[float, ...]] = (1.0, 0.0, 1.0, 0.1)

_FOV: Final[float] = 45.0
_NEAR_CLIP: Final[float] = 0.1
_FAR_CLIP: Final[float] = 10000.0


def draw_rand_tris(n: int) -> None:
    ctx = mgl.create_context(standalone=True)

    with open(COL_VERT_SHADER_PATH) as file:
        vert_shader = file.read()

    with open(COL_FRAG_SHADER_PATH) as file:
        frag_shader = file.read()

    prog = ctx.program(vertex_shader=vert_shader, fragment_shader=frag_shader)
    fbo = ctx.simple_framebuffer(_OUTPUT_RESOLUTION, components=4)
    fbo.use()

    tri_count = 10
    vert_count = tri_count * 3

    vbo = ctx.buffer(reserve=7 * 4 * vert_count)
    vao = ctx.vertex_array(prog, [(vbo, '3f4 4f4', 'in_vert', 'in_color')])

    for i in range(n):
        x = np.random.rand(vert_count) * 2 - 1
        y = np.random.rand(vert_count) * 2 - 1
        z = np.zeros(vert_count)
        r = np.random.rand(vert_count)
        g = np.random.rand(vert_count)
        b = np.random.rand(vert_count)
        a = np.ones(vert_count)

        vertices = np.dstack([x, y, z, r, g, b, a])
        vbo.write(vertices.astype('f4').tobytes())

        fbo.clear(*_CLEAR_COLOR)
        vao.render(mgl.TRIANGLES)

        render_result = img_from_fbo(fbo)
        file_path = f'{OUTPUT_DIR}render_{i}.png'
        cv2.imwrite(file_path, render_result)

    vao.release()
    vbo.release()
    fbo.release()
    prog.release()
    ctx.release()


def gltf_lib_test() -> None:
    file = f'{INPUT_DIR}mesh.glb'

    mesh_data, tex_data = gltf_extract(file)

    vertices = mesh_data.vertices
    indices = mesh_data.indices
    uvs = mesh_data.uvs

    center, _ = get_center(vertices)
    x_tsl, y_tsl, _ = center

    camera = Camera(orthogonal=False, orthogonal_size=(1024, 1024), position=Vector3([0, 0, 750]))
    camera.look_at(Vector3([0, 0, 0]))

    projection = camera.get_proj()
    view = camera.get_view()

    model = Matrix44.from_translation([x_tsl, y_tsl, -5], dtype='f4')

    ctx = mgl.create_context(standalone=True)
    ctx.enable(mgl.DEPTH_TEST)

    with open(TEX_VERT_SHADER_PATH) as file:
        vert_shader = file.read()
    with open(TEX_FRAG_SHADER_PATH) as file:
        frag_shader = file.read()
    prog = ctx.program(vertex_shader=vert_shader, fragment_shader=frag_shader)

    prog['projection'].write(projection)
    prog['view'].write(view)
    prog['model'].write(model)

    fbo = ctx.simple_framebuffer(_OUTPUT_RESOLUTION, components=4)
    fbo.use()

    tex_input = tex_data.tex_gen_input()
    tex = ctx.texture(*tex_input, dtype='f4')
    tex.use()

    vbo = ctx.buffer(reserve=5 * 4 * vertices.shape[0])
    ibo = ctx.buffer(indices.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, '3f4 2f4', 'v_in_v3_pos', 'v_in_v2_uv')], index_buffer=ibo,
                           index_element_size=4)
    # vao = ctx.vertex_array(prog, [(vetices, '3f4 2f4', 'pos_in', 'uv_cord_in')])

    x, y, z = split_components(vertices)
    u, v = split_components(uvs)

    shader_data = np.dstack([x, y, z, u, v])
    vbo.write(shader_data.astype('f4').tobytes())

    fbo.clear(*_CLEAR_COLOR)
    vao.render(mgl.TRIANGLES)

    render_result = img_from_fbo(fbo)
    file_path = f'{OUTPUT_DIR}render_mesh.png'
    cv2.imwrite(file_path, render_result)

    print()

    for releasable in (tex, vao, vbo, fbo, prog, ctx):
        releasable.release()


def test_lines():
    vertices = np.array([
        [0.0, 0.5, -1.0],
        [0.5, -0.5, -1.0],
        [-0.5, -0.5, -1.0],
    ], dtype='f4')

    col = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ], dtype='f4')

    ctx = mgl.create_context(standalone=True)

    with open(COL_VERT_SHADER_PATH) as file:
        vert_shader = file.read()

    with open(COL_FRAG_SHADER_PATH) as file:
        frag_shader = file.read()

    prog = ctx.program(vertex_shader=vert_shader, fragment_shader=frag_shader)
    fbo = ctx.simple_framebuffer(_OUTPUT_RESOLUTION, components=4)
    fbo.use()

    x, y, z = split_components(vertices)
    r, g, b = split_components(col)
    a = np.ones(x.shape)
    shader_data = np.dstack([x, y, z, r, g, b, a])
    vbo = ctx.buffer(shader_data.astype('f4').tobytes())
    vao = ctx.vertex_array(prog, [(vbo, '3f4 4f4', 'in_vert', 'in_color')])

    fbo.clear(*_CLEAR_COLOR)
    vao.render(mgl.LINE_LOOP)

    render_result = img_from_fbo(fbo)
    file_path = f'{OUTPUT_DIR}render_line.png'
    cv2.imwrite(file_path, render_result)

    vao.release()
    vbo.release()
    fbo.release()
    prog.release()
    ctx.release()


def test_crop_to_content():
    img = cv2.imread(f'{INPUT_DIR}asym_crate.png')
    img, delta = crop_to_content(img, True)
    height, width = img.shape[0:2]
    old_center = (int(width / 2 + delta.x), int(height / 2 + delta.y))
    new_center = (width // 2, height // 2)
    cv2.circle(img, new_center, 7, (255, 255, 0), -1)
    cv2.arrowedLine(img, old_center, new_center, (255, 0, 255), 2)
    cv2.circle(img, old_center, 7, (0, 255, 255), -1)
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite(f'{OUTPUT_DIR}crop.png', img)


class DoneCallback:

    def __init__(self):
        self.start = time.time()
        self.last = self.start

    def __call__(self, print_msg: bool = True):
        current = time.time()
        if print_msg:
            print(f'    Done [{(current - self.last) * 1000:.3f} ms]')
        self.last = current


def test_projection(count: int = 1, show_count: int = -1, projection_mode: ProjectMode = ProjectMode.SHOT_VIEW_RELATIVE,
                    lazy: bool = True, render_integral: bool = True, release_shots: bool = True,
                    correction: Optional[Transform] = None, suffix: str = ''):

    done = DoneCallback()
    print('Start projection process')

    ctx = mgl.create_context(standalone=True)
    ctx.enable(mgl.DEPTH_TEST)
    ctx.enable(mgl.CULL_FACE)
    ctx.cull_face = 'back'

    data_dir = f'{INPUT_DIR}data\\haag\\'
    gltf_file = f'{data_dir}dem_mesh_r2.glb'
    json_file = f'{data_dir}poses.json'

    print(f'  Reading GLTF file from "[{gltf_file}]')
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
    center.z = 1
    ortho_size = int_up(aabb.width), int_up(aabb.height)
    done()

    print(f'  Creating context and renderer')
    camera = Camera(orthogonal=True, orthogonal_size=ortho_size, position=center)
    renderer = Renderer(_OUTPUT_RESOLUTION, ctx, camera, mesh_data, texture_data)
    done()

    print(f'  Extracting shots from JSON (creating lazy shots: {lazy})')
    shots = CtxShot.from_json(json_file, ctx, count=count, correction=correction, lazy=lazy)
    done()

    print('  Rendering background')
    background = cv2.cvtColor(renderer.render_ground(), cv2.COLOR_BGRA2RGBA)
    cv2.imwrite(f'{OUTPUT_DIR}back.png', cv2.cvtColor(background, cv2.COLOR_BGRA2RGBA))
    done()

    print(f'  Projecting shots (Mode: {projection_mode}; Render integral: {render_integral}; Release single shots after projection: {release_shots})')
    file_name_iter = file_name_gen('.png', f'{OUTPUT_DIR}proj')
    results = renderer.project_shots(shots, projection_mode, integral=render_integral, save=False, save_name_iter=file_name_iter, release_shots=True)
    done()

    if render_integral:
        print('  Showing integral')
        result = results
        img = cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA)
        img = overlay(background, img)
        im_pil = Image.fromarray(img)
        im_pil.show('Integral')
        done()
    else:
        if show_count > 0:
            print(f'  Showing {show_count} individual results')
            for i, result in enumerate(results[:show_count]):
                # img = cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA)
                # im_pil = Image.fromarray(img)
                # im_pil.show(f'Shot {i}')
                cv2.imwrite(f'{OUTPUT_DIR}proj/{i}.png', result)
            done()

        print('  Computing integral')
        result = integral(results)
        done()

        print('  Processing and showing integral')
        img = cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA)
        img = replace_color(img, (0, 0, 0, 255), (0, 0, 0, 0), True)
        cv2.imwrite(f'{OUTPUT_DIR}integral{suffix}.png', cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
        img = overlay(background, img)
        im_pil = Image.fromarray(img)
        im_pil.show('Integral')
        done()

    print('  Saving integral image')
    im_pil.save(f'{OUTPUT_DIR}integral.png')
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


def test_load_all_images(img_dir: str) -> None:
    files = glob.glob(f'{img_dir}*.png')

    file_arr = []
    for file in files:
        img = cv2.imread(file)
        file_arr.append(img)


def test_image_metrics() -> None:
    prefix = f'{OUTPUT_DIR}integral'
    suffix = '.png'
    files = glob.glob(f'{prefix}*{suffix}')
    files = [file.split(prefix)[1].split(suffix)[0] for file in files]

    files = sorted(files, key=lambda x: int(x))
    for file in files:
        full_file = prefix + file + suffix
        img = cv2.imread(full_file)
        img = crop_to_content(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        # laplacian = cv2.Laplacian(img, cv2.CV_64F)
        variance = laplacian_variance(img)
        print(f'{file}: {variance:.3f}')


def main() -> None:

    test_projection(1000)

    # correction = Transform()
    # vals = np.arange(-1, 1.05, 0.1) * 0.08726646
    # for val in vals:
    # correction_rot = Quaternion.from_z_rotation(0.08726646, dtype='f4')
    # correction.rotation = correction_rot
    # test_projection(100, 0, ProjectMode.SHOT_VIEW_RELATIVE, correction, f'{0.08726646*1000:.0f}')
    # test_load_all_images(f'{INPUT_DIR}data\\haag\\')

    # test_image_metrics()


if __name__ == '__main__':
    main()
