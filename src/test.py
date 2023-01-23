from typing import Final

import cv2
import moderngl as mgl
import numpy as np
from pyrr import Matrix44, Vector3

from src.core.camera import Camera
from src.core.defs import OUTPUT_DIR, INPUT_DIR, COL_VERT_SHADER_PATH, COL_FRAG_SHADER_PATH, \
    TEX_VERT_SHADER_PATH, TEX_FRAG_SHADER_PATH, DEF_FRAG_SHADER_PATH, \
    DEF_VERT_SHADER_PATH, DEF_PASS_VERT_SHADER_PATH, DEF_PASS_FRAG_SHADER_PATH
from src.core.geo.frustum import Frustum
from src.core.iters import file_name_gen
from src.core.renderer import Renderer, ProjectMode
from src.core.shot import CtxShot
from src.core.geo.transform import Transform
from src.core.utils import img_from_fbo, gltf_extract, crop_to_content, split_components, \
    get_center, int_up, make_quad

_OUTPUT_RESOLUTION: Final[tuple[int, int]] = (1024 * 2, 1024 * 2)
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
    vao = ctx.vertex_array(prog, [(vbo, '3f4 2f4', 'v_in_v3_pos', 'v_in_v2_uv')], index_buffer=ibo, index_element_size=4)
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


def test_projection(count: int = 1):
    ctx = mgl.create_context(standalone=True)
    ctx.enable(mgl.DEPTH_TEST)

    gltf_file = f'{INPUT_DIR}mesh.glb'
    mesh_data, texture_data = gltf_extract(gltf_file)

    center, aabb = get_center(mesh_data.vertices)
    center.z = 1
    ortho_size = int_up(aabb.width), int_up(aabb.height)

    camera = Camera(orthogonal=True, orthogonal_size=ortho_size, position=center)
    renderer = Renderer(_OUTPUT_RESOLUTION, ctx, camera, mesh_data, texture_data)

    json_file = f'{INPUT_DIR}data\\poses.json'
    shots = CtxShot.from_json(json_file, ctx, count=count)

    file_name_iter = file_name_gen('.png', f'{OUTPUT_DIR}proj')
    results = renderer.project_shots(shots, ProjectMode.SHOT_VIEW, save=False, save_name_iter=file_name_iter)

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

def test_frustum() -> None:
    transform = Transform()
    frustum = Frustum(70.0, 16.0/9.0, near=0.001, far=10, transform=transform)

    corners = frustum.corners
    for i, corner in enumerate(corners):
        print(f'Corner {chr(65+i)} = ({corner.x:.9f}, {corner.y:.9f}, {corner.z:.9f}) | Length {corner.length}')

def test_projection2(count: int = 1):
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
        print(f'Corner {chr(65+i)} = ({corner.x:.9f}, {corner.y:.9f}, {corner.z:.9f})')

    #camera.transform.position.z = 600

    print(camera.transform.position)
    camera.fit_to_points(aabb.corners, 0.0)
    print(camera.transform.position)
    renderer = Renderer(_OUTPUT_RESOLUTION, ctx, camera, mesh_data, texture_data)

    json_file = f'{INPUT_DIR}data\\poses.json'
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
    # test_projection(5)

    test_projection2()


if __name__ == '__main__':
    main()
