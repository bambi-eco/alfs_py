from typing import Final

import cv2
import moderngl as mgl
import numpy as np
from numpy import pi
from pyrr import Matrix44, Quaternion, Vector3

from src.core.camera import Camera
from src.core.data import TextureData
from src.core.defs import OUTPUT_DIR, INPUT_DIR, COL_VERT_SHADER_PATH, COL_FRAG_SHADER_PATH, \
    TEX_VERT_SHADER_PATH, TEX_FRAG_SHADER_PATH
from src.core.renderer import Renderer, ProjectMode
from src.core.shot import CtxShot
from src.core.utils import img_from_fbo, gltf_extract, get_vert_center_translation, crop_to_content, split_components

_OUTPUT_RESOLUTION: Final[tuple[int, int]] = (512, 512)
_CLEAR_COLOR: Final[tuple[float, ...]] = (1.0, 0.0, 1.0, 0.1)  # TRANSPARENT

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

    x_tsl, y_tsl, _ = get_vert_center_translation(vertices)

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
    vao = ctx.vertex_array(prog, [(vbo, '3f4 2f4', 'pos_in', 'uv_cord_in')], index_buffer=ibo, index_element_size=4)
    # vao = ctx.vertex_array(prog, [(vbo, '3f4 2f4', 'pos_in', 'uv_cord_in')])

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

    tex.release()
    vao.release()
    vbo.release()
    fbo.release()
    prog.release()
    ctx.release()


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

    x = vertices[..., 0]
    y = vertices[..., 1]
    z = vertices[..., 2]
    r = col[..., 0]
    g = col[..., 1]
    b = col[..., 2]
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


def test_projection():
    ctx = mgl.create_context(standalone=True)
    ctx.enable(mgl.DEPTH_TEST)

    gltf_file = f'{INPUT_DIR}mesh.glb'
    mesh_data, texture_data = gltf_extract(gltf_file)

    json_file = 'C:\\Users\\Cleo\\Documents\\Git\\alfs-web\\data\\BAMBI_202208240731_008_Tierpark-Haag-deer1\\poses.json'
    shots = CtxShot.from_json(json_file, ctx)

    camera = Camera(position=Vector3([0, 1, 0]), forward=Vector3([0, 0, -1]), up=Vector3([0, 1, 0]))

    renderer = Renderer((512, 512), ctx, camera, mesh_data, texture_data)

    projections = renderer.project_shots(shots, ProjectMode.COMPLETE_VIEW)

    for projection in projections:
        cv2.imshow('proj', projection)
        cv2.waitKey()
        cv2.destroyAllWindows()


def main() -> None:
    test_projection()


if __name__ == '__main__':
    main()
