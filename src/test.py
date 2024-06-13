import os
import random
from typing import Final, cast

import cv2
import moderngl as mgl
import numpy as np
from pyrr import Matrix44, Vector3, Quaternion
from trimesh import Trimesh

from src.core.conv.coord_conversion import pixel_to_world_coord, world_to_pixel_coord
from src.core.rendering.camera import Camera
from src.core.rendering.data import Resolution
from src.core.util import TimeTracker
from src.core.util.basic import get_center, nearest_int, make_quad
from src.core.util.defs import OUTPUT_DIR, INPUT_DIR, DEF_FRAG_SHADER_PATH, \
    DEF_VERT_SHADER_PATH, DEF_PASS_VERT_SHADER_PATH, DEF_PASS_FRAG_SHADER_PATH
from src.core.util.gltf import gltf_extract
from src.core.util.image import split_components
from src.core.util.moderngl import img_from_fbo

_OUTPUT_RESOLUTION: Final[tuple[int, int]] = Resolution(1024 * 2, 1024 * 2).as_tuple()
_CLEAR_COLOR: Final[tuple[float, ...]] = (1.0, 0.0, 1.0, 0.1)


def test_deferred_shading() -> None:
    file = f'{INPUT_DIR}mesh.glb'

    mesh_data, tex_data = gltf_extract(file)

    vertices = mesh_data.vertices
    indices = mesh_data.indices

    center, aabb = get_center(mesh_data.vertices)
    center.z = 750
    ortho_size = nearest_int(aabb.width), nearest_int(aabb.height)

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


def test_coords_conv() -> None:
    with TimeTracker("Init Data", False):
        bambi_dev_dir = os.getenv('BAMBI_DEV_DIR')
        bambi_data_dir = os.getenv('BAMBI_DATA_DIR')
        gltf_file = os.path.join(
            bambi_dev_dir,
            'Processed', 'BW', '2023_01_18_Ktn_Feldreh_Zollfelf', '1581F5FJB22A700A0DV7_M3TE', '010_Feldreh_Zoll',
            'Data', 'dem', 'dem_mesh_r2.glb'
        )
        camera_position = Vector3([450.9566076750634, 5.060010188259184, 495.5558088407927])  # [0, 0, 500]
        camera_rotation = Quaternion.from_eulers([0.0, 0.0, 57.841499999999826])  # [360.1, 0, 57.841499999999826]
        camera = Camera(
            position=camera_position, rotation=camera_rotation,
            fovy=48.887902511473634, aspect_ratio=1.0,
            orthogonal=False)
        image_res = (1000, 1000)

    with TimeTracker("Read Gltf", False):
        mesh, _ = gltf_extract(gltf_file)

    with TimeTracker("Proc mesh", False):
        tri_mesh = Trimesh(vertices=mesh.vertices, faces=mesh.indices)

    with TimeTracker("Cast rays", False):
        errors = []
        for _ in range(1):
            x = random.random() * image_res[0]
            if random.random() > 0.5:
                x = int(x)

            y = random.random() * image_res[1]
            if random.random() > 0.5:
                y = int(y)
            input_coord = x, y

            world_coord = pixel_to_world_coord(input_coord[0], input_coord[1], image_res[0], image_res[1], tri_mesh, camera)
            pixel_coord = world_to_pixel_coord(world_coord, image_res[0], image_res[1], camera, ensure_int=False)

            error = ((input_coord[0] - pixel_coord[0]) ** 2 + (input_coord[1] - pixel_coord[1]) ** 2) ** 0.5

            print(f'({input_coord[0]:8.3f}, {input_coord[1]:8.3f}) -> '
                  f'({world_coord[0]:8.3f}, {world_coord[1]:8.3f}, {world_coord[2]:8.3f}) -> '
                  f'({pixel_coord[0]:8.3f}, {pixel_coord[1]:8.3f})'
                  f' | error: {error}')

            errors.append(error)

    avg = sum(errors) / len(errors)
    mxv = max(errors)
    mnv = min(errors)

    print('\n| metric |    avg    |    max    |    min    |\n|--------|-----------|-----------|-----------|')
    print(f'|  error | {avg:7.3e} | {mxv:7.3e} | {mnv:7.3e} |')


def main() -> None:
    print(f'running {__file__}')
    test_coords_conv()


if __name__ == '__main__':
    main()
