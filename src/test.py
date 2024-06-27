import json
import os
import random
import time
from collections import defaultdict
from typing import Final, cast

import cv2
import moderngl as mgl
import numpy as np
import pyrr
from pyrr import Matrix44, Vector3, Quaternion
from trimesh import Trimesh

from src.core.conv.coord_conversion import pixel_to_world_coord, world_to_pixel_coord
from src.core.geo.transform import Transform
from src.core.rendering.camera import Camera
from src.core.rendering.data import Resolution
from src.core.util import TimeTracker
from src.core.util.basic import get_center, nearest_int, make_quad
from src.core.util.cyclic_list import CyclicList
from src.core.util.defs import OUTPUT_DIR, INPUT_DIR, DEF_FRAG_SHADER_PATH, \
    DEF_VERT_SHADER_PATH, DEF_PASS_VERT_SHADER_PATH, DEF_PASS_FRAG_SHADER_PATH
from src.core.util.gltf import gltf_extract
from src.core.util.image import split_components
from src.core.util.moderngl import img_from_fbo
from src.core.util.pyrrs import rand_quaternion
from src.examples.rendering.data import IntegralSettings, CameraPositioningMode
from src.examples.rendering.render import render_integral, DoneCallback

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
        gltf_file = os.path.join(
            bambi_dev_dir,
            'Processed', 'BW', '2023_01_18_Ktn_Feldreh_Zollfelf', '1581F5FJB22A700A0DV7_M3TE', '010_Feldreh_Zoll',
            'Data', 'dem', 'dem_mesh_r2.glb'
        )
        camera_position = Vector3([450.9566076750634, 5.060010188259184, 495.5558088407927])  # [0, 0, 500]
        camera_eulers = [360.1, 0, 57.841499999999826]
        camera_rotation = Quaternion.from_eulers([np.deg2rad(e % 360.0) for e in camera_eulers])
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
        for _ in range(100):
            x = random.random() * image_res[0]
            if random.random() > 0.5:
                x = int(x)

            y = random.random() * image_res[1]
            if random.random() > 0.5:
                y = int(y)
            input_coord = x, y

            camera_rotation = rand_quaternion(min_x=-45, max_x=45, min_y=0.0, max_y=0.0, min_z=-45, max_z=45)
            camera.transform.rotation = camera_rotation

            world_coord = pixel_to_world_coord(input_coord[0], input_coord[1], image_res[0], image_res[1], tri_mesh, camera)
            pixel_coord = world_to_pixel_coord(world_coord, image_res[0], image_res[1], camera, ensure_int=False)

            error = ((input_coord[0] - pixel_coord[0]) ** 2 + (input_coord[1] - pixel_coord[1]) ** 2) ** 0.5

            # print(f'({input_coord[0]:8.3f}, {input_coord[1]:8.3f}) -> '
            #       f'({world_coord[0]:8.3f}, {world_coord[1]:8.3f}, {world_coord[2]:8.3f}) -> '
            #       f'({pixel_coord[0]:8.3f}, {pixel_coord[1]:8.3f})'
            #       f' | error: {error}')

            errors.append(error)

    avg = sum(errors) / len(errors)
    mxv = max(errors)
    mnv = min(errors)

    print('\n| metric |    avg    |    max    |    min    |\n|--------|-----------|-----------|-----------|')
    print(f'|  error | {avg:7.3e} | {mxv:7.3e} | {mnv:7.3e} |')


def test_render_labels() -> None:

    data_sets = [
        ('BW', '2023_01_18_Ktn_Feldreh_Zollfelf', '1581F5FJB22A700A0DV7_M3TE', '010_Feldreh_Zoll'),
        ('Spektakulair', '2023_10_05_NOe_Purkersdorf', '1581F5FJB22A700A0DW3_M3TE', '104_Schwarzwild'),
    ]  # ID 31

    # region paths

    bambi_dev_dir = os.getenv('BAMBI_DEV_DIR')
    data_dir = os.path.join(bambi_dev_dir, 'Processed', *data_sets[1])

    dem_file = os.path.join(data_dir, 'Data', 'dem', 'dem_mesh_r2.glb')
    labels_file = os.path.join(data_dir, 'labels.json')
    info_file = os.path.join(data_dir, 'info.json')
    frames_dir = os.path.join(data_dir, 'Frames_T')

    poses_file = os.path.join(frames_dir, 'matched_poses.json')
    mask_file = os.path.join(frames_dir, 'mask_T.png')

    output_file = os.path.join(OUTPUT_DIR, 'integral.png')
    labeled_output_file = os.path.join(OUTPUT_DIR, 'labeled_integral.png')
    render_camera_file = os.path.join(OUTPUT_DIR, 'render_camera.json')

    # endregion

    # region config

    input_resolution = Resolution(1024, 1024)
    render_resolution = Resolution(720 * 10, 720 * 10)
    first_frame_idx = 100  # 4838  # 31500
    last_frame_idx = 10900  # 5238  # 32700
    frame_count = last_frame_idx - first_frame_idx
    frame_range = range(first_frame_idx, last_frame_idx)

    render = False

    label_colors = CyclicList((  # BGR
        (102, 0, 255),  # '#ff0066',  #
        (255, 102, 0),  # '#0066ff',  #
        (0, 255, 102),  # '#66ff00',  #
        (255, 0, 102),  # '#6600ff',  #
        (255, 102, 0),  # '#00ff66',  #
        (0, 102, 255),  # '#ff6600',  #
    ))

    with open(info_file, 'r') as jf:
        data = json.load(jf)
    correction = data.get('correction', None)

    if correction is not None:
        translation = correction.get('translation', {'x': 0, 'y': 0, 'z': 0})
        cor_translation = Vector3([translation['x'], translation['y'], translation['z']], dtype='f4')

        rotation = correction.get('rotation', {'x': 0, 'y': 0, 'z': 0})
        cor_rotation_eulers = Vector3([rotation['x'], rotation['y'], rotation['z']], dtype='f4')
        correction = Transform(cor_translation, Quaternion.from_eulers(cor_rotation_eulers))
    else:
        cor_translation = Vector3()
        cor_rotation_eulers = Vector3()
        correction = Transform()

    # endregion

    # region render integral

    if render:
        settings = IntegralSettings(
            count=frame_count, initial_skip=first_frame_idx, add_background=True, camera_dist=10.0,
            camera_position_mode=CameraPositioningMode.shot_centered,  fovy=50.0, aspect_ratio=1.0, orthogonal=True,
            ortho_size=(256, 256), correction=correction, resolution=render_resolution,
            show_integral=False, output_file=output_file
        )

        render_camera = render_integral(dem_file, poses_file, mask_file, settings)

        render_camera_dict = render_camera.to_dict()
        with open(render_camera_file, 'w+') as jf:
            json.dump(render_camera_dict, jf)

    # endregion

    # region draw labels

    with open(render_camera_file, 'r') as jf:
        render_camera_dict = json.load(jf)
    render_camera = Camera.from_dict(render_camera_dict)
    if render_camera is None:
        raise Exception('A' * 5000)

    label_done = DoneCallback('      ')
    print('Start label projection process')
    print('  Projecting pixels')

    mesh, _ = gltf_extract(dem_file)
    tri_mesh = Trimesh(vertices=mesh.vertices, faces=mesh.indices)

    render = cv2.imread(output_file)
    frame_data = defaultdict(lambda: {
        "label_coords": [],
        "camera": None
    })

    with open(labels_file, 'r') as jf:
        label_data = json.load(jf)

    label_state_data = defaultdict(lambda: [])
    for i, label in enumerate(label_data):
        label_state_data[i].extend(label["states"])

    for label in label_state_data:
        for state in label_state_data[label]:
            label_coords = state["pxlCoordinates"][:4]
            color_idx = label
            frame_data[state["frameIdx"]]["label_coords"].append((color_idx, label_coords))

    with open(poses_file, 'r') as jf:
        poses_data = json.load(jf)
    image_data = poses_data['images']

    for frame_idx in frame_data:
        cur_frame_data = image_data[frame_idx]
        fovy = cur_frame_data['fovy'][0]

        position = Vector3(cur_frame_data['location'])
        rotation_eulers = Vector3([np.deg2rad(val % 360.0) for val in cur_frame_data['rotation']]) + cor_rotation_eulers

        position += cor_translation
        rotation = Quaternion.from_eulers(rotation_eulers)

        camera = Camera(fovy=fovy, aspect_ratio=1.0, position=position, rotation=rotation)

        frame_data[frame_idx]["camera"] = camera

    for frame, data in frame_data.items():
        camera = data['camera']

        for color_idx, poly_coords in data['label_coords']:
            render_pixels = []
            for pixel in poly_coords:
                x = pixel['x']
                y = pixel['y']
                w_pos = pixel_to_world_coord(x, y, input_resolution.width, input_resolution.height, tri_mesh, camera)
                np_pos = world_to_pixel_coord(w_pos, render_resolution.width, render_resolution.height, render_camera)
                render_pixels.append(np_pos)
            poly_lines = [np.array(render_pixels).reshape((-1, 1, 2))]
            cv2.polylines(render, poly_lines, True, label_colors[color_idx], thickness=1)
    label_done()

    print('  Saving labeled integral')
    cv2.imwrite(labeled_output_file, render)
    label_done()
    label_done.total(msg='All done', indent=False)

    # endregion


def main() -> None:
    print(f'running {__file__}')
    test_render_labels()


if __name__ == '__main__':
    main()

# further tests with flights showing boars
