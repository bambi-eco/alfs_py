from typing import Final, cast

import cv2
import moderngl as mgl
import numpy as np
from pyrr import Matrix44, Quaternion

from src.core.defs import OUTPUT_DIR, INPUT_DIR, DEF_FRAG_SHADER_PATH, \
    DEF_VERT_SHADER_PATH, DEF_PASS_VERT_SHADER_PATH, DEF_PASS_FRAG_SHADER_PATH
from src.core.geo.transform import Transform
from src.core.rendering.camera import Camera
from src.core.rendering.data import Resolution
from src.core.util.basic import get_center, int_up, make_quad
from src.core.util.gltf import gltf_extract
from src.core.util.image import split_components
from src.core.util.moderngl import img_from_fbo
from src.examples.rendering.data import IntegralSettings, CameraPositioningMode, FocusAnimationSettings, \
    ShutterAnimationSettings
from src.examples.rendering.render import render_integral, animate_focus, animate_shutter, project_shots

_OUTPUT_RESOLUTION: Final[tuple[int, int]] = Resolution(1024 * 2, 1024 * 2).as_tuple()
_CLEAR_COLOR: Final[tuple[float, ...]] = (1.0, 0.0, 1.0, 0.1)

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

def main() -> None:
    # bambi_data_dir = 'D:\\BambiData\\'
    frame_type = 't'
    data_set = r'\Hagenberg\KFV-hgb-Enew'  # NeRF Grid'

    gltf_file = rf'D:\BambiData\DEM\Hagenberg\dem_mesh_r2.glb'
    shot_json_file = rf'D:\BambiData\Processed{data_set}\Frames_{frame_type}\poses.json'
    mask_file = rf'D:\BambiData\Processed{data_set}\Frames_{frame_type}\mask_{frame_type}.png'
    count = 10
    if frame_type == 't':
        center = 35380
    elif frame_type == 'w':
        center = 35820
    else:
        center = count // 2
    first = center - count // 2

    correction = Transform()
    correction.position.z = 2.0
    correction.rotation = Quaternion.from_z_rotation(-np.deg2rad(1.0), dtype='f4')
    output_file = rf'{OUTPUT_DIR}integral'
    settings = IntegralSettings(count=count, initial_skip=first, camera_dist=10.0, add_background=True,
                                camera_position_mode=CameraPositioningMode.background_centered,
                                correction=correction, resolution=Resolution(2109, 4096), fovy=50.0, aspect_ratio=1.0,
                                orthogonal=True, show_integral=True, output_file=output_file)
    # project_shots(gltf_file, shot_json_file, mask_file, settings)
    render_integral(gltf_file, shot_json_file, mask_file, settings)

    # fps = 50
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
    #     resolution=Resolution(1024 * 2, 1024 * 2), aspect_ratio=1.0, orthogonal=False, ortho_size=(65, 65),
    #     delete_frames=False, first_frame_repetitions=fps, last_frame_repetitions=fps, output_file=output_file)
    # animate_focus(gltf_file, shot_json_file, mask_file, settings)

    # shots_grow_func = lambda x: int(np.ceil(np.exp(x * 0.2 - 0.8)))
    # correction = Transform()
    # correction.position.z = 22
    # correction.rotation = Quaternion.from_z_rotation(-np.deg2rad(1.0), dtype='f4')
    # output_file = rf'{OUTPUT_DIR}shutter_anim_{frame_type}_22'
    # settings = ShutterAnimationSettings(
    #     shots_grow_func=shots_grow_func, reference_index=center - first, grow_symmetrical=True,
    #     count=count, initial_skip=first, add_background=False, fovy=50.0, camera_dist=-22.0,
    #     camera_position_mode=CameraPositioningMode.center_shot,
    #     frame_dir='./.frames/22',
    #     resolution=Resolution(1024 * 2, 1024 * 2), aspect_ratio=1.0, orthogonal=False, ortho_size=(65, 65),
    #     correction=correction,
    #     delete_frames=False, output_file=output_file
    # )
    # animate_shutter(gltf_file, shot_json_file, mask_file, settings)

    # settings.correction.position.z = 2.0
    # settings.camera_dist = -2.0
    # settings.output_file = rf'{OUTPUT_DIR}shutter_anim_{frame_type}_2'
    # settings.frame_dir = './.frames/2'

    # test_shutter_animation(gltf_file, shot_json_file, mask_file, settings)

if __name__ == '__main__':
    main()

# TODO: Animation of increasing shot count with amount of used shots growing exponentially ( ceil(exp(x * 0.2 - 0.8)) )
# TODO: Test direct sharepoint loading
# TODO: Maybe try shifting integral calculations onto GPU using additional shader
# TODO: Fix shot reloading for future dgx use