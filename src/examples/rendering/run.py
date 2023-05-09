import numpy as np
from pyrr import Quaternion

from src.core.defs import OUTPUT_DIR
from src.core.geo.transform import Transform
from src.core.rendering.data import Resolution
from src.examples.rendering.data import ShutterAnimationSettings, CameraPositioningMode, IntegralSettings, \
    FocusAnimationSettings
from src.examples.rendering.render import animate_shutter, project_shots, render_integral, animate_focus


def run() -> None:
    bambi_data_dir = r'D:\BambiData'
    frame_type = 't'
    data_set = r'\Hagenberg\KFV-hgb-Enew'  # NeRF Grid'

    gltf_file = rf'{bambi_data_dir}\DEM\Hagenberg\dem_mesh_r2.glb'
    shot_json_file = rf'{bambi_data_dir}\Processed{data_set}\Frames_{frame_type}\poses.json'
    mask_file = rf'{bambi_data_dir}\Processed{data_set}\Frames_{frame_type}\mask_{frame_type}.png'
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
    project_shots(gltf_file, shot_json_file, mask_file, settings)
    render_integral(gltf_file, shot_json_file, mask_file, settings)

    fps = 50
    duration = 2
    start_focus = 22
    end_focus = 2
    correction = Transform()
    correction.rotation = Quaternion.from_z_rotation(-np.deg2rad(1.0), dtype='f4')
    output_file = rf'{OUTPUT_DIR}focus_anim'
    settings = FocusAnimationSettings(
        start_focus=start_focus, end_focus=end_focus, frame_count=duration * fps, fps=fps,
        count=count, initial_skip=first, add_background=False, fovy=50.0, camera_dist=0.0, #-17.5,
        camera_position_mode=CameraPositioningMode.center_shot, move_camera_with_focus=True,
        resolution=Resolution(1024 * 2, 1024 * 2), aspect_ratio=1.0, orthogonal=False, ortho_size=(65, 65),
        delete_frames=False, first_frame_repetitions=fps, last_frame_repetitions=fps, output_file=output_file)
    animate_focus(gltf_file, shot_json_file, mask_file, settings)

    shots_grow_func = lambda x: int(np.ceil(np.exp(x * 0.2 - 0.8)))
    correction = Transform()
    correction.position.z = 22
    correction.rotation = Quaternion.from_z_rotation(-np.deg2rad(1.0), dtype='f4')
    output_file = rf'{OUTPUT_DIR}shutter_anim_{frame_type}_22'
    settings = ShutterAnimationSettings(
        shots_grow_func=shots_grow_func, reference_index=center - first, grow_symmetrical=True,
        count=count, initial_skip=first, add_background=False, fovy=50.0, camera_dist=-22.0,
        camera_position_mode=CameraPositioningMode.center_shot,
        resolution=Resolution(1024 * 2, 1024 * 2), aspect_ratio=1.0, orthogonal=False, ortho_size=(65, 65),
        correction=correction,
        delete_frames=False, output_file=output_file
    )
    animate_shutter(gltf_file, shot_json_file, mask_file, settings)

if __name__ == '__main__':
    run()