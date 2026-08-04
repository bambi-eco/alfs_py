from alfspy.core.util.loggings import apply_default_config

apply_default_config()

import os
from typing import Final

import numpy as np
from pyrr import Quaternion, Vector3

from alfspy.core.geo.transform import Transform
from alfspy.core.rendering.data import Resolution
from alfspy.core.util.defs import MODULE_DIR
from alfspy.render import CameraPositioningMode, IntegralSettings, render_integral

OUTPUT_DIR: Final[str] = os.path.join(MODULE_DIR, '..', '..', 'output')
INPUT_DIR: Final[str] = os.path.join(MODULE_DIR, '..', '..', 'input')
RESOURCES_DIR: Final[str] = os.path.join(MODULE_DIR, '..', '..', 'resources')

def run() -> None:
    bambi_dev_dir = os.environ.get("BAMBI_DEV_DIR", None)
    if bambi_dev_dir is None:
        raise EnvironmentError('BAMBI_DEV_DIR environment variable not set')

    data_set = os.path.join('BW', '2023_01_18_Ktn_Feldreh_Zollfelf', '1581F5FJB22A700A0DV7_M3TE', '010_Feldreh_Zoll')
    # data_set = os.path.join('Spektakulair', '2023_10_05_NOe_Purkersdorf', '1581F5FJB22A700A0DW3_M3TE', '104_Schwarzwild')
    gltf_file = os.path.join(bambi_dev_dir, 'Processed', data_set, 'Data', 'dem', 'dem_mesh_r2.glb')

    # for frame_type in ['w', 't']:
    for frame_type in ('t',):
        shot_json_file = os.path.join(bambi_dev_dir, 'Processed', data_set, f'Frames_{frame_type}', 'matched_poses.json')
        mask_file = os.path.join(bambi_dev_dir, 'Processed', data_set, f'Frames_{frame_type}', f'mask_{frame_type}.png')
        count = 20
        if frame_type == 't':
            center = 22999
        elif frame_type == 'w':
            center = 5820
        else:
            center = count // 2
        first = center - count // 2

        for i in range(1):
            translation = Vector3([0.0, 0.0, 2.0 * (i / 3)], dtype='f4')
            rotation = Quaternion.from_z_rotation(np.deg2rad(2.0), dtype='f4')
            correction = Transform(position=translation, rotation=rotation)
            output_file = os.path.join(OUTPUT_DIR, f'integral_{i}.png')
            settings = IntegralSettings(count=count, initial_skip=first, camera_dist=10.0, add_background=False,
                                        camera_position_mode=CameraPositioningMode.CenterShot,
                                        correction=correction, resolution=Resolution(1024 * 2, 1024 * 2),
                                        fovy=50.0, aspect_ratio=1.0, ortho_size=(65, 65),
                                        orthogonal=False, show_integral=False, output_file=output_file)
            # project_shots(gltf_file, shot_json_file, mask_file, settings)
            render_integral(gltf_file, shot_json_file, mask_file, settings)

        # fps = 50
        # duration = 2
        # start_focus = 22
        # end_focus = 2
        # correction = Transform()
        # correction.rotation = Quaternion.from_z_rotation(np.deg2rad(1.0), dtype='f4')
        # output_file = rf'{OUTPUT_DIR}focus_anim_{frame_type}'
        # settings = FocusAnimationSettings(
        #     start_focus=start_focus, end_focus=end_focus, frame_count=duration * fps, fps=fps,
        #     count=count, initial_skip=first, add_background=False, fovy=50.0, camera_dist=0.0, #-17.5,
        #     camera_position_mode=CameraPositioningMode.center_shot, move_camera_with_focus=True,
        #     resolution=Resolution(1024 * 2, 1024 * 2), aspect_ratio=1.0, orthogonal=False, ortho_size=(65, 65),
        #     correction=correction,
        #     delete_frames=False, frame_dir=os.path.join(OUTPUT_DIR, 'focus_animation', frame_type),
        #     first_frame_repetitions=fps, last_frame_repetitions=fps, output_file=output_file)
        # animate_focus(gltf_file, shot_json_file, mask_file, settings)

        # fps = 30
        # shots_grow_func = lambda x: int(np.ceil(np.exp(x * 0.2 - 0.8)))
        # correction = Transform()
        # z_correction = 2
        # correction.position.z = z_correction
        # correction.rotation = Quaternion.from_z_rotation(np.deg2rad(1.0), dtype='f4')
        # output_file = rf'{OUTPUT_DIR}shutter_anim_{frame_type}'
        # settings = ShutterAnimationSettings(
        #     shots_grow_func=shots_grow_func, reference_index=center - first, grow_symmetrical=True, fps=fps,
        #     count=count, initial_skip=first, add_background=False, fovy=50.0, camera_dist=-22.0,
        #     camera_position_mode=CameraPositioningMode.center_shot,
        #     resolution=Resolution(1024 * 2, 1024 * 2), aspect_ratio=1.0, orthogonal=False, ortho_size=(65, 65),
        #     correction=correction,
        #     delete_frames=False, frame_dir=os.path.join(OUTPUT_DIR, 'focus_animation', f's_{frame_type}_{z_correction}'),
        #     output_file=output_file
        # )
        # animate_shutter(gltf_file, shot_json_file, mask_file, settings)


if __name__ == '__main__':
    run()
