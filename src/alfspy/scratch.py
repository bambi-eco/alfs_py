"""Developer scratch scripts carried over from the ModernGL version.

**This module is not part of the library API and is not covered by the test suite.**

It was ``src/alfspy/test.py`` in ``alfs_py``. Three things changed during the port:

1. **Renamed** to ``scratch.py``. The old name collided with pytest's default collection
   rules, which matters now that the project actually has a test suite under ``test/``.
2. **``test_deferred_shading`` was removed.** It was a raw ModernGL deferred-shading demo,
   unrelated to ALFS, loading shader files (``DEF_VERT_SHADER_PATH`` and friends) that do
   not exist in ``core.util.defs``, and it carried a standing
   ``# TODO: fix second pass not rendering anything``. There was nothing working to port.
3. The remaining functions still reference names that do not resolve in ``alfs_py`` either
   (``CyclicList`` from a non-existent ``core.util.cyclic_list``, ``LoggerCallback`` from
   ``render.render``, module-level ``INPUT_DIR`` / ``OUTPUT_DIR``). They are kept verbatim
   for reference; the imports are deferred into the functions so that importing this module
   does not fail.

Both remaining functions need real BAMBI flight data via the ``BAMBI_DEV_DIR`` environment
variable. For runnable examples that need no data, see ``test/integration/``.
"""

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Final, Optional

import cv2
import numpy as np
from pyrr import Vector3, Quaternion
from trimesh import Trimesh

from alfspy.core.convert import pixel_to_world_coord, world_to_pixel_coord
from alfspy.core.geo.transform import Transform
from alfspy.core.rendering.camera import Camera
from alfspy.core.rendering.data import Resolution
from alfspy.core.util import TimeTracker
from alfspy.core.util.geo import get_center
from alfspy.core.util.gltf import gltf_extract
from alfspy.core.util.pyrrs import rand_quaternion
from alfspy.render.data import IntegralSettings, CameraPositioningMode
from alfspy.render.render import render_integral

_OUTPUT_RESOLUTION: Final[tuple[int, int]] = Resolution(1024 * 2, 1024 * 2).as_tuple()

OUTPUT_DIR: str = os.getenv('ALFS_OUTPUT_DIR', '.')


def test_coords_conv() -> None:
    """
    Round-trips random pixel coordinates through pixel -> world -> pixel and reports the error.

    Needs ``BAMBI_DEV_DIR`` to point at a processed flight archive.
    """
    with TimeTracker("Init Data", False):
        bambi_dev_dir = os.getenv('BAMBI_DEV_DIR')
        gltf_file = os.path.join(
            bambi_dev_dir,
            'Processed', 'BW', '2023_01_18_Ktn_Feldreh_Zollfelf', '1581F5FJB22A700A0DV7_M3TE', '010_Feldreh_Zoll',
            'Data', 'dem', 'dem_mesh_r2.glb'
        )
        camera_position = Vector3([450.9566076750634, 5.060010188259184, 495.5558088407927])
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

            world_coord = pixel_to_world_coord(input_coord[0], input_coord[1], image_res[0], image_res[1], tri_mesh,
                                               camera)
            pixel_coord = world_to_pixel_coord(world_coord, image_res[0], image_res[1], camera, ensure_int=False)

            error = ((input_coord[0] - pixel_coord[0]) ** 2 + (input_coord[1] - pixel_coord[1]) ** 2) ** 0.5
            errors.append(error)

    avg = sum(errors) / len(errors)
    mxv = max(errors)
    mnv = min(errors)

    print('\n| metric |    avg    |    max    |    min    |\n|--------|-----------|-----------|-----------|')
    print(f'|  error | {avg:7.3e} | {mxv:7.3e} | {mnv:7.3e} |')


def test_render_labels() -> None:
    """
    Renders an integral for a real flight and draws the projected polygon labels onto it.

    Needs ``BAMBI_DEV_DIR``. ``CyclicList`` and ``LoggerCallback`` are imported lazily
    because neither resolves in the upstream project either -- see the module docstring.
    """
    from alfspy.core.util.collections.cyclic import CyclicList  # noqa: F401 - see module docstring
    from alfspy.render.render import LoggerCallback              # noqa: F401 - see module docstring

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

    @dataclass
    class FrameData:
        label_coords: list
        camera: Optional[Camera]

    # region config

    input_resolution = Resolution(1024, 1024)
    render_resolution = Resolution(720 * 16, 720 * 16)
    first_frame_idx = 100
    last_frame_idx = 10900
    frame_count = last_frame_idx - first_frame_idx

    render = False

    label_colors = CyclicList((  # BGR
        (102, 0, 255),
        (255, 102, 0),
        (0, 255, 102),
        (255, 0, 102),
        (255, 102, 0),
        (0, 102, 255),
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
            camera_position_mode=CameraPositioningMode.ShotCentered, fovy=50.0, aspect_ratio=1.0, orthogonal=True,
            ortho_size=(320, 320), correction=correction, resolution=render_resolution,
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
        raise Exception('No render camera found.')

    label_done = LoggerCallback('      ')
    print('Start label projection process')

    print('  Loading Mesh')
    mesh, _ = gltf_extract(dem_file)
    tri_mesh = Trimesh(vertices=mesh.vertices, faces=mesh.indices)
    label_done()

    print('  Projecting pixels')
    render = cv2.imread(output_file)
    frame_data = defaultdict(lambda: FrameData([], None))

    with open(labels_file, 'r') as jf:
        label_data = json.load(jf)

    label_state_data = defaultdict(lambda: [])
    for i, label in enumerate(label_data):
        label_state_data[i].extend(label["states"])

    for label in label_state_data:
        for state in label_state_data[label]:
            label_coords = state["pxlCoordinates"][:4]
            color_idx = label
            frame_data[state["frameIdx"]].label_coords.append((color_idx, label_coords))

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

        frame_data[frame_idx].camera = camera

    for frame, data in frame_data.items():
        camera = data.camera

        for color_idx, poly_coords in data.label_coords:
            pixel_xs = []
            pixel_ys = []
            for pixel in poly_coords:
                pixel_xs.append(pixel['x'])
                pixel_ys.append(pixel['y'])

            w_poses = pixel_to_world_coord(pixel_xs, pixel_ys, input_resolution.width, input_resolution.height,
                                           tri_mesh, camera)
            np_poses = world_to_pixel_coord(w_poses, render_resolution.width, render_resolution.height, render_camera)
            poly_lines = [np.array(np_poses).T.reshape((-1, 1, 2))]
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
