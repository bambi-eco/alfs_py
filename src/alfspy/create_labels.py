import json
import os
from typing import List, TYPE_CHECKING

import cv2
import numpy as np
from pyrr import Vector3, Quaternion
from trimesh import Trimesh
# TorchContext is used for annotations only, and ``alfspy.core.torchgl`` imports torch
# at module level. Importing it eagerly would make torch a hard requirement of this
# module, and so of every engine -- exactly what the optional backend extras exist to
# avoid. The annotations that use it are strings for the same reason.
if TYPE_CHECKING:  # pragma: no cover
    from alfspy.core.torchgl import TorchContext
from alfspy.core.convert import pixel_to_world_coord, world_to_pixel_coord
from alfspy.core.geo import Transform
from alfspy.core.rendering import Resolution, Camera, CtxShot
from alfspy.core.util.geo import get_aabb
from alfspy.core.util.pyrrs import quaternion_from_drone_pose
from alfspy.orthografic_projection import get_axis_aligned_bounding_box
from alfspy.render.data import BaseSettings, CameraPositioningMode
from alfspy.render.render import read_gltf, process_render_data, make_camera, make_torch_context


def get_shots_for_files(ctx: 'TorchContext', mask, correction: Transform,
                        matched_poses: dict, lazy: bool = False, fovy: float = 60.0):
    shots = []
    shot_names = []
    shots_rotation_eulers = []
    num_images = len(matched_poses["images"])
    for idx, image in enumerate(matched_poses["images"]):
            position = Vector3(image["location"])

            # from shot.py _prosses_json() 243-249
            rotation = image["rotation"]
            rotation = [val % 360.0 for val in rotation]

            rot_len = len(rotation)
            if rot_len == 3:
                rotation = quaternion_from_drone_pose(rotation)
            elif rot_len == 4:
                rotation = Quaternion(rotation)
            else:
                raise ValueError(f'Invalid rotation format of length {rot_len}: {rotation}')

            fov = image["fovy"][0]
            if fov is None:
                fov = fovy
            elif isinstance(fov, list):
                fov = fov[0]

            shots.append(
                CtxShot(ctx, mask, position, rotation, fov, 1, correction, lazy))
            print(f"{idx}/{num_images}")
            shots_rotation_eulers.append(eulers)
            shot_names.append(idx)

    return shots, shot_names, shots_rotation_eulers

if __name__ == '__main__':
    flight_keys = range(367, 396)
    base_dir = r"C:\Users\P41743\Desktop\labels"
    label_positions = [
        [2579, [
                    {
                        "x": 883,
                        "y": 121
                    },
                    {
                        "x": 921,
                        "y": 121
                    },
                    {
                        "x": 921,
                        "y": 157
                    },
                    {
                        "x": 883,
                        "y": 157
                    },
                    {
                        "x": 883,
                        "y": 121
                    }
                ]
         ],
        [3317, [
                    {
                        "x": 931,
                        "y": 499
                    },
                    {
                        "x": 988,
                        "y": 499
                    },
                    {
                        "x": 988,
                        "y": 549
                    },
                    {
                        "x": 931,
                        "y": 549
                    },
                    {
                        "x": 931,
                        "y": 499
                    }
                ]
        ]
    ]

    ctx = make_torch_context()

    # for flight_id in flight_ids:
    flight_key = 377
    dem_file = os.path.join(base_dir, f"{flight_key}_dem.glb")
    mesh_data, texture_data = read_gltf(dem_file)
    tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
    mesh_data, texture_data = process_render_data(mesh_data, texture_data)

    poses_file = os.path.join(base_dir, f"{flight_key}_matched_poses.json")
    mask_file = os.path.join(base_dir, f"{flight_key}_mask.png")
    mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
    # correction_matrix_file = os.path.join(base_dir, f"{flight_key}_correction.json")
    with open(poses_file) as f:
        poses = json.load(f)

    # TODO create virtual poses and adapt file name -> load only this one shot as part of for-loop
    shots, shot_names, shots_rotation_eulers = get_shots_for_files(ctx, mask_file, None, poses)


    settings = BaseSettings(
        count=len(poses["images"]), initial_skip=0, add_background=True, camera_dist=30,
        camera_position_mode=CameraPositioningMode.FirstShot, fovy=50, aspect_ratio=1, orthogonal=True,
        ortho_size=(70, 70), correction=None, resolution=Resolution(1024, 1024)
    )
    mesh_aabb = get_aabb(mesh_data.vertices)

    labels = []
    for label_position in label_positions:
        idx = label_position[0]
        cur_frame_data = poses["images"][idx]
        shot = shots[idx]
        shot_rotation_eulers = shots_rotation_eulers[idx]
        single_shot_camera = make_camera(mesh_aabb, [shot], settings, rotation=Quaternion.from_eulers(
            [(shot_rotation_eulers[0]), (shot_rotation_eulers[1]),
             (shot_rotation_eulers[2])]))

        fovy = cur_frame_data['fovy'][0]
        position = Vector3(cur_frame_data['location'])
        rotation_eulers = (Vector3([np.deg2rad(val % 360.0) for val in cur_frame_data['rotation']])) * -1
        rotation = Quaternion.from_eulers(rotation_eulers)

        camera = Camera(fovy=fovy, aspect_ratio=1.0, position=position, rotation=rotation)

        pixel_xs = []
        pixel_ys = []
        for pixel in label_position[1]:
            pixel_xs.append(pixel['x'])
            pixel_ys.append(pixel['y'])
        w_poses = pixel_to_world_coord(pixel_xs, pixel_ys, 1024, 1024, tri_mesh, camera)
        labels.append(w_poses)

    for idx, cur_frame_data in enumerate(poses["images"]):
        # shot = shots[idx]
        # shot_rotation_eulers = shots_rotation_eulers[idx]
        # single_shot_camera = make_camera(mesh_aabb, [shot], settings, rotation=Quaternion.from_eulers(
        #     [(shot_rotation_eulers[0]), (shot_rotation_eulers[1]),
        #      (shot_rotation_eulers[2])]))

        fovy = cur_frame_data['fovy'][0]
        position = Vector3(cur_frame_data['location'])
        rotation_eulers = (Vector3([np.deg2rad(val % 360.0) for val in cur_frame_data['rotation']])) * -1
        rotation = Quaternion.from_eulers(rotation_eulers)

        camera = Camera(fovy=fovy, aspect_ratio=1.0, position=position, rotation=rotation)

        for w_poses in labels:
            np_poses = world_to_pixel_coord(w_poses, 1024, 1024, camera)
            poly_lines = [np.array(np_poses).T.reshape((-1, 1, 2))]
            axis_aligned_bounding_box = get_axis_aligned_bounding_box(poly_lines)
        print()



