import json
import os
from typing import List, Dict

import cv2
import numpy as np
from moderngl import Context
from pyrr import Vector3, Quaternion
from trimesh import Trimesh

from alfspy.core.geo import Transform
from alfspy.core.rendering import Resolution, CtxShot, Renderer, RenderResultMode, TextureData
from alfspy.core.util.geo import get_aabb
from alfspy.core.util.pyrrs import quaternion_from_eulers
from alfspy.orthografic_projection import get_shots_for_files
from alfspy.render.data import BaseSettings, CameraPositioningMode
from alfspy.render.render import read_gltf, process_render_data, make_mgl_context, make_camera, make_shot_loader, \
    release_all

if __name__ == '__main__':
    images_folder = r"C:\Users\P41743\Desktop\New folder\Frames_T"
    poses_file = r"C:\Users\P41743\Desktop\New folder\Frames_T\matched_poses.json"
    mask_file = r"C:\Users\P41743\Desktop\New folder\Frames_T\mask_T.png"
    dem_file = r"C:\Users\P41743\Desktop\New folder\dem_mesh_r2.gltf"
    target_folder = r"C:\Users\P41743\Desktop\New folder\res"
    correction_matrix_file = None
    frame_idx = 564
    neighbor_range = 100
    start_distance = 200
    #
    # ##################################
    # mask = TextureData(CtxShot._cvt_img(cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)))
    # mesh_data, texture_data = read_gltf(dem_file)
    # tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
    # mesh_data, texture_data = process_render_data(mesh_data, texture_data)
    #
    # with open(poses_file) as f:
    #     poses = json.load(f)
    #
    # if correction_matrix_file is not None:
    #     with open(correction_matrix_file, 'r') as file:
    #         correction_data = json.load(file)
    #     translation = correction_data.get('translation', {'x': 0, 'y': 0, 'z': 0})
    #     cor_translation = Vector3([translation['x'], translation['y'], translation['z']], dtype='f4')
    #
    #     rotation = correction_data.get('rotation', {'x': 0, 'y': 0, 'z': 0})
    #     cor_rotation_eulers = Vector3([rotation['x'], rotation['y'], rotation['z']], dtype='f4')
    #     correction = Transform(cor_translation, Quaternion.from_eulers(cor_rotation_eulers))
    # else:
    #     cor_translation = Vector3()
    #     cor_rotation_eulers = Vector3()
    #     correction = Transform()
    #
    # images = poses["images"]
    # center_frame = [images[frame_idx]]
    # previous_frames = images[frame_idx - neighbor_range:frame_idx]
    # additional_frames = images[frame_idx + 1: frame_idx + 1 + neighbor_range]
    # ctx = make_mgl_context()
    #
    # def get_shots_for_files(image_files: List[Dict[str, any]], images_folder: str, ctx: Context, correction: Transform, lazy: bool = False, fovy: float = 60.0):
    #     shots = []
    #     shot_names = []
    #     shots_rotation_eulers = []
    #     for img_file in image_files:
    #         position = Vector3(img_file["location"])
    #
    #         # from shot.py _prosses_json() 243-249
    #         rotation = img_file["rotation"]
    #         rotation = [val % 360.0 for val in rotation]
    #
    #         rot_len = len(rotation)
    #         if rot_len == 3:
    #             eulers = [np.deg2rad(val) for val in rotation]
    #             rotation = quaternion_from_eulers(eulers, 'zyx')
    #         elif rot_len == 4:
    #             rotation = Quaternion(rotation)
    #         else:
    #             raise ValueError(f'Invalid rotation format of length {rot_len}: {rotation}')
    #
    #         fov = img_file["fovy"][0]
    #         if fov is None:
    #             fov = fovy
    #         elif isinstance(fov, list):
    #             fov = fov[0]
    #
    #         shots.append(CtxShot(ctx, os.path.join(images_folder, img_file["imagefile"]), position, rotation, fov, 1, correction, lazy))
    #         shots_rotation_eulers.append(eulers)
    #         shot_names.append(img_file)
    #
    #     return shots, shot_names, shots_rotation_eulers
    #
    #
    # shots, shot_names, shots_rotation_eulers = get_shots_for_files(center_frame, images_folder, ctx, correction)
    # shot_name = shot_names[0]
    # shot_rotation_eulers = shots_rotation_eulers[0]
    # prev_shots, prev_shot_names, prev_shots_rotation_eulers = get_shots_for_files(previous_frames, images_folder, ctx, correction)
    # add_shots, add_shot_names, add_shots_rotation_eulers = get_shots_for_files(additional_frames, images_folder, ctx, correction)
    # all_shots = prev_shots + shots + add_shots
    # mesh_aabb = get_aabb(mesh_data.vertices)
    #
    # settings = BaseSettings(
    #     count=len(previous_frames) + len(additional_frames) + 1, initial_skip=False, add_background=True, camera_dist=start_distance,
    #     camera_position_mode=CameraPositioningMode.FirstShot, fovy=50, aspect_ratio=1.0, orthogonal=True,
    #     ortho_size=(70, 70), correction=correction, resolution=Resolution(2048, 2048)
    # )
    #
    # single_shot_camera = make_camera(mesh_aabb, shots, settings, rotation=Quaternion.from_eulers(
    #     [(shot_rotation_eulers[0] - cor_rotation_eulers[0]), (shot_rotation_eulers[1] - cor_rotation_eulers[1]),
    #      (shot_rotation_eulers[2] - cor_rotation_eulers[2])]))
    #
    # # Create new renderer with the single-shot camera
    # renderer = Renderer(settings.resolution, ctx, single_shot_camera, mesh_data, texture_data)
    #
    # shot_loader = make_shot_loader(shots)  # Create loader for single shot
    # renderer.project_shots(
    #     shot_loader,
    #     RenderResultMode.ShotOnly,
    #     mask=mask,
    #     integral=False,
    #     save=True,
    #     release_shots=False,
    #     save_name_iter=iter([os.path.join(target_folder, f"{start_distance}.png")])
    # )
    # renderer.release()
    #
    # # for i in range(start_distance, 1, -1):
    # #     print(f"Rendering shot {i}")
    # #     settings = BaseSettings(
    # #         count=len(previous_frames) + len(additional_frames) + 1, initial_skip=False, add_background=True,
    # #         camera_dist=i,
    # #         camera_position_mode=CameraPositioningMode.CenterShot, fovy=50, aspect_ratio=1.0, orthogonal=True,
    # #         ortho_size=(70, 70), correction=correction, resolution=Resolution(2048, 2048)
    # #     )
    # #     single_shot_camera = make_camera(mesh_aabb, shots, settings, rotation=Quaternion.from_eulers(
    # #         [(shot_rotation_eulers[0] - cor_rotation_eulers[0]), (shot_rotation_eulers[1] - cor_rotation_eulers[1]),
    # #          (shot_rotation_eulers[2] - cor_rotation_eulers[2])]))
    # #     renderer = Renderer(settings.resolution, ctx, single_shot_camera, mesh_data, texture_data)
    # #     renderer.apply_matrices()
    # #
    # #     # for shot in all_shots:
    # #     #     shot.correction.position.z += 1
    # #
    # #     shot_loader = make_shot_loader(all_shots)
    # #     save_name = os.path.join(target_folder, f"{start_distance - i + 1}.png")
    # #     renderer.render_integral(shot_loader,
    # #                              mask=None,
    # #                              save=True,
    # #                              release_shots=False,
    # #                              save_name=save_name)
    # #     renderer.release()
    # release_all(ctx, all_shots)
