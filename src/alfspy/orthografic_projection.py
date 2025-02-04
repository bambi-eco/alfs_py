from collections import defaultdict
import os
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
from moderngl import Context
import numpy as np

from alfspy.core.rendering import Resolution, Camera, CtxShot, RenderResultMode
from pyrr import Quaternion, Vector3
from trimesh import Trimesh

from alfspy.core.convert.convert import pixel_to_world_coord, world_to_pixel_coord
from alfspy.core.geo.transform import Transform
from alfspy.core.rendering.renderer import Renderer
from alfspy.core.util.collections.cyclic import CyclicList
from alfspy.core.util.geo import get_aabb
from alfspy.core.util.loggings import LoggerStep
from alfspy.core.util.pyrrs import quaternion_from_eulers
from alfspy.render.data import BaseSettings, CameraPositioningMode
from alfspy.render.render import make_camera, make_mgl_context, make_shot_loader, process_render_data, read_gltf, release_all


DATASET_DIR = "C:\\Users\\p42748\\Desktop\\bambi_dataset\\test_with_correction_info" #"dataset_dir"
OUTPUT_DIR = "C:\\Users\\p42748\\Desktop\\bambi_dataset\\test_projection"
SPLITS = ["train", "val", "test"]

LABEL_COLORS = CyclicList((  # BGR
        (102, 0, 255),  # '#ff0066',  #
        (255, 102, 0),  # '#0066ff',  #
        (0, 255, 102),  # '#66ff00',  #
        (255, 0, 102),  # '#6600ff',  #
        (255, 102, 0),  # '#00ff66',  #
        (0, 102, 255),  # '#ff6600',  #
    ))

# Get shots for a list of image files
def get_shots_for_files(image_files: List[str], images_folder: str, ctx: Context, correction: Transform, matched_poses: dict, cor_rotation_eulers: Vector3, cor_translation: Vector3, lazy: bool = False, fovy: float = 60.0) -> Tuple[List[CtxShot], List[str]]:
    shots = []
    shot_names = []
    shots_rotation_eulers = []
    for img_file in image_files:
        if img_file.endswith(".png") or img_file.endswith(".jpg"):
            idx = int(img_file.split("_")[1].split(".")[0])

            position = Vector3(matched_poses["images"][idx]["location"])

            # from shot.py _prosses_json() 243-249
            rotation = matched_poses["images"][idx]["rotation"]
            print("rotation", rotation)
            rotation = [val-360 if val > 360 else val for val in rotation]
            
            rot_len = len(rotation)
            if rot_len == 3:
                eulers = [np.deg2rad(val) for val in rotation]
                print("eulers", eulers)
                rotation = quaternion_from_eulers(eulers, 'zyx')
            elif rot_len == 4:
                rotation = Quaternion(rotation)
            else:
                raise ValueError(f'Invalid rotation format of length {rot_len}: {rotation}')
            
            fov = matched_poses["images"][idx]["fovy"][0]
            if fov is None:
                fov = fovy
            elif isinstance(fov, list):
                fov = fov[0]

            shots.append(CtxShot(ctx, os.path.join(images_folder, img_file), position, rotation, fov, 1, correction, lazy))
            shots_rotation_eulers.append(eulers)
            shot_names.append(img_file)
    return shots, shot_names, shots_rotation_eulers


def get_axis_aligned_bounding_box(frame_label) -> List[np.ndarray]:
    # Extract the coordinates array (first element since it's a list with one array)
    coords = frame_label[0]
    
    # Get all x coordinates (first column) and y coordinates (second column)
    x_coords = coords[:, 0, 0]
    y_coords = coords[:, 0, 1]
    
    # Get min and max values
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)
    
    # Create axis-aligned bounding box coordinates in the same format as input
    box_coords = np.array([
        [[x_min, y_min]],
        [[x_max, y_min]],
        [[x_max, y_max]],
        [[x_min, y_max]]
    ])
    
    return [box_coords]  # Return as list with one array to match input format

def to_yolo_format(axis_aligned_bounding_box: List[np.ndarray], img_width: int, img_height: int) -> str:
    # Extract coordinates from the nested arrays (format of get_axis_aligned_bounding_box return)
    x_min = axis_aligned_bounding_box[0][0][0][0]
    y_min = axis_aligned_bounding_box[0][0][0][1]
    x_max = axis_aligned_bounding_box[0][2][0][0]
    y_max = axis_aligned_bounding_box[0][2][0][1]

    # Calculate center coordinates and dimensions
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    return f"{x_center} {y_center} {width} {height}"

def project_images_for_flight(flight_key: int, split: str, images_folder: str, labels_folder: str, dem_file: str, poses_file: str, correction_matrix_file: str):
    print("processing flight", flight_key)

    frame_files = [f for f in os.listdir(images_folder) if f.startswith(str(flight_key))]

    print(len(frame_files))

    output_images_folder = os.path.join(OUTPUT_DIR, 'images', split)
    output_labels_folder = os.path.join(OUTPUT_DIR, 'labels', split)

    with open(correction_matrix_file, 'r') as file:
        correction = json.load(file)

    # from test.py 248-258
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

    with open(poses_file, 'r') as file:
        matched_poses = json.load(file)

    # endregion

    # region config
    input_resolution = Resolution(1024, 1024)
    render_resolution = Resolution(1024, 1024)

    # TODO: check if those settings are correct, 
    # especially camera_dist (seems to change nothing), 
    # camera_position_mode (not really changing anything relevant), 
    # fovy (no changes afaict), 
    # ortho_size (changes a lot but i am not sure if it is relevant), 
    # correction, 
    # resolution
    settings = BaseSettings(
        count=len(frame_files), initial_skip=0, add_background=True, camera_dist=10.0,
        camera_position_mode=CameraPositioningMode.FirstShot, fovy=50.0, aspect_ratio=1.0, orthogonal=True,
        ortho_size=(70, 70), correction=correction, resolution=render_resolution
    )
    # endregion

    # region setup
    # basically render.py _base_steps() with the cange of using the files form the folder and not the poses file
    mesh_data, texture_data = read_gltf(dem_file)
    tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
    mesh_data, texture_data = process_render_data(mesh_data, texture_data)

    ctx = make_mgl_context()

    shots, shot_names, shots_rotation_eulers = get_shots_for_files(frame_files, images_folder, ctx, correction, matched_poses, cor_rotation_eulers, cor_translation)

    mesh_aabb = get_aabb(mesh_data.vertices)
    # endregion

    exit_after_x_shots = 3
    shots_processed = 0
    # region orthographic projection
    for shot, shot_name, shot_rotation_eulers in zip(shots, shot_names, shots_rotation_eulers):
        frame_idx = int(shot_name.split("_")[1].split(".")[0])
        # Create new camera for each shot TODO idk if this is needed, i guess every shot has a slightly different camera position, so this is just a try in figuring out what is wrong
        print("shot_rotation_eulers", shot_rotation_eulers)
        single_shot_camera = make_camera(mesh_aabb, [shot], settings, rotation=Quaternion.from_eulers([0.0, 0.0, -(shot_rotation_eulers[0] + cor_rotation_eulers[2])])) # shot_rotation_eulers zyx
        print(f'Computed camera position for shot {shot_name}: {single_shot_camera.transform.position}')

        # Create new renderer with the single-shot camera
        renderer = Renderer(settings.resolution, ctx, single_shot_camera, mesh_data, texture_data)
        
        shot_loader = make_shot_loader([shot])  # Create loader for single shot
        save_name = os.path.join(output_images_folder, f"{shot_name}")
        renderer.project_shots(
            shot_loader, 
            RenderResultMode.Complete, 
            mask=None, 
            integral=False, 
            save=True, 
            release_shots=True, 
            save_name_iter=iter([save_name])
        )
        
        # Clean up renderer after each shot
        renderer.release()

        # project labels (similar to test.py 317-346 )
        print('Start label projection process')
        render = cv2.imread(save_name)

        labels_file = os.path.join(labels_folder, shot_name.split('.')[0] + '.txt')
        
        # Read and parse the label file for the current shot/frame idx
        frame_labels = []
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                for line in f:

                    # YOLO format: class x_center y_center width height
                    values = line.strip().split()
                    if len(values) == 5:
                        class_id = int(values[0])
                        x_center = float(values[1])
                        y_center = float(values[2])
                        width = float(values[3])
                        height = float(values[4])
                        
                        # Convert to pixel coordinates
                        img_height, img_width = settings.resolution.height, settings.resolution.width
                        x1 = int((x_center - width/2) * img_width)
                        y1 = int((y_center - height/2) * img_height)
                        x2 = int((x_center + width/2) * img_width)
                        y2 = int((y_center + height/2) * img_height)

                        print("x1", x1)
                        print("y1", y1)
                        print("x2", x2)
                        print("y2", y2)

                        frame_labels.append(
                                    (class_id, [{"x": x1, "y": y1},
                                    {"x": x2, "y": y1},
                                    {"x": x2, "y": y2},
                                    {"x": x1, "y": y2}])
                        )

        # get info from matched_poses file
        cur_frame_data = matched_poses['images'][frame_idx]
        fovy = cur_frame_data['fovy'][0]

        position = Vector3(cur_frame_data['location'])
        rotation_eulers = Vector3([np.deg2rad(val % 360.0) for val in cur_frame_data['rotation']]) + cor_rotation_eulers

        position += cor_translation
        rotation = Quaternion.from_eulers(rotation_eulers)

        camera = Camera(fovy=fovy, aspect_ratio=1.0, position=position, rotation=rotation)

        labels_axis_aligned = []
        for class_id, poly_coords in frame_labels:
            pixel_xs = []
            pixel_ys = []
            for pixel in poly_coords:
                pixel_xs.append(pixel['x'])
                pixel_ys.append(pixel['y'])

            w_poses = pixel_to_world_coord(pixel_xs, pixel_ys, input_resolution.width, input_resolution.height, tri_mesh, camera)
            np_poses = world_to_pixel_coord(w_poses, render_resolution.width, render_resolution.height, single_shot_camera)

            poly_lines = [np.array(np_poses).T.reshape((-1, 1, 2))]

            axis_aligned_bounding_box = get_axis_aligned_bounding_box(poly_lines)
            cv2.polylines(render, poly_lines, True, LABEL_COLORS[class_id], thickness=1)
            cv2.polylines(render, axis_aligned_bounding_box, True, (255, 255, 0), thickness=1)
            labels_axis_aligned.append({'animal_class': class_id, 'axis_aligned_bounding_box': axis_aligned_bounding_box})

        # save labels_axis_aligned in a file in the format: animal_class center_x center_y width height
        with open(os.path.join(output_labels_folder, f"{shot_name.split('.')[0]}.txt"), 'w') as f:
            for label in labels_axis_aligned:
                f.write(f"{label['animal_class']} {to_yolo_format(label['axis_aligned_bounding_box'], settings.resolution.width, settings.resolution.height)}\n")

        print('  Saving labeled image')
        labeled_output_file = os.path.join(output_images_folder, f"labeled_{shot_name}")
        cv2.imwrite(labeled_output_file, render)

        # TODO: remove this
        shots_processed += 1
        if shots_processed >= exit_after_x_shots:
            print("done with shots", shots_processed)
            break

    release_all(ctx, renderer, shots)


# Get the included flights from the export json file
def get_included_flights(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data["metadata"]["included_flights"]


# Export images for a split (requires the dataset to be in the correct format)
def project_images_for_split(split: str):
    print("projecting images for split ", split)
    images_folder = os.path.join(DATASET_DIR, "images", split)
    labels_folder = os.path.join(DATASET_DIR, "labels", split)

    # get all flights in the split
    flight_keys = get_included_flights(os.path.join(DATASET_DIR, f"export_{split}.json"))

    for flight_key in flight_keys:
        dem_file = os.path.join(DATASET_DIR, "correction_data", f"{flight_key}_dem.glb")
        poses_file = os.path.join(DATASET_DIR, "correction_data", f"{flight_key}_matched_poses.json")
        correction_matrix_file = os.path.join(DATASET_DIR, "correction_data", f"{flight_key}_correction.json")
        
        project_images_for_flight(flight_key, split, images_folder, labels_folder, dem_file, poses_file, correction_matrix_file)

        #exit()



if __name__ == "__main__":
    # create output folders if needed
    output_images_folder = os.path.join(OUTPUT_DIR, 'images')
    output_labels_folder = os.path.join(OUTPUT_DIR, 'labels')

    if not os.path.exists(output_images_folder):
        os.makedirs(output_images_folder)
    if not os.path.exists(output_labels_folder):
        os.makedirs(output_labels_folder)

    for split in SPLITS:
        if not os.path.exists(os.path.join(output_images_folder, split)):
            os.makedirs(os.path.join(output_images_folder, split))
        if not os.path.exists(os.path.join(output_labels_folder, split)):
            os.makedirs(os.path.join(output_labels_folder, split))

    # project images for each flight
    for split in SPLITS:
        project_images_for_split(split)
        print("done with split", split)

        #exit()
    print("done")

