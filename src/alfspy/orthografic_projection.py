from collections import defaultdict
import os
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from pathlib import Path
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

import logging
import sys

LABEL_COLORS = CyclicList((  # BGR
        (102, 0, 255),  # '#ff0066',  #
        (255, 102, 0),  # '#0066ff',  #
        (0, 255, 102),  # '#66ff00',  #
        (255, 0, 102),  # '#6600ff',  #
        (255, 102, 0),  # '#00ff66',  #
        (0, 102, 255),  # '#ff6600',  #
    ))

def polyline_to_bounding_box(polyline: List[int]) -> Tuple[int, int, int, int]:
    xs = []
    ys = []
    for coordinate_idx, coordinate in enumerate(polyline):
        if coordinate_idx % 2 == 0:
            xs.append(int(coordinate))
        else:
            ys.append(int(coordinate))

    start_x = min(xs)
    start_y = min(ys)
    end_x = max(xs)
    end_y = max(ys)

    return start_x, start_y, end_x, end_y

# def extend_labels(labels, config):
#     for label in labels:
#         bb = polyline_to_bounding_box(label["coordinates"])
#         label_id = config["metadata"]["label_backward_mapping"][label["label"]]
#         label["bb"] = bb
#         label["label_id"] = label_id

# Get shots for a list of image files
def get_shots_for_files(image_files: List[str], images_folder: str, ctx: Context, correction: Transform, matched_poses: dict, lazy: bool = False, fovy: float = 60.0):
    shots = []
    shot_names = []
    shots_rotation_eulers = []
    for img_file in image_files:
        if img_file.endswith(".png") or img_file.endswith(".jpg"):
            idx = int(img_file.split("_")[1].split(".")[0])

            position = Vector3(matched_poses["images"][idx]["location"])

            # from shot.py _prosses_json() 243-249
            rotation = matched_poses["images"][idx]["rotation"]
            rotation = [val % 360.0 for val in rotation]
            
            rot_len = len(rotation)
            if rot_len == 3:
                eulers = [np.deg2rad(val) for val in rotation]
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

    # check if labels are in the frame, if not return None
    if x_center + width/2 < 0 or x_center - width/2 > 1 or y_center + height/2 < 0 or y_center - height/2 > 1:
        return None

    return f"{x_center} {y_center} {width} {height}"

def get_camera_for_frame(matched_poses, frame_idx, cor_rotation_eulers, cor_translation):
    cur_frame_data = matched_poses['images'][frame_idx]
    fovy = cur_frame_data['fovy'][0]

    position = Vector3(cur_frame_data['location'])
    rotation_eulers = (Vector3(
        [np.deg2rad(val % 360.0) for val in cur_frame_data['rotation']]) - cor_rotation_eulers) * -1

    position += cor_translation
    rotation = Quaternion.from_eulers(rotation_eulers)

    return Camera(fovy=fovy, aspect_ratio=1.0, position=position, rotation=rotation)


def project_label(label_coordinates, input_resolution, tri_mesh, camera, render_resolution, single_shot_camera):
    pixel_xs = []
    pixel_ys = []
    for pixel_id, pixel in enumerate(label_coordinates):
        if pixel_id % 2 == 0:
            pixel_xs.append(int(pixel))
        else:
            pixel_ys.append(int(pixel))

    w_poses = pixel_to_world_coord(pixel_xs, pixel_ys, input_resolution.width, input_resolution.height, tri_mesh, camera,
                                   include_misses=False)
    np_poses = world_to_pixel_coord(w_poses, render_resolution.width, render_resolution.height, single_shot_camera)

    return [np.array(np_poses).T.reshape((-1, 1, 2))]


def project_images_for_flight(flight_key: int, split: str, images_folder: str, labels_folder: str, dem_file: str, poses_file: str, correction_matrix_file: str,
                              OUTPUT_DIR: str, ORTHO_WIDTH: int, ORTHO_HEIGHT: int, RENDER_WIDTH: int, RENDER_HEIGHT: int, CAMERA_DISTANCE: int,
                              INITIAL_SKIP: int, ADD_BACKGROUND: bool, FOVY: float, ASPECT_RATIO: float, SAVE_LABELED_IMAGES: bool, INPUT_WIDTH:int, INPUT_HEIGHT:int,
                              config: Dict[str, any], project_orthogonal:bool, ADDITIONAL_ROTATIONS: int, ROTATION_LIMIT: float, merge_labels_in_alfs: bool,
                              APPLY_NMS:bool, NMS_IOU: float, rng: np.random.Generator):
    logging.info(f"processing flight: {flight_key}", )
    flight_key_str = str(flight_key)
    mission_id = config["flight_to_mission_mapping"][flight_key_str]
    nr_of_frames_after_current = config["missions"][mission_id]["flights"][flight_key_str]["nr_of_frames_after_current"]
    nr_of_frames_before_current = config["missions"][mission_id]["flights"][flight_key_str]["nr_of_frames_before_current"]

    frame_files = [f for f in os.listdir(images_folder) if f.split("_")[0] == flight_key_str]

    logging.info(len(frame_files))

    if len(frame_files) == 0:
        return

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
    input_resolution = Resolution(INPUT_WIDTH, INPUT_HEIGHT)
    render_resolution = Resolution(RENDER_WIDTH, RENDER_HEIGHT)

    settings = BaseSettings(
        count=len(frame_files), initial_skip=INITIAL_SKIP, add_background=ADD_BACKGROUND, camera_dist=CAMERA_DISTANCE,
        camera_position_mode=CameraPositioningMode.FirstShot, fovy=FOVY, aspect_ratio=ASPECT_RATIO, orthogonal=True,
        ortho_size=(ORTHO_WIDTH, ORTHO_HEIGHT), correction=correction, resolution=render_resolution
    )
    # endregion

    # region setup
    # basically render.py _base_steps() with the cange of using the files form the folder and not the poses file
    mesh_data, texture_data = read_gltf(dem_file)
    tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
    mesh_data, texture_data = process_render_data(mesh_data, texture_data)

    ctx = make_mgl_context()

    shots, shot_names, shots_rotation_eulers = get_shots_for_files(frame_files, images_folder, ctx, correction, matched_poses)

    mesh_aabb = get_aabb(mesh_data.vertices)
    # endregion

    # exit_after_x_shots = 3
    # shots_processed = 0

    for shot, shot_name, shot_rotation_eulers in zip(shots, shot_names, shots_rotation_eulers):
        frame_idx = int(shot_name.split("_")[1].split(".")[0])

        random_z_rotations = [0.0]
        if ADDITIONAL_ROTATIONS > 0:
            random_z_rotations.extend(rng.uniform(-ROTATION_LIMIT, ROTATION_LIMIT, ADDITIONAL_ROTATIONS))

        logging.info(random_z_rotations)

        prev_shots = []
        add_shots = []
        previous_frames = []
        additional_frames = []
        previous_frame_ids = []
        additional_frame_ids = []

        if not project_orthogonal:
            neighbour_folder = os.path.join(str(Path(images_folder).parent) + "_neighbours", split)
            for neighbour_frame in os.listdir(neighbour_folder):
                splits = neighbour_frame.split("_")
                if splits[0] != str(flight_key):
                    continue
                neighbour_id = int(Path(splits[1]).stem)
                if frame_idx > neighbour_id >= frame_idx - nr_of_frames_before_current:
                    previous_frames.append(neighbour_frame)
                    previous_frame_ids.append(neighbour_id)

                if frame_idx < neighbour_id <= frame_idx + nr_of_frames_after_current:
                    additional_frames.append(neighbour_frame)
                    additional_frame_ids.append(neighbour_id)

            prev_shots, _, _ = get_shots_for_files(previous_frames, neighbour_folder, ctx, correction, matched_poses)
            add_shots, _, _ = get_shots_for_files(additional_frames, neighbour_folder, ctx, correction, matched_poses)


        for random_z_rotation in random_z_rotations:	
            logging.info(f"random_z_rotation: {random_z_rotation}")
            # region image projection
            # Create new camera for each shot
            single_shot_camera = make_camera(mesh_aabb, [shot], settings, rotation= Quaternion.from_eulers([(shot_rotation_eulers[0] - cor_rotation_eulers[0]), (shot_rotation_eulers[1] - cor_rotation_eulers[1]), (shot_rotation_eulers[2] - cor_rotation_eulers[2]) + random_z_rotation]))
            logging.info(f"single_shot_camera created")
            # Create new renderer with the single-shot camera
            renderer = Renderer(settings.resolution, ctx, single_shot_camera, mesh_data, texture_data)
            logging.info(f"renderer created")
            if random_z_rotation != 0.0:
                save_name = os.path.join(output_images_folder, f"{shot_name.split('.')[0]}_{str(random_z_rotation).replace('.', '_')}.jpg")
            else:
                save_name = os.path.join(output_images_folder, f"{shot_name}")

            logging.info(f"saving image to {save_name}")

            if project_orthogonal:
                shot_loader = make_shot_loader([shot])  # Create loader for single shot
                renderer.project_shots(
                    shot_loader,
                    RenderResultMode.ShotOnly,
                    mask=None,
                    integral=False,
                    save=True,
                    release_shots=False,
                    save_name_iter=iter([save_name])
                )
            else:
                shot_loader = make_shot_loader(prev_shots + [shot] + add_shots)
                renderer.render_integral(shot_loader,
                    mask=None,
                    save=True,
                    release_shots=False,
                    save_name=save_name)
            logging.info(f"saved image to {save_name}")
            # Clean up renderer after each shot
            renderer.release()
            # endregion

            # region label projection
            # project labels (similar to test.py 317-346 )
            logging.info('Start label projection process')
            render = cv2.imread(save_name)


            # Read and parse the label file for the current shot/frame idx
            frame_labels = []
            labels_axis_aligned = []
            if project_orthogonal or (not project_orthogonal and not merge_labels_in_alfs):
                labels_file = os.path.join(labels_folder, shot_name.split('.')[0] + '.txt')
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
                                img_height, img_width = input_resolution.height, input_resolution.width
                                x1 = int((x_center - width/2) * img_width)
                                y1 = int((y_center - height/2) * img_height)
                                x2 = int((x_center + width/2) * img_width)
                                y2 = int((y_center + height/2) * img_height)

                                logging.info(f"x1: {x1}")
                                logging.info(f"y1: {y1}")
                                logging.info(f"x2: {x2}")
                                logging.info(f"y2: {y2}")

                                frame_labels.append((class_id, [x1, y1, x2, y1, x2, y2, x1, y2])
                                )

                    # project labels
                    camera = get_camera_for_frame(matched_poses, frame_idx, cor_rotation_eulers, cor_translation)
                    for class_id, poly_coords in frame_labels:
                        poly_lines = project_label(poly_coords, input_resolution, tri_mesh, camera, render_resolution, single_shot_camera)
                        axis_aligned_bounding_box = get_axis_aligned_bounding_box(poly_lines)
                        labels_axis_aligned.append(
                            {'animal_class': class_id, 'axis_aligned_bounding_box': axis_aligned_bounding_box})
            else:
                labels = config["missions"][mission_id]["flights"][flight_key_str]["frame_to_labels_mapping"]
                projected_labels = {}

                # get all labels of central frame and project them
                camera = get_camera_for_frame(matched_poses, frame_idx, cor_rotation_eulers, cor_translation)
                for label in labels.get(frame_idx, {}):
                    poly_lines = project_label(label["coordinates"][:-2], input_resolution, tri_mesh, camera, render_resolution, single_shot_camera)
                    axis_aligned_bounding_box = get_axis_aligned_bounding_box(poly_lines)
                    class_id = int(config["metadata"]['label_backward_mapping'][label["label"]])
                    if projected_labels.get(label["track"]) is None:
                        projected_labels[label["track"]] = []
                    projected_labels[label["track"]].append({'animal_class': class_id, 'axis_aligned_bounding_box': axis_aligned_bounding_box})

                # get all labels of previous frames and project them
                for previous_frame_id in previous_frame_ids:
                    l = labels.get(previous_frame_id)
                    if l is None:
                        continue
                    for previous_frame_label in l:
                        camera = get_camera_for_frame(matched_poses, previous_frame_id, cor_rotation_eulers, cor_translation)
                        poly_lines = project_label(previous_frame_label["coordinates"][:-2], input_resolution, tri_mesh, camera,
                                                   render_resolution, single_shot_camera)
                        axis_aligned_bounding_box = get_axis_aligned_bounding_box(poly_lines)
                        class_id = int(config["metadata"]['label_backward_mapping'][previous_frame_label["label"]])
                        if projected_labels.get(previous_frame_label["track"]) is None:
                            projected_labels[previous_frame_label["track"]] = []
                        projected_labels[previous_frame_label["track"]].append({'animal_class': class_id, 'axis_aligned_bounding_box': axis_aligned_bounding_box})

                # get all labels of additional frames and project them
                for additional_frame_id in additional_frame_ids:
                    l = labels.get(additional_frame_id)
                    if l is None:
                        continue
                    for additional_frame_label in l:
                        camera = get_camera_for_frame(matched_poses, additional_frame_id, cor_rotation_eulers,
                                                      cor_translation)
                        poly_lines = project_label(additional_frame_label["coordinates"][:-2], input_resolution, tri_mesh,
                                                   camera,
                                                   render_resolution, single_shot_camera)
                        axis_aligned_bounding_box = get_axis_aligned_bounding_box(poly_lines)
                        class_id = int(config["metadata"]['label_backward_mapping'][additional_frame_label["label"]])
                        if projected_labels.get(additional_frame_label["track"]) is None:
                            projected_labels[additional_frame_label["track"]] = []
                        projected_labels[additional_frame_label["track"]].append({'animal_class': class_id, 'axis_aligned_bounding_box': axis_aligned_bounding_box})

                # calculate minimal bounding box around all bounding boxes of a track in the scene
                for track_id, label_states in projected_labels.items():
                    bboxes = np.array([label_state["axis_aligned_bounding_box"] for label_state in label_states])
                    bboxes = bboxes.reshape(-1, 2)
                    x_min, y_min = bboxes.min(axis=0)
                    x_max, y_max = bboxes.max(axis=0)
                    enclosing_bbox = [np.array([
                        [[x_min, y_min]],
                        [[x_max, y_min]],
                        [[x_max, y_max]],
                        [[x_min, y_max]]
                    ])]
                    labels_axis_aligned.append({'animal_class': label_states[0]["animal_class"], 'axis_aligned_bounding_box': enclosing_bbox})

            if APPLY_NMS:
                boxes = []
                for box_dict in labels_axis_aligned:
                    box = box_dict['axis_aligned_bounding_box']
                    x_coords = [pt[0][0] for pt in box[0] if len(box) == 1]
                    y_coords = [pt[0][1] for pt in box[0] if len(box) == 1]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    boxes.append([x_min, y_min, x_max, y_max])
                # Prepare data for NMS
                boxes_array = np.array(boxes, dtype=float)

                # Define dummy scores (required by OpenCV's NMS)
                scores = np.array([1.0] * len(boxes))

                # Apply NMS
                indices = cv2.dnn.NMSBoxes(
                    bboxes=boxes_array.tolist(),
                    scores=scores.tolist(),
                    score_threshold=0.5,
                    nms_threshold=NMS_IOU
                )

                filtered_boxes = []
                for i in indices:
                    filtered_boxes.append(labels_axis_aligned[i])
                labels_axis_aligned = filtered_boxes

            # save labels_axis_aligned in a file in the format: animal_class center_x center_y width height
            if random_z_rotation != 0.0:
                labels_save_name = os.path.join(output_images_folder, f"{shot_name.split('.')[0]}_{str(random_z_rotation).replace('.', '_')}.txt")
            else:
                labels_save_name = os.path.join(output_images_folder, f"{shot_name.split('.')[0]}.txt")

            with open(os.path.join(output_labels_folder, labels_save_name), 'w') as f:
                for label in labels_axis_aligned:
                    label_str = to_yolo_format(label['axis_aligned_bounding_box'], settings.resolution.width, settings.resolution.height)
                    if label_str is not None:
                        f.write(f"{label['animal_class']} {label_str}\n")


            if SAVE_LABELED_IMAGES:
                for label in labels_axis_aligned:
                    cv2.polylines(render, label['axis_aligned_bounding_box'], True, LABEL_COLORS[label["animal_class"]], thickness=1)
                # cv2.polylines(render, poly_lines, True, LABEL_COLORS[class_id], thickness=1)
                # cv2.polylines(render, axis_aligned_bounding_box, True, (255, 255, 0), thickness=1)

                logging.info('  Saving labeled image')
                labeled_output_file = os.path.join(output_images_folder, f"labeled_{shot_name}")
                cv2.imwrite(labeled_output_file, render)
            # endregion

        if not project_orthogonal:
            release_all(prev_shots, add_shots)
        release_all(shot)

            # for testing
            # shots_processed += 1
            # if shots_processed >= exit_after_x_shots:
            #     logging.info("done with shots", shots_processed)
            #     break

    release_all(ctx, renderer, shots)



# Export images for a split (requires the dataset to be in the correct format)
def project_images_for_split(split: str, DATASET_DIR: str, OUTPUT_DIR: str, ORTHO_WIDTH: int, ORTHO_HEIGHT: int, RENDER_WIDTH: int, RENDER_HEIGHT: int, CAMERA_DISTANCE: int,
                             INITIAL_SKIP: int, ADD_BACKGROUND: bool, FOVY: float, ASPECT_RATIO: float, SAVE_LABELED_IMAGES: bool, INPUT_WIDTH:int, INPUT_HEIGHT:int,
                             project_orthogonal:bool, ADDITIONAL_ROTATIONS: int, ROTATION_LIMIT: float, merge_labels_in_alfs: bool, APPLY_NMS:bool, NMS_IOU: float, rng: np.random.Generator):
    logging.info(f"projecting images for split {split}")
    images_folder = os.path.join(DATASET_DIR, "images", split)
    labels_folder = os.path.join(DATASET_DIR, "labels", split)

    config_file = os.path.join(DATASET_DIR, f"export_{split}.json")

    if not os.path.exists(config_file):
        print(f"Could not find config file {config_file}")
        return

    with open(config_file, 'r') as f:
        config = json.load(f)

    # get all flights in the split
    flight_keys = config["metadata"]["included_flights"]

    flight_to_mission_mapping = {}
    for mission_id, flights in config["missions"].items():
        for flight_id, flight in flights["flights"].items():
            flight_to_mission_mapping[flight_id] = mission_id
            frame_to_labels_mapping = {}
            for label_id, label_states in flight["labels"].items():
                for label_state_frame_idx, label_state in label_states.items():
                    label_state_frame_idx_int = int(label_state_frame_idx)
                    if frame_to_labels_mapping.get(label_state_frame_idx_int) is None:
                        frame_to_labels_mapping[label_state_frame_idx_int] = []
                    frame_to_labels_mapping[label_state_frame_idx_int].append(label_state)
            flight["frame_to_labels_mapping"] = frame_to_labels_mapping

    config["flight_to_mission_mapping"] = flight_to_mission_mapping

    label_backward_mapping = {}
    for label_id, wikidata_id in config["metadata"]["labels"].items():
        label_backward_mapping[wikidata_id] = label_id
    config["metadata"]["label_backward_mapping"] = label_backward_mapping


    for flight_key in flight_keys:
        dem_file = os.path.join(DATASET_DIR, "correction_data", f"{flight_key}_dem.glb")
        poses_file = os.path.join(DATASET_DIR, "correction_data", f"{flight_key}_matched_poses.json")

        correction_matrix_file = os.path.join(DATASET_DIR, "correction_data", f"{flight_key}_correction.json")
        
        project_images_for_flight(flight_key, split, images_folder, labels_folder, dem_file, poses_file, correction_matrix_file,
                                  OUTPUT_DIR, ORTHO_WIDTH, ORTHO_HEIGHT, RENDER_WIDTH, RENDER_HEIGHT, CAMERA_DISTANCE,
                                  INITIAL_SKIP, ADD_BACKGROUND, FOVY, ASPECT_RATIO, SAVE_LABELED_IMAGES, INPUT_WIDTH, INPUT_HEIGHT,
                                  config, project_orthogonal, ADDITIONAL_ROTATIONS, ROTATION_LIMIT, merge_labels_in_alfs, APPLY_NMS, NMS_IOU, rng)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
    logging.info("starting orthografic projection")
    DEFAULT_DATASET_DIR = r"C:\Users\P41743\Desktop\bambi_dataset"  # "dataset_dir"
    DEFAULT_OUTPUT_DIR = r"C:\Users\P41743\Desktop\bambi_dataset\projected"

    # Argument parser can be removed since we're using environment variables
    SPLITS = os.environ.get("SPLITS", "train,val,test").split(",")
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    DATASET_DIR = os.environ.get("INPUT_DIR", DEFAULT_DATASET_DIR)
    ORTHO_WIDTH = int(os.environ.get("ORTHO_WIDTH", 70))
    ORTHO_HEIGHT = int(os.environ.get("ORTHO_HEIGHT", 70))
    INPUT_WIDTH = int(os.environ.get("INPUT_WIDTH", 1024))
    INPUT_HEIGHT = int(os.environ.get("INPUT_HEIGHT", 1024))
    RENDER_WIDTH = int(os.environ.get("RENDER_WIDTH", 2048))
    RENDER_HEIGHT = int(os.environ.get("RENDER_HEIGHT", 2048))
    CAMERA_DISTANCE = float(os.environ.get("CAMERA_DISTANCE", 10.0))
    INITIAL_SKIP = int(os.environ.get("INITIAL_SKIP", 0))
    ADD_BACKGROUND = bool(int(os.environ.get("ADD_BACKGROUND", 1)))
    FOVY = float(os.environ.get("FOVY", 50.0))
    ASPECT_RATIO = float(os.environ.get("ASPECT_RATIO", 1.0))
    SAVE_LABELED_IMAGES = bool(int(os.environ.get("SAVE_LABELED_IMAGES", 0)))
    project_orthogonal= bool(int(os.environ.get("PROJECT_ORTHOGONAL", 1)))
    merge_labels_in_alfs = bool(int(os.environ.get("MERGE_LABELS_IN_ALFS", 1)))
    ADDITIONAL_ROTATIONS = int(os.environ.get("ADDITIONAL_ROTATIONS", 0))
    ROTATION_LIMIT = float(os.environ.get("ROTATION_LIMIT", 2*np.pi))
    ROTATION_SEED = int(os.environ.get("ROTATION_SEED", -1))
    ROTATION_LIMIT_RADIAN = bool(int(os.environ.get("ROTATION_LIMIT_RADIAN", 1)))
    APPLY_NMS = bool(int(os.environ.get("APPLY_NMS", 1)))
    NMS_IOU = float(os.environ.get("NMS_IOU", 0.75))

    if not ROTATION_LIMIT_RADIAN:
        ROTATION_LIMIT = np.deg2rad(ROTATION_LIMIT)

    if ROTATION_SEED < 0:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(ROTATION_SEED)

    logging.info(f"Using configuration: {locals()}")

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
        logging.info(f"starting orthografic projection with split {split}")
        project_images_for_split(split, DATASET_DIR, OUTPUT_DIR, ORTHO_WIDTH, ORTHO_HEIGHT, RENDER_WIDTH, RENDER_HEIGHT, CAMERA_DISTANCE, INITIAL_SKIP, ADD_BACKGROUND, FOVY, ASPECT_RATIO, SAVE_LABELED_IMAGES, INPUT_WIDTH, INPUT_HEIGHT, project_orthogonal, ADDITIONAL_ROTATIONS, ROTATION_LIMIT, merge_labels_in_alfs, APPLY_NMS, NMS_IOU, rng)
        logging.info("done with split", split)

    logging.info("done")

