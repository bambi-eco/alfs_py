"""
ALFS (Airborne Light Field Sampling) Renderer

This script renders ALFS projections from drone imagery with separately configured input folders:
- images_dir: {split}/{flight_id}_{frame_idx}.jpg - key frames
- neighbors_dir: {split}/{flight_id}_{frame_idx}.jpg - neighboring frames
- corrections_dir: flat folder with DEM, poses, corrections, and masks per flight
  - {flight_id}_dem.glb
  - {flight_id}_matched_poses.json
  - {flight_id}_correction.json
  - {flight_id}_mask_t.png
- mot_dir: {split}/{flight_id}_gt.txt - MOT format labels

Output follows the same {split} structure with rendered images and projected labels.
"""

import traceback
import os
import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Set, Any
from pathlib import Path
import cv2
from alfspy.core.torchgl import TorchContext
import numpy as np
import logging
import sys
from collections import defaultdict

from alfspy.core.rendering import Resolution, Camera, CtxShot, RenderResultMode, TextureData
from pyrr import Quaternion, Vector3
from trimesh import Trimesh

from alfspy.core.convert.convert import pixel_to_world_coord, world_to_pixel_coord
from alfspy.core.geo.transform import Transform
from alfspy.core.rendering.renderer import Renderer
from alfspy.core.util.collections.cyclic import CyclicList
from alfspy.core.util.geo import get_aabb
from alfspy.core.util.pyrrs import quaternion_from_drone_pose
from alfspy.render.data import BaseSettings, CameraPositioningMode
from alfspy.render.render import make_camera, make_torch_context, make_shot_loader, process_render_data, read_gltf, release_all


LABEL_COLORS = CyclicList((  # BGR
    (102, 0, 255),
    (255, 102, 0),
    (0, 255, 102),
    (255, 0, 102),
    (255, 102, 0),
    (0, 102, 255),
))


@dataclass
class ALFSConfig:
    """Configuration for ALFS rendering."""
    # Input folders (each defined separately)
    images_dir: str              # images/{split}/{flight_id}_{frame_idx}.jpg
    neighbors_dir: str           # images_neighbors/{split}/{flight_id}_{frame_idx}.jpg
    corrections_dir: str         # flat folder with {flight_id}_dem.glb, etc.
    mot_dir: str                 # MOT/{split}/{flight_id}_gt.txt

    # Output folder
    output_dir: str

    splits: List[str] = field(default_factory=lambda: ["train", "val"])

    # Neighbor frame settings
    neighbor_range_before: int = 10  # Number of frames before current
    neighbor_range_after: int = 10   # Number of frames after current
    frame_stride: int = 1            # Use every n-th frame as neighbor

    # Rendering settings
    input_width: int = 1024
    input_height: int = 1024
    render_width: int = 2048
    render_height: int = 2048
    ortho_width: int = 70
    ortho_height: int = 70
    camera_distance: float = 10.0
    fovy: float = 50.0
    aspect_ratio: float = 1.0
    alpha_threshold: float = 2.0

    # Output settings
    save_labeled_images: bool = False
    merge_labels_in_alfs: bool = True
    apply_nms: bool = False
    nms_iou: float = 0.9

    # MOT indexing
    mot_one_indexed: bool = True  # If True, subtract 1 from MOT frame indices to match 0-indexed frames


@dataclass
class MOTLabel:
    """Represents a single MOT format label."""
    frame_idx: int
    track_id: int
    bb_left: float
    bb_top: float
    bb_width: float
    bb_height: float
    confidence: float = 1.0
    class_id: int = 0
    visibility: float = 1.0


def parse_mot_file(mot_file: str, mot_one_indexed: bool = True) -> Dict[int, List[MOTLabel]]:
    """
    Parse a MOT format ground truth file.

    MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>

    Args:
        mot_file: Path to the MOT file
        mot_one_indexed: If True, subtract 1 from frame indices to convert from 1-indexed MOT to 0-indexed frames

    Returns: Dictionary mapping frame_idx to list of labels
    """
    frame_labels = defaultdict(list)

    if not os.path.exists(mot_file):
        logging.warning(f"MOT file not found: {mot_file}")
        return frame_labels

    with open(mot_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue

            try:
                frame_idx = int(parts[0])
                if mot_one_indexed:
                    frame_idx -= 1  # Convert from 1-indexed MOT to 0-indexed frames
                track_id = int(parts[1])
                bb_left = float(parts[2])
                bb_top = float(parts[3])
                bb_width = float(parts[4])
                bb_height = float(parts[5])
                confidence = float(parts[6]) if len(parts) > 6 else 1.0
                class_id = int(parts[7]) if len(parts) > 7 else 0
                visibility = float(parts[8]) if len(parts) > 8 else 1.0

                label = MOTLabel(
                    frame_idx=frame_idx,
                    track_id=track_id,
                    bb_left=bb_left,
                    bb_top=bb_top,
                    bb_width=bb_width,
                    bb_height=bb_height,
                    confidence=confidence,
                    class_id=class_id,
                    visibility=visibility
                )
                frame_labels[frame_idx].append(label)
            except (ValueError, IndexError) as e:
                logging.warning(f"Failed to parse MOT line: {line.strip()}, error: {e}")
                continue

    return frame_labels


def mot_label_to_polygon(label: MOTLabel) -> List[float]:
    """Convert MOT bounding box to polygon coordinates [x1, y1, x2, y1, x2, y2, x1, y2]."""
    x1 = label.bb_left
    y1 = label.bb_top
    x2 = label.bb_left + label.bb_width
    y2 = label.bb_top + label.bb_height
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def parse_image_filename(filename: str) -> Tuple[int, int]:
    """
    Parse image filename to extract flight_id and frame_idx.
    Expected format: {flight_id}_{frame_idx}.jpg
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    if len(parts) != 2:
        raise ValueError(f"Invalid filename format: {filename}, expected {{flight_id}}_{{frame_idx}}.jpg")
    return int(parts[0]), int(parts[1])


def get_frame_correction(corrections_data: dict, frame_idx: int, central_frame_idx: Optional[int] = None) -> Tuple[dict, dict]:
    """
    Get the correction for a specific frame from the corrections data.

    Returns:
        Tuple of (correction_dict, correction_meta) where correction_meta describes
        the source of the correction (matched range, fallback to central, or default).
    """
    central_correction = None
    central_correction_meta = None
    for correction in corrections_data.get("corrections", []):
        if correction["start frame"] <= frame_idx <= correction["end frame"]:
            meta = {
                "source": "matched",
                "start_frame": correction["start frame"],
                "end_frame": correction["end frame"],
            }
            return correction, meta
        if central_frame_idx is not None and correction["start frame"] <= central_frame_idx <= correction["end frame"]:
            central_correction = correction
            central_correction_meta = {
                "source": "central_fallback",
                "start_frame": correction["start frame"],
                "end_frame": correction["end frame"],
                "central_frame_idx": central_frame_idx,
            }

    if central_correction is not None:
        return central_correction, central_correction_meta

    default = corrections_data.get("default", {
        "translation": {"x": 0, "y": 0, "z": 0},
        "rotation": {"x": 0, "y": 0, "z": 0}
    })
    return default, {"source": "default"}


def get_shots_for_files(
    image_files: List[str],
    images_folder: str,
    ctx: TorchContext,
    corrections_data: Dict[str, any],
    matched_poses: dict,
    fovy: float = 60.0,
    central_frame_idx: Optional[int] = None
) -> Tuple[List[CtxShot], List[str], List[Vector3], List[Transform], List[Vector3], List[dict]]:
    """Create shots for a list of image files."""
    shots = []
    shot_names = []
    shots_rotation_eulers = []
    corrections = []
    correction_eulers = []
    correction_metas = []

    for img_file in image_files:
        if not (img_file.endswith(".png") or img_file.endswith(".jpg")):
            continue

        _, frame_idx = parse_image_filename(img_file)
        frame_pose = matched_poses["images"][frame_idx]

        camera_position = Vector3(frame_pose["location"])
        camera_rotation = frame_pose["rotation"]
        camera_rotation = [val % 360.0 for val in camera_rotation]

        rot_len = len(camera_rotation)
        if rot_len == 3:
            camera_rotation = quaternion_from_drone_pose(camera_rotation)
        elif rot_len == 4:
            camera_rotation = Quaternion(camera_rotation)
        else:
            raise ValueError(f'Invalid rotation format of length {rot_len}: {camera_rotation}')

        fov = frame_pose.get("fovy", [fovy])
        if fov is None:
            fov = fovy
        elif isinstance(fov, list):
            fov = fov[0] if fov else fovy

        frame_correction, correction_meta = get_frame_correction(corrections_data, frame_idx, central_frame_idx)
        translation = frame_correction.get('translation', {'x': 0, 'y': 0, 'z': 0})
        cor_translation = Vector3([translation['x'], translation['y'], translation['z']], dtype='f4')

        rotation = frame_correction.get('rotation', {'x': 0, 'y': 0, 'z': 0})
        cor_rotation_eulers = Vector3([rotation['x'], rotation['y'], rotation['z']], dtype='f4')
        cor_quat = Quaternion.from_eulers(cor_rotation_eulers)
        correction = Transform(cor_translation, cor_quat)

        # Enrich correction metadata with the actual values applied
        correction_meta["translation"] = {"x": float(translation['x']), "y": float(translation['y']), "z": float(translation['z'])}
        correction_meta["rotation"] = {"x": float(rotation['x']), "y": float(rotation['y']), "z": float(rotation['z'])}

        shots.append(CtxShot(ctx, os.path.join(images_folder, img_file), camera_position, camera_rotation, fov, 1, correction, lazy=False))
        shots_rotation_eulers.append(cor_rotation_eulers)
        shot_names.append(img_file)
        corrections.append(correction)
        correction_eulers.append(cor_rotation_eulers)
        correction_metas.append({"frame_idx": frame_idx, "file": img_file, **correction_meta})

    return shots, shot_names, shots_rotation_eulers, corrections, correction_eulers, correction_metas


def get_axis_aligned_bounding_box(frame_label) -> List[np.ndarray]:
    """Convert projected polygon to axis-aligned bounding box."""
    coords = frame_label[0]
    x_coords = coords[:, 0, 0]
    y_coords = coords[:, 0, 1]

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    box_coords = np.array([
        [[x_min, y_min]],
        [[x_max, y_min]],
        [[x_max, y_max]],
        [[x_min, y_max]]
    ])

    return [box_coords]


def to_yolo_format(axis_aligned_bounding_box: List[np.ndarray], img_width: int, img_height: int) -> Optional[str]:
    """Convert axis-aligned bounding box to YOLO format string."""
    x_min = axis_aligned_bounding_box[0][0][0][0]
    y_min = axis_aligned_bounding_box[0][0][0][1]
    x_max = axis_aligned_bounding_box[0][2][0][0]
    y_max = axis_aligned_bounding_box[0][2][0][1]

    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    # Check if label is outside the frame
    if x_center + width/2 < 0 or x_center - width/2 > 1 or y_center + height/2 < 0 or y_center - height/2 > 1:
        return None

    return f"{x_center} {y_center} {width} {height}"


def get_camera_for_frame(matched_poses: dict, frame_idx: int, cor_rotation_eulers: Vector3, cor_translation: Vector3) -> Camera:
    """Create camera object for a specific frame."""
    pose_key = str(frame_idx) if str(frame_idx) in matched_poses["images"] else frame_idx
    cur_frame_data = matched_poses['images'][pose_key]
    fovy = cur_frame_data['fovy'][0] if isinstance(cur_frame_data['fovy'], list) else cur_frame_data['fovy']

    position = Vector3(cur_frame_data['location'])
    rotation_eulers = (Vector3([np.deg2rad(val % 360.0) for val in cur_frame_data['rotation']]) - cor_rotation_eulers) * -1
    position += cor_translation
    rotation = Quaternion.from_eulers(rotation_eulers)

    return Camera(fovy=fovy, aspect_ratio=1.0, position=position, rotation=rotation)


def project_label(label_coordinates: List[float], input_resolution: Resolution, tri_mesh: Trimesh,
                  camera: Camera, render_resolution: Resolution, single_shot_camera: Camera) -> List[np.ndarray]:
    """Project label coordinates from input image to rendered image."""
    pixel_xs = []
    pixel_ys = []
    for pixel_id, pixel in enumerate(label_coordinates):
        if pixel_id % 2 == 0:
            pixel_xs.append(int(float(pixel)))
        else:
            pixel_ys.append(int(float(pixel)))

    w_poses = pixel_to_world_coord(pixel_xs, pixel_ys, input_resolution.width, input_resolution.height,
                                   tri_mesh, camera, include_misses=False)
    np_poses = world_to_pixel_coord(w_poses, render_resolution.width, render_resolution.height, single_shot_camera)

    return [np.array(np_poses).T.reshape((-1, 1, 2))]


def discover_flights(images_dir: str, split: str) -> Dict[int, List[str]]:
    """
    Discover all flights and their frames in the images folder.
    Returns: Dictionary mapping flight_id to list of image filenames.
    """
    split_folder = os.path.join(images_dir, split)
    if not os.path.exists(split_folder):
        logging.warning(f"Split folder not found: {split_folder}")
        return {}

    flights = defaultdict(list)
    for filename in os.listdir(split_folder):
        if not (filename.endswith('.jpg') or filename.endswith('.png')):
            continue
        try:
            flight_id, frame_idx = parse_image_filename(filename)
            flights[flight_id].append(filename)
        except ValueError as e:
            logging.warning(f"Skipping file with invalid name: {filename}")
            continue

    # Sort frames within each flight
    for flight_id in flights:
        flights[flight_id].sort(key=lambda f: parse_image_filename(f)[1])

    return dict(flights)


def discover_neighbors(neighbors_dir: str, split: str, flight_id: int) -> Dict[int, str]:
    """
    Discover all neighbor frames for a flight.
    Returns: Dictionary mapping frame_idx to filename.
    """
    split_folder = os.path.join(neighbors_dir, split)
    if not os.path.exists(split_folder):
        return {}

    neighbors = {}
    for filename in os.listdir(split_folder):
        if not (filename.endswith('.jpg') or filename.endswith('.png')):
            continue
        try:
            fid, frame_idx = parse_image_filename(filename)
            if fid == flight_id:
                neighbors[frame_idx] = filename
        except ValueError:
            continue

    return neighbors


def render_alfs_for_flight(
    flight_id: int,
    split: str,
    config: ALFSConfig,
    flight_frames: List[str],
    mot_labels: Dict[int, List[MOTLabel]]
):
    """Render ALFS for all frames in a flight."""
    logging.info(f"Processing flight {flight_id} with {len(flight_frames)} frames")

    # Build paths using separate config folders
    images_folder = os.path.join(config.images_dir, split)
    neighbors_folder = os.path.join(config.neighbors_dir, split)

    dem_file = os.path.join(config.corrections_dir, f"{flight_id}_dem.glb")
    poses_file = os.path.join(config.corrections_dir, f"{flight_id}_matched_poses.json")
    correction_file = os.path.join(config.corrections_dir, f"{flight_id}_correction.json")
    mask_file = os.path.join(config.corrections_dir, f"{flight_id}_mask_t.png")

    # Verify required files exist
    if not os.path.exists(dem_file):
        logging.error(f"DEM file not found: {dem_file}")
        return
    if not os.path.exists(poses_file):
        logging.error(f"Poses file not found: {poses_file}")
        return

    output_images_folder = os.path.join(config.output_dir, 'images', split)
    output_labels_folder = os.path.join(config.output_dir, 'labels', split)
    output_metadata_folder = os.path.join(config.output_dir, 'metadata', split)

    # Check if all output files already exist - skip DEM loading if so
    frames_to_process = []
    for frame_file in flight_frames:
        output_filename = frame_file.replace('.png', '.jpg')
        output_label_path = os.path.join(output_labels_folder, Path(output_filename).stem + ".txt")
        output_meta_path = os.path.join(output_metadata_folder, Path(output_filename).stem + ".json")
        if not os.path.exists(output_label_path) or not os.path.exists(output_meta_path):
            frames_to_process.append(frame_file)

    if not frames_to_process:
        logging.info(f"All {len(flight_frames)} frames already processed for flight {flight_id}, skipping DEM loading")
        return

    logging.info(f"Flight {flight_id}: {len(frames_to_process)}/{len(flight_frames)} frames need processing")

    # Load mask if available
    mask = None
    if os.path.exists(mask_file):
        mask = TextureData(CtxShot._cvt_img(cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)))
    else:
        logging.warning(f"Mask file not found: {mask_file}")

    # Load corrections
    corrections_data = {"corrections": [], "default": {
        "translation": {"x": 0, "y": 0, "z": 0},
        "rotation": {"x": 0, "y": 0, "z": 0}
    }}

    if os.path.exists(correction_file):
        with open(correction_file, 'r') as f:
            corrections_data["default"] = json.load(f)

    # Load poses
    with open(poses_file, 'r') as f:
        matched_poses = json.load(f)

    # Setup resolutions
    input_resolution = Resolution(config.input_width, config.input_height)
    render_resolution = Resolution(config.render_width, config.render_height)

    # Load mesh
    mesh_data, texture_data = read_gltf(dem_file)
    tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
    mesh_data, texture_data = process_render_data(mesh_data, texture_data)

    # Create TorchContext and get all shots for key frames
    ctx = make_torch_context()
    shots, shot_names, shots_rotation_eulers, corrections, correction_eulers, central_correction_metas = get_shots_for_files(
        flight_frames, images_folder, ctx, corrections_data, matched_poses, config.fovy
    )

    mesh_aabb = get_aabb(mesh_data.vertices)

    # Discover all neighbors for this flight
    all_neighbors = discover_neighbors(config.neighbors_dir, split, flight_id)

    renderer = None

    for shot, shot_name, shot_rotation_eulers, correction, cor_rotation_eulers, central_cor_meta in zip(
        shots, shot_names, shots_rotation_eulers, corrections, correction_eulers, central_correction_metas
    ):
        prev_shots = None
        add_shots = None

        try:
            _, frame_idx = parse_image_filename(shot_name)
            cor_translation = correction.position

            output_filename = shot_name.replace('.png', '.jpg')
            output_image_path = os.path.join(output_images_folder, output_filename)
            output_label_path = os.path.join(output_labels_folder, Path(output_filename).stem + ".txt")

            output_meta_path = os.path.join(output_metadata_folder, Path(output_filename).stem + ".json")

            # Skip if already processed (image, label, and metadata all exist)
            if os.path.exists(output_label_path) and os.path.exists(output_meta_path):
                logging.info(f"Already exists, skipping: {output_label_path}")
                continue

            # Gather neighbor frames based on range and stride
            previous_frames = []
            previous_frame_ids = []
            additional_frames = []
            additional_frame_ids = []

            for neighbor_idx, neighbor_file in all_neighbors.items():
                # Previous frames (before current)
                if frame_idx > neighbor_idx >= frame_idx - config.neighbor_range_before:
                    if (frame_idx - neighbor_idx) % config.frame_stride == 0:
                        previous_frames.append(neighbor_file)
                        previous_frame_ids.append(neighbor_idx)

                # Additional frames (after current)
                if frame_idx < neighbor_idx <= frame_idx + config.neighbor_range_after:
                    if (neighbor_idx - frame_idx) % config.frame_stride == 0:
                        additional_frames.append(neighbor_file)
                        additional_frame_ids.append(neighbor_idx)

            # Sort frames
            if previous_frame_ids:
                combined = sorted(zip(previous_frame_ids, previous_frames))
                previous_frame_ids, previous_frames = zip(*combined)
                previous_frame_ids = list(previous_frame_ids)
                previous_frames = list(previous_frames)
            else:
                previous_frame_ids, previous_frames = [], []

            if additional_frame_ids:
                combined = sorted(zip(additional_frame_ids, additional_frames))
                additional_frame_ids, additional_frames = zip(*combined)
                additional_frame_ids = list(additional_frame_ids)
                additional_frames = list(additional_frames)
            else:
                additional_frame_ids, additional_frames = [], []

            logging.info(f"Frame {frame_idx}: {len(previous_frames)} prev, {len(additional_frames)} next neighbors")

            # Load neighbor shots
            prev_shots, _, _, _, _, prev_correction_metas = get_shots_for_files(
                previous_frames, neighbors_folder, ctx, corrections_data, matched_poses,
                config.fovy, central_frame_idx=frame_idx
            )
            add_shots, _, _, _, _, add_correction_metas = get_shots_for_files(
                additional_frames, neighbors_folder, ctx, corrections_data, matched_poses,
                config.fovy, central_frame_idx=frame_idx
            )

            # Setup rendering
            settings = BaseSettings(
                count=len(flight_frames), initial_skip=0, add_background=False,
                camera_dist=config.camera_distance,
                camera_position_mode=CameraPositioningMode.FirstShot, fovy=config.fovy,
                aspect_ratio=config.aspect_ratio, orthogonal=True,
                ortho_size=(config.ortho_width, config.ortho_height),
                correction=correction, resolution=render_resolution
            )

            # Create camera and renderer
            single_shot_camera = make_camera(
                mesh_aabb, [shot], settings,
                rotation=Quaternion.from_eulers([
                    shot_rotation_eulers[0] - cor_rotation_eulers[0],
                    shot_rotation_eulers[1] - cor_rotation_eulers[1],
                    shot_rotation_eulers[2] - cor_rotation_eulers[2]
                ])
            )

            renderer = Renderer(settings.resolution, ctx, single_shot_camera, mesh_data, texture_data)

            # Render ALFS (skip if image/labels already exist, only metadata missing)
            label_exists = os.path.exists(output_label_path)
            if not label_exists:
                all_shots = prev_shots + [shot] + add_shots
                shot_loader = make_shot_loader(all_shots)
                renderer.render_integral(
                    shot_loader,
                    mask=mask,
                    save=True,
                    release_shots=False,
                    save_name=output_image_path,
                    alpha_threshold=config.alpha_threshold
                )

                logging.info(f"Saved image: {output_image_path}")
            else:
                logging.info(f"Label exists, skipping render for metadata: {output_label_path}")
            renderer.release()

            # Project and write labels (skip if label file already exists)
            num_labels_written = 0
            if not label_exists:
                labels_axis_aligned = []

                if config.merge_labels_in_alfs:
                    projected_labels = {}
                    all_frame_ids = previous_frame_ids + [frame_idx] + additional_frame_ids

                    for label_frame_idx in all_frame_ids:
                        frame_label_list = mot_labels.get(label_frame_idx, [])
                        if not frame_label_list:
                            continue

                        camera = get_camera_for_frame(matched_poses, label_frame_idx, cor_rotation_eulers, cor_translation)

                        for label in frame_label_list:
                            poly_coords = mot_label_to_polygon(label)
                            poly_lines = project_label(
                                poly_coords, input_resolution, tri_mesh, camera,
                                render_resolution, single_shot_camera
                            )
                            axis_aligned_bounding_box = get_axis_aligned_bounding_box(poly_lines)

                            track_key = label.track_id
                            if track_key not in projected_labels:
                                projected_labels[track_key] = []
                            projected_labels[track_key].append({
                                'animal_class': label.class_id,
                                'axis_aligned_bounding_box': axis_aligned_bounding_box
                            })

                    # Merge bounding boxes per track
                    for track_id, label_states in projected_labels.items():
                        animal_class = label_states[0]["animal_class"]

                        all_bboxes = [
                            np.array(ls["axis_aligned_bounding_box"]).reshape(-1, 2).tolist()
                            for ls in label_states
                        ]
                        bboxes = np.array(all_bboxes).reshape(-1, 2)
                        x_min, y_min = bboxes.min(axis=0)
                        x_max, y_max = bboxes.max(axis=0)

                        enclosing_bbox = np.array([
                            [[x_min, y_min]],
                            [[x_max, y_min]],
                            [[x_max, y_max]],
                            [[x_min, y_max]]
                        ])

                        labels_axis_aligned.append({
                            'animal_class': animal_class,
                            'axis_aligned_bounding_box': [enclosing_bbox]
                        })
                else:
                    # Only project labels from central frame
                    central_labels = mot_labels.get(frame_idx, [])
                    camera = get_camera_for_frame(matched_poses, frame_idx, cor_rotation_eulers, cor_translation)

                    for label in central_labels:
                        poly_coords = mot_label_to_polygon(label)
                        poly_lines = project_label(
                            poly_coords, input_resolution, tri_mesh, camera,
                            render_resolution, single_shot_camera
                        )
                        axis_aligned_bounding_box = get_axis_aligned_bounding_box(poly_lines)
                        labels_axis_aligned.append({
                            'animal_class': label.class_id,
                            'axis_aligned_bounding_box': axis_aligned_bounding_box
                        })

                # Apply NMS if configured
                if config.apply_nms and labels_axis_aligned:
                    boxes = []
                    for box_dict in labels_axis_aligned:
                        box = box_dict['axis_aligned_bounding_box']
                        x_coords = [pt[0][0] for pt in box[0]]
                        y_coords = [pt[0][1] for pt in box[0]]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        boxes.append([x_min, y_min, x_max, y_max])

                    boxes_array = np.array(boxes, dtype=float)
                    scores = np.array([1.0] * len(boxes))

                    indices = cv2.dnn.NMSBoxes(
                        bboxes=boxes_array.tolist(),
                        scores=scores.tolist(),
                        score_threshold=0.5,
                        nms_threshold=config.nms_iou
                    )

                    labels_axis_aligned = [labels_axis_aligned[i] for i in indices]

                # Write labels
                with open(output_label_path, 'w') as f:
                    for label in labels_axis_aligned:
                        label_str = to_yolo_format(
                            label['axis_aligned_bounding_box'],
                            settings.resolution.width,
                            settings.resolution.height
                        )
                        if label_str is not None:
                            f.write(f"{label['animal_class']} {label_str}\n")
                            num_labels_written += 1

                logging.info(f"Saved labels: {output_label_path}")

                # Save labeled image if configured
                if config.save_labeled_images:
                    render = cv2.imread(output_image_path)
                    for label in labels_axis_aligned:
                        cv2.polylines(
                            render, label['axis_aligned_bounding_box'], True,
                            LABEL_COLORS[label["animal_class"]], thickness=2
                        )
                    labeled_path = os.path.join(output_images_folder, f"labeled_{Path(output_filename).stem}.jpg")
                    cv2.imwrite(labeled_path, render)
                    logging.info(f"Saved labeled image: {labeled_path}")
            else:
                # Count labels from existing file
                with open(output_label_path, 'r') as f:
                    num_labels_written = sum(1 for line in f if line.strip())

            # Write per-frame metadata
            metadata_folder = os.path.join(config.output_dir, 'metadata', split)
            os.makedirs(metadata_folder, exist_ok=True)
            frame_meta = {
                "flight_id": flight_id,
                "split": split,
                "output_image": output_filename,
                "central_frame": {
                    "frame_idx": frame_idx,
                    "file": shot_name,
                    "correction": central_cor_meta,
                },
                "neighbor_frames_before": list(prev_correction_metas),
                "neighbor_frames_after": list(add_correction_metas),
                "num_neighbors_before": len(previous_frames),
                "num_neighbors_after": len(additional_frames),
                "num_labels_written": num_labels_written,
            }
            metadata_path = os.path.join(metadata_folder, f"{Path(output_filename).stem}.json")
            with open(metadata_path, 'w') as f:
                json.dump(frame_meta, f, indent=2)
            logging.info(f"Saved frame metadata: {metadata_path}")

        except Exception as e:
            logging.error(f"Error processing frame {shot_name}: {traceback.format_exc()}")

        finally:
            if prev_shots is not None:
                release_all(prev_shots)
            if add_shots is not None:
                release_all(add_shots)

    release_all(ctx, renderer, shots)


def render_alfs_for_split(split: str, config: ALFSConfig):
    """Render ALFS for all flights in a split."""
    logging.info(f"Processing split: {split}")

    # Discover flights
    flights = discover_flights(config.images_dir, split)

    if not flights:
        logging.warning(f"No flights found for split: {split}")
        return

    logging.info(f"Found {len(flights)} flights in split {split}")

    for flight_id, flight_frames in flights.items():
        # Load MOT labels for this flight
        mot_file = os.path.join(config.mot_dir, split, f"{flight_id}_gt.txt")
        mot_labels = parse_mot_file(mot_file, mot_one_indexed=config.mot_one_indexed)

        logging.info(f"Flight {flight_id}: {len(flight_frames)} frames, {sum(len(v) for v in mot_labels.values())} labels")

        render_alfs_for_flight(flight_id, split, config, flight_frames, mot_labels)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Configuration from environment variables or defaults
    config = ALFSConfig(
        # Input folders (each defined separately)
        images_dir=os.environ.get("IMAGES_DIR", r"Z:\Hugo\images"),
        neighbors_dir=os.environ.get("NEIGHBORS_DIR", r"Z:\Hugo\images_neighbors"),
        corrections_dir=os.environ.get("CORRECTIONS_DIR", r"Z:\correction_data2"),
        mot_dir=os.environ.get("MOT_DIR", r"Z:\Hugo\mot_with_neighbors"),

        # Output folder
        output_dir=os.environ.get("OUTPUT_DIR", r"Z:\Hugo\alfs"),

        splits=os.environ.get("SPLITS", "train").split(","), # train,val,test
        neighbor_range_before=int(os.environ.get("NEIGHBOR_RANGE_BEFORE", 90)),
        neighbor_range_after=int(os.environ.get("NEIGHBOR_RANGE_AFTER", 90)),
        frame_stride=int(os.environ.get("FRAME_STRIDE", 3)),
        input_width=int(os.environ.get("INPUT_WIDTH", 1024)),
        input_height=int(os.environ.get("INPUT_HEIGHT", 1024)),
        render_width=int(os.environ.get("RENDER_WIDTH", 2048)),
        render_height=int(os.environ.get("RENDER_HEIGHT", 2048)),
        ortho_width=int(os.environ.get("ORTHO_WIDTH", 70)),
        ortho_height=int(os.environ.get("ORTHO_HEIGHT", 70)),
        camera_distance=float(os.environ.get("CAMERA_DISTANCE", 10.0)),
        fovy=float(os.environ.get("FOVY", 50.0)),
        aspect_ratio=float(os.environ.get("ASPECT_RATIO", 1.0)),
        alpha_threshold=float(os.environ.get("ALPHA_THRESHOLD", 2.0)),
        save_labeled_images=bool(int(os.environ.get("SAVE_LABELED_IMAGES", 0))),
        merge_labels_in_alfs=bool(int(os.environ.get("MERGE_LABELS_IN_ALFS", 0))),
        apply_nms=bool(int(os.environ.get("APPLY_NMS", 0))),
        nms_iou=float(os.environ.get("NMS_IOU", 0.9)),
        mot_one_indexed=bool(int(os.environ.get("MOT_ONE_INDEXED", 1)))
    )

    logging.info(f"ALFS Renderer Configuration:")
    logging.info(f"  Images dir: {config.images_dir}")
    logging.info(f"  Neighbors dir: {config.neighbors_dir}")
    logging.info(f"  Corrections dir: {config.corrections_dir}")
    logging.info(f"  MOT dir: {config.mot_dir}")
    logging.info(f"  Output dir: {config.output_dir}")
    logging.info(f"  Splits: {config.splits}")
    logging.info(f"  Neighbor range: -{config.neighbor_range_before} to +{config.neighbor_range_after}")
    logging.info(f"  Frame stride: {config.frame_stride}")
    logging.info(f"  MOT one-indexed: {config.mot_one_indexed}")

    # Create output directories
    for split in config.splits:
        os.makedirs(os.path.join(config.output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, 'labels', split), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, 'metadata', split), exist_ok=True)

    # Write global config metadata
    config_metadata = {
        "config": asdict(config),
        "timestamp": __import__('datetime').datetime.now().isoformat(),
    }
    config_metadata_path = os.path.join(config.output_dir, 'metadata', 'config.json')
    os.makedirs(os.path.join(config.output_dir, 'metadata'), exist_ok=True)
    with open(config_metadata_path, 'w') as f:
        json.dump(config_metadata, f, indent=2)
    logging.info(f"Saved global config metadata: {config_metadata_path}")

    # Process each split
    for split in config.splits:
        render_alfs_for_split(split, config)

    logging.info("ALFS rendering complete")


if __name__ == "__main__":
    main()