"""
Orthographic projection script for MOT17 format datasets.

Expected folder structure:
    base_folder/
        images/
            train/
            test/
            eval/
        mot/
            train/
            test/
            eval/

    correction_folder/
        {flight_id}_dem.glb
        {flight_id}_matched_poses.json
        {flight_id}_mask.png (or _mask_t.png / _mask_r.png)
        {flight_id}_correction.json

Image naming: {flight_id}_{frame_idx}.jpg (e.g., 50_80.jpg)
MOT file naming: {flight_id}_gt.txt (e.g., 50_gt.txt)

Output structure:
    output_folder/
        proj_images/
            train/
            test/
            eval/
        proj_mot/
            train/
            test/
            eval/
"""

import traceback
import os
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set
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
from alfspy.core.util.geo import get_aabb
from alfspy.core.util.pyrrs import quaternion_from_drone_pose
from alfspy.render.data import BaseSettings, CameraPositioningMode
from alfspy.render.render import make_camera, make_torch_context, make_shot_loader, process_render_data, read_gltf, \
    release_all


@dataclass
class FlightData:
    """Data container for a single flight's resources."""
    flight_id: str
    dem_file: str
    poses_file: str
    mask_file: str
    correction_file: str
    image_files: List[str] = field(default_factory=list)
    mot_file: Optional[str] = None


@dataclass
class MOTEntry:
    """Single MOT17 ground truth entry."""
    frame_id: int
    track_id: int
    bb_left: float
    bb_top: float
    bb_width: float
    bb_height: float
    conf: float
    class_id: int
    visibility: float


def parse_mot_file(mot_file_path: str) -> Dict[int, List[MOTEntry]]:
    """
    Parse MOT17 ground truth file.
    Returns a dict mapping frame_id to list of MOTEntry objects.

    MOT17 format: frame_id, track_id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
    """
    frame_to_entries: Dict[int, List[MOTEntry]] = defaultdict(list)

    if not os.path.exists(mot_file_path):
        logging.warning(f"MOT file not found: {mot_file_path}")
        return frame_to_entries

    with open(mot_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) < 9:
                logging.warning(f"Invalid MOT line (expected 9 values): {line}")
                continue

            try:
                entry = MOTEntry(
                    frame_id=int(parts[0]),
                    track_id=int(parts[1]),
                    bb_left=float(parts[2]),
                    bb_top=float(parts[3]),
                    bb_width=float(parts[4]),
                    bb_height=float(parts[5]),
                    conf=float(parts[6]),
                    class_id=int(parts[7]),
                    visibility=float(parts[8])
                )
                frame_to_entries[entry.frame_id].append(entry)
            except ValueError as e:
                logging.warning(f"Failed to parse MOT line: {line}, error: {e}")
                continue

    return frame_to_entries


def mot_entry_to_line(entry: MOTEntry) -> str:
    """Convert MOTEntry back to MOT17 format string."""
    return f"{entry.frame_id},{entry.track_id},{entry.bb_left:.2f},{entry.bb_top:.2f},{entry.bb_width:.2f},{entry.bb_height:.2f},{entry.conf},{entry.class_id},{entry.visibility:.2f}"


def collect_flight_ids_from_mot(mot_folder: str) -> Set[str]:
    """
    Collect flight IDs from MOT label files.
    Expects files named like: {flight_id}_gt.txt
    """
    flight_ids = set()

    if not os.path.exists(mot_folder):
        logging.warning(f"MOT folder not found: {mot_folder}")
        return flight_ids

    for filename in os.listdir(mot_folder):
        if filename.endswith('_gt.txt'):
            flight_id = filename.replace('_gt.txt', '')
            flight_ids.add(flight_id)

    return flight_ids


def collect_images_for_flight(images_folder: str, flight_id: str) -> List[str]:
    """
    Collect all image files for a specific flight.
    Expects files named like: {flight_id}_{frame_idx}.jpg
    """
    image_files = []

    if not os.path.exists(images_folder):
        logging.warning(f"Images folder not found: {images_folder}")
        return image_files

    pattern = re.compile(rf'^{re.escape(flight_id)}_\d+\.(jpg|png)$', re.IGNORECASE)

    for filename in os.listdir(images_folder):
        if pattern.match(filename):
            image_files.append(filename)

    # Sort by frame index
    def get_frame_idx(filename: str) -> int:
        parts = filename.split('_')
        return int(parts[1].split('.')[0])

    image_files.sort(key=get_frame_idx)
    return image_files


def check_correction_files(correction_folder: str, flight_id: str, mask_suffix: str = '') -> Tuple[
    bool, Dict[str, str]]:
    """
    Check if all required correction files exist for a flight.

    Returns:
        Tuple of (all_exist: bool, file_paths: dict)
    """
    files = {
        'dem': os.path.join(correction_folder, f"{flight_id}_dem.glb"),
        'poses': os.path.join(correction_folder, f"{flight_id}_matched_poses.json"),
        'correction': os.path.join(correction_folder, f"{flight_id}_correction.json"),
    }

    # Try different mask naming conventions
    mask_candidates = [
        os.path.join(correction_folder, f"{flight_id}_mask{mask_suffix}.png"),
        os.path.join(correction_folder, f"{flight_id}_mask.png"),
        os.path.join(correction_folder, f"{flight_id}_mask_t.png"),
        os.path.join(correction_folder, f"{flight_id}_mask_r.png"),
    ]

    mask_file = None
    for candidate in mask_candidates:
        if os.path.exists(candidate):
            mask_file = candidate
            break

    files['mask'] = mask_file

    all_exist = all(
        os.path.exists(f) if f else False
        for key, f in files.items()
        if key != 'mask'  # Mask is optional
    )

    if mask_file is None:
        logging.warning(f"No mask file found for flight {flight_id}")
        files['mask'] = None

    return all_exist, files


def get_frame_correction(corrections_data: dict, frame_idx: int, central_frame_idx: Optional[int] = None) -> dict:
    """
    Get the correction for a specific frame from the corrections data.
    """
    central_correction = None

    for correction in corrections_data.get("corrections", []):
        if correction["start frame"] <= frame_idx <= correction["end frame"]:
            return correction

        if central_frame_idx is not None and correction["start frame"] <= central_frame_idx <= correction["end frame"]:
            central_correction = correction

    if central_correction is not None:
        return central_correction

    return corrections_data.get("default", {
        'translation': {'x': 0, 'y': 0, 'z': 0},
        'rotation': {'x': 0, 'y': 0, 'z': 0}
    })


def get_shots_for_files(
        image_files: List[str],
        images_folder: str,
        ctx: TorchContext,
        corrections_data: Dict[str, any],
        matched_poses: dict,
        lazy: bool = False,
        fovy: float = 60.0,
        central_frame_idx: Optional[int] = None
) -> Tuple[List[CtxShot], List[str], List[Vector3], List[Transform], List[Vector3]]:
    """Get shots for a list of image files."""
    shots = []
    shot_names = []
    shots_rotation_eulers = []
    corrections = []
    correction_eulers = []

    for img_file in image_files:
        if not (img_file.endswith(".png") or img_file.endswith(".jpg")):
            continue

        # Extract frame index from filename: {flight_id}_{frame_idx}.jpg
        idx = int(img_file.split("_")[1].split(".")[0])

        if idx not in matched_poses.get("images", {}):
            logging.warning(f"Frame {idx} not found in poses file, skipping {img_file}")
            continue

        camera_position = Vector3(matched_poses["images"][idx]["location"])

        camera_rotation = matched_poses["images"][idx]["rotation"]
        camera_rotation = [val % 360.0 for val in camera_rotation]

        rot_len = len(camera_rotation)
        if rot_len == 3:
            camera_rotation = quaternion_from_drone_pose(camera_rotation)
        elif rot_len == 4:
            camera_rotation = Quaternion(camera_rotation)
        else:
            raise ValueError(f'Invalid rotation format of length {rot_len}: {camera_rotation}')

        fov = matched_poses["images"][idx].get("fovy", [fovy])
        if fov is None:
            fov = fovy
        elif isinstance(fov, list):
            fov = fov[0] if fov else fovy

        frame_correction = get_frame_correction(corrections_data, idx, central_frame_idx)
        translation = frame_correction.get('translation', {'x': 0, 'y': 0, 'z': 0})
        cor_translation = Vector3([translation['x'], translation['y'], translation['z']], dtype='f4')

        rotation = frame_correction.get('rotation', {'x': 0, 'y': 0, 'z': 0})
        cor_rotation_eulers = Vector3([rotation['x'], rotation['y'], rotation['z']], dtype='f4')
        cor_quat = Quaternion.from_eulers(cor_rotation_eulers)
        correction = Transform(cor_translation, cor_quat)

        shots.append(
            CtxShot(ctx, os.path.join(images_folder, img_file), camera_position, camera_rotation, fov, 1, correction,
                    lazy))
        shots_rotation_eulers.append(cor_rotation_eulers)
        shot_names.append(img_file)
        corrections.append(correction)
        correction_eulers.append(cor_rotation_eulers)

    return shots, shot_names, shots_rotation_eulers, corrections, correction_eulers


def get_camera_for_frame(matched_poses: dict, frame_idx: int, cor_rotation_eulers: Vector3,
                         cor_translation: Vector3) -> Camera:
    """Create camera for a specific frame."""
    cur_frame_data = matched_poses['images'][frame_idx]
    fovy = cur_frame_data['fovy'][0] if isinstance(cur_frame_data['fovy'], list) else cur_frame_data['fovy']

    position = Vector3(cur_frame_data['location'])
    rotation_eulers = (Vector3(
        [np.deg2rad(val % 360.0) for val in cur_frame_data['rotation']]) - cor_rotation_eulers) * -1

    position += cor_translation
    rotation = Quaternion.from_eulers(rotation_eulers)

    return Camera(fovy=fovy, aspect_ratio=1.0, position=position, rotation=rotation)


def project_bbox_corners(
        bbox: Tuple[float, float, float, float],
        input_resolution: Resolution,
        tri_mesh: Trimesh,
        camera: Camera,
        render_resolution: Resolution,
        single_shot_camera: Camera
) -> Optional[Tuple[float, float, float, float]]:
    """
    Project bounding box corners from input image to rendered image.

    Args:
        bbox: (left, top, width, height) in input image coordinates
        input_resolution: Resolution of input image
        tri_mesh: Mesh for raycasting
        camera: Camera for the input frame
        render_resolution: Resolution of rendered/output image
        single_shot_camera: Camera for the orthographic projection

    Returns:
        Projected bbox as (left, top, width, height) or None if projection fails
    """
    left, top, width, height = bbox

    # Get corner coordinates
    x1, y1 = left, top
    x2, y2 = left + width, top + height

    # All four corners
    pixel_xs = [x1, x2, x2, x1]
    pixel_ys = [y1, y1, y2, y2]

    # Project to world coordinates
    w_poses = pixel_to_world_coord(
        pixel_xs, pixel_ys,
        input_resolution.width, input_resolution.height,
        tri_mesh, camera,
        include_misses=False
    )

    if len(w_poses) == 0:
        return None

    # Project from world to output image coordinates
    np_poses = world_to_pixel_coord(
        w_poses,
        render_resolution.width, render_resolution.height,
        single_shot_camera
    )

    if len(np_poses) == 0:
        return None

    # Get axis-aligned bounding box of projected points
    coords = np.array(np_poses)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    # Clip to image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(render_resolution.width, x_max)
    y_max = min(render_resolution.height, y_max)

    new_width = x_max - x_min
    new_height = y_max - y_min

    if new_width <= 0 or new_height <= 0:
        return None

    return (x_min, y_min, new_width, new_height)


def process_flight(
        flight_data: FlightData,
        split: str,
        images_folder: str,
        mot_file_path: str,
        output_images_folder: str,
        output_mot_folder: str,
        config: 'ProjectionConfig',
        corrections_data: Optional[dict] = None
):
    """
    Process all frames for a single flight.
    Loads mesh once and processes all frames.
    """
    logging.info(f"Processing flight: {flight_data.flight_id}")

    if len(flight_data.image_files) == 0:
        logging.warning(f"No images found for flight {flight_data.flight_id}")
        return

    # Load poses
    with open(flight_data.poses_file, 'r') as f:
        matched_poses = json.load(f)

    # Load corrections
    if corrections_data is None:
        corrections_data = {"corrections": []}

    with open(flight_data.correction_file, 'r') as f:
        default_correction = json.load(f)
        corrections_data["default"] = default_correction

    # Load mask if available
    mask = None
    if flight_data.mask_file and os.path.exists(flight_data.mask_file):
        mask = TextureData(CtxShot._cvt_img(cv2.imread(flight_data.mask_file, cv2.IMREAD_UNCHANGED)))

    # Load mesh
    mesh_data, texture_data = read_gltf(flight_data.dem_file)
    tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
    mesh_data, texture_data = process_render_data(mesh_data, texture_data)

    # Create TorchContext
    ctx = make_torch_context()

    # Get all shots for this flight
    shots, shot_names, shots_rotation_eulers, corrections, correction_eulers = get_shots_for_files(
        flight_data.image_files, images_folder, ctx, corrections_data, matched_poses
    )

    if len(shots) == 0:
        logging.warning(f"No valid shots created for flight {flight_data.flight_id}")
        release_all(ctx)
        return

    mesh_aabb = get_aabb(mesh_data.vertices)

    # Parse MOT file
    mot_entries = parse_mot_file(mot_file_path)

    # Prepare output MOT entries
    output_mot_entries: List[MOTEntry] = []

    input_resolution = Resolution(config.input_width, config.input_height)
    render_resolution = Resolution(config.render_width, config.render_height)

    renderer = None

    try:
        for shot, shot_name, shot_rotation_eulers, correction, cor_rotation_eulers in zip(
                shots, shot_names, shots_rotation_eulers, corrections, correction_eulers
        ):
            frame_idx = int(shot_name.split("_")[1].split(".")[0])

            # Check if already processed
            output_image_path = os.path.join(output_images_folder, shot_name)
            if os.path.exists(output_image_path) and not config.overwrite:
                logging.info(f"Skipping existing: {shot_name}")
                continue

            try:
                settings = BaseSettings(
                    count=1,
                    initial_skip=0,
                    add_background=config.add_background,
                    camera_dist=config.camera_distance,
                    camera_position_mode=CameraPositioningMode.FirstShot,
                    fovy=config.fovy,
                    aspect_ratio=config.aspect_ratio,
                    orthogonal=True,
                    ortho_size=(config.ortho_width, config.ortho_height),
                    correction=correction,
                    resolution=render_resolution
                )

                cor_translation = correction.position

                # Create camera for this shot
                single_shot_camera = make_camera(
                    mesh_aabb, [shot], settings,
                    rotation=Quaternion.from_eulers([
                        (shot_rotation_eulers[0] - cor_rotation_eulers[0]),
                        (shot_rotation_eulers[1] - cor_rotation_eulers[1]),
                        (shot_rotation_eulers[2] - cor_rotation_eulers[2])
                    ])
                )

                # Create renderer
                renderer = Renderer(settings.resolution, ctx, single_shot_camera, mesh_data, texture_data)

                # Render image
                save_path = os.path.join(output_images_folder, shot_name)

                if config.project_orthogonal:
                    shot_loader = make_shot_loader([shot])
                    renderer.project_shots(
                        shot_loader,
                        RenderResultMode.ShotOnly,
                        mask=None,
                        integral=False,
                        save=True,
                        release_shots=False,
                        save_name_iter=iter([save_path])
                    )
                else:
                    shot_loader = make_shot_loader([shot])
                    renderer.render_integral(
                        shot_loader,
                        mask=mask,
                        save=True,
                        release_shots=False,
                        save_name=save_path,
                        alpha_threshold=config.alpha_threshold
                    )

                logging.info(f"Saved image: {save_path}")

                # Project labels for this frame
                frame_mot_entries = mot_entries.get(frame_idx, [])

                if frame_mot_entries:
                    camera = get_camera_for_frame(matched_poses, frame_idx, cor_rotation_eulers, cor_translation)

                    for entry in frame_mot_entries:
                        bbox = (entry.bb_left, entry.bb_top, entry.bb_width, entry.bb_height)

                        projected_bbox = project_bbox_corners(
                            bbox, input_resolution, tri_mesh, camera,
                            render_resolution, single_shot_camera
                        )

                        if projected_bbox is not None:
                            new_entry = MOTEntry(
                                frame_id=entry.frame_id,
                                track_id=entry.track_id,
                                bb_left=projected_bbox[0],
                                bb_top=projected_bbox[1],
                                bb_width=projected_bbox[2],
                                bb_height=projected_bbox[3],
                                conf=entry.conf,
                                class_id=entry.class_id,
                                visibility=entry.visibility
                            )
                            output_mot_entries.append(new_entry)
                        else:
                            logging.debug(f"Failed to project bbox for track {entry.track_id} in frame {frame_idx}")

                # Clean up renderer
                renderer.release()
                renderer = None

            except Exception as e:
                logging.error(f"Error processing frame {shot_name}: {traceback.format_exc()}")
                if renderer is not None:
                    renderer.release()
                    renderer = None

        # Write output MOT file
        output_mot_path = os.path.join(output_mot_folder, f"{flight_data.flight_id}_gt.txt")
        with open(output_mot_path, 'w') as f:
            # Sort by frame_id, then track_id
            output_mot_entries.sort(key=lambda e: (e.frame_id, e.track_id))
            for entry in output_mot_entries:
                f.write(mot_entry_to_line(entry) + '\n')

        logging.info(f"Saved MOT file: {output_mot_path}")

    finally:
        # Clean up
        release_all(ctx, shots)


@dataclass
class ProjectionConfig:
    """Configuration for orthographic projection."""
    ortho_width: int = 70
    ortho_height: int = 70
    input_width: int = 1024
    input_height: int = 1024
    render_width: int = 2048
    render_height: int = 2048
    camera_distance: float = 10.0
    fovy: float = 50.0
    aspect_ratio: float = 1.0
    add_background: bool = False
    project_orthogonal: bool = False
    alpha_threshold: float = 2.0
    overwrite: bool = False
    mask_suffix: str = ''  # e.g., '_t' for thermal, '_r' for RGB


def process_split(
        split: str,
        base_folder: str,
        correction_folder: str,
        output_folder: str,
        config: ProjectionConfig,
        global_corrections_file: Optional[str] = None
):
    """
    Process all flights in a split.
    """
    logging.info(f"Processing split: {split}")

    images_folder = os.path.join(base_folder, 'images', split)
    mot_folder = os.path.join(base_folder, 'mot', split)

    output_images_folder = os.path.join(output_folder, 'proj_images', split)
    output_mot_folder = os.path.join(output_folder, 'proj_mot', split)

    # Create output folders
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_mot_folder, exist_ok=True)

    # Collect flight IDs from MOT folder
    flight_ids = collect_flight_ids_from_mot(mot_folder)
    logging.info(f"Found {len(flight_ids)} flights in {split}: {sorted(flight_ids)}")

    # Load global corrections if provided
    global_corrections = None
    if global_corrections_file and os.path.exists(global_corrections_file):
        with open(global_corrections_file, 'r') as f:
            global_corrections = json.load(f)

    # Process each flight
    for flight_id in sorted(flight_ids):
        # Check for correction files
        files_exist, file_paths = check_correction_files(
            correction_folder, flight_id, config.mask_suffix
        )

        if not files_exist:
            missing = [k for k, v in file_paths.items() if v is None or not os.path.exists(v)]
            logging.warning(f"Skipping flight {flight_id}: missing files: {missing}")
            continue

        # Collect images for this flight
        image_files = collect_images_for_flight(images_folder, flight_id)

        if len(image_files) == 0:
            logging.warning(f"Skipping flight {flight_id}: no images found")
            continue

        mot_file_path = os.path.join(mot_folder, f"{flight_id}_gt.txt")

        flight_data = FlightData(
            flight_id=flight_id,
            dem_file=file_paths['dem'],
            poses_file=file_paths['poses'],
            mask_file=file_paths['mask'],
            correction_file=file_paths['correction'],
            image_files=image_files,
            mot_file=mot_file_path
        )

        # Prepare flight-specific corrections if available
        corrections_data = None
        if global_corrections and flight_id in global_corrections.get('corrections', {}):
            corrections_data = {
                'corrections': global_corrections['corrections'][flight_id]
            }

        process_flight(
            flight_data=flight_data,
            split=split,
            images_folder=images_folder,
            mot_file_path=mot_file_path,
            output_images_folder=output_images_folder,
            output_mot_folder=output_mot_folder,
            config=config,
            corrections_data=corrections_data
        )

    logging.info(f"Completed split: {split}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logging.info("Starting MOT orthographic projection")

    # Configuration from environment variables
    BASE_FOLDER = os.environ.get("BASE_FOLDER", r"D:\dataset")
    CORRECTION_FOLDER = os.environ.get("CORRECTION_FOLDER", r"D:\correction")
    OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER", r"D:\output")
    SPLITS = os.environ.get("SPLITS", "train,test,eval").split(",")

    # Optional global corrections file
    GLOBAL_CORRECTIONS_FILE = os.environ.get("GLOBAL_CORRECTIONS_FILE", None)
    if GLOBAL_CORRECTIONS_FILE and not os.path.exists(GLOBAL_CORRECTIONS_FILE):
        GLOBAL_CORRECTIONS_FILE = None

    config = ProjectionConfig(
        ortho_width=int(os.environ.get("ORTHO_WIDTH", 70)),
        ortho_height=int(os.environ.get("ORTHO_HEIGHT", 70)),
        input_width=int(os.environ.get("INPUT_WIDTH", 1024)),
        input_height=int(os.environ.get("INPUT_HEIGHT", 1024)),
        render_width=int(os.environ.get("RENDER_WIDTH", 2048)),
        render_height=int(os.environ.get("RENDER_HEIGHT", 2048)),
        camera_distance=float(os.environ.get("CAMERA_DISTANCE", 10.0)),
        fovy=float(os.environ.get("FOVY", 50.0)),
        aspect_ratio=float(os.environ.get("ASPECT_RATIO", 1.0)),
        add_background=bool(int(os.environ.get("ADD_BACKGROUND", 0))),
        project_orthogonal=bool(int(os.environ.get("PROJECT_ORTHOGONAL", 0))),
        alpha_threshold=float(os.environ.get("ALPHA_THRESHOLD", 2.0)),
        overwrite=bool(int(os.environ.get("OVERWRITE", 0))),
        mask_suffix=os.environ.get("MASK_SUFFIX", ''),  # e.g., '_t' or '_r'
    )

    # Create base output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Process each split
    for split in SPLITS:
        split = split.strip()
        if split:
            process_split(
                split=split,
                base_folder=BASE_FOLDER,
                correction_folder=CORRECTION_FOLDER,
                output_folder=OUTPUT_FOLDER,
                config=config,
                global_corrections_file=GLOBAL_CORRECTIONS_FILE
            )

    logging.info("Done")


if __name__ == "__main__":
    main()