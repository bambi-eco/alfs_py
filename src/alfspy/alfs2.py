"""
ALFS rendering for specific frames from a dataset.

Hard-coded to use:
  - poses_t.json  (list-indexed, one entry per image)
  - frames 133 and 134 from the frames_t folder
  - converted.glb as the DEM
"""

import os
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from pyrr import Quaternion, Vector3, Matrix44, Matrix33
from trimesh import Trimesh

from alfspy.core.rendering import CtxShot, TextureData
from alfspy.core.geo.transform import Transform
from alfspy.core.rendering.renderer import Renderer
from alfspy.core.util.geo import get_aabb
from alfspy.core.util.pyrrs import quaternion_from_drone_pose
from alfspy.render.data import BaseSettings, CameraPositioningMode
from alfspy.render.render import (
    make_camera,
    make_torch_context,
    make_shot_loader,
    process_render_data,
    read_gltf,
    release_all,
)
from alfspy.core.rendering import Resolution

# ── Hard-coded paths ──────────────────────────────────────────────────────────
POSES_FILE = r"C:\Users\P41743\Desktop\sweden_test2\output_converted2\poses_t.json"
FRAMES_DIR = r"C:\Users\P41743\Desktop\sweden_test2\output_converted2\frames_t"
DEM_FILE   = r"C:\Users\P41743\Desktop\sweden_test2\converted.glb"
OUTPUT_DIR = r"C:\Users\P41743\Desktop\sweden_test2\output_converted2\alfs_output"

# 1-based frame numbers as they appear in the filenames (_0133_, _0134_, …)
# The first entry is treated as the central frame; the rest are neighbors.
FRAME_NUMBERS = [133, 134]

# ── Rendering settings ────────────────────────────────────────────────────────
RENDER_WIDTH    = 2048
RENDER_HEIGHT   = 2048
ORTHO_WIDTH     = 70
ORTHO_HEIGHT    = 70
CAMERA_DISTANCE = 10.0
FOVY            = 46.8
ASPECT_RATIO    = 1.0
ALPHA_THRESHOLD = 0.5  # must be < number of shots; 2.0 would mask everything with only 2 frames


# ─────────────────────────────────────────────────────────────────────────────

def load_poses_by_frame_number(poses_file: str) -> dict:
    """
    Parse poses_t.json and return a dict mapping the 4-digit frame number
    embedded in each imagefile name to its pose entry.

    Filename pattern:  DJI_<timestamp>_<NNNN>_T.png
    The frame number is the third '_'-separated token (1-based).
    """
    with open(poses_file, "r") as f:
        data = json.load(f)

    poses = {}
    for entry in data["images"]:
        stem = Path(entry["imagefile"]).stem          # e.g. DJI_20260301081902_0133_T
        tokens = stem.split("_")
        # tokens: ['DJI', '<timestamp>', '<NNNN>', '<suffix>']
        if len(tokens) < 3:
            continue
        try:
            frame_num = int(tokens[-2])               # '0133' → 133
            poses[frame_num] = entry
        except ValueError:
            continue
    return poses


def find_image_file(frames_dir: str, frame_number: int) -> str:
    """
    Find the image file in *frames_dir* whose name contains the zero-padded
    frame number (e.g. '_0133_').  Returns the full path.
    """
    needle = f"_{frame_number:04d}_"
    for name in os.listdir(frames_dir):
        if needle in name and (name.endswith(".png") or name.endswith(".jpg")):
            return os.path.join(frames_dir, name)
    raise FileNotFoundError(
        f"No file containing '{needle}' found in {frames_dir}"
    )


def make_shot(ctx, image_path: str, pose: dict, fovy_default: float) -> CtxShot:
    """Create a CtxShot from an image path and its pose dictionary."""
    camera_position = Vector3(pose["location"])

    rot = pose["rotation"]
    rot = [v % 360.0 for v in rot]
    if len(rot) == 3:
        camera_rotation = quaternion_from_drone_pose(rot)
    elif len(rot) == 4:
        camera_rotation = Quaternion(rot)
    else:
        raise ValueError(f"Unexpected rotation length {len(rot)}: {rot}")

    fov = pose.get("fovy", fovy_default)
    if isinstance(fov, list):
        fov = fov[0] if fov else fovy_default

    # No corrections for this dataset — identity transform
    return CtxShot(
        ctx,
        image_path,
        camera_position,
        camera_rotation,
        fov,
        1,
        None,
        lazy=False,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load poses ────────────────────────────────────────────────────────────
    logging.info(f"Loading poses from {POSES_FILE}")
    poses = load_poses_by_frame_number(POSES_FILE)
    logging.info(f"Loaded {len(poses)} pose entries")

    for fn in FRAME_NUMBERS:
        if fn not in poses:
            raise KeyError(f"Frame number {fn} not found in poses file")

    # ── Load DEM ──────────────────────────────────────────────────────────────
    logging.info(f"Loading DEM from {DEM_FILE}")
    mesh_data, texture_data = read_gltf(DEM_FILE)
    tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
    mesh_data, texture_data = process_render_data(mesh_data, texture_data)
    mesh_aabb = get_aabb(mesh_data.vertices)

    # ── Create rendering context and shots ────────────────────────────────────
    ctx = make_torch_context()
    render_resolution = Resolution(RENDER_WIDTH, RENDER_HEIGHT)

    central_frame_num = FRAME_NUMBERS[0]
    neighbor_frame_nums = FRAME_NUMBERS[1:]

    central_image = find_image_file(FRAMES_DIR, central_frame_num)
    central_pose  = poses[central_frame_num]
    central_shot  = make_shot(ctx, central_image, central_pose, FOVY)
    logging.info(f"Central frame {central_frame_num}: {central_image}")

    neighbor_shots = []
    for fn in neighbor_frame_nums:
        img_path = find_image_file(FRAMES_DIR, fn)
        shot = make_shot(ctx, img_path, poses[fn], FOVY)
        neighbor_shots.append(shot)
        logging.info(f"Neighbor frame {fn}: {img_path}")

    # ── Camera & renderer setup ───────────────────────────────────────────────
    settings = BaseSettings(
        count=len(FRAME_NUMBERS),
        initial_skip=0,
        add_background=True,
        camera_dist=CAMERA_DISTANCE,
        camera_position_mode=CameraPositioningMode.FirstShot,
        fovy=FOVY,
        aspect_ratio=ASPECT_RATIO,
        orthogonal=True,
        ortho_size=(ORTHO_WIDTH, ORTHO_HEIGHT),
        correction=None,
        resolution=render_resolution,
    )

    single_shot_camera = make_camera(mesh_aabb, [central_shot], settings)
    renderer = Renderer(render_resolution, ctx, single_shot_camera, mesh_data, texture_data)

    # ── Render ALFS integral ──────────────────────────────────────────────────
    all_shots = [central_shot] + neighbor_shots
    shot_loader = make_shot_loader(all_shots)

    output_name = f"alfs_frame{'_'.join(str(n) for n in FRAME_NUMBERS)}.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    renderer.render_integral(
        shot_loader,
        mask=None,
        save=True,
        release_shots=False,
        save_name=output_path,
        alpha_threshold=ALPHA_THRESHOLD,
    )
    logging.info(f"Saved ALFS image: {output_path}")

    # Load the image
    img = cv2.imread(output_path)

    if img is None:
        print("Error: Could not load image.")
        exit()

    # Resize image to 500x500 pixels
    resized_img = cv2.resize(img, (500, 500))

    # Show the image
    cv2.imshow("Resized Image (500x500)", resized_img)

    # Wait for a key press and close window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    renderer.release()
    release_all(ctx, renderer, all_shots)
    logging.info("Done")


if __name__ == "__main__":
    main()
