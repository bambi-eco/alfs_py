"""
Embedded Field Renderer

Processes the same data as alfs.py but in DINOv3 embedding space.
For each key frame the script discovers neighboring frames, renders the ALFS
light field, and saves a (H, W, 1280) float32 numpy array instead of an RGB image.

Input folder structure (identical to alfs.py):
    images_dir/     {split}/{flight_id}_{frame_idx}.jpg   ← key frames
    neighbors_dir/  {split}/{flight_id}_{frame_idx}.jpg   ← neighbor frames
    corrections_dir/
        {flight_id}_dem.glb
        {flight_id}_matched_poses.json
        {flight_id}_correction.json
        {flight_id}_mask_t.png

Output:
    output_dir/
        embeddings/{split}/{flight_id}_{frame_idx}.npy
        metadata/{split}/{flight_id}_{frame_idx}_meta.json
"""

import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# ── shared helpers from alfs.py ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from alfs import (
    ALFSConfig,
    discover_flights,
    discover_neighbors,
    get_shots_for_files,
    parse_image_filename,
    get_frame_correction,
)

# ── alfspy rendering infrastructure ──────────────────────────────────────────
from alfspy.core.rendering import Resolution, TextureData, CtxShot
from alfspy.core.util.geo import get_aabb
from alfspy.embedding import (
    DinoV3Extractor,
    EmbeddingCtxShot,
    UpsampleAnything,
    _run_embedding_render_passes,
)
from alfspy.embedding.render import _render_integral_f32, _extract_or_load
from alfspy.render.data import BaseSettings, CameraPositioningMode
from alfspy.render.render import (
    make_camera, make_mgl_context, make_shot_loader,
    process_render_data, read_gltf, release_all,
)
from pyrr import Quaternion

# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EmbeddedFieldConfig:
    """
    Configuration for the DINOv3 embedding light field pipeline.
    Mirrors ALFSConfig for all data/rendering parameters and adds DINOv3 settings.
    """
    # Input folders (same layout as ALFSConfig)
    images_dir: str
    neighbors_dir: str
    corrections_dir: str
    output_dir: str

    splits: List[str] = field(default_factory=lambda: ["train"])

    # Neighbor frame settings (same as ALFSConfig)
    neighbor_range_before: int = 90
    neighbor_range_after: int = 90
    frame_stride: int = 3

    # Rendering settings (same names as ALFSConfig)
    render_width: int = 2048
    render_height: int = 2048
    ortho_width: int = 70
    ortho_height: int = 70
    camera_distance: float = 10.0
    fovy: float = 50.0
    aspect_ratio: float = 1.0
    alpha_threshold: float = 2.0

    # MOT indexing (kept for consistency, not used for labels here)
    mot_one_indexed: bool = True

    # DINOv3
    model_dir: str = r'D:\DINOv3'
    precomputed_dir: Optional[str] = None   # cache per-shot .npy; reused on re-runs

    # UpsampleAnything (optional)
    upa_repo_dir: str = r'D:\Upsample-Anything_Pytorch'
    upsample_before_render: bool = False    # Stage 1: UPA per shot before ALFS (always uses patch_size as scale)
    upsample_after_render: bool = False     # Stage 2: UPA on aggregated result
    upsample_scale: int = 2                 # scale factor for Stage 2; ignored when upsample_to_input=True
    upsample_to_input: bool = False         # Stage 2: use patch_size as scale instead of upsample_scale

    # PCA dimensionality reduction (optional, applied after the render)
    pca_components: Optional[int] = None   # None = disabled; e.g. 64 or 128


# ─────────────────────────────────────────────────────────────────────────────


def _apply_pca(embedding: np.ndarray, n_components: int,
               max_fit_pixels: int = 500_000) -> np.ndarray:
    """
    Reduces a (H, W, D) float32 embedding to (H, W, n_components) via PCA.

    PCA is fitted on a random pixel subsample (capped at *max_fit_pixels*) to
    keep memory usage manageable, then applied to the full array in row-chunks
    so that the 4-byte × H × W × D working set never needs to be fully resident
    in RAM at once.

    :param embedding: Input array of shape (H, W, D) — may be a numpy memmap.
    :param n_components: Target number of principal components.
    :param max_fit_pixels: Maximum pixels used for fitting (default 500 000).
    :return: (H, W, n_components) float32 array (in RAM).
    """
    from sklearn.decomposition import PCA

    H, W, D = embedding.shape
    N = H * W
    flat = embedding.reshape(N, D)   # view — no copy even for memmaps

    pca = PCA(n_components=n_components)
    if N > max_fit_pixels:
        idx = np.random.choice(N, max_fit_pixels, replace=False)
        pca.fit(flat[idx])
    else:
        pca.fit(flat)

    explained = pca.explained_variance_ratio_
    logging.info(
        f"  PCA {D}→{n_components}: "
        f"{[f'{v:.1%}' for v in explained]}  total={sum(explained):.1%}"
    )

    # Transform in chunks so only ~chunk × D floats are in RAM at a time
    chunk = 50_000
    reduced = np.empty((N, n_components), dtype=np.float32)
    for start in range(0, N, chunk):
        reduced[start:start + chunk] = pca.transform(flat[start:start + chunk])

    return reduced.reshape(H, W, n_components)


def _wrap_shots_with_embeddings(
    shots: List[CtxShot],
    extractor: DinoV3Extractor,
    ctx,
    w_out: int,
    h_out: int,
    precomputed_dir: Optional[str],
    id_offset: int = 0,
    upsampler: Optional[UpsampleAnything] = None,
) -> List[EmbeddingCtxShot]:
    """
    Extracts (or loads) DINOv3 embeddings for a list of shots and wraps them
    in ``EmbeddingCtxShot`` objects ready for multi-pass rendering.
    """
    n = len(shots)
    wrapped = []
    for i, shot in enumerate(shots):
        logging.info(f"    Embedding shot {i + 1}/{n} ...")
        shot.load_image()
        cache_hit = (
            precomputed_dir is not None
            and os.path.exists(os.path.join(precomputed_dir, f'shot_{id_offset + i:04d}.npy'))
        )
        emb = _extract_or_load(extractor, shot, id_offset + i, precomputed_dir)
        src = "cache" if cache_hit else f"DINOv3 → {emb.shape}"
        logging.info(f"    Shot {i + 1}/{n}: {src}")

        if upsampler is not None:
            img_bgr = shot.img[..., :3]
            H_img = round(emb.shape[0] * extractor.patch_size)
            W_img = round(emb.shape[1] * extractor.patch_size)
            guidance = UpsampleAnything.make_guidance_rgb(img_bgr, H_img, W_img)
            logging.info(f"    UPA stage 1: {emb.shape[:2]} → ({H_img}, {W_img})")
            emb = upsampler.upsample(emb, guidance)

        wrapped.append(EmbeddingCtxShot(shot, emb, ctx, w_out, h_out))
    return wrapped


def render_embedding_for_flight(
    flight_id: int,
    split: str,
    config: EmbeddedFieldConfig,
    flight_frames: List[str],
    extractor: DinoV3Extractor,
    upsampler: Optional[UpsampleAnything],
) -> None:
    """Renders DINOv3 embedding light fields for all key frames in a flight."""
    logging.info(f"Flight {flight_id}: {len(flight_frames)} key frames")

    images_folder    = os.path.join(config.images_dir,    split)
    neighbors_folder = os.path.join(config.neighbors_dir, split)
    dem_file         = os.path.join(config.corrections_dir, f"{flight_id}_dem.glb")
    poses_file       = os.path.join(config.corrections_dir, f"{flight_id}_matched_poses.json")
    correction_file  = os.path.join(config.corrections_dir, f"{flight_id}_correction.json")
    mask_file        = os.path.join(config.corrections_dir, f"{flight_id}_mask_t.png")

    if not os.path.exists(dem_file):
        logging.error(f"DEM not found: {dem_file}"); return
    if not os.path.exists(poses_file):
        logging.error(f"Poses not found: {poses_file}"); return

    output_emb_dir  = os.path.join(config.output_dir, 'embeddings', split)
    output_meta_dir = os.path.join(config.output_dir, 'metadata',   split)
    os.makedirs(output_emb_dir,  exist_ok=True)
    os.makedirs(output_meta_dir, exist_ok=True)

    # ── skip check ────────────────────────────────────────────────────────────
    frames_to_process = []
    for frame_file in flight_frames:
        stem = Path(frame_file).stem
        logging.info(stem)
        # if stem != "276_2605":
        #     continue
        print(stem)
        if not os.path.exists(os.path.join(output_emb_dir, f"{stem}.npy")):
            frames_to_process.append(frame_file)

    if not frames_to_process:
        logging.info(f"Flight {flight_id}: all frames already processed, skipping")
        return
    logging.info(f"Flight {flight_id}: {len(frames_to_process)}/{len(flight_frames)} frames to process")

    # ── load mesh, mask, poses, corrections (once per flight) ─────────────────
    mask = None
    if os.path.exists(mask_file):
        mask = TextureData(CtxShot._cvt_img(cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)))

    corrections_data = {"corrections": [], "default": {
        "translation": {"x": 0, "y": 0, "z": 0},
        "rotation": {"x": 0, "y": 0, "z": 0},
    }}
    if os.path.exists(correction_file):
        with open(correction_file) as f:
            corrections_data["default"] = json.load(f)

    with open(poses_file) as f:
        matched_poses = json.load(f)

    render_resolution = Resolution(config.render_width, config.render_height)
    w_out, h_out = config.render_width, config.render_height

    mesh_data, texture_data = read_gltf(dem_file)
    mesh_data, texture_data = process_render_data(mesh_data, texture_data)
    mesh_aabb = get_aabb(mesh_data.vertices)

    ctx = make_mgl_context()

    # Load central shots for all key frames (used for camera positioning)
    shots, shot_names, shots_rotation_eulers, corrections, correction_eulers, _ = \
        get_shots_for_files(frames_to_process, images_folder, ctx,
                            corrections_data, matched_poses, config.fovy)

    all_neighbors = discover_neighbors(config.neighbors_dir, split, flight_id)

    # ── per-frame loop ─────────────────────────────────────────────────────────
    for shot, shot_name, shot_rot_eulers, correction, cor_rot_eulers in zip(
        shots, shot_names, shots_rotation_eulers, corrections, correction_eulers
    ):
        try:
            _, frame_idx = parse_image_filename(shot_name)
            stem = Path(shot_name).stem
            out_npy  = os.path.join(output_emb_dir,  f"{stem}.npy")
            out_meta = os.path.join(output_meta_dir, f"{stem}_meta.json")

            if os.path.exists(out_npy):
                logging.info(f"Already exists, skipping: {out_npy}")
                continue

            cor_translation = correction.position

            # ── gather neighbors ───────────────────────────────────────────────
            previous_frames, previous_ids = [], []
            additional_frames, additional_ids = [], []

            for nb_idx, nb_file in all_neighbors.items():
                if frame_idx > nb_idx >= frame_idx - config.neighbor_range_before:
                    if (frame_idx - nb_idx) % config.frame_stride == 0:
                        previous_frames.append(nb_file)
                        previous_ids.append(nb_idx)
                if frame_idx < nb_idx <= frame_idx + config.neighbor_range_after:
                    if (nb_idx - frame_idx) % config.frame_stride == 0:
                        additional_frames.append(nb_file)
                        additional_ids.append(nb_idx)

            if previous_ids:
                previous_ids, previous_frames = zip(*sorted(zip(previous_ids, previous_frames)))
                previous_ids, previous_frames = list(previous_ids), list(previous_frames)
            if additional_ids:
                additional_ids, additional_frames = zip(*sorted(zip(additional_ids, additional_frames)))
                additional_ids, additional_frames = list(additional_ids), list(additional_frames)

            logging.info(f"Frame {frame_idx}: {len(previous_frames)} prev + 1 central + {len(additional_frames)} next")

            # ── load neighbor shots ────────────────────────────────────────────
            logging.info(f"  Loading {len(previous_frames)} prev shots ...")
            prev_shots, *_ = get_shots_for_files(
                previous_frames, neighbors_folder, ctx, corrections_data,
                matched_poses, config.fovy, central_frame_idx=frame_idx,
            )
            logging.info(f"  Loading {len(additional_frames)} next shots ...")
            add_shots, *_ = get_shots_for_files(
                additional_frames, neighbors_folder, ctx, corrections_data,
                matched_poses, config.fovy, central_frame_idx=frame_idx,
            )

            # ── camera positioned above central shot (same as alfs.py) ────────
            settings = BaseSettings(
                add_background=False,
                camera_dist=config.camera_distance,
                camera_position_mode=CameraPositioningMode.FirstShot,
                fovy=config.fovy, aspect_ratio=config.aspect_ratio,
                orthogonal=True, ortho_size=(config.ortho_width, config.ortho_height),
                correction=correction, resolution=render_resolution,
            )
            camera = make_camera(
                mesh_aabb, [shot], settings,
                rotation=Quaternion.from_eulers([
                    shot_rot_eulers[0] - cor_rot_eulers[0],
                    shot_rot_eulers[1] - cor_rot_eulers[1],
                    shot_rot_eulers[2] - cor_rot_eulers[2],
                ]),
            )

            from alfspy.core.rendering.renderer import Renderer
            renderer = Renderer(render_resolution, ctx, camera, mesh_data, texture_data)

            # ── embed all shots ────────────────────────────────────────────────
            precomp = config.precomputed_dir
            all_raw_shots = prev_shots + [shot] + add_shots
            logging.info(f"  Embedding {len(all_raw_shots)} shots (DINOv3) ...")
            embedding_shots = _wrap_shots_with_embeddings(
                all_raw_shots, extractor, ctx, w_out, h_out, precomp,
                id_offset=frame_idx * 1000,   # unique offset per frame to avoid cache collisions
                upsampler=upsampler if config.upsample_before_render else None,
            )
            logging.info(f"  All shots embedded. Starting {(extractor.embed_dim + 3) // 4} render passes ...")

            # ── multi-pass embedding render ────────────────────────────────────
            # When PCA is enabled the full-dim embedding is written to a temp
            # memmap; PCA is then applied and the reduced result is saved to
            # out_npy.  Without PCA, out_npy is written directly as a memmap.
            use_pca = config.pca_components is not None
            raw_path = out_npy + ".tmp" if use_pca else out_npy
            result = _run_embedding_render_passes(
                renderer, embedding_shots, mask,
                embed_dim=extractor.embed_dim,
                alpha_threshold=config.alpha_threshold,
                h_out=h_out, w_out=w_out,
                output_path=raw_path,
            )

            # ── Stage 2: UPA post-render ───────────────────────────────────────
            if config.upsample_after_render and upsampler is not None:
                scale2 = extractor.patch_size if config.upsample_to_input else config.upsample_scale
                h_up = h_out * scale2
                w_up = w_out * scale2
                rgb_raw = renderer.render_integral(
                    [e._wrapped for e in embedding_shots],
                    mask=mask, save=False, auto_contrast=False,
                )
                guidance = UpsampleAnything.make_guidance_rgb(rgb_raw[..., :3], h_up, w_up)
                logging.info(f"  UPA stage 2: ({h_out},{w_out}) → ({h_up},{w_up})"
                             + (f" [patch_size={extractor.patch_size}]" if config.upsample_to_input else ""))
                result = upsampler.upsample(result, guidance)

            renderer.release()

            # ── PCA dimensionality reduction ───────────────────────────────────
            if use_pca:
                logging.info(
                    f"  Applying PCA: {extractor.embed_dim} → {config.pca_components} components ..."
                )
                result_reduced = _apply_pca(result, config.pca_components)
                del result   # release memmap before deleting the temp file
                os.remove(raw_path)
                np.save(out_npy, result_reduced)
                saved_shape = result_reduced.shape
                saved_dim = config.pca_components
                del result_reduced
            else:
                saved_shape = result.shape
                saved_dim = extractor.embed_dim
                del result   # release memmap; data is already flushed to out_npy

            # ── save metadata ──────────────────────────────────────────────────
            meta = {
                "flight_id": flight_id, "frame_idx": frame_idx,
                "embed_dim": saved_dim,
                "patch_size": extractor.patch_size,
                "output_shape": list(saved_shape),
                "camera_position": list(map(float, camera.transform.position)),
                "render_resolution": [w_out, h_out],
                "neighbor_count": len(previous_frames) + len(additional_frames),
                "model_dir": config.model_dir,
            }
            if use_pca:
                meta["pca_components"] = config.pca_components
                meta["raw_embed_dim"] = extractor.embed_dim
            with open(out_meta, 'w') as f:
                json.dump(meta, f, indent=2)

            logging.info(f"Saved: {out_npy}")

            for e in embedding_shots:
                e.release()

        except Exception:
            logging.error(f"Error on frame {shot_name}:\n{traceback.format_exc()}")

    release_all(ctx)


# ─────────────────────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    config = EmbeddedFieldConfig(
        images_dir    = os.environ.get("IMAGES_DIR",    r"Z:\Hugo\thermal_images"),
        neighbors_dir = os.environ.get("NEIGHBORS_DIR", r"Z:\Hugo\thermal_images_neighbors"),
        corrections_dir = os.environ.get("CORRECTIONS_DIR", r"Z:\correction_data2"),
        output_dir    = os.environ.get("OUTPUT_DIR",    r"Z:\Hugo\embedded_fields_upa"),

        splits                = os.environ.get("SPLITS", "test").split(","),
        neighbor_range_before = int(os.environ.get("NEIGHBOR_RANGE_BEFORE", 90)),
        neighbor_range_after  = int(os.environ.get("NEIGHBOR_RANGE_AFTER",  90)),
        frame_stride          = int(os.environ.get("FRAME_STRIDE", 3)),
        render_width          = int(os.environ.get("RENDER_WIDTH",  2048)),
        render_height         = int(os.environ.get("RENDER_HEIGHT", 2048)),
        ortho_width           = int(os.environ.get("ORTHO_WIDTH",  70)),
        ortho_height          = int(os.environ.get("ORTHO_HEIGHT", 70)),
        camera_distance       = float(os.environ.get("CAMERA_DISTANCE", 10.0)),
        fovy                  = float(os.environ.get("FOVY",         50.0)),
        aspect_ratio          = float(os.environ.get("ASPECT_RATIO",  1.0)),
        alpha_threshold       = float(os.environ.get("ALPHA_THRESHOLD", 2.0)),
        model_dir             = os.environ.get("MODEL_DIR", r"D:\DINOv3"),
        precomputed_dir       = os.environ.get("PRECOMPUTED_DIR", None),
        upsample_before_render = bool(int(os.environ.get("UPSAMPLE_BEFORE", 0))),
        upsample_after_render  = bool(int(os.environ.get("UPSAMPLE_AFTER",  0))),
        upsample_scale         = int(os.environ.get("UPSAMPLE_SCALE", 2)),
        upsample_to_input      = bool(int(os.environ.get("UPSAMPLE_TO_INPUT", 1))),
        pca_components         = (
            int(os.environ["PCA_COMPONENTS"])
            if os.environ.get("PCA_COMPONENTS") else None
        ),
    )

    logging.info("Embedded Field Configuration:")
    for k, v in asdict(config).items():
        logging.info(f"  {k}: {v}")

    # Load DINOv3 model once (shared across all flights)
    logging.info(f"Loading DINOv3 from {config.model_dir}")
    extractor = DinoV3Extractor(config.model_dir)
    logging.info(f"  embed_dim={extractor.embed_dim}, patch_size={extractor.patch_size}")

    upsampler = None
    if config.upsample_before_render or config.upsample_after_render:
        logging.info(f"Loading UpsampleAnything from {config.upa_repo_dir}")
        upsampler = UpsampleAnything(config.upa_repo_dir)

    for split in config.splits:
        logging.info(f"=== Split: {split} ===")
        flights = discover_flights(config.images_dir, split)
        logging.info(f"Found {len(flights)} flights")

        for flight_id, flight_frames in flights.items():
            if flight_id != 276:
                continue
            logging.info(f"Processing flight {flight_id}")
            render_embedding_for_flight(
                flight_id, split, config, flight_frames,
                extractor, upsampler,
            )
            break


if __name__ == "__main__":
    main()
