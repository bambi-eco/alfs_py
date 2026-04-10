import json
import logging
import os
from copy import copy
from logging import getLogger, Logger
from typing import Optional, Final, cast

import moderngl as mgl
import numpy as np
from numpy.typing import NDArray

from alfspy.core.rendering.data import TextureData
from alfspy.core.rendering.renderer import Renderer
from alfspy.core.rendering.shot import CtxShot
from alfspy.core.util.defs import TRANSPARENT
from alfspy.core.util.loggings import LoggerStep
from alfspy.embedding.data import EmbeddingSettings
from alfspy.embedding.extractor import DinoV3Extractor
from alfspy.embedding.shot import EmbeddingCtxShot
from alfspy.embedding.upsampler import UpsampleAnything
from alfspy.render.render import (
    _base_steps, _ensure_or_copy_settings,
    release_all, render_integral,
)

logger: Final[Logger] = getLogger(__name__)


def _render_integral_f32(renderer: Renderer, shots: list[EmbeddingCtxShot],
                         mask: Optional[TextureData],
                         alpha_threshold: float = 0.1) -> NDArray:
    """
    Runs the GPU additive-blending integral render and returns a normalized float32
    array of shape (H, W, 4) without converting to uint8.

    This is a trimmed version of ``Renderer.render_integral`` that skips the
    ``* 255 → uint8`` step to preserve embedding precision.

    :param renderer: Configured ``Renderer`` instance (camera + mesh already applied).
    :param shots: ``EmbeddingCtxShot`` objects with their channel slice already set.
    :param mask: Optional mask texture.
    :param alpha_threshold: Minimum summed alpha for a pixel to be considered covered.
    :return: (H, W, 4) float32 array in [0, 1] (approximately), Y-flipped to match
        image convention.
    """
    logging.info("_render_integral_f32")
    # Apply mask (uses Renderer's internal method)
    renderer._use_mask(mask)

    ctx: mgl.Context = renderer._ctx
    fbo = renderer._fbo

    ctx.enable(cast(int, mgl.BLEND))
    ctx.disable(cast(int, mgl.DEPTH_TEST))
    ctx.blend_func = mgl.ADDITIVE_BLENDING

    fbo.clear(color=TRANSPARENT)
    for idx, shot in enumerate(shots):
        logging.info(f"-- project shot {idx}")
        renderer._project_shot(shot)

    integral_bytes = fbo.read(components=4, dtype='f4', clamp=False)
    integral_arr = np.frombuffer(integral_bytes, dtype=np.float32).reshape(
        (*fbo.size[1::-1], 4)
    )

    alpha = integral_arr[:, :, -1:]
    alpha_mask = alpha > alpha_threshold
    out = np.divide(integral_arr, alpha, where=alpha_mask, out=np.zeros_like(integral_arr))

    ctx.disable(cast(int, mgl.BLEND))

    return out[::-1, ...]  # flip Y to match image convention


def _run_embedding_render_passes(
    renderer: Renderer,
    embedding_shots: list,
    mask: Optional[TextureData],
    embed_dim: int,
    alpha_threshold: float,
    h_out: int,
    w_out: int,
    output_path: Optional[str] = None,
) -> NDArray:
    """
    Runs all multi-pass embedding render passes on an already-configured renderer.

    This is the low-level counterpart to ``render_embedding_integral`` for use when
    the renderer, camera, and shots are managed externally (e.g. when iterating over
    many key frames within a flight without reloading the mesh each time).

    To avoid holding the full ``(h_out, w_out, embed_dim)`` array in RAM, pass
    ``output_path``. The result is then written directly to a memory-mapped ``.npy``
    file so that only the current 4-channel chunk is in physical memory at a time.

    :param renderer: Fully configured ``Renderer`` (camera + mesh already applied).
    :param embedding_shots: List of ``EmbeddingCtxShot`` objects.
    :param mask: Optional mask texture.
    :param embed_dim: Number of embedding dimensions (e.g. 1280 for ViT-H+/16).
    :param alpha_threshold: Minimum summed alpha for a pixel to be considered covered.
    :param h_out: Output height in pixels.
    :param w_out: Output width in pixels.
    :param output_path: If given, the result is written to a memory-mapped ``.npy``
        file at this path instead of being allocated in RAM.  The returned array is
        the memmap object; call ``del result`` after use to free the mapping.
    :return: ``(h_out, w_out, embed_dim)`` float32 array (memmap if output_path set).
    """
    shape = (h_out, w_out, embed_dim)
    if output_path is not None:
        result = np.lib.format.open_memmap(
            output_path, mode='w+', dtype=np.float32, shape=shape
        )
        logger.info(f'  Result mapped to disk: {output_path} ({h_out}x{w_out}x{embed_dim})')
    else:
        result = np.zeros(shape, dtype=np.float32)

    n_passes = (embed_dim + 3) // 4

    for pass_idx, ch_start in enumerate(range(0, embed_dim, 4)):
        ch_end = min(ch_start + 4, embed_dim)
        n_ch = ch_end - ch_start

        if (pass_idx + 1) % 5 == 0 or pass_idx == 0 or pass_idx == n_passes - 1:
            logger.info(f'  Pass {pass_idx + 1}/{n_passes} (channels {ch_start}:{ch_end})')

        for e_shot in embedding_shots:
            e_shot.set_channel_slice(ch_start, ch_end)

        raw = _render_integral_f32(renderer, embedding_shots, mask, alpha_threshold)
        result[:, :, ch_start:ch_end] = raw[:, :, :n_ch]
        del raw

        if output_path is not None and (pass_idx + 1) % 32 == 0:
            result.flush()  # periodically evict written pages from RAM

    if output_path is not None:
        result.flush()

    return result


def _precomputed_path(precomputed_dir: str, idx: int) -> str:
    return os.path.join(precomputed_dir, f'shot_{idx:04d}.npy')


def _extract_or_load(extractor: DinoV3Extractor, shot: CtxShot,
                     idx: int, precomputed_dir: Optional[str]) -> NDArray:
    if precomputed_dir is not None:
        path = _precomputed_path(precomputed_dir, idx)
        if os.path.exists(path):
            return np.load(path)

    embedding = extractor.extract(shot.img[..., :3])

    if precomputed_dir is not None:
        os.makedirs(precomputed_dir, exist_ok=True)
        np.save(_precomputed_path(precomputed_dir, idx), embedding)

    return embedding


def render_embedding_integral(
    gltf_file: str,
    shot_json_file: str,
    mask_file: Optional[str] = None,
    settings: Optional[EmbeddingSettings] = None,
) -> None:
    """
    Renders a DINOv3 embedding light field and saves it as a numpy file.

    The pipeline mirrors ``render_integral`` but replaces each shot's raw pixel
    texture with its DINOv3 dense patch embeddings. The GPU render runs in
    ``embed_dim // 4`` passes (320 passes for ViT-H+ with embed_dim=1280), one
    per group of 4 embedding channels, and accumulates the results into a single
    ``(H_out, W_out, embed_dim)`` float32 array.

    :param gltf_file: Path to the GLTF file containing the DEM mesh (and optional
        background texture).
    :param shot_json_file: Path to the matched-poses JSON file describing each drone
        capture (image file, camera position, rotation, FOV).
    :param mask_file: Optional path to a binary mask image applied to each shot
        texture.
    :param settings: ``EmbeddingSettings`` instance. Defaults are used when ``None``.
    """
    with LoggerStep(logger, 'Start Render Embedding Integral', 'All done'):

        with LoggerStep(logger, 'Initializing'):
            se = _ensure_or_copy_settings(settings, EmbeddingSettings)

        ctx, camera, renderer, shots, mask = _base_steps(
            gltf_file, shot_json_file, mask_file, se
        )

        h_out, w_out = se.resolution.height, se.resolution.width

        with LoggerStep(logger, f'Loading DINOv3 model from "{se.model_dir}"'):
            extractor = DinoV3Extractor(se.model_dir)
            embed_dim = extractor.embed_dim

        if se.upsample_before_render:
            with LoggerStep(logger, f'Loading UpsampleAnything from "{se.upa_repo_dir}"'):
                upsampler = UpsampleAnything(se.upa_repo_dir)
        else:
            upsampler = None

        with LoggerStep(logger, f'Extracting embeddings for {len(shots)} shots'):
            for shot in shots:
                shot.load_image()

            embeddings: list[NDArray] = []
            for i, shot in enumerate(shots):
                logger.info(f'  Shot {i + 1}/{len(shots)}')
                emb = _extract_or_load(extractor, shot, i, se.precomputed_dir)

                # Stage 1: UPA per-shot — upsample (H//16, W//16, D) → (H, W, D)
                # guided by the original shot image. Scale = patch_size = 16.
                if upsampler is not None:
                    img_bgr = shot.img[..., :3]  # (H, W, 3) float32 RGBA → drop A, keep BGR
                    H_img = round(emb.shape[0] * extractor.patch_size)
                    W_img = round(emb.shape[1] * extractor.patch_size)
                    guidance = UpsampleAnything.make_guidance_rgb(img_bgr, H_img, W_img)
                    logger.info(f'    UPA stage 1: {emb.shape[:2]} → {guidance.shape[:2]}')
                    emb = upsampler.upsample(emb, guidance)

                embeddings.append(emb)

        with LoggerStep(logger, 'Wrapping shots in EmbeddingCtxShot'):
            embedding_shots = [
                EmbeddingCtxShot(shot, emb, ctx, w_out, h_out)
                for shot, emb in zip(shots, embeddings)
            ]

        n_passes = (embed_dim + 3) // 4
        result = np.zeros((h_out, w_out, embed_dim), dtype=np.float32)

        with LoggerStep(logger, f'Multi-pass embedding render ({n_passes} passes)'):
            for pass_idx, ch_start in enumerate(range(0, embed_dim, 4)):
                ch_end = min(ch_start + 4, embed_dim)
                n_ch = ch_end - ch_start

                if (pass_idx + 1) % 10 == 0 or pass_idx == 0 or pass_idx == n_passes - 1:
                    logger.info(f'  Pass {pass_idx + 1}/{n_passes} (channels {ch_start}:{ch_end})')

                for e_shot in embedding_shots:
                    e_shot.set_channel_slice(ch_start, ch_end)

                raw = _render_integral_f32(renderer, embedding_shots, mask,
                                           alpha_threshold=se.alpha_threshold)
                result[:, :, ch_start:ch_end] = raw[:, :, :n_ch]

        # Stage 2: UPA post-render — upsample (H_out, W_out, D) → (H_out*s, W_out*s, D)
        # guided by a bicubic-upscaled version of the integrated RGB render.
        if se.upsample_after_render:
            with LoggerStep(logger, f'UPA stage 2: upsampling result by {se.upsample_scale}x'):
                if upsampler is None:
                    upsampler = UpsampleAnything(se.upa_repo_dir)

                # Render RGB at the upsampled resolution to use as guidance
                h_up = h_out * se.upsample_scale
                w_up = w_out * se.upsample_scale

                # Get RGB render at render resolution and bicubic-upsample to guidance size
                rgb_raw = renderer.render_integral(
                    [shot._wrapped for shot in embedding_shots],
                    mask=mask, save=False, auto_contrast=False,
                )
                # rgb_raw: (H_out, W_out, 4) uint8 RGBA
                guidance_rgb = UpsampleAnything.make_guidance_rgb(
                    rgb_raw[..., :3], h_up, w_up
                )

                logger.info(f'  UPA stage 2: {result.shape[:2]} → ({h_up}, {w_up})')
                result = upsampler.upsample(result, guidance_rgb)

        with LoggerStep(logger, f'Saving embedding light field to "{se.embed_output_file}"'):
            output_path = se.embed_output_file
            if not output_path.endswith('.npy'):
                output_path += '.npy'
            np.save(output_path, result)

            meta_path = output_path.replace('.npy', '_meta.json')
            meta = {
                'model_dir': se.model_dir,
                'embed_dim': embed_dim,
                'patch_size': extractor.patch_size,
                'output_shape': list(result.shape),
                'camera_position': list(map(float, camera.transform.position)),
                'render_resolution': [w_out, h_out],
                'shot_count': len(shots),
                'shot_json_file': shot_json_file,
            }
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            logger.info(f'Metadata saved to "{meta_path}"')

        if se.also_save_rgb:
            with LoggerStep(logger, 'Rendering RGB light field for reference'):
                rgb_settings = copy(se)
                rgb_settings.output_file = output_path.replace('.npy', '_rgb.png')
                rgb_settings.show_integral = False
                render_integral(gltf_file, shot_json_file, mask_file, rgb_settings)

        with LoggerStep(logger, 'Releasing resources'):
            release_all(ctx, renderer, embedding_shots)
