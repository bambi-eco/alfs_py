"""End-to-end ``Renderer`` behaviour: background, projective texture mapping, integration.

The projective-mapping test derives the expected sample analytically from the camera
geometry, so it validates the whole chain at once -- matrix convention, perspective-correct
interpolation of the shot varying, texture orientation and the GL bottom-up framebuffer.
"""

import math

import numpy as np
import pytest
import torch
from pyrr import Quaternion, Vector3

from alfspy.core.rendering import CtxShot, RenderResultMode, Renderer, Resolution, TextureData
from alfspy.core.torchgl import create_context
from test.helpers.images import assert_images_close, coverage
from test.helpers.scenes import (
    checkerboard_rgba,
    flat_quad,
    fovy_covering,
    gradient_rgba,
    height_field,
    ortho_camera_above,
    solid_rgba,
)

RES = 64
DEM_HALF = 25.0
ORTHO = 40.0
CAM_HEIGHT = 30.0
# Deliberately smaller than half the orthographic view, so a projected shot leaves an
# uncovered border. Without that margin the coverage assertions below would be vacuous.
SHOT_HALF = 15.0


@pytest.fixture
def scene(ctx):
    """A flat DEM with an orthographic virtual camera looking straight down at it."""
    mesh = flat_quad(half=DEM_HALF)
    texture = TextureData(checkerboard_rgba())
    camera = ortho_camera_above(size=(ORTHO, ORTHO), height=CAM_HEIGHT, far=200.0)
    renderer = Renderer(Resolution(RES, RES), ctx, camera, mesh, texture)
    yield renderer
    renderer.release()


def make_shot(ctx, image, x=0.0, y=0.0, height=CAM_HEIGHT, half=SHOT_HALF):
    """A capture looking straight down from ``(x, y, height)`` covering ``+/- half``."""
    return CtxShot(ctx, image, position=Vector3([x, y, height]), rotation=Quaternion(),
                   fovy=fovy_covering(half, height), aspect_ratio=1.0)


# ────────────────────────────────── background ──────────────────────────────


def test_background_fills_the_viewport(scene):
    result = scene.render_background()

    assert result.shape == (RES, RES, 4)
    assert result.dtype == np.uint8
    assert coverage(result) == 1.0


def test_background_reproduces_the_texture(ctx):
    """
    An orthographic camera fitted exactly to a quad whose UVs span [0,1] is an identity
    resample: the render must reproduce the texture.
    """
    mesh = flat_quad(half=DEM_HALF)
    texture = TextureData(checkerboard_rgba(tiles=8, tile_size=8))  # 64x64
    camera = ortho_camera_above(size=(2 * DEM_HALF, 2 * DEM_HALF), height=CAM_HEIGHT, far=200.0)

    renderer = Renderer(Resolution(64, 64), ctx, camera, mesh, texture)
    try:
        result = renderer.render_background()
    finally:
        renderer.release()

    expected = checkerboard_rgba(tiles=8, tile_size=8).astype(np.uint8)
    assert_images_close(result, expected, tolerance=1, max_bad_fraction=0.0,
                        label='identity-resampled background')


def test_release_is_idempotent(scene):
    scene.release()
    scene.release()


# ─────────────────────────── projective texture mapping ─────────────────────


def test_projected_shot_matches_analytic_uv_mapping(scene, ctx):
    """
    The core of both ALFS and orthographic projection.

    The shot texture encodes its own UVs (red = u, green = 1 - v after the GL flip), so the
    rendered colour states directly which source texel each output pixel pulled from. Those
    UVs are recomputed here from the pinhole geometry alone.
    """
    shot_image = gradient_rgba(64, 64)
    shot = make_shot(ctx, shot_image)

    result = scene.project_shots(shot, RenderResultMode.ShotOnly)[0]
    assert result.shape == (RES, RES, 4)

    # Output row 0 is the top; the framebuffer stores rows bottom-up.
    rows = np.arange(RES)
    cols = np.arange(RES)
    grid_col, grid_row = np.meshgrid(cols, rows, indexing='xy')

    py = RES - 1 - grid_row
    world_x = ((grid_col + 0.5) / RES * 2.0 - 1.0) * (ORTHO / 2.0)
    world_y = ((py + 0.5) / RES * 2.0 - 1.0) * (ORTHO / 2.0)

    # Pinhole looking straight down from CAM_HEIGHT covering +/- SHOT_HALF at the DEM plane.
    ndc_x = world_x / SHOT_HALF
    ndc_y = world_y / SHOT_HALF
    u = (ndc_x + 1.0) / 2.0
    v = (ndc_y + 1.0) / 2.0

    inside = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    assert inside.any(), 'test setup projects nothing'

    # Stay clear of the outermost half texel, where sampling clamps to the border.
    margin = 0.5 / 64
    core = inside & (u > margin) & (u < 1 - margin) & (v > margin) & (v < 1 - margin)

    np.testing.assert_allclose(result[..., 0][core] / 255.0, u[core], atol=2 / 255)
    np.testing.assert_allclose(result[..., 1][core] / 255.0, (1.0 - v)[core], atol=2 / 255)

    # Everything the shot cannot see must have been discarded.
    assert (result[..., 3][~inside] == 0).all(), 'fragments outside the shot frustum leaked in'
    assert (result[..., 3][core] > 0).all(), 'fragments inside the shot frustum were dropped'


def test_projected_shot_footprint_has_the_expected_area(scene, ctx):
    """The projected footprint is a known fraction of the orthographic view."""
    shot = make_shot(ctx, solid_rgba((255.0, 0.0, 0.0, 255.0)))

    result = scene.project_shots(shot, RenderResultMode.ShotOnly)[0]

    expected_fraction = (2 * SHOT_HALF) ** 2 / ORTHO ** 2
    assert coverage(result) == pytest.approx(expected_fraction, abs=0.02)


def test_shot_translation_moves_the_footprint(scene, ctx):
    image = solid_rgba((255.0, 0.0, 0.0, 255.0))

    shift = 3.0  # small enough that the shifted footprint stays fully inside the view
    centred = scene.project_shots(make_shot(ctx, image), RenderResultMode.ShotOnly)[0]
    shifted = scene.project_shots(make_shot(ctx, image, x=shift), RenderResultMode.ShotOnly)[0]

    def centroid_x(img):
        mask = img[..., 3] > 0
        return float(np.nonzero(mask)[1].mean())

    expected_shift = shift / ORTHO * RES
    assert centroid_x(shifted) - centroid_x(centred) == pytest.approx(expected_shift, abs=1.0)


def test_mask_zeroes_the_masked_region(scene, ctx):
    shot = make_shot(ctx, solid_rgba((255.0, 255.0, 255.0, 255.0)))

    mask_image = np.zeros((32, 32, 1), dtype='f4')
    mask_image[:16, :, 0] = 1.0  # top half of the source frame passes

    result = scene.project_shots(shot, RenderResultMode.ShotOnly,
                                 mask=TextureData(mask_image))[0]

    covered = coverage(result)
    unmasked = scene.project_shots(shot, RenderResultMode.ShotOnly)[0]
    assert covered == pytest.approx(coverage(unmasked) / 2.0, abs=0.02)


# ──────────────────────────────── integration ───────────────────────────────


def test_integral_alpha_counts_overlapping_shots(scene, ctx):
    """
    Alpha is the ALFS overlap counter. Three identical shots must accumulate exactly 3.0 --
    no more (double-counted shared edges) and no less (dropped fragments).
    """
    image = solid_rgba((200.0, 100.0, 50.0, 255.0))
    shots = [make_shot(ctx, image) for _ in range(3)]

    scene._use_mask(None)
    scene.fbo.clear((0.0, 0.0, 0.0, 0.0))
    for shot in shots:
        scene._project_shot(shot, additive=True, depth_test=False)

    raw = scene.fbo.read_array(dtype='f4', clamp=False)
    alpha = raw[..., 3]

    covered = alpha > 0.5
    assert covered.any(), 'nothing was projected'

    # Only two populations may exist: uncovered, and covered by exactly three shots. A value
    # near 6 would mean shared edges were counted twice; near 2 would mean fragments were
    # dropped. Sub-LSB float32 error from bilinear alpha sampling is expected and ignored.
    assert (alpha[~covered] == 0.0).all()
    np.testing.assert_allclose(alpha[covered], 3.0, atol=1e-5)


def test_integral_of_identical_shots_equals_one_shot(scene, ctx):
    """Normalising by the overlap count must undo the accumulation exactly."""
    image = gradient_rgba(64, 64)

    single = scene.render_integral([make_shot(ctx, image)], auto_contrast=False)
    tripled = scene.render_integral([make_shot(ctx, image) for _ in range(3)], auto_contrast=False)

    assert_images_close(tripled, single, tolerance=1, max_bad_fraction=0.001,
                        label='integral of 3 identical shots vs 1')


def test_integral_partial_overlap_counts_correctly(scene, ctx):
    """Two shots offset in X: the shared band must report 2, the wings 1."""
    image = solid_rgba((255.0, 255.0, 255.0, 255.0))
    shots = [make_shot(ctx, image, x=-5.0), make_shot(ctx, image, x=5.0)]

    scene._use_mask(None)
    scene.fbo.clear((0.0, 0.0, 0.0, 0.0))
    for shot in shots:
        scene._project_shot(shot, additive=True, depth_test=False)

    alpha = scene.fbo.read_array(dtype='f4', clamp=False)[..., 3]

    rounded = np.round(alpha)
    np.testing.assert_allclose(alpha, rounded, atol=1e-5,
                               err_msg='overlap counts must be whole numbers')
    assert set(np.unique(rounded).tolist()) == {0.0, 1.0, 2.0}

    # Footprints span x in [-20, 10] and [-10, 20]; they overlap over x in [-10, 10] and
    # share the full y extent of 2 * SHOT_HALF.
    overlap_area = 20.0 * (2 * SHOT_HALF)
    assert (rounded == 2.0).mean() == pytest.approx(overlap_area / ORTHO ** 2, abs=0.03)


def test_integral_alpha_threshold_suppresses_thin_coverage(scene, ctx):
    image = solid_rgba((255.0, 255.0, 255.0, 255.0))
    shots = [make_shot(ctx, image, x=-5.0), make_shot(ctx, image, x=5.0)]

    lenient = scene.render_integral(shots, alpha_threshold=0.1, auto_contrast=False)
    strict = scene.render_integral(shots, alpha_threshold=1.5, auto_contrast=False)

    assert coverage(lenient) > coverage(strict), (
        'raising the alpha threshold must discard pixels seen by only one shot'
    )


def test_repeated_renders_are_bit_identical(scene, ctx):
    """
    The regression guard for the artifact class this migration exists to remove: renders must
    not depend on whatever was in the buffer beforehand.
    """
    image = gradient_rgba(32, 32)

    first = scene.render_integral([make_shot(ctx, image)], auto_contrast=False)
    _ = scene.render_background()
    _ = scene.render_integral([make_shot(ctx, image, x=12.0)], auto_contrast=False)
    second = scene.render_integral([make_shot(ctx, image)], auto_contrast=False)

    assert np.array_equal(first, second), 'render depends on prior framebuffer contents'


def test_dem_mesh_renders_without_seams(ctx):
    """
    A many-triangle DEM must not show a lattice of double-counted internal edges in the
    accumulator.
    """
    mesh = height_field(resolution=12, half=DEM_HALF, amplitude=0.0)
    camera = ortho_camera_above(size=(ORTHO, ORTHO), height=CAM_HEIGHT, far=200.0)
    renderer = Renderer(Resolution(RES, RES), ctx, camera, mesh, TextureData(checkerboard_rgba()))

    try:
        shot = make_shot(ctx, solid_rgba((255.0, 255.0, 255.0, 255.0)))
        renderer._use_mask(None)
        renderer.fbo.clear((0.0, 0.0, 0.0, 0.0))
        renderer._project_shot(shot, additive=True, depth_test=False)
        alpha = renderer.fbo.read_array(dtype='f4', clamp=False)[..., 3]
    finally:
        renderer.release()

    rounded = np.round(alpha)
    assert set(np.unique(rounded).tolist()) == {0.0, 1.0}, (
        f'internal DEM edges were counted more than once: {np.unique(rounded).tolist()}'
    )


# ─────────────────────────────── device coverage ────────────────────────────


@pytest.mark.cuda
def test_cuda_matches_cpu():
    """CPU and CUDA must agree; the scatter-min resolve is deterministic on both."""
    mesh = height_field(resolution=8, half=DEM_HALF)
    texture = TextureData(checkerboard_rgba())
    image = gradient_rgba(64, 64)

    results = []
    for device in ('cpu', 'cuda'):
        context = create_context(device=device)
        camera = ortho_camera_above(size=(ORTHO, ORTHO), height=CAM_HEIGHT, far=200.0)
        renderer = Renderer(Resolution(RES, RES), context, camera, mesh, texture)
        try:
            shot = make_shot(context, image)
            results.append(renderer.project_shots(shot, RenderResultMode.ShotOnly)[0])
        finally:
            renderer.release()
            context.release()

    assert_images_close(results[1], results[0], tolerance=1, max_bad_fraction=0.001,
                        label='CUDA vs CPU render')
