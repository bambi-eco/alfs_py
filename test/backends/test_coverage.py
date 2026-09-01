"""Coverage must be the overlap count, and must not be inferred from an alpha channel.

The renderer used to sum each shot's own alpha and divide by it, so alpha did double duty as
the overlap counter. That works only while every shot's alpha is exactly 1, and it costs a
channel: an N-channel field striped through an RGBA pipeline has its fourth channel
overwritten by the counter and then divided by itself, which is how 25% of the channels in
the embedded_light_field prototype came out as binary masks.

These tests pin coverage as a separate quantity with a checkable value.
"""

import numpy as np
import pytest
from pyrr import Quaternion, Vector3

from alfspy.core.backends import available_engines, create_context
from alfspy.core.rendering import CtxShot, Renderer
from alfspy.core.rendering.data import IntegralResult, Resolution
from test.helpers.scenes import (
    fovy_covering,
    gradient_rgba,
    height_field,
    perspective_camera_above,
    solid_rgba,
)

ENGINES = available_engines()

RESOLUTION = Resolution(64, 64)
HALF = 20.0
HEIGHT = 40.0


@pytest.fixture(params=ENGINES)
def ctx(request):
    context = create_context(engine=request.param)
    yield context
    context.release()


def _flat_dem():
    return height_field(resolution=8, half=HALF, amplitude=0.0, seed=0)


def _renderer(ctx):
    camera = perspective_camera_above(
        fovy=fovy_covering(HALF, HEIGHT), aspect_ratio=1.0, height=HEIGHT)
    return Renderer(RESOLUTION, ctx, camera, _flat_dem())


def _shot(ctx, dx=0.0, dy=0.0, half=HALF, image=None):
    return CtxShot(
        ctx,
        gradient_rgba(width=32, height=32) if image is None else image,
        Vector3([dx, dy, HEIGHT]),
        Quaternion(),
        fovy=fovy_covering(half, HEIGHT),
        aspect_ratio=1.0,
    )


def test_coverage_counts_identical_shots(ctx):
    """
    Three shots at the same pose cover the same pixels, so every covered pixel must have a
    coverage of exactly 3. Nothing about the image content can change that.
    """
    renderer = _renderer(ctx)
    shots = [_shot(ctx) for _ in range(3)]
    try:
        result = renderer.render_integral_raw(shots)
    finally:
        for shot in shots:
            shot.release()
        renderer.release()

    covered = result.coverage > 0
    assert covered.any(), 'the shots covered nothing'
    np.testing.assert_allclose(result.coverage[covered], 3.0, atol=1e-4)


def test_coverage_grows_with_the_number_of_shots(ctx):
    renderer = _renderer(ctx)
    try:
        for count in (1, 2, 5):
            shots = [_shot(ctx) for _ in range(count)]
            try:
                result = renderer.render_integral_raw(shots)
            finally:
                for shot in shots:
                    shot.release()
            covered = result.coverage > 0
            np.testing.assert_allclose(result.coverage[covered], float(count), atol=1e-4)
    finally:
        renderer.release()


def test_partially_overlapping_shots_give_a_range_of_counts(ctx):
    """Offset footprints must produce a genuine 1/2 overlap pattern, not a constant."""
    renderer = _renderer(ctx)
    shots = [_shot(ctx, dx=-8.0, half=HALF * 0.5), _shot(ctx, dx=8.0, half=HALF * 0.5)]
    try:
        result = renderer.render_integral_raw(shots)
    finally:
        for shot in shots:
            shot.release()
        renderer.release()

    counts = np.unique(np.round(result.coverage[result.coverage > 0]).astype(int))
    assert set(counts) == {1, 2}, f'expected single and doubly covered pixels, got {counts}'


def test_coverage_is_independent_of_pixel_values(ctx):
    """
    The point of separating coverage: a shot whose fourth channel is 0 still counts as one
    observation. Under the old alpha-as-counter scheme it counted as none.
    """
    renderer = _renderer(ctx)
    transparent = solid_rgba(colour=(200.0, 100.0, 50.0, 0.0), width=32, height=32)
    opaque = solid_rgba(colour=(200.0, 100.0, 50.0, 255.0), width=32, height=32)

    try:
        a = [_shot(ctx, image=transparent)]
        try:
            zero_alpha = renderer.render_integral_raw(a)
        finally:
            a[0].release()

        b = [_shot(ctx, image=opaque)]
        try:
            full_alpha = renderer.render_integral_raw(b)
        finally:
            b[0].release()
    finally:
        renderer.release()

    np.testing.assert_allclose(zero_alpha.coverage, full_alpha.coverage, atol=1e-4)
    assert (zero_alpha.coverage > 0).any()


def test_normalised_averages_and_leaves_uncovered_pixels_at_the_fill(ctx):
    renderer = _renderer(ctx)
    shots = [_shot(ctx, half=HALF * 0.5) for _ in range(3)]
    try:
        result = renderer.render_integral_raw(shots)
    finally:
        for shot in shots:
            shot.release()
        renderer.release()

    averaged = result.normalised(threshold=0.1)
    covered = result.coverage > 0.1

    assert not covered.all(), 'shots should not cover the whole frame in this scene'
    # Averaging three identical shots gives back one shot's values, so the average must lie
    # in the same [0, 1] range a single sample does -- not three times it.
    assert averaged[covered].max() <= 1.0 + 1e-4
    np.testing.assert_allclose(averaged[~covered], 0.0, atol=1e-6)


def test_normalised_does_not_leak_uninitialised_memory():
    """
    ``np.divide(..., where=...)`` leaves excluded entries at whatever was in the freshly
    allocated buffer unless an explicit ``out=`` is given. Those values then go through
    ``* 255`` and ``.astype(np.uint8)``, which is why artifacts appeared "sometimes".
    """
    accum = np.full((8, 8, 4), 7.0, dtype=np.float32)
    coverage = np.zeros((8, 8), dtype=np.float32)
    result = IntegralResult(accum=accum, coverage=coverage)

    for _ in range(20):
        out = result.normalised(threshold=0.1)
        assert np.all(out == 0.0), 'uncovered pixels must be the fill value, deterministically'


@pytest.mark.skipif(len(ENGINES) < 2, reason='need two backends to compare')
def test_backends_agree_on_coverage_counts():
    results = {}
    for engine in ENGINES:
        context = create_context(engine=engine)
        renderer = _renderer(context)
        shots = [_shot(context, dx=-6.0, half=HALF * 0.6), _shot(context, dx=6.0, half=HALF * 0.6)]
        try:
            results[engine] = renderer.render_integral_raw(shots).coverage
        finally:
            for shot in shots:
                shot.release()
            renderer.release()
            context.release()

    reference = ENGINES[0]
    for other in ENGINES[1:]:
        a = np.round(results[other]).astype(int)
        b = np.round(results[reference]).astype(int)
        disagreement = float((a != b).mean())
        assert disagreement <= 0.01, (
            f'{other} and {reference} disagree on the overlap count for '
            f'{disagreement:.2%} of pixels')
