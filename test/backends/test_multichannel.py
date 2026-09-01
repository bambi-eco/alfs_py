"""N-channel light fields.

These are written against the three defects the ``embedded_light_field`` prototype carried,
because each is silent: the output has the right shape and plausible statistics either way.

1. Feature dimension 3 of every group was packed into the alpha channel, which the normaliser
   then divided by -- so 25% of channels came back as binary masks.
   ``test_every_channel_survives`` catches that.
2. Slices were uploaded without the vertical flip that ``TextureData`` applies, so features
   were sampled from the mirrored half of each shot. ``test_field_matches_the_rgb_path``
   catches that, because a Y-asymmetric pattern rendered through both paths must agree.
3. Slices were resized to the render resolution on the CPU before upload.
   ``test_low_resolution_field_is_interpolated_by_the_gpu`` pins the behaviour that makes
   that unnecessary.
"""

import json

import numpy as np
import pytest
from pyrr import Quaternion, Vector3

from alfspy.core.backends import available_engines, create_context
from alfspy.core.rendering import CtxShot, Renderer
from alfspy.core.rendering.data import Resolution
from alfspy.io.field import (
    FieldMetadata,
    load_field,
    load_metadata,
    meta_path,
    open_field,
    save_field,
)
from alfspy.render.field import ChannelSpec, FieldShot, render_field_integral
from test.helpers.scenes import fovy_covering, gradient_rgba, height_field, perspective_camera_above

ENGINES = available_engines()

RESOLUTION = Resolution(48, 48)
HALF = 20.0
HEIGHT = 40.0


@pytest.fixture(params=ENGINES)
def ctx(request):
    context = create_context(engine=request.param)
    yield context
    context.release()


def _renderer(ctx):
    camera = perspective_camera_above(
        fovy=fovy_covering(HALF, HEIGHT), aspect_ratio=1.0, height=HEIGHT)
    return Renderer(RESOLUTION, ctx, camera, height_field(resolution=8, half=HALF, amplitude=0.0))


def _field_shot(field, dx=0.0, dy=0.0, half=HALF):
    return FieldShot(
        field=field,
        position=Vector3([dx, dy, HEIGHT]),
        rotation=Quaternion(),
        fovy=fovy_covering(half, HEIGHT),
        aspect_ratio=1.0,
    )


def _constant_field(channels, size=16):
    """Channel ``i`` is the constant ``i``, so any channel mix-up is directly readable."""
    field = np.zeros((size, size, channels), dtype=np.float32)
    for i in range(channels):
        field[:, :, i] = float(i)
    return field


# ───────────────────────────── channel integrity ─────────────────────────────


@pytest.mark.parametrize('channels', [1, 3, 4, 5, 8, 13])
def test_every_channel_survives(ctx, channels):
    """
    Channel ``i`` holds the constant ``i``, so after averaging, every covered pixel of
    channel ``i`` must still read ``i``.

    This is the defect that made 25% of the prototype's channels binary masks: with a
    four-wide group, channels 3, 7, 11, ... landed in alpha and were divided by themselves,
    coming out as exactly 1.0 where covered.
    """
    field = _constant_field(channels)
    renderer = _renderer(ctx)
    try:
        result = render_field_integral(renderer, ctx, [_field_shot(field)] * 3)
    finally:
        renderer.release()

    assert result.accum.shape == (RESOLUTION.height, RESOLUTION.width, channels)

    averaged = result.normalised(threshold=0.1)
    covered = result.coverage > 0.1
    assert covered.any()

    for i in range(channels):
        values = averaged[..., i][covered]
        np.testing.assert_allclose(
            values, float(i), atol=1e-3,
            err_msg=f'channel {i} came back as {np.unique(np.round(values, 3))[:5]} '
                    f'instead of the constant {i}')


def test_channel_three_is_not_a_coverage_mask(ctx):
    """
    A direct regression test for the prototype. Under its scheme every fourth channel came
    out as exactly 1.0 where covered and 0.0 elsewhere, regardless of its input value.
    """
    channels = 8
    field = np.full((16, 16, channels), 0.25, dtype=np.float32)
    renderer = _renderer(ctx)
    try:
        result = render_field_integral(renderer, ctx, [_field_shot(field)] * 2)
    finally:
        renderer.release()

    averaged = result.normalised(threshold=0.1)
    covered = result.coverage > 0.1

    for i in (3, 7):
        values = averaged[..., i][covered]
        assert not np.allclose(values, 1.0, atol=1e-6), (
            f'channel {i} is a binary coverage mask, not data -- alpha is being stolen')
        np.testing.assert_allclose(values, 0.25, atol=1e-3)


def test_coverage_is_shared_across_channel_groups(ctx):
    """Coverage depends only on geometry, so it must not vary with the channel count."""
    renderer = _renderer(ctx)
    try:
        narrow = render_field_integral(renderer, ctx, [_field_shot(_constant_field(2))] * 3)
        wide = render_field_integral(renderer, ctx, [_field_shot(_constant_field(11))] * 3)
    finally:
        renderer.release()

    np.testing.assert_allclose(narrow.coverage, wide.coverage, atol=1e-4)
    covered = narrow.coverage > 0
    np.testing.assert_allclose(narrow.coverage[covered], 3.0, atol=1e-4)


# ─────────────────────────── orientation and sampling ────────────────────────


def test_field_matches_the_rgb_path(ctx):
    """
    A four-channel field must render identically to the same image through the ordinary
    shot path. A missing vertical flip shows up here and almost nowhere else.

    The image is scaled into [0, 1] first so the two paths agree on value handling and this
    test is only about orientation: the RGB path rescales 0-255 input by 1/255, while the
    field path deliberately does not.
    """
    image = gradient_rgba(width=32, height=32).astype(np.float32) / 255.0

    renderer = _renderer(ctx)
    try:
        via_field = render_field_integral(renderer, ctx, [_field_shot(image)])
        shot = CtxShot(ctx, image, Vector3([0.0, 0.0, HEIGHT]), Quaternion(),
                       fovy=fovy_covering(HALF, HEIGHT), aspect_ratio=1.0)
        try:
            via_rgb = renderer.render_integral_raw([shot])
        finally:
            shot.release()
    finally:
        renderer.release()

    np.testing.assert_allclose(via_field.coverage, via_rgb.coverage, atol=1e-4)
    np.testing.assert_allclose(via_field.accum, via_rgb.accum, atol=1e-3,
                               err_msg='the field path disagrees with the RGB path -- check '
                                       'the vertical flip')


def test_low_resolution_field_is_interpolated_by_the_gpu(ctx):
    """
    A field far coarser than the render resolution must still fill the footprint. This is
    what makes CPU-side upsampling before upload unnecessary: a 128x128 patch grid uploads
    as 128x128 and the sampler does the rest.
    """
    field = _constant_field(4, size=4)  # 4x4 field, 48x48 render
    renderer = _renderer(ctx)
    try:
        result = render_field_integral(renderer, ctx, [_field_shot(field)])
    finally:
        renderer.release()

    covered = result.coverage > 0
    assert covered.mean() > 0.5, 'a coarse field should still cover the footprint'
    averaged = result.normalised(threshold=0.1)
    np.testing.assert_allclose(averaged[..., 2][covered], 2.0, atol=1e-3)


# ──────────────────────────────── plumbing ───────────────────────────────────


def test_mismatched_channel_counts_are_rejected(ctx):
    renderer = _renderer(ctx)
    try:
        with pytest.raises(ValueError, match='same channel count'):
            render_field_integral(renderer, ctx, [
                _field_shot(_constant_field(4)),
                _field_shot(_constant_field(5)),
            ])
    finally:
        renderer.release()


def test_no_shots_is_rejected(ctx):
    renderer = _renderer(ctx)
    try:
        with pytest.raises(ValueError, match='at least one shot'):
            render_field_integral(renderer, ctx, [])
    finally:
        renderer.release()


def test_progress_is_reported_per_group(ctx):
    seen = []
    renderer = _renderer(ctx)
    try:
        render_field_integral(renderer, ctx, [_field_shot(_constant_field(9))],
                              progress=lambda i, n: seen.append((i, n)))
    finally:
        renderer.release()

    assert seen == [(1, 3), (2, 3), (3, 3)]  # 9 channels -> three four-wide groups


def test_channel_spec_validates():
    with pytest.raises(ValueError, match='at least one channel'):
        ChannelSpec(count=0)
    with pytest.raises(ValueError, match='channel names'):
        ChannelSpec(count=3, names=['a', 'b'])
    assert ChannelSpec(count=1280).groups == 320
    assert ChannelSpec(count=4).groups == 1
    assert ChannelSpec(count=5).groups == 2


# ────────────────────────────────── field I/O ────────────────────────────────


def test_field_round_trips_through_disk(tmp_path):
    path = str(tmp_path / 'field.npy')
    data = np.random.default_rng(0).random((6, 7, 130), dtype=np.float32)

    meta = FieldMetadata(
        channels=ChannelSpec(count=130, source='test-model'),
        shape=data.shape,
        flight_id='276',
        frame_index=2120,
        engine='moderngl',
    )
    save_field(path, data, meta)

    loaded, loaded_meta = load_field(path)
    np.testing.assert_array_equal(np.asarray(loaded), data)
    assert loaded_meta.channels.count == 130
    assert loaded_meta.channels.source == 'test-model'
    assert loaded_meta.flight_id == '276'
    assert loaded_meta.frame_index == 2120
    assert tuple(loaded_meta.shape) == data.shape


def test_a_field_is_never_saved_without_a_sidecar(tmp_path):
    path = str(tmp_path / 'bare.npy')
    save_field(path, np.zeros((2, 2, 7), dtype=np.float32))

    meta = load_metadata(path)
    assert meta is not None, 'saving must always write a sidecar'
    assert meta.channels.count == 7
    assert json.loads(open(meta_path(path), encoding='utf-8').read())['shape'] == [2, 2, 7]


def test_memmapped_render_writes_straight_to_disk(ctx, tmp_path):
    """The path that makes a 21 GB field renderable on a machine with less RAM than that."""
    path = str(tmp_path / 'big.npy')
    channels = 9
    out = open_field(path, (RESOLUTION.height, RESOLUTION.width, channels))

    renderer = _renderer(ctx)
    try:
        result = render_field_integral(
            renderer, ctx, [_field_shot(_constant_field(channels))] * 2, out=out)
    finally:
        renderer.release()

    save_field(path, out, FieldMetadata(channels=ChannelSpec(count=channels), shape=out.shape))
    del out, result

    reloaded, meta = load_field(path)
    assert reloaded.shape == (RESOLUTION.height, RESOLUTION.width, channels)
    assert meta.channels.count == channels
    # Channel 5 held the constant 5 and two shots contributed, so the sum is 10.
    covered = np.asarray(reloaded)[..., 5] > 0
    np.testing.assert_allclose(np.asarray(reloaded)[..., 5][covered], 10.0, atol=1e-3)


def test_out_shape_is_validated(ctx):
    renderer = _renderer(ctx)
    try:
        with pytest.raises(ValueError, match='expected'):
            render_field_integral(renderer, ctx, [_field_shot(_constant_field(4))],
                                  out=np.zeros((2, 2, 4), dtype=np.float32))
    finally:
        renderer.release()
