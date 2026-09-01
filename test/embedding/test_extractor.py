"""DINOv3 descriptor extraction, and the whole embedded-light-field pipeline end to end.

These run only when the model is already in the local HuggingFace cache. They never trigger a
download: a test suite that fetches an 840M-parameter checkpoint on a cold CI runner is a
test suite people turn off. Warm the cache first if you want them::

    huggingface-cli download facebook/dinov3-vits16-pretrain-lvd1689m

``ALFS_DINOV3_MODEL`` selects which model to use; it defaults to the small 384-dimensional
variant rather than the 1280-dimensional ViT-H+/16 the production pipeline uses, because the
things worth asserting here -- grid geometry, token layout, dtype, the end-to-end join with
the renderer -- do not depend on descriptor quality.
"""

import os

import numpy as np
import pytest
from pyrr import Quaternion, Vector3

MODEL = os.environ.get('ALFS_DINOV3_MODEL', 'facebook/dinov3-vits16-pretrain-lvd1689m')


def _model_is_cached(model_id: str) -> bool:
    """
    :param model_id: A HuggingFace model id.
    :return: Whether its config is already in the local cache, i.e. whether constructing an
        extractor will hit the network.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    if os.path.isdir(model_id):
        return True
    return all(isinstance(try_to_load_from_cache(model_id, name), str)
               for name in ('config.json', 'preprocessor_config.json', 'model.safetensors'))


try:
    from alfspy.embedding.extractor import DinoV3Extractor, is_available

    HAS_DEPS = is_available()
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(
    not (HAS_DEPS and _model_is_cached(MODEL)),
    reason=f'{MODEL} is not in the local HuggingFace cache; download it to run these',
)


@pytest.fixture(scope='module')
def extractor():
    # local_files_only because the DINOv3 repos are gated: with the weights cached but no
    # token in the environment, transformers still probes the hub for optional config files
    # and gets a 401.
    return DinoV3Extractor(MODEL, device='cpu', local_files_only=True)


@pytest.fixture(scope='module')
def frame():
    """A structured frame -- gradients plus a bright square, so patches actually differ."""
    rng = np.random.default_rng(0)
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[..., 0] = np.linspace(0, 255, 128, dtype=np.uint8)[None, :]
    img[..., 1] = np.linspace(0, 255, 128, dtype=np.uint8)[:, None]
    img[32:80, 32:80] = 255
    return np.clip(img + rng.integers(-8, 8, img.shape), 0, 255).astype(np.uint8)


def test_reports_its_geometry(extractor):
    assert extractor.embed_dim > 0
    assert extractor.patch_size > 0
    # One CLS token plus the register tokens. The original implementation hard-coded 5; this
    # reads it from the config, so a model with a different layout is handled rather than
    # silently misaligned by a few tokens.
    assert extractor.prefix_tokens >= 1


def test_grid_is_the_image_divided_by_the_patch_size(extractor, frame):
    embedding = extractor.extract(frame)
    patch = extractor.patch_size
    assert embedding.shape == (frame.shape[0] // patch, frame.shape[1] // patch,
                               extractor.embed_dim)
    assert embedding.dtype == np.float32


def test_descriptors_are_not_degenerate(extractor, frame):
    """A grid of identical or zero descriptors would satisfy every shape assertion."""
    embedding = extractor.extract(frame)
    flat = embedding.reshape(-1, embedding.shape[-1])
    assert np.isfinite(flat).all()
    assert flat.std() > 1e-3
    assert np.unique(flat.round(3), axis=0).shape[0] > 1


def test_the_bright_square_is_distinguishable(extractor, frame):
    """
    Descriptors have to reflect the image. The centre square differs from the corners, so
    their mean descriptors must too.
    """
    embedding = extractor.extract(frame)
    rows, cols = embedding.shape[:2]

    centre = embedding[rows // 3:2 * rows // 3, cols // 3:2 * cols // 3].reshape(-1, embedding.shape[-1])
    corner = embedding[:rows // 4, :cols // 4].reshape(-1, embedding.shape[-1])

    def unit(v):
        v = v.mean(axis=0)
        return v / np.linalg.norm(v)

    similarity = float(unit(centre) @ unit(corner))
    assert similarity < 0.99, (
        f'centre and corner descriptors are nearly identical (cosine {similarity:.4f}); '
        'the extractor is probably not reading real patch tokens')


def test_single_extraction_is_deterministic(extractor, frame):
    np.testing.assert_array_equal(extractor.extract(frame), extractor.extract(frame))


def test_batch_matches_single_extraction(extractor, frame):
    """
    Batched and single results agree to float32 batch-reduction noise. They are not bitwise
    equal -- a batched matmul sums in a different order -- so this pins the tolerance rather
    than pretending it is exact.
    """
    single = extractor.extract(frame)
    batched = extractor.extract_batch([frame, frame])

    for result in batched:
        assert np.abs(result - single).max() < 1e-4


def test_batch_preserves_input_order_across_sizes(extractor, frame):
    """
    Different sizes are grouped into separate batches for throughput, so the results have to
    be put back in the caller's order rather than the grouping's.
    """
    small = frame[:96, :96]
    results = extractor.extract_batch([frame, small, frame])

    patch = extractor.patch_size
    assert results[0].shape[0] == frame.shape[0] // patch
    assert results[1].shape[0] == small.shape[0] // patch
    assert results[2].shape[0] == frame.shape[0] // patch
    np.testing.assert_allclose(results[0], results[2], atol=1e-4)


def test_accepts_grayscale_and_bgra(extractor, frame):
    """Thermal frames arrive single-channel and masks arrive with alpha."""
    grey = frame[..., 0]
    bgra = np.dstack([frame, np.full(frame.shape[:2], 255, dtype=np.uint8)])

    patch = extractor.patch_size
    expected = (frame.shape[0] // patch, frame.shape[1] // patch, extractor.embed_dim)
    assert extractor.extract(grey).shape == expected
    assert extractor.extract(bgra).shape == expected


def test_non_multiple_sizes_are_resized_to_whole_patches(extractor):
    """A 100x100 frame is not a whole number of 16px patches; it must still work."""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    embedding = extractor.extract(img)
    patch = extractor.patch_size
    assert embedding.shape[0] == round(100 / patch)
    assert embedding.shape[1] == round(100 / patch)


# ───────────────────────── the pipeline, end to end ──────────────────────────


@pytest.mark.slow
def test_embedded_light_field_end_to_end(extractor, tmp_path):
    """
    Frames -> descriptors -> integrated field -> reduced image, through the real model.

    This is the join the whole feature exists for, and the one the original prototype got
    wrong in three separate ways.
    """
    from alfspy.core.backends import available_engines, create_context
    from alfspy.core.rendering import Renderer
    from alfspy.core.rendering.data import Resolution
    from alfspy.embedding.reduce import FieldReducer
    from alfspy.io.field import ChannelSpec, FieldMetadata, load_field, save_field
    from alfspy.render.field import FieldShot, render_field_integral
    from test.helpers.scenes import fovy_covering, height_field, perspective_camera_above

    engines = available_engines()
    if not engines:
        pytest.skip('no render backend available')

    rng = np.random.default_rng(1)
    frames = []
    for _ in range(3):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[..., 0] = np.linspace(0, 255, 64, dtype=np.uint8)[None, :]
        img[16:48, 16:48] = 255
        frames.append(np.clip(img + rng.integers(-6, 6, img.shape), 0, 255).astype(np.uint8))

    fields = extractor.extract_batch(frames)
    assert {f.shape for f in fields} == {(64 // extractor.patch_size,
                                          64 // extractor.patch_size,
                                          extractor.embed_dim)}

    half, height = 20.0, 40.0
    shots = [
        FieldShot(field=field,
                  position=Vector3([dx, 0.0, height]), rotation=Quaternion(),
                  fovy=fovy_covering(half * 0.7, height), aspect_ratio=1.0)
        for field, dx in zip(fields, (-4.0, 0.0, 4.0))
    ]

    ctx = create_context(engine=engines[0])
    renderer = Renderer(Resolution(64, 64), ctx,
                        perspective_camera_above(fovy=fovy_covering(half, height),
                                                 height=height),
                        height_field(resolution=8, half=half, amplitude=0.0))
    try:
        result = render_field_integral(renderer, ctx, shots)
    finally:
        renderer.release()
        ctx.release()

    assert result.accum.shape == (64, 64, extractor.embed_dim)
    covered = result.coverage > 0
    assert covered.any()

    averaged = result.normalised(threshold=0.1)

    # The prototype's signature failure: every fourth channel came back constant because it
    # had been overwritten by the coverage counter.
    constant = [c for c in range(extractor.embed_dim)
                if np.ptp(averaged[..., c][covered]) < 1e-9]
    assert not constant, f'{len(constant)} channels are constant across covered pixels'

    # And it must round-trip through disk with its metadata.
    path = str(tmp_path / 'embedded.npy')
    save_field(path, averaged,
               FieldMetadata(channels=ChannelSpec(count=extractor.embed_dim,
                                                  source=extractor.model_id),
                             shape=averaged.shape))
    reloaded, meta = load_field(path)
    assert meta.channels.source == extractor.model_id
    assert reloaded.shape == averaged.shape

    rgb = FieldReducer('pca', n_components=3).fit_transform(np.asarray(reloaded), mask=covered)
    assert rgb.shape == (64, 64, 3)
    assert np.isfinite(rgb).all()
