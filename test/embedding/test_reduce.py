"""Reducing an N-channel field to something viewable.

These need only numpy and scikit-learn -- no model, no GPU -- so they are the part of the
embedding pipeline that can be checked properly everywhere.

Two behaviours are pinned that the original implementation did not have, both of which matter
for a *sequence* rather than a single frame:

* Uncovered pixels are excluded from the fit. A rendered field is zero wherever no shot
  reached, and those zeros are usually the plurality of pixels. Fitting on them drags the
  principal axes towards a point mass that carries no information.
* A fit can be reused. Refitting per frame gives every frame its own basis, so the same
  terrain changes colour frame to frame and the sequence flickers.
"""

import numpy as np
import pytest

from alfspy.embedding.reduce import FieldReducer, reduce_to_2d, reduce_to_rgb

try:
    import sklearn  # noqa: F401

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import umap  # noqa: F401

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# The reducers are an optional extra, so an environment without them must skip these rather
# than fail them.
pytestmark = pytest.mark.skipif(
    not HAS_SKLEARN,
    reason='scikit-learn is not installed; `pip install "AlfsPy[embedding]"`')


@pytest.fixture
def field_and_mask():
    """
    A field with two distinguishable 'terrain types' and a large uncovered border.

    The two populations sit far apart along one random direction, so a correct reduction has
    to separate them in its first component and a fit polluted by the uncovered zeros will
    not.
    """
    rng = np.random.default_rng(0)
    height, width, channels = 24, 32, 64

    field = np.zeros((height, width, channels), dtype=np.float32)
    covered = np.zeros((height, width), dtype=bool)
    covered[4:20, 4:28] = True

    upper = np.zeros((height, width), dtype=bool)
    upper[4:12, 4:28] = True

    basis = rng.standard_normal((2, channels)).astype(np.float32) * 4.0
    for group, vector in ((covered & upper, basis[0]), (covered & ~upper, basis[1])):
        noise = rng.standard_normal((int(group.sum()), channels)).astype(np.float32) * 0.05
        field[group] = vector + noise

    return field, covered, upper


def test_reduce_to_rgb_shape_and_dtype(field_and_mask):
    field, covered, _ = field_and_mask
    image = reduce_to_rgb(field, mask=covered)
    assert image.shape == (*field.shape[:2], 3)
    assert image.dtype == np.uint8


def test_uncovered_pixels_stay_black(field_and_mask):
    field, covered, _ = field_and_mask
    image = reduce_to_rgb(field, mask=covered)
    assert (image[~covered] == 0).all()


def test_the_two_populations_separate(field_and_mask):
    """The reduction has to preserve the structure that is actually in the field."""
    field, covered, upper = field_and_mask
    image = reduce_to_rgb(field, mask=covered)

    a = image[covered & upper].mean(axis=0)
    b = image[covered & ~upper].mean(axis=0)
    assert np.abs(a - b).max() > 200, (
        f'the two terrain types should land at opposite ends of a component, got {a} vs {b}')


def test_the_fit_uses_only_the_masked_pixels(field_and_mask):
    """
    The mask's precise meaning: fitting a field with a mask must equal fitting the selected
    pixels on their own. Anything else means uncovered pixels are influencing the basis.
    """
    field, covered, _ = field_and_mask

    masked = FieldReducer('pca', n_components=3).fit(field, mask=covered)
    subset = field[covered].reshape(1, -1, field.shape[-1])
    direct = FieldReducer('pca', n_components=3).fit(subset)

    np.testing.assert_allclose(
        masked.transform(field, mask=covered)[covered],
        direct.transform(subset)[0],
        atol=1e-4)


def test_masking_changes_the_result(field_and_mask):
    """
    Passing the mask is not a no-op. Note the effect is *not* better separation -- PCA finds
    the dominant direction either way -- it is that uncovered pixels stop being treated as
    observations, and stop being given a colour of their own.
    """
    field, covered, _ = field_and_mask

    masked = reduce_to_rgb(field, mask=covered)
    unmasked = reduce_to_rgb(field)

    assert not np.array_equal(masked, unmasked)
    # Without the mask the uncovered region is fitted like any other pixel and comes out
    # with a confident-looking false colour rather than as absent data.
    assert (unmasked[~covered] != 0).any()
    assert (masked[~covered] == 0).all()


def test_reduce_to_2d(field_and_mask):
    field, covered, _ = field_and_mask
    reduced = reduce_to_2d(field, mask=covered)
    assert reduced.shape == (*field.shape[:2], 2)
    assert reduced.dtype == np.float32
    assert np.allclose(reduced[~covered], 0.0)


def test_a_fitted_reducer_gives_a_stable_mapping(field_and_mask):
    """
    The sequence case: one reducer applied to several fields must map the same descriptor to
    the same colour every time.
    """
    field, covered, _ = field_and_mask
    reducer = FieldReducer('pca', n_components=3).fit(field, mask=covered)

    first = reducer.to_image(reducer.transform(field, mask=covered), mask=covered)
    second = reducer.to_image(reducer.transform(field, mask=covered), mask=covered)
    np.testing.assert_array_equal(first, second)

    # A second field with the same content but a different uncovered border must still map
    # its covered pixels to the same colours.
    shifted = field.copy()
    reused = reducer.to_image(reducer.transform(shifted, mask=covered), mask=covered)
    np.testing.assert_array_equal(first, reused)


def test_refitting_per_field_is_what_reuse_avoids(field_and_mask):
    """
    Demonstrates the flicker: an independent fit on a scaled field produces a different
    mapping, while a reused fit does not.
    """
    field, covered, _ = field_and_mask
    scaled = field * 3.0

    reducer = FieldReducer('pca', n_components=3).fit(field, mask=covered)
    reused = reducer.to_image(reducer.transform(scaled, mask=covered), mask=covered)
    refitted = reduce_to_rgb(scaled, mask=covered)

    assert not np.array_equal(reused, refitted)


def test_transform_before_fit_is_rejected(field_and_mask):
    field, covered, _ = field_and_mask
    with pytest.raises(RuntimeError, match='Fit the reducer'):
        FieldReducer().transform(field, mask=covered)


def test_tsne_reports_that_it_cannot_be_reused(field_and_mask):
    """
    t-SNE has no out-of-sample extension, so a fit genuinely cannot be applied elsewhere.
    Saying so is better than silently refitting and producing a different mapping.
    """
    field, covered, _ = field_and_mask
    reducer = FieldReducer('tsne', n_components=2)
    assert not reducer.reusable

    reducer.fit(field, mask=covered)
    with pytest.raises(RuntimeError, match='no out-of-sample transform'):
        reducer.transform(field, mask=covered)


def test_pca_is_reusable():
    assert FieldReducer('pca').reusable


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match='Unknown reduction method'):
        FieldReducer('magic')


def test_empty_mask_is_rejected(field_and_mask):
    field, _, _ = field_and_mask
    empty = np.zeros(field.shape[:2], dtype=bool)
    with pytest.raises(ValueError, match='Nothing to fit on'):
        FieldReducer().fit(field, mask=empty)


def test_mask_shape_is_validated(field_and_mask):
    field, _, _ = field_and_mask
    with pytest.raises(ValueError, match='entries but the field has'):
        FieldReducer().fit(field, mask=np.ones((3, 3), dtype=bool))


def test_non_field_input_is_rejected():
    with pytest.raises(ValueError, match=r'Expected an \(H, W, C\) field'):
        FieldReducer().fit(np.zeros((4, 4)))


def test_round_trips_through_files(tmp_path, field_and_mask):
    """The convenience path: read a rendered field from disk, write a PNG."""
    field, covered, _ = field_and_mask
    source = tmp_path / 'field.npy'
    np.save(source, field)

    png = tmp_path / 'reduced.png'
    image = reduce_to_rgb(str(source), mask=covered, output_file=str(png))
    assert png.exists() and png.stat().st_size > 0
    assert image.shape == (*field.shape[:2], 3)

    npy = tmp_path / 'reduced_2d.npy'
    reduce_to_2d(str(source), mask=covered, output_file=str(npy))
    assert np.load(npy).shape == (*field.shape[:2], 2)


@pytest.mark.slow
@pytest.mark.skipif(not HAS_UMAP, reason='umap-learn not installed')
def test_umap_reduces_and_is_reusable(field_and_mask):
    field, covered, _ = field_and_mask
    reducer = FieldReducer('umap', n_components=3, n_neighbors=5, min_dist=0.1,
                           random_state=0)
    assert reducer.reusable

    reduced = reducer.fit_transform(field, mask=covered)
    assert reduced.shape == (*field.shape[:2], 3)
    assert np.allclose(reduced[~covered], 0.0)

    again = reducer.transform(field, mask=covered)
    assert again.shape == reduced.shape
