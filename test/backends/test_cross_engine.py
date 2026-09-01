"""Every backend must render the same picture.

This is the test the merge exists to make possible. Previously the ModernGL and PyTorch
implementations lived in separate repositories that both installed a package called
``alfspy``, so they could not be imported into one interpreter; comparing them meant
shelling out to a second virtualenv and exchanging ``.npy`` files (see ``test/parity``).
Now they are two backends in one process and the comparison is a plain parametrised test.

The comparison is tolerance-based, not exact. A GL driver and a tensor rasteriser break
rasterisation ties differently and round texture filtering differently, so they disagree on
a thin band of edge pixels by construction. What must not differ is the geometry: if a
projection is mirrored, offset, or sampling the wrong part of a shot, the disagreement is
not confined to edges and these bounds catch it.
"""

import numpy as np
import pytest

from alfspy.core.backends import (
    available_engines,
    backend_for_context,
    create_context,
    engine_names,
    get_backend,
    resolve_engine,
)
from test.golden.cases import CASES, render_case
from test.helpers.images import image_metrics

# Per-channel deviation (0-255) tolerated before a value counts as disagreeing, and the
# fraction of values allowed to disagree.
#
# Measured on a 96x96 render, ModernGL vs PyTorch: mean absolute error 0.24-0.85 and
# 0.36%-1.26% of values off by more than 8. The bounds are set just above that, not at a
# comfortable round number -- a threshold ten times the observed disagreement would pass
# through a genuinely broken projection.
TOLERANCE = 8.0
MAX_BAD_FRACTION = 0.03


@pytest.fixture(scope='module')
def engines():
    found = available_engines()
    if len(found) < 2:
        pytest.skip(f'need two backends to compare, found: {found or "none"}')
    return found


@pytest.fixture(scope='module')
def rendered(engines):
    """Every case rendered by every available backend."""
    out = {}
    for name in engines:
        ctx = create_context(engine=name)
        try:
            out[name] = {case: render_case(case, ctx) for case in CASES}
        finally:
            release = getattr(ctx, 'release', None)
            if release is not None:
                release()
    return out


@pytest.mark.parametrize('case', sorted(CASES))
def test_backends_agree(case, engines, rendered):
    reference = engines[0]
    expected = rendered[reference][case]

    for other in engines[1:]:
        actual = rendered[other][case]
        assert actual.shape == expected.shape, (
            f'{case}: {other} produced {actual.shape}, {reference} produced {expected.shape}')

        metrics = image_metrics(actual, expected, tolerance=TOLERANCE)
        assert metrics['bad_fraction'] <= MAX_BAD_FRACTION, (
            f'{case}: {other} disagrees with {reference} beyond an edge band -- '
            f"{metrics['bad_fraction']:.2%} of values off by more than {TOLERANCE} "
            f"(allowed {MAX_BAD_FRACTION:.2%}), mae={metrics['mae']:.3f}, "
            f"max={metrics['max_abs']:.1f}")


@pytest.mark.parametrize('case', sorted(CASES))
def test_backends_agree_on_coverage(case, engines, rendered):
    """
    Coverage is the geometry, stripped of shading.

    A backend can disagree about a pixel's colour for benign reasons, but the set of pixels
    a projection reaches is decided purely by the transforms and the fill rule. This is the
    check that would fail on a mirrored or offset projection.
    """
    reference = engines[0]
    expected = rendered[reference][case][..., 3] > 0

    for other in engines[1:]:
        actual = rendered[other][case][..., 3] > 0
        # Measured disagreement between ModernGL and PyTorch is at most 0.01% of pixels.
        disagreement = float((actual != expected).mean())
        assert disagreement <= 0.005, (
            f'{case}: {other} and {reference} cover different pixels '
            f'({disagreement:.2%} differ). That is a geometry difference, not a shading one.')


def test_every_registered_engine_is_importable_or_reports_why():
    """A registered engine must either work or fail with a message naming its extra."""
    for name in engine_names():
        try:
            get_backend(name)
        except ImportError as exc:
            assert f'AlfsPy[{name}]' in str(exc), (
                f'{name}: ImportError should name the extra that provides it, got: {exc}')


def test_context_round_trips_to_its_own_backend(engines):
    for name in engines:
        ctx = create_context(engine=name)
        try:
            assert backend_for_context(ctx) is get_backend(name)
        finally:
            ctx.release()


def test_explicit_engine_beats_the_environment(monkeypatch):
    monkeypatch.setenv('ALFS_ENGINE', 'torch')
    assert resolve_engine('moderngl') == 'moderngl'
    assert resolve_engine() == 'torch'
    monkeypatch.delenv('ALFS_ENGINE')
    assert resolve_engine() == 'moderngl'


def test_unknown_engine_is_rejected():
    with pytest.raises(ValueError, match='Unknown render engine'):
        get_backend('vulkan-but-not-yet')
