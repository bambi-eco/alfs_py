"""Pins the rendered output against fixtures captured before the multi-backend refactor.

Every other test in the suite checks a property (a coverage fraction, an analytic world
position, an invertibility round-trip). Those pass just as happily against a renderer that
has silently changed its orientation, fill rule or channel order. These tests compare whole
images, which is the only check that catches that class of regression.

Regenerate with ``python -m test.golden.capture`` -- but only when a change in output is
intended, and review the diff when you do.
"""

import os

import numpy as np
import pytest

from test.golden.cases import CASES, render_case
from test.helpers.images import assert_images_close

_HERE = os.path.dirname(os.path.abspath(__file__))

# GPUs differ in rasterisation tie-breaking and texture-filter rounding, so a handful of
# edge pixels may legitimately differ by a small amount even for an unchanged renderer.
# These bounds are tight enough to catch a real change and loose enough not to flake.
TOLERANCE = 2.0        # per-channel, 0-255
MAX_BAD_FRACTION = 0.02  # at most 2% of values may exceed it


@pytest.fixture(scope='module')
def gl_ctx():
    """A ModernGL context shared by every golden case, skipped if no GL is available."""
    pytest.importorskip('moderngl')
    try:
        from alfspy.core.backends.moderngl_ import create_context
        ctx = create_context()
    except Exception as exc:  # pragma: no cover - depends on the machine, not the code
        pytest.skip(f'no OpenGL context available: {type(exc).__name__}: {exc}')
    yield ctx
    ctx.release()


def _golden(name: str) -> np.ndarray:
    path = os.path.join(_HERE, f'{name}.npy')
    if not os.path.exists(path):
        pytest.fail(f'missing golden fixture {path}; run `python -m test.golden.capture`')
    return np.load(path)


@pytest.mark.gl
@pytest.mark.parametrize('name', sorted(CASES))
def test_render_matches_golden(name, gl_ctx):
    expected = _golden(name)
    actual = render_case(name, gl_ctx)

    assert actual.shape == expected.shape, (
        f'{name}: shape changed {expected.shape} -> {actual.shape}')
    assert actual.dtype == expected.dtype, (
        f'{name}: dtype changed {expected.dtype} -> {actual.dtype}')

    assert_images_close(
        actual, expected,
        tolerance=TOLERANCE,
        max_bad_fraction=MAX_BAD_FRACTION,
        label=f'golden {name!r} (regenerate with `python -m test.golden.capture` '
              f'if this change is intended)')


@pytest.mark.gl
def test_cases_are_distinguishable(gl_ctx):
    """
    Guards the fixtures themselves: if two cases rendered the same thing, the suite would
    look green while testing far less than it appears to.
    """
    rendered = {name: _golden(name) for name in CASES}
    names = sorted(rendered)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not np.array_equal(rendered[a], rendered[b]), (
                f'golden fixtures {a!r} and {b!r} are identical')
