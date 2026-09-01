"""The public API must let a caller choose a backend, and honour the choice.

The registry can be correct while the high-level entry points still hard-wire one backend --
which is exactly what happened between the PyTorch port and this merge, when
``make_mgl_context`` silently returned a ``TorchContext``. These tests go through
``ProjectionScene`` and ``make_*_context``, not through the registry.
"""

import numpy as np
import pytest

from alfspy.core.backends import available_engines, backend_for_context, get_backend
from alfspy.render.projection import ProjectionScene, ProjectionSettings
from alfspy.render.render import make_mgl_context, make_torch_context

ENGINES = available_engines()


@pytest.mark.parametrize('engine', ENGINES)
def test_projection_scene_uses_the_requested_engine(engine, small_flight):
    scene = ProjectionScene(
        small_flight.dem_file, small_flight.poses_file, small_flight.correction_file,
        mask_file=small_flight.mask_file, engine=engine,
    )
    try:
        assert backend_for_context(scene.ctx) is get_backend(engine)
    finally:
        scene.release()


@pytest.mark.parametrize('engine', ENGINES)
def test_projection_scene_renders_on_every_engine(engine, small_flight, tmp_path):
    """A backend that is selectable but cannot actually produce an image is not selectable."""
    out = tmp_path / f'ortho_{engine}.png'
    scene = ProjectionScene(
        small_flight.dem_file, small_flight.poses_file, small_flight.correction_file,
        mask_file=small_flight.mask_file, engine=engine,
        settings=ProjectionSettings(),
    )
    try:
        scene.project_orthographic(small_flight.frame_files[0], output_image=str(out))
    finally:
        scene.release()

    assert out.exists() and out.stat().st_size > 0


def test_env_var_selects_the_engine(monkeypatch, small_flight):
    if 'torch' not in ENGINES:
        pytest.skip('torch backend not available')
    monkeypatch.setenv('ALFS_ENGINE', 'torch')
    scene = ProjectionScene(
        small_flight.dem_file, small_flight.poses_file, small_flight.correction_file,
    )
    try:
        assert backend_for_context(scene.ctx) is get_backend('torch')
    finally:
        scene.release()


def test_an_explicit_context_wins_over_engine(small_flight):
    """Passing a context selects its backend; `engine` must not override or fight it."""
    if len(ENGINES) < 2:
        pytest.skip('need two backends')
    other = ENGINES[1]
    ctx = get_backend(other).create_context()
    scene = ProjectionScene(
        small_flight.dem_file, small_flight.poses_file, small_flight.correction_file,
        ctx=ctx, engine=ENGINES[0],
    )
    try:
        assert backend_for_context(scene.ctx) is get_backend(other)
    finally:
        scene.release()
        ctx.release()


@pytest.mark.skipif('moderngl' not in ENGINES, reason='no working OpenGL context')
def test_make_mgl_context_returns_opengl_not_torch():
    """
    The regression this guards: after the PyTorch port, ``make_mgl_context`` was an alias for
    ``make_torch_context``, so code that asked for GL got a tensor rasteriser instead.
    """
    ctx = make_mgl_context()
    try:
        assert backend_for_context(ctx) is get_backend('moderngl')
    finally:
        ctx.release()


def test_make_torch_context_returns_torch():
    ctx = make_torch_context(device='cpu')
    try:
        assert backend_for_context(ctx) is get_backend('torch')
    finally:
        ctx.release()
