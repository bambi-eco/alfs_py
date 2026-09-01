"""Deterministic render cases pinned as golden images.

These fixtures were captured from the ModernGL renderer as it behaved *before* the
multi-backend refactor. They are the safety net for that refactor: every backend
(ModernGL, PyTorch, Vulkan) must reproduce them, and any change to a golden has to be a
deliberate, reviewed re-baseline rather than a silent drift.

The scenes are built in memory from :mod:`test.helpers.scenes` rather than read from disk,
so a capture depends only on this file and the renderer -- no dataset, no file I/O, no
encode/decode round-trip to perturb the pixels.

Each case is a callable taking a context and returning a ``uint8`` RGBA array.
"""

from typing import Callable, Dict

import numpy as np
from numpy.typing import NDArray
from pyrr import Quaternion, Vector3

from alfspy.core.backends.moderngl_ import CtxShot, Renderer
from alfspy.core.rendering.data import RenderResultMode, Resolution, TextureData
from test.helpers.scenes import (
    checkerboard_rgba,
    fovy_covering,
    gradient_rgba,
    height_field,
    ortho_camera_above,
    perspective_camera_above,
)

__all__ = ['CASES', 'RESOLUTION', 'reset_state', 'render_case']

# Small enough that the fixtures stay cheap to store and diff, large enough that a fill-rule
# or orientation regression is visible rather than averaged away.
RESOLUTION = Resolution(96, 96)

DEM_HALF = 25.0
DEM_AMPLITUDE = 3.0
CAMERA_HEIGHT = 30.0
SHOT_HEIGHT = 30.0

# The virtual camera sees the whole DEM; the shots see roughly half of it, so their
# footprints overlap partially and the integral has a non-trivial coverage pattern.
VIEW_FOVY = fovy_covering(DEM_HALF, CAMERA_HEIGHT)
SHOT_FOVY = fovy_covering(DEM_HALF * 0.6, SHOT_HEIGHT)

# Offsets of the three shots contributing to the integral, in world units.
SHOT_OFFSETS = ((-6.0, -4.0), (0.0, 0.0), (7.0, 5.0))


def _dem():
    """The DEM under test: a non-flat height field, i.e. many small shared-edge triangles."""
    return height_field(resolution=12, half=DEM_HALF, amplitude=DEM_AMPLITUDE, seed=7)


def _dem_texture() -> TextureData:
    return TextureData(checkerboard_rgba(tiles=8, tile_size=12))


def _shot_texture() -> NDArray:
    # A gradient makes an accidental flip or channel swap obvious; a checkerboard would not.
    return gradient_rgba(width=48, height=48)


def _view_camera():
    return perspective_camera_above(fovy=VIEW_FOVY, aspect_ratio=1.0, height=CAMERA_HEIGHT)


def _make_shot(ctx, dx: float, dy: float) -> CtxShot:
    return CtxShot(
        ctx,
        _shot_texture(),
        Vector3([dx, dy, SHOT_HEIGHT]),
        Quaternion(),
        fovy=SHOT_FOVY,
        aspect_ratio=1.0,
    )


def reset_state(ctx) -> None:
    """
    Puts the context into the pipeline's standard state.

    ``Renderer.render_integral`` disables ``DEPTH_TEST`` on the shared context and never
    re-enables it, so without this a case would depend on which cases ran before it.
    """
    import moderngl as mgl

    ctx.enable(mgl.DEPTH_TEST)
    ctx.enable(mgl.CULL_FACE)
    ctx.cull_face = 'back'
    ctx.disable(mgl.BLEND)


def _case_background(ctx) -> NDArray:
    """The textured DEM alone, from the virtual camera."""
    renderer = Renderer(RESOLUTION, ctx, _view_camera(), _dem(), _dem_texture())
    try:
        return renderer.render_background()
    finally:
        renderer.release()


def _case_shot_only(ctx) -> NDArray:
    """A single shot projected onto the DEM, without the background."""
    renderer = Renderer(RESOLUTION, ctx, _view_camera(), _dem(), _dem_texture())
    shot = _make_shot(ctx, 0.0, 0.0)
    try:
        results = renderer.project_shots(shot, RenderResultMode.ShotOnly)
        return results[0]
    finally:
        shot.release()
        renderer.release()


def _case_shot_complete(ctx) -> NDArray:
    """A single shot composited over the DEM background."""
    renderer = Renderer(RESOLUTION, ctx, _view_camera(), _dem(), _dem_texture())
    shot = _make_shot(ctx, 3.0, -2.0)
    try:
        results = renderer.project_shots(shot, RenderResultMode.Complete)
        return results[0]
    finally:
        shot.release()
        renderer.release()


def _case_integral_3shots(ctx) -> NDArray:
    """
    The ALFS integral of three overlapping shots.

    ``auto_contrast`` is off: it is a host-side post-process whose global min/max would make
    the fixture sensitive to a single outlying pixel, which is not what this case is pinning.
    """
    renderer = Renderer(RESOLUTION, ctx, _view_camera(), _dem(), _dem_texture())
    shots = [_make_shot(ctx, dx, dy) for dx, dy in SHOT_OFFSETS]
    try:
        return renderer.render_integral(shots, auto_contrast=False, alpha_threshold=0.1)
    finally:
        for shot in shots:
            shot.release()
        renderer.release()


def _case_orthographic(ctx) -> NDArray:
    """A shot projected under an orthographic virtual camera."""
    camera = ortho_camera_above(size=(2 * DEM_HALF, 2 * DEM_HALF), height=CAMERA_HEIGHT)
    renderer = Renderer(RESOLUTION, ctx, camera, _dem(), _dem_texture())
    shot = _make_shot(ctx, 0.0, 0.0)
    try:
        results = renderer.project_shots(shot, RenderResultMode.Complete)
        return results[0]
    finally:
        shot.release()
        renderer.release()


CASES: Dict[str, Callable[[object], NDArray]] = {
    'background': _case_background,
    'shot_only': _case_shot_only,
    'shot_complete': _case_shot_complete,
    'integral_3shots': _case_integral_3shots,
    'orthographic': _case_orthographic,
}


def render_case(name: str, ctx) -> NDArray:
    """
    Renders one case in a known context state.

    :param name: The case name, a key of :data:`CASES`.
    :param ctx: The render context to use.
    :return: The rendered image as a ``uint8`` RGBA array.
    """
    reset_state(ctx)
    return np.ascontiguousarray(CASES[name](ctx))
