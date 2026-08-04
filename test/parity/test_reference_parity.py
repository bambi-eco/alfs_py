"""Opt-in comparison against the reference ModernGL implementation.

**This is the acceptance gate for "the port is equivalent".**

The reference render runs in a *subprocess* under the reference project's own interpreter.
Both projects install a package named ``alfspy``, so they cannot share a process; and each
needs its own dependency set (ModernGL and older numpy there, torch here). Scene inputs and
results are exchanged as ``.npy`` files.

Enable it by pointing at a working ``alfs_py`` checkout::

    set ALFS_REFERENCE_PATH=C:\\D\\Projects\\alfs_py
    pytest test/parity -m parity -v

By default the reference interpreter is ``<ALFS_REFERENCE_PATH>/.venv`` -- override with
``ALFS_REFERENCE_PYTHON``. Run this where the old stack renders *correctly* (Windows or
on-prem Linux), not inside the Xvfb container whose artifacts prompted the migration.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from pyrr import Quaternion, Vector3

from alfspy.core.rendering import (
    CtxShot,
    RenderResultMode,
    Renderer,
    Resolution,
    TextureData,
)
from alfspy.core.torchgl import create_context
from test.helpers.images import image_metrics
from test.helpers.scenes import (
    checkerboard_rgba,
    flat_quad,
    fovy_covering,
    gradient_rgba,
    height_field,
    ortho_camera_above,
)

pytestmark = pytest.mark.parity

RES = 128
DEM_HALF = 25.0
ORTHO = 40.0
CAM_HEIGHT = 30.0
SHOT_HALF = 15.0

# Acceptance thresholds. Exact equality is not achievable: the GL pipeline quantises depth to
# 24 bits, snaps vertices to a fixed-point subpixel grid and blends in the driver's own order.
MAX_MEAN_ABSOLUTE_ERROR = 1.0      # out of 255
MAX_BAD_PIXEL_FRACTION = 0.02      # at most 2 % of pixels may differ by more than 1 LSB


def _reference_python() -> str:
    """
    :return: Path to the interpreter that can run the reference implementation.
    """
    explicit = os.getenv('ALFS_REFERENCE_PYTHON')
    if explicit:
        return explicit

    root = os.getenv('ALFS_REFERENCE_PATH', '')
    for relative in (os.path.join('.venv', 'Scripts', 'python.exe'),
                     os.path.join('.venv', 'bin', 'python')):
        candidate = os.path.join(root, relative)
        if os.path.isfile(candidate):
            return candidate
    return sys.executable


@pytest.fixture(scope='module')
def reference_root() -> str:
    root = os.getenv('ALFS_REFERENCE_PATH')
    if not root:
        pytest.skip('ALFS_REFERENCE_PATH is not set')
    src = os.path.join(root, 'src')
    if not os.path.isdir(src):
        pytest.skip(f'{src} does not exist; ALFS_REFERENCE_PATH must point at an alfs_py checkout')
    return root


def render_reference(reference_root, tmp_path, mesh, texture, shot_image, mode,
                     shot_offsets=(0.0,)):
    """
    Renders a scene with the reference ModernGL implementation.

    :param reference_root: The ``alfs_py`` checkout to run.
    :param tmp_path: Directory for the exchanged ``.npy`` files.
    :param mesh: The mesh to render.
    :param texture: The mesh texture as a float array.
    :param shot_image: The source frame as a float RGBA array.
    :param mode: ``'background'``, ``'shot'`` or ``'integral'``.
    :param shot_offsets: X offsets of the shots when integrating.
    :return: The reference render as an ``(H, W, 4)`` array.
    """
    inputs = {}
    for name, array in (('vertices', mesh.vertices), ('indices', mesh.indices),
                        ('uvs', mesh.uvs), ('texture', texture), ('shot_image', shot_image)):
        path = os.path.join(str(tmp_path), f'{name}.npy')
        np.save(path, array)
        inputs[name] = path

    output = os.path.join(str(tmp_path), f'reference_{mode}.npy')
    spec_path = os.path.join(str(tmp_path), f'spec_{mode}.json')
    with open(spec_path, 'w') as handle:
        json.dump({
            'reference_src': os.path.join(reference_root, 'src'),
            'inputs': inputs,
            'output': output,
            'resolution': RES,
            'ortho_size': ORTHO,
            'camera_height': CAM_HEIGHT,
            'shot_fovy': fovy_covering(SHOT_HALF, CAM_HEIGHT),
            'mode': mode,
            'shot_offsets': list(shot_offsets),
        }, handle)

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_reference_render.py')
    completed = subprocess.run(
        [_reference_python(), script, spec_path],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )

    if completed.returncode != 0:
        message = f'{completed.stdout}\n{completed.stderr}'.strip()
        if 'moderngl' in message.lower() or 'context' in message.lower():
            pytest.skip(f'the reference implementation could not render: {message[-500:]}')
        pytest.fail(f'reference render failed:\n{message}')

    return np.load(output)


def render_port(mesh, texture, shot_image, mode, shot_offsets=(0.0,)):
    """Renders the same scene through the PyTorch backend."""
    ctx = create_context(device='cpu')
    camera = ortho_camera_above(size=(ORTHO, ORTHO), height=CAM_HEIGHT, far=500.0)
    renderer = Renderer(Resolution(RES, RES), ctx, camera, mesh, TextureData(texture))
    fovy = fovy_covering(SHOT_HALF, CAM_HEIGHT)

    try:
        if mode == 'background':
            return renderer.render_background()

        shots = [
            CtxShot(ctx, shot_image, position=Vector3([offset, 0.0, CAM_HEIGHT]),
                    rotation=Quaternion(), fovy=fovy, aspect_ratio=1.0)
            for offset in shot_offsets
        ]

        if mode == 'shot':
            return renderer.project_shots(shots[0], RenderResultMode.ShotOnly)[0]
        if mode == 'integral':
            return renderer.render_integral(shots, auto_contrast=False, alpha_threshold=0.1)
        raise ValueError(f'Unknown mode "{mode}"')
    finally:
        renderer.release()
        ctx.release()


def assert_parity(actual, expected, label):
    metrics = image_metrics(actual, expected, tolerance=1.0)
    assert (metrics['mae'] <= MAX_MEAN_ABSOLUTE_ERROR
            and metrics['bad_fraction'] <= MAX_BAD_PIXEL_FRACTION), (
        f'{label} diverges from the ModernGL reference: '
        f"mae={metrics['mae']:.4f} (limit {MAX_MEAN_ABSOLUTE_ERROR}), "
        f"bad={metrics['bad_fraction']:.4%} (limit {MAX_BAD_PIXEL_FRACTION:.4%}), "
        f"max={metrics['max_abs']:.1f}"
    )


# ──────────────────────────────────── tests ─────────────────────────────────


def test_background_matches_reference(reference_root, tmp_path):
    """
    The plain textured pass. A failure here is a texture-orientation or matrix problem rather
    than a rasterisation one, so check this first when the others fail.
    """
    mesh = height_field(resolution=16, half=DEM_HALF, amplitude=1.5, seed=2)
    texture = checkerboard_rgba()
    shot_image = gradient_rgba(64, 64)

    reference = render_reference(reference_root, tmp_path, mesh, texture, shot_image, 'background')
    port = render_port(mesh, texture, shot_image, 'background')

    assert_parity(port, reference, 'background render')


def test_projected_shot_on_a_flat_dem_matches_reference(reference_root, tmp_path):
    mesh = flat_quad(half=DEM_HALF)
    texture = checkerboard_rgba()
    shot_image = gradient_rgba(64, 64)

    reference = render_reference(reference_root, tmp_path, mesh, texture, shot_image, 'shot')
    port = render_port(mesh, texture, shot_image, 'shot')

    assert_parity(port, reference, 'projected shot on a flat DEM')


def test_projected_shot_on_an_undulating_dem_matches_reference(reference_root, tmp_path):
    mesh = height_field(resolution=24, half=DEM_HALF, amplitude=2.0, seed=5)
    texture = checkerboard_rgba()
    shot_image = gradient_rgba(64, 64)

    reference = render_reference(reference_root, tmp_path, mesh, texture, shot_image, 'shot')
    port = render_port(mesh, texture, shot_image, 'shot')

    assert_parity(port, reference, 'projected shot on an undulating DEM')


def test_light_field_integral_matches_reference(reference_root, tmp_path):
    """The ALFS path itself: several overlapping shots accumulated and normalised."""
    mesh = height_field(resolution=16, half=DEM_HALF, amplitude=1.0, seed=3)
    texture = checkerboard_rgba()
    shot_image = gradient_rgba(64, 64)
    offsets = (-6.0, -2.0, 2.0, 6.0)

    reference = render_reference(reference_root, tmp_path, mesh, texture, shot_image,
                                 'integral', offsets)
    port = render_port(mesh, texture, shot_image, 'integral', offsets)

    assert_parity(port, reference, 'light field integral')
