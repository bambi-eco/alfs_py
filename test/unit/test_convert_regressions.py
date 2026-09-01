"""Regression tests for defects that were fixed during the PyTorch migration.

This file used to *pin* wrong-but-load-bearing behaviour. All of it has since been fixed in
both ``alfs_pytorch`` and ``alfs_py``, so the tests now assert the corrected behaviour and
guard against the old bugs coming back. The docstrings keep the history, because several of
these bugs were invisible in production for reasons worth remembering.

See ``docs/MIGRATION_REVIEW.md`` section 8.
"""

import math

import numpy as np
import pytest
import trimesh
from pyrr import Quaternion, Vector3

from alfspy.core.convert import change_pixel_origin, pixel_to_world_coord, world_to_pixel_coord
from alfspy.core.convert.data import PixelOrigin
from alfspy.core.rendering import Camera


@pytest.fixture
def plane():
    half = 500.0
    vertices = np.array([[-half, -half, 0.0], [half, -half, 0.0],
                         [half, half, 0.0], [-half, half, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


# ──────────────────── 1. ray casting follows the camera ─────────────────────


def test_ray_casting_follows_the_camera_orientation(plane):
    """
    ``pixel_to_world_coord`` used to build rays with ``np.dot(dirs, R33.T)``.

    pyrr is a row-vector library, so ``v @ R33`` is the camera-to-world rotation -- the one
    ``Transform.forward`` uses. Using the transpose made the centre ray point the *opposite*
    way from where the camera was actually facing. It was cancelled downstream by
    ``ProjectionScene`` negating its Euler angles; both halves were fixed together.
    """
    pitch = math.radians(20.0)
    height = 100.0
    camera = Camera(fovy=60.0, aspect_ratio=1.0, position=Vector3([0.0, 0.0, height]),
                    rotation=Quaternion.from_eulers([pitch, 0.0, 0.0]))

    forward = np.asarray(camera.transform.forward, dtype=np.float64)
    assert forward[1] < 0.0, 'setup: the camera should be pitched toward -Y'

    hit = np.asarray(pixel_to_world_coord([500], [500], 1000, 1000, plane, camera)[0])

    # Where a ray along `forward` from the camera meets z = 0.
    expected_y = height * forward[1] / -forward[2]

    assert float(hit[1]) == pytest.approx(expected_y, rel=1e-3), (
        'the centre ray no longer agrees with camera.transform.forward'
    )
    assert float(hit[1]) < 0.0, 'the ray must land on the side the camera faces'


def test_straight_down_camera_hits_directly_below(plane):
    """Why the transposition survived in production: with no tilt it is invisible."""
    camera = Camera(fovy=60.0, aspect_ratio=1.0, position=Vector3([0.0, 0.0, 100.0]))

    hit = np.asarray(pixel_to_world_coord([500], [500], 1000, 1000, plane, camera)[0])

    np.testing.assert_allclose(hit, [0.0, 0.0, 0.0], atol=1e-4)


@pytest.mark.parametrize('yaw_deg', [0.0, 37.0, 180.0, 275.0])
def test_ray_casting_round_trips_through_world_to_pixel(plane, yaw_deg):
    """pixel -> world -> pixel must be the identity for any orientation."""
    camera = Camera(fovy=60.0, aspect_ratio=1.0, position=Vector3([0.0, 0.0, 120.0]),
                    rotation=Quaternion.from_eulers([math.radians(10.0), 0.0,
                                                     math.radians(yaw_deg)]))

    pixels_x = [400, 500, 600, 550]
    pixels_y = [450, 500, 520, 610]

    world = pixel_to_world_coord(pixels_x, pixels_y, 1000, 1000, plane, camera,
                                 include_misses=False)
    assert len(world) == len(pixels_x), 'every ray should hit the plane'

    xs, ys = world_to_pixel_coord(np.asarray(world), 1000, 1000, camera, ensure_int=False)

    np.testing.assert_allclose(np.atleast_1d(xs), pixels_x, atol=0.5)
    np.testing.assert_allclose(np.atleast_1d(ys), pixels_y, atol=0.5)


# ─────────────── 2. world_to_pixel_coord accepts any point count ────────────


@pytest.mark.parametrize('count', [1, 2, 3, 4, 5, 8, 100])
def test_world_to_pixel_coord_accepts_any_point_count(count):
    """
    ``ndc_coords / ndc_coords[:, 3]`` divided an ``(N, 4)`` array by an ``(N,)`` one, which
    only broadcasts at ``N == 1`` or ``N == 4``. Every other count raised ``ValueError``.

    Production only ever passed the four corners of a bounding box, which is the sole reason
    this was not a crash in the field.
    """
    camera = Camera(orthogonal=True, orthogonal_size=(100.0, 100.0),
                    position=Vector3([0.0, 0.0, 100.0]))
    points = np.zeros((count, 3))
    points[:, 0] = np.linspace(-40.0, 40.0, count)

    xs, ys = world_to_pixel_coord(points, 1000, 1000, camera, ensure_int=False)

    assert np.atleast_1d(xs).shape == (count,)
    assert np.atleast_1d(ys).shape == (count,)


def test_world_to_pixel_coord_divides_each_point_by_its_own_w():
    """
    Even at ``N == 4`` the old broadcast computed ``clip[i, j] / w[j]`` instead of ``/ w[i]``,
    so every point was divided by the *first* point's depth and the projection collapsed.

    Orthographic cameras leave every ``w`` at 1.0, so the error cancelled there -- and the
    render camera in ``ProjectionScene`` is always orthographic.
    """
    fovy, height = 60.0, 100.0
    camera = Camera(fovy=fovy, aspect_ratio=1.0, position=Vector3([0.0, 0.0, height]))

    # Same world X, four clearly different depths.
    points = np.array([
        [10.0, 0.0, 0.0],
        [10.0, 0.0, -100.0],
        [10.0, 0.0, -300.0],
        [10.0, 0.0, 50.0],
    ])

    xs, _ = world_to_pixel_coord(points, 1000, 1000, camera, ensure_int=False)
    xs = np.atleast_1d(np.asarray(xs, dtype=float))

    cot_half_fov = 1.0 / math.tan(math.radians(fovy) / 2.0)
    w = height - points[:, 2]
    expected = ((10.0 * cot_half_fov / w) + 1.0) * 500.0

    np.testing.assert_allclose(xs, expected, atol=1e-6)
    # A receding point must move toward the image centre; the old code kept them together.
    assert xs[2] < xs[1] < xs[0] < xs[3]


def test_world_to_pixel_coord_is_correct_for_orthographic_cameras():
    """Depth must not affect an orthographic projection."""
    camera = Camera(orthogonal=True, orthogonal_size=(100.0, 100.0),
                    position=Vector3([0.0, 0.0, 100.0]))

    points = np.array([[0.0, 0.0, 0.0], [25.0, 0.0, -10.0], [0.0, 25.0, -50.0], [-25.0, 0.0, 5.0]])
    xs, ys = world_to_pixel_coord(points, 1000, 1000, camera, ensure_int=False)

    np.testing.assert_allclose(np.atleast_1d(xs), [500.0, 750.0, 500.0, 250.0], atol=1e-3)
    np.testing.assert_allclose(np.atleast_1d(ys), [500.0, 500.0, 250.0, 500.0], atol=1e-3)


# ──────────────────── 3. change_pixel_origin round-trips ────────────────────


@pytest.mark.parametrize('origin', list(PixelOrigin))
def test_change_pixel_origin_round_trips_for_every_origin(origin):
    """
    The old implementation translated first and negated afterwards, so converting to an
    origin whose axes point the other way and back left a residual of ``2 * max * offset``.
    Only ``TopLeft`` and ``TopCenter`` round-tripped. The conversion now routes through
    top-left coordinates and is exact for every pair.
    """
    width, height = 100, 200

    x, y = change_pixel_origin(37.0, 21.0, width, height, PixelOrigin.TopLeft, origin)
    back_x, back_y = change_pixel_origin(x, y, width, height, origin, PixelOrigin.TopLeft)

    assert back_x == pytest.approx(37.0)
    assert back_y == pytest.approx(21.0)


@pytest.mark.parametrize('source', list(PixelOrigin))
@pytest.mark.parametrize('target', list(PixelOrigin))
def test_change_pixel_origin_round_trips_between_arbitrary_origins(source, target):
    width, height = 128, 256
    start_x, start_y = 11.0, 47.0

    x, y = change_pixel_origin(start_x, start_y, width, height, source, target)
    back_x, back_y = change_pixel_origin(x, y, width, height, target, source)

    assert back_x == pytest.approx(start_x)
    assert back_y == pytest.approx(start_y)


# ──────────── 4. the orthographic branch of pixel_to_world_coord ────────────


def test_orthographic_pixel_to_world_spreads_rays_across_the_frustum(plane):
    """
    The branch carried a standing ``# TODO: fix and test`` and had never worked: it evaluated
    ``camera.transform.rotation @ origin_offset``, multiplying a 4-element quaternion by a
    ``(3, N)`` array, and it divided the already normalised NDC offset by ``width``/``height``
    a second time, which would have collapsed every ray onto the camera axis.

    Production was unaffected -- labels are ray-cast through the *source* camera, which is
    perspective -- but the path is now usable.
    """
    size = 100.0
    camera = Camera(orthogonal=True, orthogonal_size=(size, size),
                    position=Vector3([0.0, 0.0, 100.0]))

    # Centre, and the horizontal/vertical extremes.
    hits = pixel_to_world_coord([500, 0, 1000, 500], [500, 500, 500, 0],
                                1000, 1000, plane, camera)

    assert all(hit is not None for hit in hits), 'all rays should hit the plane'
    centre, left, right, top = (np.asarray(h) for h in hits)

    np.testing.assert_allclose(centre, [0.0, 0.0, 0.0], atol=1e-4)
    np.testing.assert_allclose(left[0], -size / 2.0, atol=1e-3)
    np.testing.assert_allclose(right[0], size / 2.0, atol=1e-3)
    np.testing.assert_allclose(top[1], size / 2.0, atol=1e-3)


def test_orthographic_pixel_world_round_trip(plane):
    camera = Camera(orthogonal=True, orthogonal_size=(80.0, 80.0),
                    position=Vector3([5.0, -3.0, 100.0]))

    pixels_x = [100, 400, 700, 512]
    pixels_y = [200, 500, 640, 900]

    world = pixel_to_world_coord(pixels_x, pixels_y, 1024, 1024, plane, camera,
                                 include_misses=False)
    xs, ys = world_to_pixel_coord(np.asarray(world), 1024, 1024, camera, ensure_int=False)

    np.testing.assert_allclose(np.atleast_1d(xs), pixels_x, atol=0.5)
    np.testing.assert_allclose(np.atleast_1d(ys), pixels_y, atol=0.5)
