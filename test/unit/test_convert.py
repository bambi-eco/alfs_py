"""Coordinate conversion helpers.

Extends the upstream ``test/core/convert/convert.py`` (which had five ``pass  # TODO``
bodies) with real coverage of the ray-casting and projection paths.

The *quirks* of these functions are pinned separately in
``test_convert_characterisation.py``; this file covers the behaviour that is simply correct.
"""

import math

import numpy as np
import pytest
import trimesh
from pyrr import Quaternion, Vector3

from alfspy.core.convert import (
    adjacent_angle,
    cast_ray,
    change_pixel_origin,
    pixel_to_world_coord,
    undistort_coords,
    world_to_pixel_coord,
)
from alfspy.core.convert.data import Distortion, PixelOrigin
from alfspy.core.rendering import Camera


@pytest.fixture
def plane():
    """A large flat plane at ``z = 0`` to cast rays against."""
    half = 500.0
    vertices = np.array([[-half, -half, 0.0], [half, -half, 0.0],
                         [half, half, 0.0], [-half, half, 0.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return trimesh.Trimesh(vertices=vertices, faces=faces)


# ────────────────────────────── change_pixel_origin ─────────────────────────


def test_change_pixel_origin():
    """Carried over from the upstream suite."""
    for width, height in ((0, 0), (99, 99), (99, 199), (200, 99)):
        width_half = width / 2
        height_half = height / 2
        for in_args, output in (
                ((0, 0, width, height, PixelOrigin.TopLeft, PixelOrigin.TopLeft), (0, 0)),
                ((0, 0, width, height, PixelOrigin.TopLeft, PixelOrigin.TopCenter), (-width_half, 0)),
                ((0, 0, width, height, PixelOrigin.TopLeft, PixelOrigin.TopRight), (width, 0)),
                ((0, 0, width, height, PixelOrigin.TopLeft, PixelOrigin.CenterLeft), (0, height_half)),
                ((0, 0, width, height, PixelOrigin.TopLeft, PixelOrigin.Center), (-width_half, height_half)),
                ((0, 0, width, height, PixelOrigin.TopLeft, PixelOrigin.CenterRight), (width, height_half)),
                ((0, 0, width, height, PixelOrigin.TopLeft, PixelOrigin.BottomLeft), (0, height)),
                ((0, 0, width, height, PixelOrigin.TopLeft, PixelOrigin.BottomCenter), (-width_half, height)),
                ((0, 0, width, height, PixelOrigin.TopLeft, PixelOrigin.BottomRight), (width, height)),
        ):
            assert change_pixel_origin(*in_args) == output


@pytest.mark.parametrize('origin', list(PixelOrigin))
def test_change_pixel_origin_round_trips(origin):
    """Exhaustive round-trip coverage lives in ``test_convert_regressions.py``."""
    x, y = change_pixel_origin(37.0, 21.0, 100, 200, PixelOrigin.TopLeft, origin)
    back_x, back_y = change_pixel_origin(x, y, 100, 200, origin, PixelOrigin.TopLeft)

    assert back_x == pytest.approx(37.0)
    assert back_y == pytest.approx(21.0)


# ──────────────────────────────── adjacent_angle ────────────────────────────


def test_get_cos_angle():
    """Carried over from the upstream suite."""
    pi_fourth = math.pi / 4
    pi_eighth = math.pi / 8
    sin_ratio = np.sin(pi_eighth) / np.sin(pi_fourth)
    for side_length in (0.001, 0.1, 1, 10):
        o_side_length = side_length * sin_ratio
        for in_args, output in (
                ((pi_fourth, o_side_length, side_length), pi_eighth),
                ((pi_eighth, side_length, o_side_length), pi_fourth),
        ):
            assert adjacent_angle(*in_args) == pytest.approx(output)


def test_adjacent_angle_is_zero_for_zero_offset():
    assert adjacent_angle(math.pi / 4, 0.0, 1.0) == pytest.approx(0.0)


# ───────────────────────────────── undistort ────────────────────────────────


def test_undistort_coords_is_identity_without_distortion():
    distortion = Distortion(k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0)

    x, y = undistort_coords([0.1, -0.2], [0.3, 0.4], distortion, None)

    np.testing.assert_allclose(np.atleast_1d(x), [0.1, -0.2], atol=1e-5)
    np.testing.assert_allclose(np.atleast_1d(y), [0.3, 0.4], atol=1e-5)


def test_undistort_coords_pulls_points_inward_for_barrel_distortion():
    distortion = Distortion(k1=0.2, k2=0.0, p1=0.0, p2=0.0, k3=0.0)

    x, y = undistort_coords([0.5], [0.0], distortion, None)

    assert float(np.atleast_1d(x)[0]) < 0.5, 'positive k1 must contract the radius'
    assert float(np.atleast_1d(y)[0]) == pytest.approx(0.0, abs=1e-6)


# ────────────────────────────────── cast_ray ────────────────────────────────


def test_cast_ray_hits_a_plane_straight_down(plane):
    hits = cast_ray(np.array([[3.0, -4.0, 10.0]]), np.array([[0.0, 0.0, -1.0]]), plane)

    assert len(hits) == 1
    np.testing.assert_allclose(np.asarray(hits[0]), [3.0, -4.0, 0.0], atol=1e-5)


def test_cast_ray_reports_misses_as_none(plane):
    hits = cast_ray(np.array([[0.0, 0.0, 10.0]]), np.array([[0.0, 0.0, 1.0]]), plane,
                    include_misses=True)

    assert hits == [None]


def test_cast_ray_can_drop_misses(plane):
    origins = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 10.0]])
    directions = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])

    hits = cast_ray(origins, directions, plane, include_misses=False)

    assert len(hits) == 1


def test_cast_ray_preserves_index_alignment(plane):
    origins = np.array([[0.0, 0.0, 10.0], [1.0, 1.0, 10.0], [2.0, 2.0, 10.0]])
    directions = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])

    hits = cast_ray(origins, directions, plane, include_misses=True)

    assert hits[0] is None
    assert hits[2] is None
    np.testing.assert_allclose(np.asarray(hits[1]), [1.0, 1.0, 0.0], atol=1e-5)


def test_cast_ray_accepts_mesh_data():
    from alfspy.core.rendering import MeshData

    mesh = MeshData(
        vertices=np.array([[-5.0, -5.0, 0.0], [5.0, -5.0, 0.0], [5.0, 5.0, 0.0], [-5.0, 5.0, 0.0]]),
        indices=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
    )

    hits = cast_ray(np.array([[0.0, 0.0, 5.0]]), np.array([[0.0, 0.0, -1.0]]), mesh)

    np.testing.assert_allclose(np.asarray(hits[0]), [0.0, 0.0, 0.0], atol=1e-5)


# ────────────────────────── pixel_to_world_coord (perspective) ──────────────


def test_pixel_to_world_centre_pixel_maps_below_a_downward_camera(plane):
    camera = Camera(fovy=60.0, aspect_ratio=1.0, position=Vector3([0.0, 0.0, 100.0]))

    hits = pixel_to_world_coord([500], [500], 1000, 1000, plane, camera)

    np.testing.assert_allclose(np.asarray(hits[0]), [0.0, 0.0, 0.0], atol=1e-4)


def test_pixel_to_world_footprint_matches_the_field_of_view(plane):
    """A pixel at the frustum edge must land at ``height * tan(fovy / 2)``."""
    fovy = 60.0
    height = 100.0
    camera = Camera(fovy=fovy, aspect_ratio=1.0, position=Vector3([0.0, 0.0, height]))

    hits = pixel_to_world_coord([1000], [500], 1000, 1000, plane, camera)

    expected = height * math.tan(math.radians(fovy) / 2.0)
    assert float(np.asarray(hits[0])[0]) == pytest.approx(expected, rel=1e-4)


def test_pixel_to_world_returns_none_for_empty_input(plane):
    camera = Camera(position=Vector3([0.0, 0.0, 10.0]))
    assert pixel_to_world_coord([], [], 100, 100, plane, camera) == [None]


def test_pixel_to_world_marks_rays_that_miss(plane):
    """A camera below the plane looking down never reaches it."""
    camera = Camera(fovy=60.0, position=Vector3([0.0, 0.0, -100.0]))

    hits = pixel_to_world_coord([500], [500], 1000, 1000, plane, camera, include_misses=True)

    assert hits[0] is None


# ────────────────────────────── world_to_pixel_coord ────────────────────────


def test_world_to_pixel_centres_the_origin_under_an_orthographic_camera():
    camera = Camera(orthogonal=True, orthogonal_size=(100.0, 100.0),
                    position=Vector3([0.0, 0.0, 100.0]))

    xs, ys = world_to_pixel_coord(np.array([[0.0, 0.0, 0.0]]), 1000, 1000, camera,
                                  ensure_int=False)

    assert float(np.atleast_1d(xs)[0]) == pytest.approx(500.0)
    assert float(np.atleast_1d(ys)[0]) == pytest.approx(500.0)


def test_world_to_pixel_scale_matches_the_orthographic_size():
    camera = Camera(orthogonal=True, orthogonal_size=(100.0, 100.0),
                    position=Vector3([0.0, 0.0, 100.0]))

    points = np.array([[0.0, 0.0, 0.0], [25.0, 0.0, 0.0], [0.0, 25.0, 0.0], [-25.0, 0.0, 0.0]])
    xs, ys = world_to_pixel_coord(points, 1000, 1000, camera, ensure_int=False)

    xs = np.atleast_1d(xs)
    ys = np.atleast_1d(ys)

    # 25 m of a 100 m wide view across 1000 px is 250 px.
    assert xs[1] - xs[0] == pytest.approx(250.0)
    assert xs[0] - xs[3] == pytest.approx(250.0)
    assert ys[0] - ys[2] == pytest.approx(250.0), 'world +Y must map to smaller pixel rows'


def test_world_to_pixel_rounds_when_requested():
    camera = Camera(orthogonal=True, orthogonal_size=(100.0, 100.0),
                    position=Vector3([0.0, 0.0, 100.0]))

    xs, ys = world_to_pixel_coord(np.array([[0.13, 0.0, 0.0]]), 1000, 1000, camera,
                                  ensure_int=True)

    assert np.issubdtype(np.asarray(xs).dtype, np.integer)


def test_pixel_world_round_trip_under_an_orthographic_camera(plane):
    """
    The production round trip: source pixels go to world space through the shot camera and
    back to render pixels through the orthographic render camera.
    """
    render_camera = Camera(orthogonal=True, orthogonal_size=(100.0, 100.0),
                           position=Vector3([0.0, 0.0, 100.0]))

    world = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]])
    xs, ys = world_to_pixel_coord(world, 1000, 1000, render_camera, ensure_int=False)

    xs = np.atleast_1d(xs)
    ys = np.atleast_1d(ys)

    np.testing.assert_allclose(xs, [500.0, 600.0, 600.0, 500.0], atol=1e-3)
    np.testing.assert_allclose(ys, [500.0, 500.0, 400.0, 400.0], atol=1e-3)
