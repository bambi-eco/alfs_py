"""Geometry, transform and camera maths.

These are ModernGL-independent and were carried over unchanged from ``alfs_py``; the tests
exist so the port has a baseline that proves nothing drifted.
"""

import math

import numpy as np
import pytest
from pyrr import Matrix44, Quaternion, Vector3

from alfspy.core.geo import AABB, Frustum, Quad, Transform, Triangle
from alfspy.core.rendering import Camera
from alfspy.core.util.geo import get_aabb, get_center
from alfspy.core.util.pyrrs import quaternion_from_eulers, vector_mean, vector_project


# ─────────────────────────────────── Transform ──────────────────────────────


def test_transform_defaults_to_identity():
    transform = Transform()

    np.testing.assert_allclose(np.asarray(transform.position), [0, 0, 0])
    np.testing.assert_allclose(np.asarray(transform.scale), [1, 1, 1])
    np.testing.assert_allclose(np.asarray(transform.mat()), np.eye(4), atol=1e-6)


def test_transform_applies_scale_then_rotation_then_translation():
    transform = Transform(
        position=Vector3([1.0, 2.0, 3.0]),
        rotation=Quaternion.from_y_rotation(math.pi / 2),
        scale=Vector3([2.0, 2.0, 2.0]),
    )

    point = np.array([1.0, 0.0, 0.0, 1.0])
    result = point @ np.asarray(transform.mat(), dtype=np.float64)

    # scale -> (2,0,0); rotate about y -> (0,0,2); translate -> (1,2,5)
    np.testing.assert_allclose(result[:3], [1.0, 2.0, 5.0], atol=1e-5)


def test_transform_basis_vectors_are_orthonormal():
    transform = Transform(rotation=quaternion_from_eulers([0.3, -0.7, 1.1]))

    forward = np.asarray(transform.forward, dtype=np.float64)
    up = np.asarray(transform.up, dtype=np.float64)
    right = np.asarray(transform.right, dtype=np.float64)

    for vector in (forward, up, right):
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)

    assert np.dot(forward, up) == pytest.approx(0.0, abs=1e-5)
    assert np.dot(forward, right) == pytest.approx(0.0, abs=1e-5)
    assert np.dot(up, right) == pytest.approx(0.0, abs=1e-5)


def test_transform_default_forward_points_down_negative_z():
    np.testing.assert_allclose(np.asarray(Transform().forward), [0.0, 0.0, -1.0], atol=1e-6)


def test_transform_round_trips_through_dict():
    original = Transform(Vector3([1.5, -2.0, 0.25]), quaternion_from_eulers([0.1, 0.2, 0.3]))

    restored = Transform.from_dict(original.to_dict())

    np.testing.assert_allclose(np.asarray(restored.position), np.asarray(original.position), atol=1e-6)
    np.testing.assert_allclose(np.asarray(restored.rotation), np.asarray(original.rotation), atol=1e-6)


def test_transform_translation_helpers():
    transform = Transform()
    transform.move_forward(5.0)

    np.testing.assert_allclose(np.asarray(transform.position), [0.0, 0.0, 5.0], atol=1e-6)


# ──────────────────────────────── Frustum / Camera ──────────────────────────


def test_orthographic_projection_maps_half_extent_to_half_ndc():
    frustum = Frustum(orthogonal=True, orthogonal_size=(10.0, 10.0), near=0.1, far=100.0)
    proj = np.asarray(frustum.get_proj(), dtype=np.float64)

    point = np.array([2.5, 0.0, 0.0, 1.0])
    clip = point @ proj

    assert clip[0] / clip[3] == pytest.approx(0.5)


def test_perspective_projection_places_a_point_on_the_frustum_edge():
    fovy = 90.0
    frustum = Frustum(fovy=fovy, aspect_ratio=1.0, near=0.1, far=100.0)
    proj = np.asarray(frustum.get_proj(), dtype=np.float64)

    # At depth -d the frustum half-height is d * tan(fovy / 2) = d for fovy = 90 degrees.
    point = np.array([0.0, 5.0, -5.0, 1.0])
    clip = point @ proj

    assert clip[1] / clip[3] == pytest.approx(1.0, abs=1e-6)


def test_camera_fovy_is_zero_when_orthographic():
    camera = Camera(fovy=60.0, orthogonal=True)
    assert camera.fovy == 0.0
    assert camera.fovx == 0.0


def test_camera_fovx_follows_aspect_ratio():
    camera = Camera(fovy=40.0, aspect_ratio=2.0)
    assert camera.fovx == pytest.approx(80.0)

    camera.fovx = 40.0
    assert camera.fovy == pytest.approx(20.0)


def test_camera_view_matrix_translates_by_negative_distance():
    camera = Camera(position=Vector3([0.0, 0.0, 5.0]))
    view = np.asarray(camera.get_view(), dtype=np.float64)

    origin = np.array([0.0, 0.0, 0.0, 1.0]) @ view
    assert origin[2] == pytest.approx(-5.0, abs=1e-5)


def test_camera_round_trips_through_dict():
    camera = Camera(fovy=42.0, aspect_ratio=1.5, orthogonal=True, orthogonal_size=(8.0, 4.0),
                    near=0.5, far=250.0, position=Vector3([1.0, 2.0, 3.0]))

    restored = Camera.from_dict(camera.to_dict())

    assert restored is not None
    assert restored.orthogonal is True
    assert restored.orthogonal_size == (8.0, 4.0)
    assert restored.near == pytest.approx(0.5)
    assert restored.far == pytest.approx(250.0)
    np.testing.assert_allclose(np.asarray(restored.transform.position), [1.0, 2.0, 3.0], atol=1e-6)


def test_camera_look_at_orients_forward_towards_the_target():
    camera = Camera(position=Vector3([0.0, 0.0, 10.0]))
    camera.look_at(Vector3([0.0, 0.0, 0.0]))

    np.testing.assert_allclose(np.asarray(camera.transform.forward), [0.0, 0.0, -1.0], atol=1e-5)


def test_frustum_captures_points_inside_and_rejects_points_outside():
    frustum = Frustum(orthogonal=True, orthogonal_size=(10.0, 10.0), near=0.1, far=100.0,
                      transform=Transform(Vector3([0.0, 0.0, 10.0])))

    assert frustum.captures(Vector3([0.0, 0.0, 0.0]))
    assert not frustum.captures(Vector3([50.0, 0.0, 0.0]))


def test_frustum_corners_widen_with_depth_for_a_perspective_camera():
    frustum = Frustum(fovy=90.0, aspect_ratio=1.0, near=1.0, far=10.0)

    near_corners = np.stack([np.asarray(c) for c in frustum.corners_at(1.0)])
    far_corners = np.stack([np.asarray(c) for c in frustum.corners_at(10.0)])

    near_width = near_corners[:, 0].max() - near_corners[:, 0].min()
    far_width = far_corners[:, 0].max() - far_corners[:, 0].min()

    assert far_width > near_width


# ──────────────────────────────────── AABB ──────────────────────────────────


def test_get_aabb_bounds_a_point_cloud():
    points = np.array([[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]])

    aabb = get_aabb(points)

    np.testing.assert_allclose(np.asarray(aabb.p_min), [-1.0, -2.0, -3.0])
    np.testing.assert_allclose(np.asarray(aabb.p_max), [4.0, 5.0, 6.0])
    assert aabb.width == pytest.approx(5.0)
    assert aabb.height == pytest.approx(7.0)
    assert aabb.depth == pytest.approx(9.0)
    np.testing.assert_allclose(np.asarray(aabb.center), [1.5, 1.5, 1.5])


def test_get_aabb_accepts_a_sequence_of_vectors():
    aabb = get_aabb([Vector3([1.0, 1.0, 1.0]), Vector3([-1.0, -1.0, -1.0])])
    np.testing.assert_allclose(np.asarray(aabb.center), [0.0, 0.0, 0.0], atol=1e-6)


def test_get_center_returns_centre_and_aabb():
    centre, aabb = get_center(np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]))

    np.testing.assert_allclose(np.asarray(centre), [1.0, 1.0, 1.0])
    assert isinstance(aabb, AABB)


def test_aabb_has_eight_corners():
    aabb = AABB(Vector3([0.0, 0.0, 0.0]), Vector3([1.0, 1.0, 1.0]))
    assert len(aabb.corners) == 8


# ─────────────────────────── primitives and pyrr utils ──────────────────────


def test_triangle_normal_is_unit_length_and_perpendicular():
    triangle = Triangle(Vector3([0.0, 0.0, 0.0]), Vector3([1.0, 0.0, 0.0]), Vector3([0.0, 1.0, 0.0]))

    normal = np.asarray(triangle.normal, dtype=np.float64)

    np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1e-6)


def test_quad_is_flat_when_coplanar():
    quad = Quad(Vector3([0.0, 0.0, 0.0]), Vector3([1.0, 0.0, 0.0]),
                Vector3([1.0, 1.0, 0.0]), Vector3([0.0, 1.0, 0.0]))
    assert quad.flat


def test_quad_is_not_flat_when_a_corner_is_lifted():
    quad = Quad(Vector3([0.0, 0.0, 0.0]), Vector3([1.0, 0.0, 0.0]),
                Vector3([1.0, 1.0, 5.0]), Vector3([0.0, 1.0, 0.0]))
    assert not quad.flat


@pytest.mark.parametrize('order', ['xyz', 'zyx', 'yxz'])
def test_quaternion_from_eulers_is_normalised(order):
    quaternion = quaternion_from_eulers([0.3, 0.4, 0.5], order)
    assert np.linalg.norm(np.asarray(quaternion, dtype=np.float64)) == pytest.approx(1.0, abs=1e-6)


def test_quaternion_from_eulers_rejects_unknown_order():
    with pytest.raises(ValueError):
        quaternion_from_eulers([0.0, 0.0, 0.0], 'abc')


def test_euler_order_changes_the_result():
    xyz = np.asarray(quaternion_from_eulers([0.5, 0.6, 0.7], 'xyz'), dtype=np.float64)
    zyx = np.asarray(quaternion_from_eulers([0.5, 0.6, 0.7], 'zyx'), dtype=np.float64)

    assert not np.allclose(xyz, zyx)


def test_vector_project_onto_axis():
    result = vector_project(Vector3([3.0, 4.0, 0.0]), Vector3([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(np.asarray(result), [3.0, 0.0, 0.0], atol=1e-6)


def test_vector_mean():
    result = vector_mean([Vector3([0.0, 0.0, 0.0]), Vector3([2.0, 4.0, 6.0])])
    np.testing.assert_allclose(np.asarray(result), [1.0, 2.0, 3.0], atol=1e-6)
