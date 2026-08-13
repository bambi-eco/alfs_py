"""Absolute pointing of a drone-pose camera.

Every other test in this suite is *round-trip* consistent -- it renders through a
pose and un-projects with the same construction -- so a camera that points the
wrong way in world space passes them all. These tests pin the absolute direction.

The defect they guard against: composing the pose as
``quaternion_from_eulers([tilt, roll, heading], 'zyx')`` turns the heading into a
rotation about the camera's *own* optical axis. The tilt then always leans the same
way in world space and the heading merely spins the image. At nadir that is
indistinguishable from the correct result; at the horizon it is 128 degrees wrong.
"""
import numpy as np
import pytest
from pyrr import Vector3

from alfspy.core.geo.transform import Transform
from alfspy.core.util.pyrrs import quaternion_from_drone_pose


def forward(tilt, roll, heading):
    """World-space forward vector of a camera at pose ``[tilt, roll, heading]``."""
    quat = quaternion_from_drone_pose([tilt, roll, heading])
    vec = np.asarray(
        Transform(Vector3([0.0, 0.0, 0.0]), quat, None).forward, dtype=float)
    return vec / np.linalg.norm(vec)


def expected(tilt, heading):
    """ENU direction for a gimbal tilted ``tilt`` off nadir on a given heading."""
    t, h = np.radians(tilt), np.radians(heading)
    return np.array([np.sin(t) * np.sin(h), np.sin(t) * np.cos(h), -np.cos(t)])


def angle_between(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


@pytest.mark.parametrize("tilt,heading", [
    (0.0, 0.0), (0.0, 51.6), (0.0, 317.0),          # nadir, any heading
    (45.0, 0.0), (45.0, 90.0), (45.0, 180.0),       # oblique, cardinal headings
    (90.0, 51.6), (76.4, 50.0), (12.5, 317.0),      # the lion flight's range
    (30.0, -140.0), (80.0, 200.0),                  # negative / wrapped headings
])
def test_camera_points_where_the_gimbal_points(tilt, heading):
    assert angle_between(forward(tilt, 0.0, heading),
                         expected(tilt, heading)) < 1e-4


def test_nadir_ignores_heading():
    """Looking straight down, every heading gives the same direction."""
    for heading in (0.0, 90.0, 217.0):
        assert np.allclose(forward(0.0, 0.0, heading), [0.0, 0.0, -1.0],
                           atol=1e-6)


def test_heading_steers_the_tilt():
    """The regression: these two poses must NOT produce the same direction."""
    north = forward(45.0, 0.0, 0.0)
    east = forward(45.0, 0.0, 90.0)
    assert angle_between(north, east) == pytest.approx(60.0, abs=1e-3)
    assert north[0] == pytest.approx(0.0, abs=1e-6)   # due north: no east
    assert east[1] == pytest.approx(0.0, abs=1e-6)    # due east: no north


def test_horizon_is_horizontal():
    for heading in (0.0, 51.6, 190.0):
        assert forward(90.0, 0.0, heading)[2] == pytest.approx(0.0, abs=1e-6)


def test_heading_is_clockwise_from_north():
    """Heading 90 must look east, not west."""
    assert forward(90.0, 0.0, 90.0)[0] == pytest.approx(1.0, abs=1e-6)


def test_tilt_is_measured_from_nadir():
    """Tilt 45 must be halfway between straight down and the horizon."""
    assert forward(45.0, 0.0, 0.0)[2] == pytest.approx(-np.sqrt(0.5), abs=1e-6)


def test_degrees_in_and_wrapping():
    """Headings beyond 360 and below 0 behave the same as their wrapped value."""
    assert np.allclose(forward(30.0, 0.0, 380.0), forward(30.0, 0.0, 20.0),
                       atol=1e-6)
    assert np.allclose(forward(30.0, 0.0, -90.0), forward(30.0, 0.0, 270.0),
                       atol=1e-6)
