import numpy as np
from pyrr import Quaternion


def rand_quaternion(*, min_x: float = 0.0, max_x: float = 360.0,
                    min_y: float = 0.0, max_y: float = 360.0,
                    min_z: float = 0.0, max_z: float = 360.0) -> Quaternion:
    """
    Creates a random quaternion using random Euler degree angles in the XYZ order.
    :param min_x: The minimum pitch / x Euler value in degrees (defaults to ``0.0``).
    :param max_x: The maximum pitch / x Euler value in degrees (defaults to ``360.0``).
    :param min_y: The minimum roll / y Euler value in degrees (defaults to ``0.0``).
    :param max_y: The maximum roll / y Euler value in degrees (defaults to ``360.0``).
    :param min_z: The minimum yaw / z Euler value in degrees (defaults to ``0.0``).
    :param max_z: The maximum yaw / z Euler value in degrees (defaults to ``360.0``).
    :return: A random quaternion.
    """
    x_angle = np.deg2rad(np.random.uniform(min_x, max_x))
    y_angle = np.deg2rad(np.random.uniform(min_y, max_y))
    z_angle = np.deg2rad(np.random.uniform(min_z, max_z))

    q = Quaternion.from_eulers([x_angle, y_angle, z_angle])
    return q