from statistics import fmean
from typing import Sequence, Collection, Optional, Literal, Union

import numpy as np
from pyrr import Quaternion, Vector3


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

def get_vector_center(vectors: Sequence[Vector3]) -> Vector3:
    """
    Computes the center point of the given point cloud, i.e., the average vector of the given vector sequence.
    :param vectors: The vector sequence.
    :return: The average vector.
    """
    count = len(vectors)
    vec_sum = np.sum(vectors, axis=0)
    vec_avg = vec_sum / count
    return Vector3(vec_avg)


def vector_project(a: Vector3, b: Vector3) -> Vector3:
    """
    Projects the vector ``a`` onto the vector ``b``.
    :param a: The vector to be projected.
    :param b: The vector to be projected onto.
    :return: A new vector representing the portion of the vector a pointing in the direction of vector b.
    """
    return (a.dot(b) / b.dot(b)) * b


def vector_mean(vectors: Collection[Vector3]) -> Vector3:
    """
    Computes the mean vector of a set of vectors.
    :param vectors: The vectors to aggregate.
    :return: The mean vector.
    """
    x_vals = []
    y_vals = []
    z_vals = []
    for vector in vectors:
        x_vals.append(vector.x)
        y_vals.append(vector.y)
        z_vals.append(vector.z)

    return Vector3((fmean(x_vals), fmean(y_vals), fmean(z_vals)))


def vector_to_geogebra(vec: Vector3, vec_name: Optional[str] = None, decimals: int = 3) -> str:
    """
    Turns a vector into a GeoGebra point definition.
    :param vec: The vector to transform.
    :param vec_name: Name to be assigned to the vector in GeoGebra.
    :param decimals: The amount of decimals each coordinate should be rounded to (defaults to 3).
    :return: A string that defines a point in GeoGebra.
    """
    definition = f'({round(vec.x, decimals)}, {round(vec.y, decimals)}, {round(vec.z, decimals)})'
    assignment = '' if vec_name is None else f'{vec_name} = '
    return assignment + definition


EulerOrder = Union[
    Literal['xyz'], Literal['xzy'],
    Literal['yxz'], Literal['yzx'],
    Literal['zxy'], Literal['zyx'],
]


def quaternion_from_eulers(eulers: Sequence[float], order: EulerOrder = 'xyz', dtype: object = None) -> Quaternion:
    """
    Converts Euler angles to a Quaternion.
    :param eulers: The Euler angles to convert.
    :param order: The Euler order to use (defaults to ``xyz``).
    :param dtype: The data type to use for the Quaternion (optional). If not specified, it is derived from the Euler
    angles.
    :return: The constructed Quaternion.
    """
    eulers = np.asarray(eulers)

    if dtype is None:
        dtype = eulers.dtype

    try:
        roll, pitch, yaw = eulers[0:3]
    except ValueError:
        raise ValueError('Not enough Euler angles to convert')

    qx = Quaternion.from_x_rotation(roll, dtype=dtype)
    qy = Quaternion.from_y_rotation(pitch, dtype=dtype)
    qz = Quaternion.from_z_rotation(yaw, dtype=dtype)

    order = order.lower()
    if order == 'xyz':
        q = qx * qy * qz
    elif order == 'xzy':
        q = qx * qz * qy
    elif order == 'yxz':
        q = qy * qx * qz
    elif order == 'yzx':
        q = qy * qz * qx
    elif order == 'zxy':
        q = qz * qx * qy
    elif order == 'zyx':
        q = qz * qy * qx
    else:
        raise ValueError(f'Invalid rotation order "{order}"')

    return q
