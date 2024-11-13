from typing import Union, Sequence

import numpy as np
from numpy.typing import NDArray
from pyrr import Vector3

from alfspy.core.geo import AABB


def get_aabb(points: Union[Sequence[Vector3], NDArray]) -> AABB:
    """
    Determines the smallest convex AABB for the given points.
    :param points: The vertices.
    :return: An AABB.
    """
    if isinstance(points[0], Vector3):
        points = np.stack(points)

    max_x, max_y, max_z = np.max(points, axis=0)
    min_x, min_y, min_z = np.min(points, axis=0)
    max_p = Vector3((max_x, max_y, max_z))
    min_p = Vector3((min_x, min_y, min_z))

    return AABB(min_p, max_p)


def get_center(vertices: NDArray) -> tuple[Vector3, AABB]:
    """
    Computes the center position and AABB of a set of vertices.
    :param vertices: A numpy array containing vertices.
    :return: A tuple containing the center of the vertices and the two points defining the vertices AABB.
    """
    aabb = get_aabb(vertices)
    return aabb.center, aabb