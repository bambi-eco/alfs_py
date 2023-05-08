from statistics import fmean
from typing import Union, Optional, Iterable, Any, Collection, Sequence

import cv2
import numpy as np
from numpy import ndarray
from numpy.typing import NDArray
from pyrr import Vector3

from src.core.rendering.data import MeshData
from src.core.geo.aabb import AABB
from src.core.defs import Color


def get_first_valid(in_dict: dict[Any], keys: Iterable[Any], default: Optional[Any] = None) -> Any:
    """
    Returns the value within a dict associated with the first valid key of an iterable key source
    :param in_dict: The dictionary to search within
    :param keys: An iterable source of keys
    :param default: The value to be returned when no valid key was passed (optional)
    :return: The value of default if no valid key was passed; otherwise the value associated with the first valid key
    """
    for key in keys:
        if key in in_dict:
            return in_dict.get(key)
    return default


def compare_color(col_a: Union[NDArray, Iterable, int, float], col_b: Union[NDArray, Iterable, int, float]) -> bool:
    """
    Compares two color values
    :param col_a: The first color
    :param col_b: The second color
    :return: ``False`` if any component of the two colors differs; otherwise ``True``
    """
    if not isinstance(col_a, ndarray):
        col_a = np.array(col_a)
    if not isinstance(col_b, ndarray):
        col_b = np.array(col_b)
    return (col_a == col_b).all()



def get_aabb(points: Union[Collection[Vector3], NDArray]) -> AABB:
    """
    Determines the smallest convex AABB for the given points
    :param points: The vertices
    :return: An AABB
    """
    if not isinstance(points, np.ndarray):
        points = np.stack(points)

    max_x, max_y, max_z = np.max(points, axis=0)
    min_x, min_y, min_z = np.min(points, axis=0)
    max_p = Vector3((max_x, max_y, max_z))
    min_p = Vector3((min_x, min_y, min_z))

    return AABB(min_p, max_p)

def get_center(vertices: NDArray) -> tuple[Vector3, AABB]:
    """
    Computes the center position and AABB of a set of vertices
    :param vertices: A numpy array containing vertices
    :return: A tuple containing the center of the vertices and the two points defining the vertices AABB
    """
    aabb = get_aabb(vertices)
    return aabb.center, aabb

def get_vector_center(vectors: Sequence[Vector3]) -> Vector3:
    """
    Computes the center point of the given point cloud, i.e., the average vector of the given vector sequence
    :param vectors: The vector sequence
    :return: The average vector
    """
    count = len(vectors)
    vec_sum = np.sum(vectors, axis=0)
    vec_avg = vec_sum / count
    return Vector3(vec_avg)


def make_plane(size: float = 1.0, y: float = 0.0) -> MeshData:
    """
    Creates an axis aligned plane of the given size
    :param size: Side length of the plane
    :param y: Value to apply to the Y component of all vertices
    :return: A ``MeshData`` object representing the generated plane
    """
    size_h = size / 2.0
    vertices = np.array([[-size_h, y, size_h], [-size_h, y, -size_h],
                         [size_h, y, -size_h], [size_h, y, size_h]])
    indices = np.array([0, 1, 2, 2, 3, 0])
    uvs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    return MeshData(vertices, indices, uvs)


def make_quad() -> MeshData:
    """
    :return: Mesh data representing a quad covering the entire screen in a deferred shading scenario
    """
    vertices = np.array([[-1.0, 1.0, 0.0], [-1.0, -1.0, 0.0],
                         [1.0, -1.0, 0.0], [1.0, 1.0, 0.0]])

    uvs = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    indices = np.array([0, 1, 2, 1, 2, 3])

    return MeshData(vertices=vertices, indices=indices, uvs=uvs)


def int_up(val: float) -> int:
    """
    Rounds a float up and casts it to an integer
    :param val: The value to round up
    :return: The rounded up integer
    """
    return int(val + 0.5)


def is_same(obj: Any, other: Any) -> bool:
    """
    Checks whether two objects are instances of the same class
    :param obj: The first object to check
    :param other: The object whose type should be used for the check
    :return: ``True`` if `obj` is an instance of the same type as `other`; otherwise ``False``
    """
    return isinstance(obj, type(other))


def gen_checkerboard_tex(tile_per_side: int, tile_size: int, tile_color: Color, non_tile_color: Color,
                         dtype: object = float) -> NDArray:
    """
    Generates a numpy array representing a checkerboard texture
    :param tile_per_side: The amount of tiles per side
    :param tile_size: The side length of a tile in pixels
    :param tile_color: The color of a tile
    :param non_tile_color: The color of the lack of a tile
    :param dtype: The dtype to convert all values within the numpy to (defaults to float)
    :return: A numpy array representing a checkerboard texture
    """
    t_c = np.array(tile_color).reshape((-1))
    nt_c = np.array(non_tile_color).reshape((-1))

    if t_c.shape != nt_c.shape:
        raise ValueError('Both given colors have to have the same amount of components')

    if tile_per_side == 0 or tile_size == 0:
        return np.empty((0, 0, t_c.shape[0]), dtype=dtype)

    depth = t_c.shape[0]
    tile = np.empty((tile_size, tile_size, depth), dtype=dtype)
    n_tile = np.empty_like(tile)
    tile[..., :] = t_c
    n_tile[..., :] = nt_c

    odd_line = cv2.hconcat([tile if i % 2 == 0 else n_tile for i in range(0, tile_per_side)])
    even_line = cv2.hconcat([tile if i % 2 == 1 else n_tile for i in range(0, tile_per_side)])
    result = cv2.vconcat([odd_line if i % 2 == 0 else even_line for i in range(0, tile_per_side)])

    return result


def vector_project(a: Vector3, b: Vector3) -> Vector3:
    """
    Projects the vector ``a`` onto the vector b
    :param a: The vector to be projected
    :param b: The vector to be projected onto
    :return: A new vector representing the portion of the vector a pointing in the direction of vector b
    """
    return (a.dot(b) / b.dot(b)) * b


def vector_mean(vectors: Collection[Vector3]) -> Vector3:
    """
    Computes the mean vector of a set of vectors
    :param vectors: The vectors to aggregate
    :return: The mean vector
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
    Turns a vector into a GeoGebra point definition
    :param vec: The vector to transform
    :param vec_name: Name to be assigned to the vector in GeoGebra
    :param decimals: The amount of decimals each coordinate should be rounded to (defaults to 3)
    :return: A string that defines a point in GeoGebra
    """
    definition = f'({round(vec.x, decimals)}, {round(vec.y, decimals)}, {round(vec.z, decimals)})'
    assignment = '' if vec_name is None else f'{vec_name} = '
    return assignment + definition
