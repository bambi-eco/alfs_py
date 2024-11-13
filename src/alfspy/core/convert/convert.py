from collections import defaultdict
from typing import Optional, Union, Sequence, Callable, Final

import cv2
import numpy as np
import trimesh
from numpy.typing import NDArray, ArrayLike
from pyrr import Vector3
from trimesh import Trimesh

from alfspy.core.convert.data import PixelOrigin, Distortion
from alfspy.core.rendering.camera import Camera
from alfspy.core.rendering.data import MeshData
from alfspy.core.util.defs import Number

_DEFAULT_POT_FUNC: Final[Callable[[], tuple[float, float, bool, bool]]] = lambda: (0, 0, True, False)
_POT_LOOKUP: dict[PixelOrigin, tuple[float, float, bool, bool]] = defaultdict(_DEFAULT_POT_FUNC, {
        PixelOrigin.TopLeft: (0.0, 0.0, True, False),
        PixelOrigin.TopCenter: (0.5, 0.0, True, False),
        PixelOrigin.TopRight: (1.0, 0.0, False, False),
        PixelOrigin.CenterLeft: (0.0, 0.5, True, True),
        PixelOrigin.Center: (0.5, 0.5, True, True),
        PixelOrigin.CenterRight: (1.0, 0.5, False, True),
        PixelOrigin.BottomLeft: (0.0, 1.0, True, True),
        PixelOrigin.BottomCenter: (0.5, 1.0, True, True),
        PixelOrigin.BottomRight: (1.0, 1.0, False, True),
    }
)
"""
A static constant dictionary expressing the behaviour of different origins.
The tuple of an origin describes it's deviation from the top left origin in percent in x and y direction, whether
the x-axis points left and whether the y-axis points up.
This information allows a generic definition of origin conversion.
"""


def change_pixel_origin(x: ArrayLike, y: ArrayLike, max_x: Number, max_y: Number, from_origin: PixelOrigin,
                        to_origin: PixelOrigin) -> tuple[ArrayLike, ArrayLike]:
    """
    Converts the given pixel coordinates from one common image origin to another.
    :param x: The x-coordinate of the pixel.
    :param y: The y-coordinate of the pixel.
    :param max_x: The maximum index along the x-axis.
    :param max_y: The maximum index along the y-axis.
    :param from_origin: The origin of the passed coordinate.
    :param to_origin: The origin to convert the coordinates to.
    :return: A tuple containing converted coordinates ``(x, y)``.
    """
    lookup = _POT_LOOKUP
    x_tf, y_tf, p_xa_tf, p_ya_tf = lookup[from_origin]
    x_tt, y_tt, p_xa_tt, p_ya_tt = lookup[to_origin]

    x_tr = x_tt - x_tf
    y_tr = y_tt - y_tf

    n_x = x - max_x * x_tr
    n_y = y - max_y * y_tr

    if p_xa_tf != p_xa_tt:
        n_x *= -1.0

    if p_ya_tf != p_ya_tt:
        n_y *= -1.0

    return n_x, n_y


def undistort_coords(x: ArrayLike, y: ArrayLike, distortion: Distortion,
                     camera_matrix: Optional[NDArray]) -> tuple[ArrayLike, ArrayLike]:
    """
    Removes distortions from 2D coordinates.
    :param x: The x-coordinates.
    :param y: The y-coordinates.
    :param distortion: The distortion to be removed.
    :param camera_matrix: The camera matrix to be used (optional). If not specified an identity matrix will be used.
    :return: The undistorted coordinates.
    """
    if camera_matrix is None:
        camera_matrix = np.identity(3)

    dst_coords = np.column_stack((x, y)).reshape(-1, 1, 2).astype(np.float32)
    dst = distortion.to_array(dtype=np.float32)
    norm_coords = cv2.undistortPoints(dst_coords, camera_matrix, dst)

    norm_x = np.squeeze(norm_coords[..., 0])
    norm_y = np.squeeze(norm_coords[..., 1])
    return norm_x, norm_y


def adjacent_angle(ref_angle: float, offset: ArrayLike, ref_offset: float) -> float:
    """
    This function calculates the angle of a right-angled triangle, which is nested within a reference right-angled
    triangle whose angle is known. Hence, both triangles share the same adjacent, but have opposite sides of different
    length.
    .. code-block::
        Consider the following example:

        |                    o ... the opposite side length of the different triangle
        |\\                 ro ... the opposite side of the reference triangle
        ||\\  ra             a ... the angle to calculate, i.e., the return value
        || \\               ra ... the angle of the reference triangle
        | \\ \\
        |  | \\
        |  |  \\
        | a \\  \\
        |    |  \\
        |    |   \\
        |     \\   \\
        |      |   \\
        |      |    \\
        |_______\\____\\
        |       |    |
        |<- o ->|    |
        |<--- ro --->|

    :param ref_angle: The angle of the reference triangle with a known angle.
    :param offset: The opposite side of the smaller triangle.
    :param ref_offset: The opposite side of the bigger triangle.
    :return: The angle of the smaller triangle.
    """
    return np.arcsin((offset * np.sin(ref_angle)) / ref_offset)


def cast_ray(ray_origins: Union[Vector3, Sequence[Vector3]], ray_directions: Union[Vector3, Sequence[Vector3]],
             mesh: Union[MeshData, Trimesh], include_misses: bool = True) -> Sequence[Optional[ArrayLike]]:
    """
    Casts a ray to find an intersection with the given mesh.
    :param ray_origins: One or more vectors describing the ray origins.
    :param ray_directions: One or more vectors describing the ray directions from their origin.
    :param mesh: The mesh to intersect with.
    :param include_misses: Whether non-intersecting rays should be included in the result via ``None`` values (default
    value is ``True``). If set to ``True``, the result indices correspond to the indices of a rays  origin or direction.
    :return: If there were no hits ``None``; otherwise the coordinate of the first hit.
    """
    if not isinstance(mesh, Trimesh):
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.indices)

    origins = np.array(ray_origins).reshape((-1, 3))
    directions = np.array(ray_directions).reshape((-1, 3))

    hits, ray_indices, *_ = mesh.ray.intersects_location(origins, directions, multiple_hits=False)

    if include_misses:
        if isinstance(ray_origins, Vector3):
            res_count = 1
        elif isinstance(ray_origins, np.ndarray):
            res_count = ray_origins.shape[0]
        else:
            res_count = len(ray_origins)

        res_dict = {ray_indices[i]: hits[i] for i in range(len(hits))}
        res = [res_dict.get(i, None) for i in range(res_count)]
    else:
        res = hits

    return res


def world_to_pixel_coord(coordinates: Union[ArrayLike, Sequence[ArrayLike]],
                         width: int, height: int, camera: Camera, ensure_int: bool = True,
                         viewport_origin: PixelOrigin = PixelOrigin.TopLeft) -> tuple[ArrayLike, ArrayLike]:
    """
    Converts world coordinates into pixel coordinates by projecting them onto a camera.
    :param coordinates: The coordinates to convert.
    :param width: The width of the camera's render resolution.
    :param height: The height of the camera's render resolution.
    :param camera: The camera to be used for projecting the coordinate.
    :param ensure_int: Whether the output coordinates should be rounded to the next integer (defaults to ``True``).
    :param viewport_origin: The origin of the viewport (defaults to top left).
    :return: The pixel coordinates of the projected world coordinates.
    """
    coordinates = np.reshape(coordinates, (-1, 3))
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    z = coordinates[:, 2]

    homo_coords = np.stack((x, y, z, np.ones_like(x)), axis=-1)
    camera_coords = np.dot(homo_coords, camera.get_view())
    ndc_coords = np.dot(camera_coords, camera.get_proj())
    ndc_coords_norm = ndc_coords / ndc_coords[:, 3]

    # convert normalized coordinates to top-left oriented pixel coordinates
    pixel_xs = (ndc_coords_norm[:, 0] + 1.0) * width / 2.0
    pixel_ys = height - (ndc_coords_norm[:, 1] + 1) / 2.0 * height

    if viewport_origin != PixelOrigin.TopLeft:
        pixel_xs, pixel_ys = change_pixel_origin(pixel_xs, pixel_ys, width, height, PixelOrigin.TopLeft, viewport_origin)

    if ensure_int:
        pixel_xs = np.floor(pixel_xs + 0.5).astype(int)
        pixel_ys = np.floor(pixel_ys + 0.5).astype(int)

    if len(coordinates.shape) <= 1:
        pixel_xs = pixel_xs[0]
        pixel_ys = pixel_ys[0]

    return pixel_xs, pixel_ys


def pixel_to_world_coord(x: ArrayLike, y: ArrayLike, width: int, height: int, mesh: Union[MeshData, Trimesh],
                         camera: Camera, distortion: Optional[Distortion] = None,
                         camera_matrix: Optional[NDArray] = None) -> Sequence[Optional[ArrayLike]]:
    """
    Converts pixel coordinates to 3D world coordinates by projecting them from a camera onto a mesh. The viewport origin
    is assumed to be in the top left corner of the image, with the positive x-axis pointing to the right and the y-axis
    pointing down.
    :param x: The x-coordinates of the pixels to convert.
    :param y: The y-coordinates of the pixels to convert.
    :param width: The total width of the respective viewport.
    :param height: The total height of the respective viewport.
    :param mesh: The mesh to project the coordinates onto; Its surface represents the set of all possible world
    coordinate outputs. Passing and reusing the same ``Trimesh`` instance can lead to a significant performance boost
    compared to passing the raw mesh data or a new instance for every conversion.
    :param camera: The camera to project the coordinates through. In the case of reversing a render projection, this
    camera should be equal to the camera used for rendering.
    :param distortion: The distortion to be removed from the given pixel coordinates (optional). If specified, the pixel
    coordinates will be remapped according to counteract the passed distortion.
    :param camera_matrix: The camera matrix to be used for removing distortion (optional). If not specified when a
    distortion is given, an identity matrix will be used.
    :return: If the given pixel coordinates were invalid or did not result in a hit ``None```; otherwise the 3D world
    coordinates associated with the given pixel.
    """
    x = np.array(x)
    y = np.array(y)

    count = len(x)
    if count == 0:
        return [None]

    if distortion is not None:
        x, y = undistort_coords(x, y, distortion, camera_matrix)

    # convert pixel to normalized coordinates
    ndc_x = (2.0 * x / width) - 1.0
    ndc_y = 1.0 - (2.0 * y / height)

    if camera.orthogonal:
        # TODO: fix and test
        # the ray direction is parallel to the camera forward vector
        ray_dirs = np.array([camera.transform.forward] * count)

        # the ray origin lies on the same plane as the camera but is offset by the pixels distance on the image plane
        img_plane_width, img_plane_height = camera.orthogonal_size
        origin_offset_x = (ndc_x * img_plane_width) / width
        origin_offset_y = (ndc_y * img_plane_height) / height
        origin_offset = np.array([origin_offset_x, origin_offset_y, np.zeros_like(x)])
        ray_origins = camera.transform.position + camera.transform.rotation @ origin_offset
    else:  # camera has a perspective projection
        # the ray direction is the projection of the pixel coordinates onto the image plane
        tan_fov_y = np.tan(np.deg2rad(camera.fovy) / 2.0)
        n_ray_dirs = np.stack((
            ndc_x * camera.aspect_ratio * tan_fov_y,  # account for fov x via aspect ratio (more efficient)
            ndc_y * tan_fov_y,
            np.full_like(x, -1.0)  # the camera looks along the negative z-axis
        ), axis=-1)

        ray_dirs = np.dot(n_ray_dirs, camera.transform.rotation.matrix33.T)

        # the ray origin is the same as the camera origin
        ray_origins = np.array([camera.transform.position] * count)

    res = cast_ray(ray_origins, ray_dirs, mesh)
    return res
