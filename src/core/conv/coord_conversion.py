from collections import defaultdict
from functools import cache
from typing import Optional, Union, Sequence

import cv2
import numpy as np
import trimesh
from numpy.typing import NDArray, ArrayLike
from pyrr import Vector3, Vector4
from trimesh import Trimesh

from src.core.conv.data import PixelOrigin, Distortion
from src.core.rendering.camera import Camera
from src.core.rendering.data import MeshData
from src.core.util.basic import nearest_int


@cache
def _po_t_lookup() -> dict[PixelOrigin, tuple[float, float, bool, bool]]:
    # return tuple indicates the percentage change relative to top left in x and y direction
    # and whether the x-axis and the y-axis point in the positive direction
    res: dict[PixelOrigin, tuple[float, float, bool, bool]] = defaultdict(lambda: (0.0, 0.0, True, True))

    res[PixelOrigin.TopLeft] = (0.0, 0.0, True, False)
    res[PixelOrigin.TopCenter] = (0.5, 0.0, True, False)
    res[PixelOrigin.TopRight] = (1.0, 0.0, False, False)

    res[PixelOrigin.CenterLeft] = (0.0, 0.5, True, True)
    res[PixelOrigin.Center] = (0.5, 0.5, True, True)
    res[PixelOrigin.CenterRight] = (1.0, 0.5, False, True)

    res[PixelOrigin.BottomLeft] = (0.0, 1.0, True, True)
    res[PixelOrigin.BottomCenter] = (0.5, 1.0, True, True)
    res[PixelOrigin.BottomRight] = (1.0, 1.0, False, True)

    return res


def cvt_pixel_origin(x: ArrayLike, y: ArrayLike, img_width, img_height, from_origin: PixelOrigin,
                     to_origin: PixelOrigin) -> tuple[ArrayLike, ArrayLike]:
    """
    Converts the given pixel coordinates from one common image origin to another.
    :param x: The x-coordinate of the pixel
    :param y: The y-coordinate of the pixel
    :param img_width: The width of the image
    :param img_height: The height of the image
    :param from_origin: The origin of the given coordinates
    :param to_origin: The origin the given coordinates should be converted to
    :return: The converted coordinates
    """

    lookup = _po_t_lookup()
    x_tf, y_tf, p_xa_tf, p_ya_tf = lookup[from_origin]
    x_tt, y_tt, p_xa_tt, p_ya_tt = lookup[to_origin]

    x_tr = x_tt - x_tf
    y_tr = y_tt - y_tf

    n_x = x - img_width * x_tr
    n_y = y - img_height * y_tr

    if p_xa_tf != p_xa_tt:
        n_x *= -1.0

    if p_ya_tf != p_ya_tt:
        n_y *= -1.0

    return n_x, n_y


def undistort_coords(x: ArrayLike, y: ArrayLike, distortion: Distortion,
                     camera_matrix: Optional[NDArray]) -> tuple[ArrayLike, ArrayLike]:
    """
    Removes a distortion from 2D coordinates
    :param x: The x-coordinate
    :param y: The y-coordinate
    :param distortion: The distortion to be removed
    :param camera_matrix: The camera matrix to be used (optional). If not specified a neutral matrix will be used.
    :return: The undistorted coordinate
    """
    if camera_matrix is None:
        camera_matrix = np.identity(3)

    dst_coords = np.column_stack((x, y)).reshape(-1, 1, 2).astype(np.float32)
    dst = distortion.to_array(dtype=np.float32)
    undst_coords = cv2.undistortPoints(dst_coords, camera_matrix, dst)

    undst_x = np.squeeze(undst_coords[..., 0])
    undst_y = np.squeeze(undst_coords[..., 1])
    return undst_x, undst_y


def get_cos_angle(ref_angle: float, offset: ArrayLike, ref_offset: float) -> float:
    """
    This function calculates the angle of a right-angled triangle, which is nested within a reference right-angled
    triangle whose angle is known. Both triangles share the same adjacent side, but have opposite sides of different
    length.
    .. code-block::
        Consider the following example:

        |                    o ... the opposite side length of the different triangle
        |\\                 ro ... the opposite side of the reference triangle
        ||\\  rq             a ... the angle to calculate, i.e., the return value
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
    :param offset: The opposite side of the smaller triangle
    :param ref_offset: The opposite side of the bigger triangle
    :return: The angle of the smaller triangle.
    """
    return np.arctan((offset * np.tan(ref_angle)) / ref_offset)


def cast_ray(ray_origins: Union[Vector3, Sequence[Vector3]], ray_directions: Union[Vector3, Sequence[Vector3]],
             mesh: Union[MeshData, Trimesh], include_misses: bool = True) -> Sequence[Optional[ArrayLike]]:
    """
    Casts a ray to find an intersection with the given mesh.
    :param ray_origins: The origin coordinate of the rays
    :param ray_directions: The ray directions
    :param mesh: The mesh to be used for hits
    :param include_misses: Whether non-intersecting rays should be included in the result via ``None`` values (default
    value is ``True``). If set to ``True``, the result indices correspond to the indices of a rays  origin or direction.
    :return: If there were no hits ``None``; otherwise the coordinate of the first hit
    """
    if not isinstance(mesh, Trimesh):
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.indices)

    origins = np.array(ray_origins).reshape((-1, 3))
    directions = np.array(ray_directions).reshape((-1, 3))

    hits, ray_indices, *_ = mesh.ray.intersects_location(origins, directions, multiple_hits=False)

    if include_misses:
        if len(ray_origins.shape) > 1:
            res_count = ray_origins.shape[0]
        else:
            res_count = 1

        res_dict = {ray_indices[i]: hits[i] for i in range(len(hits))}
        res = [res_dict.get(i, None) for i in range(res_count)]
    else:
        res = hits

    return res


def world_to_pixel_coord(coords: Union[ArrayLike, Sequence[ArrayLike]],
                         width: int, height: int, camera: Camera, ensure_int: bool = True,
                         viewport_origin: PixelOrigin = PixelOrigin.TopLeft) -> tuple[ArrayLike, ArrayLike]:
    """
    Converts world coordinates into pixel coordinates of a shot projection.
    :param coords: The coordinates to convert
    :param width: The width of the image containing the pixel coordinate
    :param height: The height of the image containing the pixel coordinate
    :param camera: The camera to be used for projecting the coordinate
    :param ensure_int: Whether the output coordinates should be rounded to the next integer
    :param viewport_origin: The origin of the viewport (defaults to top left)
    :return: The pixel coordinates of the projected world point
    """
    coords = np.reshape(coords, (-1, 3))
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    homo_coords = np.stack((x, y, z, np.ones_like(x)), axis=-1)
    camera_coords = np.dot(homo_coords, camera.get_view())
    ndc_coords = np.dot(camera_coords, camera.get_proj())
    ndc_coords_norm = ndc_coords / ndc_coords[:, 3]

    # convert normalized coordinates to top-left oriented pixel coordinates
    pixel_xs = (ndc_coords_norm[:, 0] + 1.0) * width / 2.0
    pixel_ys = height - (ndc_coords_norm[:, 1] + 1) / 2.0 * height

    if viewport_origin != PixelOrigin.TopLeft:
        pixel_xs, pixel_ys = cvt_pixel_origin(pixel_xs, pixel_ys, width, height, PixelOrigin.TopLeft, viewport_origin)

    if ensure_int:
        pixel_xs = np.floor(pixel_xs + 0.5).astype(int)
        pixel_ys = np.floor(pixel_ys + 0.5).astype(int)

    if len(coords.shape) <= 1:
        pixel_xs = pixel_xs[0]
        pixel_ys = pixel_ys[0]

    return pixel_xs, pixel_ys


def pixel_to_world_coord(x: ArrayLike, y: ArrayLike, width: int, height: int, mesh: Union[MeshData, Trimesh],
                         camera: Camera, distortion: Optional[Distortion] = None,
                         camera_matrix: Optional[NDArray] = None) -> Sequence[Optional[ArrayLike]]:
    """
    Converts pixel coordinates of a shot projection back to 3D coordinates. The pixel origin is assumed to be in the
    top left corner of the image, with the positive x-axis pointing to the right and the y-axis pointing down.
    :param x: The x-coordinates of the pixels to convert
    :param y: The y-coordinates of the pixels to convert
    :param width: The total width of the image
    :param height: The total height of the image
    :param mesh: The mesh associated with the pixel. Its surface represents the set of possible valid 3D coordinates.
    Passing and reusing the same ``Trimesh`` instance can lead to a significant performance boost compared to passing
    the raw mesh data or a new instance for every conversion.
    :param camera: A camera configured to be equal to the camera used during rendering
    :param distortion: The distortion to be removed from the given pixel coordinates (optional)
    :param camera_matrix: The camera matrix to be used for removing distortion (optional). If not specified when a
    distortion is given, a neutral matrix will be used instead
    :return: If the given pixel coordinates were invalid or did not result in a hit ``None```; otherwise the 3D
    coordinates associated with the given pixel
    :TODO: Allow x and y to be numpy arrays to convert entire arrays of coordinates.
    """
    x = np.array(x)
    y = np.array(y)

    count = len(x)
    if count == 0:
        return

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
            ndc_x * camera.aspect_ratio * tan_fov_y,  # account for fovx via aspect ratio (more efficient)
            ndc_y * tan_fov_y,
            np.full_like(x, -1.0)  # the camera looks along the negative z-axis
        ), axis=-1)

        ray_dirs = np.dot(n_ray_dirs, camera.transform.rotation.matrix33.T)

        # the ray origin is the same as the camera origin
        ray_origins = np.array([camera.transform.position] * count)

    res = cast_ray(ray_origins, ray_dirs, mesh)
    return res
