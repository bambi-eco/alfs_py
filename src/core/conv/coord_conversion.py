from collections import defaultdict
from functools import cache
from typing import Optional, Union

import cv2
import numpy as np
import trimesh
from numpy.typing import NDArray
from pyrr import Vector3, Vector4, Matrix44
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


def cvt_pixel_origin(x: float, y: float, img_width, img_height, from_origin: PixelOrigin,
                     to_origin: PixelOrigin) -> tuple[float, float]:
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


def undistort_coords(x: float, y: float, distortion: Distortion, camera_matrix: Optional[NDArray]) -> tuple[float, float]:
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

    dist_coords = np.array([x, y, 1])
    undist_coords = cv2.undistortPoints(dist_coords, camera_matrix, distortion.to_array())
    x = undist_coords[0, 0, 0]
    y = undist_coords[0, 0, 1]
    return x, y


def get_cos_angle(ref_angle: float, offset: float, ref_offset: float) -> float:
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


def cast_ray(ray_origin: Vector3, ray_direction: Vector3, mesh: Union[MeshData, Trimesh]) -> Optional[Vector3]:
    """
    Casts a ray to find an intersection with the given mesh.
    :param ray_origin: The origin coordinate of the ray
    :param ray_direction: The rays' direction
    :param mesh: The mesh to be used for hits
    :return: If there were no hits ``None``; otherwise the coordinate of the first hit
    """
    if not isinstance(mesh, Trimesh):
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.indices)

    origins = np.array([ray_origin[:3]])
    directions = np.array([ray_direction[:3]])
    point, *_ = mesh.ray.intersects_location(origins, directions, multiple_hits=False)

    return Vector3(point[0]) if len(point) > 0 else None


def world_to_pixel_coord(coord: Vector3, width: int, height: int, camera: Camera, ensure_int: bool = True,
                         viewport_origin: PixelOrigin = PixelOrigin.TopLeft) -> Union[tuple[int, int], tuple[float, float]]:
    """
    Converts world coordinates into pixel coordinates of a shot projection.
    :param coord: The coordinate to convert
    :param width: The width of the image containing the pixel coordinate
    :param height: The height of the image containing the pixel coordinate
    :param camera: The camera to be used for projecting the coordinate
    :param ensure_int: Whether the output coordinates should be rounded to the next integer
    :param viewport_origin: The origin of the viewport (defaults to top left)
    :return: The pixel coordinates of the projected world point
    """
    # transform coordinate to camera space
    camera_coord = camera.get_view() * coord
    camera_coord = Vector4.from_vector3(camera_coord, w=1.0)

    # project coordinate onto the normalized camera plane
    ndc_coord = camera.get_proj() * camera_coord
    ndc_coord /= ndc_coord[3]

    # convert normalized coordinates to top-left oriented pixel coordinates
    pixel_coord = (ndc_coord.x + 1.0) * width / 2.0, height - (ndc_coord.y + 1) / 2 * height

    if viewport_origin != PixelOrigin.TopLeft:
        pixel_coord = cvt_pixel_origin(pixel_coord[0], pixel_coord[1], width, height,
                                       PixelOrigin.TopLeft, viewport_origin)

    if ensure_int:
        pixel_coord = (nearest_int(pixel_coord[0]), nearest_int(pixel_coord[1]))

    return pixel_coord


def pixel_to_world_coord(x: float, y: float, width: int, height: int, mesh: Union[MeshData, Trimesh],
                         camera: Camera, distortion: Optional[Distortion] = None,
                         camera_matrix: Optional[NDArray] = None) -> Optional[Vector3]:
    """
    Converts pixel coordinates of a shot projection back to 3D coordinates. The pixel origin is assumed to be in the
    top left corner of the image, with the positive x-axis pointing to the right and the y-axis pointing down.
    :param x: The x-coordinate of the pixel to convert
    :param y: The y-coordinate of the pixel to convert
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
    if distortion is not None:
        x, y = undistort_coords(x, y, distortion, camera_matrix)

    # convert pixel to normalized coordinates
    ndc_x = (2.0 * x / width) - 1.0
    ndc_y = 1.0 - (2.0 * y / height)

    if camera.orthogonal:
        # the ray direction is parallel to the camera forward vector
        ray_dir = camera.transform.forward

        # the ray origin lies on the same plane as the camera but is offset by the pixels distance on the image plane
        img_plane_width, img_plane_height = camera.orthogonal_size
        origin_offset_x = (ndc_x * img_plane_width) / width
        origin_offset_y = (ndc_y * img_plane_height) / height
        origin_offset = Vector3([origin_offset_x, origin_offset_y, 0])
        ray_origin = camera.transform.position + camera.transform.rotation * origin_offset
    else:  # camera has a perspective projection
        # the ray direction is the projection of the pixel coordinates onto the image plane
        tan_fov_y = np.tan(np.deg2rad(camera.fovy) / 2.0)
        ray_dir = Vector3([
            ndc_x * camera.aspect_ratio * tan_fov_y,  # account for fovx via aspect ratio (more efficient)
            ndc_y * tan_fov_y,
            -1.0  # the camera looks along the negative z-axis
        ])
        ray_dir = camera.transform.rotation * ray_dir

        # the ray origin is the same as the camera origin
        ray_origin = camera.transform.position

    res = cast_ray(ray_origin, ray_dir, mesh)
    return res
