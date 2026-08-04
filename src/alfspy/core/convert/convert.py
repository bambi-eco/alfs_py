from collections import defaultdict
from typing import Optional, Union, Sequence, Callable, Final, Tuple

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


def _to_top_left(x: ArrayLike, y: ArrayLike, max_x: Number, max_y: Number,
                 origin: tuple[float, float, bool, bool]) -> tuple[ArrayLike, ArrayLike]:
    """
    Converts coordinates expressed in the given origin into top-left coordinates.
    :param x: The x-coordinate in the source origin.
    :param y: The y-coordinate in the source origin.
    :param max_x: The maximum index along the x-axis.
    :param max_y: The maximum index along the y-axis.
    :param origin: The source origin's lookup tuple.
    :return: The equivalent top-left coordinates.
    """
    offset_x, offset_y, points_right, points_up = origin
    return (offset_x * max_x + (x if points_right else -x),
            offset_y * max_y + (-y if points_up else y))


def _from_top_left(x: ArrayLike, y: ArrayLike, max_x: Number, max_y: Number,
                   origin: tuple[float, float, bool, bool]) -> tuple[ArrayLike, ArrayLike]:
    """
    Converts top-left coordinates into coordinates expressed in the given origin.
    :param x: The top-left x-coordinate.
    :param y: The top-left y-coordinate.
    :param max_x: The maximum index along the x-axis.
    :param max_y: The maximum index along the y-axis.
    :param origin: The target origin's lookup tuple.
    :return: The equivalent coordinates in the target origin.
    """
    offset_x, offset_y, points_right, points_up = origin
    shifted_x = x - offset_x * max_x
    shifted_y = y - offset_y * max_y
    return (shifted_x if points_right else -shifted_x,
            -shifted_y if points_up else shifted_y)


def change_pixel_origin(x: ArrayLike, y: ArrayLike, max_x: Number, max_y: Number, from_origin: PixelOrigin,
                        to_origin: PixelOrigin) -> tuple[ArrayLike, ArrayLike]:
    """
    Converts the given pixel coordinates from one common image origin to another.

    The conversion routes through top-left coordinates. The previous implementation
    translated first and negated afterwards, which made the function asymmetric: converting
    to an origin whose axes point the other way and back left a residual of
    ``2 * max * offset``. Round-tripping is now exact for every origin pair.

    :param x: The x-coordinate of the pixel.
    :param y: The y-coordinate of the pixel.
    :param max_x: The maximum index along the x-axis.
    :param max_y: The maximum index along the y-axis.
    :param from_origin: The origin of the passed coordinate.
    :param to_origin: The origin to convert the coordinates to.
    :return: A tuple containing converted coordinates ``(x, y)``.
    """
    lookup = _POT_LOOKUP
    top_left_x, top_left_y = _to_top_left(x, y, max_x, max_y, lookup[from_origin])
    return _from_top_left(top_left_x, top_left_y, max_x, max_y, lookup[to_origin])


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
    coordinates = np.reshape(np.asarray(coordinates, dtype=np.float64), (-1, 3))
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    z = coordinates[:, 2]

    homo_coords = np.stack((x, y, z, np.ones_like(x)), axis=-1)
    camera_coords = np.dot(homo_coords, np.asarray(camera.get_view(), dtype=np.float64))
    ndc_coords = np.dot(camera_coords, np.asarray(camera.get_proj(), dtype=np.float64))

    # Divide each point by its *own* w. The previous `ndc_coords / ndc_coords[:, 3]` divided
    # an (N, 4) array by an (N,) one, which only broadcasts when N is 1 or 4 -- and even at
    # N == 4 it produced `clip[i, j] / w[j]`, dividing every point by the first point's
    # depth. Orthographic cameras leave every w at 1.0, so the error was invisible there.
    w = ndc_coords[:, 3:4]
    w = np.where(np.abs(w) < np.finfo(np.float64).tiny, 1.0, w)
    ndc_coords_norm = ndc_coords / w

    # convert normalized coordinates to top-left oriented pixel coordinates
    pixel_xs = (ndc_coords_norm[:, 0] + 1.0) * width / 2.0
    pixel_ys = height - (ndc_coords_norm[:, 1] + 1) / 2.0 * height

    if viewport_origin != PixelOrigin.TopLeft:
        pixel_xs, pixel_ys = change_pixel_origin(pixel_xs, pixel_ys, width, height, PixelOrigin.TopLeft, viewport_origin)

    if ensure_int:
        pixel_xs = np.floor(pixel_xs + 0.5).astype(int)
        pixel_ys = np.floor(pixel_ys + 0.5).astype(int)

    return pixel_xs, pixel_ys

def world_to_pixel_coord2(
    px: ArrayLike,
    py: ArrayLike,
    pz: ArrayLike,
    width: int,
    height: int,
    camera: "Camera",
    distortion: Optional["Distortion"] = None,
    camera_matrix: Optional[NDArray] = None,
    include_misses: bool = True,
    clip_to_view: bool = True,
) -> Sequence[Optional[Tuple[float, float]]]:
    """
    Inverts `pixel_to_world_coord` by projecting world points to pixel coordinates.

    Conventions:
    - Viewport origin at the top-left, +x right, +y down.
    - Camera looks along negative z in its local frame (same as in `pixel_to_world_coord`).
    - When `distortion` is provided, ideal pixel coords are distorted before returning,
      mirroring the `undistort_coords` step in the forward direction.

    :param px,py,pz: Components of world points to project (same length).
    :param width,height: Viewport size in pixels.
    :param camera: Same camera used in `pixel_to_world_coord`.
    :param distortion: If provided, apply *distortion* to the ideal pixel coords before returning.
    :param camera_matrix: Intrinsics used by the distortion model. If None, identity is assumed (as in your code).
    :param include_misses: If True, return None for points that cannot be projected; else drop them.
    :param clip_to_view: If True, points outside [0,width)×[0,height) are treated as misses.
    :return: List of (x_px, y_px) or None, index-aligned with the inputs.
    """
    px = np.asarray(px)
    py = np.asarray(py)
    pz = np.asarray(pz)
    assert px.shape == py.shape == pz.shape, "px, py, pz must have same shape"
    n = px.size
    if n == 0:
        return [None]

    # Bring the points into camera space. pyrr is a row-vector library, so `v @ R33` rotates
    # camera-space into world-space; the world-to-camera direction therefore needs the
    # transpose. The previous code used `@ R`, the wrong direction.
    P_world = np.stack([px, py, pz], axis=-1)
    t = np.asarray(camera.transform.position, dtype=np.float64)
    R = np.asarray(camera.transform.rotation.matrix33, dtype=np.float64)
    P_cam = (P_world - t) @ R.T

    x_c = P_cam[..., 0]
    y_c = P_cam[..., 1]
    z_c = P_cam[..., 2]

    # Points at/behind camera cannot project (camera looks toward -z)
    valid = z_c < 0.0

    # Prepare holders
    x_px = np.full_like(x_c, np.nan, dtype=float)
    y_px = np.full_like(y_c, np.nan, dtype=float)

    if camera.orthogonal:
        # Inverse of the (corrected) forward mapping in `pixel_to_world_coord`:
        #   origin_offset_x = ndc_x * img_plane_width / 2   =>   ndc_x = 2 * x_c / img_plane_width
        img_plane_w, img_plane_h = camera.orthogonal_size
        # Avoid division by zero
        eps = 1e-12
        img_plane_w = max(float(img_plane_w), eps)
        img_plane_h = max(float(img_plane_h), eps)

        ndc_x = (2.0 * x_c) / img_plane_w
        ndc_y = (2.0 * y_c) / img_plane_h

        # Map NDC -> pixels (origin top-left, +y down)
        x_px_ideal = (ndc_x + 1.0) * 0.5 * width
        y_px_ideal = (1.0 - ndc_y) * 0.5 * height
        x_px[valid] = x_px_ideal[valid]
        y_px[valid] = y_px_ideal[valid]
    else:
        # Perspective projection inverse of your ray construction:
        # You used n_ray_dirs = [ ndc_x * aspect * tan(fovy/2), ndc_y * tan(fovy/2), -1 ]
        # For a *point* we have: x_c / -z_c = ndc_x * aspect * tanF,  y_c / -z_c = ndc_y * tanF
        tanF = np.tan(np.deg2rad(camera.fovy) / 2.0)
        asp = float(camera.aspect_ratio)
        denom = -z_c  # positive for valid points (since z_c < 0)
        eps = 1e-12
        denom = np.where(np.abs(denom) < eps, np.sign(denom) * eps, denom)

        ndc_x = (x_c / denom) / (asp * tanF)
        ndc_y = (y_c / denom) / tanF

        # Map NDC -> pixels
        x_px_ideal = (ndc_x + 1.0) * 0.5 * width
        y_px_ideal = (1.0 - ndc_y) * 0.5 * height
        x_px[valid] = x_px_ideal[valid]
        y_px[valid] = y_px_ideal[valid]

    # Optionally clip to image bounds
    if clip_to_view:
        in_view = (x_px >= 0.0) & (x_px < width) & (y_px >= 0.0) & (y_px < height)
        valid = valid & in_view

    # Apply lens distortion to ideal pixels (mirror of undistort in forward dir)
    if distortion is not None:
        # Assumes you have a function that maps ideal pixel coords -> distorted pixel coords
        # todo
        pass
        # xd, yd = distort_coords(x_px[valid], y_px[valid], distortion, camera_matrix)
        # x_px[valid] = xd
        # y_px[valid] = yd

    # Build result list
    result = []
    for i in range(n):
        if valid[i]:
            result.append((float(x_px[i]), float(y_px[i])))
        else:
            if include_misses:
                result.append(None)
            # else: skip — but to keep indices aligned like your forward function, we still append None
            else:
                result.append(None)

    return result


def pixel_to_world_coord(x: ArrayLike, y: ArrayLike, width: int, height: int, mesh: Union[MeshData, Trimesh],
                         camera: Camera, distortion: Optional[Distortion] = None,
                         camera_matrix: Optional[NDArray] = None, include_misses: bool = True) -> Sequence[Optional[ArrayLike]]:
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
    :param include_misses: Whether non-intersecting rays should be included in the result via ``None`` values (default
    value is ``True``). If set to ``True``, the result indices correspond to the indices of a rays  origin or direction.
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

    # pyrr is a row-vector library, so `v @ R33` is the camera-to-world rotation -- the same
    # one `Transform.forward` uses. The previous code multiplied by `R33.T`, i.e. the inverse,
    # which made the centre ray point opposite to where the camera actually faced. That was
    # cancelled downstream by ProjectionScene negating its Euler angles; both halves were
    # fixed together. See docs/pyrr_conventions.md.
    rotation = np.asarray(camera.transform.rotation.matrix33, dtype=np.float64)
    camera_position = np.asarray(camera.transform.position, dtype=np.float64)

    if camera.orthogonal:
        # The ray direction is parallel to the camera forward vector.
        forward = np.asarray(camera.transform.forward, dtype=np.float64)
        ray_dirs = np.broadcast_to(forward, (count, 3))

        # The ray origin lies on the camera plane, offset by the pixel's position on the
        # image plane. `ndc_x` is already normalised to [-1, 1], so it scales by half the
        # frustum size; the previous code divided by `width`/`height` a second time, which
        # collapsed every ray onto the camera axis.
        img_plane_width, img_plane_height = camera.orthogonal_size
        origin_offset = np.stack((
            ndc_x * img_plane_width / 2.0,
            ndc_y * img_plane_height / 2.0,
            np.zeros_like(ndc_x),
        ), axis=-1)

        ray_origins = camera_position + np.dot(origin_offset, rotation)
    else:  # camera has a perspective projection
        # the ray direction is the projection of the pixel coordinates onto the image plane
        tan_fov_y = np.tan(np.deg2rad(camera.fovy) / 2.0)
        n_ray_dirs = np.stack((
            ndc_x * camera.aspect_ratio * tan_fov_y,  # account for fov x via aspect ratio (more efficient)
            ndc_y * tan_fov_y,
            np.full_like(ndc_x, -1.0)  # the camera looks along the negative z-axis
        ), axis=-1)

        ray_dirs = np.dot(n_ray_dirs, rotation)

        # the ray origin is the same as the camera origin
        ray_origins = np.broadcast_to(camera_position, (count, 3))

    res = cast_ray(ray_origins, ray_dirs, mesh, include_misses=include_misses)
    return res
