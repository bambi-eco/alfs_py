from typing import Optional, Collection, Final

from numpy import deg2rad, sign, arctan, arccos, tan
from pyrr import Vector3, Quaternion, Matrix44, Vector4

from src.core.defs import FORWARD, RIGHT, UP, StrEnum
from src.core.geo.quad import Quad
from src.core.geo.transform import Transform

class Side(StrEnum):
    TOP: Final[str] = 'top'
    RIGHT: Final[str] = 'right'
    BOTTOM: Final[str] = 'bottom'
    LEFT: Final[str] = 'left'
    NEAR: Final[str] = 'near'
    FAR: Final[str] = 'far'

class Frustum:
    """
    Class representing a frustum
    """
    _perspective_fovy: float
    aspect_ratio: float
    orthogonal: bool
    orthogonal_size: tuple[int, int]
    near: float
    far: float
    transform: Transform

    _SIDE_NAMES: Final[tuple[Side]] = (Side.TOP, Side.RIGHT, Side.BOTTOM, Side.LEFT, Side.NEAR, Side.FAR)
    _FULL_CORNER_NAMES: Final[tuple[str]] = ('NBL', 'NTL', 'NBR', 'NTR', 'FBL', 'FTL', 'FBR', 'FTR')
    _CORNER_NAMES: Final[tuple[str]] = ('BL', 'TL', 'BR', 'TR')

    def __init__(self, fovy: float = 60.0, aspect_ratio: float = 1.0,
                 orthogonal: bool = False,
                 orthogonal_size: tuple[int, int] = (16, 16),
                 near: float = 0.1, far: float = 10000,
                 transform: Optional[Transform] = None
                 ):
        """
        Initializes a new ``Camera`` object
        :param fovy: The field of view in y direction in degrees (defaults to 60)
        :param aspect_ratio: The aspect ratio of the view (defaults to 1.0)
        :param orthogonal: Whether the camera is orthographic (defaults to False)
        :param orthogonal_size: The cameras size when in orthographic mode (defaults to (16, 16))
        :param near: Distance from the camera to the near clipping plane (defaults to 0.1)
        :param far: Distance from the camera to the far clipping plane (defaults to 10000)
        :param transform: The world transformation of the camera in 3D space (defaults to a neutral transformation)
        The options to pass a rotation or a pair of up and forward vector are mutually exclusive.
        Rotation will be preferred if only one or less of these vectors are passed.
        """
        self._perspective_fovy = fovy
        self.aspect_ratio = aspect_ratio
        self.orthogonal = orthogonal
        self.orthogonal_size = orthogonal_size
        self.near = near
        self.far = far
        self.transform = transform if transform is not None else Transform()

    @property
    def fovy(self) -> float:
        """
        :return: The frustum's fov angle in y direction in degrees. Always equals 0.0 when the frustum is orthogonal
        """
        return self._perspective_fovy if not self.orthogonal else 0.0

    @fovy.setter
    def fovy(self, value) -> None:
        """
        Sets the perspective fov in y direction of the frustum
        :param value: The new value
        """
        self._perspective_fovy = value

    @property
    def fovx(self) -> float:
        """
        :return: The frustum's fov angle in x direction in degrees. Always equals 0.0 when the frustum is orthogonal
        """
        return self.fovy * self.aspect_ratio if not self.orthogonal else 0.0

    @fovx.setter
    def fovx(self, value) -> None:
        """
        Sets the perspective fov in x direction of the frustum
        :param value: The new value
        """
        self.fovy = value / self.aspect_ratio

    @property
    def corners(self) -> tuple[Vector3, Vector3, Vector3, Vector3, Vector3, Vector3, Vector3, Vector3]:
        """
        :return: The corners of the frustum defined by the near and far clipping distance.
        Points are ordered from near to far, bottom to top, and left to right (i.e., NBL, NTL, NBR, NTR, FBL, FTL, FBR, FTR)
        """
        return self.corners_at(self.near) + self.corners_at(self.far)

    @property
    def corners_dict(self) -> dict[str, Vector3]:
        """
        :return: The corners of the frustum defined by the near and far clipping distance.
        Points are named as NBL, NTL, NBR, NTR, FBL, FTL, FBR, and FTR
        """
        return {key: corner for key, corner in zip(Frustum._FULL_CORNER_NAMES, self.corners)}


    def corners_at(self, depth: float) -> tuple[Vector3, Vector3, Vector3, Vector3]:
        """
        Computes the corners of the frustums slice at the given depth.
        :param depth: The depth the frustum should be sliced at
        :return: A tuple containing the corner points going from bottom to top and from left to right (i.e., BL, TL, BR, TR)
        """
        x_rad = deg2rad(self.fovx / 2.0)
        y_rad = deg2rad(self.fovy / 2.0)
        l = Quaternion.from_eulers([0.0, x_rad, 0.0]) * FORWARD  # center of the left side
        t = Quaternion.from_eulers([y_rad, 0.0, 0.0]) * FORWARD  # center of the top side
        direction = Vector3([l.x + t.x, l.y + t.y, l.z]).normalized  # take depth in forward direction only once
        ref = direction * depth  # top left corner
        mat = self.transform.mat  # translation, rotation, and scale

        return mat * Vector3([-ref.x, -ref.y, ref.z]), \
               mat * Vector3([-ref.x,  ref.y, ref.z]), \
               mat * Vector3([ ref.x, -ref.y, ref.z]), \
               mat * ref

    def corners_at_dict(self, depth: float) -> dict[str, Vector3]:
        """
        Computes the corners of the frustums slice at the given depth.
        :param depth: The depth the frustum should be sliced at
        :return: A map holding all corner points as BL, TL, BR, and TR
        """
        return {key: corner for key, corner in zip(Frustum._CORNER_NAMES, self.corners_at(depth))}

    @property
    def sides(self) -> tuple[Quad, Quad, Quad, Quad, Quad, Quad]:
        """
        :return: Returns the six sides of the frustum in the order top, right, bottom, left, near, far.
        """
        corners = self.corners
        # corners come in order NBL, NTL, NBR, NTR, FBL, FTL, FBR, FTR
        near = Quad(corners[0], corners[1], corners[2], corners[3])
        far = Quad(corners[4], corners[5], corners[6], corners[7])
        left = Quad(corners[0], corners[4], corners[1], corners[5])
        right = Quad(corners[2], corners[6], corners[3], corners[7])
        top = Quad(corners[1], corners[5], corners[3], corners[7])
        bottom = Quad(corners[2], corners[6], corners[0], corners[4])

        return top, right, bottom, left, near, far

    @property
    def sides_dict(self) -> dict[str, Quad]:
        """
        :return: Returns a map storing the six sides of the frustum as top, right, bottom, left, near, and far.
        """
        return {key: quad for key, quad in zip(Frustum._SIDE_NAMES, self.sides)}

    def get_proj(self, dtype: object = 'f4') -> Matrix44:
        """
        Generates a projection matrix based on the cameras current parameters
        :param dtype: The dtype all values in the matrix should be converted to
        :return: A matrix representing a projection
        """
        if self.orthogonal:
            width_vec = self.orthogonal_size[0] / 2.0
            height_vec = self.orthogonal_size[1] / 2.0
            return Matrix44.orthogonal_projection(-width_vec, width_vec, -height_vec, height_vec, self.near, self.far,
                                                  dtype=dtype)
        else:
            return Matrix44.perspective_projection(self.fovy, self.aspect_ratio, self.near, self.far, dtype=dtype)

    def get_view(self, dtype: object = 'f4') -> Matrix44:
        """
        Generates a view matrix based on the cameras current parameters
        :param dtype: The dtype all values in the matrix should be converted to
        :return: A matrix representing a view
        """
        return Matrix44.look_at(self.transform.position, self.transform.target, self.transform.up, dtype=dtype)

    def get_mat(self, dtype: object = 'f4') -> Matrix44:
        """
        Generates a matrix based on the cameras current parameters
        :param dtype: The dtype all values in the matrix should be converted to
        :return: A matrix representing the combined projection and view matrix
        """
        return self.get_proj(dtype=dtype) * self.get_view(dtype=dtype)

    def captures(self, point: Vector3) -> bool:
        """
        Returns whether a point lies within the frustum or not
        :param point: The point to be checked
        :return: Whether a point lies within the frustum or not
        """
        mat = self.get_mat()
        p4 = Vector4([point.x, point.y, point.z, 1.0])
        pp = mat * p4
        # GLSL clipping rule
        return abs(pp.x) <= pp.w and abs(pp.y) <= pp.w and abs(pp.z) <= pp.w

    def fit_to_points(self, points: Collection[Vector3], leeway: float) -> None:
        """
        Translates frustum along its viewing direction to capture all given points.
        This process ignores near and far clipping settings and does not modify them.
        :param points: A collection of points to capture
        :param leeway: The maximum angle allowed between a point a side of the frustum expressed by its cosine value
        """
        sides = self.sides_dict
        sides.pop('near', None)
        sides.pop('far', None)
        side_map = {key: (side.normal, side.a, side.center) for key, side in sides.items()}
        max_up_angle = arctan(1.0 / self.aspect_ratio)

        min_angle = float('inf')
        min_side_name = None
        min_side_origin = None
        min_side_normal = None
        min_point = None
        min_rel_point = None
        min_t_point = None
        min_t_proj_r = None
        min_t_proj_u = None

        points = tuple(points)
        for i, point in enumerate(points):
            # project point onto relative plane of the frustums up and right vector to remove depth
            t_point = self.transform.mat.inverse * point
            t_proj_r = t_point * RIGHT  # vector_project(t_point, RIGHT)
            t_proj_u = t_point * UP     # vector_project(t_point, UP)
            t_proj = t_proj_r + t_proj_u

            # find angle between local up and projection
            t_angle = arccos(UP.dot(t_proj) / t_proj.length)

            # determine which side is associated with the point via t_angle
            if t_angle > max_up_angle:  # if angle between up and proj > arctan(1 / aspect_ratio) it is either left or right
                if t_proj.x < 0.0:  # if x is negative cant be right side => left side
                    side_name = Side.LEFT
                else:  # else => right side
                    side_name = Side.RIGHT
            else:  # else it is either top or bottom
                if t_proj.y < 0.0:  # if y is negative cant be top => bottom
                    side_name = Side.BOTTOM
                else:  # else => top
                    side_name = Side.TOP

            # compute angle between side normal and relative point
            side_normal, side_origin, side_center = side_map[side_name]
            rel_point = point - side_origin
            cur_angle = side_normal.dot(rel_point) / rel_point.length

            if cur_angle < min_angle:
                min_angle = cur_angle
                min_side_name = side_name
                min_side_normal = side_normal
                min_side_origin = side_origin
                min_point = point
                min_rel_point = rel_point
                min_t_point = t_point
                min_t_proj_r = t_proj_r
                min_t_proj_u = t_proj_u


        if min_angle < 0.0 or min_angle > leeway: # if the point lies outside the frustum or too far away from the frustum's sides
            # get the sign of the delta by feeding the angle between forward and point to the sign function
            # forward = self.transform.forward
            # angle = forward.dot(min_point) / min_t_point.length
            fact = sign(min_angle)

            if min_side_name in (Side.TOP, Side.BOTTOM):
                beta = deg2rad(90.0 - self.fovy / 2.0)
                b = min_t_proj_u.length
                a = b * tan(beta)
            else:
                beta = deg2rad(90.0 - self.fovx / 2.0)
                b = min_t_proj_r.length
                a = b * tan(beta)
            delta = FORWARD * a

            delta = self.transform.scale_mat * delta

            # translate transform by delta in its forward direction
            self.transform.translate(self.transform.forward * delta.length * fact)
