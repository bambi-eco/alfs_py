from typing import Optional, Collection

from numpy import deg2rad
from pyrr import Vector3, Quaternion, Matrix44, Vector4

from src.core.defs import FORWARD, EPSILON
from src.core.geo.quad import Quad
from src.core.transform import Transform

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
               mat * Vector3([-ref.x, ref.y, ref.z]), \
               mat * Vector3([ref.x, -ref.y, ref.z]), \
               mat * ref

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
        return Matrix44.look_at(self.transform.position, self.transform.forward + self.transform.position,
                                self.transform.up, dtype=dtype)

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
        return abs(pp.x) < pp.w and abs(pp.y) < pp.w and 0 < pp.z < pp.w

    def fit_to_points(self, points: Collection[Vector3]) -> None:
        """
        Translates frustum along its viewing direction to capture all given points.
        This process ignores near and far clipping settings and does not modify them.
        :param points: A collection of points to capture
        """
        pass
