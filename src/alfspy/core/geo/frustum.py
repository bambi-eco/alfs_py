from typing import Optional, Final, Any

from numpy import deg2rad
from pyrr import Vector3, Quaternion, Matrix44, Vector4

from alfspy.core.util.defs import FORWARD
from alfspy.core.geo import Quad
from alfspy.core.geo import Side
from alfspy.core.geo import Transform


class Frustum:
    """
    Class representing a frustum.
    """
    _perspective_fovy: float
    aspect_ratio: float
    orthogonal: bool
    orthogonal_size: tuple[float, float]
    near: float
    far: float
    transform: Transform

    _SIDE_NAMES: Final[tuple[Side]] = (Side.Top, Side.Right, Side.Bottom, Side.Left, Side.Near, Side.Far)
    _FULL_CORNER_NAMES: Final[tuple[str]] = ('NBL', 'NTL', 'NBR', 'NTR', 'FBL', 'FTL', 'FBR', 'FTR')
    _CORNER_NAMES: Final[tuple[str]] = ('BL', 'TL', 'BR', 'TR')

    def __init__(self, fovy: float = 60.0, aspect_ratio: float = 1.0,
                 orthogonal: bool = False,
                 orthogonal_size: tuple[float, float] = (16, 16),
                 near: float = 0.1, far: float = 10000,
                 transform: Optional[Transform] = None
                 ):
        """
        Initializes a new ``Camera`` object.
        :param fovy: The field of view in y direction in degrees (defaults to 60).
        :param aspect_ratio: The aspect ratio of the view (defaults to 1.0).
        :param orthogonal: Whether the camera is orthographic (defaults to False).
        :param orthogonal_size: The cameras size when in orthographic mode (defaults to (16, 16)).
        :param near: Distance from the camera to the near clipping plane (defaults to 0.1).
        :param far: Distance from the camera to the far clipping plane (defaults to 10000).
        :param transform: The world transformation of the camera in 3D space (defaults to a neutral transformation).
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
        :return: The frustum's fov angle in y direction in degrees. Always equals 0.0 when the frustum is orthogonal.
        """
        return self._perspective_fovy if not self.orthogonal else 0.0

    @fovy.setter
    def fovy(self, value) -> None:
        """
        Sets the perspective fov in y direction of the frustum.
        :param value: The new value
        """
        self._perspective_fovy = value

    @property
    def fovx(self) -> float:
        """
        :return: The frustum's fov angle in x direction in degrees. Always equals 0.0 when the frustum is orthogonal.
        """
        return self.fovy * self.aspect_ratio if not self.orthogonal else 0.0

    @fovx.setter
    def fovx(self, value) -> None:
        """
        Sets the perspective fov in x direction of the frustum.
        :param value: The new fovx value.
        """
        self.fovy = value / self.aspect_ratio

    @property
    def corners(self) -> tuple[Vector3, Vector3, Vector3, Vector3, Vector3, Vector3, Vector3, Vector3]:
        """
        :return: The corners of the frustum defined by the near and far clipping distance.
        Points are ordered from near to far, bottom to top, and left to right
        (i.e., NBL, NTL, NBR, NTR, FBL, FTL, FBR, FTR).
        """
        return self.corners_at(self.near) + self.corners_at(self.far)

    @property
    def corners_dict(self) -> dict[str, Vector3]:
        """
        :return: The corners of the frustum defined by the near and far clipping distance.
        Points are named as NBL, NTL, NBR, NTR, FBL, FTL, FBR, and FTR.
        """
        return {key: corner for key, corner in zip(Frustum._FULL_CORNER_NAMES, self.corners)}

    def corners_at(self, depth: float) -> tuple[Vector3, Vector3, Vector3, Vector3]:
        """
        Computes the corners of the frustums slice at the given depth.
        :param depth: The depth the frustum should be sliced at.
        :return: A tuple containing the corner points going from bottom to top and from left to right
        (i.e., BL, TL, BR, TR).
        """
        x_rad = deg2rad(self.fovx / 2.0)
        y_rad = deg2rad(self.fovy / 2.0)
        c_left = Quaternion.from_eulers([0.0, x_rad, 0.0]) * FORWARD  # center of the left side
        c_top = Quaternion.from_eulers([y_rad, 0.0, 0.0]) * FORWARD  # center of the top side
        # take depth in forward direction only once
        direction = Vector3([c_left.x + c_top.x, c_left.y + c_top.y, c_left.z]).normalized
        ref = direction * depth  # top left corner
        mat = self.transform.mat()  # translation, rotation, and scale

        return mat * Vector3([-ref.x, -ref.y, ref.z]), \
            mat * Vector3([-ref.x, ref.y, ref.z]), \
            mat * Vector3([ref.x, -ref.y, ref.z]), \
            mat * ref

    def corners_at_dict(self, depth: float) -> dict[str, Vector3]:
        """
        Computes the corners of the frustums slice at the given depth.
        :param depth: The depth the frustum should be sliced at.
        :return: A map holding all corner points as BL, TL, BR, and TR.
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
        Generates a projection matrix based on the cameras current parameters.
        :param dtype: The dtype all values in the matrix should be converted to.
        :return: A matrix representing a projection.
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
        Generates a view matrix based on the cameras current parameters.
        :param dtype: The dtype all values in the matrix should be converted to.
        :return: A matrix representing a view.
        """
        return Matrix44.look_at(self.transform.position, self.transform.target, self.transform.up, dtype=dtype)

    def get_mat(self, dtype: object = 'f4') -> Matrix44:
        """
        Generates a matrix based on the cameras current parameters.
        :param dtype: The dtype all values in the matrix should be converted to.
        :return: A matrix representing the combined projection and view matrix.
        """
        return self.get_proj(dtype=dtype) * self.get_view(dtype=dtype)

    def captures(self, point: Vector3) -> bool:
        """
        Returns whether a point lies within the frustum or not.
        :param point: The point to be checked.
        :return: Whether a point lies within the frustum or not.
        """
        mat = self.get_mat()
        p4 = Vector4([point.x, point.y, point.z, 1.0])
        pp = mat * p4
        # GLSL clipping rule
        return abs(pp.x) <= pp.w and abs(pp.y) <= pp.w and abs(pp.z) <= pp.w

    def to_dict(self) -> dict[str, Any]:
        """
        :return: A dictionary that can be used to reconstruct this frustum instance.
        """
        return {
            'fovy': self._perspective_fovy,
            'aspect_ratio': self.aspect_ratio,
            'orthogonal': self.orthogonal,
            'orthogonal_size': self.orthogonal_size,
            'near': self.near,
            'far': self.far,
            'transform': self.transform.to_dict()
        }

    @staticmethod
    def from_dict(dictionary: dict[str, Any]) -> 'Frustum':
        """
        Creates a frustum instance from a dictionary of parameters.
        :return: If the dictionary is missing properties or cannot be used for constructing a frustum ``None``;
        otherwise the constructed frustum instance.
        """
        try:
            input_dict = {
                k: v for k, v in dictionary.items() if k in {
                    'fovy',
                    'aspect_ratio',
                    'orthogonal',
                    'orthogonal_size',
                    'near',
                    'far',
                    'transform'
                }
            }
            if 'transform' in input_dict:
                input_dict["transform"] = Transform.from_dict(input_dict['transform'])

            return Frustum(**input_dict)
        except (TypeError, IndexError, KeyError):
            pass
