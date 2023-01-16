import copy
from typing import Optional

from pyrr import Matrix44, Quaternion, Vector3

from src.core.geo.frustum import Frustum
from src.core.transform import Transform


class Camera:
    """
    Class representing a camera in 3D space
    """
    _frustum: Frustum

    def __init__(self, fovy: float = 60.0, aspect_ratio: float = 1.0,
                 orthogonal: bool = False,
                 orthogonal_size: tuple[int, int] = (16, 16),
                 near: float = 0.1, far: float = 10000,
                 position: Vector3 = Vector3([0.0, 0.0, 0.0]),
                 rotation: Optional[Quaternion] = None,
                 forward: Optional[Vector3] = None, up: Optional[Vector3] = None) -> None:
        """
        Initializes a new ``Camera`` object
        :param fovy: The field of view in y direction in degrees (defaults to 60)
        :param aspect_ratio: The aspect ratio of the view (defaults to 1.0)
        :param orthogonal: Whether the camera is orthographic (defaults to False)
        :param orthogonal_size: The cameras size when in orthographic mode (defaults to (16, 16))
        :param near: Distance from the camera to the near clipping plane (defaults to 0.1)
        :param far: Distance from the camera to the far clipping plane (defaults to 10000)
        :param position: The position of the camera in 3D space (defaults to [0.0, 0.0, 0.0])
        :param rotation: The rotation of the camera in 3D space (optional)
        :param forward: The forward vector of the camera in 3D space (optional)
        :param up: The up vector of the camera in 3D space (optional)
        The options to pass a rotation or a pair of up and forward vector are mutually exclusive.
        Rotation will be preferred if only one or less of these vectors are passed.
        """
        if forward is None or up is None:
            transform = Transform(position, rotation, None)
        else:
            transform = Transform.from_up_forward(up, forward, position, None)

        self._frustum = Frustum(fovy, aspect_ratio, orthogonal, orthogonal_size, near, far, transform)

    @staticmethod
    def from_frustum(frustum: Frustum) -> 'Camera':
        """
        Creates a ``Camera`` instance based on the given frustum
        :param frustum: The ``Frustum`` object the camera should be based on
        :return: A ``Camera`` instance initialized respectively
        """
        camera = Camera()
        camera._frustum = copy.deepcopy(frustum)
        return camera

    @property
    def fovy(self) -> float:
        """
        :return: The camera's fov angle in y direction in degrees. Always equals 0.0 when the camera is orthogonal
        """
        return self._frustum.fovy

    @fovy.setter
    def fovy(self, value) -> None:
        """
        Sets the fov in y direction of the camera
        :param value: The new value
        """
        self._frustum.fovy = value

    @property
    def fovx(self) -> float:
        """
        :return: The camera's fov angle in x direction. Always equals 0.0 when the camera is orthogonal
        """
        return self._frustum.fovx

    @fovx.setter
    def fovx(self, value) -> None:
        """
        Sets the perspective fov in x direction of the camera
        :param value: The new value
        """
        self._frustum.fovx = value

    @property
    def aspect_ratio(self) -> float:
        """
        :return: The camera's aspect ratio
        """
        return self._frustum.aspect_ratio

    @aspect_ratio.setter
    def aspect_ratio(self, value) -> None:
        """
        Sets the aspect ratio of the camera
        :param value: The new value
        """
        self._frustum.aspect_ratio = value

    @property
    def orthogonal(self) -> bool:
        """
        :return: Whether the camera is orthogonal or not
        """
        return self._frustum.orthogonal

    @orthogonal.setter
    def orthogonal(self, value) -> None:
        self._frustum.orthogonal = value

    @property
    def orthogonal_size(self) -> tuple[int, int]:
        return self._frustum.orthogonal_size

    @orthogonal_size.setter
    def orthogonal_size(self, value) -> None:
        self._frustum.orthogonal_size = value

    @property
    def near(self) -> float:
        return self._frustum.near

    @near.setter
    def near(self, value) -> None:
        self._frustum.near = value

    @property
    def far(self) -> float:
        return self._frustum.far

    @far.setter
    def far(self, value) -> None:
        self._frustum.far = value

    @property
    def transform(self) -> Transform:
        return self._frustum.transform


    @property
    def target(self) -> Vector3:
        """
        :return: The forward vector in world coordinates
        """
        return self.transform.position + self.transform.forward

    def get_proj(self, dtype: object = 'f4') -> Matrix44:
        """
        Generates a projection matrix based on the cameras current parameters
        :param dtype: The dtype all values in the matrix should be converted to
        :return: A matrix representing a projection
        """
        return self._frustum.get_proj(dtype)

    def get_view(self, dtype: object = 'f4') -> Matrix44:
        """
        Generates a view matrix based on the cameras current parameters
        :param dtype: The dtype all values in the matrix should be converted to
        :return: A matrix representing a view
        """
        return self._frustum.get_view(dtype)

    def get_mat(self, dtype: object = 'f4') -> Matrix44:
        """
        Generates a matrix based on the cameras current parameters
        :param dtype: The dtype all values in the matrix should be converted to
        :return: A matrix representing the combined projection and view matrix
        """
        return self._frustum.get_mat(dtype)

    def look_at(self, target: Vector3, up: Optional[Vector3] = None) -> None:
        """
        Changes the rotation of this camera so its forward vector points at the target
        :param target: The target to point to
        :param up: The new up vector (optional)
        """
        self.transform.look_at(target, up)
