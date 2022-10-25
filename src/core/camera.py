from typing import Optional

from pyrr import Matrix44, Quaternion, Vector3

from src.core.errors import MutuallyExclusiveArgsError
from src.core.transform import Transform


class Camera:
    """
    Class representing a camera in 3D space
    :cvar fovy: The field of view in y direction in degrees
    :cvar aspect_ratio: The aspect ratio of the view
    :cvar near: Distance from the camera to the near clipping plane
    :cvar far: Distance from the camera to the far clipping plane
    :cvar transform: Transform describing position and rotation of the camera
    """
    fovy: float
    aspect_ratio: float
    orthogonal: bool
    orthogonal_size: tuple[int, int]
    near: float
    far: float
    transform: Transform

    def __init__(self, fovy: float = 60, aspect_ratio: float = 1.0,
                 orthogonal: bool = False,
                 orthogonal_size: tuple[int, int] = (16, 16),
                 near: float = 0.1, far: float = 10000,
                 position: Vector3 = Vector3([0.0, 0.0, 0.0]),
                 rotation: Optional[Quaternion] = None,
                 forward: Optional[Vector3] = None, up: Optional[Vector3] = None):
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
        self.fovy = fovy
        self.aspect_ratio = aspect_ratio
        self.orthogonal = orthogonal
        self.orthogonal_size = orthogonal_size
        self.near = near
        self.far = far

        rot_set = rotation is not None
        dirs_set = up is not None and forward is not None

        if dirs_set:
            if rot_set:
                raise MutuallyExclusiveArgsError('Up and front are mutually exclusive from rotation')

            if up.dot(forward) > 0.000125:
                raise ValueError('The up and forward vector have to be orthogonal')

            target = position + forward.normalized
            rotation = Quaternion.from_matrix(Matrix44.look_at(position, target, up.normalized))
        else:
            rotation = rotation if rot_set else Quaternion([0.0, 0.0, 0.0, 1.0])

        self.transform = Transform(position, rotation)

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
        return Matrix44.look_at(self.transform.position, self.target, self.transform.up, dtype=dtype)

    def get_mat(self, dtype: object = 'f4') -> Matrix44:
        """
        Generates a matrix based on the cameras current parameters
        :param dtype: The dtype all values in the matrix should be converted to
        :return: A matrix representing the combined projection and view matrix
        """
        return self.get_proj(dtype=dtype) * self.get_view(dtype=dtype)

    def look_at(self, target: Vector3, up: Optional[Vector3] = None) -> None:
        """
        Changes the rotation of this camera so its forward vector points at the target
        :param target: The target to point to
        :param up: The new up vector (optional)
        """
        self.transform.look_at(target, up)
