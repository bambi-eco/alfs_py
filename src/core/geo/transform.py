from typing import Optional

from pyrr import Quaternion, Vector3, Matrix44

from src.core.defs import UP, DOWN, LEFT, RIGHT, FORWARD, BACK


class Transform:
    """
    Class that represents a position, rotation and scale in 3D space
    """
    _position: Vector3
    _rotation: Quaternion
    _scale: Vector3

    def __init__(self, position: Optional[Vector3] = None, rotation: Optional[Quaternion] = None,
                 scale: Optional[Vector3] = None):
        """
        Class representing a position, rotation, and scale of geometry in 3D space
        :param position: The position in 3D space (optional)
        :param rotation: The rotation in 3D space (optional)
        :param scale: The scale along each axis (optional)
        """
        self._position = Vector3(position) if position is not None else Vector3([0.0, 0.0, 0.0])
        self._rotation = Quaternion(rotation) if rotation is not None else Quaternion([0.0, 0.0, 0.0, 1.0])
        self._scale = Vector3(scale) if scale is not None else Vector3([1.0, 1.0, 1.0])

    @staticmethod
    def from_up_forward(up: Vector3, forward: Vector3, position: Optional[Vector3] = None, scale: Optional[Vector3] = None):
        """
        Creates a ``Transform`` object based on the rotation defined by an up and forward vector
        :param up: The up vector of the transform
        :param forward: The forward vector of the transform
        :param position: The position in 3D space (optional)
        :param scale: The scale along each axis (optional)
        :return: A ``Transform`` instance initialized respectively
        """
        if up.dot(forward) > 0.000125:
            raise ValueError('The up and forward vector have to be orthogonal')

        target = position + forward.normalized
        rotation = Quaternion.from_matrix(Matrix44.look_at(position, target, up.normalized))
        return Transform(position, rotation, scale)

    @property
    def trans_mat(self) -> Matrix44:
        """
        :return: A matrix representing the translation of the transform
        """
        return Matrix44.from_translation(self._position)

    @property
    def rot_mat(self) -> Matrix44:
        """
        :return: A matrix representing the rotation of the transform
        """
        return Matrix44.from_quaternion(self._rotation)

    @property
    def scale_mat(self) -> Matrix44:
        """
        :return: A matrix representing the scale of the transform
        """
        return Matrix44.from_scale(self._scale)

    @property
    def mat(self) -> Matrix44:
        """
        :return: A matrix representing the entire transform
        """
        return self.trans_mat * self.rot_mat * self.scale_mat

    @property
    def up(self) -> Vector3:
        """
        :return: The up vector of this transform
        """
        return self.rot_mat * UP

    @property
    def down(self) -> Vector3:
        """
        :return: The down vector of this transform
        """
        return self.rot_mat * DOWN

    @property
    def left(self) -> Vector3:
        """
        :return: The left vector of this transform
        """
        return self.rot_mat * LEFT

    @property
    def right(self) -> Vector3:
        """
        :return: The right vector of this transform
        """
        return self.rot_mat * RIGHT

    @property
    def forward(self) -> Vector3:
        """
        :return: The forward vector of this transform
        """
        return self.rot_mat * FORWARD

    @property
    def back(self) -> Vector3:
        """
        :return: The back vector of this transform
        """
        return self.rot_mat * BACK

    @property
    def target(self) -> Vector3:
        """
        :return: The forward vector in world coordinates
        """
        return self.position + self.forward

    @property
    def position(self) -> Vector3:
        """
        :return: The position held by this transform
        """
        return Vector3(self._position)

    @position.setter
    def position(self, position: Vector3) -> None:
        """
        Sets the position held by this transform
        :param position: The new position
        """
        self._position = Vector3(position)

    @property
    def rotation(self) -> Quaternion:
        """
        :return: The rotation held by this transform
        """
        return Quaternion(self._rotation)

    @rotation.setter
    def rotation(self, rotation: Quaternion) -> None:
        """
        Sets the rotation held by this transform
        :param rotation: The new rotation
        """
        self._rotation = Quaternion(rotation)

    @property
    def scale(self) -> Vector3:
        """
        :return: The scale held by this transform
        """
        return Vector3(self._scale)

    @scale.setter
    def scale(self, scale: Vector3) -> None:
        """
        Sets the scale held by this transform
        :param scale: The new scale
        """
        self._scale = Vector3(scale)

    def look_at(self, target: Vector3, up: Optional[Vector3] = None) -> None:
        """
        Changes the rotation of this transform so its forward vector points at the target.
        :param target: The target to point to
        :param up: The new up vector (optional)
        """
        if up is None:
            up = UP

        self._rotation = Quaternion.from_matrix(Matrix44.look_at(self._position, target, up))

    def translate(self, translation: Vector3) -> None:
        """
        Translates the transform by the given translation
        :param translation: The translation to apply
        """
        self._position += translation

    def move_up(self, distance: float) -> None:
        """
        Translates the transform along its up vector by the given distance
        :param distance: The distance to translate
        """
        self.translate(self._rotation * Vector3([0, distance, 0]))

    def move_down(self, distance: float) -> None:
        """
        Translates the transform along its down vector by the given distance
        :param distance: The distance to translate
        """
        self.move_up(-distance)

    def move_right(self, distance: float) -> None:
        """
        Translates the transform along its right vector by the given distance
        :param distance: The distance to translate
        """
        self.translate(self._rotation * Vector3([distance, 0, 0]))

    def move_left(self, distance: float) -> None:
        """
        Translates the transform along its left vector by the given distance
        :param distance: The distance to translate
        """
        self.move_right(-distance)

    def move_forward(self, distance: float) -> None:
        """
        Translates the transform along its forward vector by the given distance
        :param distance: The distance to translate
        """
        self.translate(self._rotation * Vector3([0, 0, distance]))

    def move_back(self, distance: float) -> None:
        """
        Translates the transform along its back vector by the given distance
        :param distance: The distance to translate
        """
        self.move_forward(-distance)
