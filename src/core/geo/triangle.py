from dataclasses import dataclass

from pyrr import Vector3


@dataclass
class Triangle:
    """
    Class representing a triangle
    """
    a: Vector3
    b: Vector3
    c: Vector3

    @property
    def normal(self) -> Vector3:
        """
        :return: The normal spanned by a, b, and c
        """
        return (self.b-self.a).cross(self.c-self.a).normalized