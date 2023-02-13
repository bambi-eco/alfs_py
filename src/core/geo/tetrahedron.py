from dataclasses import dataclass

from pyrr import Vector3


@dataclass
class Tetrahedron:
    """
    Class representing a tetrahedron
    """
    a: Vector3
    b: Vector3
    c: Vector3
    d: Vector3

    @property
    def volume(self):
        """
        :return: The volume captured by the given points
        """
        return ((self.a - self.b).dot((self.b - self.d).cross(self.c - self.d))) / 6.0
