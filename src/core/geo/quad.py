from dataclasses import dataclass

from pyrr import Vector3

from src.core.geo.tetrahedron import Tetrahedron
from src.core.util.basic import vector_mean
from src.core.util.defs import EPSILON


@dataclass
class Quad:
    """
    Class representing a quad
    """
    a: Vector3
    b: Vector3
    c: Vector3
    d: Vector3

    @property
    def normal(self) -> Vector3:
        """
        :return: The normal spanned by a, b, and d
        """
        return (self.b - self.a).cross(self.d - self.a).normalized

    @property
    def flat(self) -> bool:
        """
        :return: Whether all points of the quad lie on the same plane
        """
        return abs(Tetrahedron(self.a, self.b, self.c, self.d).volume) < EPSILON

    @property
    def center(self) -> Vector3:
        """
        :return: The average of the corner points
        """
        return vector_mean((self.a, self.b, self.c, self.d))
