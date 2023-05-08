from dataclasses import dataclass

from pyrr import Vector3


@dataclass
class AABB:
    """
    Represents a 3D axis aligned bounding box
    :cvar p_s: The start corner of the bounding box
    :cvar p_e: The end corner of the bounding box
    """
    p_s: Vector3
    p_e: Vector3

    @property
    def center(self) -> Vector3:
        max_x, max_y, max_z = self.p_e
        min_x, min_y, min_z = self.p_s
        dhx = (max_x - min_x) / 2.0
        dhy = (max_y - min_y) / 2.0
        dhz = (max_z - min_z) / 2.0
        return Vector3((min_x + dhx, min_y + dhy, min_z + dhz))


    @property
    def width(self) -> float:
        """
        :return: The width / length in X direction of the AABB
        """
        return abs(self.p_e.x - self.p_s.x)

    @property
    def height(self) -> float:
        """
        :return: The height / length in Y direction of the AABB
        """
        return abs(self.p_e.y - self.p_s.y)

    @property
    def depth(self) -> float:
        """
        :return: The depth / length in Z direction of the AABB
        """
        return abs(self.p_e.z - self.p_s.z)

    @property
    def corners(self) -> tuple[Vector3, Vector3, Vector3, Vector3, Vector3, Vector3, Vector3, Vector3]:
        """
        :return: The corners of the AABB from front to back, bottom to top, and left to right
        """
        min_x, min_y, min_z = self.p_s
        max_x, max_y, max_z = self.p_e
        return Vector3([min_x, min_y, min_z]), Vector3([min_x, min_y, max_z]), \
            Vector3([min_x, max_y, min_z]), Vector3([min_x, max_y, max_z]), \
            Vector3([max_x, min_y, min_z]), Vector3([max_x, min_y, max_z]), \
            Vector3([max_x, max_y, min_z]), Vector3([max_x, max_y, max_z])
