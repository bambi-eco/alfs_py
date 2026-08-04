from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray, DTypeLike


class PixelOriginFlag(Enum):
    """
    Enum representing flags to check pixel coordinate origins to via bitwise comparison.
    """
    Top              = 0b100000
    VerticalCenter   = 0b010000
    Bottom           = 0b001000
    Left             = 0b000100
    HorizontalCenter = 0b000010
    Right            = 0b000001


class PixelOrigin(Enum):
    """
    Enum representing a 2D coordinate origin.
    Each bit of the values represents a flag:
    [Top][Vertical Center][Bottom][Left][Horizontal Center][Right]
    """
    TopLeft =      0b100100
    TopCenter =    0b100010
    TopRight =     0b100001

    CenterLeft =   0b010100
    Center =       0b010010
    CenterRight =  0b010001

    BottomLeft =   0b001100
    BottomCenter = 0b001010
    BottomRight =  0b001001


@dataclass
class Distortion:
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float

    def to_array(self, dtype: DTypeLike = np.float32) -> NDArray:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=dtype)
