from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class PixelOrigin(Enum):
    """
    Enum representing a 2D coordinate origin.
    The first two bits represent the vertical while the last two represent horizontal positioning:
    - 00xx: top
    - 01xx: vertical center
    - 10xx: bottom
    - xx00: left
    - xx01: horizontal center
    - xx10: right
    """
    TopLeft =      0b0000
    TopCenter =    0b0001
    TopRight =     0b0010

    CenterLeft =   0b0100
    Center =       0b0101
    CenterRight =  0b0110

    BottomLeft =   0b1000
    BottomCenter = 0b1001
    BottomRight =  0b1010

@dataclass
class Distortion:
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float

    def to_array(self) -> NDArray:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])
