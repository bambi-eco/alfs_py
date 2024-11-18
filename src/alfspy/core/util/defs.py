import os
import pathlib
import sys
from enum import Enum
from typing import Final, Type, Union

from pyrr import Vector3

# FUNDAMENTAL
PATH_SEP: Final[str] = os.sep
EPSILON: Final[float] = sys.float_info.epsilon

CPP_INT_MAX: Final[int] = 2147483648
MAX_TEX_DIM: Final[int] = 4096 * 2

# TYPES
Number: Type = Union[int, float]
Color: Type = Union[Number, tuple[Number, Number, Number], tuple[Number, Number, Number, Number]]


class StrEnum(str, Enum):
    pass

# META
PACKAGE_NAME: Final[str] = __name__.split('.')[0]

# DIRECTORIES
MODULE_DIR: Final[str] = str(pathlib.Path(__file__).parent.parent.parent.absolute())

# COLORS
BLACK: Final[tuple[float, float, float]] = (0.0, 0.0, 0.0)
MAGENTA: Final[tuple[float, float, float]] = (255.0, 255.0, 255.0)
TRANSPARENT: Final[tuple[float, float, float, float]] = (0.0, 0.0, 0.0, 0.0)

# VECTORS
ORIGIN: Final[Vector3] = Vector3([0, 0, 0], dtype=float)
UP: Final[Vector3] = Vector3([0, 1, 0], dtype=float)
FORWARD: Final[Vector3] = Vector3([0, 0, -1], dtype=float)
RIGHT: Final[Vector3] = Vector3([1, 0, 0], dtype=float)
DOWN: Final[Vector3] = Vector3([0, -1, 0], dtype=float)
BACK: Final[Vector3] = Vector3([0, 0, 1], dtype=float)
LEFT: Final[Vector3] = Vector3([-1, 0, 0], dtype=float)


if __name__ == '__main__':
    def main() -> None:
        globs = globals().copy()
        for name, value in globs.items():
            if not name.startswith('__') and name.isupper():
                print(f'{name}: {value}')
    main()
