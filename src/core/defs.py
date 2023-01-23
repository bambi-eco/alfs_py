import os
import pathlib
import sys
from enum import Enum
from typing import Final, Type, Union

from pyrr import Vector3

# FUNDAMENTAL
PATH_SEP: Final[str] = os.sep
EPSILON: Final[float] = sys.float_info.epsilon

# TYPES
Number: Type = Union[int, float]
Color: Type = Union[Number, tuple[Number, Number, Number], tuple[Number, Number, Number, Number]]

class StrEnum(str, Enum):
    pass

# DIRECTORIES
ROOT_DIR: Final[str] = f'{str(pathlib.Path(__file__).parent.parent.parent.absolute())}{PATH_SEP}'
MODULE_DIR: Final[str] = f'{str(pathlib.Path(__file__).parent.parent.absolute())}{PATH_SEP}'
INPUT_DIR: Final[str] = f'{ROOT_DIR}input{PATH_SEP}'
OUTPUT_DIR: Final[str] = f'{ROOT_DIR}output{PATH_SEP}'

# SHADERS
SHADERS_PATH: Final[str] = f'{MODULE_DIR}{PATH_SEP}shaders{PATH_SEP}'
COL_VERT_SHADER_PATH: Final[str] = f'{SHADERS_PATH}col.vert.glsl'
COL_FRAG_SHADER_PATH: Final[str] = f'{SHADERS_PATH}col.frag.glsl'
FLAT_VERT_SHADER_PATH: Final[str] = f'{SHADERS_PATH}flat.vert.glsl'
FLAT_FRAG_SHADER_PATH: Final[str] = f'{SHADERS_PATH}flat.frag.glsl'
TEX_VERT_SHADER_PATH: Final[str] = f'{SHADERS_PATH}tex.vert.glsl'
TEX_FRAG_SHADER_PATH: Final[str] = f'{SHADERS_PATH}tex.frag.glsl'
DEF_VERT_SHADER_PATH: Final[str] = f'{SHADERS_PATH}def.vert.glsl'
DEF_FRAG_SHADER_PATH: Final[str] = f'{SHADERS_PATH}def.frag.glsl'
DEF_PASS_VERT_SHADER_PATH: Final[str] = f'{SHADERS_PATH}def_pass.vert.glsl'
DEF_PASS_FRAG_SHADER_PATH: Final[str] = f'{SHADERS_PATH}def_pass.frag.glsl'

# COLORS
BLACK: Final[tuple[float, float, float]] = (0.0, 0.0, 0.0)
MAGENTA:  Final[tuple[float, float, float]] = (255.0, 255.0, 255.0)
TRANSPARENT: Final[tuple[float, float, float, float]] = (0.0, 0.0, 0.0, 0.0)

# VECTORS
ORIGIN: Final[Vector3] = Vector3([0, 0, 0], dtype=float)
UP: Final[Vector3] = Vector3([0, 1, 0], dtype=float)
FORWARD: Final[Vector3] = Vector3([0, 0, -1], dtype=float)
RIGHT: Final[Vector3] = Vector3([1, 0, 0], dtype=float)
DOWN: Final[Vector3] = Vector3([0, -1, 0], dtype=float)
BACK: Final[Vector3] = Vector3([0, 0, 1], dtype=float)
LEFT: Final[Vector3] = Vector3([-1, 0, 0], dtype=float)
