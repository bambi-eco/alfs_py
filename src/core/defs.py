import os
import pathlib
import sys
from typing import Final

from pyrr import Vector3

# FUNDAMENTAL
PATH_SEP: Final[str] = os.sep
EPSILON: Final[float] = sys.float_info.epsilon

# DIRECTORIES
ROOT_DIR: Final[str] = f'{str(pathlib.Path(__file__).parent.parent.parent.absolute())}{PATH_SEP}'
MODULE_DIR: Final[str] = f'{str(pathlib.Path(__file__).parent.parent.absolute())}{PATH_SEP}'
INPUT_DIR: Final[str] = f'{ROOT_DIR}input{PATH_SEP}'
OUTPUT_DIR: Final[str] = f'{ROOT_DIR}output{PATH_SEP}'

# SHADERS
SHADERS_PATH: Final[str] = f'{MODULE_DIR}{PATH_SEP}shaders{PATH_SEP}'
COL_VERT_SHADER_PATH: Final[str] = f'{SHADERS_PATH}col.vert.glsl'
COL_FRAG_SHADER_PATH: Final[str] = f'{SHADERS_PATH}col.frag.glsl'
TEX_VERT_SHADER_PATH: Final[str] = f'{SHADERS_PATH}tex.vert.glsl'
TEX_FRAG_SHADER_PATH: Final[str] = f'{SHADERS_PATH}tex.frag.glsl'

# COLORS
BLACK: Final[tuple[float, float, float]] = (0.0, 0.0, 0.0)
TRANSPARENT: Final[tuple[float, float, float, float]] = (0.0, 0.0, 0.0, 0.0)

# VECTORS
ORIGIN: Final[Vector3] = Vector3([0, 0, 0])
UP: Final[Vector3] = Vector3([0, 1, 0])
FORWARD: Final[Vector3] = Vector3([0, 0, -1])
RIGHT: Final[Vector3] = Vector3([1, 0, 0])
DOWN: Final[Vector3] = Vector3([0, -1, 0])
BACK: Final[Vector3] = Vector3([0, 0, 1])
LEFT: Final[Vector3] = Vector3([-1, 0, 0])
