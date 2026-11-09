from typing import Union, Optional, Iterable, Any

import cv2
import numpy as np
from numpy.typing import NDArray

from alfspy.core.util.defs import Color


def get_first_valid(in_dict: dict[Any], keys: Iterable[Any], default: Optional[Any] = None) -> Any:
    """
    Returns the value within a dict associated with the first valid key of an iterable key source.
    :param in_dict: The dictionary to search within.
    :param keys: An iterable source of keys.
    :param default: The value to be returned when no valid key was passed (optional).
    :return: The value of default if no valid key was passed; otherwise the value associated with the first valid key.
    """
    for key in keys:
        if key in in_dict:
            return in_dict.get(key)
    return default


def is_duplicated(image: dict) -> bool:
    """
    Tells whether a shot JSON ``images`` entry is flagged as a duplicated frame.

    bambi_detection's ``TimedPoseExtractor.extract(flag_duplicates=True)`` keeps frozen thermal
    frames (e.g. while the shutter is closed for calibration) and marks them with
    ``"duplicated": true``. Their pose is valid, but their image is a copy of the previous frame.
    :param image: One entry of the ``images`` list.
    :return: ``True`` if the entry carries a truthy ``duplicated`` key; otherwise ``False``.
    """
    return bool(image.get('duplicated', False))


def frame_is_duplicated(poses: dict, frame_idx: int) -> bool:
    """
    Tells whether the frame at ``frame_idx`` of a loaded shot JSON file is flagged as duplicated.

    Accepts the usual list form of ``images`` and the dict form keyed by the frame index as a
    string. A frame that is not in the file counts as not duplicated.
    :param poses: The loaded JSON data.
    :param frame_idx: The frame index, i.e., the position in the ``images`` list.
    :return: ``True`` if the frame exists and carries ``"duplicated": true``; otherwise ``False``.
    """
    images = poses.get('images', [])
    if isinstance(images, dict):
        image = images.get(str(frame_idx), images.get(frame_idx))
    elif 0 <= frame_idx < len(images):
        image = images[frame_idx]
    else:
        image = None
    return image is not None and is_duplicated(image)


def compare_color(col_a: Union[NDArray, Iterable, int, float], col_b: Union[NDArray, Iterable, int, float]) -> bool:
    """
    Compares two color values.
    :param col_a: The first color.
    :param col_b: The second color.
    :return: ``False`` if any component of the two colors differs; otherwise ``True``.
    """
    if not isinstance(col_a, np.ndarray):
        col_a = np.array(col_a)
    if not isinstance(col_a, np.ndarray):
        col_b = np.array(col_b)
    return (col_a == col_b).all()


def nearest_int(val: float) -> int:
    """
    Rounds a float to its nearest integer. A value of a half will be rounded up, i.e., this function rounds half up.
    :param val: The value to round.
    :return: The nearest integer of the given value.
    """
    return int(val + 0.5)


def is_same(obj: Any, other: Any) -> bool:
    """
    Checks whether two objects are instances of the same class.
    :param obj: The first object to check.
    :param other: The object whose type should be used for the check.
    :return: ``True`` if `obj` is an instance of the same type as `other`; otherwise ``False``.
    """
    return isinstance(obj, type(other))


def gen_checkerboard_tex(tile_per_side: int, tile_size: int, tile_color: Color, non_tile_color: Color,
                         dtype: object = float) -> NDArray:
    """
    Generates a numpy array representing a checkerboard texture.
    :param tile_per_side: The amount of tiles per side.
    :param tile_size: The side length of a tile in pixels.
    :param tile_color: The color of a tile.
    :param non_tile_color: The color of the lack of a tile.
    :param dtype: The dtype to convert all values within the numpy to (defaults to float).
    :return: A numpy array representing a checkerboard texture.
    """
    t_c = np.array(tile_color).reshape((-1))
    nt_c = np.array(non_tile_color).reshape((-1))

    if t_c.shape != nt_c.shape:
        raise ValueError('Both given colors have to have the same amount of components')

    if tile_per_side == 0 or tile_size == 0:
        return np.empty((0, 0, t_c.shape[0]), dtype=dtype)

    depth = t_c.shape[0]
    tile = np.empty((tile_size, tile_size, depth), dtype=dtype)
    n_tile = np.empty_like(tile)
    tile[..., :] = t_c
    n_tile[..., :] = nt_c

    odd_line = cv2.hconcat([tile if i % 2 == 0 else n_tile for i in range(0, tile_per_side)])
    even_line = cv2.hconcat([tile if i % 2 == 1 else n_tile for i in range(0, tile_per_side)])
    result = cv2.vconcat([odd_line if i % 2 == 0 else even_line for i in range(0, tile_per_side)])

    return result
