"""
Backend-neutral reading of shot JSON files (``poses.json`` / ``matched_poses.json``).

Every backend's ``CtxShot.from_json`` builds its shots from the parameter tuples returned here,
so all of them select and parse the ``images`` entries the same way.
"""
from typing import Any, Optional, Sequence

from pyrr import Vector3, Quaternion

from alfspy.core.geo import Transform
from alfspy.core.util.basic import get_first_valid, is_duplicated
from alfspy.core.util.defs import PATH_SEP
from alfspy.core.util.pyrrs import quaternion_from_drone_pose


def select_json_images(images: Sequence[dict], count: Optional[int] = None, initial_skip: int = 0,
                       skip: int = 1, skip_duplicates: bool = True) -> list[dict]:
    """
    Selects the ``images`` entries of a shot JSON file to turn into shots.

    ``initial_skip``, ``count`` and ``skip`` are positions in the full ``images`` list, so the
    chosen window does not move when duplicated frames are dropped. Duplicated frames (entries
    with ``"duplicated": true``, e.g. frozen thermal frames during shutter calibration) are only
    removed from that window afterwards.

    :param images: The ``images`` list of the JSON file.
    :param count: The number of entries in the window (optional, defaults to all remaining ones).
    :param initial_skip: The number of entries to skip at the beginning.
    :param skip: The step between two selected entries.
    :param skip_duplicates: Whether entries flagged as duplicated should be left out (defaults to ``True``).
    :return: The selected entries in their original order.
    """
    if skip < 1:
        raise ValueError(f'skip must be at least 1, got {skip}')
    stop = None if count is None else initial_skip + count
    window = images[initial_skip:stop:skip]
    if skip_duplicates:
        window = [image for image in window if not is_duplicated(image)]
    return list(window)


def shot_params_from_json(data: dict, ctx: Any, count: Optional[int] = None, image_dir: Optional[str] = None,
                          fovy: float = 60.0, correction: Optional[Transform] = None, lazy: bool = False,
                          initial_skip: int = 0, skip: int = 1, skip_duplicates: bool = True) \
        -> list[tuple[Any, str, Vector3, Quaternion, float, float, Optional[Transform], bool]]:
    """
    Parses a loaded shot JSON file into the constructor arguments of a backend ``CtxShot``.

    :param data: The loaded JSON data.
    :param ctx: The context to attach the shots to.
    :param count: The number of ``images`` entries to read (optional), see :func:`select_json_images`.
    :param image_dir: The directory of the referenced images.
    :param fovy: The default fovy to use when an entry does not provide one.
    :param correction: The general correction to be applied to all shots (optional).
    :param lazy: Whether the shots should be lazy loaded.
    :param initial_skip: The number of ``images`` entries to skip at the beginning.
    :param skip: The step between two selected entries.
    :param skip_duplicates: Whether entries flagged as duplicated should be left out (defaults to ``True``).
    :return: One ``(ctx, file, position, rotation, fovy, 1, correction, lazy)`` tuple per shot.
    """
    images = select_json_images(data.get('images', []), count, initial_skip, skip, skip_duplicates)

    shot_params = []
    for image in images:
        img_file = get_first_valid(image, ['imagefile', 'file', 'image'])
        position = get_first_valid(image, ['location', 'pos', 'loc'])
        rotation = get_first_valid(image, ['rotation', 'rot', 'quaternion'])
        fov = get_first_valid(image, ['fovy', 'fov', 'fieldofview'])

        if img_file is None or position is None or rotation is None:
            raise ValueError('The given JSON file does not contain valid data')

        rot_len = len(rotation)
        if rot_len == 3:
            rotation = quaternion_from_drone_pose(rotation)
        elif rot_len == 4:
            rotation = Quaternion(rotation)
        else:
            raise ValueError(f'Invalid rotation format of length {rot_len}: {rotation}')

        if fov is None:
            fov = fovy
        elif isinstance(fov, list):
            fov = fov[0]

        img_file = f'{image_dir}{PATH_SEP}{img_file}'

        shot_params.append((ctx, img_file, Vector3(position), rotation, fov, 1, correction, lazy))
    return shot_params
