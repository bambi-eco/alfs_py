"""Frames flagged ``"duplicated": true`` in a shot JSON file.

bambi_detection's ``TimedPoseExtractor.extract(flag_duplicates=True)`` keeps frozen thermal
frames (shutter calibration) in ``poses.json`` and flags them instead of dropping them, so the
``images`` list stays aligned with the video. Readers must leave those frames out of renders
without shifting any index.
"""
import json
import os

import pytest

from alfspy.core.rendering.shot_json import select_json_images, shot_params_from_json
from alfspy.core.util.basic import frame_is_duplicated, is_duplicated


def _images(n=8, duplicated=(3, 4)):
    images = []
    for i in range(n):
        entry = {'imagefile': f'{i}.png', 'location': [float(i), 0.0, 30.0], 'rotation': [0.0, 0.0, 0.0]}
        if i in duplicated:
            entry['duplicated'] = True
        images.append(entry)
    return images


def _names(images):
    return [image['imagefile'] for image in images]


def test_is_duplicated():
    assert is_duplicated({'duplicated': True})
    assert not is_duplicated({'duplicated': False})
    assert not is_duplicated({})


def test_frame_is_duplicated_list_and_dict_forms():
    poses = {'images': _images()}
    assert frame_is_duplicated(poses, 3)
    assert not frame_is_duplicated(poses, 2)
    assert not frame_is_duplicated(poses, 99)  # missing frame
    assert not frame_is_duplicated(poses, -1)

    as_dict = {'images': {str(i): image for i, image in enumerate(_images())}}
    assert frame_is_duplicated(as_dict, 4)
    assert not frame_is_duplicated(as_dict, 5)


def test_select_drops_duplicated_frames():
    assert _names(select_json_images(_images())) == ['0.png', '1.png', '2.png', '5.png', '6.png', '7.png']


def test_select_keeps_duplicated_frames_on_request():
    assert len(select_json_images(_images(), skip_duplicates=False)) == 8


def test_window_is_taken_on_raw_indices():
    # Frames 2..5 are asked for; 3 and 4 are frozen. The window must not slide on to 6 and 7.
    assert _names(select_json_images(_images(), count=4, initial_skip=2)) == ['2.png', '5.png']


def test_step_is_taken_on_raw_indices():
    assert _names(select_json_images(_images(), skip=2)) == ['0.png', '2.png', '6.png']
    assert _names(select_json_images(_images(), count=5, initial_skip=1, skip=2)) == ['1.png', '5.png']


def test_select_without_duplicates_matches_plain_slicing():
    images = _images(duplicated=())
    assert select_json_images(images, count=3, initial_skip=2, skip=1) == images[2:5]
    assert select_json_images(images) == images


def test_select_rejects_non_positive_skip():
    with pytest.raises(ValueError):
        select_json_images(_images(), skip=0)


def test_shot_params_skip_duplicated_frames():
    params = shot_params_from_json({'images': _images()}, ctx=None, image_dir='dir', fovy=42.0)
    assert [os.path.basename(p[1]) for p in params] == [f'{i}.png' for i in (0, 1, 2, 5, 6, 7)]
    assert all(p[4] == 42.0 for p in params)  # fovy fallback still applies


def test_from_json_and_read_shots_skip_duplicated_frames(tmp_path, ctx):
    from alfspy.core.rendering import CtxShot
    from alfspy.render.data import BaseSettings
    from alfspy.render.render import read_shots

    poses_file = tmp_path / 'poses.json'
    poses_file.write_text(json.dumps({'images': _images()}))

    def stems(shots):
        return [int(os.path.basename(shot._img_file).split('.')[0]) for shot in shots]

    shots = CtxShot.from_json(str(poses_file), ctx, lazy=True)
    assert stems(shots) == [0, 1, 2, 5, 6, 7]
    assert len(CtxShot.from_json(str(poses_file), ctx, lazy=True, skip_duplicates=False)) == 8

    se = BaseSettings(count=4, initial_skip=2, lazy=True)
    assert stems(read_shots(str(poses_file), ctx, se)) == [2, 5]

    se = BaseSettings(count=4, initial_skip=2, lazy=True, skip_duplicates=False)
    assert stems(read_shots(str(poses_file), ctx, se)) == [2, 3, 4, 5]
