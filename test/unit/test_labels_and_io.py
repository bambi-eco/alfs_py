"""Label parsing, YOLO conversion, image helpers and texture data handling."""

import json
import os

import numpy as np
import pytest
import torch

from alfspy.core.rendering import Resolution, TextureData
from alfspy.core.util.basic import gen_checkerboard_tex, get_first_valid, nearest_int
from alfspy.core.util.image import integral, overlay, replace_color, split_components
from alfspy.render.projection import (
    Label,
    _load_correction,
    _to_yolo,
    _write_yolo,
    frame_index_from_path,
    parse_mot_labels,
)


# ──────────────────────────────── MOT parsing ───────────────────────────────


def test_parse_mot_labels_groups_by_frame(tmp_path):
    mot = tmp_path / 'gt.txt'
    mot.write_text(
        '1,7,10.0,20.0,5.0,6.0,1,1,1\n'
        '1,8,30.0,40.0,5.0,6.0,1,1,1\n'
        '2,7,12.0,22.0,5.0,6.0,1,1,1\n'
    )

    labels = parse_mot_labels(str(mot))

    assert set(labels) == {1, 2}
    assert len(labels[1]) == 2
    assert labels[1][0].track_id == 7
    assert labels[1][0].bbox == (10.0, 20.0, 5.0, 6.0)


def test_parse_mot_labels_defaults_class_id_to_the_track_column(tmp_path):
    mot = tmp_path / 'gt.txt'
    mot.write_text('1,7,10.0,20.0,5.0,6.0,1,1,1\n')

    labels = parse_mot_labels(str(mot))

    assert labels[1][0].class_id == 7, 'column 1 is the documented default'


def test_parse_mot_labels_honours_an_explicit_class_column(tmp_path):
    mot = tmp_path / 'gt.txt'
    mot.write_text('1,7,10.0,20.0,5.0,6.0,1,3,1\n')

    labels = parse_mot_labels(str(mot), class_id_column=7)

    assert labels[1][0].class_id == 3


def test_parse_mot_labels_skips_blank_and_short_lines(tmp_path):
    mot = tmp_path / 'gt.txt'
    mot.write_text('\n1,7,10.0,20.0,5.0,6.0\n1,8,1\n   \n')

    labels = parse_mot_labels(str(mot))

    assert len(labels[1]) == 1


def test_parse_mot_labels_returns_empty_for_a_missing_file(tmp_path):
    assert parse_mot_labels(str(tmp_path / 'nope.txt')) == {}


# ───────────────────────────────── YOLO output ──────────────────────────────


def test_to_yolo_centres_and_normalises():
    result = _to_yolo((100.0, 200.0, 300.0, 400.0), 1000, 1000)

    x_centre, y_centre, width, height = (float(v) for v in result.split())
    assert x_centre == pytest.approx(0.2)
    assert y_centre == pytest.approx(0.3)
    assert width == pytest.approx(0.2)
    assert height == pytest.approx(0.2)


def test_to_yolo_returns_none_for_a_box_entirely_outside():
    assert _to_yolo((-500.0, -500.0, -400.0, -400.0), 1000, 1000) is None


def test_to_yolo_keeps_a_partially_visible_box():
    assert _to_yolo((-50.0, 100.0, 50.0, 200.0), 1000, 1000) is not None


def test_write_yolo_skips_unprojectable_labels(tmp_path):
    from alfspy.render.projection import ProjectedLabel

    labels = [
        ProjectedLabel(1, (0.0, 0.0, 10.0, 10.0), 1, '0.5 0.5 0.1 0.1'),
        ProjectedLabel(2, (0.0, 0.0, 10.0, 10.0), 2, None),
    ]
    out = tmp_path / 'labels.txt'

    written = _write_yolo(str(out), labels)

    assert written == 1
    assert out.read_text().strip() == '1 0.5 0.5 0.1 0.1'


def test_write_yolo_creates_the_parent_directory(tmp_path):
    from alfspy.render.projection import ProjectedLabel

    out = tmp_path / 'nested' / 'deeper' / 'labels.txt'
    _write_yolo(str(out), [ProjectedLabel(0, (0, 0, 1, 1), None, '0.5 0.5 0.1 0.1')])

    assert out.exists()


def test_label_corners_are_clockwise_from_the_top_left():
    label = Label(class_id=0, bbox=(10.0, 20.0, 30.0, 40.0))

    assert label.corners() == [10.0, 20.0, 40.0, 20.0, 40.0, 60.0, 10.0, 60.0]


def test_frame_index_from_path():
    assert frame_index_from_path('146_00002120.png') == 2120
    assert frame_index_from_path(os.path.join('a', 'b', 'flight_42.jpg')) == 42


def test_frame_index_from_path_rejects_a_nameless_index():
    with pytest.raises((IndexError, ValueError)):
        frame_index_from_path('noindex.png')


# ───────────────────────────────── correction ───────────────────────────────


def test_load_correction_reads_translation_and_rotation(tmp_path):
    path = tmp_path / 'correction.json'
    path.write_text(json.dumps({
        'translation': {'x': 1.0, 'y': -2.0, 'z': 0.5},
        'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.25},
    }))

    transform, translation, eulers = _load_correction(str(path))

    np.testing.assert_allclose(np.asarray(translation), [1.0, -2.0, 0.5], atol=1e-6)
    np.testing.assert_allclose(np.asarray(eulers), [0.0, 0.0, 0.25], atol=1e-6)
    np.testing.assert_allclose(np.asarray(transform.position), [1.0, -2.0, 0.5], atol=1e-6)


def test_load_correction_defaults_to_neutral(tmp_path):
    path = tmp_path / 'correction.json'
    path.write_text('{}')

    _, translation, eulers = _load_correction(str(path))

    np.testing.assert_allclose(np.asarray(translation), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(eulers), [0.0, 0.0, 0.0])


# ─────────────────────────────── image helpers ──────────────────────────────


def test_overlay_uses_the_foreground_alpha():
    background = np.zeros((4, 4, 4), dtype=np.uint8)
    foreground = np.zeros((4, 4, 4), dtype=np.uint8)
    foreground[..., 0] = 255
    foreground[..., 3] = 255

    result = overlay(background, foreground)

    assert (result[..., 0] == 255).all()


def test_overlay_keeps_the_background_where_the_foreground_is_transparent():
    background = np.full((4, 4, 4), 200, dtype=np.uint8)
    foreground = np.zeros((4, 4, 4), dtype=np.uint8)

    result = overlay(background, foreground)

    assert (result[..., 0] == 200).all()


def test_overlay_rejects_mismatched_shapes():
    assert overlay(np.zeros((4, 4, 4), np.uint8), np.zeros((5, 5, 4), np.uint8)) is None


def test_replace_color():
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    img[1, 1] = (255, 0, 0)

    result = replace_color(img, (255, 0, 0), (0, 255, 0))

    np.testing.assert_array_equal(result[1, 1], [0, 255, 0])


def test_split_components():
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    xs, ys, zs = split_components(arr)

    np.testing.assert_array_equal(xs, [1.0, 4.0])
    np.testing.assert_array_equal(ys, [2.0, 5.0])
    np.testing.assert_array_equal(zs, [3.0, 6.0])


def test_integral_normalises_by_the_alpha_overlap_count():
    a = np.zeros((2, 2, 4), dtype=np.float32)
    a[..., 0] = 1.0
    a[..., 3] = 1.0

    result = integral([a, a.copy()], dtype=np.float32)

    np.testing.assert_allclose(result[..., 0], 1.0, atol=1e-6)


def test_gen_checkerboard_tex_alternates():
    tex = gen_checkerboard_tex(2, 2, (255, 255, 255), (0, 0, 0))

    assert tex.shape == (4, 4, 3)
    assert tex[0, 0, 0] != tex[0, 2, 0]


def test_gen_checkerboard_tex_rejects_mismatched_colour_widths():
    with pytest.raises(ValueError):
        gen_checkerboard_tex(2, 2, (255, 255, 255), (0, 0))


def test_get_first_valid_picks_the_first_present_key():
    assert get_first_valid({'b': 2, 'c': 3}, ['a', 'b', 'c']) == 2
    assert get_first_valid({}, ['a'], default='fallback') == 'fallback'


def test_nearest_int_rounds_half_up():
    assert nearest_int(2.5) == 3
    assert nearest_int(2.49) == 2


# ─────────────────────────────── TextureData ────────────────────────────────


def test_texture_data_to_tensor_flips_and_normalises():
    img = np.zeros((2, 2, 3), dtype='f4')
    img[0, :, 0] = 255.0  # top row red

    tensor = TextureData(img).to_tensor()

    assert tensor.shape == (2, 2, 3)
    assert float(tensor.max()) <= 1.0
    # After the flip into GL bottom-up order the red row is last.
    assert float(tensor[1, 0, 0]) == pytest.approx(1.0)
    assert float(tensor[0, 0, 0]) == pytest.approx(0.0)


def test_texture_data_to_tensor_leaves_normalised_input_alone():
    img = np.full((2, 2, 3), 0.5, dtype='f4')

    tensor = TextureData(img).to_tensor()

    np.testing.assert_allclose(tensor.numpy(), 0.5, atol=1e-6)


def test_texture_data_to_tensor_matches_to_bytes():
    """The two encoders must agree, since lazy shot loading still goes through to_bytes."""
    rng = np.random.default_rng(3)
    img = rng.uniform(0, 255, size=(5, 7, 4)).astype('f4')
    data = TextureData(img)

    from_bytes = np.frombuffer(data.to_bytes(), dtype='f4').reshape(5, 7, 4)

    np.testing.assert_allclose(data.to_tensor().numpy(), from_bytes, atol=1e-6)


def test_texture_data_reports_dimensions():
    data = TextureData(np.zeros((7, 5, 4), dtype='f4'))

    assert data.width == 5
    assert data.height == 7
    assert data.byte_size('f4') == 5 * 7 * 4 * 4


def test_texture_data_scale_to_fit_shrinks_oversized_textures():
    data = TextureData(np.zeros((64, 64, 4), dtype='f4'))
    original = data.byte_size('f4')

    data.scale_to_fit(original // 4, dtype='f4')

    assert data.byte_size('f4') < original


def test_resolution_is_iterable_and_indexable():
    resolution = Resolution(1920, 1080)

    assert tuple(resolution) == (1920, 1080)
    assert resolution[0] == 1920
    assert resolution[1] == 1080
