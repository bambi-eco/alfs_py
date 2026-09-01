"""End-to-end tests of ``ProjectionScene`` on a generated synthetic flight.

The fixture flight has a flat DEM and nadir cameras with a known footprint, so the world
scale of every render is analytically known: an output pixel covers exactly
``ortho_size / resolution`` metres. That turns "did the render look right" into a number.
"""

import os

import numpy as np
import pytest

from alfspy.render.projection import (
    Label,
    ProjectionScene,
    ProjectionSettings,
    parse_mot_labels,
)
from alfspy.core.rendering import Resolution
from test.helpers.images import coverage

RENDER = 128


@pytest.fixture
def settings(flight):
    return ProjectionSettings(
        ortho_size=(60.0, 60.0),
        render_resolution=Resolution(RENDER, RENDER),
        fovy=flight.fovy,
        aspect_ratio=1.0,
        add_background=False,
        input_resolution=Resolution(*flight.frame_size),
    )


@pytest.fixture
def scene(flight, settings):
    with ProjectionScene(
        dem_file=flight.dem_file,
        poses_file=flight.poses_file,
        correction_file=flight.correction_file,
        mask_file=flight.mask_file,
        settings=settings,
        device='cpu',
    ) as active:
        yield active


# ──────────────────────────────── construction ──────────────────────────────


def test_scene_loads_the_flight(scene, flight):
    assert len(scene.images) == flight.frame_count
    assert scene.mask_texture is not None
    assert scene.tri_mesh is not None
    assert scene.mesh_data is not None


def test_scene_raises_after_release(scene):
    scene.release()

    with pytest.raises(RuntimeError, match='released'):
        scene.project_orthographic('unused.png', frame_idx=0)


def test_scene_release_is_idempotent(scene):
    scene.release()
    scene.release()


def test_scene_accepts_a_shared_context(flight, settings, ctx):
    scene = ProjectionScene(
        dem_file=flight.dem_file,
        poses_file=flight.poses_file,
        correction_file=flight.correction_file,
        settings=settings,
        ctx=ctx,
    )
    try:
        assert scene.ctx is ctx
    finally:
        scene.release()

    assert not ctx.released, 'a borrowed context must not be released by the scene'


# ──────────────────────────── orthographic projection ───────────────────────


def test_project_orthographic_writes_an_image(scene, flight, tmp_path):
    out = tmp_path / 'ortho' / 'frame.png'

    scene.project_orthographic(flight.frame_files[2], frame_idx=2, output_image=str(out))

    assert out.exists()

    import cv2
    rendered = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert rendered.shape[:2] == (RENDER, RENDER)


def test_project_orthographic_footprint_matches_the_camera_geometry(scene, flight, tmp_path):
    """
    A frame covering ``2 * footprint_half`` metres, rendered into a ``ortho_size`` view,
    must occupy exactly the square of their ratio.
    """
    out = tmp_path / 'frame.png'
    scene.project_orthographic(flight.frame_files[2], frame_idx=2, output_image=str(out))

    import cv2
    rendered = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert rendered.shape[2] == 4, 'expected BGRA output'

    covered = float((rendered[..., 3] > 0).mean())
    expected = (2 * flight.footprint_half) ** 2 / 60.0 ** 2

    assert covered == pytest.approx(expected, abs=0.03)


def test_project_orthographic_infers_the_frame_index_from_the_file_name(scene, flight, tmp_path):
    out = tmp_path / 'frame.png'

    scene.project_orthographic(flight.frame_files[3], output_image=str(out))

    assert out.exists()


def test_project_orthographic_without_output_still_returns_labels(scene, flight):
    labels = parse_mot_labels(flight.mot_file)

    projected = scene.project_orthographic(
        flight.frame_files[2], frame_idx=2, labels=labels.get(2, []),
    )

    assert len(projected) == len(labels.get(2, []))


# ──────────────────────────────── label projection ──────────────────────────


def test_projected_labels_land_on_the_planted_objects(scene, flight):
    """
    The full label round trip: source pixels are ray-cast onto the DEM through the source
    camera, then projected into the render camera. Both objects were planted at known world
    coordinates, so the expected render pixel is computable in closed form.
    """
    frame_idx = 2
    labels = parse_mot_labels(flight.mot_file)
    projected = scene.project_orthographic(
        flight.frame_files[frame_idx], frame_idx=frame_idx, labels=labels.get(frame_idx, []),
    )

    assert projected, 'the fixture should place at least one object in this frame'

    ortho = 60.0
    camera_x, camera_y, _ = flight.positions[frame_idx]

    by_track = {label.track_id: label for label in projected}
    for track_id, world_x, world_y in flight.objects:
        if track_id not in by_track:
            continue

        # The render camera is centred on the frame's own camera position.
        expected_x = ((world_x - camera_x) / ortho + 0.5) * RENDER
        expected_y = (0.5 - (world_y - camera_y) / ortho) * RENDER

        box = by_track[track_id].bbox
        centre_x = (box[0] + box[2]) / 2.0
        centre_y = (box[1] + box[3]) / 2.0

        assert centre_x == pytest.approx(expected_x, abs=4.0), f'track {track_id} x'
        assert centre_y == pytest.approx(expected_y, abs=4.0), f'track {track_id} y'


def test_projected_labels_are_written_in_yolo_format(scene, flight, tmp_path):
    frame_idx = 2
    labels = parse_mot_labels(flight.mot_file)
    out = tmp_path / 'labels.txt'

    scene.project_orthographic(
        flight.frame_files[frame_idx], frame_idx=frame_idx,
        labels=labels.get(frame_idx, []), output_labels=str(out),
    )

    assert out.exists()
    for line in out.read_text().strip().splitlines():
        parts = line.split()
        assert len(parts) == 5
        values = [float(v) for v in parts[1:]]
        assert all(-1.0 <= v <= 2.0 for v in values), f'YOLO values out of range: {values}'


def test_label_box_scales_with_the_render_resolution(scene, flight):
    """A 6 px box in a 64 px frame covering 40 m must scale predictably into the render."""
    frame_idx = 2
    labels = parse_mot_labels(flight.mot_file)
    source = labels.get(frame_idx, [])
    assert source

    projected = scene.project_orthographic(
        flight.frame_files[frame_idx], frame_idx=frame_idx, labels=source,
    )

    source_box = source[0].bbox[2]  # width in source pixels
    metres = source_box / flight.frame_size[0] * (2 * flight.footprint_half)
    expected = metres / 60.0 * RENDER

    rendered = projected[0].bbox
    assert (rendered[2] - rendered[0]) == pytest.approx(expected, abs=3.0)


# ────────────────────────────── light field / ALFS ──────────────────────────


def test_render_lightfield_writes_an_image(scene, flight, tmp_path):
    out = tmp_path / 'alfs' / 'integral.png'
    central = 2
    neighbours = [0, 1, 3, 4]

    scene.render_lightfield(
        central_image=flight.frame_files[central],
        neighbor_images=[flight.frame_files[i] for i in neighbours],
        central_idx=central,
        neighbor_indices=neighbours,
        output_image=str(out),
    )

    assert out.exists()

    import cv2
    rendered = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert rendered.shape[:2] == (RENDER, RENDER)
    assert (rendered[..., 3] > 0).any(), 'the integral is entirely empty'


def test_render_lightfield_covers_more_than_a_single_frame(scene, flight, tmp_path):
    """Neighbouring frames must extend the footprint along the flight direction."""
    import cv2

    single = tmp_path / 'single.png'
    scene.project_orthographic(flight.frame_files[2], frame_idx=2, output_image=str(single))

    integral = tmp_path / 'integral.png'
    neighbours = [0, 1, 3, 4]
    scene.render_lightfield(
        central_image=flight.frame_files[2],
        neighbor_images=[flight.frame_files[i] for i in neighbours],
        central_idx=2, neighbor_indices=neighbours,
        output_image=str(integral), alpha_threshold=0.1,
    )

    single_cover = coverage(cv2.imread(str(single), cv2.IMREAD_UNCHANGED))
    integral_cover = coverage(cv2.imread(str(integral), cv2.IMREAD_UNCHANGED))

    assert integral_cover > single_cover


def test_render_lightfield_merges_labels_per_track(scene, flight, tmp_path):
    central = 2
    neighbours = [1, 3]
    labels = parse_mot_labels(flight.mot_file)

    merged = scene.render_lightfield(
        central_image=flight.frame_files[central],
        neighbor_images=[flight.frame_files[i] for i in neighbours],
        central_idx=central, neighbor_indices=neighbours,
        labels=labels,
    )

    track_ids = [label.track_id for label in merged]
    assert len(track_ids) == len(set(track_ids)), 'each track must merge into one box'

    expected_tracks = set()
    for frame in [central] + neighbours:
        for label in labels.get(frame, []):
            expected_tracks.add(label.track_id)

    assert set(track_ids) == expected_tracks


def test_merged_label_encloses_every_contributing_box(scene, flight):
    central = 2
    neighbours = [1, 3]
    labels = parse_mot_labels(flight.mot_file)

    merged = scene.render_lightfield(
        central_image=flight.frame_files[central],
        neighbor_images=[flight.frame_files[i] for i in neighbours],
        central_idx=central, neighbor_indices=neighbours,
        labels=labels,
    )
    single = scene.project_orthographic(
        flight.frame_files[central], frame_idx=central, labels=labels.get(central, []),
    )

    merged_by_track = {label.track_id: label.bbox for label in merged}
    for label in single:
        if label.track_id not in merged_by_track:
            continue
        box = merged_by_track[label.track_id]
        assert box[0] <= label.bbox[0] + 1e-3
        assert box[1] <= label.bbox[1] + 1e-3
        assert box[2] >= label.bbox[2] - 1e-3
        assert box[3] >= label.bbox[3] - 1e-3


def test_render_lightfield_rejects_mismatched_index_lengths(scene, flight):
    with pytest.raises(ValueError, match='neighbor_indices'):
        scene.render_lightfield(
            central_image=flight.frame_files[2],
            neighbor_images=[flight.frame_files[1], flight.frame_files[3]],
            central_idx=2, neighbor_indices=[1],
        )


# ──────────────────────────────── determinism ───────────────────────────────


def test_repeated_projection_is_bit_identical(scene, flight, tmp_path):
    """
    The direct regression guard for the artifact class this migration removes. Renders must
    depend only on their inputs.
    """
    import cv2

    first = tmp_path / 'a.png'
    second = tmp_path / 'b.png'

    scene.project_orthographic(flight.frame_files[1], frame_idx=1, output_image=str(first))
    # Do unrelated work in between, so any shared state would show up.
    scene.render_lightfield(
        central_image=flight.frame_files[3],
        neighbor_images=[flight.frame_files[4]],
        central_idx=3, neighbor_indices=[4],
        output_image=str(tmp_path / 'noise.png'),
    )
    scene.project_orthographic(flight.frame_files[1], frame_idx=1, output_image=str(second))

    assert np.array_equal(
        cv2.imread(str(first), cv2.IMREAD_UNCHANGED),
        cv2.imread(str(second), cv2.IMREAD_UNCHANGED),
    ), 'the same frame rendered twice produced different output'


def test_lightfield_render_is_deterministic(scene, flight, tmp_path):
    import cv2

    paths = []
    for name in ('one.png', 'two.png'):
        path = tmp_path / name
        scene.render_lightfield(
            central_image=flight.frame_files[2],
            neighbor_images=[flight.frame_files[1], flight.frame_files[3]],
            central_idx=2, neighbor_indices=[1, 3],
            output_image=str(path),
        )
        paths.append(path)

    assert np.array_equal(
        cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED),
        cv2.imread(str(paths[1]), cv2.IMREAD_UNCHANGED),
    )
