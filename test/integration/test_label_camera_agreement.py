"""The label camera must agree with the camera the renderer actually uses.

This is the invariant that the two cancelling bugs used to satisfy only by accident, and only
for near-nadir poses: **a source pixel's label must project to the same place its image
content lands.**

The tests below deliberately use *tilted and rolled* cameras and a non-neutral correction --
the cases the old ``-(eulers - correction)`` compensation got wrong by several degrees,
because ``R(-e) == R(e).T`` holds for a single-axis rotation but not for composed Euler
angles.
"""

import json
import math
import os

import numpy as np
import pytest
from pyrr import Quaternion, Vector3

from alfspy.core.convert import pixel_to_world_coord, world_to_pixel_coord
from alfspy.core.rendering import Camera, CtxShot, RenderResultMode, Renderer, Resolution, TextureData
from alfspy.render.projection import ProjectionScene, ProjectionSettings, Label
from test.helpers.dataset import write_gltf
from test.helpers.scenes import checkerboard_rgba, flat_quad, fovy_covering, ortho_camera_above

RENDER = 128
DEM_HALF = 120.0
CAM_HEIGHT = 100.0
FOOTPRINT_HALF = 30.0
FRAME = 64


def _marker_image(size=FRAME, marker_px=(20, 44)):
    """A dark frame with one bright marker pixel block at a known location."""
    image = np.zeros((size, size, 4), dtype='f4')
    image[..., 3] = 255.0
    col, row = marker_px
    image[row - 1:row + 2, col - 1:col + 2, :3] = 255.0
    return image


def _render_shot(ctx, rotation, ortho_size, image):
    """Renders one shot orthographically from straight above and returns the RGBA image."""
    mesh = flat_quad(half=DEM_HALF)
    camera = ortho_camera_above(size=(ortho_size, ortho_size), height=CAM_HEIGHT + 50.0,
                                far=500.0)
    renderer = Renderer(Resolution(RENDER, RENDER), ctx, camera, mesh,
                        TextureData(checkerboard_rgba()))
    try:
        shot = CtxShot(ctx, image, position=Vector3([0.0, 0.0, CAM_HEIGHT]), rotation=rotation,
                       fovy=fovy_covering(FOOTPRINT_HALF, CAM_HEIGHT), aspect_ratio=1.0)
        rendered = renderer.project_shots(shot, RenderResultMode.ShotOnly)[0]
    finally:
        renderer.release()
    return rendered, camera


@pytest.mark.parametrize('eulers_deg', [
    (0.0, 0.0, 0.0),        # nadir: the case that hid the bug
    (12.0, 0.0, 0.0),       # pitch only: single-axis, the old code coped
    (12.0, 8.0, 35.0),      # pitch + roll + yaw: the old code was wrong here
    (-9.0, 15.0, 200.0),
])
def test_ray_cast_of_a_source_pixel_lands_where_the_renderer_puts_it(ctx, eulers_deg):
    """
    Render a frame containing a single bright marker, then independently ray-cast that
    marker's source pixel onto the DEM and project it into the render camera. The two must
    agree.

    This ties `pixel_to_world_coord` to the renderer's shot camera directly, with no
    reference values invented by hand.
    """
    import trimesh

    marker_px = (20, 44)
    image = _marker_image(marker_px=marker_px)
    rotation = Quaternion.from_eulers([math.radians(v) for v in eulers_deg])
    ortho_size = 2 * DEM_HALF

    rendered, render_camera = _render_shot(ctx, rotation, ortho_size, image)

    bright = (rendered[..., :3].astype(np.int32).sum(axis=-1) > 600) & (rendered[..., 3] > 0)
    assert bright.any(), 'the marker was not visible in the render'
    rows, cols = np.nonzero(bright)
    rendered_centre = (float(cols.mean()), float(rows.mean()))

    # Independently: ray-cast the marker's source pixel onto the DEM through the shot camera.
    mesh = flat_quad(half=DEM_HALF)
    tri_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.indices)
    shot_camera = Camera(fovy=fovy_covering(FOOTPRINT_HALF, CAM_HEIGHT), aspect_ratio=1.0,
                         position=Vector3([0.0, 0.0, CAM_HEIGHT]), rotation=rotation)

    world = pixel_to_world_coord([marker_px[0]], [marker_px[1]], FRAME, FRAME,
                                 tri_mesh, shot_camera, include_misses=False)
    assert len(world) == 1, 'the marker ray missed the DEM'

    xs, ys = world_to_pixel_coord(np.asarray(world), RENDER, RENDER, render_camera,
                                  ensure_int=False)
    projected = (float(np.atleast_1d(xs)[0]), float(np.atleast_1d(ys)[0]))

    assert projected[0] == pytest.approx(rendered_centre[0], abs=2.5), (
        f'x mismatch for eulers {eulers_deg}: ray-cast says {projected[0]:.1f}, '
        f'the render puts the marker at {rendered_centre[0]:.1f}'
    )
    assert projected[1] == pytest.approx(rendered_centre[1], abs=2.5), (
        f'y mismatch for eulers {eulers_deg}: ray-cast says {projected[1]:.1f}, '
        f'the render puts the marker at {rendered_centre[1]:.1f}'
    )


# ───────────────── ProjectionScene, including a real correction ─────────────


@pytest.fixture
def tilted_flight(tmp_path):
    """
    A one-frame flight with a tilted, rolled, yawed camera and a non-neutral correction.

    Both are things the previous label camera handled incorrectly.
    """
    import cv2

    root = tmp_path / 'tilted'
    root.mkdir()

    half, resolution = DEM_HALF, 8
    n = resolution + 1
    axis = np.linspace(-half, half, n, dtype='f4')
    grid_x, grid_y = np.meshgrid(axis, axis, indexing='xy')
    vertices = np.stack((grid_x, grid_y, np.zeros_like(grid_x)), axis=-1).reshape(-1, 3)

    unit = np.linspace(0.0, 1.0, n, dtype='f4')
    grid_u, grid_v = np.meshgrid(unit, unit, indexing='xy')
    uvs = np.stack((grid_u, grid_v), axis=-1).reshape(-1, 2)

    faces = []
    for row in range(resolution):
        for col in range(resolution):
            bl = row * n + col
            faces.append((bl, bl + 1, bl + n + 1))
            faces.append((bl, bl + n + 1, bl + n))
    indices = np.array(faces, dtype=np.uint32)

    dem = write_gltf(str(root / '900_matched_dem.gltf'), vertices, indices, uvs,
                     np.full((32, 32, 3), 120, dtype=np.uint8))

    # Deliberately steep and multi-axis. A mild tilt is not enough: the old
    # `-(eulers - correction)` camera is only ~0.5 px wrong for a 10-degree pose, which would
    # slip under any sane tolerance and make this test decorative.
    eulers = [22.0, 14.0, 125.0]
    poses = {'images': [{
        'imagefile': '900_00000000.png',
        'location': [0.0, 0.0, CAM_HEIGHT],
        'rotation': eulers,
        'fovy': [fovy_covering(FOOTPRINT_HALF, CAM_HEIGHT)],
    }]}
    poses_file = root / '900_matched_poses.json'
    poses_file.write_text(json.dumps(poses))

    correction_file = root / '900_correction.json'
    correction_file.write_text(json.dumps({
        'translation': {'x': 1.5, 'y': -0.75, 'z': 0.5},
        'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.03},
    }))

    frame = root / '900_00000000.png'
    marker = _marker_image()
    cv2.imwrite(str(frame), marker[..., :3][..., ::-1].astype(np.uint8))

    return {
        'dem': dem,
        'poses': str(poses_file),
        'correction': str(correction_file),
        'frame': str(frame),
        'eulers': eulers,
    }


def test_projection_scene_label_lands_on_the_rendered_marker(tilted_flight, ctx):
    """
    End to end through ``ProjectionScene``, with a tilted pose *and* a non-neutral
    correction: a label drawn around the marker pixel must land on the marker in the render.

    This is the test that discriminates the fixed label camera from the old one; with the
    previous ``-(eulers - correction)`` construction the projected box misses by tens of
    pixels.
    """
    import cv2

    settings = ProjectionSettings(
        ortho_size=(2 * DEM_HALF, 2 * DEM_HALF),
        render_resolution=Resolution(RENDER, RENDER),
        fovy=fovy_covering(FOOTPRINT_HALF, CAM_HEIGHT),
        aspect_ratio=1.0,
        add_background=False,
        input_resolution=Resolution(FRAME, FRAME),
    )

    output = os.path.join(os.path.dirname(tilted_flight['frame']), 'out.png')

    with ProjectionScene(
        dem_file=tilted_flight['dem'],
        poses_file=tilted_flight['poses'],
        correction_file=tilted_flight['correction'],
        settings=settings,
        ctx=ctx,
    ) as scene:
        marker_col, marker_row = 20, 44
        label = Label(class_id=0, bbox=(marker_col - 1.5, marker_row - 1.5, 3.0, 3.0))

        projected = scene.project_orthographic(
            tilted_flight['frame'], frame_idx=0, labels=[label], output_image=output,
        )

    assert len(projected) == 1
    box = projected[0].bbox
    label_centre = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    rendered = cv2.imread(output, cv2.IMREAD_UNCHANGED)
    assert rendered is not None

    bright = (rendered[..., :3].astype(np.int32).sum(axis=-1) > 600) & (rendered[..., 3] > 0)
    assert bright.any(), 'the marker was not visible in the rendered projection'
    rows, cols = np.nonzero(bright)
    marker_centre = (float(cols.mean()), float(rows.mean()))

    assert label_centre[0] == pytest.approx(marker_centre[0], abs=1.5), (
        f'projected label x={label_centre[0]:.1f} but the marker rendered at '
        f'x={marker_centre[0]:.1f}'
    )
    assert label_centre[1] == pytest.approx(marker_centre[1], abs=1.5), (
        f'projected label y={label_centre[1]:.1f} but the marker rendered at '
        f'y={marker_centre[1]:.1f}'
    )


def test_label_camera_reproduces_the_corrected_shot_view(tilted_flight, ctx):
    """
    The structural guarantee behind the test above: ``_camera_for_frame`` must reconstruct
    exactly the view matrix the shot program uses, ``V_shot @ C``.
    """
    settings = ProjectionSettings(
        ortho_size=(2 * DEM_HALF, 2 * DEM_HALF),
        render_resolution=Resolution(RENDER, RENDER),
        fovy=fovy_covering(FOOTPRINT_HALF, CAM_HEIGHT),
        aspect_ratio=1.0,
        input_resolution=Resolution(FRAME, FRAME),
    )

    with ProjectionScene(
        dem_file=tilted_flight['dem'],
        poses_file=tilted_flight['poses'],
        correction_file=tilted_flight['correction'],
        settings=settings,
        ctx=ctx,
    ) as scene:
        shot = scene._make_shot(tilted_flight['frame'], 0)
        try:
            expected = (np.asarray(shot.get_view(), dtype=np.float64)
                        @ np.asarray(shot.get_correction(), dtype=np.float64))
            actual = np.asarray(scene._camera_for_frame(0).get_view(), dtype=np.float64)
        finally:
            shot.release()

    np.testing.assert_allclose(actual, expected, atol=1e-4)


@pytest.mark.parametrize('module', ['alfs', 'orthografic_projection', 'orthographic_projection'])
def test_legacy_get_camera_for_frame_reproduces_the_corrected_shot_view(module):
    """
    The pipeline modules build their label camera with ``get_camera_for_frame`` rather than
    through ``ProjectionScene``. It must reproduce ``V_shot @ C`` for the shot those modules
    construct (``quaternion_from_drone_pose`` pose, ``Transform(t, from_eulers(cor))``
    correction) -- with a tilted pose and a non-neutral correction, where the old
    ``position + t`` / ``-(eulers - cor)`` construction was off by twice the yaw correction.
    """
    import importlib

    from alfspy.core.geo.transform import Transform
    from alfspy.core.rendering.shot import label_camera_for
    from alfspy.core.util.pyrrs import quaternion_from_drone_pose

    get_camera_for_frame = importlib.import_module(f'alfspy.{module}').get_camera_for_frame
    poses = {'images': [{'location': [12.0, -30.0, 140.0], 'rotation': [6.0, -3.0, 213.0],
                         'fovy': [48.9]}]}
    cor_eulers = Vector3([0.01, -0.02, 0.157], dtype='f4')
    cor_translation = Vector3([1.5, -0.5, 23.3], dtype='f4')

    shot_camera = Camera(fovy=48.9, aspect_ratio=1.0, position=Vector3(poses['images'][0]['location']),
                         rotation=quaternion_from_drone_pose(poses['images'][0]['rotation']))
    correction = Transform(cor_translation, Quaternion.from_eulers(cor_eulers), None, dtype='f4')
    expected = (np.asarray(shot_camera.get_view(), dtype=np.float64)
                @ np.asarray(correction.trans_mat.inverse * correction.rot_mat, dtype=np.float64))

    actual = np.asarray(get_camera_for_frame(poses, 0, cor_eulers, cor_translation).get_view(),
                        dtype=np.float64)
    np.testing.assert_allclose(actual, expected, atol=1e-4)
    # and the shot itself agrees with the free function
    direct = np.asarray(label_camera_for(shot_camera, correction).get_view(), dtype=np.float64)
    np.testing.assert_allclose(direct, expected, atol=1e-4)


# ───────────── the legacy pipelines, end to end against a real render ─────────────


def _legacy_pose(eulers_deg, cor_eulers, cor_translation):
    """
    The pose dictionary, correction vectors and shot the legacy pipelines build in their
    ``get_shots_for_files``: a ``quaternion_from_drone_pose`` rotation and a
    ``Transform(t, from_eulers(cor))`` correction.
    """
    from alfspy.core.geo.transform import Transform
    from alfspy.core.util.pyrrs import quaternion_from_drone_pose

    fovy = fovy_covering(FOOTPRINT_HALF, CAM_HEIGHT)
    poses = {'images': [{'location': [0.0, 0.0, CAM_HEIGHT], 'rotation': list(eulers_deg),
                         'fovy': [fovy]}]}
    cor_eulers = Vector3(cor_eulers, dtype='f4')
    cor_translation = Vector3(cor_translation, dtype='f4')
    correction = Transform(cor_translation, Quaternion.from_eulers(cor_eulers))
    rotation = quaternion_from_drone_pose([v % 360.0 for v in eulers_deg])
    return poses, cor_eulers, cor_translation, correction, rotation, fovy


def _render_legacy_shot(ctx, correction, rotation, fovy, image):
    """Renders one *corrected* shot orthographically from above; returns the image and camera."""
    mesh = flat_quad(half=DEM_HALF)
    camera = ortho_camera_above(size=(2 * DEM_HALF, 2 * DEM_HALF), height=CAM_HEIGHT + 50.0,
                                far=500.0)
    renderer = Renderer(Resolution(RENDER, RENDER), ctx, camera, mesh,
                        TextureData(checkerboard_rgba()))
    try:
        shot = CtxShot(ctx, image, Vector3([0.0, 0.0, CAM_HEIGHT]), rotation, fovy, 1, correction)
        rendered = renderer.project_shots(shot, RenderResultMode.ShotOnly)[0]
    finally:
        renderer.release()
    return rendered, camera


def _marker_centre(rendered):
    bright = (rendered[..., :3].astype(np.int32).sum(axis=-1) > 600) & (rendered[..., 3] > 0)
    assert bright.any(), 'the marker was not visible in the render'
    rows, cols = np.nonzero(bright)
    return float(cols.mean()), float(rows.mean())


def _project_marker(camera, render_camera, marker_px):
    """
    Projects the 3x3 marker block through ``camera`` the way the pipelines' ``project_label``
    does (ray-cast onto the DEM, then into the render camera) and returns the box centre.
    """
    import trimesh

    mesh = flat_quad(half=DEM_HALF)
    tri_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.indices)
    col, row = marker_px
    xs = [col - 1, col + 2, col + 2, col - 1]
    ys = [row - 1, row - 1, row + 2, row + 2]
    world = pixel_to_world_coord(xs, ys, FRAME, FRAME, tri_mesh, camera, include_misses=False)
    assert len(world) == 4, 'a marker corner ray missed the DEM'
    px, py = world_to_pixel_coord(np.asarray(world), RENDER, RENDER, render_camera,
                                  ensure_int=False)
    return float(np.mean(px)), float(np.mean(py))


LEGACY_POSES = [
    # tilted, rolled and yawed with a real correction: the case that was wrong
    ((22.0, 14.0, 125.0), (0.0, 0.0, 0.03), (1.5, -0.75, 0.5)),
    # a mild survey pose with a larger yaw correction and a height offset
    ((6.0, -3.0, 213.0), (0.01, -0.02, 0.157), (1.5, -0.5, 23.3)),
    # nadir without any correction: the case that hid the bug must keep working
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
]


@pytest.mark.parametrize('module', ['alfs', 'orthografic_projection', 'orthographic_projection'])
@pytest.mark.parametrize('eulers_deg,cor_eulers,cor_translation', LEGACY_POSES)
def test_legacy_label_lands_on_the_rendered_marker(ctx, module, eulers_deg, cor_eulers,
                                                    cor_translation):
    """
    End to end for the legacy pipelines: render a corrected shot the way their
    ``get_shots_for_files`` builds it, then project the marker pixel through the camera their
    ``get_camera_for_frame`` returns. The label must land on the marker in the render.

    Unlike the view-matrix test above this does not assume what the right matrix is; the
    renderer is the reference.
    """
    import importlib

    get_camera_for_frame = importlib.import_module(f'alfspy.{module}').get_camera_for_frame
    marker_px = (20, 44)
    poses, cor_eulers, cor_translation, correction, rotation, fovy = _legacy_pose(
        eulers_deg, cor_eulers, cor_translation)

    rendered, render_camera = _render_legacy_shot(ctx, correction, rotation, fovy,
                                                  _marker_image(marker_px=marker_px))
    marker_centre = _marker_centre(rendered)

    camera = get_camera_for_frame(poses, 0, cor_eulers, cor_translation)
    label_centre = _project_marker(camera, render_camera, marker_px)

    assert label_centre[0] == pytest.approx(marker_centre[0], abs=1.5), (
        f'{module}: label x={label_centre[0]:.1f} but the marker rendered at '
        f'x={marker_centre[0]:.1f}'
    )
    assert label_centre[1] == pytest.approx(marker_centre[1], abs=1.5), (
        f'{module}: label y={label_centre[1]:.1f} but the marker rendered at '
        f'y={marker_centre[1]:.1f}'
    )


def test_old_hand_built_label_camera_misses_the_rendered_marker(ctx):
    """
    The scene above is discriminative: the ``position + t`` / ``-(eulers - cor)`` camera the
    legacy pipelines built before 3.0.1 misses the marker by many pixels in it, so the test
    above would have failed on the old code rather than passing by luck.
    """
    marker_px = (20, 44)
    eulers_deg, cor_eulers, cor_translation = LEGACY_POSES[0]
    poses, cor_eulers, cor_translation, correction, rotation, fovy = _legacy_pose(
        eulers_deg, cor_eulers, cor_translation)

    rendered, render_camera = _render_legacy_shot(ctx, correction, rotation, fovy,
                                                  _marker_image(marker_px=marker_px))
    marker_centre = _marker_centre(rendered)

    frame = poses['images'][0]
    old_eulers = (Vector3([np.deg2rad(v % 360.0) for v in frame['rotation']]) - cor_eulers) * -1
    old_camera = Camera(fovy=fovy, aspect_ratio=1.0,
                        position=Vector3(frame['location']) + cor_translation,
                        rotation=Quaternion.from_eulers(old_eulers))
    label_centre = _project_marker(old_camera, render_camera, marker_px)

    miss = math.hypot(label_centre[0] - marker_centre[0], label_centre[1] - marker_centre[1])
    assert miss > 5.0, (
        f'the old camera only missed by {miss:.1f} px; the scene is not discriminative'
    )


def test_orthographic_project_bbox_corners_lands_on_the_rendered_marker(ctx):
    """
    ``orthographic_projection.project_bbox_corners`` end to end: the projected
    ``(left, top, width, height)`` box must enclose the marker in the render.

    It used to raise on every call: ``world_to_pixel_coord`` returns the x and y coordinates
    as two parallel sequences, and the function took its extremes as if they were ``(x, y)``
    pairs, so ``x_min, y_min = coords.min(axis=0)`` unpacked one value per corner.
    """
    import trimesh

    from alfspy.orthographic_projection import get_camera_for_frame, project_bbox_corners

    marker_px = (20, 44)
    eulers_deg, cor_eulers, cor_translation = LEGACY_POSES[0]
    poses, cor_eulers, cor_translation, correction, rotation, fovy = _legacy_pose(
        eulers_deg, cor_eulers, cor_translation)

    rendered, render_camera = _render_legacy_shot(ctx, correction, rotation, fovy,
                                                  _marker_image(marker_px=marker_px))
    marker_centre = _marker_centre(rendered)

    mesh = flat_quad(half=DEM_HALF)
    tri_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.indices)
    camera = get_camera_for_frame(poses, 0, cor_eulers, cor_translation)
    col, row = marker_px
    box = project_bbox_corners((col - 1, row - 1, 3, 3), Resolution(FRAME, FRAME), tri_mesh,
                               camera, Resolution(RENDER, RENDER), render_camera)

    assert box is not None, 'the marker box did not project'
    left, top, width, height = box
    assert width > 0 and height > 0
    # a 3 px block of a 64 px frame covering 60 m, drawn into 128 px covering 240 m: a few px
    assert width < 12 and height < 12, f'the projected box is implausibly large: {box}'
    assert left + width / 2.0 == pytest.approx(marker_centre[0], abs=1.5), (
        f'projected box centre x={left + width / 2.0:.1f} but the marker rendered at '
        f'x={marker_centre[0]:.1f}'
    )
    assert top + height / 2.0 == pytest.approx(marker_centre[1], abs=1.5), (
        f'projected box centre y={top + height / 2.0:.1f} but the marker rendered at '
        f'y={marker_centre[1]:.1f}'
    )
