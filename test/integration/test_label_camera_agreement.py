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
