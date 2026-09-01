"""Rasteriser correctness against independently computed answers.

Every expected value here is derived with plain numpy from the camera matrices, not by
recording what the rasteriser produced. That is what makes these proofs rather than
regression snapshots.
"""

import numpy as np
import pytest
import torch

from alfspy.core.torchgl import (
    iter_fragments,
    resolve_depth,
    setup_triangles,
    transform_vertices,
)
from test.helpers.scenes import (
    flat_quad,
    height_field,
    ortho_camera_above,
    single_triangle,
    two_quads_at_depths,
)

RES = 48


# ────────────────────────────── analytic reference ──────────────────────────


def screen_positions(mesh, camera, width, height):
    """
    Projects mesh vertices to screen space with plain numpy, independently of the rasteriser.

    :return: ``(V, 2)`` screen positions in pixels, y-up, and ``(V,)`` NDC depths.
    """
    view = np.asarray(camera.get_view(), dtype=np.float64)
    proj = np.asarray(camera.get_proj(), dtype=np.float64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    homogeneous = np.hstack([verts, np.ones((len(verts), 1))])

    clip = homogeneous @ view @ proj
    ndc = clip[:, :3] / clip[:, 3:4]

    screen = np.stack(((ndc[:, 0] + 1.0) * 0.5 * width,
                       (ndc[:, 1] + 1.0) * 0.5 * height), axis=-1)
    return screen, ndc[:, 2]


def analytic_coverage(mesh, camera, width, height):
    """
    Builds the expected coverage mask by point-in-triangle testing every pixel centre.

    :return: ``(H, W)`` boolean mask in the framebuffer's bottom-up row order.
    """
    screen, _ = screen_positions(mesh, camera, width, height)

    xs = np.arange(width) + 0.5
    ys = np.arange(height) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys, indexing='xy')

    mask = np.zeros((height, width), dtype=bool)
    for tri in np.asarray(mesh.indices, dtype=np.int64):
        (ax, ay), (bx, by), (cx, cy) = screen[tri]
        area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        if area == 0:
            continue
        w0 = ((cx - bx) * (grid_y - by) - (cy - by) * (grid_x - bx)) / area
        w1 = ((ax - cx) * (grid_y - cy) - (ay - cy) * (grid_x - cx)) / area
        w2 = ((bx - ax) * (grid_y - ay) - (by - ay) * (grid_x - ax)) / area
        mask |= (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    return mask


def rasterise(mesh, camera, width, height, cull_face='back', device='cpu',
              dtype=torch.float32):
    """Runs triangle setup plus a depth-resolved rasterisation for a mesh and camera."""
    verts = torch.as_tensor(np.asarray(mesh.vertices), device=device, dtype=dtype)
    indices = torch.as_tensor(np.asarray(mesh.indices, dtype=np.int64), device=device)

    view = torch.as_tensor(np.asarray(camera.get_view(), dtype=np.float64), device=device, dtype=dtype)
    proj = torch.as_tensor(np.asarray(camera.get_proj(), dtype=np.float64), device=device, dtype=dtype)

    clip = transform_vertices(verts, view @ proj)
    setup = setup_triangles(clip, indices, width, height, cull_face=cull_face)
    return setup, resolve_depth(setup, width, height)


def coverage_mask(fragments, width, height):
    mask = torch.zeros(height * width, dtype=torch.bool)
    mask[fragments.pixel_index] = True
    return mask.reshape(height, width).numpy()


# ──────────────────────────────────── tests ─────────────────────────────────


def test_full_screen_quad_covers_every_pixel():
    mesh = flat_quad(half=5.0)
    camera = ortho_camera_above(size=(10.0, 10.0))

    _, fragments = rasterise(mesh, camera, RES, RES)

    assert coverage_mask(fragments, RES, RES).all(), 'an exactly fitting quad must fill the viewport'
    assert len(fragments) == RES * RES


def test_triangle_coverage_matches_analytic_mask():
    mesh = single_triangle()
    camera = ortho_camera_above(size=(10.0, 10.0))

    _, fragments = rasterise(mesh, camera, RES, RES)

    actual = coverage_mask(fragments, RES, RES)
    expected = analytic_coverage(mesh, camera, RES, RES)

    disagreement = actual != expected
    # Pixel centres landing exactly on an edge may legitimately differ, since the rasteriser
    # applies GL's top-left fill rule while the reference uses a plain >= 0 test.
    assert disagreement.sum() <= RES, (
        f'coverage differs on {disagreement.sum()} pixels, more than the {RES}-pixel '
        f'boundary allowance'
    )
    # Interior agreement must be exact.
    assert not (disagreement & _erode(expected)).any(), 'coverage differs away from the boundary'


def _erode(mask):
    """Shrinks a boolean mask by one pixel so boundary disagreements are excluded."""
    inner = mask.copy()
    inner[1:, :] &= mask[:-1, :]
    inner[:-1, :] &= mask[1:, :]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    return inner


def test_backface_culling_discards_reversed_winding():
    camera = ortho_camera_above(size=(10.0, 10.0))

    _, front = rasterise(single_triangle(flip_winding=False), camera, RES, RES)
    _, back = rasterise(single_triangle(flip_winding=True), camera, RES, RES)

    assert len(front) > 0
    assert len(back) == 0, 'a back-facing triangle must produce no fragments when culling'


def test_culling_disabled_keeps_both_windings():
    camera = ortho_camera_above(size=(10.0, 10.0))

    _, front = rasterise(single_triangle(flip_winding=False), camera, RES, RES, cull_face=None)
    _, back = rasterise(single_triangle(flip_winding=True), camera, RES, RES, cull_face=None)

    assert len(front) == len(back) > 0


@pytest.mark.parametrize('near_first', [True, False])
def test_depth_test_picks_the_nearest_surface(near_first):
    """The nearest quad must win everywhere, whichever order the triangles are submitted in."""
    mesh = two_quads_at_depths(near_z=2.0, far_z=0.0, near_first=near_first)
    camera = ortho_camera_above(size=(10.0, 10.0), height=10.0)

    setup, fragments = rasterise(mesh, camera, RES, RES)

    assert len(fragments) == RES * RES

    # The near quad is whichever pair of triangles sits at z = 2.
    verts = np.asarray(mesh.vertices)
    tri_z = verts[np.asarray(mesh.indices, dtype=np.int64)][:, :, 2].mean(axis=1)
    near_triangles = set(np.nonzero(tri_z == 2.0)[0].tolist())

    won = set(fragments.tri_index.tolist())
    assert won <= near_triangles, 'the farther quad must not win any pixel'


def test_depth_result_is_independent_of_submission_order():
    camera = ortho_camera_above(size=(10.0, 10.0), height=10.0)

    _, first = rasterise(two_quads_at_depths(near_first=True), camera, RES, RES)
    _, second = rasterise(two_quads_at_depths(near_first=False), camera, RES, RES)

    assert torch.equal(first.pixel_index, second.pixel_index)
    np.testing.assert_allclose(first.depth.numpy(), second.depth.numpy(), atol=1e-6)


def test_shared_edges_are_covered_exactly_once():
    """
    The fill-rule guarantee. Without it, every internal DEM edge is claimed by both adjoining
    triangles, and since alpha is the ALFS overlap counter that corrupts the integral.
    """
    mesh = height_field(resolution=8, half=25.0)
    camera = ortho_camera_above(size=(50.0, 50.0), height=30.0)

    verts = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32)
    indices = torch.as_tensor(np.asarray(mesh.indices, dtype=np.int64))
    view = torch.as_tensor(np.asarray(camera.get_view(), dtype=np.float64), dtype=torch.float32)
    proj = torch.as_tensor(np.asarray(camera.get_proj(), dtype=np.float64), dtype=torch.float32)

    clip = transform_vertices(verts, view @ proj)
    setup = setup_triangles(clip, indices, RES, RES, cull_face='back')

    counts = torch.zeros(RES * RES, dtype=torch.int64)
    for fragments in iter_fragments(setup, RES, RES):
        counts.index_add_(0, fragments.pixel_index,
                          torch.ones_like(fragments.pixel_index))

    unique = set(counts.unique().tolist())
    assert unique <= {0, 1}, (
        f'pixels covered more than once by a closed mesh: counts seen = {sorted(unique)}. '
        f'This is the top-left fill rule failing.'
    )
    assert counts.sum().item() == RES * RES, 'the mesh should tile the viewport exactly'


def test_perspective_correct_barycentrics():
    """
    Interpolation weights must be divided by w. A naive screen-space linear interpolation
    passes every orthographic test and fails only here.
    """
    width = height = 32

    # Clip-space triangle built by hand so each vertex has a different w.
    clip = torch.tensor([
        [-0.8, -0.8, 0.0, 1.0],
        [3.0, -1.5, 0.0, 4.0],
        [-0.4, 0.9, 0.0, 1.0],
    ], dtype=torch.float64)
    indices = torch.tensor([[0, 1, 2]], dtype=torch.int64)

    setup = setup_triangles(clip, indices, width, height, cull_face=None)
    assert len(setup) == 1

    fragments = resolve_depth(setup, width, height)
    assert len(fragments) > 0

    screen = setup.screen[0].numpy()
    inv_w = setup.inv_w[0].numpy()

    px = (fragments.pixel_index % width).numpy() + 0.5
    py = (fragments.pixel_index // width).numpy() + 0.5

    (ax, ay), (bx, by), (cx, cy) = screen
    area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    l0 = ((cx - bx) * (py - by) - (cy - by) * (px - bx)) / area
    l1 = ((ax - cx) * (py - cy) - (ay - cy) * (px - cx)) / area
    l2 = ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) / area

    weights = np.stack((l0 * inv_w[0], l1 * inv_w[1], l2 * inv_w[2]), axis=-1)
    expected = weights / weights.sum(axis=-1, keepdims=True)

    np.testing.assert_allclose(fragments.bary.numpy(), expected, atol=1e-9)

    # And confirm the correction is not a no-op for this triangle.
    screen_linear = np.stack((l0, l1, l2), axis=-1)
    assert np.abs(expected - screen_linear).max() > 0.05, (
        'test setup is degenerate: perspective correction had no effect'
    )


def test_offscreen_geometry_produces_nothing():
    mesh = flat_quad(half=1.0)
    camera = ortho_camera_above(size=(10.0, 10.0), position=None)
    # Move the mesh far outside the frustum.
    mesh.vertices = mesh.vertices + np.array([500.0, 0.0, 0.0], dtype='f4')

    setup, fragments = rasterise(mesh, camera, RES, RES)

    assert len(setup) == 0
    assert len(fragments) == 0


def test_geometry_behind_the_camera_is_rejected():
    """Vertices with w <= 0 cannot be divided through; they must be dropped, not wrapped."""
    from test.helpers.scenes import perspective_camera_above

    mesh = flat_quad(half=5.0, z=50.0)  # above a camera that sits at z = 10 looking down
    camera = perspective_camera_above(fovy=60.0, height=10.0, far=1000.0)

    setup, fragments = rasterise(mesh, camera, RES, RES)

    assert len(fragments) == 0


def test_degenerate_triangle_is_dropped():
    clip = torch.tensor([
        [-0.5, -0.5, 0.0, 1.0],
        [0.5, 0.5, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],  # collinear -> zero area
    ], dtype=torch.float32)
    indices = torch.tensor([[0, 1, 2]], dtype=torch.int64)

    setup = setup_triangles(clip, indices, RES, RES, cull_face=None)
    assert len(setup) == 0


def test_sample_budget_does_not_change_the_result():
    """Chunking is a memory-management detail and must be invisible in the output."""
    mesh = height_field(resolution=6, half=25.0)
    camera = ortho_camera_above(size=(50.0, 50.0), height=30.0)

    verts = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32)
    indices = torch.as_tensor(np.asarray(mesh.indices, dtype=np.int64))
    view = torch.as_tensor(np.asarray(camera.get_view(), dtype=np.float64), dtype=torch.float32)
    proj = torch.as_tensor(np.asarray(camera.get_proj(), dtype=np.float64), dtype=torch.float32)
    clip = transform_vertices(verts, view @ proj)
    setup = setup_triangles(clip, indices, RES, RES, cull_face='back')

    generous = resolve_depth(setup, RES, RES, sample_budget=1 << 20)
    stingy = resolve_depth(setup, RES, RES, sample_budget=64)

    assert torch.equal(generous.pixel_index, stingy.pixel_index)
    assert torch.equal(generous.tri_index, stingy.tri_index)
    np.testing.assert_allclose(generous.bary.numpy(), stingy.bary.numpy(), atol=1e-6)


# ─────────────────────────────── bucket batching ────────────────────────────


@pytest.mark.parametrize('boxes', [
    # (x0, y0, x1, y1), inclusive. Shapes chosen to exercise the merge policy:
    # many identical small boxes, a few odd ones, and one very large box.
    [(0, 0, 7, 7)] * 200,
    [(0, 0, 7, 8)] * 200 + [(0, 0, 0, 0)] * 50 + [(0, 0, 63, 2)] * 3,
    [(i, i, i + (i % 13), i + (i % 7)) for i in range(500)],
    [(0, 0, 1023, 1023)],
])
@pytest.mark.parametrize('budget', [64, 4096, 1 << 21])
def test_bucket_batches_partition_every_triangle(boxes, budget):
    """
    Each triangle must appear in exactly one batch, and its box must fit the batch's grid.

    The batching loop is greedy and has several cut conditions; a triangle silently dropped
    here would vanish from the render rather than raise.
    """
    from alfspy.core.torchgl.raster import _bucket_batches

    bbox = torch.tensor(boxes, dtype=torch.int64)
    batches = _bucket_batches(bbox, budget)

    seen = torch.cat([b[0] for b in batches]) if batches else torch.zeros(0, dtype=torch.int64)
    assert sorted(seen.tolist()) == list(range(len(boxes))), 'batches do not partition the triangles'

    for indices, grid_w, grid_h in batches:
        assert indices.numel() > 0
        widths = bbox[indices, 2] - bbox[indices, 0] + 1
        heights = bbox[indices, 3] - bbox[indices, 1] + 1
        assert int(widths.max()) <= grid_w, 'a box is wider than its batch grid'
        assert int(heights.max()) <= grid_h, 'a box is taller than its batch grid'
        # A batch may exceed the budget only when it holds a single oversized box.
        if indices.numel() > 1:
            assert indices.numel() * grid_w * grid_h <= budget


def test_bucket_batches_are_tight_for_a_uniform_dem():
    """
    The regular grid an orthographic DEM produces must cost no more candidate samples than
    its bounding boxes need. This is the case the square power-of-two bucketing inflated by
    2.6x, which was the rasteriser's single largest source of wasted work.
    """
    from alfspy.core.torchgl.raster import _bucket_batches

    mesh = height_field(resolution=32, half=25.0)
    camera = ortho_camera_above(size=(50.0, 50.0), height=30.0)
    setup, _ = rasterise(mesh, camera, 256, 256)

    bbox = setup.bbox
    exact = int(((bbox[:, 2] - bbox[:, 0] + 1) * (bbox[:, 3] - bbox[:, 1] + 1)).sum())
    candidates = sum(i.numel() * w * h for i, w, h in _bucket_batches(bbox, 1 << 21))

    assert candidates <= 1.1 * exact, f'{candidates} candidates for {exact} bounding-box samples'
