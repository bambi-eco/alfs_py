"""Depth-buffered triangle rasteriser in pure PyTorch.

This is the component that replaces the OpenGL fixed-function pipeline. It implements the
subset the ALFS renderer needs and nothing more: triangle setup, back-face culling,
near/far rejection, coverage, depth resolution and perspective-correct attribute
interpolation.

Design notes
------------

*Matrices are in pyrr's row-vector convention* -- clip coordinates are ``v_row @ MVP``.
See ``docs/pyrr_conventions.md``.

*Coverage uses bucketed binning.* Triangles are grouped by the shape of their screen
bounding box, so each group expands to a dense ``w x h`` candidate grid in one broadcast
instead of a ragged gather. Groups are processed in chunks bounded by a sample budget, which
keeps peak memory independent of both mesh size and the triangle size distribution. Grouping
by the *exact* box shape rather than a rounded-up square matters more than it looks: see
:func:`_bucket_batches`.

*Barycentric weights are computed after compaction, not over the candidate grid*, and
:func:`resolve_depth` skips the perspective-correct ones entirely because it recomputes the
surviving fragments' weights once the depth test has picked them.

*Depth resolution is a single scatter-min over packed int64 keys*
(``key = (quantised_depth << 32) | triangle_index``). One pass resolves both the nearest
fragment and the tie-break (lowest triangle index), with no sorting and no atomics race, so
results are bit-identical on CPU and CUDA and across runs.
"""

from dataclasses import dataclass
from typing import Iterator, Optional

import torch

from alfspy.core.torchgl.compat import scatter_min_

__all__ = [
    'Fragments',
    'DEFAULT_SAMPLE_BUDGET',
    'transform_vertices',
    'TriangleSetup',
    'setup_triangles',
    'iter_fragments',
    'resolve_depth',
]

DEFAULT_SAMPLE_BUDGET: int = 1 << 21
"""Maximum number of candidate samples materialised at once. Bounds peak memory."""

_DEPTH_QUANT_BITS: int = 31
_DEPTH_QUANT_MAX: int = (1 << _DEPTH_QUANT_BITS) - 1
_TRI_ID_BITS: int = 32
_TRI_ID_MASK: int = (1 << _TRI_ID_BITS) - 1
_INT64_MAX: int = torch.iinfo(torch.int64).max

_W_EPSILON: float = 1e-7

_SHAPE_KEY_MASK: int = (1 << 32) - 1
"""Low half of the ``(height, width)`` key :func:`_bucket_batches` sorts bounding boxes by."""


@dataclass
class Fragments:
    """
    A batch of rasterised fragments.

    :cvar pixel_index: ``(N,)`` flat pixel indices into a row-major ``H x W`` image
        (row ``0`` is the bottom, as in GL).
    :cvar tri_index: ``(N,)`` index of the triangle covering each fragment.
    :cvar bary: ``(N, 3)`` perspective-correct barycentric coordinates, ready for attribute
        interpolation. Empty when the chunk was produced with ``need_bary=False``.
    :cvar bary_screen: ``(N, 3)`` screen-space (non perspective-corrected) barycentrics, used
        for depth interpolation. Empty when the chunk was produced with ``need_bary=False``.
    :cvar depth: ``(N,)`` interpolated NDC depth in ``[-1, 1]``.
    """
    pixel_index: torch.Tensor
    tri_index: torch.Tensor
    bary: torch.Tensor
    bary_screen: torch.Tensor
    depth: torch.Tensor

    def __len__(self) -> int:
        return int(self.pixel_index.numel())

    @staticmethod
    def concat(chunks: 'list[Fragments]') -> 'Fragments':
        """
        Joins fragment chunks into one batch.

        :param chunks: The chunks to join. Must not be empty.
        :return: A single :class:`Fragments` holding every input fragment.
        """
        if len(chunks) == 1:
            return chunks[0]
        return Fragments(
            pixel_index=torch.cat([c.pixel_index for c in chunks]),
            tri_index=torch.cat([c.tri_index for c in chunks]),
            bary=torch.cat([c.bary for c in chunks]),
            bary_screen=torch.cat([c.bary_screen for c in chunks]),
            depth=torch.cat([c.depth for c in chunks]),
        )

    def interpolate(self, vertex_attributes: torch.Tensor, tri_indices: torch.Tensor) -> torch.Tensor:
        """
        Perspective-correctly interpolates a per-vertex attribute at these fragments.

        :param vertex_attributes: ``(V, K)`` per-vertex attribute values.
        :param tri_indices: ``(T, 3)`` triangle vertex indices.
        :return: ``(N, K)`` interpolated values.
        """
        corners = tri_indices[self.tri_index]                       # (N, 3)
        gathered = vertex_attributes[corners]                       # (N, 3, K)
        return (gathered * self.bary.unsqueeze(-1)).sum(dim=1)


def transform_vertices(vertices: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """
    Transforms positions into clip space using the row-vector convention.

    :param vertices: ``(V, 3)`` or ``(V, 4)`` positions.
    :param matrix: A ``(4, 4)`` pyrr-convention matrix (or product of them).
    :return: ``(V, 4)`` clip-space positions, i.e. ``v_row @ matrix``.
    """
    if vertices.ndim != 2 or vertices.shape[1] not in (3, 4):
        raise ValueError(f'Expected (V, 3) or (V, 4) vertices but got {tuple(vertices.shape)}')

    if vertices.shape[1] == 3:
        ones = torch.ones((vertices.shape[0], 1), device=vertices.device, dtype=vertices.dtype)
        vertices = torch.cat((vertices, ones), dim=1)

    return vertices @ matrix.to(dtype=vertices.dtype)


@dataclass
class TriangleSetup:
    """
    Per-triangle screen-space data produced by :func:`setup_triangles`.

    Only triangles that survived culling are present; :cvar:`tri_index` maps back to the
    original triangle numbering so fragment output stays meaningful to the caller.

    :cvar tri_index: ``(T',)`` original indices of the surviving triangles.
    :cvar screen: ``(T', 3, 2)`` screen-space vertex positions in pixels.
    :cvar depth: ``(T', 3)`` per-vertex NDC depth.
    :cvar inv_w: ``(T', 3)`` per-vertex ``1 / w_clip``, for perspective correction.
    :cvar bbox: ``(T', 4)`` integer pixel bounding boxes as ``(x0, y0, x1, y1)``, inclusive.
    :cvar area: ``(T',)`` signed screen-space area (twice the triangle area).
    :cvar edge_origin: ``(T', 3, 2)`` canonical first endpoint of each edge.
    :cvar edge_delta: ``(T', 3, 2)`` canonical edge vector.
    :cvar edge_sign: ``(T', 3)`` factor restoring the triangle's own edge orientation.
    """
    tri_index: torch.Tensor
    screen: torch.Tensor
    depth: torch.Tensor
    inv_w: torch.Tensor
    bbox: torch.Tensor
    area: torch.Tensor
    edge_origin: torch.Tensor
    edge_delta: torch.Tensor
    edge_sign: torch.Tensor

    def __len__(self) -> int:
        return int(self.tri_index.numel())


def setup_triangles(clip: torch.Tensor, indices: torch.Tensor, width: int, height: int,
                    cull_face: Optional[str] = 'back',
                    front_face: str = 'ccw') -> TriangleSetup:
    """
    Performs triangle setup: perspective divide, viewport mapping, culling and binning boxes.

    :param clip: ``(V, 4)`` clip-space vertex positions.
    :param indices: ``(T, 3)`` triangle vertex indices.
    :param width: Viewport width in pixels.
    :param height: Viewport height in pixels.
    :param cull_face: ``'back'``, ``'front'``, ``'front_and_back'`` or ``None`` to disable
        culling (defaults to ``'back'``, matching ``make_torch_context``).
    :param front_face: Winding that counts as front-facing, ``'ccw'`` (GL default) or ``'cw'``.
    :return: A :class:`TriangleSetup` holding only the surviving triangles.
    """
    device = clip.device
    dtype = clip.dtype

    tri = indices.to(device=device, dtype=torch.int64)
    tri_clip = clip[tri]                                              # (T, 3, 4)
    w = tri_clip[..., 3]

    # Triangles crossing the near plane cannot be handled by a divide alone. GL clips them;
    # we reject them whole. For the top-down DEM cameras used here nothing ever straddles it.
    keep = (w > _W_EPSILON).all(dim=1)

    if not bool(keep.any()):
        return _empty_setup(device, dtype)

    tri_index = torch.nonzero(keep, as_tuple=False).squeeze(1)
    tri_clip = tri_clip[keep]
    w = w[keep]

    inv_w = 1.0 / w
    ndc = tri_clip[..., :3] * inv_w.unsqueeze(-1)                     # (T', 3, 3)

    screen_x = (ndc[..., 0] + 1.0) * (0.5 * width)
    screen_y = (ndc[..., 1] + 1.0) * (0.5 * height)
    screen = torch.stack((screen_x, screen_y), dim=-1)                # (T', 3, 2)
    depth = ndc[..., 2]

    # Signed area, twice the triangle area. Positive == counter-clockwise in a y-up viewport.
    edge1 = screen[:, 1] - screen[:, 0]
    edge2 = screen[:, 2] - screen[:, 0]
    area = edge1[:, 0] * edge2[:, 1] - edge2[:, 0] * edge1[:, 1]

    alive = area != 0.0

    if cull_face:
        mode = cull_face.lower()
        front_is_positive = front_face.lower() == 'ccw'
        front = area > 0 if front_is_positive else area < 0
        if mode == 'back':
            alive = alive & front
        elif mode == 'front':
            alive = alive & ~front
        elif mode in ('front_and_back', 'front_and_back_face'):
            alive = torch.zeros_like(alive)
        else:
            raise ValueError(f'Unknown cull_face mode "{cull_face}"')

    # Reject triangles wholly outside the viewport or wholly beyond the depth range.
    min_xy = screen.amin(dim=1)
    max_xy = screen.amax(dim=1)
    alive = alive & (max_xy[:, 0] >= 0.0) & (min_xy[:, 0] <= width)
    alive = alive & (max_xy[:, 1] >= 0.0) & (min_xy[:, 1] <= height)
    alive = alive & (depth.amax(dim=1) >= -1.0) & (depth.amin(dim=1) <= 1.0)

    if not bool(alive.any()):
        return _empty_setup(device, dtype)

    tri_index = tri_index[alive]
    screen = screen[alive]
    depth = depth[alive]
    inv_w = inv_w[alive]
    area = area[alive]
    min_xy = min_xy[alive]
    max_xy = max_xy[alive]

    # Pixel i has its centre at i + 0.5, so it is covered when min <= i + 0.5 <= max.
    x0 = torch.ceil(min_xy[:, 0] - 0.5).clamp_(0, width - 1).to(torch.int64)
    y0 = torch.ceil(min_xy[:, 1] - 0.5).clamp_(0, height - 1).to(torch.int64)
    x1 = torch.floor(max_xy[:, 0] - 0.5).clamp_(0, width - 1).to(torch.int64)
    y1 = torch.floor(max_xy[:, 1] - 0.5).clamp_(0, height - 1).to(torch.int64)

    non_empty = (x1 >= x0) & (y1 >= y0)
    if not bool(non_empty.any()):
        return _empty_setup(device, dtype)

    bbox = torch.stack((x0, y0, x1, y1), dim=1)[non_empty]
    screen = screen[non_empty]
    edge_origin, edge_delta, edge_sign = _canonical_edges(screen)

    return TriangleSetup(
        tri_index=tri_index[non_empty],
        screen=screen,
        depth=depth[non_empty],
        inv_w=inv_w[non_empty],
        bbox=bbox,
        area=area[non_empty],
        edge_origin=edge_origin,
        edge_delta=edge_delta,
        edge_sign=edge_sign,
    )


def _canonical_edges(screen: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rewrites each triangle edge in a canonical endpoint order.

    Two triangles sharing an edge traverse it in opposite directions, so each evaluates the
    edge function from a *different* anchor vertex. Those two expressions are exact negatives
    in real arithmetic but not in floating point: when a pixel centre lies on the edge, both
    can round to a small positive value and both triangles claim the pixel. On a DEM that
    turns every internal edge into a double-counted seam, and since alpha is the ALFS overlap
    counter the integral then normalises those pixels by the wrong number of shots.

    Anchoring every edge at the lexicographically smaller endpoint makes both triangles
    compute the *identical* float, so negating it to restore each triangle's orientation is
    exact and the top-left fill rule can decide unambiguously. This is the floating-point
    equivalent of the fixed-point vertex snapping a hardware rasteriser uses.

    :param screen: ``(T, 3, 2)`` screen-space triangle vertices.
    :return: ``(origin, delta, sign)`` of shapes ``(T, 3, 2)``, ``(T, 3, 2)`` and ``(T, 3)``,
        where edge ``i`` is the one opposite vertex ``i``.
    """
    # Edge i runs from vertex (i+1) % 3 to vertex (i+2) % 3.
    starts = screen[:, [1, 2, 0], :]
    ends = screen[:, [2, 0, 1], :]

    # Lexicographic comparison on (x, y): swap when the start is the larger endpoint.
    swap = (starts[..., 0] > ends[..., 0]) | (
        (starts[..., 0] == ends[..., 0]) & (starts[..., 1] > ends[..., 1])
    )
    swap_expanded = swap.unsqueeze(-1)

    origin = torch.where(swap_expanded, ends, starts)
    far = torch.where(swap_expanded, starts, ends)

    ones = torch.ones_like(starts[..., 0])
    sign = torch.where(swap, -ones, ones)
    return origin, far - origin, sign


def _empty_setup(device: torch.device, dtype: torch.dtype) -> TriangleSetup:
    return TriangleSetup(
        tri_index=torch.zeros(0, device=device, dtype=torch.int64),
        screen=torch.zeros((0, 3, 2), device=device, dtype=dtype),
        depth=torch.zeros((0, 3), device=device, dtype=dtype),
        inv_w=torch.zeros((0, 3), device=device, dtype=dtype),
        bbox=torch.zeros((0, 4), device=device, dtype=torch.int64),
        area=torch.zeros(0, device=device, dtype=dtype),
        edge_origin=torch.zeros((0, 3, 2), device=device, dtype=dtype),
        edge_delta=torch.zeros((0, 3, 2), device=device, dtype=dtype),
        edge_sign=torch.zeros((0, 3), device=device, dtype=dtype),
    )


_MAX_BUCKET_WASTE: float = 1.5
"""How much larger than a triangle's own bounding box its candidate grid may get when
triangles of differing sizes are merged into one batch."""

_MIN_BATCH_SAMPLES: int = 1 << 18
"""Candidate samples a batch must reach before :data:`_MAX_BUCKET_WASTE` may split it.

Every batch costs a fixed number of tensor dispatches regardless of how few triangles it
holds, so splitting to save arithmetic is only worth it once there is enough arithmetic to
save. Without this floor a tilted perspective view -- which produces hundreds of distinct
box shapes -- fragments into dozens of batches of a couple of dozen triangles each, and the
dispatch overhead costs more than the wasted samples ever did."""


def _bucket_batches(bbox: torch.Tensor, sample_budget: int) -> list[tuple[torch.Tensor, int, int]]:
    """
    Groups triangles into batches that can share a single dense candidate grid.

    Coverage is evaluated by expanding each triangle to a rectangular grid of candidate
    samples, so every sample the grid holds beyond the triangle's own bounding box is wasted
    work. An earlier version rounded the *longer* box side up to a power of two and used a
    square grid of that side, which cost 1.9x to 2.7x more candidates than the boxes actually
    needed: a DEM triangle spanning 9 pixels was rasterised on a 16x16 grid.

    Grouping by the exact box shape removes that overhead entirely. Real meshes produce few
    distinct shapes -- an orthographic DEM produces about four -- but a tilted perspective
    view can produce a few hundred, and one dispatch per shape would trade the saved
    arithmetic for Python overhead. So shapes are sorted by ``(height, width)``, which puts
    similar ones next to each other, and adjacent shapes are merged while the merged grid
    stays within :data:`_MAX_BUCKET_WASTE` of the smallest box in the batch -- a limit that
    only starts applying once the batch holds :data:`_MIN_BATCH_SAMPLES` candidates, so that
    small shapes never split into batches too thin to be worth dispatching.

    Every triangle lands in exactly one batch, and its box always fits inside that batch's
    grid; :func:`_rasterise_batch` masks off the samples that overhang its own box.

    :param bbox: ``(T, 4)`` inclusive pixel bounding boxes.
    :param sample_budget: Maximum candidate samples a single batch may materialise.
    :return: A list of ``(triangle_indices, grid_width, grid_height)`` batches.
    """
    widths = bbox[:, 2] - bbox[:, 0] + 1
    heights = bbox[:, 3] - bbox[:, 1] + 1

    # Sort by (height, width) so equal shapes are contiguous and neighbours are similar.
    # `stable=True` keeps the batching -- and therefore the fragment order -- reproducible.
    key = (heights << 32) | widths
    order = torch.argsort(key, stable=True)

    unique_keys, counts = torch.unique_consecutive(key[order], return_counts=True)

    # One host round-trip; the loop below runs over distinct shapes, not triangles.
    shape_w = (unique_keys & _SHAPE_KEY_MASK).tolist()
    shape_h = (unique_keys >> 32).tolist()
    shape_n = counts.tolist()

    budget = max(1, int(sample_budget))
    batches: list[tuple[torch.Tensor, int, int]] = []

    start = 0          # offset into `order` of the current batch
    taken = 0          # triangles already in the current batch
    grid_w = grid_h = 0
    smallest = 0       # smallest exact box area in the current batch

    def flush(end: int) -> None:
        if end > start:
            batches.append((order[start:end], grid_w, grid_h))

    cursor = 0
    for width, height, count in zip(shape_w, shape_h, shape_n):
        area = width * height
        remaining = count
        while remaining:
            new_w = max(grid_w, width)
            new_h = max(grid_h, height)
            new_smallest = min(smallest, area) if taken else area

            # Merging this shape in must not dilute the batch beyond the waste limit -- but
            # only once the batch is big enough that a separate dispatch would pay for itself.
            if (taken * new_w * new_h >= _MIN_BATCH_SAMPLES
                    and new_w * new_h > _MAX_BUCKET_WASTE * new_smallest):
                flush(cursor)
                start, taken, grid_w, grid_h, smallest = cursor, 0, 0, 0, 0
                continue

            room = budget // (new_w * new_h) - taken
            if room <= 0:
                if taken:
                    flush(cursor)
                    start, taken, grid_w, grid_h, smallest = cursor, 0, 0, 0, 0
                    continue
                # A single box larger than the whole budget still has to be rasterised.
                room = 1

            step = min(remaining, room)
            grid_w, grid_h, smallest = new_w, new_h, new_smallest
            taken += step
            cursor += step
            remaining -= step

    flush(cursor)
    return batches


def iter_fragments(setup: TriangleSetup, width: int, height: int,
                   sample_budget: int = DEFAULT_SAMPLE_BUDGET,
                   need_bary: bool = True) -> Iterator[Fragments]:
    """
    Yields chunks of covered fragments for every triangle in ``setup``.

    All fragments are produced -- no depth resolution is applied. Callers that need a depth
    test feed the chunks to :func:`resolve_depth`; the additive-blending path consumes them
    directly, which is exactly what ``disable(DEPTH_TEST)`` did in the GL version.

    :param setup: The triangle setup to rasterise.
    :param width: Viewport width in pixels.
    :param height: Viewport height in pixels.
    :param sample_budget: Maximum candidate samples materialised per chunk.
    :param need_bary: Whether perspective-correct barycentrics are wanted. :func:`resolve_depth`
        passes ``False``: it keeps only one fragment per pixel and recomputes their weights
        afterwards, so computing them for every covered sample first is pure waste. The
        ``bary`` and ``bary_screen`` fields of the yielded chunks are then empty.
    :return: An iterator over :class:`Fragments` chunks.
    """
    if len(setup) == 0:
        return

    device = setup.screen.device
    dtype = setup.screen.dtype

    for batch, grid_w, grid_h in _bucket_batches(setup.bbox, sample_budget):
        off_y, off_x = torch.meshgrid(
            torch.arange(grid_h, device=device, dtype=torch.int64),
            torch.arange(grid_w, device=device, dtype=torch.int64),
            indexing='ij')
        fragments = _rasterise_batch(setup, batch, off_x.reshape(-1), off_y.reshape(-1),
                                     width, height, device, dtype, need_bary)
        if fragments is not None:
            yield fragments


def _covers(edge_value: torch.Tensor, edge_dx: torch.Tensor, edge_dy: torch.Tensor,
            area: torch.Tensor) -> torch.Tensor:
    """
    Applies GL's top-left fill rule to one edge.

    Without it, a pixel centre lying exactly on an edge shared by two triangles is claimed by
    *both*, so the DEM's internal edges each get counted twice. That is harmless for an
    opaque depth-tested draw but corrupts :func:`iter_fragments`-based additive integration,
    where the alpha channel is an overlap counter: shared edges would report one extra
    contributing shot and normalise to the wrong colour.

    The rule: a fragment exactly on an edge is inside only if that edge is a *top* or *left*
    edge. Exactly one of the two triangles sharing an edge sees it that way, because the edge
    is traversed in opposite directions by each.

    Screen space here is y-up (row 0 is the bottom) and front faces are counter-clockwise, so
    a left edge runs downward (``dy < 0``) and a top edge is horizontal running leftward
    (``dy == 0 and dx < 0``). ``area`` supplies the winding: for a clockwise triangle every
    edge direction is reversed, which is what multiplying the deltas by its sign expresses.

    :param edge_value: The edge function evaluated at the sample points.
    :param edge_dx: The edge's x delta.
    :param edge_dy: The edge's y delta.
    :param area: The triangle's signed area, broadcastable against the samples.
    :return: A boolean tensor selecting the samples this edge admits.
    """
    orientation = torch.sign(area)
    oriented_value = edge_value * orientation
    oriented_dx = edge_dx * orientation
    oriented_dy = edge_dy * orientation

    top_left = (oriented_dy < 0) | ((oriented_dy == 0) & (oriented_dx < 0))
    return (oriented_value > 0) | ((oriented_value == 0) & top_left)


def _rasterise_batch(setup: TriangleSetup, batch: torch.Tensor,
                     off_x: torch.Tensor, off_y: torch.Tensor,
                     width: int, height: int,
                     device: torch.device, dtype: torch.dtype,
                     need_bary: bool = True) -> Optional[Fragments]:
    """Rasterises one bucket chunk. Returns ``None`` when nothing is covered."""
    bbox = setup.bbox[batch]                                            # (K, 4)
    depth = setup.depth[batch]                                          # (K, 3)

    px = bbox[:, 0:1] + off_x.unsqueeze(0)                              # (K, S)
    py = bbox[:, 1:2] + off_y.unsqueeze(0)

    in_bbox = (px <= bbox[:, 2:3]) & (py <= bbox[:, 3:4])

    # Pixel centres.
    cx = px.to(dtype) + 0.5
    cy = py.to(dtype) + 0.5

    origin = setup.edge_origin[batch]                                   # (K, 3, 2)
    delta = setup.edge_delta[batch]                                     # (K, 3, 2)
    edge_sign = setup.edge_sign[batch]                                  # (K, 3)
    area = setup.area[batch].unsqueeze(-1)                              # (K, 1)

    # Edge functions, evaluated from the canonical endpoint so that two triangles sharing an
    # edge produce bit-identical magnitudes. e_i is the edge opposite vertex i and is
    # proportional to that vertex's barycentric weight.
    values = []
    for edge in range(3):
        ax = origin[:, edge, 0:1]
        ay = origin[:, edge, 1:2]
        dx = delta[:, edge, 0:1]
        dy = delta[:, edge, 1:2]
        values.append(edge_sign[:, edge:edge + 1] * (dx * (cy - ay) - dy * (cx - ax)))

    e0, e1, e2 = values
    total = e0 + e1 + e2

    inside = in_bbox & (total != 0)
    for edge, value in enumerate(values):
        oriented = edge_sign[:, edge:edge + 1]
        inside = inside & _covers(value,
                                  delta[:, edge, 0:1] * oriented,
                                  delta[:, edge, 1:2] * oriented,
                                  area)

    hit = torch.nonzero(inside, as_tuple=False)
    if hit.numel() == 0:
        return None

    rows, cols = hit[:, 0], hit[:, 1]

    # Barycentrics are wanted only where the triangle actually covers the sample, which is
    # well under half the candidate grid. Computing them after compaction rather than over
    # the dense grid is the same arithmetic on a fraction of the elements.
    safe_total = total[rows, cols]
    safe_total = torch.where(safe_total == 0, torch.ones_like(safe_total), safe_total)
    l0 = e0[rows, cols] / safe_total
    l1 = e1[rows, cols] / safe_total
    l2 = e2[rows, cols] / safe_total

    frag_depth = l0 * depth[rows, 0] + l1 * depth[rows, 1] + l2 * depth[rows, 2]

    # Near/far rejection. Whole triangles outside the range are already gone; this catches
    # the fragments of a triangle that only partly straddles it.
    in_range = (frag_depth >= -1.0) & (frag_depth <= 1.0)
    if not bool(in_range.all()):
        keep = torch.nonzero(in_range, as_tuple=False).squeeze(1)
        if keep.numel() == 0:
            return None
        rows = rows.index_select(0, keep)
        cols = cols.index_select(0, keep)
        l0 = l0.index_select(0, keep)
        l1 = l1.index_select(0, keep)
        l2 = l2.index_select(0, keep)
        frag_depth = frag_depth.index_select(0, keep)

    pixel_index = py[rows, cols] * width + px[rows, cols]

    if need_bary:
        bary_screen = torch.stack((l0, l1, l2), dim=1)
        # Perspective correction: weight by 1/w and renormalise.
        weights = bary_screen * setup.inv_w[batch][rows]
        weight_sum = weights.sum(dim=1, keepdim=True)
        weight_sum = torch.where(weight_sum == 0, torch.ones_like(weight_sum), weight_sum)
        bary = weights / weight_sum
    else:
        bary = bary_screen = torch.zeros((0, 3), device=device, dtype=dtype)

    return Fragments(
        pixel_index=pixel_index,
        tri_index=setup.tri_index[batch][rows],
        bary=bary,
        bary_screen=bary_screen,
        depth=frag_depth,
    )


def _pack_key(depth: torch.Tensor, tri_index: torch.Tensor) -> torch.Tensor:
    """
    Packs NDC depth and a triangle index into one sortable int64.

    :param depth: ``(N,)`` NDC depth in ``[-1, 1]``.
    :param tri_index: ``(N,)`` triangle indices, which must fit in 32 bits.
    :return: ``(N,)`` int64 keys ordered first by depth, then by triangle index.
    """
    normalised = ((depth.to(torch.float64) + 1.0) * 0.5).clamp_(0.0, 1.0)
    quantised = (normalised * _DEPTH_QUANT_MAX).round().to(torch.int64)
    return (quantised << _TRI_ID_BITS) | (tri_index & _TRI_ID_MASK)


def resolve_depth(setup: TriangleSetup, width: int, height: int,
                  sample_budget: int = DEFAULT_SAMPLE_BUDGET) -> Fragments:
    """
    Rasterises with a depth test, returning one fragment per covered pixel.

    :param setup: The triangle setup to rasterise.
    :param width: Viewport width in pixels.
    :param height: Viewport height in pixels.
    :param sample_budget: Maximum candidate samples materialised per chunk.
    :return: The winning :class:`Fragments`, one entry per covered pixel, ordered by pixel
        index.
    """
    device = setup.screen.device
    dtype = setup.screen.dtype

    if len(setup) == 0:
        empty = torch.zeros(0, device=device, dtype=torch.int64)
        return Fragments(empty, empty.clone(),
                         torch.zeros((0, 3), device=device, dtype=dtype),
                         torch.zeros((0, 3), device=device, dtype=dtype),
                         torch.zeros(0, device=device, dtype=dtype))

    if int(setup.tri_index.max()) > _TRI_ID_MASK:
        raise ValueError(
            f'Mesh has more than {_TRI_ID_MASK} triangles, which exceeds the depth-key '
            f'packing budget of the rasteriser.'
        )

    buffer = torch.full((height * width,), _INT64_MAX, device=device, dtype=torch.int64)

    # Only coverage, depth and triangle identity survive the depth test; the winners' weights
    # are recomputed by ``_shade_winners`` below, so asking for them here would be wasted.
    for fragments in iter_fragments(setup, width, height, sample_budget, need_bary=False):
        keys = _pack_key(fragments.depth, fragments.tri_index)
        scatter_min_(buffer, fragments.pixel_index, keys)

    covered = buffer != _INT64_MAX
    pixel_index = torch.nonzero(covered, as_tuple=False).squeeze(1)

    if pixel_index.numel() == 0:
        empty = torch.zeros(0, device=device, dtype=torch.int64)
        return Fragments(empty, empty.clone(),
                         torch.zeros((0, 3), device=device, dtype=dtype),
                         torch.zeros((0, 3), device=device, dtype=dtype),
                         torch.zeros(0, device=device, dtype=dtype))

    winners = buffer[pixel_index]
    tri_index = winners & _TRI_ID_MASK

    # Map original triangle indices back to rows of the setup arrays.
    lookup = torch.full((int(setup.tri_index.max()) + 1,), -1, device=device, dtype=torch.int64)
    lookup[setup.tri_index] = torch.arange(len(setup), device=device, dtype=torch.int64)
    rows = lookup[tri_index]

    return _shade_winners(setup, rows, tri_index, pixel_index, width, dtype)


def _shade_winners(setup: TriangleSetup, rows: torch.Tensor, tri_index: torch.Tensor,
                   pixel_index: torch.Tensor, width: int, dtype: torch.dtype) -> Fragments:
    """Recomputes barycentrics for the depth-test winners only."""
    screen = setup.screen[rows]
    depth = setup.depth[rows]
    inv_w = setup.inv_w[rows]

    cx = (pixel_index % width).to(dtype) + 0.5
    cy = torch.div(pixel_index, width, rounding_mode='floor').to(dtype) + 0.5

    x0, y0 = screen[:, 0, 0], screen[:, 0, 1]
    x1, y1 = screen[:, 1, 0], screen[:, 1, 1]
    x2, y2 = screen[:, 2, 0], screen[:, 2, 1]

    e0 = (x2 - x1) * (cy - y1) - (y2 - y1) * (cx - x1)
    e1 = (x0 - x2) * (cy - y2) - (y0 - y2) * (cx - x2)
    e2 = (x1 - x0) * (cy - y0) - (y1 - y0) * (cx - x0)

    total = e0 + e1 + e2
    total = torch.where(total == 0, torch.ones_like(total), total)
    bary_screen = torch.stack((e0 / total, e1 / total, e2 / total), dim=1)

    weights = bary_screen * inv_w
    weight_sum = weights.sum(dim=1, keepdim=True)
    weight_sum = torch.where(weight_sum == 0, torch.ones_like(weight_sum), weight_sum)

    return Fragments(
        pixel_index=pixel_index,
        tri_index=tri_index,
        bary=weights / weight_sum,
        bary_screen=bary_screen,
        depth=(bary_screen * depth).sum(dim=1),
    )
