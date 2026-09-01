"""Rendering light fields with an arbitrary number of channels.

An ALFS integral is not inherently a three-channel operation: the renderer projects a source
image onto the DEM and averages the contributions. Nothing in that requires the image to be
RGB. Feeding it a dense feature map instead -- DINOv3 patch embeddings, say -- gives an
*embedded light field*: a novel view where every pixel carries a 1280-dimensional descriptor
rather than a colour.

The GPU is the constraint: an OpenGL colour attachment holds at most four components, so a
field wider than that is rendered in groups and reassembled. This module owns that grouping
so it is written once rather than per backend, and so the three defects the original
prototype carried cannot come back:

**Coverage is not stolen from the data.** The prototype packed feature dimension
``ch_start + 3`` into the alpha channel, which the normaliser then divided by -- so channels
3, 7, ... 1279, a full 25% of the output, came back as binary coverage masks and the rest
were divided by an arbitrary signed activation. Coverage is its own render target now (see
:class:`~alfspy.core.rendering.data.IntegralResult`), and it is computed once and reused for
every group, because geometry does not change between them.

**Textures keep their orientation.** The prototype uploaded slices by calling ``ctx.texture``
directly, bypassing ``TextureData``, which is where the vertical flip into GL's bottom-up
order lives -- so features were sampled from the mirrored half of every shot. Everything here
goes through ``TextureData``.

**Fields upload at their own resolution.** The prototype resized every slice up to the render
resolution on the CPU and re-uploaded it per group, moving roughly 1.25 TB per output frame.
Texture sampling is already bilinear, so the GPU performs that upsample for free: a 128x128
patch grid is uploaded as a 128x128 texture. That alone is a ~256x reduction in per-slice
upload.
"""

import math
from dataclasses import dataclass, field as dataclass_field
from typing import Iterable, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from pyrr import Quaternion, Vector3

from alfspy.core.geo import Transform
from alfspy.core.rendering import CtxShot, TextureData
from alfspy.core.rendering.data import IntegralResult

__all__ = ['ChannelSpec', 'FieldShot', 'render_field_integral']

# An OpenGL colour attachment holds at most four components.
_GROUP = 4


@dataclass
class ChannelSpec:
    """
    What the channels of a field mean.

    ``.npy`` carries no metadata, so without this a saved field is an anonymous ``(H, W, C)``
    block and nothing records what produced it or what the channels are.

    :cvar count: How many channels the field has.
    :cvar names: A name per channel (optional). Useful for small semantic fields; normally
        left unset for learned descriptors, where the dimensions have no individual meaning.
    :cvar source: What produced the channels, e.g. a model identifier.
    :cvar dtype: The storage dtype.
    """
    count: int
    names: Optional[List[str]] = None
    source: Optional[str] = None
    dtype: str = 'float32'

    def __post_init__(self):
        if self.count < 1:
            raise ValueError(f'A field needs at least one channel, got {self.count}')
        if self.names is not None and len(self.names) != self.count:
            raise ValueError(
                f'Got {len(self.names)} channel names for {self.count} channels')

    @property
    def groups(self) -> int:
        """
        :return: How many render passes this field needs on a four-component backend.
        """
        return math.ceil(self.count / _GROUP)

    def as_dict(self) -> dict:
        """
        :return: A JSON-serialisable description, for the sidecar metadata of a saved field.
        """
        return {
            'count': self.count,
            'names': self.names,
            'source': self.source,
            'dtype': self.dtype,
        }


@dataclass
class FieldShot:
    """
    One capture whose image is an N-channel field rather than an RGB frame.

    This is pure data: no context, no texture. The per-group textures are built during the
    render, which is what lets the same shot list be handed to any backend.

    :cvar field: The ``(H, W, C)`` field. Typically far lower resolution than the render --
        a patch grid, for instance -- which is fine and desirable: texture sampling
        interpolates it on the GPU.
    :cvar position: The camera position of the capture.
    :cvar rotation: The camera rotation of the capture.
    :cvar fovy: Vertical field of view in degrees.
    :cvar aspect_ratio: The camera aspect ratio.
    :cvar correction: Pose correction to apply to this shot (optional).
    """
    field: NDArray
    position: Vector3
    rotation: Quaternion
    fovy: float = 60.0
    aspect_ratio: float = 1.0
    correction: Optional[Transform] = None

    def __post_init__(self):
        arr = np.asarray(self.field)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        if arr.ndim != 3:
            raise ValueError(f'A field shot needs an (H, W, C) array, got shape {arr.shape}')
        self.field = arr

    @property
    def channels(self) -> int:
        """
        :return: The number of channels this shot carries.
        """
        return self.field.shape[2]

    def slice_texture(self, start: int, end: int) -> TextureData:
        """
        Builds the texture for one channel group, padded to four components.

        The slice is uploaded at the field's own resolution. Resizing it to the render
        resolution first -- as the original prototype did -- costs a CPU interpolation and a
        far larger upload for a result the GPU's bilinear sampling produces anyway.

        :param start: First channel of the group.
        :param end: One past the last channel of the group.
        :return: Texture data for the group.
        """
        chunk = self.field[:, :, start:end]
        if chunk.shape[2] < _GROUP:
            pad = np.zeros((*chunk.shape[:2], _GROUP - chunk.shape[2]), dtype=chunk.dtype)
            chunk = np.concatenate([chunk, pad], axis=2)
        return TextureData(np.ascontiguousarray(chunk, dtype=np.float32))


def _group_shots(ctx, shots: Sequence[FieldShot], start: int, end: int) -> List[CtxShot]:
    return [
        CtxShot(
            ctx,
            shot.slice_texture(start, end).texture,
            shot.position,
            shot.rotation,
            fovy=shot.fovy,
            aspect_ratio=shot.aspect_ratio,
            correction=shot.correction,
            # A feature activation of 3.0 is a value, not an 8-bit colour. Without this the
            # "rescale anything above 1 by 1/255" heuristic silently divides the field.
            normalise=False,
        )
        for shot in shots
    ]


def render_field_integral(renderer, ctx, shots: Sequence[FieldShot],
                          channels: Optional[int] = None,
                          mask: Optional[TextureData] = None,
                          out: Optional[NDArray] = None,
                          progress=None) -> IntegralResult:
    """
    Renders the ALFS integral of an N-channel field.

    :param renderer: A renderer already built for the target camera and DEM.
    :param ctx: The render context the renderer belongs to.
    :param shots: The field shots to integrate. All must have the same channel count.
    :param channels: How many channels to render (optional). Defaults to every channel the
        shots carry; pass fewer to render a prefix.
    :param mask: The projection mask (optional).
    :param out: An array to write the accumulation into (optional). Pass a memmap for fields
        too large to hold in RAM -- a 2048x2048x1280 float32 field is about 21 GB.
    :param progress: Called as ``progress(group_index, group_count)`` after each render pass
        (optional).
    :return: The accumulated field and the per-pixel coverage. Divide with
        :meth:`~alfspy.core.rendering.data.IntegralResult.normalised`.
    :raises ValueError: If the shots disagree about their channel count, or there are none.
    """
    shots = list(shots)
    if not shots:
        raise ValueError('render_field_integral needs at least one shot')

    counts = {shot.channels for shot in shots}
    if len(counts) != 1:
        raise ValueError(f'All field shots must have the same channel count, got {sorted(counts)}')

    available = counts.pop()
    total = available if channels is None else channels
    if not 1 <= total <= available:
        raise ValueError(f'Asked for {total} channels but the shots carry {available}')

    height, width = renderer.render_shape[0], renderer.render_shape[1]
    if out is None:
        out = np.zeros((height, width, total), dtype=np.float32)
    elif out.shape != (height, width, total):
        raise ValueError(
            f'`out` has shape {out.shape}, expected {(height, width, total)}')

    coverage: Optional[NDArray] = None
    group_count = math.ceil(total / _GROUP)

    for index, start in enumerate(range(0, total, _GROUP)):
        end = min(start + _GROUP, total)
        group = _group_shots(ctx, shots, start, end)
        try:
            result = renderer.render_integral_raw(group, mask=mask)
        finally:
            for shot in group:
                shot.release()

        out[:, :, start:end] = result.accum[:, :, :end - start]

        if coverage is None:
            # Coverage depends only on geometry, which does not change between groups, so it
            # is rendered once and reused. The prototype had no coverage pass at all.
            coverage = result.coverage

        if hasattr(out, 'flush') and (index + 1) % 32 == 0:
            out.flush()  # evict written pages when `out` is a memmap

        if progress is not None:
            progress(index + 1, group_count)

    if hasattr(out, 'flush'):
        out.flush()

    return IntegralResult(accum=out, coverage=coverage)
