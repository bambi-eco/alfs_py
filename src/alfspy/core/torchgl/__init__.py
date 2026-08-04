"""PyTorch replacement for the ModernGL rendering primitives used by ALFS.

The package provides exactly what the ALFS renderer needs -- a context, textures, a render
target, a depth-buffered rasteriser and the two shader programs -- and nothing else. It is
not a general OpenGL emulation.

See ``docs/MIGRATION_REVIEW.md`` for the design rationale and
``docs/pyrr_conventions.md`` for the matrix conventions everything here assumes.
"""

from .compat import COMPAT_NOTES, HAS_SCATTER_REDUCE, TORCH_VERSION, as_matrix, as_tensor, to_numpy
from .context import (
    ADDITIVE_BLENDING,
    BLEND,
    CULL_FACE,
    DEFAULT_BLENDING,
    DEPTH_TEST,
    TorchContext,
    create_context,
)
from .framebuffer import TorchFramebuffer
from .programs import shade_object, shade_shot, shot_clip_coords
from .raster import (
    DEFAULT_SAMPLE_BUDGET,
    Fragments,
    TriangleSetup,
    iter_fragments,
    resolve_depth,
    setup_triangles,
    transform_vertices,
)
from .texture import TorchTexture, sample_texture

__all__ = [
    'ADDITIVE_BLENDING',
    'BLEND',
    'COMPAT_NOTES',
    'CULL_FACE',
    'DEFAULT_BLENDING',
    'DEFAULT_SAMPLE_BUDGET',
    'DEPTH_TEST',
    'Fragments',
    'HAS_SCATTER_REDUCE',
    'TORCH_VERSION',
    'TorchContext',
    'TorchFramebuffer',
    'TorchTexture',
    'TriangleSetup',
    'as_matrix',
    'as_tensor',
    'create_context',
    'iter_fragments',
    'resolve_depth',
    'sample_texture',
    'setup_triangles',
    'shade_object',
    'shade_shot',
    'shot_clip_coords',
    'to_numpy',
    'transform_vertices',
]
