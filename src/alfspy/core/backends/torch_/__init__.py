"""The PyTorch render backend.

A pure-tensor reimplementation of the fixed-function pipeline the ModernGL backend gets from
the driver: vertex transform, triangle setup, binned rasterisation with GL's top-left fill
rule, a scatter-min depth resolve, and ``grid_sample``-based texture lookup. It needs no GL
driver and no display, which is what makes headless deployment possible without Xvfb.

The low-level layer lives in :mod:`alfspy.core.torchgl`; this package is the adapter that
presents it through the interface :mod:`alfspy.core.backends.registry` expects.
"""

from alfspy.core.torchgl import BLEND, CULL_FACE, DEPTH_TEST, TorchContext

from .context import create_context, is_available, reset_state
from .data import RenderObject
from .framebuffer import img_from_fbo
from .renderer import Renderer
from .shot import CtxShot

__all__ = [
    'BLEND',
    'CULL_FACE',
    'DEPTH_TEST',
    'CtxShot',
    'RenderObject',
    'Renderer',
    'TorchContext',
    'create_context',
    'img_from_fbo',
    'is_available',
    'owns_context',
    'reset_state',
]


def owns_context(ctx) -> bool:
    """
    :param ctx: Any render context.
    :return: Whether it belongs to this backend.
    """
    return isinstance(ctx, TorchContext)
