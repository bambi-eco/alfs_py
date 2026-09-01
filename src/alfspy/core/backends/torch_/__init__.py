"""The PyTorch render backend.

A pure-tensor reimplementation of the fixed-function pipeline the ModernGL backend gets from
the driver: vertex transform, triangle setup, binned rasterisation with GL's top-left fill
rule, a scatter-min depth resolve, and ``grid_sample``-based texture lookup. It needs no GL
driver and no display, which is what makes headless deployment possible without Xvfb.

The low-level layer lives in :mod:`alfspy.core.torchgl`; this package is the adapter that
presents it through the interface :mod:`alfspy.core.backends.registry` expects.
"""

from alfspy.core.torchgl import BLEND, CULL_FACE, DEPTH_TEST, TorchContext, create_context

from .data import RenderObject
from .framebuffer import img_from_fbo
from .renderer import Renderer
from .shot import CtxShot

__all__ = [
    'create_context',
    'is_available',
    'owns_context',
    'reset_state',
    'img_from_fbo',
    'CtxShot',
    'RenderObject',
    'Renderer',
    'TorchContext',
]


def reset_state(ctx: TorchContext) -> None:
    """
    Returns a context to the pipeline's standard state.

    ``Renderer.render_integral`` switches to additive blending and disables depth testing, so
    anything reusing a context across renders should reset it first.

    :param ctx: The context to reset.
    """
    ctx.enable(DEPTH_TEST)
    ctx.enable(CULL_FACE)
    ctx.cull_face = 'back'
    ctx.disable(BLEND)


def is_available() -> bool:
    """
    :return: Whether this backend can render here. Always ``True`` once torch imports -- the
        rasteriser runs on the CPU, so unlike the GL backend there is nothing to probe.
    """
    return True


def owns_context(ctx) -> bool:
    """
    :param ctx: Any render context.
    :return: Whether it belongs to this backend.
    """
    return isinstance(ctx, TorchContext)
