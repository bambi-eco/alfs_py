"""Torch context creation.

An adapter over :func:`alfspy.core.torchgl.create_context` that gives this backend the same
``create_context(device=None, **options)`` signature as every other one, so
``create_context(engine=...)`` behaves identically whichever engine is chosen.
"""

from typing import Optional

from alfspy.core.torchgl import BLEND, CULL_FACE, DEPTH_TEST, TorchContext
from alfspy.core.torchgl import create_context as _torchgl_create_context

__all__ = ['create_context', 'is_available', 'reset_state']


def create_context(device: Optional[str] = None, **options) -> TorchContext:
    """
    Creates a torch render context in the pipeline's standard state.

    :param device: The torch device, e.g. ``"cpu"``, ``"cuda"`` or ``"cuda:1"`` (optional).
        Defaults to CUDA when one is visible, otherwise the CPU.
    :param options: Backend-specific extras. This one understands ``dtype`` (``float32`` by
        default; ``float64`` separates algorithmic error from float32 rounding) and
        ``sample_budget`` (how many rasterisation samples to process per chunk, which caps
        peak memory independently of mesh size). Anything else is ignored, so options meant
        for another engine do not raise.
    :return: A configured ``TorchContext`` with depth testing and back-face culling enabled.
    """
    known = {key: options[key] for key in ('dtype', 'sample_budget') if key in options}
    return _torchgl_create_context(device=device, **known)


def is_available() -> bool:
    """
    :return: Whether this backend can render here. Always ``True`` once torch imports -- the
        rasteriser runs on the CPU, so unlike the GL backend there is nothing to probe.
    """
    return True


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
