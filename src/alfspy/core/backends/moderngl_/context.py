"""ModernGL context creation.

Previously this lived in two places that had drifted apart: ``render.render.make_mgl_context``
enabled depth testing and back-face culling, while ``ProjectionScene`` created its context
inline and enabled neither. Renders therefore depended on which entry point had been used.
There is one factory now, and it sets the state.
"""

from typing import Optional, cast

import moderngl as mgl

__all__ = ['create_context', 'is_available']


def create_context(device: Optional[str] = None, **options) -> mgl.Context:
    """
    Creates a ModernGL context in the pipeline's standard state.

    Every backend takes this same signature, so ``create_context(engine=...)`` behaves
    identically whichever engine is chosen and only the engine changes the result.

    :param device: Accepted for interface compatibility and ignored. OpenGL offers no device
        selection -- the driver picks the adapter -- so unlike the torch and Vulkan backends
        there is nothing to honour here.
    :param options: Backend-specific extras. This one understands ``backend`` (an explicit
        ModernGL backend such as ``"egl"`` for headless Linux) and ``standalone`` (whether to
        create a windowless context, default ``True``). Anything else is ignored, so options
        meant for another engine do not raise.
    :return: A configured ModernGL context.
    """
    backend = options.get('backend')
    standalone = options.get('standalone', True)

    if backend is not None:
        ctx = mgl.create_standalone_context(backend=backend)
    else:
        ctx = mgl.create_context(standalone=standalone)

    ctx.enable(cast(int, mgl.DEPTH_TEST))
    ctx.enable(cast(int, mgl.CULL_FACE))
    ctx.cull_face = 'back'
    return ctx


def reset_state(ctx: mgl.Context) -> None:
    """
    Returns a context to the pipeline's standard state.

    ``Renderer.render_integral`` switches to additive blending and disables depth testing, and
    only re-disables blending afterwards -- so a context that has integrated once is left with
    depth testing off. Anything reusing a context across renders should reset it first.

    :param ctx: The context to reset.
    """
    ctx.enable(cast(int, mgl.DEPTH_TEST))
    ctx.enable(cast(int, mgl.CULL_FACE))
    ctx.cull_face = 'back'
    ctx.disable(cast(int, mgl.BLEND))


def is_available() -> bool:
    """
    :return: Whether a ModernGL context can actually be created here. Importing ``moderngl``
        succeeds on machines with no usable GL driver, so availability has to be probed by
        creating a context rather than by catching an ``ImportError``.
    """
    try:
        ctx = create_context()
    except Exception:
        return False
    ctx.release()
    return True
