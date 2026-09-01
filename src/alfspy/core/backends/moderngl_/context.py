"""ModernGL context creation.

Previously this lived in two places that had drifted apart: ``render.render.make_mgl_context``
enabled depth testing and back-face culling, while ``ProjectionScene`` created its context
inline and enabled neither. Renders therefore depended on which entry point had been used.
There is one factory now, and it sets the state.
"""

from typing import Optional, cast

import moderngl as mgl

__all__ = ['create_context', 'is_available']


def create_context(backend: Optional[str] = None, standalone: bool = True) -> mgl.Context:
    """
    Creates a ModernGL context in the pipeline's standard state.

    :param backend: An explicit ModernGL backend, e.g. ``"egl"`` for headless Linux
        (optional). Leave as ``None`` for the platform default.
    :param standalone: Whether to create a standalone (windowless) context. Defaults to
        ``True``; there is no on-screen surface in this pipeline.
    :return: A configured ModernGL context.
    """
    if backend is not None:
        ctx = mgl.create_standalone_context(backend=backend)
    else:
        ctx = mgl.create_context(standalone=standalone)

    ctx.enable(cast(int, mgl.DEPTH_TEST))
    ctx.enable(cast(int, mgl.CULL_FACE))
    ctx.cull_face = 'back'
    return ctx


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
