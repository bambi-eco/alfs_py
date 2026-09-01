"""The ModernGL (OpenGL) render backend.

This is the original AlfsPy renderer, moved here unchanged in behaviour when the PyTorch
implementation was merged in. It owns everything that touches an OpenGL context: the two
GLSL programs, the framebuffer, VAO/buffer/texture handles and the readback.

Backend-agnostic types (``Camera``, ``MeshData``, ``TextureData``, ``Resolution``,
``Transform``) are shared with every other backend and continue to live in
``alfspy.core.rendering`` and ``alfspy.core.geo``.
"""

import moderngl as _mgl

from .context import create_context, is_available, reset_state
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
]


def owns_context(ctx) -> bool:
    """
    :param ctx: Any render context.
    :return: Whether it belongs to this backend.
    """
    return isinstance(ctx, _mgl.Context)
