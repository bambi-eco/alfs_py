"""The ModernGL (OpenGL) render backend.

This is the original AlfsPy renderer, moved here unchanged in behaviour when the PyTorch
implementation was merged in. It owns everything that touches an OpenGL context: the two
GLSL programs, the framebuffer, VAO/buffer/texture handles and the readback.

Backend-agnostic types (``Camera``, ``MeshData``, ``TextureData``, ``Resolution``,
``Transform``) are shared with every other backend and continue to live in
``alfspy.core.rendering`` and ``alfspy.core.geo``.
"""

from .context import create_context, is_available
from .data import RenderObject
from .framebuffer import img_from_fbo
from .renderer import Renderer
from .shot import CtxShot

__all__ = [
    'create_context',
    'is_available',
    'img_from_fbo',
    'CtxShot',
    'RenderObject',
    'Renderer',
]
