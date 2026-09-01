"""Rendering types shared by every backend, plus the backend-dispatching facades.

``Camera``, ``Resolution``, ``MeshData``, ``TextureData`` and ``RenderResultMode`` are plain
numpy/pyrr and carry no device state, so they are the same object regardless of which backend
renders them. ``Renderer`` and ``CtxShot`` are facades that dispatch on the context they are
given -- see :mod:`alfspy.core.backends.registry`.

``RenderObject`` is deliberately *not* re-exported here any more: it is a mesh already
uploaded to a device (a VAO under ModernGL, tensors under torch), so it is backend-specific
by nature. Import it from the backend package if you need it.
"""

from .camera import Camera
from .data import MeshData, RenderResultMode, Resolution, TextureData

from .shot import CtxShot

from .shot_loader import SyncShotLoader, AsyncShotLoader

from .renderer import Renderer

__all__ = [
    'Camera',
    'CtxShot',
    'MeshData',
    'RenderResultMode',
    'Renderer',
    'Resolution',
    'SyncShotLoader',
    'AsyncShotLoader',
    'TextureData',
]
