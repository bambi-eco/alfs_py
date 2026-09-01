"""Render backends.

Each backend implements the same three GPU operations -- render the textured DEM, project a
shot onto it, and integrate several shots -- against a different API. Backends are imported
lazily, so a missing optional dependency only fails when that backend is actually requested;
importing ``alfspy`` pulls in neither torch nor moderngl.

    from alfspy.core.backends import make_context, available_engines

    ctx = make_context('torch', device='cuda')   # or 'moderngl' / 'vulkan', or $ALFS_ENGINE

Every backend implements the same ``create_context(device=None, **options)`` signature, so
only the engine argument changes what you get.
"""

from .registry import (
    DEFAULT_ENGINE,
    DEVICE_ENV_VAR,
    ENGINE_ENV_VAR,
    available_engines,
    backend_for_context,
    create_context,
    engine_names,
    make_context,
    get_backend,
    resolve_device,
    resolve_engine,
)

__all__ = [
    'DEFAULT_ENGINE',
    'DEVICE_ENV_VAR',
    'ENGINE_ENV_VAR',
    'available_engines',
    'backend_for_context',
    'create_context',
    'engine_names',
    'make_context',
    'get_backend',
    'resolve_device',
    'resolve_engine',
]
