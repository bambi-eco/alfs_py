"""Render backends.

Each backend implements the same three GPU operations -- render the textured DEM, project a
shot onto it, and integrate several shots -- against a different API. Backends are imported
lazily, so a missing optional dependency only fails when that backend is actually requested;
importing ``alfspy`` pulls in neither torch nor moderngl.

    from alfspy.core.backends import create_context, available_engines

    ctx = create_context(engine='torch')     # or 'moderngl', or $ALFS_ENGINE
"""

from .registry import (
    DEFAULT_ENGINE,
    ENGINE_ENV_VAR,
    available_engines,
    backend_for_context,
    create_context,
    engine_names,
    get_backend,
    resolve_engine,
)

__all__ = [
    'DEFAULT_ENGINE',
    'ENGINE_ENV_VAR',
    'available_engines',
    'backend_for_context',
    'create_context',
    'engine_names',
    'get_backend',
    'resolve_engine',
]
