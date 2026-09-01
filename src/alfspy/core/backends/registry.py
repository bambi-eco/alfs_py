"""Render-backend registry.

Every backend implements the same three GPU operations -- render the textured DEM, project a
shot onto it, integrate several shots -- against a different API. This module is how one gets
chosen.

A backend module must expose:

``create_context(device=None, **options)``  build a render context in the standard state
``reset_state(ctx)``                        return a context to that standard state
``is_available()``                          whether a context can be created here
``owns_context(ctx)``                       whether a context belongs to this backend
``Renderer``/``CtxShot``/``RenderObject``/``img_from_fbo``

That signature is deliberately identical across backends -- ``device`` first, then
``**options`` -- so :func:`make_context` behaves the same whichever engine is chosen and only
the engine argument changes the result. Backends honour the options that apply to them and
ignore the rest.

Backends are imported lazily and cached, so an uninstalled optional dependency only fails
when that backend is actually requested -- importing ``alfspy`` never pulls in torch or
moderngl by itself.
"""

import importlib
import os
from types import ModuleType
from typing import Dict, List, Optional

__all__ = [
    'DEFAULT_ENGINE',
    'DEVICE_ENV_VAR',
    'ENGINE_ENV_VAR',
    'engine_names',
    'get_backend',
    'available_engines',
    'resolve_device',
    'resolve_engine',
    'create_context',
    'make_context',
    'backend_for_context',
]

ENGINE_ENV_VAR = 'ALFS_ENGINE'

# The device counterpart of ALFS_ENGINE. A deployment that selects an engine through the
# environment has to be able to select the device the same way -- otherwise ALFS_ENGINE=torch
# is only half an answer, and every caller has to plumb `device=` through itself to finish it.
DEVICE_ENV_VAR = 'ALFS_DEVICE'

# ModernGL is the default because it is what this project rendered with historically, so an
# unqualified call keeps producing what it produced before. It is a deliberate, explicit
# default rather than "whichever backend happens to import": a silent fallback would make a
# render depend on which machine it ran on, which is not acceptable for a scientific tool.
# Headless deployments set ALFS_ENGINE=torch (the Dockerfile does).
DEFAULT_ENGINE = 'moderngl'

_MODULES: Dict[str, str] = {
    'moderngl': 'alfspy.core.backends.moderngl_',
    'torch': 'alfspy.core.backends.torch_',
    'vulkan': 'alfspy.core.backends.wgpu_',
}

# The extra that provides each backend, for the error message when one is missing.
_EXTRAS: Dict[str, str] = {
    'moderngl': 'moderngl',
    'torch': 'torch',
    'vulkan': 'vulkan',
}

_CACHE: Dict[str, ModuleType] = {}


def engine_names() -> tuple:
    """
    :return: The names of every registered backend, whether or not it is installed.
    """
    return tuple(_MODULES)


def get_backend(name: str) -> ModuleType:
    """
    Imports and returns a backend module.

    :param name: The backend name, e.g. ``"moderngl"`` or ``"torch"``.
    :return: The backend module.
    :raises ValueError: If no backend is registered under that name.
    :raises ImportError: If the backend's dependency is not installed. The message names the
        extra that provides it, because "No module named 'moderngl'" is not actionable.
    """
    if name in _CACHE:
        return _CACHE[name]

    if name not in _MODULES:
        raise ValueError(
            f'Unknown render engine {name!r}. Registered engines: {", ".join(_MODULES)}.')

    try:
        module = importlib.import_module(_MODULES[name])
    except ImportError as exc:
        raise ImportError(
            f'The {name!r} render backend is registered but its dependency is not installed. '
            f'Install it with `pip install "AlfsPy[{_EXTRAS[name]}]"`. Original error: {exc}'
        ) from exc

    _CACHE[name] = module
    return module


def available_engines() -> List[str]:
    """
    :return: The backends that are installed *and* can actually create a context here. This
        probes rather than merely importing: ``moderngl`` imports fine on a machine with no
        usable GL driver and only fails when a context is created.
    """
    found = []
    for name in _MODULES:
        try:
            backend = get_backend(name)
        except ImportError:
            continue
        try:
            if backend.is_available():
                found.append(name)
        except Exception:
            continue
    return found


def resolve_engine(engine: Optional[str] = None) -> str:
    """
    Determines which backend to use.

    Precedence: an explicit argument, then ``$ALFS_ENGINE``, then :data:`DEFAULT_ENGINE`.

    :param engine: An explicit engine name (optional).
    :return: The resolved engine name.
    """
    if engine is not None:
        return engine
    return os.environ.get(ENGINE_ENV_VAR) or DEFAULT_ENGINE


def resolve_device(device: Optional[str] = None) -> Optional[str]:
    """
    Determines which device to render on.

    Precedence matches :func:`resolve_engine`: an explicit argument, then ``$ALFS_DEVICE``,
    then ``None`` -- which leaves the choice to the backend, since each has its own sensible
    answer (torch picks CUDA when it is there, ModernGL has no device to pick).

    :param device: An explicit device name (optional).
    :return: The resolved device, or ``None`` to let the backend decide.
    """
    if device is not None:
        return device
    return os.environ.get(DEVICE_ENV_VAR) or None


def make_context(engine: Optional[str] = None, device: Optional[str] = None, **options):
    """
    Creates a render context for the selected backend.

    Every backend implements this same signature, so **only ``engine`` changes what you get**:
    the same call works unchanged across ModernGL, torch and Vulkan. Backends honour what
    applies to them and ignore the rest rather than raising, which is what lets a caller
    switch engines without also rewriting its arguments.

        make_context()                              # $ALFS_ENGINE / $ALFS_DEVICE, or defaults
        make_context('torch', device='cuda')
        make_context('vulkan', device='cpu')        # software adapter
        make_context('moderngl', backend='egl')     # headless Linux GL

    :param engine: The backend to use (optional). Defaults to ``$ALFS_ENGINE`` and then to
        :data:`DEFAULT_ENGINE`.
    :param device: Which device to render on -- ``"cpu"``, ``"cuda"``, ``"cuda:1"``
        (optional). Defaults to ``$ALFS_DEVICE``, then to the backend's own choice. Torch uses
        it directly; Vulkan maps ``"cpu"`` onto the software fallback adapter; ModernGL
        ignores it, because OpenGL offers no device selection.
    :param options: Backend-specific extras, documented on each backend's ``create_context``.
        Unknown keys are ignored so that options meant for one engine do not break another.
    :return: A backend-specific render context.
    """
    name = resolve_engine(engine)
    return get_backend(name).create_context(device=resolve_device(device), **options)


#: Alias of :func:`make_context`, kept because 3.0.0 shipped with this name.
create_context = make_context


def backend_for_context(ctx) -> ModuleType:
    """
    Finds the backend that owns a context.

    This is what lets ``Renderer(resolution, ctx, ...)`` keep its original signature while
    dispatching to the right implementation: the context *is* the engine handle.

    :param ctx: A render context produced by some backend.
    :return: The backend module that owns it.
    :raises TypeError: If no registered backend recognises the context.
    """
    for name in _MODULES:
        try:
            backend = get_backend(name)
        except ImportError:
            continue
        try:
            if backend.owns_context(ctx):
                return backend
        except Exception:
            continue

    raise TypeError(
        f'No registered render backend recognises a context of type '
        f'{type(ctx).__module__}.{type(ctx).__name__}. '
        f'Create one with alfspy.core.backends.make_context(engine); '
        f'available here: {", ".join(available_engines()) or "none"}.')
