"""Ray-caster registry.

Ray-mesh intersection is pluggable in the same way the render backend is: ``embree`` by
default, ``warp`` when a workload is large enough to pay for the GPU.

    from alfspy.core.raycast import create_raycaster

    caster = create_raycaster(mesh_data)                 # or backend='warp'
    hits, ray_indices = caster.intersects_first(origins, directions)

Selection precedence is an explicit argument, then ``$ALFS_RAYCASTER``, then
:data:`DEFAULT_RAYCASTER`.
"""

import importlib
import os
from typing import Dict, List, Optional, Union

from numpy.typing import NDArray

from .base import RayCaster

__all__ = [
    'DEFAULT_RAYCASTER',
    'RAYCASTER_ENV_VAR',
    'RayCaster',
    'available_raycasters',
    'create_raycaster',
    'get_raycaster',
    'raycaster_names',
    'resolve_raycaster',
]

RAYCASTER_ENV_VAR = 'ALFS_RAYCASTER'

# embree is the default because label projection casts tens of rays per frame, where the GPU
# backend cannot win: below roughly 1e4 rays, launch and transfer overhead dominate.
DEFAULT_RAYCASTER = 'embree'

_MODULES: Dict[str, str] = {
    'embree': 'alfspy.core.raycast.embree_',
    'warp': 'alfspy.core.raycast.warp_',
}

_CLASSES: Dict[str, str] = {
    'embree': 'EmbreeRayCaster',
    'warp': 'WarpRayCaster',
}


def raycaster_names() -> tuple:
    """
    :return: The names of every registered ray caster, installed or not.
    """
    return tuple(_MODULES)


def get_raycaster(name: str):
    """
    Imports and returns a ray-caster class.

    :param name: The backend name, ``"embree"`` or ``"warp"``.
    :return: The ``RayCaster`` subclass.
    :raises ValueError: If no ray caster is registered under that name.
    :raises ImportError: If its dependency is not installed, with a message naming the extra
        that provides it -- the extra is named after the caster, as the engine extras are
        named after the engines.
    """
    if name not in _MODULES:
        raise ValueError(
            f'Unknown ray caster {name!r}. Registered: {", ".join(_MODULES)}.')
    try:
        module = importlib.import_module(_MODULES[name])
    except ImportError as exc:
        raise ImportError(
            f'The {name!r} ray caster is registered but its dependency is not installed. '
            f'Install it with `pip install "AlfsPy[{name}]"`. Original error: {exc}'
        ) from exc
    return getattr(module, _CLASSES[name])


def available_raycasters() -> List[str]:
    """
    :return: The ray casters that are installed and usable here.
    """
    found = []
    for name in _MODULES:
        try:
            module = importlib.import_module(_MODULES[name])
        except ImportError:
            continue
        try:
            if module.is_available():
                found.append(name)
        except Exception:
            continue
    return found


def resolve_raycaster(backend: Optional[str] = None) -> str:
    """
    :param backend: An explicit backend name (optional).
    :return: The resolved name -- the argument, else ``$ALFS_RAYCASTER``, else
        :data:`DEFAULT_RAYCASTER`.
    """
    if backend is not None:
        return backend
    return os.environ.get(RAYCASTER_ENV_VAR) or DEFAULT_RAYCASTER


def create_raycaster(mesh, backend: Optional[str] = None, **kwargs) -> RayCaster:
    """
    Builds a ray caster for a mesh. The acceleration structure is built here, once.

    :param mesh: The mesh to intersect against -- a ``MeshData``, a ``Trimesh``, or a
        ``(vertices, faces)`` pair.
    :param backend: Which ray caster to use (optional).
    :param kwargs: Forwarded to the backend, e.g. ``device=`` for Warp.
    :return: A ready-to-use ray caster.
    """
    vertices, faces = _as_vertices_faces(mesh)
    return get_raycaster(resolve_raycaster(backend))(vertices, faces, **kwargs)


def _as_vertices_faces(mesh):
    """Accepts the several mesh spellings this codebase passes around."""
    if isinstance(mesh, RayCaster):
        raise TypeError('Expected a mesh, got a RayCaster')

    vertices = getattr(mesh, 'vertices', None)
    if vertices is not None:
        faces = getattr(mesh, 'faces', None)
        if faces is None:
            faces = getattr(mesh, 'indices', None)
        if faces is None:
            raise ValueError('Mesh has vertices but no faces/indices')
        return vertices, faces

    try:
        vertices, faces = mesh
    except (TypeError, ValueError):
        raise TypeError(f'Cannot interpret {type(mesh).__name__} as a mesh')
    return vertices, faces
