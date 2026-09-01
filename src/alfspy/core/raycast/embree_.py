"""The default ray caster: trimesh, accelerated by Embree via embreex.

trimesh picks its intersector at import: ``ray_pyembree`` when the Embree bindings are
importable, ``ray_triangle`` (pure Python/NumPy) otherwise. That fallback is silent, and it
is not merely slow -- it returns hits in a different index order, which broke five
coordinate-conversion round-trip tests. :func:`using_embree` exists so a caller (or a test)
can tell which one is actually in play rather than assuming.

Note trimesh 3.x looks for the long-obsolete ``pyembree`` module and never finds ``embreex``,
so it always falls back. The package requires trimesh >= 4 for this reason.
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from trimesh import Trimesh

from .base import RayCaster

__all__ = ['EmbreeRayCaster', 'is_available', 'using_embree']


def using_embree(mesh: Trimesh) -> bool:
    """
    :param mesh: A ``Trimesh``.
    :return: Whether trimesh selected the Embree-backed intersector for it, rather than
        silently falling back to the pure-Python one.
    """
    return type(mesh.ray).__module__.endswith('ray_pyembree')


def is_available() -> bool:
    """
    :return: Always ``True`` -- trimesh is a hard dependency and always provides *an*
        intersector. Whether it is the accelerated one is :func:`using_embree`.
    """
    return True


class EmbreeRayCaster(RayCaster):
    """
    Ray casting through ``trimesh.Trimesh.ray``.
    """

    name = 'embree'

    def _build(self, vertices: NDArray, faces: NDArray) -> None:
        self._mesh = Trimesh(vertices=vertices, faces=faces, process=False)
        # Touch the intersector so the structure is built here rather than on the first cast,
        # keeping construction cost where a caller can see it.
        self._accelerated = using_embree(self._mesh)

    @property
    def accelerated(self) -> bool:
        """
        :return: Whether this caster got the Embree intersector rather than the pure-Python
            fallback.
        """
        return self._accelerated

    def _intersect(self, origins: NDArray, directions: NDArray) -> Tuple[NDArray, NDArray]:
        hits, ray_indices, *_ = self._mesh.ray.intersects_location(
            origins, directions, multiple_hits=False)
        return hits, ray_indices

    def release(self) -> None:
        if not self._released:
            self._mesh = None
        super().release()
