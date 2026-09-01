"""The ray-caster interface.

Ray-mesh intersection is used for label projection: ``pixel_to_world_coord`` un-projects
label corners onto the DEM. The whole surface is one operation -- first hit of a batch of
rays against a static mesh -- so the interface is one method.

Two things live in the base class because they are backend-independent and both backends
need them:

**The acceleration structure is built once.** ``cast_ray`` used to take a *mesh* per call and
rebuild a ``Trimesh``, and therefore its BVH, whenever it was handed raw ``MeshData``. The
docstring warned about this in prose, which meant performance correctness depended on every
caller having read it. Measured on a 131k-triangle DEM, an accidental rebuild costs 0.75 s
with the pure-Python intersector and 0.095 s with embree. A ray caster is an object that owns
its structure, so reuse is structural.

**Rays are cast in a mesh-local frame.** Both embree and Warp traverse in ``float32``, and
GLTF stores positions as ``float32`` too. A geo-referenced DEM with UTM northings around
5e6 therefore has only about 0.5 m of representable precision. Subtracting the mesh's AABB
centre in ``float64`` before casting and adding it back afterwards costs one vector add per
ray and removes that cliff.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

__all__ = ['RayCaster']


class RayCaster(ABC):
    """
    Casts batches of rays against one static mesh.

    :cvar name: The registry name of this backend.
    """

    name: str = ''

    def __init__(self, vertices: NDArray, faces: NDArray):
        """
        :param vertices: The mesh vertices as an ``(V, 3)`` array.
        :param faces: The triangle indices as an ``(F, 3)`` array.
        """
        vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(faces).reshape(-1, 3)

        # Cast in a local frame; see the module docstring.
        self._origin = ((vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
                        if len(vertices) else np.zeros(3))
        self._local_vertices = vertices - self._origin
        self._faces = faces
        self._released = False
        self._build(self._local_vertices, faces)

    @abstractmethod
    def _build(self, vertices: NDArray, faces: NDArray) -> None:
        """
        Builds the acceleration structure. Called once, from ``__init__``.

        :param vertices: Mesh vertices already translated into the local frame.
        :param faces: The triangle indices.
        """

    @abstractmethod
    def _intersect(self, origins: NDArray, directions: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Casts rays that are already expressed in the local frame.

        :param origins: ``(N, 3)`` ray origins.
        :param directions: ``(N, 3)`` ray directions.
        :return: ``(hits, ray_indices)`` -- the ``(M, 3)`` hit positions and the ``(M,)``
            indices of the rays that produced them, first hit only, misses omitted.
        """

    def intersects_first(self, origins: NDArray, directions: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Casts rays and returns their first intersection with the mesh.

        :param origins: ``(N, 3)`` ray origins in world space.
        :param directions: ``(N, 3)`` ray directions.
        :return: ``(hits, ray_indices)`` in world space -- the ``(M, 3)`` hit positions and
            the ``(M,)`` indices of the rays that produced them. Misses are omitted, so
            ``M <= N`` and ``ray_indices`` is what re-aligns the result with the input.
        """
        if self._released:
            raise RuntimeError('This ray caster has been released')

        origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3) - self._origin
        directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)

        if len(origins) == 0:
            return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)

        hits, ray_indices = self._intersect(origins, directions)
        hits = np.asarray(hits, dtype=np.float64).reshape(-1, 3)
        return hits + self._origin, np.asarray(ray_indices, dtype=np.int64).reshape(-1)

    def release(self) -> None:
        """
        Releases the acceleration structure. Idempotent.
        """
        self._released = True
