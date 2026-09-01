"""GPU ray casting through NVIDIA Warp.

Optional; install with ``pip install "AlfsPy[warp]"``.

Whether this is worth selecting depends entirely on ray count. Measured on a 131k-triangle
DEM (RTX 500 Ada laptop, CUDA; timings include host<->device transfer):

===================  ==========  ==========  ===========
rays                 embree      Warp CPU    Warp CUDA
===================  ==========  ==========  ===========
80 (a frame's labels)  <1 ms       <1 ms       <1 ms
4k                      3 ms        2 ms       <1 ms
64k                    33 ms       28 ms        2 ms
4.2M (2048 squared)  1.810 s     1.678 s     0.085 s
===================  ==========  ==========  ===========

The crossover is around 1e4 rays: below it, launch and transfer overhead dominate and this
backend merely ties embree. Label projection casts tens of rays per frame, so ``embree``
remains the default and this is for bulk work.

Warp JIT-compiles its kernels on first use (about 0.7 s on CUDA, 2.5 s on CPU) and then
caches them to disk, so a fresh container pays that once unless the cache is pre-warmed.

Note that for *dense* un-projection there is a better answer than either backend: the
renderer already rasterises this mesh from the same camera, so a world-position render target
yields every hit as a by-product with no BVH at all. Reach for this backend when the rays do
not originate at the render camera.
"""

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .base import RayCaster

__all__ = ['WarpRayCaster', 'is_available']

_MAX_T = 1.0e6

_wp = None
_kernel = None


def _warp():
    """Imports and initialises Warp once, and builds the kernel on first use."""
    global _wp, _kernel
    if _wp is not None:
        return _wp

    import warp as wp

    wp.init()

    @wp.kernel
    def _raycast(mesh_id: wp.uint64,
                 origins: wp.array(dtype=wp.vec3),
                 directions: wp.array(dtype=wp.vec3),
                 out_pos: wp.array(dtype=wp.vec3),
                 out_hit: wp.array(dtype=wp.int32)):
        tid = wp.tid()
        query = wp.mesh_query_ray(mesh_id, origins[tid], directions[tid], _MAX_T)
        if query.result:
            out_pos[tid] = origins[tid] + directions[tid] * query.t
            out_hit[tid] = 1
        else:
            out_hit[tid] = 0

    _wp, _kernel = wp, _raycast
    return _wp


def is_available() -> bool:
    """
    :return: Whether Warp is installed and can initialise here.
    """
    try:
        _warp()
    except Exception:
        return False
    return True


class WarpRayCaster(RayCaster):
    """
    Ray casting against a ``warp.Mesh`` BVH.
    """

    name = 'warp'

    def __init__(self, vertices: NDArray, faces: NDArray, device: Optional[str] = None):
        """
        :param vertices: The mesh vertices as an ``(V, 3)`` array.
        :param faces: The triangle indices as an ``(F, 3)`` array.
        :param device: The Warp device, e.g. ``"cuda:0"`` or ``"cpu"`` (optional). Defaults to
            the first CUDA device when one is visible, otherwise the CPU.
        """
        wp = _warp()
        if device is None:
            device = 'cuda:0' if wp.get_cuda_device_count() > 0 else 'cpu'
        self.device = device
        super().__init__(vertices, faces)

    def _build(self, vertices: NDArray, faces: NDArray) -> None:
        wp = _warp()
        self._mesh = wp.Mesh(
            points=wp.array(np.ascontiguousarray(vertices, dtype=np.float32),
                            dtype=wp.vec3, device=self.device),
            indices=wp.array(np.ascontiguousarray(faces.reshape(-1), dtype=np.int32),
                             dtype=wp.int32, device=self.device),
        )

    def _intersect(self, origins: NDArray, directions: NDArray) -> Tuple[NDArray, NDArray]:
        wp = _warp()
        count = len(origins)

        d_origins = wp.array(np.ascontiguousarray(origins, dtype=np.float32),
                             dtype=wp.vec3, device=self.device)
        d_directions = wp.array(np.ascontiguousarray(directions, dtype=np.float32),
                                dtype=wp.vec3, device=self.device)
        d_positions = wp.zeros(count, dtype=wp.vec3, device=self.device)
        d_hit = wp.zeros(count, dtype=wp.int32, device=self.device)

        wp.launch(_kernel, dim=count,
                  inputs=[self._mesh.id, d_origins, d_directions, d_positions, d_hit],
                  device=self.device)
        wp.synchronize_device(self.device)

        hit = d_hit.numpy().astype(bool)
        positions = d_positions.numpy()
        ray_indices = np.flatnonzero(hit)
        return positions[hit], ray_indices

    def release(self) -> None:
        if not self._released:
            self._mesh = None
        super().release()
