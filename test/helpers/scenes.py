"""Synthetic geometry, textures and cameras with analytically known answers.

Nothing here loads a file. Every scene is small enough to reason about by hand, which is
what makes the rasteriser tests provable rather than golden-image regressions.
"""

import math
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pyrr import Quaternion, Vector3

from alfspy.core.rendering import Camera, MeshData, Resolution, TextureData

__all__ = [
    'flat_quad',
    'single_triangle',
    'two_quads_at_depths',
    'height_field',
    'checkerboard_rgba',
    'gradient_rgba',
    'solid_rgba',
    'ortho_camera_above',
    'perspective_camera_above',
    'fovy_covering',
]


# ───────────────────────────────── geometry ─────────────────────────────────


def flat_quad(half: float = 5.0, z: float = 0.0) -> MeshData:
    """
    A flat square in the XY plane, wound counter-clockwise as seen from ``+Z``.

    :param half: Half the side length, so the quad spans ``[-half, half]`` in X and Y.
    :param z: The height of the quad.
    :return: Mesh data for two triangles with full-range UVs.
    """
    vertices = np.array([
        [-half, -half, z],
        [half, -half, z],
        [half, half, z],
        [-half, half, z],
    ], dtype='f4')
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype='f4')
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    return MeshData(vertices=vertices, indices=indices, uvs=uvs)


def single_triangle(a: tuple = (-4.0, -4.0), b: tuple = (4.0, -3.0), c: tuple = (0.0, 4.0),
                    z: float = 0.0, flip_winding: bool = False) -> MeshData:
    """
    One triangle in the XY plane.

    :param a: First vertex as ``(x, y)``.
    :param b: Second vertex as ``(x, y)``.
    :param c: Third vertex as ``(x, y)``.
    :param z: The height of the triangle.
    :param flip_winding: Whether to reverse the winding, making it back-facing from ``+Z``.
    :return: Mesh data for a single triangle.
    """
    order = (a, c, b) if flip_winding else (a, b, c)
    vertices = np.array([[p[0], p[1], z] for p in order], dtype='f4')
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype='f4')
    indices = np.array([[0, 1, 2]], dtype=np.uint32)
    return MeshData(vertices=vertices, indices=indices, uvs=uvs)


def two_quads_at_depths(near_z: float = 2.0, far_z: float = 0.0, half: float = 5.0,
                        near_first: bool = True) -> MeshData:
    """
    Two overlapping coplanar-in-XY quads at different heights, for depth-test checks.

    The near quad is offset in UV space so the two are distinguishable after shading.

    :param near_z: Height of the quad closer to a camera looking down from ``+Z``.
    :param far_z: Height of the farther quad.
    :param half: Half the side length of both quads.
    :param near_first: Whether the near quad is emitted first. Flipping this must not change
        the depth-tested result.
    :return: Mesh data containing four triangles.
    """
    def quad(z, u0, u1):
        verts = np.array([
            [-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z],
        ], dtype='f4')
        uvs = np.array([[u0, 0.0], [u1, 0.0], [u1, 1.0], [u0, 1.0]], dtype='f4')
        return verts, uvs

    near_v, near_uv = quad(near_z, 0.0, 0.5)
    far_v, far_uv = quad(far_z, 0.5, 1.0)

    if near_first:
        vertices = np.concatenate((near_v, far_v))
        uvs = np.concatenate((near_uv, far_uv))
    else:
        vertices = np.concatenate((far_v, near_v))
        uvs = np.concatenate((far_uv, near_uv))

    indices = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.uint32)
    return MeshData(vertices=vertices, indices=indices, uvs=uvs)


def height_field(resolution: int = 16, half: float = 25.0, amplitude: float = 0.0,
                 seed: int = 0) -> MeshData:
    """
    A regular triangulated grid, optionally with smooth height variation.

    This is the shape of a real DEM: many small triangles sharing edges, which is exactly the
    configuration that exposes fill-rule errors.

    :param resolution: Number of quads per side.
    :param half: Half the side length in world units.
    :param amplitude: Peak height deviation. ``0`` yields an exactly flat mesh.
    :param seed: Seed for the height variation.
    :return: Mesh data with per-vertex UVs covering ``[0, 1]``.
    """
    n = resolution + 1
    xs = np.linspace(-half, half, n, dtype='f4')
    ys = np.linspace(-half, half, n, dtype='f4')
    grid_x, grid_y = np.meshgrid(xs, ys, indexing='xy')

    if amplitude:
        rng = np.random.default_rng(seed)
        phase = rng.uniform(0.0, 2.0 * math.pi, size=4)
        grid_z = amplitude * (
            np.sin(grid_x / half * math.pi + phase[0]) * np.cos(grid_y / half * math.pi + phase[1])
            + 0.5 * np.sin(2.0 * grid_x / half * math.pi + phase[2])
            * np.cos(2.0 * grid_y / half * math.pi + phase[3])
        )
    else:
        grid_z = np.zeros_like(grid_x)

    vertices = np.stack((grid_x, grid_y, grid_z), axis=-1).reshape(-1, 3).astype('f4')

    u = np.linspace(0.0, 1.0, n, dtype='f4')
    grid_u, grid_v = np.meshgrid(u, u, indexing='xy')
    uvs = np.stack((grid_u, grid_v), axis=-1).reshape(-1, 2).astype('f4')

    faces = []
    for row in range(resolution):
        for col in range(resolution):
            bl = row * n + col
            br = bl + 1
            tl = bl + n
            tr = tl + 1
            # Counter-clockwise seen from +Z.
            faces.append((bl, br, tr))
            faces.append((bl, tr, tl))

    indices = np.array(faces, dtype=np.uint32)
    return MeshData(vertices=vertices, indices=indices, uvs=uvs)


# ───────────────────────────────── textures ─────────────────────────────────


def checkerboard_rgba(tiles: int = 8, tile_size: int = 8,
                      colour_a: tuple = (255.0, 255.0, 255.0),
                      colour_b: tuple = (0.0, 0.0, 0.0)) -> NDArray:
    """
    :param tiles: Number of tiles per side.
    :param tile_size: Side length of one tile in pixels.
    :param colour_a: RGB of the first tile colour.
    :param colour_b: RGB of the second tile colour.
    :return: An ``(H, W, 4)`` float32 RGBA checkerboard with opaque alpha in ``[0, 255]``.
    """
    side = tiles * tile_size
    img = np.zeros((side, side, 4), dtype='f4')
    for row in range(tiles):
        for col in range(tiles):
            colour = colour_a if (row + col) % 2 == 0 else colour_b
            img[row * tile_size:(row + 1) * tile_size,
                col * tile_size:(col + 1) * tile_size, :3] = colour
    img[..., 3] = 255.0
    return img


def gradient_rgba(width: int = 32, height: int = 32) -> NDArray:
    """
    A texture whose red channel encodes U and green channel encodes V.

    Sampling it makes the recovered UV coordinates directly readable from the output, which
    is how the perspective-correctness tests get an analytic reference.

    :param width: Texture width.
    :param height: Texture height.
    :return: An ``(H, W, 4)`` float32 RGBA image in ``[0, 255]``.
    """
    us = (np.arange(width, dtype='f4') + 0.5) / width
    vs = (np.arange(height, dtype='f4') + 0.5) / height
    grid_u, grid_v = np.meshgrid(us, vs, indexing='xy')

    img = np.zeros((height, width, 4), dtype='f4')
    img[..., 0] = grid_u * 255.0
    img[..., 1] = grid_v * 255.0
    img[..., 3] = 255.0
    return img


def solid_rgba(colour: tuple = (255.0, 0.0, 0.0, 255.0), width: int = 16, height: int = 16) -> NDArray:
    """
    :param colour: The RGBA colour to fill with, in ``[0, 255]``.
    :param width: Texture width.
    :param height: Texture height.
    :return: An ``(H, W, 4)`` float32 RGBA image.
    """
    img = np.zeros((height, width, 4), dtype='f4')
    img[...] = np.array(colour, dtype='f4')
    return img


# ────────────────────────────────── cameras ─────────────────────────────────


def fovy_covering(half_extent: float, distance: float) -> float:
    """
    The vertical FOV in degrees that makes a perspective camera at ``distance`` see exactly
    ``[-half_extent, half_extent]``.

    :param half_extent: Half the world-space extent to cover.
    :param distance: Distance from the camera to the plane.
    :return: The field of view in degrees.
    """
    return float(np.rad2deg(2.0 * math.atan(half_extent / distance)))


def ortho_camera_above(size: tuple = (10.0, 10.0), height: float = 10.0,
                       near: float = 0.1, far: float = 100.0,
                       position: Optional[Vector3] = None) -> Camera:
    """
    An orthographic camera above the origin looking straight down.

    The default rotation leaves the camera's forward vector at ``(0, 0, -1)``.

    :param size: The orthographic frustum size as ``(width, height)`` in world units.
    :param height: The camera's Z position when ``position`` is not given.
    :param near: Near clipping distance.
    :param far: Far clipping distance.
    :param position: An explicit camera position (optional).
    :return: A configured :class:`~alfspy.core.rendering.camera.Camera`.
    """
    if position is None:
        position = Vector3([0.0, 0.0, height])
    return Camera(orthogonal=True, orthogonal_size=size, position=position, near=near, far=far)


def perspective_camera_above(fovy: float = 60.0, aspect_ratio: float = 1.0, height: float = 10.0,
                             near: float = 0.1, far: float = 100.0,
                             rotation: Optional[Quaternion] = None,
                             position: Optional[Vector3] = None) -> Camera:
    """
    A perspective camera above the origin looking straight down.

    :param fovy: Vertical field of view in degrees.
    :param aspect_ratio: The aspect ratio.
    :param height: The camera's Z position when ``position`` is not given.
    :param near: Near clipping distance.
    :param far: Far clipping distance.
    :param rotation: An explicit rotation (optional).
    :param position: An explicit camera position (optional).
    :return: A configured :class:`~alfspy.core.rendering.camera.Camera`.
    """
    if position is None:
        position = Vector3([0.0, 0.0, height])
    return Camera(fovy=fovy, aspect_ratio=aspect_ratio, orthogonal=False,
                  position=position, rotation=rotation, near=near, far=far)
