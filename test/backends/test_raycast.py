"""The ray-caster backends must agree, and embree must actually be embree.

The second point is not pedantry. trimesh selects its intersector at import and falls back to
a pure-Python one without saying so, and that fallback does not merely run slower: it returns
hits in a different index order. Five coordinate-conversion round-trip tests failed under
trimesh 3.21.7 for exactly that reason and pass under 4.x. A test that asserts which
intersector is live is the only thing that stops the pin drifting back.
"""

import numpy as np
import pytest

from alfspy.core.convert.convert import cast_ray, pixel_to_world_coord
from alfspy.core.raycast import (
    DEFAULT_RAYCASTER,
    available_raycasters,
    create_raycaster,
    get_raycaster,
    raycaster_names,
    resolve_raycaster,
)
from alfspy.core.rendering.data import MeshData
from test.helpers.scenes import height_field, perspective_camera_above

BACKENDS = available_raycasters()


@pytest.fixture(scope='module')
def dem():
    """A non-flat DEM: many small shared-edge triangles, as a real one has."""
    return height_field(resolution=24, half=40.0, amplitude=4.0, seed=3)


def _downward_rays(count, rng):
    origins = np.zeros((count, 3))
    origins[:, 0] = rng.uniform(-30, 30, count)
    origins[:, 1] = rng.uniform(-30, 30, count)
    origins[:, 2] = 200.0
    directions = np.tile(np.array([0.0, 0.0, -1.0]), (count, 1))
    return origins, directions


def test_embree_is_actually_embree(dem):
    """Guards the silent pure-Python fallback."""
    caster = create_raycaster(dem, backend='embree')
    assert caster.accelerated, (
        'trimesh fell back to its pure-Python ray intersector. That returns hits in a '
        'different index order, not just more slowly. Check that trimesh >= 4 and embreex '
        'are installed -- trimesh 3.x looks for the obsolete `pyembree` module.')


@pytest.mark.parametrize('backend', BACKENDS)
def test_hits_land_on_the_mesh(backend, dem):
    """Every reported hit must actually lie on the surface it claims to have hit."""
    rng = np.random.default_rng(0)
    origins, directions = _downward_rays(64, rng)

    caster = create_raycaster(dem, backend=backend)
    hits, ray_indices = caster.intersects_first(origins, directions)

    assert len(hits) == 64, f'{backend}: expected every downward ray to hit the DEM'
    # A vertical ray keeps its x/y, so the hit must sit directly under the origin.
    np.testing.assert_allclose(hits[:, :2], origins[ray_indices][:, :2], atol=1e-4)
    # And its height must be within the DEM's own range.
    zs = dem.vertices[:, 2]
    assert hits[:, 2].min() >= zs.min() - 1e-3
    assert hits[:, 2].max() <= zs.max() + 1e-3


@pytest.mark.skipif(len(BACKENDS) < 2, reason='need two ray casters to compare')
def test_backends_agree(dem):
    rng = np.random.default_rng(1)
    origins, directions = _downward_rays(256, rng)

    results = {}
    for backend in BACKENDS:
        caster = create_raycaster(dem, backend=backend)
        hits, ray_indices = caster.intersects_first(origins, directions)
        order = np.argsort(ray_indices)
        results[backend] = (ray_indices[order], hits[order])

    reference = BACKENDS[0]
    ref_idx, ref_hits = results[reference]
    for other in BACKENDS[1:]:
        idx, hits = results[other]
        np.testing.assert_array_equal(
            idx, ref_idx, err_msg=f'{other} and {reference} disagree about which rays hit')
        # float32 traversal in both, on a mesh spanning 80 world units.
        np.testing.assert_allclose(
            hits, ref_hits, atol=1e-3,
            err_msg=f'{other} and {reference} disagree about where the rays hit')


@pytest.mark.parametrize('backend', BACKENDS)
def test_misses_are_reported_and_index_alignment_survives(backend, dem):
    """A miss must drop out of the hit list without shifting the surviving indices."""
    origins = np.array([
        [0.0, 0.0, 200.0],       # hits
        [1e5, 1e5, 200.0],       # misses, far outside the DEM
        [10.0, -5.0, 200.0],     # hits
    ])
    directions = np.tile(np.array([0.0, 0.0, -1.0]), (3, 1))

    caster = create_raycaster(dem, backend=backend)
    hits, ray_indices = caster.intersects_first(origins, directions)

    assert list(ray_indices) == [0, 2], f'{backend}: miss handling shifted the ray indices'
    assert len(hits) == 2


@pytest.mark.parametrize('backend', BACKENDS)
def test_cast_ray_accepts_a_prebuilt_caster(backend, dem):
    """Reusing one caster must give the same answer as letting cast_ray build one."""
    origins = np.array([[0.0, 0.0, 200.0], [12.0, -7.0, 200.0]])
    directions = np.tile(np.array([0.0, 0.0, -1.0]), (2, 1))

    caster = create_raycaster(dem, backend=backend)
    via_object = cast_ray(origins, directions, caster, include_misses=True)
    via_mesh = cast_ray(origins, directions, dem, include_misses=True, raycaster=backend)

    for a, b in zip(via_object, via_mesh):
        np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                                   atol=1e-6)


def test_geo_referenced_coordinates_keep_their_precision():
    """
    Both backends traverse in float32, where a UTM northing near 5e6 resolves to about 0.5 m.
    The base class casts in a mesh-local frame so that cliff does not reach the caller.
    """
    offset = np.array([500000.0, 5000000.0, 0.0])
    local = height_field(resolution=8, half=20.0, amplitude=0.0, seed=0)
    shifted = MeshData(vertices=local.vertices.astype(np.float64) + offset,
                       indices=local.indices, uvs=local.uvs)

    origins = np.array([[offset[0] + 3.25, offset[1] - 7.125, 100.0]])
    directions = np.array([[0.0, 0.0, -1.0]])

    caster = create_raycaster(shifted, backend='embree')
    hits, ray_indices = caster.intersects_first(origins, directions)

    assert len(hits) == 1
    # The DEM is flat at z=0 here, so the hit is fully determined; a float32 world-space
    # traversal would smear x/y by decimetres.
    np.testing.assert_allclose(hits[0], [offset[0] + 3.25, offset[1] - 7.125, 0.0], atol=1e-3)


@pytest.mark.parametrize('backend', BACKENDS)
def test_pixel_to_world_works_through_every_backend(backend, dem):
    """The actual consumer: un-projecting label corners onto the DEM."""
    camera = perspective_camera_above(fovy=60.0, height=80.0)
    caster = create_raycaster(dem, backend=backend)

    world = pixel_to_world_coord(
        [128, 200], [128, 60], 256, 256, caster, camera, include_misses=False)

    assert len(world) == 2
    for point in world:
        assert point is not None


def test_default_and_env_selection(monkeypatch):
    assert DEFAULT_RAYCASTER == 'embree'
    monkeypatch.setenv('ALFS_RAYCASTER', 'warp')
    assert resolve_raycaster() == 'warp'
    assert resolve_raycaster('embree') == 'embree'
    monkeypatch.delenv('ALFS_RAYCASTER')
    assert resolve_raycaster() == 'embree'


def test_unknown_raycaster_is_rejected():
    with pytest.raises(ValueError, match='Unknown ray caster'):
        get_raycaster('optix')


def test_missing_backend_names_its_extra():
    """
    The extra is named after the caster, so the error message is directly actionable:
    `raycaster='warp'` failing tells you to install `AlfsPy[warp]`.
    """
    for name in raycaster_names():
        try:
            get_raycaster(name)
        except ImportError as exc:
            assert f'AlfsPy[{name}]' in str(exc)
