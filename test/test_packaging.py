"""The declared extras.

Extras are the install-time half of the engine and ray-caster choice, and nothing else checks
them: a typo in `pyproject.toml` produces a package that imports fine and simply cannot
render. These assert the shape rather than resolving anything, so they need no network.
"""

import os
import re

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYPROJECT = os.path.join(_ROOT, 'pyproject.toml')

pytestmark = pytest.mark.skipif(
    not os.path.exists(_PYPROJECT), reason='running against an installed package, not a checkout')


def _load():
    try:
        import tomllib
    except ImportError:  # Python 3.9/3.10
        tomllib = pytest.importorskip('tomli', reason='needs tomllib (3.11+) or tomli')
    with open(_PYPROJECT, 'rb') as handle:
        return tomllib.load(handle)


@pytest.fixture(scope='module')
def project():
    return _load()['project']


@pytest.fixture(scope='module')
def extras(project):
    return project['optional-dependencies']


def test_version_matches_the_package(project):
    import alfspy

    assert project['version'] == alfspy.__version__


@pytest.mark.parametrize('name', ['moderngl', 'torch', 'vulkan', 'embree', 'warp',
                                  'embedding', 'umap', 'all', 'dev', 'test'])
def test_expected_extras_exist(name, extras):
    assert name in extras, f'extra {name!r} is missing; declared: {sorted(extras)}'


def test_extra_names_match_the_runtime_values(extras):
    """
    `AlfsPy[torch]` goes with `engine='torch'`, `AlfsPy[warp]` with `raycaster='warp'`. If the
    two vocabularies drift apart the docs stop being followable.
    """
    from alfspy.core.backends import engine_names
    from alfspy.core.raycast import raycaster_names

    for engine in engine_names():
        assert engine in extras, f'no extra named after the {engine!r} engine'
    for caster in raycaster_names():
        assert caster in extras, f'no extra named after the {caster!r} ray caster'


def test_no_render_backend_is_a_base_dependency(project):
    """
    Backends are opt-in; a bare install must not drag in torch or a GL binding. The lazy-import
    test in test_public_api.py checks the runtime half of this.
    """
    base = ' '.join(project['dependencies']).lower()
    for forbidden in ('torch', 'moderngl', 'wgpu', 'warp-lang'):
        assert forbidden not in base, f'{forbidden} must be an extra, not a base dependency'


def test_embree_arrives_with_the_base_install(project):
    """
    The default ray caster must work out of the box. `trimesh[easy]` provides embreex and
    rtree, which is why the `embree` extra is only a pin -- if the base pin ever loses
    `[easy]`, embree silently degrades to a pure-Python intersector that returns hits in a
    different order.
    """
    base = ' '.join(project['dependencies'])
    assert re.search(r'trimesh\[[^\]]*easy[^\]]*\]', base), (
        'trimesh must be installed with the `easy` extra, which is what provides embreex')


def test_all_covers_every_feature_extra(extras):
    """`all` must not quietly omit a feature; that is the whole promise of the name."""
    referenced = set()
    for spec in extras['all']:
        match = re.fullmatch(r'AlfsPy\[([a-z-]+)\]', spec.strip())
        assert match, f'`all` should reference other extras, got {spec!r}'
        referenced.add(match.group(1))

    meta = {'all', 'dev', 'test', 'accel', 'raycast-gpu'}
    missing = set(extras) - meta - referenced
    assert not missing, f'`all` omits: {sorted(missing)}'


def test_dev_is_all_plus_test(extras):
    assert set(s.strip() for s in extras['dev']) == {'AlfsPy[all]', 'AlfsPy[test]'}


@pytest.mark.parametrize('old,new', [('accel', 'embree'), ('raycast-gpu', 'warp')])
def test_deprecated_extra_aliases_still_point_somewhere(old, new, extras):
    """3.0/3.1 install commands must keep working."""
    assert [s.strip() for s in extras[old]] == [f'AlfsPy[{new}]']


def test_vulkan_extra_is_gated_on_python_version(extras):
    """
    wgpu needs 3.11 while everything else runs on 3.9+. Without the marker, `AlfsPy[all]`
    would be uninstallable on 3.9 rather than simply omitting the Vulkan backend.
    """
    assert any('python_version' in spec for spec in extras['vulkan']), (
        'the vulkan extra needs a python_version marker so `all` stays installable on 3.9')
