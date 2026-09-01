"""The package's public surface.

``alfspy/__init__.py`` was empty in both merged projects, so there was no public API at all --
callers reached into ``alfspy.core.rendering`` and ``alfspy.render`` directly and nothing
recorded which of those names were meant to be stable.
"""

import importlib
import os
import subprocess
import sys

import pytest

import alfspy


def test_version_is_exposed():
    assert alfspy.__version__.startswith('3.')


@pytest.mark.parametrize('name', sorted(alfspy.__all__))
def test_every_exported_name_resolves(name):
    assert getattr(alfspy, name) is not None


def test_unknown_names_still_raise():
    with pytest.raises(AttributeError, match='has no attribute'):
        alfspy.definitely_not_exported


def test_dir_lists_the_public_api():
    assert set(dir(alfspy)) == set(alfspy.__all__)


def test_importing_alfspy_pulls_in_no_render_backend():
    """
    Backends are optional extras, so importing the package must not import any of them.
    Run in a subprocess because this one has already imported everything.
    """
    code = (
        'import sys, alfspy; '
        'loaded = [m for m in ("torch", "moderngl", "wgpu") if m in sys.modules]; '
        'print(",".join(loaded))'
    )
    # pytest's `pythonpath` ini option only affects this process, so hand the child the
    # same search path rather than relying on the package being installed.
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         check=True, env=env)
    assert out.stdout.strip() == '', (
        f'importing alfspy pulled in {out.stdout.strip()}; backends must stay lazy')


def test_lazy_names_come_from_their_documented_module():
    for name, module_name in alfspy._LAZY.items():
        module = importlib.import_module(module_name)
        assert getattr(alfspy, name) is getattr(module, name)
