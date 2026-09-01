"""The public modules must import without any render backend installed.

No backend is a base dependency - that is the point of the engine extras - so a module that
imports one at module level quietly makes it mandatory. ``alfspy/render/render.py`` did
exactly that with ``alfspy.core.torchgl``, and because ``alfspy/render/__init__.py`` imports
it, `AlfsPy[moderngl]` alone could not import ``alfspy.render`` at all. The failure was a bare
``No module named 'torch'`` from three frames down, which names neither the cause nor the fix.

Checked in a subprocess with torch blocked rather than by importing here: torch is installed
in any full dev environment, so an in-process import test would pass whether or not the bug
is back.
"""

import os
import subprocess
import sys
import textwrap

import pytest

# pytest's `pythonpath` setting only touches this process, so a subprocess would not find
# alfspy in a checkout that has not been installed. Hand it the paths we were given.
_ENV = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))

# Kept in one place: the modules a caller reaches for that must not drag in a backend.
PUBLIC_MODULES = (
    'alfspy',
    'alfspy.render',
    'alfspy.render.render',
    'alfspy.render.data',
    'alfspy.render.projection',
    'alfspy.core.backends',
    'alfspy.core.raycast',
    'alfspy.core.convert.convert',
    'alfspy.core.rendering',
)

_PROBE = textwrap.dedent('''
    import sys

    class Blocker:
        """Refuses %(pkg)s the way an environment without that extra would."""

        def find_spec(self, name, path=None, target=None):
            if name == %(pkg)r or name.startswith(%(pkg)r + "."):
                raise ImportError("No module named %(pkg)r")
            return None

    sys.meta_path.insert(0, Blocker())
    for name in [n for n in sys.modules
                 if n == %(pkg)r or n.startswith(%(pkg)r + ".")]:
        del sys.modules[name]

    failed = []
    for module in %(modules)r:
        try:
            __import__(module)
        except Exception as exc:
            failed.append("%%s: %%s: %%s" %% (module, type(exc).__name__, exc))
    if failed:
        print("\\n".join(failed))
        sys.exit(1)
''')


def _import_without(package):
    """Import every public module in a subprocess where *package* is unavailable."""
    return subprocess.run(
        [sys.executable, '-c',
         _PROBE % {'pkg': package, 'modules': PUBLIC_MODULES}],
        capture_output=True, text=True, env=_ENV)


@pytest.mark.parametrize('package', ['torch', 'moderngl'])
def test_the_public_modules_import_without_a_backend(package):
    result = _import_without(package)
    assert result.returncode == 0, (
        f'importing the public modules without {package} failed:\n'
        f'{result.stdout}{result.stderr}')


def test_a_backend_is_still_reachable_without_the_other(tmp_path):
    """The point is not merely that the imports survive - the engine that *is* installed has
    to remain usable, or the deferral would have hidden a real breakage."""
    script = textwrap.dedent('''
        import sys

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name == "torch" or name.startswith("torch."):
                    raise ImportError("No module named 'torch'")
                return None

        sys.meta_path.insert(0, Blocker())
        for name in [n for n in sys.modules
                     if n == "torch" or n.startswith("torch.")]:
            del sys.modules[name]

        from alfspy.core.backends import available_engines
        engines = available_engines()
        assert "torch" not in engines, engines
        print(",".join(engines))
    ''')
    result = subprocess.run([sys.executable, '-c', script],
                            capture_output=True, text=True, env=_ENV)
    assert result.returncode == 0, result.stdout + result.stderr
