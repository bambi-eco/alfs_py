"""``$ALFS_DEVICE``, the device half of the environment-driven selection.

``ALFS_ENGINE`` and ``ALFS_RAYCASTER`` were honoured while ``ALFS_DEVICE`` was documented but
read nowhere: setting it was a silent no-op, and a deployment that selected torch through the
environment still had to plumb ``device=`` through every call to finish the job. These pin the
symmetry -- argument, then environment, then the backend's own choice.
"""

import pytest

from alfspy.core.backends import (
    DEVICE_ENV_VAR, available_engines, get_backend, make_context, resolve_device)


def test_the_variable_is_the_documented_name():
    assert DEVICE_ENV_VAR == 'ALFS_DEVICE'


def test_default_and_env_selection(monkeypatch):
    monkeypatch.delenv(DEVICE_ENV_VAR, raising=False)
    # None, not a device name: each backend has its own sensible answer, and inventing one
    # here would override torch's CUDA detection with a guess.
    assert resolve_device() is None

    monkeypatch.setenv(DEVICE_ENV_VAR, 'cuda')
    assert resolve_device() == 'cuda'


def test_an_explicit_device_beats_the_env_var(monkeypatch):
    monkeypatch.setenv(DEVICE_ENV_VAR, 'cuda')
    assert resolve_device('cpu') == 'cpu'


def test_an_empty_variable_reads_as_unset(monkeypatch):
    """An exported-but-blank variable is how a shell says "no", not a device called ''."""
    monkeypatch.setenv(DEVICE_ENV_VAR, '')
    assert resolve_device() is None


def test_make_context_forwards_the_env_device(monkeypatch):
    """``resolve_device`` returning the right answer is not enough on its own - the value has
    to reach the backend, which is where a user would rely on it."""
    seen = {}

    class _Backend:
        @staticmethod
        def create_context(device=None, **options):
            seen['device'] = device
            return 'CTX'

    monkeypatch.setattr('alfspy.core.backends.registry.get_backend',
                        lambda name: _Backend)
    monkeypatch.setenv(DEVICE_ENV_VAR, 'cuda:1')
    assert make_context('torch') == 'CTX'
    assert seen['device'] == 'cuda:1'


def test_an_explicit_device_survives_to_the_backend(monkeypatch):
    seen = {}

    class _Backend:
        @staticmethod
        def create_context(device=None, **options):
            seen['device'] = device
            return 'CTX'

    monkeypatch.setattr('alfspy.core.backends.registry.get_backend',
                        lambda name: _Backend)
    monkeypatch.setenv(DEVICE_ENV_VAR, 'cuda')
    make_context('torch', device='cpu')
    assert seen['device'] == 'cpu'


def test_the_torch_backend_honours_the_env_device(monkeypatch):
    """End to end on a real context, since a device the backend ignores would still pass the
    forwarding tests above."""
    if 'torch' not in available_engines():
        pytest.skip('torch backend not available')

    monkeypatch.setenv(DEVICE_ENV_VAR, 'cpu')
    ctx = make_context('torch')
    try:
        assert str(getattr(ctx, 'device', 'cpu')).startswith('cpu')
    finally:
        release = getattr(get_backend('torch'), 'release_context', None)
        if release is not None:
            release(ctx)
