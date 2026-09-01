"""Shared pytest fixtures and marker handling."""

import os
import sys

import pytest
import torch

# Ensure `src` and the repo root are importable regardless of how pytest was invoked.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (_ROOT, os.path.join(_ROOT, 'src')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from alfspy.core.torchgl import create_context  # noqa: E402
from test.helpers.dataset import build_flight  # noqa: E402


def pytest_collection_modifyitems(config, items):
    """
    Skips device- and reference-dependent tests instead of failing them.
    """
    no_cuda = pytest.mark.skip(reason='no CUDA device available')
    no_reference = pytest.mark.skip(
        reason='set ALFS_REFERENCE_PATH to a ModernGL alfs_py checkout to run parity tests'
    )

    has_cuda = torch.cuda.is_available()
    has_reference = bool(os.getenv('ALFS_REFERENCE_PATH'))

    for item in items:
        if 'cuda' in item.keywords and not has_cuda:
            item.add_marker(no_cuda)
        if 'parity' in item.keywords and not has_reference:
            item.add_marker(no_reference)


@pytest.fixture
def ctx():
    """
    A CPU render context with the pipeline's standard state.

    Pinned to the CPU so results are reproducible in CI; CUDA is exercised explicitly by the
    tests marked ``cuda``.
    """
    context = create_context(device='cpu')
    yield context
    context.release()


@pytest.fixture
def ctx_f64():
    """
    A double-precision CPU context, for tests that need to separate algorithmic error from
    float32 rounding.
    """
    context = create_context(device='cpu', dtype=torch.float64)
    yield context
    context.release()


@pytest.fixture(scope='session')
def flight(tmp_path_factory):
    """
    A complete synthetic flight generated once per session.

    Session scope because generating and encoding the frames dominates the runtime of the
    integration tests, and nothing mutates the files.
    """
    root = tmp_path_factory.mktemp('flight')
    return build_flight(str(root))


@pytest.fixture(scope='session')
def small_flight(tmp_path_factory):
    """
    A three-frame flight with tiny frames, for tests that only need the plumbing to run.
    """
    root = tmp_path_factory.mktemp('small_flight')
    return build_flight(str(root), frame_count=3, frame_size=(32, 32), dem_resolution=6)
