"""One context factory, identical across every backend.

The point is that **only the engine argument changes what you get**. Before this, each
backend had its own signature -- ``backend=``/``standalone=`` for ModernGL,
``device=``/``dtype=`` for torch, ``power_preference=`` for Vulkan -- so switching engines
meant rewriting the call, and ``ProjectionScene`` carried an if/elif deciding which keyword
each backend would tolerate. These tests are what stops that divergence coming back.
"""

import inspect
import warnings

import pytest

from alfspy.core.backends import (
    available_engines,
    backend_for_context,
    create_context,
    engine_names,
    get_backend,
    make_context,
)

ENGINES = available_engines()


def test_make_context_and_create_context_are_the_same_function():
    """3.0.0 shipped the name `create_context`; `make_context` is the name going forward."""
    assert make_context is create_context


@pytest.mark.parametrize('name', engine_names())
def test_every_backend_declares_the_same_signature(name):
    """
    A backend that takes different parameters is not interchangeable, however well the
    registry hides it.
    """
    try:
        backend = get_backend(name)
    except ImportError:
        pytest.skip(f'{name} backend not installed')

    params = list(inspect.signature(backend.create_context).parameters.values())
    kinds = [(p.name, p.kind) for p in params]

    assert kinds[0][0] == 'device', f'{name}: first parameter should be `device`, got {kinds}'
    assert params[0].default is None, f'{name}: `device` should default to None'
    assert kinds[1][1] is inspect.Parameter.VAR_KEYWORD, (
        f'{name}: second parameter should be **options, got {kinds}')
    assert len(params) == 2, f'{name}: signature should be (device=None, **options), got {kinds}'


@pytest.mark.parametrize('engine', ENGINES)
def test_the_same_call_works_on_every_engine(engine):
    """The whole point: one call, any engine."""
    ctx = make_context(engine, device='cpu')
    try:
        assert backend_for_context(ctx) is get_backend(engine)
    finally:
        ctx.release()


@pytest.mark.parametrize('engine', ENGINES)
def test_options_meant_for_another_engine_are_ignored(engine):
    """
    Passing ModernGL's `backend` or torch's `dtype` to a backend that has no such notion must
    not raise -- otherwise a caller switching engines still has to rewrite its arguments.
    """
    ctx = make_context(engine, device='cpu', backend=None, standalone=True,
                       power_preference='low-power')
    try:
        assert backend_for_context(ctx) is get_backend(engine)
    finally:
        ctx.release()


@pytest.mark.parametrize('engine', ENGINES)
def test_device_is_accepted_everywhere(engine):
    """
    Every backend accepts `device`, whether or not it can honour it. ModernGL ignores it --
    OpenGL has no device selection -- but it must not be an error to pass.
    """
    for device in (None, 'cpu'):
        ctx = make_context(engine, device=device)
        try:
            assert ctx is not None
        finally:
            ctx.release()


def test_torch_actually_honours_the_device():
    """Accepting `device` uniformly is only useful if the backends that can, use it."""
    if 'torch' not in ENGINES:
        pytest.skip('torch backend not available')
    ctx = make_context('torch', device='cpu')
    try:
        assert str(ctx.device) == 'cpu'
    finally:
        ctx.release()


def test_deprecated_factories_still_work_and_say_so():
    from alfspy.render.render import make_mgl_context, make_torch_context

    if 'torch' in ENGINES:
        with pytest.warns(DeprecationWarning, match='make_context'):
            ctx = make_torch_context(device='cpu')
        try:
            assert backend_for_context(ctx) is get_backend('torch')
        finally:
            ctx.release()

    if 'moderngl' in ENGINES:
        with pytest.warns(DeprecationWarning, match='make_context'):
            ctx = make_mgl_context()
        try:
            assert backend_for_context(ctx) is get_backend('moderngl')
        finally:
            ctx.release()


def test_make_context_emits_no_warning():
    """The replacement must be silent, or the deprecation is just noise."""
    if not ENGINES:
        pytest.skip('no backend available')
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        ctx = make_context(ENGINES[0])
        ctx.release()
