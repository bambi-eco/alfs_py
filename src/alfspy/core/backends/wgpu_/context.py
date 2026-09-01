"""WebGPU device and render state.

The Vulkan backend goes through wgpu-py, which speaks WebGPU and dispatches to Vulkan on
Windows and Linux (Metal on macOS, DX12 as a fallback). That buys prebuilt wheels, a
maintained binding and a validation layer, where the raw Vulkan bindings for Python are a
single-maintainer package last released in 2024 and about 1500 lines of boilerplate before
the first triangle.

``WgpuContext`` presents the same small GL-shaped state surface the other two backends do
(``enable``/``disable``/``cull_face``/``blend_func``), so the renderer, ``reset_state`` and
the backend registry all work unchanged. WebGPU has no global state -- everything is baked
into a pipeline object -- so this class records the state and the renderer selects the
matching pre-built pipeline.
"""

from typing import Final, Optional, Set

__all__ = [
    'BLEND',
    'CULL_FACE',
    'DEPTH_TEST',
    'ADDITIVE_BLENDING',
    'DEFAULT_BLENDING',
    'WgpuContext',
    'create_context',
    'is_available',
]

# Same values the torch backend uses, so callers can share constants.
DEPTH_TEST: Final[int] = 0x001
CULL_FACE: Final[int] = 0x002
BLEND: Final[int] = 0x004

ADDITIVE_BLENDING: Final[str] = 'additive'
DEFAULT_BLENDING: Final[str] = 'default'

# Requested when the adapter offers them. WebGPU forbids both by default:
#   float32-blendable  -- blending into an rgba32float target, which *is* the ALFS integral
#   float32-filterable -- bilinear sampling of a float32 texture, which is how a shot is read
# Without them the backend has to fall back to 16-bit float targets; see `Renderer`.
_OPTIONAL_FEATURES: Final[tuple] = ('float32-blendable', 'float32-filterable')


class WgpuContext:
    """
    A WebGPU device plus the render state the pipeline needs.

    :cvar device: The underlying ``wgpu.GPUDevice``.
    :cvar adapter: The adapter it came from.
    """

    def __init__(self, power_preference: str = 'high-performance',
                 force_fallback_adapter: bool = False):
        """
        :param power_preference: ``"high-performance"`` or ``"low-power"``.
        :param force_fallback_adapter: Whether to force the software adapter.
        """
        import wgpu

        self.adapter = wgpu.gpu.request_adapter_sync(
            power_preference=power_preference,
            force_fallback_adapter=force_fallback_adapter,
        )

        available: Set[str] = set(self.adapter.features)
        self.features: Set[str] = {f for f in _OPTIONAL_FEATURES if f in available}
        self.device = self.adapter.request_device_sync(
            required_features=sorted(self.features))

        self._enabled: int = 0
        self.cull_face: str = 'back'
        self.front_face: str = 'ccw'
        self.blend_func: str = DEFAULT_BLENDING
        self._released = False

    # region GL-shaped state

    def enable(self, flag: int) -> None:
        """
        :param flag: One of :data:`DEPTH_TEST`, :data:`CULL_FACE`, :data:`BLEND`.
        """
        self._enabled |= flag

    def disable(self, flag: int) -> None:
        """
        :param flag: One of :data:`DEPTH_TEST`, :data:`CULL_FACE`, :data:`BLEND`.
        """
        self._enabled &= ~flag

    def is_enabled(self, flag: int) -> bool:
        """
        :param flag: The flag to query.
        :return: Whether it is enabled.
        """
        return bool(self._enabled & flag)

    @property
    def depth_test(self) -> bool:
        """:return: Whether depth testing is enabled."""
        return self.is_enabled(DEPTH_TEST)

    @property
    def blend(self) -> bool:
        """:return: Whether blending is enabled."""
        return self.is_enabled(BLEND)

    # endregion

    @property
    def float32_blendable(self) -> bool:
        """
        :return: Whether this device can blend into a 32-bit float render target. When it
            cannot, the renderer accumulates into ``rgba16float`` instead, which is
            representable for the counts involved but carries roughly three decimal digits.
        """
        return 'float32-blendable' in self.features

    @property
    def float32_filterable(self) -> bool:
        """
        :return: Whether this device can sample a 32-bit float texture bilinearly. When it
            cannot, shot textures are stored as ``rgba16float`` so filtering still works.
        """
        return 'float32-filterable' in self.features

    def release(self) -> None:
        """
        Releases the device. Idempotent.
        """
        self._released = True

    def __repr__(self) -> str:
        info = self.adapter.info
        name = info.get('description') or info.get('device') or 'unknown'
        return (f'<WgpuContext {name!r} backend={info.get("backend_type")} '
                f'features={sorted(self.features) or "none"}>')


def create_context(device: Optional[str] = None, **options) -> 'WgpuContext':
    """
    Creates a WebGPU context in the pipeline's standard state.

    Every backend takes this same signature, so ``create_context(engine=...)`` behaves
    identically whichever engine is chosen and only the engine changes the result.

    :param device: ``"cpu"`` selects the software fallback adapter; anything else, including
        ``None``, asks for the highest-performance adapter. A specific index such as
        ``"cuda:1"`` is not honoured -- WebGPU exposes adapters, not CUDA devices -- but is
        accepted so the same call works across engines.
    :param options: Backend-specific extras. This one understands ``power_preference``
        (``"high-performance"`` or ``"low-power"``) and ``force_fallback_adapter``. Anything
        else is ignored, so options meant for another engine do not raise.
    :return: A configured context.
    """
    wants_cpu = isinstance(device, str) and device.lower().startswith('cpu')

    power_preference = options.get(
        'power_preference', 'low-power' if wants_cpu else 'high-performance')
    force_fallback_adapter = options.get('force_fallback_adapter', wants_cpu)

    ctx = WgpuContext(power_preference=power_preference,
                      force_fallback_adapter=force_fallback_adapter)
    ctx.enable(DEPTH_TEST)
    ctx.enable(CULL_FACE)
    ctx.cull_face = 'back'
    return ctx


def reset_state(ctx: WgpuContext) -> None:
    """
    Returns a context to the pipeline's standard state.

    :param ctx: The context to reset.
    """
    ctx.enable(DEPTH_TEST)
    ctx.enable(CULL_FACE)
    ctx.cull_face = 'back'
    ctx.disable(BLEND)


def is_available() -> bool:
    """
    :return: Whether a WebGPU adapter can be obtained here. Importing ``wgpu`` succeeds
        without a usable GPU, so this has to request an adapter rather than catch an
        ``ImportError``.
    """
    try:
        import wgpu

        return wgpu.gpu.request_adapter_sync(power_preference='high-performance') is not None
    except Exception:
        return False
