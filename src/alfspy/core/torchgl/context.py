"""Rendering context replacement.

``TorchContext`` occupies the same slot in the API that ``moderngl.Context`` did: it is
threaded through :class:`~alfspy.core.rendering.shot.CtxShot`,
:class:`~alfspy.core.rendering.renderer.Renderer` and the high-level pipelines, and it owns
the device/dtype choice so call sites never have to mention either.

What it deliberately does **not** carry is a bound render target. Draw calls take their
target explicitly. That removes the class of failure described in
``docs/MIGRATION_REVIEW.md`` §3, where a context-level clear could hit a different surface
than the one being drawn to.
"""

from typing import Any, Final, Optional, Union

import torch
from numpy.typing import NDArray

from alfspy.core.torchgl.framebuffer import TorchFramebuffer
from alfspy.core.torchgl.raster import DEFAULT_SAMPLE_BUDGET
from alfspy.core.torchgl.texture import TorchTexture

__all__ = [
    'TorchContext',
    'create_context',
    'DEPTH_TEST',
    'CULL_FACE',
    'BLEND',
    'ADDITIVE_BLENDING',
    'DEFAULT_BLENDING',
]

# Flag constants mirroring the moderngl names the original code used.
DEPTH_TEST: Final[int] = 0x001
CULL_FACE: Final[int] = 0x002
BLEND: Final[int] = 0x004

ADDITIVE_BLENDING: Final[str] = 'additive'
DEFAULT_BLENDING: Final[str] = 'default'

_DTYPES: Final[dict] = {
    'f2': torch.float16, 'f4': torch.float32, 'f8': torch.float64,
    'float16': torch.float16, 'float32': torch.float32, 'float64': torch.float64,
    'half': torch.float16, 'float': torch.float32, 'double': torch.float64,
}


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _resolve_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    resolved = _DTYPES.get(str(dtype))
    if resolved is None:
        raise ValueError(f'Unknown dtype "{dtype}"; expected one of {sorted(_DTYPES)}')
    return resolved


class TorchContext:
    """
    Owns the device, the numeric precision and the GL-shaped render state.
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = None,
                 dtype: Union[str, torch.dtype] = torch.float32,
                 sample_budget: int = DEFAULT_SAMPLE_BUDGET,
                 reject_behind_camera: bool = False,
                 depth_test_in_integral: bool = False) -> None:
        """
        Initialises a new ``TorchContext``.

        :param device: The torch device to render on (optional). Defaults to CUDA when
            available, otherwise CPU.
        :param dtype: The floating point precision for rasterisation and accumulation
            (defaults to ``torch.float32``, matching the GL ``f4`` framebuffer).
        :param sample_budget: Maximum candidate samples the rasteriser materialises at once.
            Lower it to cap peak memory, raise it for throughput.
        :param reject_behind_camera: Whether the shot program rejects fragments that fall
            behind the shot camera (defaults to ``False``). The GL shader intended to do this
            but its guard was dead code; see ``docs/MIGRATION_REVIEW.md`` §2.2.
        :param depth_test_in_integral: Whether each shot is depth-resolved before being
            accumulated (defaults to ``False``, matching the GL version's
            ``disable(DEPTH_TEST)``). Enabling it stops overhanging geometry double-counting.
        """
        self.device: Final[torch.device] = _resolve_device(device)
        self.dtype: Final[torch.dtype] = _resolve_dtype(dtype)
        self.sample_budget = int(sample_budget)
        self.reject_behind_camera = bool(reject_behind_camera)
        self.depth_test_in_integral = bool(depth_test_in_integral)

        self._flags: int = 0
        self.cull_face: str = 'back'
        self.front_face: str = 'ccw'
        self.blend_func: str = DEFAULT_BLENDING
        self._released = False

    # region GL-shaped state

    def enable(self, flag: int) -> None:
        """
        Enables a render state flag.

        :param flag: One of :data:`DEPTH_TEST`, :data:`CULL_FACE`, :data:`BLEND`.
        """
        self._flags |= int(flag)

    def disable(self, flag: int) -> None:
        """
        Disables a render state flag.

        :param flag: One of :data:`DEPTH_TEST`, :data:`CULL_FACE`, :data:`BLEND`.
        """
        self._flags &= ~int(flag)

    def is_enabled(self, flag: int) -> bool:
        """
        :param flag: The flag to query.
        :return: Whether the given flag is currently enabled.
        """
        return bool(self._flags & int(flag))

    @property
    def depth_test(self) -> bool:
        """
        :return: Whether depth testing is enabled.
        """
        return self.is_enabled(DEPTH_TEST)

    @property
    def culling(self) -> Optional[str]:
        """
        :return: The active cull mode, or ``None`` when culling is disabled.
        """
        return self.cull_face if self.is_enabled(CULL_FACE) else None

    @property
    def blending(self) -> bool:
        """
        :return: Whether blending is enabled.
        """
        return self.is_enabled(BLEND)

    # endregion

    # region resource factories

    def texture(self, size: Optional[tuple[int, int]] = None, components: Optional[int] = None,
                data: Optional[Union[NDArray, torch.Tensor, bytes]] = None,
                dtype: Any = None, **kwargs) -> TorchTexture:
        """
        Creates a texture on this context's device.

        The signature tolerates the positional ``(size, components, data)`` form used by
        ``moderngl.Context.texture`` so that ported call sites keep working, but the torch
        backend takes its shape from the array itself.

        :param size: The texture size as ``(width, height)`` (accepted, unused).
        :param components: The component count (accepted, unused).
        :param data: The image data as an array or tensor.
        :param dtype: Accepted for compatibility; the context's dtype is used.
        :param kwargs: Forwarded to :class:`~alfspy.core.torchgl.texture.TorchTexture`.
        :return: A new :class:`TorchTexture`.
        """
        self._ensure_active()
        if data is None:
            raise ValueError('TorchContext.texture requires image data')
        if isinstance(data, bytes):
            raise TypeError(
                'TorchContext.texture does not accept raw bytes. Pass the image array '
                'directly, or use TextureData.to_tensor().'
            )
        return TorchTexture(data, device=self.device, dtype=self.dtype, **kwargs)

    def framebuffer(self, size: tuple[int, int], components: int = 4) -> TorchFramebuffer:
        """
        Creates a render target on this context's device.

        :param size: The size as ``(width, height)``.
        :param components: The number of colour components (defaults to ``4``).
        :return: A new :class:`TorchFramebuffer`.
        """
        self._ensure_active()
        return TorchFramebuffer(size, device=self.device, components=components, dtype=self.dtype)

    def simple_framebuffer(self, size: tuple[int, int], components: int = 4,
                           dtype: Any = None) -> TorchFramebuffer:
        """
        Alias of :meth:`framebuffer` matching ``moderngl.Context.simple_framebuffer``.

        :param size: The size as ``(width, height)``.
        :param components: The number of colour components (defaults to ``4``).
        :param dtype: Accepted for compatibility; the context's dtype is used.
        :return: A new :class:`TorchFramebuffer`.
        """
        return self.framebuffer(size, components)

    # endregion

    def _ensure_active(self) -> None:
        if self._released:
            raise RuntimeError('TorchContext has been released')

    @property
    def released(self) -> bool:
        """
        :return: Whether this context has been released.
        """
        return self._released

    def release(self) -> None:
        """
        Releases the context.

        There is no driver resource to hand back, so this drops the CUDA caching allocator's
        free blocks (when on CUDA) and marks the context unusable.
        """
        if self._released:
            return
        self._released = True
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        state = []
        if self.depth_test:
            state.append('depth')
        if self.is_enabled(CULL_FACE):
            state.append(f'cull:{self.cull_face}')
        if self.blending:
            state.append(f'blend:{self.blend_func}')
        return f'<TorchContext {self.device} {self.dtype} [{" ".join(state) or "-"}]>'


def create_context(device: Optional[Union[str, torch.device]] = None,
                   dtype: Union[str, torch.dtype] = torch.float32,
                   **kwargs) -> TorchContext:
    """
    Creates a context with the render state the ALFS pipelines expect.

    Mirrors what ``make_mgl_context`` set up in the ModernGL version: depth testing on,
    back-face culling on.

    :param device: The torch device to render on (optional).
    :param dtype: The floating point precision (defaults to ``torch.float32``).
    :param kwargs: Forwarded to :class:`TorchContext`.
    :return: A configured :class:`TorchContext`.
    """
    ctx = TorchContext(device=device, dtype=dtype, **kwargs)
    ctx.enable(DEPTH_TEST)
    ctx.enable(CULL_FACE)
    ctx.cull_face = 'back'
    return ctx
