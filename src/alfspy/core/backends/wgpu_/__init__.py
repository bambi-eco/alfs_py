"""The Vulkan render backend, via WebGPU.

wgpu-py wraps wgpu-native, which dispatches to Vulkan on Windows and Linux (Metal on macOS,
DX12 as a fallback). It was chosen over the raw Vulkan bindings deliberately: those are a
single-maintainer package last released in early 2024 and need roughly 1500 lines of
instance/device/descriptor boilerplate before the first triangle, whereas wgpu-py ships
prebuilt wheels, is actively released, and brings a validation layer.

Its practical value here is headless rendering. The ModernGL backend needs a GL driver and,
in Docker, Xvfb -- which the README documents as a source of buffer artifacts. This backend
needs neither.
"""

from .context import (
    BLEND,
    CULL_FACE,
    DEPTH_TEST,
    WgpuContext,
    create_context,
    is_available,
    reset_state,
)
from .data import RenderObject
from .framebuffer import img_from_fbo
from .renderer import Renderer
from .shot import CtxShot

__all__ = [
    'BLEND',
    'CULL_FACE',
    'DEPTH_TEST',
    'CtxShot',
    'RenderObject',
    'Renderer',
    'WgpuContext',
    'create_context',
    'img_from_fbo',
    'is_available',
    'owns_context',
    'reset_state',
]


def owns_context(ctx) -> bool:
    """
    :param ctx: Any render context.
    :return: Whether it belongs to this backend.
    """
    return isinstance(ctx, WgpuContext)
