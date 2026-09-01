"""Airborne Light Field Sampling.

Renders novel aerial views by projecting geo-referenced drone imagery onto a digital
elevation model and integrating the overlapping captures -- a reconstruction-free
alternative to photogrammetry or a neural radiance field. It also produces orthographic
projections and carries 2D labels through either transform.

The renderer has three interchangeable backends. Which one you get is decided by the render
context, so nothing else in the API changes with it::

    from alfspy import create_context, render_integral

    render_integral(dem, poses, mask, engine='moderngl')   # OpenGL; needs a GL driver
    render_integral(dem, poses, mask, engine='torch')      # tensor rasteriser; headless
    render_integral(dem, poses, mask, engine='vulkan')     # WebGPU -> Vulkan; headless

``$ALFS_ENGINE`` sets the default. Backends are optional extras -- ``AlfsPy[moderngl]``,
``AlfsPy[torch]``, ``AlfsPy[vulkan]`` -- and are imported lazily, so importing this package
pulls in none of them.

A light field is not limited to three channels: :func:`~alfspy.render.field.render_field_integral`
integrates fields of any width, which is what makes an *embedded* light field -- one whose
pixels carry learned descriptors rather than colours -- a first-class result.

Ray-mesh intersection, used to project labels onto the DEM, is pluggable the same way:
``embree`` by default, ``warp`` for GPU when the ray count justifies it.
"""

from alfspy.core.backends import (
    available_engines,
    backend_for_context,
    create_context,
    engine_names,
    resolve_engine,
)
from alfspy.core.raycast import available_raycasters, create_raycaster
from alfspy.core.rendering import (
    Camera,
    CtxShot,
    MeshData,
    Renderer,
    RenderResultMode,
    Resolution,
    TextureData,
)
from alfspy.core.rendering.data import IntegralResult

__version__ = '3.0.0'

__all__ = [
    '__version__',
    # backend selection
    'create_context',
    'available_engines',
    'engine_names',
    'resolve_engine',
    'backend_for_context',
    # ray casting
    'create_raycaster',
    'available_raycasters',
    # core types
    'Camera',
    'CtxShot',
    'IntegralResult',
    'MeshData',
    'Renderer',
    'RenderResultMode',
    'Resolution',
    'TextureData',
    # high-level rendering (lazily resolved; see __getattr__)
    'render_integral',
    'project_shots',
    'animate_focus',
    'animate_shutter',
    'ProjectionScene',
    'ProjectionSettings',
    'IntegralSettings',
    'Label',
    'ProjectedLabel',
    'parse_mot_labels',
    'render_field_integral',
    'FieldShot',
    'ChannelSpec',
]

# `alfspy.render` pulls in OpenCV, trimesh and the GLTF reader, which is a slow import for
# someone who only wants the data types. These names resolve on first access instead.
_LAZY = {
    'render_integral': 'alfspy.render.render',
    'project_shots': 'alfspy.render.render',
    'animate_focus': 'alfspy.render.render',
    'animate_shutter': 'alfspy.render.render',
    'ProjectionScene': 'alfspy.render.projection',
    'ProjectionSettings': 'alfspy.render.projection',
    'Label': 'alfspy.render.projection',
    'ProjectedLabel': 'alfspy.render.projection',
    'parse_mot_labels': 'alfspy.render.projection',
    'IntegralSettings': 'alfspy.render.data',
    'render_field_integral': 'alfspy.render.field',
    'FieldShot': 'alfspy.render.field',
    'ChannelSpec': 'alfspy.render.field',
}


def __getattr__(name):
    """
    Resolves the high-level rendering API on first access.

    :param name: The attribute being looked up.
    :return: The requested object.
    :raises AttributeError: If the name is not part of the public API.
    """
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # cache, so this runs once per name
    return value


def __dir__():
    return sorted(__all__)
