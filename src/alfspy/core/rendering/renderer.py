"""Backend-dispatching ``Renderer``.

``Renderer`` used to be a single concrete class. It is now a facade: the concrete
implementations live in ``alfspy.core.backends.<name>``, and which one you get is decided by
the context you pass.

That works because the context was always the first thing a ``Renderer`` was given --
``Renderer(resolution, ctx, camera, mesh, texture)`` -- so the context is already the engine
handle and every existing call site keeps working unchanged::

    ctx = create_context(engine='torch')      # or 'moderngl', or $ALFS_ENGINE
    renderer = Renderer(resolution, ctx, camera, mesh)   # same signature as before
"""

from abc import ABCMeta
from typing import Optional

from alfspy.core.backends.registry import backend_for_context
from alfspy.core.rendering.camera import Camera
from alfspy.core.rendering.data import MeshData, Resolution, TextureData

__all__ = ['Renderer']


class Renderer(metaclass=ABCMeta):
    """
    Renders light-field projections of a mesh.

    Instantiating this returns a backend-specific renderer rather than an instance of this
    class; the backend is whichever one owns ``ctx``. The concrete classes are registered as
    virtual subclasses on first use, so ``isinstance(r, Renderer)`` holds.
    """

    def __new__(cls, resolution: Resolution, ctx, camera: Camera, mesh: MeshData,
                texture: Optional[TextureData] = None):
        """
        Initializes a new ``Renderer`` for the backend that owns ``ctx``.

        :param resolution: The resolution of the images to render.
        :param ctx: The render context to be used, from
            :func:`alfspy.core.backends.create_context`. This selects the backend.
        :param camera: The camera to be used by the renderer.
        :param mesh: The mesh data of the main mesh the renderer should work with. It
            represents the canvas and or background of all done projections or renders.
        :param texture: The texture data of the main mesh (optional). If no texture is given a
            single colored texture will be generated.
        :return: A backend-specific renderer.
        """
        impl = backend_for_context(ctx).Renderer
        Renderer.register(impl)
        return impl(resolution, ctx, camera, mesh, texture)
