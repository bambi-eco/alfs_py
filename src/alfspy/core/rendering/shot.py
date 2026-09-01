"""Backend-dispatching ``CtxShot``.

Like :class:`~alfspy.core.rendering.renderer.Renderer`, this is a facade over the concrete
per-backend implementations. A shot is a source image plus the camera it was taken with, and
it holds that image as a texture on the render device -- which is why it is backend-specific
and why it takes a context.

(Fusing pose data with device residency is the reason a shot cannot currently be handed to a
second backend without being rebuilt. Splitting ``Shot`` into pure data plus a per-backend
texture cache is a later step; this facade is what makes both implementations reachable
through one name in the meantime.)
"""

from abc import ABCMeta
from typing import Optional, Union

from numpy.typing import NDArray
from pyrr import Quaternion, Vector3

from alfspy.core.backends.registry import backend_for_context
from alfspy.core.geo import Transform
from alfspy.core.util.image import to_rgba_f4

__all__ = ['CtxShot']


class CtxShot(metaclass=ABCMeta):
    """
    One captured image together with the camera that took it.

    Instantiating this returns a backend-specific shot, chosen by the context passed in.
    """

    def __new__(cls, ctx, img: Union[str, NDArray], position: Vector3, rotation: Quaternion,
                fovy: float = 60.0, aspect_ratio: float = 1, correction: Optional[Transform] = None,
                lazy: bool = False, normalise: bool = True):
        """
        Initializes a new shot for the backend that owns ``ctx``.

        :param ctx: The context the shot should be associated with. This selects the backend.
        :param img: Either the path to an image file as a string, or an already loaded image
            as an RGB numpy array.
        :param position: The position of the camera associated with the shot.
        :param rotation: The rotation of the camera associated with the shot.
        :param fovy: The field of view in y direction in degrees (defaults to 60).
        :param aspect_ratio: The aspect ratio of the view (defaults to 1).
        :param correction: Correction transform to be applied to the shot (optional).
        :param lazy: Whether the shot should be loaded lazily (defaults to ``False``).
        :param normalise: Whether values above 1 should be rescaled by 1/255 on upload
            (defaults to ``True``). Set ``False`` for an N-channel feature field.
        :return: A backend-specific shot.
        """
        impl = backend_for_context(ctx).CtxShot
        CtxShot.register(impl)
        return impl(ctx, img, position, rotation, fovy, aspect_ratio, correction, lazy,
                    normalise)

    @staticmethod
    def _cvt_img(img: NDArray) -> NDArray:
        """
        Converts an image to the renderers' RGBA ``f4`` format.

        Kept here because several pipelines reach for ``CtxShot._cvt_img`` to prepare a mask,
        which needs no context and therefore no backend. The implementation is
        :func:`alfspy.core.util.image.to_rgba_f4`; prefer calling that directly.

        :param img: The image to convert.
        :return: The converted image.
        """
        return to_rgba_f4(img)

    @staticmethod
    def from_json(file: str, ctx, count: Optional[int] = None, image_dir: Optional[str] = None,
                  fovy: float = 60.0, correction: Optional[Transform] = None,
                  lazy: bool = False) -> list:
        """
        Creates shots from a JSON file, for the backend that owns ``ctx``.

        :param file: The path of the JSON file to process.
        :param ctx: The context to attach the shots to. This selects the backend.
        :param count: The maximum amount of shots to be created (optional).
        :param image_dir: The directory of the images referenced in the JSON file (defaults to
            the JSON file's directory).
        :param fovy: The default fovy to use when a JSON entry does not provide one.
        :param correction: The general correction to be applied to all shots (optional).
        :param lazy: Whether the created shots should be lazy loaded (defaults to ``False``).
        :return: A list of backend-specific shots.
        """
        impl = backend_for_context(ctx).CtxShot
        CtxShot.register(impl)
        return impl.from_json(file, ctx, count, image_dir, fovy, correction, lazy)
