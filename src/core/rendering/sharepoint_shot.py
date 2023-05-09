from typing import Optional

from moderngl import Context
from numpy.typing import NDArray

#noinspection PyPackageRequirements
from office365.sharepoint.client_context import ClientContext
from office365.sharepoint.files.file import File
from pyrr import Vector3, Quaternion

from src.core.geo.transform import Transform
from src.core.rendering.data import TextureData
from src.core.rendering.shot import CtxShot


class SharepointCtxShot(CtxShot):
    _cctx: ClientContext

    def __init__(self, ctx: Context, cctx: ClientContext, img_path: str, position: Vector3, rotation: Quaternion,
                 fovy: float = 60.0, aspect_ratio: float = 1, correction: Optional[Transform] = None,
                 lazy: bool = False):
        """
        Initializes a new ``CtxShot`` object
        :param ctx: The ModernGL context the shot should be associated with
        :param cctx: The client context to be used for sharepoint queries and requests. This context is required to
        already fulfill all requirements for file access.
        :param img_path: The sharepoint path pointing at the image associated with this shot
        :param position: The position of the camera associated with the shot
        :param rotation: The rotation of the camera associated with the shot
        :param fovy: The field of view in y direction in degrees of the camera associated with the shot (defaults to 60)
        :param aspect_ratio: The aspect ratio of the view of the camera associated with the shot (defaults to 1)
        :param correction: Correction transform to be applied to the shot (optional)
        :param lazy: Whether the shot should be loaded lazily (defaults to ``False``). This also loads the image lazily
        from the drive when an image-file-path is given.
        """
        super().__init__(ctx, img_path, position, rotation, fovy, aspect_ratio, correction, lazy)
        self._cctx = cctx

    @staticmethod
    def _load_image_from_sharepoint(img_path: str, cctx: ClientContext) -> NDArray:
        response = File.open_binary(cctx, img_path)
        if response.ok:
            data = response.content
        else:
            raise ValueError('')

    def load_image(self):
        """
        When the shot was initialized using a file path, loads the associated image
        """
        if self.tex_data is None and self._can_initialize:  # ensures img_file is set
            self.tex_data = TextureData(self._load_image_from_sharepoint(str(self._img_file), self._cctx))

    @staticmethod
    def from_sharepoint_json(json_path: str, cctx: ClientContext):
        pass
