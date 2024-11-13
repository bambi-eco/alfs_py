import json
import pathlib
from typing import Optional

import cv2
import numpy as np
from moderngl import Context
from numpy.typing import NDArray
from pyrr import Vector3, Quaternion

#noinspection PyPackageRequirements
from office365.runtime.client_request_exception import ClientRequestException

from alfspy.core.geo.transform import Transform
from alfspy.core.rendering.data import TextureData
from alfspy.core.rendering.shot import CtxShot
from alfspy.core.sharepoint.sharepoint_client import SharepointClient


class SharepointCtxShot(CtxShot):
    _spc: SharepointClient
    _tries: int
    _default: NDArray

    def __init__(self, spc: SharepointClient, ctx: Context, img_path: str, position: Vector3, rotation: Quaternion,
                 fovy: float = 60.0, aspect_ratio: float = 1, correction: Optional[Transform] = None,
                 lazy: bool = False, tries: int = 1, default: Optional[NDArray] = None):
        """
        Initializes a new ``CtxShot`` object.
        :param ctx: The ModernGL context the shot should be associated with.
        :param spc: The client to be used for sharepoint queries and requests. This context is required to
        already fulfill all requirements for file access.
        :param img_path: The sharepoint path pointing at the image associated with this shot.
        :param position: The position of the camera associated with the shot.
        :param rotation: The rotation of the camera associated with the shot.
        :param fovy: The field of view in y direction in degrees of the camera associated with the shot (defaults to 60).
        :param aspect_ratio: The aspect ratio of the view of the camera associated with the shot (defaults to 1).
        :param correction: Correction transform to be applied to the shot (optional).
        :param lazy: Whether the shot should be loaded lazily (defaults to ``False``). This also loads the image lazily
        from the drive when an image-file-path is given.
        :param tries: How many attempts should be made to download the image to account for exceptions (defaults to 1).
        Will always be at least one.
        :param default: The image to be used as a default in case no image has been downloaded in the given amount of tries (optional).
        If not specified creates a small image with all RGBA values set to zero.
        """
        super().__init__(ctx, img_path, position, rotation, fovy, aspect_ratio, correction, lazy)
        self._spc = spc
        self._tries = max(tries, 1)

        if default is None:
            default = np.zeros((2,2,4), dtype='f4')
        self._default = default

    @staticmethod
    def _load_image_from_sharepoint(img_path: str, scp: SharepointClient, tries: int, default: NDArray) -> NDArray:

        img_bytes = None
        for _ in range(tries):
            img_bytes = scp.get_file_bytes_by_path(img_path)
            if img_bytes is None:
                continue
            else:
                break

        if img_bytes is not None:
            data_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(data_arr, flags=cv2.IMREAD_COLOR)
            return SharepointCtxShot._cvt_img(img)
        else:
            return default


    def load_image(self):
        """
        When the shot was initialized using a file path, loads the associated image.
        """
        if self.tex_data is None and self._can_initialize:  # ensures img_file is set
            self.tex_data = TextureData(self._load_image_from_sharepoint(self._img_file, self._spc, self._tries, self._default))

    @staticmethod
    def from_sharepoint_json(json_path: str, spc: SharepointClient, ctx: Context, count: Optional[int] = None,
                             image_dir: Optional[str] = None, fovy: float = 60.0, correction: Optional[Transform] = None,
                             lazy: bool = False) -> list['SharepointCtxShot']:
        """
        Create sharepoint shots based on a j
        :param json_path:
        :param spc:
        :param ctx:
        :param count:
        :param image_dir:
        :param fovy:
        :param correction:
        :param lazy:
        :return:
        """
        if image_dir is None:
            # assume images are within the same directory as the json file
            image_dir = str(pathlib.Path(json_path).parent)

        if correction is None:
            correction = Transform()

        # org_path = pathlib.Path(json_path)
        # spc.rename_item_by_path(json_path, alt_path.name)

        json_bytes = spc.get_file_bytes_by_path(json_path)
        if json_bytes is None:
            raise ValueError(f'The server did not successfully retrieve the JSON file at {json_path}')
        data = json.loads(json_bytes)

        shot_params = CtxShot._process_json(data, ctx, count, image_dir, fovy, correction, lazy)

        # revert renaming
        # spc.rename_item_by_path(str(alt_path), org_path.name)
        return [SharepointCtxShot(spc, *shot_param) for shot_param in shot_params]
