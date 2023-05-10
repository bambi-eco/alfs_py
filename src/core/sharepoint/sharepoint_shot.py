import json
import pathlib
from typing import Optional

import cv2
import numpy as np
from moderngl import Context
from numpy.typing import NDArray
from pyrr import Vector3, Quaternion

from src.core.geo.transform import Transform
from src.core.rendering.data import TextureData
from src.core.rendering.shot import CtxShot
from src.core.sharepoint.sharepoint_client import SharepointClient
from src.core.util.image import bytes_to_img


class SharepointCtxShot(CtxShot):
    _spc: SharepointClient

    def __init__(self, spc: SharepointClient, ctx: Context, img_path: str, position: Vector3, rotation: Quaternion,
                 fovy: float = 60.0, aspect_ratio: float = 1, correction: Optional[Transform] = None,
                 lazy: bool = False):
        """
        Initializes a new ``CtxShot`` object
        :param ctx: The ModernGL context the shot should be associated with
        :param spc: The client to be used for sharepoint queries and requests. This context is required to
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
        self._spc = spc

    @staticmethod
    def _load_image_from_sharepoint(img_path: str, scp: SharepointClient) -> NDArray:
        img_bytes = scp.get_bytes(img_path)
        data_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(data_arr, flags=cv2.IMREAD_COLOR)
        return SharepointCtxShot._cvt_img(img)


    def load_image(self):
        """
        When the shot was initialized using a file path, loads the associated image
        """
        if self.tex_data is None and self._can_initialize:  # ensures img_file is set
            self.tex_data = TextureData(self._load_image_from_sharepoint(self._img_file, self._spc))

    @staticmethod
    def from_sharepoint_json(json_path: str, spc: SharepointClient, ctx: Context, count: Optional[int] = None,
                             image_dir: Optional[str] = None, fovy: float = 60.0, correction: Optional[Transform] = None,
                             lazy: bool = False) -> list['SharepointCtxShot']:
        shots = []

        if image_dir is None:
            # assume images are within the same directory as the json file
            image_dir = str(pathlib.Path(json_path).parent)

        if correction is None:
            correction = Transform()

        # rename JSON file to avoid epic office365 package bug
        org_path = pathlib.Path(json_path)
        alt_path = org_path.with_suffix('.jsont')

        spc.rename(json_path, alt_path.name)

        json_bytes = spc.get_bytes(str(alt_path))
        data = json.loads(json_bytes)

        shot_params = CtxShot._process_json(data, ctx, count, image_dir, fovy, correction, lazy)

        # revert renaming
        spc.rename(str(alt_path), org_path.name)
        return [SharepointCtxShot(spc, *shot_param) for shot_param in shot_params]
