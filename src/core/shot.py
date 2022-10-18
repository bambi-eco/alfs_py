from typing import Union, Final

import cv2
import numpy as np
from moderngl import Context, Texture
from numpy.typing import NDArray
from pyrr import Vector3, Quaternion, Matrix44

from src.core.camera import Camera
from src.core.data import TextureData


class Shot:
    """
    Represents the combination of a camera and a picture taken by that camera
    """
    camera: Final[Camera]
    _tex_data: TextureData

    def __init__(self, img: Union[str, NDArray], position: Vector3, rotation: Quaternion, fovy: float = 60.0,
                 aspect_ratio: float = 1):
        """
        Initializes a new ``Shot`` object
        :param img: Either the path to an image file as a string, or an already loaded image as an RGB numpy array
        :param position: The position of the camera associated with the shot
        :param rotation: The rotation of the camera associated with the shot
        :param fovy: The field of view in y direction in degrees of the camera associated with the shot (defaults to 60)
        :param aspect_ratio: The aspect ratio of the view of the camera associated with the shot (defaults to 1)
        """
        self._released = False
        self.camera = Camera(fovy, aspect_ratio, position=position, rotation=rotation)
        if isinstance(img, NDArray):
            self._tex_data = TextureData(img.copy())
        else:
            self._tex_data = TextureData(self._load_image(str(img)))

    @property
    def img(self) -> NDArray:
        """
        :return: A copy of the picture associated with the shot
        """
        return TextureData.texture.copy()

    @staticmethod
    def _load_image(texture_filename) -> NDArray:
        img = cv2.imread(texture_filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB, opencv uses BGR
        img = np.flip(img, 0).copy(order="C")  # flip image vertically
        return img

    def get_proj(self) -> Matrix44:
        return self.camera.get_proj()

    def get_view(self) -> Matrix44:
        return self.camera.get_view()

    def get_mat(self) -> Matrix44:
        return self.camera.get_mat()


class CtxShot(Shot):
    _released: bool
    tex: Texture

    def __init__(self, ctx: Context, img: Union[str, NDArray], position: Vector3, rotation: Quaternion,
                 fovy: float = 60.0, aspect_ratio: float = 1):
        """
        Initializes a new ``CtxShot`` object
        :param ctx: The context the shot should be associated with
        :param img: Either the path to an image file as a string, or an already loaded image as an RGB numpy array
        :param position: The position of the camera associated with the shot
        :param rotation: The rotation of the camera associated with the shot
        :param fovy: The field of view in y direction in degrees of the camera associated with the shot (defaults to 60)
        :param aspect_ratio: The aspect ratio of the view of the camera associated with the shot (defaults to 1)
        """
        super().__init__(img, position, rotation, fovy, aspect_ratio)
        self._released = False
        self.tex = ctx.texture(*self._tex_data.tex_gen_input(), dtype='f4')

    def release(self) -> None:
        """
        Releases all objects associated with the given context
        """
        if not self._released:
            self.tex.release()
            self._released = True

    def use(self) -> None:
        self.tex.use()

