import json
import pathlib
from typing import Union, Final, Optional

import cv2
import numpy as np
from moderngl import Context, Texture
from numpy.typing import NDArray
from pyrr import Vector3, Quaternion, Matrix44

from src.core.camera import Camera
from src.core.data import TextureData
from src.core.defs import PATH_SEP
from src.core.utils import get_first_valid


class Shot:
    """
    Represents the combination of a camera and a picture taken by that camera
    """
    camera: Final[Camera]
    # TODO: Add model matrix for corrections
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
        if isinstance(img, np.ndarray):
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
        """
        :return: A matrix representing the projection of this shots camera
        """
        return self.camera.get_proj(dtype='f4')

    def get_view(self) -> Matrix44:
        """
        :return: A matrix representing the view of this shots camera
        """
        return self.camera.get_view(dtype='f4')

    def get_mat(self) -> Matrix44:
        """
        :return: A matrix representing the combination of projection and view of this shots camera
        """
        return self.camera.get_mat(dtype='f4')


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

    def tex_use(self) -> None:
        """
        Binds the texture of this object to a texture unit
        """
        self.tex.use()

    @staticmethod
    def from_json(file: str, ctx: Context, count: Optional[int] = None, image_dir: Optional[str] = None,
                  fovy: float = 60.0) -> list['CtxShot']:
        """
        Creates context shots from a JSON file
        :param file: The path of the JSON file to process
        :param ctx: The context to attach the context shots to
        :param count: The maximum amount of shots to be created (optional)
        :param image_dir: The directory of the images referenced in the JSON file (defaults to the JSON files directory)
        :param fovy: The default fovy value to be used when a JSON entry does not provide one
        :return: A list of ``CtxShot`` objects
        """
        shots = []

        if image_dir is None:
            image_dir = str(pathlib.Path(file).parent.absolute())

        with open(file, 'r') as f:
            data = json.load(f)

        if count is not None:
            images_dat = data.get('images', [])
            act_count = min(len(images_dat), count)
            images = images_dat[0:act_count]
        else:
            images = data.get('images', [])

        for image in images:
            img_file = get_first_valid(image, ['imagefile', 'file', 'image'])
            position = get_first_valid(image, ['location', 'pos', 'loc'])
            rotation = get_first_valid(image, ['rotation', 'rot', 'quaternion'])
            fov = get_first_valid(image, ['fovy', 'fov', 'fieldofview'])

            if img_file is None or position is None or rotation is None:
                raise ValueError('The given JSON file does not contain valid data')

            rot_len = len(rotation)
            for _ in range(4 - rot_len):
                rotation.append(0.0)

            if fov is None:
                fov = fovy

            img_file = f'{image_dir}{PATH_SEP}{img_file}'

            shots.append(CtxShot(ctx, img_file, Vector3(position), Quaternion(rotation), fov))
        return shots

