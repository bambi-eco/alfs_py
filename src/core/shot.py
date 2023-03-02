import copy
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
from src.core.geo.transform import Transform
from src.core.util.basic import get_first_valid


class Shot:
    """
    Represents the combination of a camera and a picture taken by that camera
    """
    camera: Final[Camera]
    correction: Transform
    _tex_data: TextureData

    def __init__(self, img: Union[str, NDArray], position: Vector3, rotation: Quaternion, fovy: float = 60.0,
                 aspect_ratio: float = 1, correction: Optional[Transform] = None):
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
        if correction is None:
            self.correction = Transform(dtype='f4')
        else:
            self.correction = copy.deepcopy(correction)

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
        img = cv2.imread(texture_filename, cv2.IMREAD_UNCHANGED)
        channel_count = 1 if len(img.shape) == 2 else img.shape[2]

        # convert to and guarantee RGBA
        if channel_count == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
        elif channel_count == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        img = cv2.flip(img, 1)  # flip image horizontally
        img = img.astype('f4')
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

    def get_correction(self) -> Matrix44:
        """
        :return: A matrix representing the correction to be applied to this shot
        """
        return self.correction.mat

    def get_mat(self) -> Matrix44:
        """
        :return: A matrix representing the combination of projection, view, and correction of this shot
        """
        return self.camera.get_mat(dtype='f4') * self.get_correction()


class CtxShot(Shot):
    _released: bool
    tex: Texture

    def __init__(self, ctx: Context, img: Union[str, NDArray], position: Vector3, rotation: Quaternion,
                 fovy: float = 60.0, aspect_ratio: float = 1, correction: Optional[Transform] = None):
        """
        Initializes a new ``CtxShot`` object
        :param ctx: The context the shot should be associated with
        :param img: Either the path to an image file as a string, or an already loaded image as an RGB numpy array
        :param position: The position of the camera associated with the shot
        :param rotation: The rotation of the camera associated with the shot
        :param fovy: The field of view in y direction in degrees of the camera associated with the shot (defaults to 60)
        :param aspect_ratio: The aspect ratio of the view of the camera associated with the shot (defaults to 1)
        """
        super().__init__(img, position, rotation, fovy, aspect_ratio, correction)
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
                  fovy: float = 60.0, correction: Optional[Transform] = None) -> list['CtxShot']:
        """
        Creates context shots from a JSON file
        :param file: The path of the JSON file to process
        :param ctx: The context to attach the context shots to
        :param count: The maximum amount of shots to be created (optional)
        :param image_dir: The directory of the images referenced in the JSON file (defaults to the JSON files directory)
        :param fovy: The default fovy value to be used when a JSON entry does not provide one
        :param correction: The general correction to be applied to all shots (optional)
        :return: A list of ``CtxShot`` objects
        """
        shots = []

        if image_dir is None:
            image_dir = str(pathlib.Path(file).parent.absolute())

        if correction is None:
            correction = Transform()

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

            shots.append(CtxShot(ctx, img_file, Vector3(position), Quaternion(rotation), fov, correction=correction))
        return shots

