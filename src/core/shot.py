import copy
import json
import pathlib
from typing import Union, Final, Optional

import cv2
import numpy as np
from moderngl import Context, Texture
from numpy import deg2rad
from numpy.typing import NDArray
from pyrr import Vector3, Quaternion, Matrix44

from src.core.camera import Camera
from src.core.data import TextureData
from src.core.defs import PATH_SEP
from src.core.geo.transform import Transform
from src.core.util.basic import get_first_valid


class CtxShot:
    camera: Final[Camera]
    correction: Transform
    lazy: Final[bool]

    tex_data: Optional[TextureData]
    tex: Optional[Texture]

    _ctx: Final[Context]
    _img_file: Optional[str] = None
    _released: bool

    def __init__(self, ctx: Context, img: Union[str, NDArray], position: Vector3, rotation: Quaternion,
                 fovy: float = 60.0, aspect_ratio: float = 1, correction: Optional[Transform] = None,
                 lazy: bool = False):
        """
        Initializes a new ``CtxShot`` object
        :param ctx: The context the shot should be associated with
        :param img: Either the path to an image file as a string, or an already loaded image as an RGB numpy array
        :param position: The position of the camera associated with the shot
        :param rotation: The rotation of the camera associated with the shot
        :param fovy: The field of view in y direction in degrees of the camera associated with the shot (defaults to 60)
        :param aspect_ratio: The aspect ratio of the view of the camera associated with the shot (defaults to 1)
        :param correction: Correction transform to be applied to the shot (optional)
        :param lazy: Whether the shot should be loaded lazily (defaults to ``False``). This also loads the image lazily
        from the drive when an image-file-path is given.
        """
        self._released = False

        self.camera = Camera(fovy, aspect_ratio, position=position, rotation=rotation)
        if correction is None:
            self.correction = Transform(dtype='f4')
        else:
            self.correction = copy.deepcopy(correction)

        self._ctx = ctx
        if isinstance(img, np.ndarray):
            self.tex_data = TextureData(img.copy())
        else:
            self._img_file = img
        self.lazy = lazy
        self.tex_data = None
        self.tex = None
        if not lazy:
            self.load_image()
            self._init_texture()

    @staticmethod
    def _load_image_from_path(img_path: str) -> NDArray:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        channel_count = 1 if len(img.shape) == 2 else img.shape[2]

        # convert to and guarantee RGBA
        if channel_count == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
        elif channel_count == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        img = img.astype('f4')
        return img

    @property
    def _can_initialize(self):
        """
        :return: Whether the shot can currently be initialized
        """
        return not self._released or self._img_file is not None

    def load_image(self):
        """
        When the shot was initialized using a file path, loads the associated image
        """
        if self.tex_data is None and self._can_initialize:  # ensures img_file is set
            self.tex_data = TextureData(self._load_image_from_path(str(self._img_file)))

    def _init_texture(self):
        if self.tex is None and self._can_initialize:
            self.tex = self._ctx.texture(*self.tex_data.tex_gen_input(), dtype='f4')

    @property
    def img(self) -> Optional[NDArray]:
        """
        :return: If the shot was initialized using a path and that path was not yet loaded ``None``;
        Otherwise a copy of the associated image
        """
        if self.tex_data is None:
            return None
        else:
            return self.tex_data.texture.copy()

    def release(self) -> None:
        """
        Releases all objects associated with the given context
        """
        if not self._released:
            if self.tex_data is not None:
                del self.tex_data
                self.tex_data = None

            if self.tex is not None:
                self.tex.release()
                del self.tex
                self.tex = None

            self._released = True

    def tex_use(self, location: int = 0) -> None:
        """
        Binds the texture of this object to a texture unit
        """
        if self._released and not self._can_initialize:
            raise RuntimeError('Shot cannot be used as it was initialized using image data and has already been released')
        if self.tex_data is None:
            self.load_image()
        if self.tex is None:
            self._init_texture()
        self.tex.use(location)

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
        return self.correction.mat(dtype='f4')

    def get_mat(self) -> Matrix44:
        """
        :return: A matrix representing the combination of projection, view, and correction of this shot
        """
        return self.camera.get_mat(dtype='f4') * self.get_correction()

    @staticmethod
    def from_json(file: str, ctx: Context, count: Optional[int] = None, image_dir: Optional[str] = None,
                  fovy: float = 60.0, correction: Optional[Transform] = None, lazy: bool = False) -> list['CtxShot']:
        """
        Creates context shots from a JSON file
        :param file: The path of the JSON file to process
        :param ctx: The context to attach the context shots to
        :param count: The maximum amount of shots to be created (optional)
        :param image_dir: The directory of the images referenced in the JSON file (defaults to the JSON files directory)
        :param fovy: The default fovy value to be used when a JSON entry does not provide one
        :param correction: The general correction to be applied to all shots (optional)
        :param lazy: Whether the created shots should be lazy loaded (defaults to ``False``)
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
            if rot_len == 3:
                rotation = Quaternion.from_eulers([deg2rad(val) for val in rotation])
            elif rot_len == 4:
                rotation = Quaternion(rotation)
            else:
                raise ValueError(f'Invalid rotation format of length {rot_len}: {rotation}')

            if fov is None:
                fov = fovy

            img_file = f'{image_dir}{PATH_SEP}{img_file}'

            shots.append(CtxShot(ctx, img_file, Vector3(position), rotation, fov, correction=correction, lazy=lazy))
        return shots
