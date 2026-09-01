import json
import pathlib
from typing import Union, Final, Optional

import cv2
from numpy import deg2rad
from numpy.typing import NDArray
from pyrr import Vector3, Quaternion, Matrix44

from alfspy.core.geo import Transform
from alfspy.core.rendering.camera import Camera
from alfspy.core.rendering.data import TextureData
from alfspy.core.torchgl import TorchContext, TorchTexture
from alfspy.core.util.basic import get_first_valid
from alfspy.core.util.defs import PATH_SEP
from alfspy.core.util.image import to_rgba_f4
from alfspy.core.util.pyrrs import quaternion_from_drone_pose


class CtxShot:
    """
    A single captured frame together with the camera pose it was taken from.

    The class name and every public member are carried over from the ModernGL version; only
    the texture type changed, from ``moderngl.Texture`` to
    :class:`~alfspy.core.torchgl.texture.TorchTexture`.
    """

    camera: Final[Camera]
    correction: Transform
    lazy: Final[bool]

    tex_data: Optional[TextureData]
    tex: Optional[TorchTexture]
    _tex_gen_input: Optional[tuple[tuple[int, int], int, bytes]]

    _ctx: Final[TorchContext]
    _img_file: Optional[str]
    _released: bool

    def __init__(self, ctx: TorchContext, img: Union[str, NDArray], position: Vector3, rotation: Quaternion,
                 fovy: float = 60.0, aspect_ratio: float = 1, correction: Optional[Transform] = None,
                 lazy: bool = False):
        """
        Initializes a new ``CtxShot`` object
        :param ctx: The context the shot should be associated with.
        :param img: Either the path to an image file as a string, or an already loaded image as an RGB numpy array.
        :param position: The position of the camera associated with the shot.
        :param rotation: The rotation of the camera associated with the shot.
        :param fovy: The field of view in y direction in degrees of the camera associated with the shot (defaults to 60).
        :param aspect_ratio: The aspect ratio of the view of the camera associated with the shot (defaults to 1).
        :param correction: Correction transform to be applied to the shot (optional). The translation stated in the
        correction transform is inverted to make it move according to the background coordinate system.
        :param lazy: Whether the shot should be loaded lazily (defaults to ``False``). This also loads the image lazily
        from the drive when an image-file-path is given.
        """
        self._released = False

        self.camera = Camera(fovy, aspect_ratio, position=position, rotation=rotation)
        if correction is None:
            self.correction = Transform(dtype='f4')
        else:
            self.correction = Transform(correction.position, correction.rotation, correction.scale, dtype='f4')

        self._ctx = ctx
        if isinstance(img, str):
            self._img_file = img
            self.tex_data = None
        else:
            self._img_file = None
            self.tex_data = TextureData(img.copy())
        self.lazy = lazy
        self.tex = None
        self._tex_gen_input = None
        if not lazy:
            self._init_texture()

    @staticmethod
    def _cvt_img(img: NDArray) -> NDArray:
        """
        Converts the given image to the format used by the renderer: RGBA with a f4 dtype.
        :param img: The image to convert.
        :return: The converted image.
        """
        return to_rgba_f4(img)

    @staticmethod
    def _load_image_from_path(img_path: str) -> NDArray:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f'Could not read shot image "{img_path}"')
        return CtxShot._cvt_img(img)

    def create_anew(self) -> 'CtxShot':
        """
        Initializes a new ``CtxShot`` object using the current values of this instance.
        Laziness is carried over and the new instance would have to reload the associated image data.
        :return: A new ``CtxShot`` instance.
        """
        img = self._img_file if self.lazy else self.img
        position = self.camera.transform.position
        rotation = self.camera.transform.rotation
        fovy = self.camera.fovy
        aspect_ratio = self.camera.aspect_ratio
        return CtxShot(self._ctx, img, position, rotation, fovy, aspect_ratio, self.correction, self.lazy)

    @property
    def _can_initialize(self):
        """
        :return: Whether the shot can currently be initialized.
        """
        return not self._released

    def load_image(self):
        """
        When the shot was initialized using a file path, loads the associated image.
        """
        if self.tex_data is None and self._can_initialize:  # ensures img_file is set
            self.tex_data = TextureData(self._load_image_from_path(str(self._img_file)))

    def load_tex_input(self):
        """
        Determines and caches the input required to create a texture on the render device.
        This data is released and removed when the texture gets initialized.

        Kept as the pre-upload step ``AsyncShotLoader`` performs off the main thread. The
        torch backend uploads from the array directly, so the cached bytes are no longer
        forwarded; the call still forces the (expensive) image decode to happen in the worker.
        """
        self.load_image()
        if self._tex_gen_input is None:
            self._tex_gen_input = self.tex_data.tex_gen_input()

    def _init_texture(self, ctx: Optional[TorchContext] = None):
        if self.tex is None and self._can_initialize:
            self.load_image()
            self._tex_gen_input = None
            context = ctx if ctx is not None else self._ctx
            self.tex = TorchTexture(self.tex_data.texture, device=context.device, dtype=context.dtype)

    @property
    def width(self) -> Optional[int]:
        """
        :return: If the shot was initialized using a path and that path was not yet loaded ``None``;
        Otherwise the texture width.
        """
        if self.tex_data is None:
            return None
        else:
            return self.tex_data.width

    @property
    def height(self) -> Optional[int]:
        """
        :return: If the shot was initialized using a path and that path was not yet loaded ``None``;
        Otherwise the texture height.
        """
        if self.tex_data is None:
            return None
        else:
            return self.tex_data.height

    @property
    def img(self) -> Optional[NDArray]:
        """
        :return: If the shot was initialized using a path and that path was not yet loaded ``None``;
        Otherwise a copy of the associated image in RGBA format.
        """
        if self.tex_data is None:
            return None
        else:
            return self.tex_data.texture.copy()

    def release(self) -> None:
        """
        Releases all objects associated with this shot.
        """
        if not self._released:
            if self.tex_data is not None:
                del self.tex_data
                self.tex_data = None

            if self.tex is not None:
                self.tex.release()
                del self.tex
                self.tex = None

            self._tex_gen_input = None
            self._released = True

    def tex_use(self, location: int = 0) -> None:
        """
        Ensures the texture of this shot is resident and binds it to a texture unit.
        """
        self.get_texture()

    def get_texture(self, ctx: Optional[TorchContext] = None) -> TorchTexture:
        """
        Returns this shot's texture, uploading it on first use.

        Replaces the ModernGL ``tex_use`` / sampler-binding pair: the torch backend passes
        textures to the shading functions explicitly rather than through global texture units.

        :param ctx: The context to upload to (optional; defaults to the shot's own context).
        :return: The texture associated with this shot.
        """
        if self._released:
            raise RuntimeError(
                'Shot cannot be used as it was initialized using image data and has already been released'
            )
        if self.tex_data is None:
            self.load_image()
        if self.tex is None:
            self._init_texture(ctx)
        return self.tex

    def get_proj(self) -> Matrix44:
        """
        :return: A matrix representing the projection of this shots camera.
        """
        return self.camera.get_proj(dtype='f4')

    def get_view(self) -> Matrix44:
        """
        :return: A matrix representing the view of this shots camera.
        """
        return self.camera.get_view(dtype='f4')

    def get_correction(self) -> Matrix44:
        """
        :return: A matrix representing the correction to be applied to this shot.
        """
        return self.correction.trans_mat.inverse * self.correction.rot_mat

    def get_mat(self) -> Matrix44:
        """
        :return: A matrix representing the combination of projection, view, and correction of this shot.
        """
        return self.camera.get_mat(dtype='f4') * self.get_correction()

    @staticmethod
    def _process_json(data: dict, ctx: TorchContext, count: Optional[int] = None, image_dir: Optional[str] = None,
                      fovy: float = 60.0, correction: Optional[Transform] = None, lazy: bool = False) \
            -> list[tuple[TorchContext, str, Vector3, Quaternion, float, float, Transform, bool]]:
        shot_params = []

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
                rotation = quaternion_from_drone_pose(rotation)
            elif rot_len == 4:
                rotation = Quaternion(rotation)
            else:
                raise ValueError(f'Invalid rotation format of length {rot_len}: {rotation}')

            if fov is None:
                fov = fovy
            elif isinstance(fov, list):
                fov = fov[0]

            img_file = f'{image_dir}{PATH_SEP}{img_file}'

            shot_params.append((ctx, img_file, Vector3(position), rotation, fov, 1, correction, lazy))
        return shot_params

    @staticmethod
    def from_json(file: str, ctx: TorchContext, count: Optional[int] = None, image_dir: Optional[str] = None,
                  fovy: float = 60.0, correction: Optional[Transform] = None, lazy: bool = False) -> list['CtxShot']:
        """
        Creates context shots from a JSON file.
        :param file: The path of the JSON file to process.
        :param ctx: The context to attach the context shots to.
        :param count: The maximum amount of shots to be created (optional).
        :param image_dir: The directory of the images referenced in the JSON file (defaults to the JSON files directory).
        :param fovy: The default fovy value to be used when a JSON entry does not provide one.
        :param correction: The general correction to be applied to all shots (optional).
        :param lazy: Whether the created shots should be lazy loaded (defaults to ``False``).
        :return: A list of ``CtxShot`` objects.
        """
        if image_dir is None:
            image_dir = str(pathlib.Path(file).parent.absolute())

        if correction is None:
            correction = Transform()

        with open(file, 'r') as f:
            data = json.load(f)

        shot_params = CtxShot._process_json(data, ctx, count, image_dir, fovy, correction, lazy)

        return [CtxShot(*shot_param) for shot_param in shot_params]
