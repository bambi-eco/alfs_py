import base64
from typing import Union, Optional, Iterable, Sequence, Any
from urllib.request import urlopen

import cv2
import numpy as np
from gltflib import GLTF
from moderngl import Framebuffer
from numpy import ndarray
from numpy.typing import NDArray
from pyrr import Vector3

from src.core.data import MeshData, TextureData, AABB
from src.core.defs import Color


def get_first_valid(in_dict: dict[Any], keys: Iterable[Any], default: Optional[Any] = None) -> Any:
    """
    Returns the value within a dict associated with the first valid key of a iterable key source
    :param in_dict: The dictionary to search within
    :param keys: An iterable source of keys
    :param default: The value to be returned when no valid key was passed (optional)
    :return: The value of default if no valid key was passed; otherwise the value associated with the first valid key
    """
    for key in keys:
        if key in in_dict:
            return in_dict.get(key)
    return default


def img_from_fbo(fbo: Framebuffer) -> NDArray[np.uint8]:
    """
    Reads image data from the FBO and turns it into an OpenCV representation (BGRA)
    :param fbo: The frame buffer to read image data from
    :return: A numpy array containing the image
    """
    raw = fbo.read(components=4, dtype='f1')
    img = np.frombuffer(raw, dtype=np.uint8).reshape((*fbo.size[1::-1], 4))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    return cv2.flip(img, 0)  # modern gl seems to vertically flip output


def base64_to_img(base64_data: Union[bytes, str]) -> NDArray[np.number]:
    """
    Turns a base64 representation of an image into an OpenCV representation
    :param base64_data: The base64 data representing an image
    :return: A numpy array containing the image
    """
    return bytes_to_img(base64.b64decode(base64_data))


def bytes_to_img(data: Union[bytes, str]) -> NDArray[np.number]:
    """
    Turns a base 8 byte representation of an image into an OpenCV representation
    :param data: The byte data representing an image
    :return: A numpy array containing the image
    """
    data_arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(data_arr, flags=cv2.IMREAD_COLOR)
    return img


def split_components(arr: NDArray) -> tuple[NDArray, ...]:
    """
    Slices a numpy array into arrays of its highest order axis.
    ``[[x1, y1, z1], [x2, y2, z2]] => [x1, x2], [y1, y2], [z1, z2]``
    :param arr: The numpy array to be sliced
    :return: A tuple of slices of the given numpy aray
    """
    depth = len(arr.shape) - 1
    layers = arr.shape[-1]
    results = np.split(arr, layers, axis=depth)
    results = [result.reshape(([-1])) for result in results]
    return tuple(results)


def compare_color(col_a: Union[NDArray, Iterable, int, float], col_b: Union[NDArray, Iterable, int, float]) -> bool:
    """
    Compares two color values
    :param col_a: The first color
    :param col_b: The second color
    :return: ``False`` if any component of the two colors differs; otherwise ``True``
    """
    if not isinstance(col_a, ndarray):
        col_a = np.array(col_a)
    if not isinstance(col_b, ndarray):
        col_b = np.array(col_b)
    return (col_a == col_b).all()


def overlay(img_a: NDArray, img_b: NDArray) -> Optional[NDArray]:
    """
    Tries to overlay an image onto another
    :param img_a: The image serving as background
    :param img_b: The image serving as foreground
    :return: If the images do not share the same shape ``None``; otherwise the generated image
    """
    if img_a.shape != img_b.shape:  # Images are not compatible
        return None
    if len(img_a.shape) < 3 or img_a.shape[2] < 4:  # images have no alpha channel, overlay would just show img_b
        return img_b.copy()

    result = img_a.copy()
    over = img_b.copy()

    alpha = img_b[..., 3]
    fact = (255.0 - alpha) / 255.0

    # scale all layers accept alpha according to the overlays alpha values
    for i in range(0, result.shape[-1] - 1):
        result[..., i] = (result[..., i] * fact).astype(img_a.dtype)
        over[..., i] = (over[..., i] * alpha).astype(img_b.dtype)
    result += over
    return result


def _get_border_size(border: tuple[int, int, int, int], width: int, height: int) -> int:
    result = 0
    top, right, bottom, left = border
    r_w = width - right
    c_w = r_w - left

    result += (left + (width - right)) * height
    result += (top + (height - bottom)) * c_w

    return result


def _get_largest_border(img: NDArray, color: NDArray) -> tuple[int, int, int, int]:
    mask = img[:, :] == color

    height, width = mask.shape[0:2]
    top = left = -1
    bottom = height + 1
    right = width + 1

    done = False
    while not done and left < width:
        left += 1
        done = not mask[:, left:left + 1].all()

    done = False
    while not done and right > left:
        right -= 1
        done = not mask[:, right - 1: right].all()

    done = False
    while not done and top < height:
        top += 1
        done = not mask[top:top + 1, left:right].all()

    done = False
    while not done and bottom > top:
        bottom -= 1
        done = not mask[bottom - 1:bottom, left:right].all()

    return top, right, bottom, left


def crop_to_content(img: NDArray, return_delta: bool = False) -> Union[NDArray, tuple[NDArray, Vector3]]:
    """
    Crops the given image by removing the largest possible border area of the same color
    :param img: The image to be cropped
    :param return_delta: Whether the delta should also be returned (defaults to False). The delta is the vector to be
    added to the center of the cropped image to obtain the coordinates of the original center of the image.
    Its Z component will always equal 0.
    :return: A copy of the cropped segment of the given image
    """
    if img.shape[0] == 0 or img.shape[1] == 0:  # image has no pixels
        return img.copy()

    height, width = img.shape[0:2]
    max_x = width - 1
    max_y = height - 1

    # get all unique corner colors
    unq_colors = []
    for color in img[0, 0], img[max_y, 0], img[max_y, max_x], img[0, max_x]:
        if not any([compare_color(c, color) for c in unq_colors]):
            unq_colors.append(color)

    if len(unq_colors) >= 4:  # cannot crop since all corners have different colors
        return img.copy()

    # search the biggest border possible with any of the unique corner colors
    max_border_size = -1
    max_border = None
    for color in unq_colors:
        border = _get_largest_border(img, color)
        border_size = _get_border_size(border, width, height)
        if border_size > max_border_size:
            max_border_size = border_size
            max_border = border

    top, right, bottom, left = max_border
    crop = img[top:bottom, left:right].copy()
    if return_delta:
        old_center = Vector3([width / 2.0, height / 2.0, 0.0])
        new_center = Vector3([left + (right - left) / 2.0, top + (bottom - top) / 2.0, 0.0])
        delta = old_center - new_center
        return crop, delta
    else:
        return crop


def blend(images: Sequence[NDArray]) -> Optional[NDArray]:
    """
    Blends multiple images into one
    :param images: A sequence of images to blend together into one
    :return: None if no images are passed or if the images have different resolutions; otherwise the blended image
    """
    img_count = len(images)
    if img_count == 0:
        return None
    shape = images[0].shape
    if any([img for img in images if img.shape != shape]):
        return None

    weight = 1.0 / img_count
    result = np.sum(images) * weight
    return result


def _get_from_buffer(idx: int, gltf: GLTF, comp: int, dtype: str = 'f4') -> NDArray:
    """
    Interprets the portion of the first buffer of the given ``GLTF`` object
    associated with the given buffer view index as a vector with given amount of components
    :param idx: Buffer view index of the data to be read
    :param gltf: The ``GLTF`` object holding all data
    :param comp: The amount of components of the vectors to read
    :param dtype: The type the read values should be converted to
    :return: A numpy array representing all read vectors
    """
    buffer = gltf.resources[0].data
    accessor = list(filter(lambda acc: acc.bufferView == idx, gltf.model.accessors))[0]
    buf_view = gltf.model.bufferViews[accessor.bufferView]
    data_bytes = buffer[buf_view.byteOffset:buf_view.byteOffset + buf_view.byteLength]
    return np.frombuffer(data_bytes, dtype=dtype).reshape((-1, comp))


def gltf_to_mesh_data(gltf: GLTF) -> Optional[MeshData]:
    """
    Extracts the first mesh found within the ``GLTF`` object
    :param gltf: The ``GLTF`` object to extract the mesh from
    :return: If the ``GLTF`` object contains no meshes returns ``None``;
    otherwise returns a ``MeshData`` object containing vertices, indices and UV coordinates.
    If indices and/or UV coordinates cannot be extracted they will be set to None.
    """
    if len(gltf.model.meshes) < 1 or len(gltf.model.meshes[0].primitives) < 1:
        return None
    mesh_attrs = gltf.model.meshes[0].primitives[0].attributes

    vertices = _get_from_buffer(mesh_attrs.POSITION, gltf, 3)

    indices_idx = gltf.model.meshes[0].primitives[0].indices
    indices = _get_from_buffer(indices_idx, gltf, 3, dtype='u4') if indices_idx is not None else None

    uvs_idx = mesh_attrs.TEXCOORD_0
    uvs = _get_from_buffer(uvs_idx, gltf, 2) if uvs_idx is not None else None

    return MeshData(vertices, indices, uvs)


def gltf_to_texture_data(gltf: GLTF) -> Optional[TextureData]:
    """
    Extracts the texture of the main mesh found within the ``GLTF`` object
    :param gltf: The ``GLTF`` object to extract the texture from
    :return: If the there is no mesh within the ``GLTF`` object or the main mesh has no texture returns ``None``;
    otherwise returns a ``TextureData`` object containing the extracted texture
    """

    if len(gltf.model.meshes) < 1 or len(gltf.model.meshes[0].primitives) < 1:  # No main mesh exists
        return None

    material_idx = gltf.model.meshes[0].primitives[0].material  # Main mesh has no material
    if material_idx is None:
        return None

    material = gltf.model.materials[material_idx]
    texture_info = material.pbrMetallicRoughness.baseColorTexture
    if texture_info is None or texture_info.index is None:  # Material has no texture assigned
        return None

    # assume texture is embedded using a base64 URI
    uri = gltf.model.images[texture_info.index].uri

    with urlopen(uri) as resp:
        data = resp.read()
    texture = bytes_to_img(data)

    return TextureData(texture) if texture is not None else None


def gltf_extract(file: str) -> tuple[MeshData, TextureData]:
    """
    Extracts mesh and texture data from a GLTF file via ``get_mesh_data`` and ``get_texture_data``
    :param file: Path to a GLTF file
    :return: A tuple containing the results of ``get_mesh_data`` and ``get_texture_data``
    """
    gltf_file = GLTF.load(file, load_file_resources=True)
    return gltf_to_mesh_data(gltf_file), gltf_to_texture_data(gltf_file)


def get_center(vertices: NDArray) -> tuple[Vector3, AABB]:
    """
    Computes the center position and AABB of a set of vertices
    :param vertices: A numpy array containing vertices
    :return: A tuple containing the center of the vertices and the two points defining the vertices AABB
    """
    max_x, max_y, max_z = np.max(vertices, axis=0)
    min_x, min_y, min_z = np.min(vertices, axis=0)
    dhx = abs(max_x - min_x) / 2.0
    dhy = abs(max_y - min_y) / 2.0
    dhz = abs(max_z - min_z) / 2.0

    center = Vector3((min_x + dhx, min_y + dhy, min_z + dhz))
    max_p = Vector3((max_x, max_y, max_z))
    min_p = Vector3((min_x, min_y, min_z))
    return center, AABB(min_p, max_p)


def make_plane(size: float = 1.0, y: float = 0.0) -> MeshData:
    """
    Creates an axis aligned plane of the given size
    :param size: Side length of the plane
    :param y: Value to apply to the Y component of all vertices
    :return: A ``MeshData`` object representing the generated plane
    """
    size_h = size / 2.0
    vertices = np.array([[-size_h, y, size_h], [-size_h, y, -size_h],
                         [size_h, y, -size_h], [size_h, y, size_h]])
    indices = np.array([0, 1, 2, 2, 3, 0])
    uvs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    return MeshData(vertices, indices, uvs)


def int_up(val: float) -> int:
    """
    Rounds a float up and casts it to an integer
    :param val: The value to round up
    :return: The rounded up integer
    """
    return int(val + 0.5)


def gen_checkerboard_tex(tile_per_side: int, tile_size: int, tile_color: Color, non_tile_color: Color,
                         dtype: object = float) -> NDArray:
    """
    Generates a numpy array representing a checkerboard texture
    :param tile_per_side: The amount of tiles per side
    :param tile_size: The side length of a tile in pixels
    :param tile_color: The color of a tile
    :param non_tile_color: The color of the lack of a tile
    :param dtype: The dtype to convert all values within the numpy to (defaults to float)
    :return: A numpy array representing a checkerboard texture
    """
    t_c = np.array(tile_color).reshape((-1))
    nt_c = np.array(non_tile_color).reshape((-1))

    if t_c.shape != nt_c.shape:
        raise ValueError('Both given colors have to have the same amount of components')

    if tile_per_side == 0 or tile_size == 0:
        return np.empty((0, 0, t_c.shape[0]), dtype=dtype)

    depth = t_c.shape[0]
    tile = np.empty((tile_size, tile_size, depth), dtype=dtype)
    n_tile = np.empty_like(tile)
    tile[..., :] = t_c
    n_tile[..., :] = nt_c

    odd_line = cv2.hconcat([tile if i % 2 == 0 else n_tile for i in range(0, tile_per_side)])
    even_line = cv2.hconcat([tile if i % 2 == 1 else n_tile for i in range(0, tile_per_side)])
    result = cv2.vconcat([odd_line if i % 2 == 0 else even_line for i in range(0, tile_per_side)])

    return result

