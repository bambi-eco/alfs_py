import base64
from numbers import Integral
from typing import Union, Optional, Sequence, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from pyrr import Vector3

from src.core.util.basic import compare_color
from src.core.util.defs import Color


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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


def replace_color(img: NDArray, color_from: Color, color_to: Color, inplace: bool = False) -> NDArray:
    if inplace:
        img = img.copy()
    width, height = img.shape[1::-1]
    img = np.reshape(img, (height, width, -1))
    channels = img.shape[-1]
    color_from = color_from[:channels]
    color_to = color_to[:channels]

    selection = np.all(img == color_from, axis=-1)
    img[selection] = color_to
    return img


def overlay(img_a: NDArray, img_b: NDArray) -> Optional[NDArray]:
    """
    Tries to overlay an image onto another
    :param img_a: The image serving as background
    :param img_b: The image serving as foreground
    :return: If the images do not share the same shape or dtype ``None``; otherwise the generated image
    """
    if img_a.shape != img_b.shape or img_a.dtype != img_b.dtype:  # Images are not compatible
        return None
    if len(img_a.shape) < 3 or img_a.shape[2] < 4:  # images have no alpha channel, overlay would just show img_b
        return img_b.copy()

    dtype = img_a.dtype
    overlay_strength = np.resize(img_b[..., 3], (img_b.shape[0], img_b.shape[1], 1)) / 1.0
    if not isinstance(dtype, np.floating):
        overlay_strength /= 255.0
    return ((img_b * overlay_strength) + (img_a * (1.0 - overlay_strength))).astype(dtype)


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
        return img.copy(), Vector3() if return_delta else img.copy()

    height, width = img.shape[0:2]
    max_x = width - 1
    max_y = height - 1

    # get all unique corner colors
    unq_colors = []
    for color in img[0, 0], img[max_y, 0], img[max_y, max_x], img[0, max_x]:
        if not any([compare_color(c, color) for c in unq_colors]):
            unq_colors.append(color)

    if len(unq_colors) >= 4:  # cannot crop since all corners have different colors
        return img.copy(), Vector3() if return_delta else img.copy()

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
        delta = new_center - old_center
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


def integral(images: Sequence[NDArray], dtype=None) -> Optional[NDArray]:
    """
    Overlaps all given images and blends them together weighted by the amount of non-transparent overlaps per pixel.
    :param images: A sequence of images to blend together
    :param dtype:
    :return: None if no images are passed or if the images have different resolutions; otherwise the blended image
    """
    img_count = len(images)
    if img_count == 0:
        return None
    shape = images[0].shape

    if any([True for img in images if img.shape != shape]):
        return None

    if dtype is None:
        dtype = images[0].dtype

    stack = np.sum(images, axis=0)
    overlaps = stack[:, :, -1]
    overlaps[overlaps <= 1.0] = 1.0
    result = np.divide(stack, overlaps[:, :, np.newaxis])

    # if dtype is any integer, assume color values are between 0 and 255
    if issubclass(np.dtype(dtype).type, Integral):
        result *= 255

    return result.astype(dtype)


def laplacian_variance(img: NDArray) -> float:
    """
    Computes the variance of the edges detected within an image using laplacian edge detection.
    :param img: The image to compute the variance for
    :return: The variance of laplacian filtered image
    """
    return cast(float, np.var(cv2.Laplacian(img, cv2.CV_64F)))
