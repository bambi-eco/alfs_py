"""Framebuffer readback for the WebGPU backend."""

import numpy as np
from numpy.typing import NDArray

__all__ = ['img_from_fbo']


def img_from_fbo(renderer, attachment: int = 0) -> NDArray:
    """
    Reads a colour attachment back as an image.

    Note there is no vertical flip here, unlike the OpenGL backend: WebGPU's framebuffer row
    0 is the top of the image while OpenGL's is the bottom, so the data already arrives in
    image order.

    :param renderer: The renderer whose attachments to read.
    :param attachment: ``0`` for the samples, ``1`` for the coverage count.
    :return: An RGBA ``uint8`` image for attachment 0, or the raw float coverage for 1.
    """
    result = renderer._read_targets()
    if attachment == 0:
        return (np.clip(result.accum, 0.0, 1.0) * 255).astype(np.uint8)
    return result.coverage
