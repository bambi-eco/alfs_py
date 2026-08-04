"""Framebuffer readback helpers.

Replaces ``alfspy.core.util.moderngl`` from the ModernGL version. The vertical flip is
retained because :class:`~alfspy.core.torchgl.framebuffer.TorchFramebuffer` deliberately
keeps GL's bottom-up row order.
"""

import numpy as np
from numpy.typing import NDArray

from alfspy.core.torchgl import TorchFramebuffer


def img_from_fbo(fbo: TorchFramebuffer, attachment: int = 0) -> NDArray[np.uint8]:
    """
    Reads image data from the framebuffer and turns it into a top-down 8-bit image.
    :param fbo: The frame buffer to read image data from.
    :param attachment: Number of the color attachment to read from.
    :return: A numpy array containing the image in the RGBA format.
    """
    if attachment != 0:
        raise ValueError(f'TorchFramebuffer has a single attachment but {attachment} was requested')
    img = fbo.read_array(dtype='f1')
    return img[::-1, ...].copy()  # framebuffer rows are stored bottom-up, as in GL
