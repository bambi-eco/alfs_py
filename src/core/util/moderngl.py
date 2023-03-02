import cv2
import numpy as np
from moderngl import Framebuffer
from numpy.typing import NDArray


def img_from_fbo(fbo: Framebuffer, attachment: int = 0) -> NDArray[np.uint8]:
    """
    Reads image data from the FBO and turns it into an OpenCV representation (BGRA)
    :param fbo: The frame buffer to read image data from
    :param attachment: Number of the color attachment to read from
    :return: A numpy array containing the image in the BGRA format
    """
    raw = fbo.read(components=4, attachment=attachment, dtype='f1')
    img = np.frombuffer(raw, dtype=np.uint8).reshape((*fbo.size[1::-1], 4))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    return cv2.flip(img, 0)  # modern gl seems to vertically flip output