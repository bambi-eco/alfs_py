from typing import Union, Collection, Optional, Callable

import cv2
from numpy.typing import NDArray


def video_from_images(images: Collection[Union[str, NDArray]], video_file: str, fps: float = 30.0, fourcc: str = 'mp4v',
                      img_filter: Optional[Callable] = None, release_images: bool = False) -> None:
    """
    Creates a video from a collection of images
    :param images: A collection of images or image file paths to be concatenated to one video
    :param video_file: The name of the video file to create
    :param fps: The frames per seconds the video should have
    :param fourcc: The four-digit code associated with the file format to be used for video creation
    :param img_filter: The filter to apply to every image before adding it to the video (optional)
    :param release_images: Whether to release image resources after usage (defaults to ``False``)
    """

    if len(images) <= 0:
        raise ValueError('At least one image has to be passed')

    if '.' not in video_file:
        video_file += '.avi'

    if img_filter is None:
        images = (cv2.imread(img) if isinstance(img, str) else img for img in images)
    else:
        images = (img_filter(cv2.imread(img) if isinstance(img, str) else img) for img in images)

    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    height, width = next(iter(images)).shape[:2]

    video_writer = cv2.VideoWriter(video_file, fourcc_code, fps, (width, height))
    for img in images:
        video_writer.write(img)
        if release_images:
            del img

    video_writer.release()
