from itertools import repeat, chain
from typing import Union, Optional, Callable, Sequence

import cv2
from numpy.typing import NDArray

def _ensure_img(img: Union[str, NDArray]) -> NDArray:
    return cv2.imread(img) if isinstance(img, str) else img

def video_from_images(images: Sequence[Union[str, NDArray]], video_file: str, fps: float = 30.0, fourcc: str = 'mp4v',
                      img_filter: Optional[Callable] = None, first_frame_repetitions: Optional[int] = None,
                      last_frame_repetitions: Optional[int] = None, release_images: bool = False) -> None:
    """
    Creates a video from a collection of images
    :param images: A collection of images or image file paths to be concatenated to one video
    :param video_file: The name of the video file to create
    :param fps: The frames per seconds the video should have
    :param fourcc: The four-digit code associated with the file format to be used for video creation
    :param img_filter: The filter to apply to every image before adding it to the video (optional)
    :param first_frame_repetitions: How often the first frame should be repeated at the beginning of the video (defaults to ``0``)
    :param last_frame_repetitions: How often the last frame should be repeated at the end of the video (defaults to ``0``)
    :param release_images: Whether to release image resources after usage (defaults to ``False``)
    """

    img_count = len(images)
    if img_count <= 0:
        raise ValueError('At least one image has to be passed')

    if '.' not in video_file:
        video_file += '.avi'

    if first_frame_repetitions is None or first_frame_repetitions < 0:
        first_frame_repetitions = 0

    if last_frame_repetitions is None or last_frame_repetitions < 0:
        last_frame_repetitions = 0

    if img_count == 1:
        single_image = _ensure_img(images[0])
        if img_filter is not None:
            single_image = img_filter(single_image)
        images = repeat(single_image, 1 + first_frame_repetitions + last_frame_repetitions)
    else:
        first, *mid, last = images
        first_image = _ensure_img(first)
        last_image = _ensure_img(last)
        mid = (_ensure_img(img) for img in mid)

        if img_filter is not None:
            first_image = img_filter(first_image)
            last_image = img_filter(last_image)
            mid = (img_filter(img) for img in mid)

        images = chain(repeat(first_image, first_frame_repetitions + 1), mid, repeat(last_image, 1 + last_frame_repetitions))

    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    height, width = next(iter(images)).shape[:2]

    video_writer = cv2.VideoWriter(video_file, fourcc_code, fps, (width, height))
    for img in images:
        video_writer.write(img)
        if release_images:
            del img

    video_writer.release()
