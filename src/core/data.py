from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable

from src.core.geo.transform import Transform
from src.core.rendering.data import Resolution


class CameraPositioningMode(Enum):
    """
    Selection of defined camera positioning
    :cvar background_centered: The camera is placed at the XY-center and just outside the AABB along the Z-axis of the background object
    :cvar first_shot: The camera takes the position of the first shot
    :cvar center_shot: The camera takes the position of the shot at position ``n/2`` given ``n`` shots
    :cvar last_shot: The camera takes the position of the last shot
    :cvar average_shot: The camera is placed at the average position of all shots
    :cvar shot_centered: The camera is placed at the center of the smallest AABB enclosing all shot positions
    """
    background_centered = 0x000001
    first_shot          = 0x000010
    center_shot         = 0x000100
    last_shot           = 0x001000
    average_shot        = 0x010000
    shot_centered       = 0x100000

@dataclass
class BaseSettings:
    """
    Class storing basic settings used by rendering processes
    :cvar count: The max amount of shots to be used
    :cvar initial_skip: The number of shots read from the JSON file to be skipped from the beginning
    :cvar skip: The number of shots to be skipped after every shot projection
    :cvar lazy: Whether the shots should be loaded lazy
    :cvar add_background: Whether to overlay the result over a render of the background object
    :cvar camera_dist: The distance between the background objects and the camera along the Z-axis
    :cvar camera_position_mode: The camera position mode to be used for camera placement
    :cvar resolution: The resolution to render at
    :cvar near_clipping: Distance to the near clipping plane
    :cvar far_clipping: Distance to the far clipping plane
    :cvar fovy: The FOV along the Y-axis when using a perspective camera
    :cvar aspect_ratio: The aspect ratio when using a perspective camera
    :cvar orthogonal: Whether the camera to be used should be orthogonal
    :cvar ortho_size: The dimensions of the orthogonal frustum when using an orthogonal camera
    :cvar release_shots: Whether the renderer should release shots as soon as he used them
    :cvar correction: The correction to be applied to every single shot
    :cvar output_file: The path and name of the output to be generated
    """
    count: int = 1
    initial_skip: int = 0
    skip: int = 1
    lazy: bool = True
    add_background: bool = True
    camera_dist: float = 1.0
    camera_position_mode: CameraPositioningMode = CameraPositioningMode.background_centered
    resolution: Resolution = Resolution(1024, 1024)
    near_clipping: float = 0.1
    far_clipping: float = 10000
    fovy: Optional[float] = None
    aspect_ratio: Optional[float] = None
    orthogonal: bool = True
    ortho_size: Optional[tuple[float, float]] = None
    release_shots: bool = True
    correction: Optional[Transform] = None
    output_file: str = ''

@dataclass
class BaseAnimationSettings(BaseSettings):
    """
    Class storing basic settings used by rendering processes that create animations
    :cvar frame_count: The amount of frames to be rendered in total
    :cvar fps: The amount of frames per second to be used when creating the video
    :cvar frame_dir: The directory to which all frames will be saved
    :cvar delete_frames: Whether the frames saved to the frame directory should be deleted after they were written to the video file
    :cvar first_frame_repetitions: How often the first frame should be repeated at the beginning of the animation
    :cvar last_frame_repetitions: How often the last frame should be repeated at the end of the animation
    """
    frame_count: int = 3600
    fps: float = 30.0
    frame_dir: str = './.frames'
    delete_frames: bool = True
    first_frame_repetitions: Optional[int] = None
    last_frame_repetitions: Optional[int] = None


@dataclass
class ProjectionSettings(BaseSettings):
    """
    Class storing settings used by the projection process.
    Extends upon ``BaseSettings`` parameters.
    :cvar show_integral: Whether to show the resulting integral in a photo viewer
    """
    show_integral: bool = False


@dataclass
class FocusAnimationSettings(BaseAnimationSettings):
    """
    Class storing settings used by the focus animation process.
    Extends upon ``BaseAnimationSettings`` parameters.
    :cvar start_focus: The starting value for the focus
    :cvar end_focus: The focus the animation should have on the last frame
    :cvar move_camera_with_focus: Whether the focus change should also cause the render camera to move
    """
    start_focus: float = 0.0
    end_focus: float = 5.0
    move_camera_with_focus: bool = False


@dataclass
class ShutterAnimationSettings(BaseAnimationSettings):
    """
    Class storing settings used by the shutter animation process.
    Extends upon ``BaseAnimationSettings`` parameters.
    :cvar shots_grow_func: Function yielding how many shots should be added after rendering the frame of the given number
    :cvar reference_index: The index of reference shot
    :cvar grow_symmetrical: Whether the shots before the initial shot should be added in addition to the shots after.
    If ``True`` shots get added symmetrically around the initial shot.
    """
    shots_grow_func: Callable[[int], int] = lambda x: 1
    reference_index: int = 0
    grow_symmetrical: bool = False
