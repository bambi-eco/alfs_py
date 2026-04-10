from dataclasses import dataclass, field
from typing import Optional

from alfspy.render.data import IntegralSettings


@dataclass
class EmbeddingSettings(IntegralSettings):
    """
    Settings for the DINOv3 embedding light field pipeline.
    Extends ``IntegralSettings`` with DINOv3-specific parameters and a convenience
    interface for selecting a central frame and its neighbors.

    Frame selection (convenience):
    --------------------------------
    Set ``central_frame_idx`` together with ``neighbor_range_before``,
    ``neighbor_range_after``, and ``frame_stride``. On initialization these are
    translated into the underlying ``initial_skip``, ``count``, and ``skip`` fields
    that ``_base_steps`` / ``read_shots`` use.

    Example — central frame 100, ±10 neighbors, every other frame::

        EmbeddingSettings(
            central_frame_idx=100,
            neighbor_range_before=10,
            neighbor_range_after=10,
            frame_stride=2,
        )

    Alternatively, set ``initial_skip``, ``count``, and ``skip`` directly (the
    inherited ``BaseSettings`` approach).

    :cvar model_dir: Path to the local HuggingFace model directory (safetensors + config).
    :cvar embed_output_file: Output path for the (H, W, embed_dim) float32 numpy array.
    :cvar precomputed_dir: Optional directory to cache/reload per-shot embedding .npy files.
        If a shot embedding file already exists it will be reloaded instead of recomputed.
    :cvar also_save_rgb: Whether to additionally produce a standard RGB render alongside
        the embedding light field.
    :cvar central_frame_idx: Index of the central (key) frame in the poses JSON.
        When set, ``initial_skip`` / ``count`` / ``skip`` are derived automatically.
    :cvar neighbor_range_before: Number of frames to include before the central frame.
    :cvar neighbor_range_after: Number of frames to include after the central frame.
    :cvar frame_stride: Step size when sampling neighbor frames (1 = every frame,
        2 = every other frame, …).
    """
    model_dir: str = r'D:\DINOv3'
    embed_output_file: str = 'embedding_lightfield.npy'
    precomputed_dir: Optional[str] = None
    also_save_rgb: bool = False
    alpha_threshold: float = 0.1   # minimum summed alpha for a pixel to be considered covered

    # UpsampleAnything integration
    upa_repo_dir: str = r'D:\Upsample-Anything_Pytorch'
    upsample_before_render: bool = False  # Stage 1: UPA per-shot embedding before ALFS
    upsample_after_render: bool = False   # Stage 2: UPA aggregated result after ALFS
    upsample_scale: int = 2               # Scale factor for Stage 2 only

    # Convenience frame-selection fields
    central_frame_idx: Optional[int] = None
    neighbor_range_before: int = 0
    neighbor_range_after: int = 0
    frame_stride: int = 1

    def __post_init__(self):
        if self.central_frame_idx is not None:
            first = self.central_frame_idx - self.neighbor_range_before
            if first < 0:
                raise ValueError(
                    f'central_frame_idx ({self.central_frame_idx}) - '
                    f'neighbor_range_before ({self.neighbor_range_before}) = {first} < 0'
                )
            total_frames = self.neighbor_range_before + 1 + self.neighbor_range_after
            self.initial_skip = first
            self.count = total_frames
            self.skip = self.frame_stride
