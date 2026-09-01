"""Reading and writing N-channel light fields.

A rendered field is an ``(H, W, C)`` float32 array -- 2048x2048x1280 is about 21 GB, so it is
written as ``.npy`` through a memmap and streamed straight into its final file rather than
built in RAM first.

``.npy`` carries no metadata, so every field gets a ``<stem>_meta.json`` sidecar recording
what the channels are and what produced them. Without it a saved field is an anonymous block
of numbers.
"""

import json
import os
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from alfspy.render.field import ChannelSpec

__all__ = ['FieldMetadata', 'meta_path', 'open_field', 'save_field', 'load_field', 'load_metadata']

_SUFFIX = '_meta.json'


@dataclass
class FieldMetadata:
    """
    What a saved field is.

    :cvar channels: The channel description.
    :cvar shape: The ``(H, W, C)`` shape of the saved array.
    :cvar flight_id: The flight the field was rendered from (optional).
    :cvar frame_index: The central frame index (optional).
    :cvar camera_position: The virtual camera position (optional).
    :cvar render_resolution: The ``(width, height)`` rendered (optional).
    :cvar shot_count: How many shots contributed (optional).
    :cvar engine: Which render backend produced it (optional).
    :cvar extra: Anything else worth recording.
    """
    channels: ChannelSpec
    shape: Tuple[int, int, int]
    flight_id: Optional[str] = None
    frame_index: Optional[int] = None
    camera_position: Optional[Tuple[float, float, float]] = None
    render_resolution: Optional[Tuple[int, int]] = None
    shot_count: Optional[int] = None
    engine: Optional[str] = None
    extra: Dict[str, Any] = dataclass_field(default_factory=dict)

    def as_dict(self) -> dict:
        """
        :return: A JSON-serialisable representation.
        """
        return {
            'channels': self.channels.as_dict(),
            'shape': list(self.shape),
            'flight_id': self.flight_id,
            'frame_index': self.frame_index,
            'camera_position': (list(self.camera_position)
                                if self.camera_position is not None else None),
            'render_resolution': (list(self.render_resolution)
                                  if self.render_resolution is not None else None),
            'shot_count': self.shot_count,
            'engine': self.engine,
            'extra': self.extra,
        }

    @staticmethod
    def from_dict(data: dict) -> 'FieldMetadata':
        """
        :param data: A dict as produced by :meth:`as_dict`.
        :return: The reconstructed metadata.
        """
        channels = data.get('channels') or {}
        return FieldMetadata(
            channels=ChannelSpec(
                count=channels.get('count', 0),
                names=channels.get('names'),
                source=channels.get('source'),
                dtype=channels.get('dtype', 'float32'),
            ),
            shape=tuple(data.get('shape', ())),
            flight_id=data.get('flight_id'),
            frame_index=data.get('frame_index'),
            camera_position=(tuple(data['camera_position'])
                             if data.get('camera_position') else None),
            render_resolution=(tuple(data['render_resolution'])
                               if data.get('render_resolution') else None),
            shot_count=data.get('shot_count'),
            engine=data.get('engine'),
            extra=data.get('extra', {}),
        )


def meta_path(npy_path: str) -> str:
    """
    :param npy_path: Path of the ``.npy`` field.
    :return: Path of its metadata sidecar.
    """
    stem = npy_path[:-4] if npy_path.lower().endswith('.npy') else npy_path
    return f'{stem}{_SUFFIX}'


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def open_field(path: str, shape: Tuple[int, int, int],
               dtype: Any = np.float32) -> np.memmap:
    """
    Creates an on-disk array to render a field into.

    Pass the result as ``out=`` to
    :func:`~alfspy.render.field.render_field_integral` so the render streams straight into
    its final file. A 2048x2048x1280 float32 field is about 21 GB, which is why this exists.

    :param path: Where to write the ``.npy``.
    :param shape: The ``(H, W, C)`` shape to allocate.
    :param dtype: The storage dtype.
    :return: A writable memmap backed by ``path``.
    """
    _ensure_parent(path)
    return np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)


def save_field(path: str, data: NDArray, metadata: Optional[FieldMetadata] = None) -> str:
    """
    Writes a field and its metadata sidecar.

    :param path: Where to write the ``.npy``.
    :param data: The ``(H, W, C)`` field.
    :param metadata: The metadata to record (optional). When omitted a minimal record is
        derived from the array itself, so a field is never saved without a sidecar.
    :return: The path the metadata was written to.
    """
    _ensure_parent(path)

    if isinstance(data, np.memmap):
        data.flush()
    else:
        np.save(path, data)

    if metadata is None:
        metadata = FieldMetadata(
            channels=ChannelSpec(count=int(data.shape[-1])),
            shape=tuple(int(v) for v in data.shape),
        )

    target = meta_path(path)
    with open(target, 'w', encoding='utf-8') as handle:
        json.dump(metadata.as_dict(), handle, indent=2)
    return target


def load_field(path: str, mmap: bool = True) -> Tuple[NDArray, Optional[FieldMetadata]]:
    """
    Loads a field and its metadata.

    :param path: Path of the ``.npy``.
    :param mmap: Whether to memory-map rather than read into RAM (defaults to ``True``,
        because these arrays are routinely larger than memory).
    :return: The field and its metadata, or ``None`` for the metadata if no sidecar exists.
    """
    data = np.load(path, mmap_mode='r' if mmap else None)
    return data, load_metadata(path)


def load_metadata(path: str) -> Optional[FieldMetadata]:
    """
    :param path: Path of the ``.npy`` or of the sidecar itself.
    :return: The metadata, or ``None`` if there is no sidecar.
    """
    target = path if path.endswith(_SUFFIX) else meta_path(path)
    if not os.path.exists(target):
        return None
    with open(target, 'r', encoding='utf-8') as handle:
        return FieldMetadata.from_dict(json.load(handle))
