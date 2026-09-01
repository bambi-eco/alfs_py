"""Torch version-compatibility shims.

The port deliberately avoids the fast-moving parts of the torch API. What remains is a
handful of primitives whose behaviour has been stable for a long time, plus one
(``Tensor.scatter_reduce_``) that is recent enough to deserve a runtime probe.

The probes run once at import and are cheap (a few elements). If a future torch release
changes semantics, the affected feature degrades to a slower fallback and
:data:`COMPAT_NOTES` records why -- the renderer never silently produces different images.
"""

from typing import Any, Final, Optional

import numpy as np
import torch
from numpy.typing import NDArray

__all__ = [
    'TORCH_VERSION',
    'HAS_SCATTER_REDUCE',
    'COMPAT_NOTES',
    'as_matrix',
    'as_tensor',
    'to_numpy',
    'scatter_min_',
    'require',
]


def _parse_version(raw: str) -> tuple[int, ...]:
    """Extracts the leading numeric components of a version string (``'2.4.1+cu121'`` -> ``(2, 4, 1)``)."""
    parts: list[int] = []
    for chunk in raw.split('+')[0].split('.'):
        digits = ''
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


TORCH_VERSION: Final[tuple[int, ...]] = _parse_version(torch.__version__)

_MIN_TORCH: Final[tuple[int, int]] = (1, 13)

if TORCH_VERSION[:2] < _MIN_TORCH:
    raise ImportError(
        f'alfspy requires torch >= {_MIN_TORCH[0]}.{_MIN_TORCH[1]} '
        f'(found {torch.__version__}). The rasteriser depends on Tensor.scatter_reduce_, '
        f'which stabilised in 1.13.'
    )

COMPAT_NOTES: list[str] = []


def _probe_scatter_reduce() -> bool:
    """
    Verifies that ``Tensor.scatter_reduce_(dim, index, src, 'amin', include_self=False)``
    behaves as the rasteriser needs: per-index minimum, ignoring the destination's initial
    values for indices that receive at least one source element.
    """
    try:
        dst = torch.zeros(3, dtype=torch.int64)
        idx = torch.tensor([0, 0, 2, 2, 2], dtype=torch.int64)
        src = torch.tensor([7, 3, 9, 4, 5], dtype=torch.int64)
        dst.scatter_reduce_(0, idx, src, reduce='amin', include_self=False)
        return bool(torch.equal(dst, torch.tensor([3, 0, 4], dtype=torch.int64)))
    except (AttributeError, TypeError, RuntimeError) as exc:  # pragma: no cover - version dependent
        COMPAT_NOTES.append(f'scatter_reduce_ probe failed ({exc!r}); using sort-based fallback')
        return False


HAS_SCATTER_REDUCE: Final[bool] = _probe_scatter_reduce()

if not HAS_SCATTER_REDUCE and not COMPAT_NOTES:  # pragma: no cover - version dependent
    COMPAT_NOTES.append('scatter_reduce_ returned unexpected values; using sort-based fallback')


def scatter_min_(dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """
    Scatters ``src`` into ``dst`` at ``index`` keeping the minimum per destination slot.

    ``dst`` is expected to be pre-filled with a sentinel that is larger than every value in
    ``src`` (the rasteriser uses ``iinfo(int64).max``), which makes ``include_self=True``
    and ``include_self=False`` equivalent and lets the fallback stay simple.

    :param dst: The 1D destination tensor, modified in place.
    :param index: The flat destination indices, same shape as ``src``.
    :param src: The values to scatter.
    :return: ``dst``, for chaining.
    """
    if index.numel() == 0:
        return dst

    if HAS_SCATTER_REDUCE:
        dst.scatter_reduce_(0, index, src, reduce='amin', include_self=True)
        return dst

    # Fallback: sort by (index, value) so the first entry of each index run is its minimum,
    # then write only those firsts. Deterministic and allocation-heavy but correct.
    order = torch.argsort(src, stable=True)
    idx_sorted = index[order]
    src_sorted = src[order]
    order2 = torch.argsort(idx_sorted, stable=True)
    idx_sorted = idx_sorted[order2]
    src_sorted = src_sorted[order2]

    keep = torch.ones_like(idx_sorted, dtype=torch.bool)
    keep[1:] = idx_sorted[1:] != idx_sorted[:-1]
    winners_idx = idx_sorted[keep]
    winners_val = src_sorted[keep]
    dst[winners_idx] = torch.minimum(dst[winners_idx], winners_val)
    return dst


def as_matrix(matrix: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Converts a pyrr ``Matrix44`` (or anything array-like) into a ``(4, 4)`` torch tensor
    **without transposing**.

    The port keeps matrices in pyrr's row-vector convention throughout, so points are
    transformed as ``v_row @ M``. See ``docs/pyrr_conventions.md``.

    :param matrix: The matrix to convert.
    :param device: The device to place the tensor on.
    :param dtype: The floating point dtype to use.
    :return: A ``(4, 4)`` tensor.
    """
    arr = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    return torch.as_tensor(arr, device=device, dtype=dtype)


def as_tensor(data: Any, device: torch.device, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    Converts array-like data into a tensor on the given device, copying only when needed.

    :param data: The data to convert.
    :param device: The device to place the tensor on.
    :param dtype: The dtype to use (optional; inferred when omitted).
    :return: A tensor holding the given data.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype) if dtype is not None else data.to(device=device)
    arr = np.ascontiguousarray(data)
    if arr.dtype == np.float16:
        arr = arr.astype(np.float32)
    return torch.as_tensor(arr, device=device, dtype=dtype)


def to_numpy(tensor: torch.Tensor) -> NDArray:
    """
    Moves a tensor to host memory and returns it as a numpy array.

    :param tensor: The tensor to convert.
    :return: A numpy array sharing no storage with the tensor.
    """
    return tensor.detach().to('cpu').numpy()


def require(condition: bool, message: str) -> None:
    """
    Raises a ``RuntimeError`` with an actionable message when ``condition`` is falsy.

    Used for invariants whose violation means a torch behaviour change rather than a bug in
    the caller's data.

    :param condition: The condition that must hold.
    :param message: The message to raise.
    """
    if not condition:
        raise RuntimeError(message)
