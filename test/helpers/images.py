"""Tolerance-based image comparison.

Bare ``np.array_equal`` on rendered output makes a suite useless across platforms and torch
builds. Everything here reports *why* a comparison failed -- mean absolute error, worst
pixel, and how much of the image is out of tolerance -- so a failure is diagnosable without
re-running under a debugger.
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray

__all__ = ['image_metrics', 'assert_images_close', 'coverage', 'describe']


def _as_float(img: NDArray) -> NDArray:
    return np.asarray(img).astype(np.float64)


def image_metrics(actual: NDArray, expected: NDArray, tolerance: float = 1.0) -> dict:
    """
    Computes comparison metrics for two same-shaped images.

    :param actual: The image produced by the code under test.
    :param expected: The reference image.
    :param tolerance: The per-channel absolute deviation considered acceptable.
    :return: A dict with ``mae``, ``max_abs``, ``bad_fraction`` and ``rmse``.
    """
    a, e = _as_float(actual), _as_float(expected)
    if a.shape != e.shape:
        raise AssertionError(f'Shape mismatch: actual {a.shape} vs expected {e.shape}')

    diff = np.abs(a - e)
    return {
        'mae': float(diff.mean()),
        'rmse': float(np.sqrt((diff ** 2).mean())),
        'max_abs': float(diff.max()) if diff.size else 0.0,
        'bad_fraction': float((diff > tolerance).mean()) if diff.size else 0.0,
    }


def assert_images_close(actual: NDArray, expected: NDArray, tolerance: float = 1.0,
                        max_bad_fraction: float = 0.0, label: str = 'image') -> None:
    """
    Asserts two images agree within tolerance.

    :param actual: The image produced by the code under test.
    :param expected: The reference image.
    :param tolerance: Acceptable per-channel absolute deviation (defaults to one 8-bit LSB).
    :param max_bad_fraction: Fraction of pixels allowed to exceed ``tolerance``. Use a small
        non-zero value when comparing anti-aliased or resampled edges.
    :param label: A name used in the failure message.
    """
    metrics = image_metrics(actual, expected, tolerance)
    if metrics['bad_fraction'] > max_bad_fraction:
        raise AssertionError(
            f'{label} differs beyond tolerance {tolerance}: '
            f"{metrics['bad_fraction']:.4%} of values off (allowed {max_bad_fraction:.4%}), "
            f"mae={metrics['mae']:.4f}, rmse={metrics['rmse']:.4f}, max={metrics['max_abs']:.4f}"
        )


def coverage(img: NDArray, channel: int = 3, threshold: float = 0.0) -> float:
    """
    :param img: An image with an alpha channel.
    :param channel: The channel to measure (defaults to alpha).
    :param threshold: The value a pixel must exceed to count as covered.
    :return: The fraction of pixels considered covered.
    """
    return float((np.asarray(img)[..., channel] > threshold).mean())


def describe(img: NDArray, name: str = 'image') -> str:
    """
    :param img: The image to describe.
    :param name: A name for the image.
    :return: A one-line summary useful in assertion messages.
    """
    arr = np.asarray(img)
    return (f'{name}: shape={arr.shape} dtype={arr.dtype} '
            f'min={arr.min() if arr.size else "-"} max={arr.max() if arr.size else "-"} '
            f'coverage={coverage(arr):.3f}' if arr.ndim == 3 and arr.shape[-1] >= 4
            else f'{name}: shape={arr.shape} dtype={arr.dtype}')
