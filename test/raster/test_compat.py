"""Probes for the torch primitives the rasteriser depends on.

torch is intentionally unpinned (see ``docs/MIGRATION_REVIEW.md`` section 5). These tests are
the tripwire: if a future release changes ``scatter_reduce_`` or ``grid_sample`` semantics,
they fail with an explicit message instead of the renderer quietly producing different
images.
"""

import numpy as np
import pytest
import torch

from alfspy.core.torchgl import compat
from alfspy.core.torchgl.compat import HAS_SCATTER_REDUCE, TORCH_VERSION, as_matrix, scatter_min_
from alfspy.core.torchgl.texture import sample_texture


def test_torch_version_meets_floor():
    assert TORCH_VERSION[:2] >= (1, 13), (
        f'torch {torch.__version__} is below the supported floor of 1.13'
    )


def test_scatter_reduce_available():
    assert HAS_SCATTER_REDUCE, (
        'Tensor.scatter_reduce_(reduce="amin") did not behave as the rasteriser requires. '
        f'The sort-based fallback will be used instead. Notes: {compat.COMPAT_NOTES}'
    )


def test_scatter_min_resolves_per_index_minimum():
    sentinel = torch.iinfo(torch.int64).max
    dst = torch.full((5,), sentinel, dtype=torch.int64)
    index = torch.tensor([0, 0, 0, 3, 3], dtype=torch.int64)
    src = torch.tensor([9, 2, 7, 40, 11], dtype=torch.int64)

    scatter_min_(dst, index, src)

    assert dst[0].item() == 2
    assert dst[3].item() == 11
    assert dst[1].item() == sentinel, 'untouched slots must keep the sentinel'


def test_scatter_min_is_order_independent():
    """The depth resolve must not depend on the order fragments happen to be generated in."""
    sentinel = torch.iinfo(torch.int64).max
    index = torch.tensor([2, 2, 2, 2, 0, 0], dtype=torch.int64)
    src = torch.tensor([50, 8, 31, 99, 4, 4], dtype=torch.int64)

    first = torch.full((4,), sentinel, dtype=torch.int64)
    scatter_min_(first, index, src)

    permutation = torch.tensor([3, 0, 5, 2, 4, 1])
    second = torch.full((4,), sentinel, dtype=torch.int64)
    scatter_min_(second, index[permutation], src[permutation])

    assert torch.equal(first, second)


def test_scatter_min_empty_input_is_noop():
    dst = torch.zeros(3, dtype=torch.int64)
    scatter_min_(dst, torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64))
    assert torch.equal(dst, torch.zeros(3, dtype=torch.int64))


def test_sort_based_fallback_matches_scatter_reduce(monkeypatch):
    """The fallback path must agree with the fast path, since it is what ships on old torch."""
    sentinel = torch.iinfo(torch.int64).max
    generator = torch.Generator().manual_seed(11)
    index = torch.randint(0, 16, (500,), generator=generator, dtype=torch.int64)
    src = torch.randint(0, 10_000, (500,), generator=generator, dtype=torch.int64)

    fast = torch.full((16,), sentinel, dtype=torch.int64)
    scatter_min_(fast, index, src)

    monkeypatch.setattr(compat, 'HAS_SCATTER_REDUCE', False)
    slow = torch.full((16,), sentinel, dtype=torch.int64)
    compat.scatter_min_(slow, index, src)

    assert torch.equal(fast, slow), 'sort-based fallback disagrees with scatter_reduce_'


def test_as_matrix_does_not_transpose():
    """
    Matrices stay in pyrr's row-vector convention. Transposing here would silently misalign
    every projected shot; see docs/pyrr_conventions.md.
    """
    from pyrr import Matrix44, Vector3

    matrix = Matrix44.from_translation(Vector3([1.0, 2.0, 3.0]))
    tensor = as_matrix(matrix, torch.device('cpu'), torch.float64)

    np.testing.assert_allclose(tensor[3, :3].numpy(), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(tensor[:3, 3].numpy(), [0.0, 0.0, 0.0])

    point = torch.tensor([10.0, 20.0, 30.0, 1.0], dtype=torch.float64)
    np.testing.assert_allclose((point @ tensor).numpy(), [11.0, 22.0, 33.0, 1.0])


@pytest.mark.parametrize('mode', ['linear', 'nearest'])
def test_texture_sampling_hits_texel_centres(mode):
    """
    GL's normalised texture coordinates put texel *centres* at ``(i + 0.5) / size``. This is
    what ``align_corners=False`` buys us; getting it wrong shifts every projection by half a
    texel.
    """
    texture = torch.arange(16, dtype=torch.float32).reshape(4, 4, 1)

    centres = (torch.arange(4, dtype=torch.float32) + 0.5) / 4.0
    grid_u, grid_v = torch.meshgrid(centres, centres, indexing='xy')

    sampled = sample_texture(texture, grid_u.reshape(-1), grid_v.reshape(-1), mode)

    expected = texture.reshape(-1, 1)
    np.testing.assert_allclose(sampled.numpy(), expected.numpy(), atol=1e-5)


def test_texture_sampling_interpolates_linearly():
    texture = torch.tensor([[[0.0], [1.0]]], dtype=torch.float32)  # 1 row, 2 texels

    midpoint = torch.tensor([0.5])
    value = sample_texture(texture, midpoint, torch.tensor([0.5]), 'linear')

    np.testing.assert_allclose(value.numpy(), [[0.5]], atol=1e-6)


def test_texture_wrap_repeat_is_exact():
    """GL_REPEAT is fract(uv); no grid_sample padding mode reproduces it."""
    texture = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)

    inside = sample_texture(texture, torch.tensor([0.375]), torch.tensor([0.5]), 'nearest', 'repeat')
    wrapped = sample_texture(texture, torch.tensor([1.375]), torch.tensor([0.5]), 'nearest', 'repeat')
    negative = sample_texture(texture, torch.tensor([-0.625]), torch.tensor([0.5]), 'nearest', 'repeat')

    np.testing.assert_allclose(wrapped.numpy(), inside.numpy())
    np.testing.assert_allclose(negative.numpy(), inside.numpy())
