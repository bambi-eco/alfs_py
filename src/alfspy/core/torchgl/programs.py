"""The two GLSL programs of the ALFS renderer, expressed as torch functions.

Each function is a line-by-line translation of the shader it replaces; the original GLSL is
quoted above it so the correspondence stays auditable. Both consume
:class:`~alfspy.core.torchgl.raster.Fragments` produced by the rasteriser.
"""

from typing import Optional

import torch

from alfspy.core.torchgl.raster import Fragments, transform_vertices
from alfspy.core.torchgl.texture import TorchTexture

__all__ = ['shade_object', 'shot_clip_coords', 'shade_shot', 'shade_shot_projected']


def shade_object(fragments: Fragments, uvs: torch.Tensor, indices: torch.Tensor,
                 texture: Optional[TorchTexture]) -> torch.Tensor:
    """
    The ``OBJ`` program: plain textured rendering of the background mesh.

    Replaces::

        // vertex
        v_out_v2_uv = v_in_v2_uv.xy;
        gl_Position = u_m4_proj * u_m4_view * u_m4_model * vec4(v_in_v3_pos.xyz, 1.0);

        // fragment
        f_out_v4_color = texture(u_s2_tex, v_out_v2_uv);

    :param fragments: The rasterised fragments to shade.
    :param uvs: ``(V, 2)`` per-vertex UV coordinates.
    :param indices: ``(T, 3)`` triangle vertex indices.
    :param texture: The mesh texture (optional). When ``None`` the fragments are shaded
        opaque white, matching an unbound sampler on most drivers.
    :return: ``(N, 4)`` RGBA colours.
    """
    count = len(fragments)
    device = fragments.bary.device
    dtype = fragments.bary.dtype

    if count == 0:
        return torch.zeros((0, 4), device=device, dtype=dtype)

    if texture is None:
        return torch.ones((count, 4), device=device, dtype=dtype)

    uv = fragments.interpolate(uvs, indices)
    colour = texture.sample(uv[:, 0], uv[:, 1])
    return _to_rgba(colour, dtype)


def shot_clip_coords(world_positions: torch.Tensor, shot_view: torch.Tensor,
                     shot_correction: torch.Tensor, shot_proj: torch.Tensor) -> torch.Tensor:
    """
    Computes the per-vertex varying of the ``SHOT`` vertex shader.

    Replaces::

        v_out_v4_shot_uv = u_m4_shot_proj * u_m4_shot_correction * u_m4_shot_view * world_pos;

    Note the multiplication order: matrices are in pyrr's row-vector convention, so the GLSL
    product ``P * C * V * p`` becomes ``p_row @ V @ C @ P``. The correction sits *between*
    projection and view -- getting that wrong silently misaligns every projected shot. See
    ``docs/pyrr_conventions.md``.

    :param world_positions: ``(V, 3)`` or ``(V, 4)`` world-space vertex positions.
    :param shot_view: The shot camera's ``(4, 4)`` view matrix.
    :param shot_correction: The shot's ``(4, 4)`` correction matrix.
    :param shot_proj: The shot camera's ``(4, 4)`` projection matrix.
    :return: ``(V, 4)`` clip-space coordinates in the shot camera's frustum.
    """
    matrix = shot_view @ shot_correction @ shot_proj
    return transform_vertices(world_positions, matrix)


def shade_shot(fragments: Fragments, shot_clip: torch.Tensor, indices: torch.Tensor,
               texture: TorchTexture, mask: Optional[TorchTexture] = None,
               reject_behind_camera: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The ``SHOT`` program: projective texture mapping of a source frame onto the mesh.

    Replaces::

        vec4 uv = v_out_v4_shot_uv;
        uv = vec4(uv.xyz / uv.w / 2.0 + .5, 1.0);
        if (uv.w <= 0.0 || uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            discard;
        } else {
            f_out_v4_color = vec4(texture(u_s2_tex, uv.xy).rgba);
            if (u_f_mask > 0.0) f_out_v4_color.rgba *= texture(u_s2_mask, uv.xy).r;
        }

    The ``uv.w <= 0.0`` guard in the original is dead code: ``uv`` has just been rebuilt with
    ``w = 1.0``, so points behind the shot camera are never rejected by it. That behaviour is
    reproduced by default; pass ``reject_behind_camera=True`` to apply the test the shader
    evidently intended, against the pre-divide ``w``.

    :param fragments: The rasterised fragments to shade.
    :param shot_clip: ``(V, 4)`` per-vertex shot clip coordinates from :func:`shot_clip_coords`.
    :param indices: ``(T, 3)`` triangle vertex indices.
    :param texture: The source frame texture.
    :param mask: The projection mask (optional). Its red channel scales the colour.
    :param reject_behind_camera: Whether to reject fragments behind the shot camera.
    :return: A tuple of ``(colours, keep)`` where ``colours`` is ``(M, 4)`` RGBA and ``keep``
        is an ``(M,)`` index tensor selecting the surviving fragments (the shader's
        ``discard``).
    """
    if len(fragments) == 0:
        device = fragments.bary.device
        dtype = fragments.bary.dtype
        return (torch.zeros((0, 4), device=device, dtype=dtype),
                torch.zeros(0, device=device, dtype=torch.int64))

    # Varyings interpolate before the perspective divide, so the divide is per fragment.
    clip = fragments.interpolate(shot_clip, indices)                    # (N, 4)
    return shade_shot_projected(clip, texture, mask, reject_behind_camera)


def shade_shot_projected(clip: torch.Tensor, texture: TorchTexture,
                         mask: Optional[TorchTexture] = None,
                         reject_behind_camera: bool = False,
                         return_weight: bool = False):
    """
    The fragment half of the ``SHOT`` program, taking clip coordinates already in fragment space.

    :func:`shade_shot` interpolates the per-vertex varying and then calls this. The renderer
    calls it directly, because the interpolation can be hoisted out of the per-shot loop:
    barycentric interpolation and the shot matrix are *both linear*, so

        interpolate(world @ M) == interpolate(world) @ M

    Interpolating the world position once per rasterisation and multiplying by each shot's
    matrix therefore gives the same clip coordinates as re-interpolating a fresh per-vertex
    varying for every shot -- but it replaces an ``(N, 3, 4)`` gather per shot with an
    ``(N, 4) @ (4, 4)`` matmul. The two differ only in floating-point rounding order.

    :param clip: ``(N, 4)`` per-fragment coordinates in the shot camera's clip space.
    :param texture: The source frame texture.
    :param mask: The projection mask (optional). Its red channel scales the colour.
    :param reject_behind_camera: Whether to reject fragments behind the shot camera.
    :param return_weight: Whether to also return the per-fragment mask weight. The weight is
        the coverage contribution of this shot -- ``1`` for an unmasked fragment, the mask's
        red channel otherwise. It is already folded into ``colours``; returning it separately
        is what lets the caller accumulate coverage without inferring it from an alpha
        channel that an N-channel field needs for data.
    :return: ``(colours, keep)``, or ``(colours, keep, weights)`` when ``return_weight``.
        ``colours`` is ``(M, 4)`` RGBA and ``keep`` is an ``(M,)`` index tensor selecting the
        surviving fragments.
    """
    device = clip.device
    dtype = clip.dtype

    if clip.shape[0] == 0:
        empty = (torch.zeros((0, 4), device=device, dtype=dtype),
                 torch.zeros(0, device=device, dtype=torch.int64))
        return empty + (torch.zeros((0, 1), device=device, dtype=dtype),) if return_weight else empty

    w = clip[:, 3]
    safe_w = torch.where(w.abs() < 1e-12, torch.ones_like(w), w)

    # Only x and y are ever sampled or tested; the shader's uv.z is dead.
    uv = clip[:, :2] / safe_w.unsqueeze(-1) / 2.0 + 0.5

    inside = (uv[:, 0] >= 0.0) & (uv[:, 0] <= 1.0) & (uv[:, 1] >= 0.0) & (uv[:, 1] <= 1.0)
    if reject_behind_camera:
        inside = inside & (w > 0.0)

    # An index tensor rather than a boolean mask: `index_select` is markedly faster than
    # advanced indexing, and the caller needs the same selection for its pixel indices.
    keep = torch.nonzero(inside, as_tuple=False).squeeze(1)
    if keep.numel() == 0:
        empty = torch.zeros((0, 4), device=device, dtype=dtype)
        if return_weight:
            return empty, keep, torch.zeros((0, 1), device=device, dtype=dtype)
        return empty, keep

    kept_uv = uv.index_select(0, keep)
    colour = _to_rgba(texture.sample(kept_uv[:, 0], kept_uv[:, 1]), dtype)

    if mask is not None:
        weight = mask.sample(kept_uv[:, 0], kept_uv[:, 1])[:, 0:1]
        colour = colour * weight
    else:
        weight = torch.ones((colour.shape[0], 1), device=device, dtype=dtype)

    if return_weight:
        return colour, keep, weight
    return colour, keep


def _to_rgba(colour: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Expands a sampled colour to four channels the way a GLSL sampler does.

    GL fills missing components with ``(0, 0, 0, 1)``: a one-channel texture reads as
    ``(r, 0, 0, 1)`` and a three-channel one as ``(r, g, b, 1)``.

    :param colour: ``(N, C)`` sampled values with ``C`` in ``1..4``.
    :param dtype: The dtype of the result.
    :return: ``(N, 4)`` RGBA values.
    """
    count, channels = colour.shape
    if channels == 4:
        return colour.to(dtype)

    out = torch.zeros((count, 4), device=colour.device, dtype=dtype)
    out[:, :channels] = colour.to(dtype)
    out[:, 3] = 1.0
    return out
