import sys
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray


class UpsampleAnything:
    """
    Thin wrapper around the UPA (Upsample Anything) function from
    https://github.com/seominseok0429/Upsample-Anything_Pytorch.

    UPA performs guided feature upsampling via test-time optimisation (~51 steps).
    It accepts an arbitrary-channel feature map and a high-resolution RGB guidance
    image, and returns the feature map upsampled to the guidance resolution.

    Constraint: ``guidance_H / feature_H`` and ``guidance_W / feature_W`` must be
    the same integer (the upsampling scale factor).
    """

    def __init__(self, upa_repo_dir: str):
        """
        :param upa_repo_dir: Path to the cloned Upsample-Anything_Pytorch repository.
            Its ``src/`` subdirectory is added to ``sys.path`` so that
            ``upsample_anything.py`` can be imported.
        """
        src_dir = str(Path(upa_repo_dir) / 'src')
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        from upsample_anything import UPA  # noqa: PLC0415
        self._UPA = UPA

    def upsample(self, embedding: NDArray, guidance_rgb: NDArray) -> NDArray:
        """
        Upsamples a dense feature map guided by a high-resolution RGB image.

        :param embedding: Low-resolution feature map as ``(H_e, W_e, C)`` float32
            numpy array (CPU).
        :param guidance_rgb: High-resolution guidance image as ``(H_g, W_g, 3)``
            uint8 numpy array in **RGB** order.  ``H_g`` must equal ``H_e * scale``
            and ``W_g`` must equal ``W_e * scale`` for some integer ``scale``.
        :return: Upsampled feature map as ``(H_g, W_g, C)`` float32 numpy array
            (CPU).
        """
        H_e, W_e, C = embedding.shape
        H_g, W_g = guidance_rgb.shape[:2]

        scale_h = H_g / H_e
        scale_w = W_g / W_e
        if scale_h != scale_w or not scale_h.is_integer():
            raise ValueError(
                f'UPA requires an integer scale factor: guidance ({H_g}x{W_g}) / '
                f'embedding ({H_e}x{W_e}) = ({scale_h:.3f}, {scale_w:.3f})'
            )

        # [1, C, H_e, W_e] CUDA tensor
        lr = torch.from_numpy(embedding).permute(2, 0, 1).unsqueeze(0).cuda()

        # UPA handles the HR image conversion internally (numpy → float CUDA tensor)
        hr_tensor = self._UPA(guidance_rgb, lr)  # [1, C, H_g, W_g]

        return hr_tensor[0].permute(1, 2, 0).float().cpu().numpy()

    @staticmethod
    def make_guidance_rgb(img_bgr: NDArray, target_h: int, target_w: int) -> NDArray:
        """
        Converts a BGR image (as loaded by OpenCV) to a uint8 RGB guidance image
        resized to ``(target_h, target_w)``.

        :param img_bgr: Source image in BGR format.
        :param target_h: Target height.
        :param target_w: Target width.
        :return: ``(target_h, target_w, 3)`` uint8 RGB numpy array.
        """
        if img_bgr.dtype != np.uint8:
            img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)

        # drop alpha if present
        if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
            img_bgr = img_bgr[..., :3]

        h, w = img_bgr.shape[:2]
        if h != target_h or w != target_w:
            img_bgr = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
