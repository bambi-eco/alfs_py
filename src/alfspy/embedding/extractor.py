"""Dense DINOv3 patch embeddings.

Turns a source frame into the per-patch feature map that an embedded light field integrates.
The frame is divided into patches (16x16 for the ``/16`` variants), and each patch gets one
descriptor -- so the output is a grid ``patch_size`` times coarser than the input, which is
exactly what the field renderer wants: it uploads at that resolution and lets the GPU's
bilinear sampling handle the rest.

Requires ``pip install "AlfsPy[embedding]"``.
"""

import os
from typing import List, Optional, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

__all__ = ['DinoV3Extractor', 'DEFAULT_MODEL', 'is_available']

#: The model the original embedded-light-field work used: ViT-H+/16, 1280-dimensional.
#: Roughly 840M parameters. ``facebook/dinov3-vits16-pretrain-lvd1689m`` (384-d) is a far
#: cheaper choice when the descriptor width matters more than its quality.
DEFAULT_MODEL = 'facebook/dinov3-vith16plus-pretrain-lvd1689m'

# DINOv3 prepends a CLS token and four register tokens before the patch tokens. Read from
# the model config where it is exposed; this is the fallback for configs that omit it.
_FALLBACK_REGISTER_TOKENS = 4


def is_available() -> bool:
    """
    :return: Whether the optional embedding dependencies are installed.
    """
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


class DinoV3Extractor:
    """
    Extracts dense DINOv3 patch embeddings from images.

    :cvar model_id: What was loaded -- a HuggingFace id or a local directory.
    """

    def __init__(self, model: str = DEFAULT_MODEL, device: Optional[str] = None,
                 dtype: Optional[str] = None, token: Optional[str] = None,
                 local_files_only: bool = False):
        """
        :param model: A HuggingFace model id or a path to a local model directory. Ids are
            downloaded and cached by ``huggingface_hub`` on first use.
        :param device: The torch device to run on (optional). Defaults to CUDA when visible.
        :param dtype: The compute dtype, e.g. ``"float16"`` (optional). Defaults to float32.
            Half precision roughly halves the memory a large model needs on the GPU.
        :param token: A HuggingFace access token for gated repositories (optional). Falls
            back to ``$HF_TOKEN``/``$HUGGINGFACE_HUB_TOKEN``, then to a cached login. Prefer
            the environment variable: a token passed here is easy to commit by accident.
        :param local_files_only: Load strictly from the local cache, never touching the
            network. Worth setting on a render machine with no outbound access, and needed
            for a cached *gated* model when no token is present -- transformers otherwise
            probes the hub for optional config files and a gated repo answers 401.
        :raises ImportError: If the optional embedding dependencies are missing.
        """
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                'DinoV3Extractor needs torch and transformers. '
                'Install them with `pip install "AlfsPy[embedding]"`.'
            ) from exc

        self._torch = torch
        self.model_id = model

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self._dtype = getattr(torch, dtype) if dtype else torch.float32

        token = token or os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
        options = {'local_files_only': local_files_only}
        if token:
            options['token'] = token

        self._processor = AutoImageProcessor.from_pretrained(model, **options)
        self._model = AutoModel.from_pretrained(model, **options)
        self._model.eval()
        self._model.to(device=device, dtype=self._dtype)

        config = self._model.config
        self._embed_dim = int(config.hidden_size)
        self._patch_size = int(getattr(config, 'patch_size', 16))
        registers = int(getattr(config, 'num_register_tokens', _FALLBACK_REGISTER_TOKENS))
        # One CLS token plus the registers, all before the first patch token.
        self._prefix_tokens = 1 + registers

    @property
    def embed_dim(self) -> int:
        """
        :return: The descriptor width, and therefore the channel count of the field.
        """
        return self._embed_dim

    @property
    def patch_size(self) -> int:
        """
        :return: The patch edge length in pixels. The embedding grid is this much coarser
            than the input image.
        """
        return self._patch_size

    @property
    def prefix_tokens(self) -> int:
        """
        :return: How many non-patch tokens the sequence starts with (CLS plus registers).
        """
        return self._prefix_tokens

    def extract(self, image: NDArray) -> NDArray:
        """
        Extracts patch embeddings from one image.

        :param image: A BGR image as ``cv2.imread`` returns it -- ``(H, W)``, ``(H, W, 3)``
            or ``(H, W, 4)``, uint8 or float.
        :return: An ``(H // patch, W // patch, embed_dim)`` float32 array.
        """
        return self.extract_batch([image])[0]

    def extract_batch(self, images: Sequence[NDArray]) -> List[NDArray]:
        """
        Extracts patch embeddings for several images.

        Images of the same size go through the model in one batch, which is most of the win
        on a GPU; differently sized ones are grouped and batched per size.

        :param images: BGR images.
        :return: One ``(h, w, embed_dim)`` float32 array per input, in the input order.
        """
        torch = self._torch
        prepared = [self._prepare(img) for img in images]

        groups = {}
        for index, img in enumerate(prepared):
            groups.setdefault(img.shape[:2], []).append(index)

        results: List[Optional[NDArray]] = [None] * len(prepared)

        for (height, width), indices in groups.items():
            batch = [prepared[i] for i in indices]
            inputs = self._processor(images=batch, return_tensors='pt', do_resize=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to(self._dtype)

            with torch.no_grad():
                outputs = self._model(**inputs)

            tokens = outputs.last_hidden_state[:, self._prefix_tokens:, :]
            rows, cols = height // self._patch_size, width // self._patch_size

            expected = rows * cols
            if tokens.shape[1] != expected:
                raise RuntimeError(
                    f'Expected {expected} patch tokens for a {height}x{width} image at patch '
                    f'size {self._patch_size}, got {tokens.shape[1]}. The model prepends '
                    f'{self._prefix_tokens} non-patch tokens; if this model differs, that is '
                    'the assumption to check.')

            grids = tokens.reshape(len(indices), rows, cols, self._embed_dim)
            grids = grids.float().cpu().numpy()
            for slot, index in enumerate(indices):
                results[index] = grids[slot]

        return results

    def _prepare(self, image: NDArray) -> NDArray:
        """
        Converts to RGB and resizes to a whole number of patches.

        Note the resize changes the image's effective focal length relative to the pose's
        recorded field of view. The change is at most half a patch on each edge -- under 1%
        for a 1024px frame at patch size 16 -- and the alternative, padding, would put
        meaningless descriptors around the border. It is worth knowing about when comparing
        an embedded field against an RGB one rendered from the same pose.

        :param image: The input image.
        :return: An RGB uint8 image whose dimensions are multiples of the patch size.
        """
        img = np.asarray(image)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = img[..., :3]

        if img.dtype != np.uint8:
            scale = 255.0 if img.max(initial=0.0) <= 1.0 else 1.0
            img = np.clip(img * scale, 0, 255).astype(np.uint8)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        height, width = rgb.shape[:2]
        patch = self._patch_size
        new_h = max(patch, int(round(height / patch)) * patch)
        new_w = max(patch, int(round(width / patch)) * patch)

        if (new_h, new_w) != (height, width):
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        return rgb

    def __repr__(self) -> str:
        return (f'<DinoV3Extractor {self.model_id!r} dim={self._embed_dim} '
                f'patch={self._patch_size} device={self.device!r}>')
