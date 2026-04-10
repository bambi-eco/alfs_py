"""
Quick-test embedding script

Runs DINOv3 on a single image, optionally upsamples the result with
UpsampleAnything, saves the embedding as a .npy file, and optionally
launches visualize_embedding.py on it.

Usage:
    python src/embed_image.py image.jpg
    python src/embed_image.py image.jpg --output out.npy
    python src/embed_image.py image.jpg --upsample-scale 2
    python src/embed_image.py image.jpg --upsample-scale 4 --visualize
    python src/embed_image.py image.jpg --visualize --viz-method pca --viz-components 3
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from alfspy.embedding import DinoV3Extractor, UpsampleAnything


def embed_image(
    image_path: str,
    output_path: str,
    model_dir: str,
    upsample_scale: int = 0,
    upsample_to_input: bool = False,
    upa_repo_dir: str = r'D:\Upsample-Anything_Pytorch',
) -> str:
    """
    Extracts DINOv3 embeddings from *image_path* and saves them to *output_path*.

    :param image_path: Input image (any format readable by OpenCV).
    :param output_path: Destination .npy path.
    :param model_dir: Path to the local DINOv3 model directory.
    :param upsample_scale: Integer scale factor for UpsampleAnything.
        0 or 1 means no upsampling (raw patch-grid resolution is saved).
        Ignored when *upsample_to_input* is True.
    :param upsample_to_input: When True, upsample the patch-grid embedding back
        to the (rounded) input image resolution using patch_size as the scale
        factor.  Takes precedence over *upsample_scale*.
    :param upa_repo_dir: Path to the Upsample-Anything_Pytorch repo (only
        used when upsampling is enabled).
    :return: The resolved output path.
    """
    img_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    print(f"Image loaded: {img_bgr.shape}  ({image_path})")

    print(f"Loading DINOv3 from {model_dir} ...")
    extractor = DinoV3Extractor(model_dir)
    print(f"  embed_dim={extractor.embed_dim}, patch_size={extractor.patch_size}")

    print("Extracting embeddings ...")
    embedding = extractor.extract(img_bgr)
    print(f"  Raw embedding shape: {embedding.shape}")  # (H//16, W//16, D)

    if upsample_to_input:
        scale = extractor.patch_size
        H_e, W_e = embedding.shape[:2]
        target_h, target_w = H_e * scale, W_e * scale
        print(f"Upsampling to input size x{scale} ({target_h}x{target_w}) with UpsampleAnything ...")
        upsampler = UpsampleAnything(upa_repo_dir)
        guidance = UpsampleAnything.make_guidance_rgb(img_bgr, target_h, target_w)
        embedding = upsampler.upsample(embedding, guidance)
        print(f"  Upsampled embedding shape: {embedding.shape}")
    elif upsample_scale >= 2:
        print(f"Upsampling x{upsample_scale} with UpsampleAnything ...")
        upsampler = UpsampleAnything(upa_repo_dir)

        H_e, W_e = embedding.shape[:2]
        target_h = H_e * upsample_scale
        target_w = W_e * upsample_scale
        guidance = UpsampleAnything.make_guidance_rgb(img_bgr, target_h, target_w)
        embedding = upsampler.upsample(embedding, guidance)
        print(f"  Upsampled embedding shape: {embedding.shape}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embedding)
    print(f"Saved: {output_path}  shape={embedding.shape}  dtype={embedding.dtype}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract DINOv3 embeddings from a single image."
    )
    parser.add_argument("--image",
                        default=r"Z:\Hugo\alfs\images\train\101_962.jpg",
                        help="Input image path")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output .npy path (default: <image_stem>.npy next to the image)",
    )
    parser.add_argument(
        "--model-dir", default=r"D:\DINOv3",
        help="Path to the local DINOv3 model directory (default: D:\\DINOv3)",
    )
    parser.add_argument(
        "--upsample-scale", type=int, default=0, metavar="N",
        help="Integer scale factor for UpsampleAnything (0 = disabled). "
             "Ignored when --upsample-to-input is set.",
    )
    parser.add_argument(
        "--upsample-to-input", action="store_true",
        help="Upsample embeddings back to the input image resolution "
             "(scale = patch_size = 16). Takes precedence over --upsample-scale.",
    )
    parser.add_argument(
        "--upa-repo-dir", default=r"D:\Upsample-Anything_Pytorch",
        help="Path to the Upsample-Anything_Pytorch repo (used when upsampling is enabled)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Launch visualize_embedding.py on the saved .npy after embedding",
    )
    parser.add_argument(
        "--viz-method", choices=["pca", "umap", "tsne"], default="pca",
        help="Dimensionality reduction method passed to visualize_embedding.py (default: pca)",
    )
    parser.add_argument(
        "--viz-components", type=int, choices=[1, 2, 3], default=3,
        help="Number of components passed to visualize_embedding.py (default: 3)",
    )
    parser.add_argument(
        "--viz-save", metavar="OUTPUT_PNG", default=None,
        help="Save the visualization figure to this path",
    )
    parser.add_argument(
        "--no-show", dest="show", action="store_false",
        help="Do not open an interactive window (pass --no-show to visualize_embedding.py)",
    )
    args = parser.parse_args()
    args.visualize = True

    image_path = args.image
    if args.output is None:
        stem = Path(image_path).stem
        if args.upsample_to_input:
            suffix = "_upa_input"
        elif args.upsample_scale >= 2:
            suffix = f"_upa{args.upsample_scale}x"
        else:
            suffix = ""
        output_path = str(Path(image_path).parent / f"{stem}_embedding{suffix}.npy")
    else:
        output_path = args.output

    embed_image(
        image_path=image_path,
        output_path=output_path,
        model_dir=args.model_dir,
        upsample_scale=args.upsample_scale,
        upsample_to_input=args.upsample_to_input,
        upa_repo_dir=args.upa_repo_dir,
    )

    if args.visualize:
        viz_script = str(Path(__file__).parent / "visualize_embedding.py")
        cmd = [
            sys.executable, viz_script,
            "--npy", output_path,
            "--method", args.viz_method,
            "--components", str(args.viz_components),
            "--side-by-side", image_path,
        ]
        if args.viz_save:
            cmd += ["--save", args.viz_save]
        if not args.show:
            cmd.append("--no-show")
        print(f"\nLaunching visualizer: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
