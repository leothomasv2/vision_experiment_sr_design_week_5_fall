
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

MODEL_ID = "mattmdjaga/segformer_b2_clothes"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = SCRIPT_DIR / "examples" / "page-1-image-1.png"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Separate a race walker's jersey and shorts with SegFormer B2 Clothes."
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"input image (default: {DEFAULT_IMAGE.relative_to(SCRIPT_DIR)})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"output directory (default: {DEFAULT_OUTPUT_DIR.relative_to(SCRIPT_DIR)})",
    )
    return parser.parse_args()


def class_id(model: SegformerForSemanticSegmentation, label: str) -> int:
    """Look up a class ID without relying on undocumented numeric constants."""
    normalized_label = label.lower().replace("-", "").replace("_", "")
    for index, name in model.config.id2label.items():
        normalized_name = name.lower().replace("-", "").replace("_", "")
        if normalized_name == normalized_label:
            return int(index)
    raise ValueError(f"Model does not contain a {label!r} class")


def save_cutout(image: Image.Image, mask: np.ndarray, path: Path) -> None:
    """Save original pixels with a transparent background."""
    cutout = image.copy()
    cutout.putalpha(Image.fromarray(mask))
    cutout.save(path)


def main() -> None:
    args = parse_args()
    image_path = args.image.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    model = model.to(device).eval()

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image)
    inputs = {
        key: value.to(device)
        for key, value in processor(images=image, return_tensors="pt").items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)
    segmentation = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[(image.height, image.width)]
    )[0].cpu().numpy()

    jersey_mask = (segmentation == class_id(model, "Upper-clothes")).astype(np.uint8) * 255
    # This checkpoint calls both trousers and shorts "Pants".
    shorts_mask = (segmentation == class_id(model, "Pants")).astype(np.uint8) * 255

    save_cutout(image, jersey_mask, output_dir / "jersey.png")
    save_cutout(image, shorts_mask, output_dir / "shorts.png")
    Image.fromarray(jersey_mask).save(output_dir / "jersey_mask.png")
    Image.fromarray(shorts_mask).save(output_dir / "shorts_mask.png")

    overlay = image_np.copy()
    overlay[jersey_mask > 0] = [0, 220, 80]
    overlay[shorts_mask > 0] = [0, 150, 255]
    visualization = (image_np * 0.6 + overlay * 0.4).astype(np.uint8)
    Image.fromarray(visualization).save(output_dir / "visualization.png")

    print(f"Input: {image_path}")
    print(f"Device: {device}")
    print(f"Jersey pixels: {np.count_nonzero(jersey_mask)}")
    print(f"Shorts pixels: {np.count_nonzero(shorts_mask)}")
    print(f"Images saved in: {output_dir}")


if __name__ == "__main__":
    main()
