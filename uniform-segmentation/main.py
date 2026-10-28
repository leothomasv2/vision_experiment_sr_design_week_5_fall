
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

MODEL_ID = "mattmdjaga/segformer_b2_clothes"
IMAGE_PATH = "athlete.jpg"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load model once, not once per image.
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
model = model.to(device).eval()

# Load athlete image.
image = Image.open(IMAGE_PATH).convert("RGB")
image_np = np.array(image)

# Predict segmentation labels.
inputs = processor(images=image, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.inference_mode():
    outputs = model(**inputs)

    logits = F.interpolate(
        outputs.logits,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )

    segmentation = logits.argmax(dim=1)[0].cpu().numpy()

# Class indices from model configuration.
upper_id = 4
pants_id = 6

upper_mask = (segmentation == upper_id).astype(np.uint8) * 255
pants_mask = (segmentation == pants_id).astype(np.uint8) * 255

def save_masked_image(mask, filename):
    # Original RGB values remain unchanged within mask.
    result = cv2.bitwise_and(image_np, image_np, mask=mask)
    Image.fromarray(result).save(OUTPUT_DIR / filename)

save_masked_image(upper_mask, "upper_clothes.png")
save_masked_image(pants_mask, "pants.png")

Image.fromarray(upper_mask).save(OUTPUT_DIR / "upper_mask.png")
Image.fromarray(pants_mask).save(OUTPUT_DIR / "pants_mask.png")

# Create an overlay for debugging.
overlay = image_np.copy()
overlay[upper_mask > 0] = [0, 220, 80]
overlay[pants_mask > 0] = [0, 180, 255]

visualization = cv2.addWeighted(
    image_np, 0.6, overlay, 0.4, 0
)

Image.fromarray(visualization).save(
    OUTPUT_DIR / "visualization.png"
)

print("Device:", device)
print("Upper clothes pixels:", np.count_nonzero(upper_mask))
print("Pants pixels:", np.count_nonzero(pants_mask))
print("Images saved in:", OUTPUT_DIR.resolve())
