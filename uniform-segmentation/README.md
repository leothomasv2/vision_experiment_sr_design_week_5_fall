# Race-walker uniform segmentation

This proof of concept uses
[`mattmdjaga/segformer_b2_clothes`](https://huggingface.co/mattmdjaga/segformer_b2_clothes)
to separate each athlete's jersey and shorts. The checkpoint's `Upper-clothes`
class is treated as the jersey and its `Pants` class is treated as shorts.

## Setup and usage

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

With no input argument, all PNG and JPG files directly inside `examples/` are
processed. The model is loaded once and reused for the whole batch.

To process one image:

```bash
python main.py examples/page-1-image-1.png
```

To select another output location:

```bash
python main.py examples --output-dir /tmp/uniform-output
```

Every image gets its own output directory:

```text
output/<image-name>/
├── jersey.png
├── shorts.png
├── jersey_mask.png
├── shorts_mask.png
└── visualization.png
```

The cutouts preserve the original RGB pixels and use transparency outside the
mask. In `visualization.png`, green is the jersey prediction and blue is the
shorts prediction.

## Evaluation

The batch completed on all 10 example images without manual coordinates or
per-image settings. The observed CPU run took about 65 seconds including model
loading. This is a qualitative visual review, not a pixel-accuracy measurement;
there are no hand-labeled ground-truth masks.

- `page-1-image-1`: Good jersey and shorts coverage. Minor edge omissions.
- `page-1-image-2`: Partial jersey coverage. The shorts prediction includes
  substantial skin below the one-piece uniform and is not usable as-is.
- `page-2-image-1`: Jersey is mostly usable, but the model predicts only a
  small part of the one-piece uniform as shorts.
- `page-3-image-1`: Good jersey and shorts coverage with clean separation.
- `page-4-image-1`: Good overall. Small holes occur where arms overlap the
  jersey; shorts remain usable.
- `page-5-image-1`: Good jersey and shorts coverage with little contamination.
- `page-6-image-1`: Good coverage of both sections of the one-piece uniform.
- `page-7-image-1`: Good overall; the jersey boundary has a small omission
  around the exposed midriff.
- `page-8-image-1`: Shorts are good. The jersey mask includes several patches
  of exposed abdomen, so the jersey needs semantic correction.
- `page-9-image-1`: Good target-athlete coverage. A small prediction from the
  neighboring athlete appears near the left edge.

Qualitatively, 8 of 10 jersey masks and 8 of 10 shorts masks are usable without
manual correction. Standard two-piece uniforms perform best. The main failures
are one-piece racing suits, exposed skin classified as clothing, and nearby
athletes entering the mask.

## Mask cleanup decision

No morphological cleanup is applied. The failed regions are large, connected
semantic errors; erosion, dilation, or opening cannot distinguish them from
real fabric and would remove valid uniform pixels from the successful images.
The next useful improvement would be person-instance isolation or a model/class
mapping that explicitly supports one-piece athletic uniforms, followed by
evaluation against hand-labeled masks.
