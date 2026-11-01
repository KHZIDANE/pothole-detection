"""Generate a tiny SYNTHETIC 'pothole' dataset in YOLO format.

Purpose: smoke-test the whole pipeline (explore -> train -> evaluate -> video -> app)
in a few minutes, before you spend hours on the real dataset or a GPU session.
Do not report metrics from this dataset anywhere; it is not real data.

Usage:
    python src/make_toy_dataset.py --out dataset_toy
    python src/explore_dataset.py --data dataset_toy
    python src/train.py --data dataset_toy --epochs 3 --imgsz 320 --batch 8 --name toy
"""
import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter


def make_image(rng, size=320):
    base = rng.integers(90, 140)
    noise = rng.normal(0, 12, (size, size, 1)).repeat(3, axis=2)
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    boxes = []
    for _ in range(rng.integers(0, 4)):
        w, h = rng.integers(25, 90), rng.integers(18, 60)
        x0 = rng.integers(0, size - w)
        y0 = rng.integers(0, size - h)
        draw.ellipse([x0 - 3, y0 - 3, x0 + w + 3, y0 + h + 3], fill=(150, 150, 150))  # rim
        draw.ellipse([x0, y0, x0 + w, y0 + h], fill=(35, 35, 35))
        boxes.append(((x0 + w / 2) / size, (y0 + h / 2) / size, w / size, h / size))
    return im.filter(ImageFilter.GaussianBlur(0.8)), boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_toy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.out)
    counts = {"train": 120, "valid": 30, "test": 30}
    for split, n in counts.items():
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            im, boxes = make_image(rng)
            stem = f"{split}_{i:04d}"
            im.save(root / split / "images" / f"{stem}.jpg", quality=92)
            (root / split / "labels" / f"{stem}.txt").write_text(
                "\n".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cx, cy, w, h in boxes)
            )
    (root / "data.yaml").write_text(yaml.safe_dump({"names": ["pothole"], "nc": 1}))
    print(f"Toy dataset written to {root} ({sum(counts.values())} images)")


if __name__ == "__main__":
    main()
