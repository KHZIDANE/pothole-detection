"""Dataset sanity checks. Run this BEFORE training.

What it reports (and why it matters):
  * images / boxes / empty images per split
  * class balance per split
  * box size distribution (small objects are the hard ones for detectors)
  * image-size variety
  * LEAKAGE: the same source photo appearing in train AND val/test. Roboflow
    exports create several augmented copies of one photo ("name_jpg.rf.<hash>").
    If copies land in different splits, your test score is inflated.

Usage:
    python src/explore_dataset.py --data dataset
"""
import argparse
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

from common import class_names, find_split_dir, label_path_for, list_images, read_boxes


def source_key(path):
    """Group augmented copies of one photo: strip Roboflow's '.rf.<hash>' suffix."""
    parts = re.split(r"[._](?:jpg|jpeg|png)\.rf\.", path.name, flags=re.I)
    return parts[0] if len(parts) > 1 else path.stem


def file_md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset", help="folder containing data.yaml")
    ap.add_argument("--out", default="docs/dataset_report", help="where to save plots/report")
    args = ap.parse_args()

    root = Path(args.data)
    cfg = yaml.safe_load((root / "data.yaml").read_text())
    names = class_names(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    splits = {}
    for split in ("train", "val", "test"):
        d = find_split_dir(root, split)
        if d is not None:
            splits[split] = list_images(d)
    if not splits:
        raise SystemExit(f"No <split>/images folders found under {root}")

    lines = [f"# Dataset report for `{root}`", "", f"Classes ({len(names)}): {names}", ""]
    stats = {}
    for split, images in splits.items():
        class_count = Counter()
        areas, boxes_per_image = [], []
        missing_labels = empty = 0
        sizes = Counter()
        for img in images:
            lp = label_path_for(img)
            if not lp.exists():
                missing_labels += 1
            boxes = read_boxes(lp)
            if not boxes:
                empty += 1
            boxes_per_image.append(len(boxes))
            for cls, _, _, w, h in boxes:
                class_count[cls] += 1
                areas.append(w * h)
            with Image.open(img) as im:
                sizes[im.size] += 1
        stats[split] = dict(class_count=class_count, areas=np.array(areas),
                            boxes_per_image=boxes_per_image)
        lines += [
            f"## {split}",
            f"- images: {len(images)}",
            f"- boxes: {sum(class_count.values())}",
            f"- images with no boxes (background): {empty}",
            f"- images with no label file at all: {missing_labels}",
            f"- most common image sizes: {sizes.most_common(3)}",
        ]
        if len(areas):
            a = np.array(areas)
            lines += [
                f"- box area as % of image: median {np.median(a) * 100:.2f}%, "
                f"5th pct {np.percentile(a, 5) * 100:.3f}%, 95th pct {np.percentile(a, 95) * 100:.2f}%",
                f"- tiny boxes (<0.1% of image): {(a < 0.001).mean() * 100:.1f}%",
            ]
        lines.append("")

    # ---- leakage checks -------------------------------------------------
    lines.append("## Leakage check")
    keys = {s: defaultdict(list) for s in splits}
    hashes = {s: {} for s in splits}
    for s, images in splits.items():
        for img in images:
            keys[s][source_key(img)].append(img)
            hashes[s][file_md5(img)] = img
    names_list = list(splits)
    leak_found = False
    for i, a in enumerate(names_list):
        for b in names_list[i + 1:]:
            shared_src = set(keys[a]) & set(keys[b])
            shared_hash = set(hashes[a]) & set(hashes[b])
            if shared_src or shared_hash:
                leak_found = True
            lines.append(
                f"- {a} vs {b}: {len(shared_src)} shared source photos, "
                f"{len(shared_hash)} byte-identical files"
            )
    lines.append("")
    lines.append(
        "**WARNING: overlap found. Re-split by source photo before trusting any metric.**"
        if leak_found else "No overlap found between splits (by source name and exact bytes)."
    )
    lines.append("")
    lines.append("Note: this cannot catch near-duplicates of *different* files "
                 "(e.g. consecutive dashcam frames). Eyeball a few val images next to train.")

    # ---- plots ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    width = 0.8 / len(splits)
    for k, (split, st) in enumerate(stats.items()):
        counts = [st["class_count"].get(c, 0) for c in range(len(names))]
        axes[0].bar(np.arange(len(names)) + k * width, counts, width, label=split)
    axes[0].set_xticks(np.arange(len(names)) + width * (len(splits) - 1) / 2)
    axes[0].set_xticklabels(names, rotation=30, ha="right")
    axes[0].set_title("Boxes per class")
    axes[0].legend()

    all_areas = np.concatenate([st["areas"] for st in stats.values() if len(st["areas"])])
    if len(all_areas):
        axes[1].hist(np.log10(all_areas * 100), bins=40)
        axes[1].set_xlabel("log10(box area as % of image)")
        axes[1].set_title("Box size (left = small objects)")

    for split, st in stats.items():
        axes[2].hist(st["boxes_per_image"], bins=range(0, max(st["boxes_per_image"] + [1]) + 2),
                     alpha=0.6, label=split)
    axes[2].set_title("Boxes per image")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=120)

    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved {out / 'report.md'} and {out / 'overview.png'}")


if __name__ == "__main__":
    main()
