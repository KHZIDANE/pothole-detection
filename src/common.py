"""Shared helpers.

Roboflow / YOLO-format datasets ship a data.yaml whose paths (for example
"../train/images") often break depending on where you run training from.
resolve_data_yaml() writes a data_resolved.yaml with absolute, correct paths,
so every script in this repo uses the same, reliable config.
"""
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": ["train"], "val": ["valid", "val"], "test": ["test"]}


def find_split_dir(root, split):
    """Return <root>/<split>/images if it exists, else None."""
    root = Path(root)
    for name in SPLIT_ALIASES[split]:
        d = root / name / "images"
        if d.is_dir():
            return d
    return None


def list_images(images_dir):
    return sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMG_EXT)


def label_path_for(image_path):
    """YOLO layout: .../images/x.jpg  ->  .../labels/x.txt"""
    image_path = Path(image_path)
    return image_path.parent.parent / "labels" / (image_path.stem + ".txt")


def read_boxes(label_file):
    """Return a list of (class_id, cx, cy, w, h), all normalised to 0-1."""
    label_file = Path(label_file)
    if not label_file.exists():
        return []
    boxes = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            boxes.append((int(float(parts[0])), *map(float, parts[1:5])))
    return boxes


def class_names(cfg):
    names = cfg["names"]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names)


def resolve_data_yaml(root):
    """Write <root>/data_resolved.yaml with absolute paths and return its path."""
    root = Path(root).resolve()
    src = root / "data.yaml"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Download a YOLO-format dataset and unzip it into "
            f"'{root}' (see README, step 2)."
        )
    cfg = yaml.safe_load(src.read_text())
    out = {"path": str(root), "names": cfg["names"]}
    for split in ("train", "val", "test"):
        d = find_split_dir(root, split)
        if d is not None:
            out[split] = str(d.relative_to(root))
    if "train" not in out or "val" not in out:
        raise RuntimeError(
            f"Expected train/images and valid/images folders under {root}. Found: {sorted(out)}"
        )
    dst = root / "data_resolved.yaml"
    dst.write_text(yaml.safe_dump(out, sort_keys=False))
    return dst
