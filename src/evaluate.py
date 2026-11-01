"""Evaluate on the held-out split and save failure cases.

Reports mAP50, mAP50-95, precision, recall (via Ultralytics), then runs its own
matching pass at a fixed confidence so you can SEE what goes wrong:
  green  = correct detection        red    = false positive (model invented a box)
  yellow = missed pothole (FN)

Usage:
    python src/evaluate.py --weights runs/n_baseline/weights/best.pt --split test
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from common import find_split_dir, label_path_for, list_images, read_boxes, resolve_data_yaml


def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def gt_to_xyxy(boxes, w, h):
    return [((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
            for _, cx, cy, bw, bh in boxes]


def match(preds, confs, gts, thr):
    """Greedy matching, highest confidence first. Returns (tp, fp, fn) as index lists."""
    used, tp, fp = set(), [], []
    for i in np.argsort(-confs):
        best_j, best_iou = -1, 0.0
        for j, g in enumerate(gts):
            if j in used:
                continue
            v = iou_xyxy(preds[i], g)
            if v > best_iou:
                best_j, best_iou = j, v
        if best_iou >= thr:
            used.add(best_j)
            tp.append(i)
        else:
            fp.append(i)
    fn = [j for j in range(len(gts)) if j not in used]
    return tp, fp, fn


def draw(img, boxes, color, thick=2):
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--conf", type=float, default=0.25, help="threshold for the failure analysis")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--max-failures", type=int, default=24)
    ap.add_argument("--out", default="docs/eval")
    args = ap.parse_args()

    data_yaml = resolve_data_yaml(args.data)
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    if args.split not in cfg:
        raise SystemExit(f"Dataset has no '{args.split}' split. Use --split val or re-export with a test set.")

    out = Path(args.out)
    (out / "failures").mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    # 1) Official metrics -------------------------------------------------
    m = model.val(data=str(data_yaml), split=args.split, plots=True,
                  project=str(out), name=f"val_{args.split}", exist_ok=True)
    summary = [
        f"# Evaluation on `{args.split}` split ({args.weights})", "",
        "| metric | value |", "|---|---|",
        f"| mAP50 | {m.box.map50:.3f} |",
        f"| mAP50-95 | {m.box.map:.3f} |",
        f"| precision | {m.box.mp:.3f} |",
        f"| recall | {m.box.mr:.3f} |", "",
    ]

    # 2) Failure analysis ---------------------------------------------------
    images = list_images(find_split_dir(cfg["path"], args.split))
    tp_n = fp_n = fn_n = 0
    scored = []
    for p in images:
        r = model.predict(str(p), conf=args.conf, verbose=False)[0]
        h, w = r.orig_shape
        preds = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        gts = gt_to_xyxy(read_boxes(label_path_for(p)), w, h)
        tp, fp, fn = match(preds, confs, gts, args.iou_thr)
        tp_n, fp_n, fn_n = tp_n + len(tp), fp_n + len(fp), fn_n + len(fn)
        if fp or fn:
            scored.append((len(fp) + len(fn), p, preds, tp, fp, gts, fn))

    prec = tp_n / max(tp_n + fp_n, 1)
    rec = tp_n / max(tp_n + fn_n, 1)
    summary += [
        f"At conf={args.conf}, IoU>={args.iou_thr}: TP={tp_n}, FP={fp_n}, FN={fn_n}, "
        f"precision={prec:.3f}, recall={rec:.3f}", "",
        f"{len(scored)} of {len(images)} images contain at least one error. "
        f"Worst {min(len(scored), args.max_failures)} saved in `{out / 'failures'}` "
        "(green = correct, red = false positive, yellow = missed).",
    ]
    scored.sort(key=lambda t: -t[0])
    for k, (_, p, preds, tp, fp, gts, fn) in enumerate(scored[: args.max_failures]):
        img = cv2.imread(str(p))
        draw(img, [preds[i] for i in tp], (0, 200, 0))
        draw(img, [preds[i] for i in fp], (0, 0, 255))
        draw(img, [gts[j] for j in fn], (0, 220, 255))
        cv2.imwrite(str(out / "failures" / f"{k:02d}_{p.stem}.jpg"), img)

    (out / "summary.md").write_text("\n".join(summary))
    print("\n".join(summary))


if __name__ == "__main__":
    main()
