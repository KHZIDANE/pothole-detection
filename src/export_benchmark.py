"""Export to ONNX and compare CPU inference speed: PyTorch vs ONNX Runtime.

Usage:
    python src/export_benchmark.py --weights weights/best.pt --image some_road.jpg
"""
import argparse
import statistics
import time
from pathlib import Path

from ultralytics import YOLO


def bench(model, image, imgsz, runs, warmup=5):
    for _ in range(warmup):
        model.predict(image, imgsz=imgsz, device="cpu", verbose=False)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(image, imgsz=imgsz, device="cpu", verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(times), statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True, help="any road image for timing")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--out", default="docs/export_benchmark.md")
    args = ap.parse_args()

    pt_path = Path(args.weights)
    onnx_path = Path(YOLO(args.weights).export(format="onnx", imgsz=args.imgsz, simplify=False))

    rows = []
    for label, path in (("PyTorch (.pt)", pt_path), ("ONNX Runtime (.onnx)", onnx_path)):
        mean_ms, med_ms = bench(YOLO(str(path), task="detect"), args.image, args.imgsz, args.runs)
        rows.append((label, path.stat().st_size / 1e6, mean_ms, med_ms, 1000 / mean_ms))

    table = ["| format | size (MB) | mean ms/img | median ms/img | FPS |", "|---|---|---|---|---|"]
    table += [f"| {l} | {s:.1f} | {m:.1f} | {md:.1f} | {f:.1f} |" for l, s, m, md, f in rows]
    text = (f"CPU benchmark, imgsz={args.imgsz}, {args.runs} runs after warmup "
            f"(includes pre/post-processing)\n\n" + "\n".join(table))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
