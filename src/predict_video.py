"""Run the detector on a video (or webcam) and save an annotated mp4 + optional GIF.

The GIF is what you put at the top of your README / LinkedIn post.

Usage:
    python src/predict_video.py --weights weights/best.pt --source road.mp4 --out demo.mp4 --gif demo.gif
    python src/predict_video.py --weights weights/best.pt --source 0      # webcam
"""
import argparse
import time

import cv2
from PIL import Image
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="video path, or 0 for webcam")
    ap.add_argument("--out", default="demo.mp4")
    ap.add_argument("--gif", default="", help="optional GIF output path")
    ap.add_argument("--gif-fps", type=int, default=10)
    ap.add_argument("--gif-width", type=int, default=480)
    ap.add_argument("--gif-max-frames", type=int, default=120)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(0 if args.source == "0" else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))

    gif_step = max(1, round(src_fps / args.gif_fps))
    gif_frames, n, t_total = [], 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        r = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                          device=args.device or None, verbose=False)[0]
        t_total += time.perf_counter() - t0
        annotated = r.plot()  # BGR with boxes drawn
        n += 1
        cv2.putText(annotated, f"potholes: {len(r.boxes)}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        cv2.putText(annotated, f"potholes: {len(r.boxes)}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1)
        writer.write(annotated)
        if args.gif and n % gif_step == 0 and len(gif_frames) < args.gif_max_frames:
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb)
            im = im.resize((args.gif_width, int(h * args.gif_width / w)))
            gif_frames.append(im)
    cap.release()
    writer.release()

    if n:
        print(f"{n} frames, model time {t_total / n * 1000:.1f} ms/frame "
              f"({n / t_total:.1f} FPS, excludes video decode/encode)")
    print(f"Saved {args.out}")
    if args.gif and gif_frames:
        gif_frames[0].save(args.gif, save_all=True, append_images=gif_frames[1:],
                           duration=int(1000 / args.gif_fps), loop=0, optimize=True)
        print(f"Saved {args.gif} ({len(gif_frames)} frames)")


if __name__ == "__main__":
    main()
