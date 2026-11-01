"""Gradio demo: upload a road photo, get potholes boxed.

Local:   python app.py
Weights: put your trained model at weights/best.pt (or set the WEIGHTS env var).
Deploy:  see README, "Deploy a free demo on Hugging Face Spaces".
"""
import os
from pathlib import Path

import gradio as gr
from ultralytics import YOLO

WEIGHTS = os.environ.get("WEIGHTS", "weights/best.pt")
if not Path(WEIGHTS).exists():
    raise SystemExit(f"Weights not found at '{WEIGHTS}'. Copy your best.pt there or set WEIGHTS=...")

model = YOLO(WEIGHTS)


def detect(image, conf):
    # Gradio gives RGB; Ultralytics treats numpy arrays as BGR, so flip both ways.
    result = model.predict(image[..., ::-1], conf=conf, verbose=False)[0]
    annotated = result.plot()[..., ::-1]
    n = len(result.boxes)
    if n == 0:
        return annotated, "No potholes detected at this confidence."
    scores = ", ".join(f"{c:.2f}" for c in result.boxes.conf.tolist())
    return annotated, f"{n} pothole(s) detected. Confidences: {scores}"


examples = sorted(str(p) for p in Path("examples").glob("*.jpg"))[:6] if Path("examples").is_dir() else None

demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="numpy", label="Road image"),
        gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold"),
    ],
    outputs=[gr.Image(label="Detections"), gr.Textbox(label="Summary")],
    title="Pothole detector (YOLO)",
    description=(
        "Fine-tuned YOLO model for pothole detection. Lower the threshold to catch more "
        "potholes (and more false alarms). Works best on dashcam-style road photos; "
        "wet roads, shadows and night images are known weak spots."
    ),
    examples=examples,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
