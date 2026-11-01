# Pothole Detection with YOLO

Fine-tuning a pretrained YOLO model to detect potholes in road images and dashcam video,
with an honest evaluation, failure analysis, ONNX export benchmark, and a live demo.

> **Status:** scaffold. Fill the results section only with numbers you measured yourself.

<!-- Put your demo GIF here once you have it: ![demo](docs/demo.gif) -->

## Results (fill in after training)

| model | frozen layers | imgsz | mAP50 | mAP50-95 | precision | recall | CPU ms/img |
|---|---|---|---|---|---|---|---|
| yolo11n | 0 | 640 | | | | | |
| yolo11n | 10 | 640 | | | | | |
| yolo11s | 0 | 640 | | | | | |

Metrics are on a held-out **test** split that was never used for training or model selection.
Known failure modes (from `docs/eval/failures/`): _write what you actually observe_.

## Project layout

```
src/common.py            shared helpers (fixes dataset paths, reads YOLO labels)
src/explore_dataset.py   dataset report + leakage check (run before training)
src/make_toy_dataset.py  synthetic dataset for a 5-minute pipeline smoke test
src/train.py             fine-tune a pretrained YOLO (supports layer freezing)
src/evaluate.py          mAP/precision/recall + saved false-positive/miss images
src/predict_video.py     annotated video + README GIF
src/export_benchmark.py  ONNX export and CPU speed comparison
app.py                   Gradio demo
```

## 1. Setup

```bash
cd pothole-detection
source ../venv/bin/activate        # your existing venv with torch + ultralytics
pip install -r requirements.txt    # torch/torchvision are already installed
```

## 2. Smoke-test the whole pipeline first (5 minutes, CPU is fine)

```bash
python src/make_toy_dataset.py --out dataset_toy
python src/explore_dataset.py --data dataset_toy
python src/train.py --data dataset_toy --epochs 3 --imgsz 320 --batch 8 --name toy
python src/evaluate.py --weights runs/toy/weights/best.pt --data dataset_toy --split test
```

The toy data is synthetic. It only proves the code runs; never report its metrics.

## 3. Get a real dataset

Download a pothole dataset in **YOLO format** and unzip it so you have:

```
dataset/
  data.yaml
  train/images, train/labels
  valid/images, valid/labels
  test/images,  test/labels
```

Where to look: search "pothole" on Roboflow Universe (export as YOLOv11 / "YOLO" format), on
Kaggle, or use RDD2022 (road damage, multiple countries). **Check the license** before you
publish anything, and write the dataset name, source and license in this README.

Then:

```bash
python src/explore_dataset.py --data dataset
```

Read `docs/dataset_report/report.md`. If it reports overlap between splits, re-split by
source photo before training, otherwise your test score is inflated.

## 4. Train (use a GPU: Colab or Kaggle)

On your machine (CPU) a real run will be very slow. In Colab (Runtime -> GPU):

```python
!git clone https://github.com/<you>/pothole-detection.git
%cd pothole-detection
!pip install -q ultralytics
# upload/unzip your dataset into ./dataset, then:
!python src/train.py --data dataset --model yolo11n.pt --epochs 60 --device 0 --name n_baseline
```

Download `runs/n_baseline/weights/best.pt` to your machine and place it in `weights/best.pt`.
If `yolo11n.pt` is not found by your Ultralytics version, check the model names in the
Ultralytics docs and pass any detection checkpoint with `--model`.

## 5. Evaluate honestly

```bash
python src/evaluate.py --weights weights/best.pt --split test
```

Look at the images in `docs/eval/failures/`. Describe the patterns in your write-up.

## 6. Demo video and GIF

```bash
python src/predict_video.py --weights weights/best.pt --source road.mp4 --out demo.mp4 --gif docs/demo.gif
```

Use footage you recorded yourself or footage whose license allows reuse.

## 7. Export and speed

```bash
python src/export_benchmark.py --weights weights/best.pt --image path/to/road.jpg
```

## 8. Run the app

```bash
python app.py
```

The trained checkpoint and demo media are intentionally excluded from this
code-only repository. To run the app, place a compatible checkpoint at
`weights/best.pt` or set the `WEIGHTS` environment variable to its path.

## 9. Deploy a free demo on Hugging Face Spaces

1. Create a new Space (SDK: Gradio).
2. Upload `app.py`, `weights/best.pt`, and `requirements-space.txt` renamed to `requirements.txt`.
3. Optionally add a few example images in `examples/` (only images you have rights to share).

## Limitations

- Trained on one dataset; expect a performance drop on other cameras, countries, weather, night.
- Detection only: no depth or severity estimation.
- Not validated for any safety-critical use.

## Credits

- Ultralytics YOLO
- Dataset: _name, link, license_
