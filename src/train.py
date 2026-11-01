"""Fine-tune a pretrained YOLO model (transfer learning, chapter 14 style).

Examples:
    # baseline: small pretrained model, all layers trainable
    python src/train.py --data dataset --model yolo11n.pt --epochs 60 --name n_baseline

    # transfer-learning experiment: freeze the first 10 layers (the backbone)
    python src/train.py --data dataset --model yolo11n.pt --freeze 10 --name n_freeze10

    # bigger model
    python src/train.py --data dataset --model yolo11s.pt --name s_baseline

Use --device 0 on a GPU machine (Colab/Kaggle). Leave it empty to auto-select.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from common import resolve_data_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--model", default="yolo11n.pt", help="pretrained weights to start from")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="", help="'' = auto, '0' = first GPU, 'cpu'")
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--freeze", type=int, default=0, help="freeze the first N layers")
    ap.add_argument("--patience", type=int, default=15, help="early stopping patience")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", action="store_true", help="cache images in RAM (faster epochs)")
    args = ap.parse_args()

    data_yaml = resolve_data_yaml(args.data)
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project="runs",
        name=args.name,
        seed=args.seed,
        patience=args.patience,
        workers=args.workers,
        cache=args.cache,
        freeze=args.freeze or None,
        plots=True,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nBest weights: {best}")
    print("Next: python src/evaluate.py --weights", best)


if __name__ == "__main__":
    main()
