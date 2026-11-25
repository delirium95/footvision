"""Fine-tune YOLO on football classes.

Usage:
    python src/train.py --data configs/data.yaml --model yolo11m.pt --epochs 80

The ball is the hard class: small, fast, motion-blurred. Expect high mAP on
players and low mAP on the ball, and report both separately.
"""
import argparse

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--model", default="yolo11m.pt", help="pretrained weights to start from")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=1280, help="high res helps the ball")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="footvision")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=20,
        # a few extra augs on top of the ultralytics defaults
        hsv_v=0.5,
        degrees=5,
        mosaic=1.0,
        close_mosaic=10,
    )

    # Per-class metrics on the val split — this is what goes in the README table.
    metrics = model.val()
    print("\nmAP@50 overall:", round(float(metrics.box.map50), 4))
    for i, name in model.names.items():
        try:
            print(f"  {name:>11}: mAP@50 = {metrics.box.ap50[i]:.4f}")
        except (IndexError, TypeError):
            print(f"  {name:>11}: no val instances")


if __name__ == "__main__":
    main()
