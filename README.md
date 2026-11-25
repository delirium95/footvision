# FootVision — player detection, tracking and movement heatmaps

Fine-tunes YOLO11 on football classes (`player`, `goalkeeper`, `referee`,
`ball`), tracks every player with a stable ID across a match clip, and turns
the tracks into a movement heatmap on a top-down pitch.

![demo](docs/demo.gif) <!-- raw clip → boxes + IDs → heatmap -->

## Results

| Class | mAP@50 |
|---|---|
| player | _fill in_ |
| goalkeeper | _fill in_ |
| referee | _fill in_ |
| ball | _fill in (expect this to be low)_ |

Tracking: BoT-SORT. Inference: _X_ FPS on _which GPU_.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Data

Any dataset in YOLO format works. Two easy starts:

- **Roboflow Universe** — search "football player detection", export in YOLO
  format. Fastest way to a first result.
- **SoccerNet** — larger, purpose-built for tracking (gated, needs sign-up).

Put it under `dataset/` as `images/{train,val}` + `labels/{train,val}`, or edit
`configs/data.yaml` to point elsewhere. Keep the class order in `data.yaml`.

## 2. Train

```bash
python src/train.py --data configs/data.yaml --model yolo11m.pt --epochs 80
```

Prints per-class mAP@50 at the end — copy it into the table above. The ball
scores low on purpose; say so rather than hiding it.

## 3. Track

```bash
python src/track.py --weights runs/detect/footvision/weights/best.pt \
    --source match.mp4 --out out.mp4 --tracks tracks.jsonl
```

Writes an annotated video (boxes, IDs, motion traces) and per-frame foot
positions to `tracks.jsonl`. Try it first with `--weights yolo11m.pt` before
you have a trained model.

## 4. Heatmap

```bash
python src/heatmap.py --tracks tracks.jsonl --id 7 --out player7.png
```

Without homography the heatmap is in image space. For the top-down pitch
version, fill in 4+ point correspondences in `configs/pitch.json` (pick pitch
corners / box corners you can see in one frame and their real metre positions
on a 105×68 pitch).

## Notes / next

- Team split by jersey colour and possession % are natural add-ons on top of
  `tracks.jsonl`.
- Deploy step: export `best.pt` to ONNX/TensorRT, wrap in a small FastAPI
  endpoint, measure latency. This is the part that reads as engineering, not a
  Colab notebook — worth doing.
- The homography approach here is deliberately minimal; the Roboflow `sports`
  repo has a more robust pitch-keypoint method if you want to go further.
