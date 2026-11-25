"""Detect + track players on a video and write an annotated copy.

Also dumps per-frame tracks to a .jsonl you can feed into heatmap.py.

Usage:
    python src/track.py --weights runs/detect/footvision/weights/best.pt \
        --source match.mp4 --out out.mp4 --tracks tracks.jsonl

Pass --weights yolo11m.pt to try it before you have a fine-tuned model
(players detect as COCO 'person').
"""
import argparse
import json

import numpy as np
import supervision as sv
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/footvision/weights/best.pt")
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="out.mp4")
    ap.add_argument("--tracks", default="tracks.jsonl")
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()

    model = YOLO(args.weights)
    box = sv.BoxAnnotator(thickness=2)
    labeler = sv.LabelAnnotator(text_scale=0.4, text_thickness=1)
    trace = sv.TraceAnnotator(trace_length=45, thickness=2)

    info = sv.VideoInfo.from_video_path(args.source)
    frames = sv.get_video_frames_generator(args.source)
    tracks_out = open(args.tracks, "w")

    with sv.VideoSink(args.out, info) as sink:
        for frame_idx, frame in enumerate(frames):
            result = model.track(
                frame, persist=True, tracker="botsort.yaml",
                conf=args.conf, imgsz=1280, verbose=False,
            )[0]
            det = sv.Detections.from_ultralytics(result)
            if det.tracker_id is None:
                det.tracker_id = np.array([], dtype=int)

            labels = [
                f"#{tid} {model.names[cid]}"
                for tid, cid in zip(det.tracker_id, det.class_id)
            ]
            annotated = trace.annotate(frame.copy(), det)
            annotated = box.annotate(annotated, det)
            annotated = labeler.annotate(annotated, det, labels)
            sink.write_frame(annotated)

            for xyxy, tid, cid in zip(det.xyxy, det.tracker_id, det.class_id):
                x1, y1, x2, y2 = (float(v) for v in xyxy)
                tracks_out.write(json.dumps({
                    "frame": frame_idx,
                    "id": int(tid),
                    "cls": model.names[int(cid)],
                    # foot position = bottom-centre of the box
                    "foot": [round((x1 + x2) / 2, 1), round(y2, 1)],
                }) + "\n")

    tracks_out.close()
    print(f"Wrote {args.out} and {args.tracks}")


if __name__ == "__main__":
    main()
