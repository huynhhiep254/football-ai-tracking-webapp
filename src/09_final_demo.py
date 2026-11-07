
from pathlib import Path
import argparse
import csv

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace


# ======================================================
# 1. PROJECT CONFIGURATION
# ======================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "outputs" / "training"
    / "yolov8n_100epochs" / "weights" / "best.pt"
)

TRACKER_PATH = ROOT / "configs" / "bytetrack_v3.yaml"

OUTPUT_DIR = ROOT / "outputs" / "final_demo"

DETECTION_CONF = 0.15
DETECTION_IOU = 0.70

# Minimum confidence for DISPLAYING non-player classes.
# These thresholds are preliminary and can be tested later.
DISPLAY_THRESHOLDS = {
    "goalkeeper": 0.40,
    "referee": 0.40,
    "ball": 0.45,
}

# BGR colors for OpenCV
CLASS_COLORS = {
    "player": (215, 120, 45),       # Blue
    "goalkeeper": (70, 185, 245),   # Yellow
    "referee": (60, 120, 225),      # Orange
    "ball": (105, 156, 41),         # Green
}


# ======================================================
# 2. DRAW RESPONSIVE LABELS AND BOXES
# ======================================================

def draw_object(frame, xyxy, label, class_name):

    h, w = frame.shape[:2]

    scale = max(0.65, min(h / 1080.0, 2.5))
    line_width = max(1, round(2 * scale))
    font_scale = 0.48 * scale
    font_thickness = max(1, round(scale))
    padding = max(4, round(6 * scale))

    color = CLASS_COLORS[class_name]

    x1, y1, x2, y2 = map(
        lambda v: int(round(v)), xyxy
    )

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        line_width,
        cv2.LINE_AA
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), baseline = cv2.getTextSize(
        label, font, font_scale, font_thickness
    )

    label_w = tw + 3 * padding
    label_h = th + baseline + 2 * padding

    lx = min(x1, max(0, w - label_w))

    if y1 > label_h + 4:
        ly = y1 - label_h - 3
    else:
        ly = min(y1 + 3, max(0, h - label_h))

    cv2.rectangle(
        frame,
        (lx, ly),
        (lx + label_w, ly + label_h),
        (28, 34, 39),
        -1
    )

    cv2.rectangle(
        frame,
        (lx, ly),
        (lx + max(3, round(4 * scale)), ly + label_h),
        color,
        -1
    )

    cv2.putText(
        frame,
        label,
        (lx + 2 * padding, ly + padding + th),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA
    )


def draw_dashboard(frame, counts):

    h, w = frame.shape[:2]
    scale = max(0.65, min(h / 1080.0, 2.5))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48 * scale
    thickness = max(1, round(scale))

    text = (
        f"YOLOv8 + ByteTrack  |  "
        f"P: {counts['player']}  "
        f"GK: {counts['goalkeeper']}  "
        f"REF: {counts['referee']}  "
        f"BALL: {counts['ball']}"
    )

    (tw, th), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    pad = max(8, round(12 * scale))

    x1 = pad
    panel_w = tw + 2 * pad
    panel_h = th + baseline + 2 * pad
    y1 = max(0, h - panel_h - pad)

    x2 = min(w, x1 + panel_w)
    y2 = min(h, y1 + panel_h)

    roi = frame[y1:y2, x1:x2]

    if roi.size > 0:
        background = np.full_like(
            roi, (28, 34, 39)
        )

        cv2.addWeighted(
            background, 0.78,
            roi, 0.22,
            0, roi
        )

    cv2.putText(
        frame,
        text,
        (x1 + pad, y1 + pad + th),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ======================================================
# 3. MAIN VIDEO PROCESSING
# ======================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="videos/match_stable.mp4",
        help="Path to input football video"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640
    )

    parser.add_argument(
        "--seconds",
        type=float,
        default=0,
        help="Preview first N seconds; 0 processes full video"
    )

    parser.add_argument(
        "--buffer-seconds",
        type=float,
        default=None,
        help="Optional FPS-adjusted track buffer"
    )

    args = parser.parse_args()

    video_path = Path(args.input)

    if not video_path.is_absolute():
        video_path = ROOT / video_path

    for path in [video_path, MODEL_PATH, TRACKER_PATH]:
        if not path.is_file():
            raise FileNotFoundError(path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    suffix = "preview" if args.seconds > 0 else "final"

    video_output = (
        OUTPUT_DIR / f"{video_path.stem}_{suffix}.mp4"
    )

    csv_output = (
        OUTPUT_DIR / f"{video_path.stem}_{suffix}.csv"
    )

    model = YOLO(str(MODEL_PATH))

    # Read actual class IDs from trained model
    class_ids = {
        name.lower(): int(class_id)
        for class_id, name in model.names.items()
    }

    required = {
        "player", "goalkeeper", "referee", "ball"
    }

    if not required.issubset(class_ids):
        raise ValueError(
            f"Missing classes: {required - set(class_ids)}"
        )

    player_id = class_ids["player"]

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError("Invalid video FPS")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read ByteTrack V3 config
    tracker_cfg = YAML.load(str(TRACKER_PATH))

    if args.buffer_seconds is not None:
        if args.buffer_seconds <= 0:
            raise ValueError(
                "--buffer-seconds must be positive"
            )

        tracker_cfg["track_buffer"] = max(
            1, round(fps * args.buffer_seconds)
        )

    # One tracker for player class only
    tracker = BYTETracker(
        args=IterableSimpleNamespace(**tracker_cfg)
    )

    device = 0 if torch.cuda.is_available() else "cpu"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(video_output),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create MP4 output")

    columns = [
        "frame", "timestamp", "class_id",
        "class_name", "track_id", "confidence",
        "x1", "y1", "x2", "y2"
    ]

    frame_index = 0
    unique_player_ids = set()

    print("INPUT:", video_path)
    print("MODEL:", MODEL_PATH)
    print("TRACKER:", TRACKER_PATH)
    print("VIDEO:", width, "x", height)
    print("FPS:", fps)
    print("TRACK BUFFER:", tracker_cfg["track_buffer"])
    print("DEVICE:", device)

    try:
        with open(
            csv_output, "w",
            newline="", encoding="utf-8"
        ) as csv_file:

            csv_writer = csv.DictWriter(
                csv_file, fieldnames=columns
            )
            csv_writer.writeheader()

            while True:

                ok, frame = cap.read()

                if not ok:
                    break

                frame_index += 1

                timestamp = (frame_index - 1) / fps

                # YOLO inference: exactly one call per frame
                result = model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    conf=DETECTION_CONF,
                    iou=DETECTION_IOU,
                    device=device,
                    verbose=False
                )[0]

                if result.boxes is None:
                    all_boxes = Boxes(
                        np.empty((0, 6), dtype=np.float32),
                        orig_shape=(height, width)
                    )
                else:
                    all_boxes = result.boxes.cpu().numpy()

                # Extract only player detections
                player_mask = (
                    all_boxes.cls.astype(int) == player_id
                )

                player_boxes = all_boxes[player_mask]

                # ByteTrack receives player boxes only
                tracks = tracker.update(
                    player_boxes,
                    frame
                )

                canvas = frame.copy()

                counts = {
                    "player": 0,
                    "goalkeeper": 0,
                    "referee": 0,
                    "ball": 0
                }

                # --------------------------------------
                # A. DRAW TRACKED PLAYERS
                # --------------------------------------

                for track in tracks:

                    x1, y1, x2, y2 = track[:4]

                    track_id = int(track[4])
                    confidence = float(track[5])

                    unique_player_ids.add(track_id)
                    counts["player"] += 1

                    draw_object(
                        canvas,
                        (x1, y1, x2, y2),
                        f"P-{track_id:02d}",
                        "player"
                    )

                    csv_writer.writerow({
                        "frame": frame_index,
                        "timestamp": round(timestamp, 3),
                        "class_id": player_id,
                        "class_name": "player",
                        "track_id": track_id,
                        "confidence": round(confidence, 4),
                        "x1": round(float(x1), 2),
                        "y1": round(float(y1), 2),
                        "x2": round(float(x2), 2),
                        "y2": round(float(y2), 2)
                    })

                # --------------------------------------
                # B. DRAW OTHER THREE CLASSES
                # --------------------------------------

                for box, conf, cls in zip(
                    all_boxes.xyxy,
                    all_boxes.conf,
                    all_boxes.cls
                ):

                    class_id = int(cls)
                    name = model.names[class_id].lower()
                    confidence = float(conf)

                    if name == "player":
                        continue

                    if name not in DISPLAY_THRESHOLDS:
                        continue

                    if confidence < DISPLAY_THRESHOLDS[name]:
                        continue

                    counts[name] += 1

                    draw_object(
                        canvas,
                        box,
                        name.capitalize(),
                        name
                    )

                    csv_writer.writerow({
                        "frame": frame_index,
                        "timestamp": round(timestamp, 3),
                        "class_id": class_id,
                        "class_name": name,
                        "track_id": "",
                        "confidence": round(confidence, 4),
                        "x1": round(float(box[0]), 2),
                        "y1": round(float(box[1]), 2),
                        "x2": round(float(box[2]), 2),
                        "y2": round(float(box[3]), 2)
                    })

                draw_dashboard(canvas, counts)

                writer.write(canvas)

                if frame_index % 50 == 0:
                    print(
                        f"Processed {frame_index} frames"
                    )

                if (
                    args.seconds > 0
                    and frame_index / fps >= args.seconds
                ):
                    break

    finally:
        cap.release()
        writer.release()

    print("\n===== COMPLETED =====")
    print("Processed frames:", frame_index)
    print("Unique player track IDs:", len(unique_player_ids))
    print("Video output:", video_output)
    print("CSV output:", csv_output)


if __name__ == "__main__":
    main()
