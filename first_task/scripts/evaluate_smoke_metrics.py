from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO


PROJECT = Path(__file__).resolve().parents[1]
RUN_DIR = PROJECT / "runs" / "smoke_cpu_window_20260928"
WEIGHTS = RUN_DIR / "weights" / "best.pt"
VAL_IMAGES = PROJECT / "prepared_window_only" / "images" / "val"
VAL_LABELS = PROJECT / "prepared_window_only" / "labels" / "val"
RUNTIME_REPORT = PROJECT / "reports" / "smoke_cpu_runtime.json"
OUTPUT_REPORT = PROJECT / "reports" / "smoke_cpu_metrics.json"

CONFIDENCE = 0.25
MATCH_IOU = 0.50
IMAGE_SIZE = 320
MAX_DET = 576


def box_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = box_area + boxes_area - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def load_ground_truth(label_path: Path, width: int, height: int) -> np.ndarray:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.empty((0, 4), dtype=np.float32)
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = map(float, parts[:5])
        boxes.append(
            [
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
            ]
        )
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def main() -> None:
    runtime = json.loads(RUNTIME_REPORT.read_text(encoding="utf-8"))
    trainer = runtime["trainer_metrics"]
    precision = float(trainer["metrics/precision(B)"])
    recall = float(trainer["metrics/recall(B)"])
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    image_paths = sorted(
        path
        for path in VAL_IMAGES.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    model = YOLO(str(WEIGHTS))
    predictions = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        iou=0.70,
        max_det=MAX_DET,
        device="cpu",
        stream=True,
        verbose=False,
    )

    total_tp = total_fp = total_fn = total_gt = total_predictions = 0
    for image_path, result in zip(image_paths, predictions):
        with Image.open(image_path) as image:
            width, height = image.size
        gt_boxes = load_ground_truth(VAL_LABELS / f"{image_path.stem}.txt", width, height)
        total_gt += len(gt_boxes)

        if result.boxes is None or len(result.boxes) == 0:
            pred_boxes = np.empty((0, 4), dtype=np.float32)
            confidences = np.empty((0,), dtype=np.float32)
        else:
            pred_boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            confidences = result.boxes.conf.cpu().numpy().astype(np.float32)
        total_predictions += len(pred_boxes)

        matched_gt: set[int] = set()
        tp = 0
        for pred_index in np.argsort(-confidences):
            ious = box_iou_xyxy(pred_boxes[pred_index], gt_boxes)
            if ious.size == 0:
                continue
            for gt_index in np.argsort(-ious):
                if ious[gt_index] < MATCH_IOU:
                    break
                if int(gt_index) not in matched_gt:
                    matched_gt.add(int(gt_index))
                    tp += 1
                    break

        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

    fixed_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    fixed_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    fixed_f1 = (
        2 * fixed_precision * fixed_recall / (fixed_precision + fixed_recall)
        if fixed_precision + fixed_recall
        else 0.0
    )

    report = {
        "model": str(WEIGHTS),
        "validation_images": len(image_paths),
        "validation_instances": total_gt,
        "ultralytics_summary_metrics": {
            "precision": precision,
            "recall": recall,
            "f1_from_reported_precision_recall": f1,
            "map_50": float(trainer["metrics/mAP50(B)"]),
            "map_50_95": float(trainer["metrics/mAP50-95(B)"]),
            "threshold_note": "Ultralytics validation summary; P/R are taken at the confidence point selected by its metric pipeline.",
        },
        "fixed_threshold_metrics": {
            "confidence": CONFIDENCE,
            "match_iou": MATCH_IOU,
            "max_det": MAX_DET,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "predictions": total_predictions,
            "precision": fixed_precision,
            "recall": fixed_recall,
            "f1": fixed_f1,
            "fp_per_image": total_fp / len(image_paths) if image_paths else 0.0,
        },
        "artifacts": {
            "pr_curve": str(RUN_DIR / "BoxPR_curve.png"),
            "confusion_matrix": str(RUN_DIR / "confusion_matrix.png"),
            "normalized_confusion_matrix": str(RUN_DIR / "confusion_matrix_normalized.png"),
            "best_weights": str(WEIGHTS),
            "last_weights": str(RUN_DIR / "weights" / "last.pt"),
        },
    }
    OUTPUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
