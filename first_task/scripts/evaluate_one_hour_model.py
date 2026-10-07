from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO


PROJECT = Path(__file__).resolve().parents[1]
RUN_NAME = "yolo11n_window_cpu_1h_20260928"
RUN_DIR = PROJECT / "runs" / RUN_NAME
WEIGHTS = RUN_DIR / "weights" / "best.pt"
DATA = PROJECT / "configs" / "data_clean_window_only.yaml"
VAL_IMAGES = PROJECT / "prepared_window_only" / "images" / "val"
VAL_LABELS = PROJECT / "prepared_window_only" / "labels" / "val"
EVAL_NAME = f"{RUN_NAME}_evaluation"
EVAL_DIR = PROJECT / "runs" / EVAL_NAME
SAMPLE_NAME = f"{RUN_NAME}_sample20"
SAMPLE_DIR = PROJECT / "runs" / SAMPLE_NAME
RUNTIME_REPORT = PROJECT / "reports" / "one_hour_cpu_runtime.json"
OUTPUT_REPORT = PROJECT / "reports" / "one_hour_cpu_metrics.json"

CONFIDENCES = (0.05, 0.10, 0.25)
MATCH_IOU = 0.50
IMAGE_SIZE = 640
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


def match_counts(pred_boxes: np.ndarray, confidences: np.ndarray, gt_boxes: np.ndarray, threshold: float):
    keep = confidences >= threshold
    boxes = pred_boxes[keep]
    scores = confidences[keep]
    matched_gt: set[int] = set()
    tp = 0
    for pred_index in np.argsort(-scores):
        ious = box_iou_xyxy(boxes[pred_index], gt_boxes)
        if ious.size == 0:
            continue
        for gt_index in np.argsort(-ious):
            if ious[gt_index] < MATCH_IOU:
                break
            if int(gt_index) not in matched_gt:
                matched_gt.add(int(gt_index))
                tp += 1
                break
    return tp, len(boxes) - tp, len(gt_boxes) - tp, len(boxes)


def make_contact_sheet(image_paths: list[Path], output: Path) -> None:
    thumbs = []
    for path in image_paths:
        with Image.open(path) as image:
            item = image.convert("RGB")
            item.thumbnail((360, 260))
            canvas = Image.new("RGB", (380, 300), "white")
            canvas.paste(item, ((380 - item.width) // 2, 10))
            ImageDraw.Draw(canvas).text((10, 275), path.name, fill="black")
            thumbs.append(canvas)
    if not thumbs:
        return
    sheet = Image.new("RGB", (380 * 4, 300 * 5), "white")
    for index, thumb in enumerate(thumbs[:20]):
        sheet.paste(thumb, ((index % 4) * 380, (index // 4) * 300))
    sheet.save(output, quality=88, optimize=True)


def main() -> None:
    if not WEIGHTS.exists():
        raise FileNotFoundError(WEIGHTS)
    runtime = json.loads(RUNTIME_REPORT.read_text(encoding="utf-8"))
    image_paths = sorted(
        path
        for path in VAL_IMAGES.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    model = YOLO(str(WEIGHTS))
    evaluation_start = time.perf_counter()
    metrics = model.val(
        data=str(DATA),
        imgsz=IMAGE_SIZE,
        batch=2,
        device="cpu",
        workers=0,
        max_det=MAX_DET,
        plots=True,
        project=str(PROJECT / "runs"),
        name=EVAL_NAME,
        exist_ok=True,
        verbose=True,
    )
    summary = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map_50": float(metrics.box.map50),
        "map_50_95": float(metrics.box.map),
    }
    summary["f1"] = (
        2 * summary["precision"] * summary["recall"]
        / (summary["precision"] + summary["recall"])
        if summary["precision"] + summary["recall"]
        else 0.0
    )

    fixed_metrics = {}
    total_gt = 0
    for image_path in image_paths:
        with Image.open(image_path) as image:
            total_gt += len(load_ground_truth(VAL_LABELS / f"{image_path.stem}.txt", *image.size))
    for threshold in CONFIDENCES:
        threshold_metrics = model.val(
            data=str(DATA),
            imgsz=IMAGE_SIZE,
            batch=2,
            device="cpu",
            workers=0,
            conf=threshold,
            iou=0.70,
            max_det=MAX_DET,
            # Ultralytics only populates the confusion-matrix accumulator when
            # plotting is enabled, so keep this True even though the numeric
            # matrix is the primary output of these threshold-specific runs.
            plots=True,
            project=str(PROJECT / "runs"),
            name=f"{EVAL_NAME}_conf_{threshold:.2f}".replace(".", ""),
            exist_ok=True,
            verbose=False,
        )
        matrix = np.asarray(threshold_metrics.confusion_matrix.matrix)
        tp = int(round(float(matrix[0, 0])))
        fp = int(round(float(matrix[0, 1])))
        fn = int(round(float(matrix[1, 0])))
        counts = {"tp": tp, "fp": fp, "fn": fn, "predictions": tp + fp}
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        fixed_metrics[f"{threshold:.2f}"] = {
            **counts,
            "precision": p,
            "recall": r,
            "f1": f1,
            "fp_per_image": fp / len(image_paths),
        }

    rng = random.Random(42)
    sample_paths = sorted(rng.sample(image_paths, k=min(20, len(image_paths))))
    if not SAMPLE_DIR.exists():
        model.predict(
            source=[str(path) for path in sample_paths],
            imgsz=IMAGE_SIZE,
            conf=0.10,
            iou=0.70,
            max_det=MAX_DET,
            device="cpu",
            save=True,
            project=str(PROJECT / "runs"),
            name=SAMPLE_NAME,
            verbose=False,
        )
    saved_samples = sorted(
        path for path in SAMPLE_DIR.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    contact_sheet = SAMPLE_DIR / "sample20_contact_sheet.jpg"
    make_contact_sheet(saved_samples, contact_sheet)

    if summary["precision"] >= 0.71 and summary["recall"] >= 0.68 and summary["map_50"] >= 0.72:
        grade = "challenge_target_met"
    elif summary["precision"] >= 0.50 and summary["recall"] >= 0.50:
        grade = "effective_progress_metrics_met"
    else:
        grade = "pipeline_only"

    report = {
        "model": str(WEIGHTS),
        "evaluation_elapsed_seconds": time.perf_counter() - evaluation_start,
        "training_elapsed_seconds": runtime["elapsed_seconds"],
        "total_elapsed_seconds": runtime["elapsed_seconds"] + (time.perf_counter() - evaluation_start),
        "completed_epochs": runtime["completed_epochs"],
        "validation_images": len(image_paths),
        "validation_instances": total_gt,
        "paper_targets": {"precision": 0.760, "recall": 0.728, "map_50": 0.777},
        "challenge_thresholds": {"precision": 0.71, "recall": 0.68, "map_50": 0.72},
        "ultralytics_summary_metrics": summary,
        "fixed_threshold_metrics": fixed_metrics,
        "automated_grade": grade,
        "artifacts": {
            "pr_curve": str(EVAL_DIR / "BoxPR_curve.png"),
            "f1_curve": str(EVAL_DIR / "BoxF1_curve.png"),
            "confusion_matrix": str(EVAL_DIR / "confusion_matrix.png"),
            "normalized_confusion_matrix": str(EVAL_DIR / "confusion_matrix_normalized.png"),
            "sample_predictions": str(SAMPLE_DIR),
            "sample_contact_sheet": str(contact_sheet),
            "best_weights": str(WEIGHTS),
            "last_weights": str(RUN_DIR / "weights" / "last.pt"),
        },
    }
    OUTPUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nONE_HOUR_METRICS_REPORT")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
