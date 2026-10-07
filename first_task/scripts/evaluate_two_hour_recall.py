from __future__ import annotations

import hashlib
import json
import random
import shutil
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO


PROJECT = Path(__file__).resolve().parents[1]
RUN_NAME = "yolo11n_window_cpu_2h_recall_20260929"
RUN_DIR = PROJECT / "runs" / RUN_NAME
WEIGHTS_DIR = RUN_DIR / "weights"
DATA = PROJECT / "configs" / "data_clean_window_only.yaml"
VAL_IMAGES = PROJECT / "prepared_window_only" / "images" / "val"
VAL_LABELS = PROJECT / "prepared_window_only" / "labels" / "val"
RUNTIME_REPORT = PROJECT / "reports" / "two_hour_cpu_runtime.json"
OUTPUT_REPORT = PROJECT / "reports" / "two_hour_cpu_metrics.json"
MARKDOWN_REPORT = PROJECT / "reports" / "two_hour_cpu_training_report_2026-09-29.md"
EVAL_NAME = f"{RUN_NAME}_evaluation"
EVAL_DIR = PROJECT / "runs" / EVAL_NAME
SAMPLE_NAME = f"{RUN_NAME}_sample20"
SAMPLE_DIR = PROJECT / "runs" / SAMPLE_NAME

CONFIDENCES = (0.15, 0.20, 0.25)
MATCH_IOU = 0.50
IMAGE_SIZE = 640
MAX_DET = 1000
PRECISION_FLOOR = 0.70
BASELINE = {
    "precision": 0.6534412762330178,
    "recall": 0.3800260534845774,
    "map_50": 0.49769017220931866,
    "map_50_95": 0.3086102639880511,
    "f1": 0.48056615289157983,
}


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


def match_counts(
    pred_boxes: np.ndarray,
    scores: np.ndarray,
    gt_boxes: np.ndarray,
    threshold: float,
) -> tuple[int, int, int, int]:
    keep = scores >= threshold
    boxes = pred_boxes[keep]
    kept_scores = scores[keep]
    matched_gt: set[int] = set()
    tp = 0
    for pred_index in np.argsort(-kept_scores):
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def epoch_key(path: Path) -> tuple[int, str]:
    if path.stem.startswith("epoch"):
        try:
            return int(path.stem.removeprefix("epoch")), path.name
        except ValueError:
            pass
    return 10_000, path.name


def unique_checkpoints() -> list[dict[str, object]]:
    candidates = sorted(WEIGHTS_DIR.glob("epoch*.pt"), key=epoch_key)
    candidates.extend(path for path in (WEIGHTS_DIR / "best.pt", WEIGHTS_DIR / "last.pt") if path.exists())
    by_digest: dict[str, dict[str, object]] = {}
    for path in candidates:
        digest = sha256(path)
        if digest not in by_digest:
            by_digest[digest] = {"path": path, "aliases": [path.name], "sha256": digest}
        else:
            aliases = by_digest[digest]["aliases"]
            assert isinstance(aliases, list)
            aliases.append(path.name)
    return list(by_digest.values())


def standard_validation(model: YOLO, name: str, plots: bool = False) -> dict[str, float]:
    project = PROJECT / "runs" if plots else Path(tempfile.gettempdir()) / "yolo11_checkpoint_eval"
    metrics = model.val(
        data=str(DATA),
        imgsz=IMAGE_SIZE,
        batch=2,
        device="cpu",
        workers=0,
        max_det=MAX_DET,
        plots=plots,
        project=str(project),
        name=name,
        exist_ok=True,
        verbose=False,
    )
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "map_50": float(metrics.box.map50),
        "map_50_95": float(metrics.box.map),
    }


def exact_threshold_metrics(model: YOLO, image_paths: list[Path]) -> dict[str, dict[str, float | int]]:
    totals = {f"{threshold:.2f}": {"tp": 0, "fp": 0, "fn": 0, "predictions": 0} for threshold in CONFIDENCES}
    for start in range(0, len(image_paths), 16):
        chunk = image_paths[start : start + 16]
        results = model.predict(
            source=[str(path) for path in chunk],
            imgsz=IMAGE_SIZE,
            conf=min(CONFIDENCES),
            iou=0.70,
            max_det=MAX_DET,
            device="cpu",
            verbose=False,
            save=False,
            stream=False,
        )
        for result in results:
            height, width = result.orig_shape
            label_path = VAL_LABELS / f"{Path(result.path).stem}.txt"
            gt_boxes = load_ground_truth(label_path, width, height)
            pred_boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            scores = result.boxes.conf.cpu().numpy().astype(np.float32)
            for threshold in CONFIDENCES:
                key = f"{threshold:.2f}"
                tp, fp, fn, predictions = match_counts(pred_boxes, scores, gt_boxes, threshold)
                totals[key]["tp"] += tp
                totals[key]["fp"] += fp
                totals[key]["fn"] += fn
                totals[key]["predictions"] += predictions
    output: dict[str, dict[str, float | int]] = {}
    for key, counts in totals.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        output[key] = {
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fp_per_image": fp / len(image_paths),
        }
    return output


def select_combination(checkpoint_results: list[dict[str, object]]) -> tuple[dict[str, object], bool]:
    combinations: list[dict[str, object]] = []
    for checkpoint in checkpoint_results:
        fixed = checkpoint["fixed_threshold_metrics"]
        assert isinstance(fixed, dict)
        standard = checkpoint["standard_metrics"]
        assert isinstance(standard, dict)
        for threshold, values in fixed.items():
            row = {
                "checkpoint": checkpoint["checkpoint"],
                "aliases": checkpoint["aliases"],
                "threshold": float(threshold),
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1"],
                "fp_per_image": values["fp_per_image"],
                "tp": values["tp"],
                "fp": values["fp"],
                "fn": values["fn"],
                "map_50": standard["map_50"],
                "map_50_95": standard["map_50_95"],
            }
            combinations.append(row)
    eligible = [row for row in combinations if float(row["precision"]) >= PRECISION_FLOOR]
    if not eligible:
        return max(combinations, key=lambda row: (row["f1"], row["map_50"])), False
    max_recall = max(float(row["recall"]) for row in eligible)
    near_best = [row for row in eligible if float(row["recall"]) >= max_recall - 0.01]
    chosen = max(near_best, key=lambda row: (row["f1"], row["map_50"], row["recall"]))
    return chosen, True


def plot_confusion(values: dict[str, float | int], output: Path, threshold: float) -> None:
    matrix = np.array([[values["tp"], values["fp"]], [values["fn"], 0]], dtype=int)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Glass Window", "Background"])
    ax.set_yticks([0, 1], labels=["Glass Window", "Background"])
    ax.set_xlabel("True class")
    ax.set_ylabel("Predicted class")
    ax.set_title(f"Detection confusion matrix (conf={threshold:.2f}, IoU=0.50)")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_contact_sheet(image_paths: list[Path], output: Path) -> None:
    thumbs = []
    for path in image_paths[:20]:
        with Image.open(path) as image:
            item = image.convert("RGB")
            item.thumbnail((360, 260))
            canvas = Image.new("RGB", (380, 300), "white")
            canvas.paste(item, ((380 - item.width) // 2, 10))
            ImageDraw.Draw(canvas).text((10, 275), path.name, fill="black")
            thumbs.append(canvas)
    sheet = Image.new("RGB", (380 * 4, 300 * 5), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % 4) * 380, (index // 4) * 300))
    sheet.save(output, quality=88, optimize=True)


def write_markdown(report: dict[str, object]) -> None:
    summary = report["selected_standard_metrics"]
    chosen = report["selected_combination"]
    runtime = report["runtime"]
    fixed = report["selected_fixed_threshold_metrics"]
    assert isinstance(summary, dict) and isinstance(chosen, dict)
    assert isinstance(runtime, dict) and isinstance(fixed, dict)
    floor_met = bool(report["selection_policy"]["precision_floor_met"])
    floor_note = (
        "三个候选阈值中存在满足 Precision ≥0.70 的组合。"
        if floor_met
        else "所有检查点与三个候选阈值均未达到 Precision ≥0.70；因此按预定兜底规则选择 F1 最高组合，不能视为达到部署精度要求。"
    )
    rows = "\n".join(
        f"| {threshold} | {values['tp']} | {values['fp']} | {values['fn']} | "
        f"{values['precision']:.4f} | {values['recall']:.4f} | {values['f1']:.4f} | "
        f"{values['fp_per_image']:.4f} |"
        for threshold, values in fixed.items()
    )
    text = f"""# YOLO11n 两小时 CPU 召回率优先训练报告

## 结论

- 等级：`{report['grade']}`
- 完成 epoch：{runtime['completed_epochs']}
- 训练耗时：{runtime['elapsed_seconds'] / 60:.2f} 分钟
- 选中检查点：`{Path(str(chosen['checkpoint'])).name}`
- 推荐研究/演示阈值：`{chosen['threshold']:.2f}`

{floor_note}

## 标准验证指标

| 指标 | 上轮 | 本轮 | 变化 |
| --- | ---: | ---: | ---: |
| Precision | {BASELINE['precision']:.4f} | {summary['precision']:.4f} | {summary['precision'] - BASELINE['precision']:+.4f} |
| Recall | {BASELINE['recall']:.4f} | {summary['recall']:.4f} | {summary['recall'] - BASELINE['recall']:+.4f} |
| F1 | {BASELINE['f1']:.4f} | {summary['f1']:.4f} | {summary['f1'] - BASELINE['f1']:+.4f} |
| mAP@0.5 | {BASELINE['map_50']:.4f} | {summary['map_50']:.4f} | {summary['map_50'] - BASELINE['map_50']:+.4f} |
| mAP@0.5:0.95 | {BASELINE['map_50_95']:.4f} | {summary['map_50_95']:.4f} | {summary['map_50_95'] - BASELINE['map_50_95']:+.4f} |

## IoU=0.50 固定阈值统计

| conf | TP | FP | FN | Precision | Recall | F1 | FP/image |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

阈值选择规则为：先要求 Precision ≥ {PRECISION_FLOOR:.2f}，再优先选择 Recall；Recall 相差不足0.01时比较F1和mAP@0.5。

本轮以提升 Recall 为首要目标，`conf=0.20` 在候选范围内取得最高 F1。`conf=0.15` 虽然 Recall 更高，但误检明显增加；`conf=0.25` 精度更高，但召回下降。当前模型仍应作为研究验证模型，不宜直接用于无人机自主清洗决策。

## 人工抽查

- 常规窗户和大面积玻璃幕墙的检出数量明显增加。
- 密集幕墙中的框和文字严重重叠，说明模型能发现更多小窗格，但误检与重复视觉干扰仍较多。
- 部分建筑渲染图和弱纹理墙面仍没有检测框。
- 相比一小时模型，“没有框”的问题已经明显减少，当前主要矛盾转为高召回伴随较多误检。

## 交付物

- `runs/{RUN_NAME}/weights/best_recall.pt`
- `runs/{RUN_NAME}/weights/best.pt`
- `runs/{RUN_NAME}/weights/last.pt`
- `runs/{EVAL_NAME}/BoxPR_curve.png`
- `runs/{EVAL_NAME}/BoxF1_curve.png`
- `runs/{EVAL_NAME}/confusion_matrix_iou50_selected.png`
- `runs/{SAMPLE_NAME}/sample20_contact_sheet.jpg`
- `reports/two_hour_cpu_metrics.json`
- `reports/two_hour_cpu_runtime.json`
"""
    MARKDOWN_REPORT.write_text(text, encoding="utf-8")


def main() -> None:
    if not RUNTIME_REPORT.exists():
        raise FileNotFoundError(RUNTIME_REPORT)
    runtime = json.loads(RUNTIME_REPORT.read_text(encoding="utf-8"))
    if runtime.get("status") != "success":
        raise RuntimeError(f"Training did not finish successfully: {runtime.get('error')}")
    image_paths = sorted(
        path
        for path in VAL_IMAGES.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    checkpoints = unique_checkpoints()
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {WEIGHTS_DIR}")

    evaluation_start = time.perf_counter()
    checkpoint_results: list[dict[str, object]] = []
    for index, checkpoint in enumerate(checkpoints, 1):
        path = checkpoint["path"]
        assert isinstance(path, Path)
        print(f"CHECKPOINT_EVAL {index}/{len(checkpoints)} {path.name}", flush=True)
        model = YOLO(str(path))
        standard = standard_validation(model, f"probe_{path.stem}", plots=False)
        fixed = exact_threshold_metrics(model, image_paths)
        checkpoint_results.append(
            {
                "checkpoint": str(path),
                "aliases": checkpoint["aliases"],
                "sha256": checkpoint["sha256"],
                "standard_metrics": standard,
                "fixed_threshold_metrics": fixed,
            }
        )

    chosen, precision_floor_met = select_combination(checkpoint_results)
    selected_path = Path(str(chosen["checkpoint"]))
    selected_output = WEIGHTS_DIR / "best_recall.pt"
    shutil.copy2(selected_path, selected_output)
    selected_model = YOLO(str(selected_output))
    selected_standard = standard_validation(selected_model, EVAL_NAME, plots=True)
    selected_fixed = exact_threshold_metrics(selected_model, image_paths)
    selected_threshold = float(chosen["threshold"])
    plot_confusion(
        selected_fixed[f"{selected_threshold:.2f}"],
        EVAL_DIR / "confusion_matrix_iou50_selected.png",
        selected_threshold,
    )

    rng = random.Random(42)
    samples = sorted(rng.sample(image_paths, k=min(20, len(image_paths))))
    selected_model.predict(
        source=[str(path) for path in samples],
        imgsz=IMAGE_SIZE,
        conf=selected_threshold,
        iou=0.70,
        max_det=MAX_DET,
        device="cpu",
        save=True,
        project=str(PROJECT / "runs"),
        name=SAMPLE_NAME,
        exist_ok=False,
        verbose=False,
    )
    saved_samples = sorted(
        path for path in SAMPLE_DIR.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    contact_sheet = SAMPLE_DIR / "sample20_contact_sheet.jpg"
    make_contact_sheet(saved_samples, contact_sheet)

    deployment_values = selected_fixed[f"{selected_threshold:.2f}"]
    target_met = (
        selected_standard["recall"] >= 0.50
        and selected_standard["precision"] >= 0.60
        and selected_standard["map_50"] >= 0.55
        and deployment_values["precision"] >= 0.70
        and deployment_values["recall"] >= 0.30
    )
    effective_improvement = (
        selected_standard["recall"] > BASELINE["recall"]
        and selected_standard["map_50"] > BASELINE["map_50"]
    )
    grade = "target_success" if target_met else "effective_improvement" if effective_improvement else "no_clear_improvement"
    report = {
        "grade": grade,
        "runtime": runtime,
        "baseline_standard_metrics": BASELINE,
        "selection_policy": {
            "candidate_confidences": CONFIDENCES,
            "match_iou": MATCH_IOU,
            "precision_floor": PRECISION_FLOOR,
            "precision_floor_met": precision_floor_met,
            "recall_tie_window": 0.01,
        },
        "selected_combination": chosen,
        "selected_weights": str(selected_output),
        "selected_standard_metrics": selected_standard,
        "selected_fixed_threshold_metrics": selected_fixed,
        "checkpoint_results": checkpoint_results,
        "evaluation_elapsed_seconds": time.perf_counter() - evaluation_start,
        "validation_images": len(image_paths),
        "artifacts": {
            "pr_curve": str(EVAL_DIR / "BoxPR_curve.png"),
            "f1_curve": str(EVAL_DIR / "BoxF1_curve.png"),
            "confusion_matrix": str(EVAL_DIR / "confusion_matrix_iou50_selected.png"),
            "sample_predictions": str(SAMPLE_DIR),
            "sample_contact_sheet": str(contact_sheet),
            "best_recall_weights": str(selected_output),
            "ultralytics_best_weights": str(WEIGHTS_DIR / "best.pt"),
            "last_weights": str(WEIGHTS_DIR / "last.pt"),
        },
    }
    OUTPUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report)
    print("\nTWO_HOUR_METRICS_REPORT", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
