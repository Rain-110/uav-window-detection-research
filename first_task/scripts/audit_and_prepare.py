from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


RAW = Path(__file__).resolve().parents[2] / "dataset" / "raw"
PROJECT = Path(__file__).resolve().parents[1]
PREPARED_6 = PROJECT / "prepared_6class"
PREPARED_1 = PROJECT / "prepared_window_only"
QUARANTINE = PROJECT / "quarantine"
REPORTS = PROJECT / "reports"
PREVIEWS = REPORTS / "previews"
CONFIGS = PROJECT / "configs"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["glass window", "pillar", "wall", "person", "door", "car"]
COLORS = [
    (36, 173, 95),
    (52, 152, 219),
    (155, 89, 182),
    (241, 196, 15),
    (230, 126, 34),
    (231, 76, 60),
]


@dataclass
class ParsedLabel:
    valid_lines: list[str] = field(default_factory=list)
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    removed_zero_area: list[dict] = field(default_factory=list)
    uncertain_errors: list[dict] = field(default_factory=list)
    original_empty: bool = False


@dataclass
class Sample:
    split: str
    image: Path
    label: Path
    parsed: ParsedLabel
    image_hash: str
    quarantined: bool = False
    reason: str = ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_empty_targets() -> None:
    targets = [PREPARED_6, PREPARED_1, QUARANTINE, REPORTS]
    nonempty = []
    for target in targets:
        if target.exists() and any(path.is_file() for path in target.rglob("*")):
            nonempty.append(str(target))
    if nonempty:
        raise RuntimeError(
            "Refusing to overwrite non-empty output directories: " + "; ".join(nonempty)
        )


def make_dirs() -> None:
    for base in (PREPARED_6, PREPARED_1):
        for split in ("train", "val"):
            (base / "images" / split).mkdir(parents=True, exist_ok=True)
            (base / "labels" / split).mkdir(parents=True, exist_ok=True)
    QUARANTINE.mkdir(parents=True, exist_ok=True)
    (PREVIEWS / "annotations").mkdir(parents=True, exist_ok=True)
    (PREVIEWS / "manual_review").mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(parents=True, exist_ok=True)


def parse_label(path: Path) -> ParsedLabel:
    parsed = ParsedLabel()
    text = path.read_text(encoding="utf-8", errors="replace")
    nonempty = [line.strip() for line in text.splitlines() if line.strip()]
    parsed.original_empty = not nonempty

    for line_no, line in enumerate(nonempty, 1):
        parts = line.split()
        if len(parts) != 5:
            parsed.uncertain_errors.append(
                {"line": line_no, "reason": f"field_count={len(parts)}", "text": line[:200]}
            )
            continue
        try:
            class_id = int(parts[0])
            x, y, w, h = [float(value) for value in parts[1:]]
        except ValueError:
            parsed.uncertain_errors.append(
                {"line": line_no, "reason": "parse_error", "text": line[:200]}
            )
            continue
        if not 0 <= class_id < len(CLASS_NAMES):
            parsed.uncertain_errors.append(
                {"line": line_no, "reason": f"class_id={class_id}", "text": line[:200]}
            )
            continue
        if w <= 0 or h <= 0:
            parsed.removed_zero_area.append(
                {"line": line_no, "reason": "zero_or_negative_area", "text": line[:200]}
            )
            continue
        if any(value < 0 or value > 1 for value in (x, y, w, h)):
            parsed.uncertain_errors.append(
                {"line": line_no, "reason": "coordinate_out_of_range", "text": line[:200]}
            )
            continue
        normalized = f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        parsed.valid_lines.append(normalized)
        parsed.boxes.append((class_id, x, y, w, h))
    return parsed


def copy_to_quarantine(
    source: Path,
    reason: str,
    split: str,
    manifest: list[dict],
    related: str = "",
) -> None:
    target_dir = QUARANTINE / reason / split
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if not target.exists():
        shutil.copy2(source, target)
    manifest.append(
        {
            "reason": reason,
            "split": split,
            "source": str(source),
            "quarantine_copy": str(target),
            "related": related,
        }
    )


def clean_family(stem: str) -> str:
    value = re.sub(r"^aug_\d+_", "", stem, flags=re.IGNORECASE)
    return value.lower().replace(" ", "")


def hardlink_or_copy(source: Path, target: Path) -> str:
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def draw_boxes(image_path: Path, sample: Sample, target: Path) -> None:
    with Image.open(image_path) as original:
        image = original.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for class_id, x, y, w, h in sample.parsed.boxes:
        left = max(0, int((x - w / 2) * image.width))
        top = max(0, int((y - h / 2) * image.height))
        right = min(image.width - 1, int((x + w / 2) * image.width))
        bottom = min(image.height - 1, int((y + h / 2) * image.height))
        color = COLORS[class_id]
        width = max(2, round(max(image.width, image.height) / 400))
        draw.rectangle((left, top, right, bottom), outline=color, width=width)
        label = f"{class_id}:{CLASS_NAMES[class_id]}"
        box = draw.textbbox((left, top), label, font=font)
        draw.rectangle(box, fill=color)
        draw.text((left, top), label, fill="white", font=font)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.thumbnail((1600, 1200))
    image.save(target, quality=90)


def make_contact_sheets(paths: list[Path], prefix: str, columns: int = 4, rows: int = 3) -> list[Path]:
    outputs = []
    per_sheet = columns * rows
    font = ImageFont.load_default()
    for sheet_index, start in enumerate(range(0, len(paths), per_sheet), 1):
        group = paths[start : start + per_sheet]
        cell_w, cell_h = 320, 240
        canvas = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
        draw = ImageDraw.Draw(canvas)
        for index, path in enumerate(group):
            try:
                with Image.open(path) as source:
                    thumb = source.convert("RGB")
                thumb.thumbnail((cell_w - 12, cell_h - 32))
                x = (index % columns) * cell_w + (cell_w - thumb.width) // 2
                y = (index // columns) * cell_h + 22
                canvas.paste(thumb, (x, y))
                draw.text(((index % columns) * cell_w + 4, (index // columns) * cell_h + 4), path.name[:44], fill="black", font=font)
            except Exception as exc:
                draw.text(((index % columns) * cell_w + 4, (index // columns) * cell_h + 4), f"ERROR {path.name}: {exc}", fill="red", font=font)
        output = PREVIEWS / "manual_review" / f"{prefix}_{sheet_index:02d}.jpg"
        canvas.save(output, quality=88)
        outputs.append(output)
    return outputs


def choose_preview_samples(samples: Iterable[Sample], count: int = 30) -> list[Sample]:
    candidates = [sample for sample in samples if not sample.quarantined]
    selected: list[Sample] = []
    selected_paths: set[Path] = set()
    for class_id in range(len(CLASS_NAMES)):
        class_samples = [s for s in candidates if any(box[0] == class_id for box in s.parsed.boxes)]
        class_samples.sort(key=lambda s: (-sum(box[0] == class_id for box in s.parsed.boxes), s.image.name))
        for sample in class_samples[:3]:
            if sample.image not in selected_paths:
                selected.append(sample)
                selected_paths.add(sample.image)
    remaining = sorted(candidates, key=lambda s: (-len(s.parsed.boxes), s.image.name))
    for sample in remaining:
        if len(selected) >= count:
            break
        if sample.image not in selected_paths:
            selected.append(sample)
            selected_paths.add(sample.image)
    return selected[:count]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not RAW.is_dir():
        raise FileNotFoundError(f"Raw dataset not found: {RAW}")
    ensure_empty_targets()
    make_dirs()

    manifest_rows = []
    for path in sorted(p for p in RAW.rglob("*") if p.is_file()):
        manifest_rows.append(
            {
                "relative_path": path.relative_to(RAW).as_posix(),
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_csv(
        REPORTS / "file_manifest.csv",
        manifest_rows,
        ["relative_path", "extension", "size_bytes", "sha256"],
    )

    samples: list[Sample] = []
    quarantine_manifest: list[dict] = []
    issue_rows: list[dict] = []
    manual_preview_sources: dict[str, list[Path]] = defaultdict(list)
    initial_counts = {}

    for source_split, output_split in (("train", "train"), ("test", "val")):
        folder = RAW / source_split
        images = {p.stem: p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
        labels = {p.stem: p for p in folder.glob("*.txt")}
        initial_counts[source_split] = {"images": len(images), "labels": len(labels)}

        for stem in sorted(set(images) - set(labels)):
            image = images[stem]
            copy_to_quarantine(image, "missing_label", output_split, quarantine_manifest)
            manual_preview_sources["missing_label"].append(image)
            issue_rows.append({"split": output_split, "file": image.name, "issue": "missing_label", "action": "quarantined"})

        for stem in sorted(set(labels) - set(images)):
            label = labels[stem]
            copy_to_quarantine(label, "orphan_label", output_split, quarantine_manifest)
            issue_rows.append({"split": output_split, "file": label.name, "issue": "orphan_label", "action": "quarantined"})

        for stem in sorted(set(images) & set(labels)):
            image = images[stem]
            label = labels[stem]
            parsed = parse_label(label)
            sample = Sample(output_split, image, label, parsed, sha256(image))
            if parsed.original_empty:
                sample.quarantined = True
                sample.reason = "empty_label_review"
                copy_to_quarantine(image, sample.reason, output_split, quarantine_manifest, label.name)
                copy_to_quarantine(label, sample.reason, output_split, quarantine_manifest, image.name)
                manual_preview_sources[sample.reason].append(image)
                issue_rows.append({"split": output_split, "file": image.name, "issue": "empty_label", "action": "quarantined_for_review"})
            elif parsed.uncertain_errors:
                sample.quarantined = True
                sample.reason = "uncertain_label_error"
                copy_to_quarantine(image, sample.reason, output_split, quarantine_manifest, label.name)
                copy_to_quarantine(label, sample.reason, output_split, quarantine_manifest, image.name)
                manual_preview_sources[sample.reason].append(image)
                for error in parsed.uncertain_errors:
                    issue_rows.append({"split": output_split, "file": label.name, "issue": error["reason"], "action": "quarantined_for_review"})
            elif not parsed.valid_lines:
                sample.quarantined = True
                sample.reason = "empty_after_auto_fix_review"
                copy_to_quarantine(image, sample.reason, output_split, quarantine_manifest, label.name)
                copy_to_quarantine(label, sample.reason, output_split, quarantine_manifest, image.name)
                manual_preview_sources[sample.reason].append(image)
                issue_rows.append({"split": output_split, "file": label.name, "issue": "empty_after_auto_fix", "action": "quarantined_for_review"})
            for removed in parsed.removed_zero_area:
                issue_rows.append({"split": output_split, "file": label.name, "issue": removed["reason"], "action": "line_removed"})
            samples.append(sample)

    hash_groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        if not sample.quarantined:
            hash_groups[sample.image_hash].append(sample)

    for image_hash, group in sorted(hash_groups.items()):
        if len(group) < 2:
            continue
        splits = {sample.split for sample in group}
        if len(splits) > 1:
            for sample in [item for item in group if item.split == "train"]:
                sample.quarantined = True
                sample.reason = "cross_split_duplicate"
                related = ";".join(item.image.name for item in group if item.split == "val")
                copy_to_quarantine(sample.image, sample.reason, sample.split, quarantine_manifest, related)
                copy_to_quarantine(sample.label, sample.reason, sample.split, quarantine_manifest, related)
                issue_rows.append({"split": sample.split, "file": sample.image.name, "issue": "cross_split_duplicate", "action": "train_copy_quarantined"})

    for split in ("train", "val"):
        local_groups: dict[str, list[Sample]] = defaultdict(list)
        for sample in samples:
            if sample.split == split and not sample.quarantined:
                local_groups[sample.image_hash].append(sample)
        for image_hash, group in sorted(local_groups.items()):
            if len(group) < 2:
                continue
            signatures = {"\n".join(sample.parsed.valid_lines) for sample in group}
            if len(signatures) == 1:
                canonical = sorted(group, key=lambda item: item.image.name)[0]
                for sample in group:
                    if sample is canonical:
                        continue
                    sample.quarantined = True
                    sample.reason = "exact_duplicate"
                    copy_to_quarantine(sample.image, sample.reason, split, quarantine_manifest, canonical.image.name)
                    copy_to_quarantine(sample.label, sample.reason, split, quarantine_manifest, canonical.label.name)
                    issue_rows.append({"split": split, "file": sample.image.name, "issue": "exact_duplicate", "action": f"quarantined_keep={canonical.image.name}"})
            else:
                for sample in group:
                    sample.quarantined = True
                    sample.reason = "duplicate_label_conflict"
                    related = ";".join(item.image.name for item in group if item is not sample)
                    copy_to_quarantine(sample.image, sample.reason, split, quarantine_manifest, related)
                    copy_to_quarantine(sample.label, sample.reason, split, quarantine_manifest, related)
                    manual_preview_sources[sample.reason].append(sample.image)
                    issue_rows.append({"split": split, "file": sample.image.name, "issue": "duplicate_label_conflict", "action": "entire_group_quarantined"})

    prepared_samples = [sample for sample in samples if not sample.quarantined]
    link_modes = Counter()
    class_counts_6 = Counter()
    class_counts_1 = Counter()
    prepared_counts = defaultdict(lambda: Counter())

    for sample in prepared_samples:
        split = sample.split
        target_image_6 = PREPARED_6 / "images" / split / sample.image.name
        target_label_6 = PREPARED_6 / "labels" / split / f"{sample.image.stem}.txt"
        shutil.copy2(sample.image, target_image_6)
        target_label_6.write_text("\n".join(sample.parsed.valid_lines) + "\n", encoding="utf-8")

        target_image_1 = PREPARED_1 / "images" / split / sample.image.name
        target_label_1 = PREPARED_1 / "labels" / split / f"{sample.image.stem}.txt"
        link_modes[hardlink_or_copy(target_image_6, target_image_1)] += 1
        window_lines = [line for line in sample.parsed.valid_lines if line.startswith("0 ")]
        target_label_1.write_text(("\n".join(window_lines) + "\n") if window_lines else "", encoding="utf-8")

        prepared_counts[split]["images"] += 1
        prepared_counts[split]["labels"] += 1
        for class_id, *_ in sample.parsed.boxes:
            class_counts_6[(split, class_id)] += 1
            if class_id == 0:
                class_counts_1[(split, 0)] += 1

    preview_rows = []
    annotation_previews = []
    for index, sample in enumerate(choose_preview_samples(prepared_samples), 1):
        target = PREVIEWS / "annotations" / f"{index:02d}_{sample.split}_{sample.image.stem}.jpg"
        draw_boxes(sample.image, sample, target)
        annotation_previews.append(target)
        preview_rows.append(
            {
                "index": index,
                "split": sample.split,
                "source_image": str(sample.image),
                "preview": str(target),
                "box_count": len(sample.parsed.boxes),
                "classes": ",".join(sorted({CLASS_NAMES[box[0]] for box in sample.parsed.boxes})),
            }
        )

    annotation_contact = []
    if annotation_previews:
        annotation_contact = make_contact_sheets(annotation_previews, "annotation_preview")
    manual_contact = {}
    for reason, paths in sorted(manual_preview_sources.items()):
        manual_contact[reason] = [str(path) for path in make_contact_sheets(paths, reason)]

    yaml_6 = f"""path: {PREPARED_6.as_posix()}\ntrain: images/train\nval: images/val\n\nnames:\n  0: glass window\n  1: pillar\n  2: wall\n  3: person\n  4: door\n  5: car\n"""
    yaml_1 = f"""path: {PREPARED_1.as_posix()}\ntrain: images/train\nval: images/val\n\nnames:\n  0: glass window\n"""
    (CONFIGS / "data_clean_6class.yaml").write_text(yaml_6, encoding="utf-8")
    (CONFIGS / "data_clean_window_only.yaml").write_text(yaml_1, encoding="utf-8")

    write_csv(REPORTS / "issues_and_fixes.csv", issue_rows, ["split", "file", "issue", "action"])
    write_csv(
        REPORTS / "quarantine_manifest.csv",
        quarantine_manifest,
        ["reason", "split", "source", "quarantine_copy", "related"],
    )
    write_csv(
        REPORTS / "preview_index.csv",
        preview_rows,
        ["index", "split", "source_image", "preview", "box_count", "classes"],
    )

    summary = {
        "raw_dataset": str(RAW),
        "project": str(PROJECT),
        "initial_counts": initial_counts,
        "prepared_counts": {split: dict(values) for split, values in prepared_counts.items()},
        "prepared_6class_box_counts": {
            split: {CLASS_NAMES[class_id]: class_counts_6[(split, class_id)] for class_id in range(len(CLASS_NAMES))}
            for split in ("train", "val")
        },
        "prepared_window_only_box_counts": {
            split: class_counts_1[(split, 0)] for split in ("train", "val")
        },
        "automatic_fixes": {
            "zero_area_lines_removed": sum(1 for row in issue_rows if row["action"] == "line_removed"),
            "duplicate_samples_quarantined": sum(1 for row in issue_rows if row["issue"] == "exact_duplicate"),
            "cross_split_train_samples_quarantined": sum(1 for row in issue_rows if row["issue"] == "cross_split_duplicate"),
        },
        "manual_review": {
            reason: len(paths) for reason, paths in sorted(manual_preview_sources.items())
        },
        "quarantine_manifest_entries": len(quarantine_manifest),
        "window_only_image_materialization": dict(link_modes),
        "annotation_previews": len(annotation_previews),
        "annotation_contact_sheets": [str(path) for path in annotation_contact],
        "manual_review_contact_sheets": manual_contact,
        "configs": [
            str(CONFIGS / "data_clean_6class.yaml"),
            str(CONFIGS / "data_clean_window_only.yaml"),
        ],
    }
    (REPORTS / "audit_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    reason_counts = Counter(row["reason"] for row in quarantine_manifest)
    report_lines = [
        "# 玻璃幕墙数据集清洗与准备报告",
        "",
        "## 执行结论",
        "",
        "- 原始数据保持不变，所有处理结果位于独立项目目录。",
        f"- 六类清洗集：训练 {prepared_counts['train']['images']} 张，验证 {prepared_counts['val']['images']} 张。",
        f"- 单类清洗集：训练 {prepared_counts['train']['images']} 张，验证 {prepared_counts['val']['images']} 张；标签中仅保留 Glass Window（ID 0）。",
        f"- 自动删除零面积标注行 {summary['automatic_fixes']['zero_area_lines_removed']} 条。",
        f"- 自动隔离同集合完全重复样本 {summary['automatic_fixes']['duplicate_samples_quarantined']} 张。",
        f"- 自动隔离跨训练/验证重复的训练样本 {summary['automatic_fixes']['cross_split_train_samples_quarantined']} 张。",
        f"- 已生成标注预览 {len(annotation_previews)} 张。",
        "",
        "## 需要人工确认",
        "",
    ]
    for reason, count in sorted(summary["manual_review"].items()):
        report_lines.append(f"- {reason}: {count} 张图片，已生成联系表并复制到隔离目录。")
    report_lines.extend(["", "## 隔离文件分类", ""])
    for reason, count in sorted(reason_counts.items()):
        report_lines.append(f"- {reason}: {count} 个文件副本。")
    report_lines.extend(
        [
            "",
            "## 生成文件",
            "",
            "- `reports/file_manifest.csv`：原始文件清单和 SHA256。",
            "- `reports/audit_report.json`：机器可读审计结果。",
            "- `reports/issues_and_fixes.csv`：问题及处理动作。",
            "- `reports/quarantine_manifest.csv`：隔离文件索引。",
            "- `reports/preview_index.csv`：标注预览索引。",
            "- `configs/data_clean_6class.yaml`：六类训练配置。",
            "- `configs/data_clean_window_only.yaml`：Glass Window 单类训练配置。",
            "",
            "## 重要说明",
            "",
            "- 隔离目录保存的是原文件副本，原始数据没有删除或覆盖。",
            "- 单类数据集图片通过 NTFS 硬链接复用六类清洗集图片；标签是独立文件。",
            "- 在人工确认空标签和冲突重复样本前，不把这些文件加入训练集。",
        ]
    )
    (REPORTS / "audit_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
