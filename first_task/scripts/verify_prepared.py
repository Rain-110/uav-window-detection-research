from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DATASETS = {
    "prepared_6class": set(range(6)),
    "prepared_window_only": {0},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset(name: str, allowed_classes: set[int]) -> dict:
    root = PROJECT / name
    result: dict = {}
    hashes: dict[str, set[str]] = {}
    for split in ("train", "val"):
        images = sorted(path for path in (root / "images" / split).iterdir() if path.is_file())
        labels = sorted((root / "labels" / split).glob("*.txt"))
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        bad_lines: list[str] = []
        box_count = 0
        for label in labels:
            for number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split()
                valid = len(parts) == 5
                if valid:
                    try:
                        class_id = int(parts[0])
                        x, y, width, height = map(float, parts[1:])
                        valid = (
                            class_id in allowed_classes
                            and all(0 <= value <= 1 for value in (x, y, width, height))
                            and width > 0
                            and height > 0
                        )
                    except ValueError:
                        valid = False
                if not valid:
                    bad_lines.append(f"{label.name}:{number}")
                box_count += 1
        hashes[split] = {sha256(path) for path in images}
        result[split] = {
            "images": len(images),
            "labels": len(labels),
            "missing_labels": sorted(image_stems - label_stems),
            "orphan_labels": sorted(label_stems - image_stems),
            "bad_lines": bad_lines,
            "boxes": box_count,
        }
    result["cross_split_duplicate_hashes"] = len(hashes["train"] & hashes["val"])
    return result


def main() -> None:
    report = {name: verify_dataset(name, classes) for name, classes in DATASETS.items()}
    yaml_checks = {}
    for yaml_path in sorted((PROJECT / "configs").glob("*.yaml")):
        text = yaml_path.read_text(encoding="utf-8")
        yaml_checks[yaml_path.name] = {
            "has_train": "train: images/train" in text,
            "has_val": "val: images/val" in text,
            "has_absolute_project_path": PROJECT.as_posix() in text.replace("\\", "/"),
        }
    report["yaml_checks"] = yaml_checks
    output = PROJECT / "reports" / "verification_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    problems = []
    for name in DATASETS:
        for split in ("train", "val"):
            item = report[name][split]
            if item["images"] != item["labels"] or item["missing_labels"] or item["orphan_labels"] or item["bad_lines"]:
                problems.append(f"{name}/{split}")
        if report[name]["cross_split_duplicate_hashes"]:
            problems.append(f"{name}/cross_split_duplicates")
    if not all(all(check.values()) for check in yaml_checks.values()):
        problems.append("yaml")
    if problems:
        raise SystemExit("Verification failed: " + ", ".join(problems))


if __name__ == "__main__":
    main()
