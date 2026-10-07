from __future__ import annotations

import json
import os
import platform
import statistics
import threading
import time
from pathlib import Path

import psutil
import torch
import ultralytics
from ultralytics import YOLO


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_WEIGHTS = (
    PROJECT
    / "runs"
    / "yolo11n_window_cpu_1h_20260928"
    / "weights"
    / "best.pt"
)
DATA = PROJECT / "configs" / "data_clean_window_only.yaml"
RUN_NAME = "yolo11n_window_cpu_2h_recall_20260929"
RUN_DIR = PROJECT / "runs" / RUN_NAME
REPORT_PATH = PROJECT / "reports" / "two_hour_cpu_runtime.json"
PROGRESS_PATH = PROJECT / "reports" / "two_hour_cpu_epoch_progress.json"

MAX_EPOCHS = 14
TRAINING_CUTOFF_SECONDS = 105 * 60


def main() -> None:
    if not SOURCE_WEIGHTS.exists():
        raise FileNotFoundError(SOURCE_WEIGHTS)
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    if RUN_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {RUN_DIR}")

    process = psutil.Process(os.getpid())
    monitor_stop = threading.Event()
    samples: list[dict[str, float]] = []
    epoch_progress: list[dict[str, float | int | bool]] = []
    train_start = time.perf_counter()

    def monitor() -> None:
        process.cpu_percent(None)
        psutil.cpu_percent(None)
        while not monitor_stop.wait(2.0):
            samples.append(
                {
                    "process_cpu_percent": process.cpu_percent(None),
                    "system_cpu_percent": psutil.cpu_percent(None),
                    "rss_bytes": float(process.memory_info().rss),
                }
            )

    def on_fit_epoch_end(trainer) -> None:
        epoch_number = int(trainer.epoch + 1)
        # A final best.pt validation can trigger this callback once more.
        if epoch_number > MAX_EPOCHS or (
            epoch_progress and epoch_number <= int(epoch_progress[-1]["epoch"])
        ):
            return
        elapsed = time.perf_counter() - train_start
        epoch_seconds = float(getattr(trainer, "epoch_time", 0.0) or 0.0)
        estimate = epoch_seconds if epoch_seconds > 0 else elapsed / max(epoch_number, 1)
        stop_for_budget = (
            epoch_number < MAX_EPOCHS
            and elapsed + estimate > TRAINING_CUTOFF_SECONDS
        )
        row = {
            "epoch": epoch_number,
            "elapsed_seconds": elapsed,
            "epoch_seconds": epoch_seconds,
            "estimated_next_epoch_seconds": estimate,
            "stop_for_budget": stop_for_budget,
        }
        epoch_progress.append(row)
        PROGRESS_PATH.write_text(
            json.dumps(epoch_progress, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("\nTWO_HOUR_TIME_BUDGET_STATUS", flush=True)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if stop_for_budget:
            trainer.stop = True

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    status = "success"
    error = ""
    trainer_metrics: dict[str, float] = {}
    try:
        model = YOLO(str(SOURCE_WEIGHTS))
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model.train(
            data=str(DATA),
            epochs=MAX_EPOCHS,
            imgsz=640,
            batch=2,
            device="cpu",
            workers=0,
            optimizer="AdamW",
            lr0=0.0005,
            lrf=0.1,
            cos_lr=True,
            warmup_epochs=0.5,
            project=str(PROJECT / "runs"),
            name=RUN_NAME,
            seed=42,
            deterministic=True,
            cache="ram",
            plots=True,
            amp=False,
            max_det=1000,
            save_period=1,
            patience=MAX_EPOCHS,
            mosaic=1.0,
            close_mosaic=2,
            mixup=0.0,
            copy_paste=0.0,
            verbose=True,
        )
        raw_metrics = getattr(model.trainer, "metrics", {}) or {}
        trainer_metrics = {
            str(key): float(value)
            for key, value in raw_metrics.items()
            if isinstance(value, (int, float))
        }
    except Exception as exc:
        status = "failed"
        error = repr(exc)
        raise
    finally:
        elapsed = time.perf_counter() - train_start
        monitor_stop.set()
        thread.join(timeout=5)
        process_cpu = [sample["process_cpu_percent"] for sample in samples]
        system_cpu = [sample["system_cpu_percent"] for sample in samples]
        rss = [sample["rss_bytes"] for sample in samples]
        report = {
            "status": status,
            "error": error,
            "started_at": started_at,
            "elapsed_seconds": elapsed,
            "run_dir": str(RUN_DIR),
            "source_weights": str(SOURCE_WEIGHTS),
            "data": str(DATA),
            "completed_epochs": len(epoch_progress),
            "epoch_progress": epoch_progress,
            "parameters": {
                "epochs_max": MAX_EPOCHS,
                "training_cutoff_seconds": TRAINING_CUTOFF_SECONDS,
                "imgsz": 640,
                "batch": 2,
                "device": "cpu",
                "workers": 0,
                "optimizer": "AdamW",
                "lr0": 0.0005,
                "lrf": 0.1,
                "cos_lr": True,
                "warmup_epochs": 0.5,
                "cache": "ram",
                "max_det": 1000,
                "save_period": 1,
                "mosaic": 1.0,
                "close_mosaic": 2,
                "mixup": 0.0,
                "copy_paste": 0.0,
                "seed": 42,
                "deterministic": True,
                "amp": False,
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "ultralytics": ultralytics.__version__,
                "cuda_available": torch.cuda.is_available(),
                "logical_cpu_count": psutil.cpu_count(logical=True),
                "physical_cpu_count": psutil.cpu_count(logical=False),
                "total_memory_gib": psutil.virtual_memory().total / 1024**3,
            },
            "resources": {
                "sample_count": len(samples),
                "process_cpu_percent_mean": statistics.fmean(process_cpu)
                if process_cpu
                else None,
                "process_cpu_percent_peak": max(process_cpu) if process_cpu else None,
                "system_cpu_percent_mean": statistics.fmean(system_cpu)
                if system_cpu
                else None,
                "system_cpu_percent_peak": max(system_cpu) if system_cpu else None,
                "process_rss_peak_gib": max(rss) / 1024**3 if rss else None,
            },
            "trainer_metrics": trainer_metrics,
        }
        REPORT_PATH.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("\nTWO_HOUR_RUNTIME_REPORT", flush=True)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
