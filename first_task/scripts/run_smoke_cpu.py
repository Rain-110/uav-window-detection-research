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
PYTHON_ENV = Path(os.sys.executable)
WEIGHTS = PROJECT / "yolo11n.pt"
DATA = PROJECT / "configs" / "data_clean_window_only.yaml"
RUN_NAME = "smoke_cpu_window_20260928"
RUN_DIR = PROJECT / "runs" / RUN_NAME
REPORT_PATH = PROJECT / "reports" / "smoke_cpu_runtime.json"


def main() -> None:
    if not WEIGHTS.exists():
        raise FileNotFoundError(WEIGHTS)
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    if RUN_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {RUN_DIR}")

    process = psutil.Process(os.getpid())
    stop = threading.Event()
    samples: list[dict[str, float]] = []

    def monitor() -> None:
        process.cpu_percent(None)
        psutil.cpu_percent(None)
        while not stop.wait(1.0):
            samples.append(
                {
                    "process_cpu_percent": process.cpu_percent(None),
                    "system_cpu_percent": psutil.cpu_percent(None),
                    "rss_bytes": float(process.memory_info().rss),
                }
            )

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    start = time.perf_counter()
    status = "success"
    error = ""
    trainer_metrics: dict[str, float] = {}
    try:
        model = YOLO(str(WEIGHTS))
        model.train(
            data=str(DATA),
            epochs=1,
            imgsz=320,
            batch=2,
            device="cpu",
            workers=0,
            fraction=0.05,
            project=str(PROJECT / "runs"),
            name=RUN_NAME,
            seed=42,
            deterministic=True,
            cache=False,
            plots=True,
            amp=False,
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
        elapsed = time.perf_counter() - start
        stop.set()
        thread.join(timeout=3)
        process_cpu = [sample["process_cpu_percent"] for sample in samples]
        system_cpu = [sample["system_cpu_percent"] for sample in samples]
        rss = [sample["rss_bytes"] for sample in samples]
        report = {
            "status": status,
            "error": error,
            "started_at": started_at,
            "elapsed_seconds": elapsed,
            "run_dir": str(RUN_DIR),
            "weights": str(WEIGHTS),
            "data": str(DATA),
            "parameters": {
                "epochs": 1,
                "imgsz": 320,
                "batch": 2,
                "device": "cpu",
                "workers": 0,
                "fraction": 0.05,
                "seed": 42,
                "deterministic": True,
                "amp": False,
            },
            "environment": {
                "python": platform.python_version(),
                "python_executable": str(PYTHON_ENV),
                "torch": torch.__version__,
                "ultralytics": ultralytics.__version__,
                "cuda_available": torch.cuda.is_available(),
                "logical_cpu_count": psutil.cpu_count(logical=True),
                "physical_cpu_count": psutil.cpu_count(logical=False),
                "total_memory_gib": psutil.virtual_memory().total / 1024**3,
            },
            "resources": {
                "sample_count": len(samples),
                "process_cpu_percent_mean": statistics.fmean(process_cpu) if process_cpu else None,
                "process_cpu_percent_peak": max(process_cpu) if process_cpu else None,
                "system_cpu_percent_mean": statistics.fmean(system_cpu) if system_cpu else None,
                "system_cpu_percent_peak": max(system_cpu) if system_cpu else None,
                "process_rss_peak_gib": max(rss) / 1024**3 if rss else None,
            },
            "trainer_metrics": trainer_metrics,
        }
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nSMOKE_RUNTIME_REPORT")
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
