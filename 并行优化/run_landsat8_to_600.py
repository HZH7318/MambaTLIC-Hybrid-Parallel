"""Resume the seven Landsat8 hybrid models to epoch 600 and finalize results.

The script keeps the existing run directories/checkpoint history intact.  It
creates explicit runtime configs, runs four lambdas sequentially on GPU 0 and
three on GPU 1, then refreshes the full-test RD results and benchmarks the
final real codec.  All publication-facing outputs are copied into one folder.
"""

from __future__ import annotations

import configparser
import csv
import json
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path("/data/huzhenghao/anaconda3/envs/mambah/bin/python")
TRAIN = ROOT / "run_mamba_hybrid_parallel.py"
SOURCE_CONFIGS = ROOT / "config" / "auto_to_train"
RUNTIME_CONFIGS = ROOT / "config" / "landsat8_resume_600"
FINAL = ROOT / "final_results" / "landsat8_hybrid_600"
LOGS = FINAL / "logs"

GPU_GROUPS = {
    0: (1024, 4096, 16384, 65536),
    1: (2048, 8192, 32768),
}


def prepare_config(lambda_value: int, gpu: int) -> Path:
    source = SOURCE_CONFIGS / f"{lambda_value}.ini"
    parser = configparser.ConfigParser()
    if not parser.read(source):
        raise FileNotFoundError(source)
    train = parser["trainconfig"]
    train["train_max_epoch"] = "600"
    train["device"] = f"cuda:{gpu}"
    train["tmuxh"] = f"landsat8_hybrid_600_gpu{gpu}"
    train["info"] = "hybrid_parallel_tail_landsat8_resume_to_600"
    test = parser["testconfig"]
    test["device"] = f"cuda:{gpu}"
    destination = RUNTIME_CONFIGS / f"gpu{gpu}" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        parser.write(handle)
    return destination


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{datetime.now().isoformat()}] RUN {' '.join(command)}\n")
        handle.flush()
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)


def train_gpu(gpu: int, lambdas: tuple[int, ...], failures: list[dict]) -> None:
    for lambda_value in lambdas:
        config = prepare_config(lambda_value, gpu)
        try:
            run_logged(
                [str(PYTHON), "-u", str(TRAIN), "--config", str(config)],
                LOGS / f"train_gpu{gpu}_lambda{lambda_value}.log",
            )
        except Exception as error:  # preserve the other GPU's useful work
            failures.append({"gpu": gpu, "lambda": lambda_value, "error": repr(error)})
            return


def benchmark_final_codec() -> list[dict]:
    rows = []
    runner = ROOT / "run_codec_benchmark.sh"
    methods = ("hybrid_python_serial", "hybrid_native_1w", "hybrid_native_4w")
    for method in methods:
        command = [
            str(runner), "--method", method,
            "--dataset", "landsat8", "--lambda-value", "16384",
            "--num-images", "30", "--warmup-images", "5", "--device", "cuda:0",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=True)
        (LOGS / f"codec_{method}.log").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        result_lines = [line for line in completed.stdout.splitlines() if line.startswith("RESULT_JSON=")]
        if not result_lines:
            raise RuntimeError(f"No RESULT_JSON from {method}")
        rows.append(json.loads(result_lines[-1].split("=", 1)[1]))
    output = FINAL / "landsat8_hybrid_codec_30_images.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def collect_final_outputs(codec_rows: list[dict]) -> None:
    comparison = ROOT / "experiments" / "comparison_all_methods"
    full_csv = comparison / "full_test_rd_all_methods.csv"
    shutil.copy2(full_csv, FINAL / full_csv.name)
    shutil.copy2(
        comparison / "landsat8_full_test_rd.png",
        FINAL / "landsat8_full_test_rd.png",
    )
    shutil.copy2(
        ROOT / "codec_time_comparison.csv",
        FINAL / "codec_time_comparison_pilot_5_images.csv",
    )

    # A Landsat-only table avoids making the final reader filter the combined file.
    with full_csv.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = [row for row in reader if row["Dataset"] == "landsat8"]
        with (FINAL / "landsat8_full_rd.csv").open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "completed_at": datetime.now().isoformat(),
        "training_target_epochs": 600,
        "rd_protocol": "forward likelihood on the complete 1700-image Landsat8 test set",
        "codec_protocol": "30 measured images after 5 warm-up images; real compress/decompress",
        "codec_rows": codec_rows,
        "primary_files": [
            "landsat8_full_rd.csv",
            "landsat8_full_test_rd.png",
            "landsat8_hybrid_codec_30_images.csv",
        ],
    }
    (FINAL / "result_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    FINAL.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    status_path = FINAL / "status.json"
    failures: list[dict] = []
    status_path.write_text(json.dumps({
        "state": "training",
        "started_at": datetime.now().isoformat(),
        "gpu_groups": GPU_GROUPS,
        "estimated_training_hours": "24-25",
    }, indent=2), encoding="utf-8")

    workers = [
        threading.Thread(target=train_gpu, args=(gpu, lambdas, failures), daemon=False)
        for gpu, lambdas in GPU_GROUPS.items()
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if failures:
        status_path.write_text(json.dumps({
            "state": "training_failed", "failures": failures,
            "updated_at": datetime.now().isoformat(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(f"Training failures: {failures}")

    status_path.write_text(json.dumps({
        "state": "evaluating", "updated_at": datetime.now().isoformat()
    }, indent=2), encoding="utf-8")
    run_logged(
        [str(PYTHON), "-u", str(ROOT / "run_full_comparison_eval.py"), "landsat8", "--device", "cuda:0"],
        LOGS / "full_test_landsat8.log",
    )
    run_logged(
        [str(PYTHON), "-u", str(ROOT / "organize_test_comparisons.py")],
        LOGS / "organize_results.log",
    )
    codec_rows = benchmark_final_codec()
    collect_final_outputs(codec_rows)
    status_path.write_text(json.dumps({
        "state": "complete", "completed_at": datetime.now().isoformat()
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
