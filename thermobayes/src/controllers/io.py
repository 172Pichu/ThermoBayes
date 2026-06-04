from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from controllers.metrics import flatten_metrics_record
from models.results import MetricsRecord, SamplingResult, SingleFoldResult


class ExperimentPaths:
    """Output path helper. It saves files, but does not decide experiment flow."""

    def __init__(self, save_root: str | Path = "./result", run_name: str = "experiment") -> None:
        self.save_root = Path(save_root)
        self.run_name = str(run_name)

    @property
    def run_dir(self) -> Path:
        path = self.save_root / self.run_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def figures_dir(self) -> Path:
        path = self.run_dir / "figures"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def arrays_dir(self) -> Path:
        path = self.run_dir / "arrays"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def records_dir(self) -> Path:
        path = self.run_dir / "records"
        path.mkdir(parents=True, exist_ok=True)
        return path


def _json_ready(obj: Any) -> Any:
    if is_dataclass(obj):
        return _json_ready(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_metrics_json(record: MetricsRecord, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(record), f, indent=2, ensure_ascii=False)
    return path


def save_metrics_csv(record: MetricsRecord, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = flatten_metrics_record(record)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return path


def save_sampling_arrays(result: SamplingResult, paths: ExperimentPaths) -> dict[str, str]:
    out: dict[str, str] = {}
    np.save(paths.arrays_dir / "chains.npy", np.asarray(result.chains))
    np.save(paths.arrays_dir / "samples.npy", np.asarray(result.samples))
    out["chains"] = str(paths.arrays_dir / "chains.npy")
    out["samples"] = str(paths.arrays_dir / "samples.npy")
    return out


def save_run_summary(result: SingleFoldResult, path: str | Path | None = None) -> Path:
    out_path = Path(path) if path is not None else result.run_dir / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_dir": result.run_dir,
        "sampling_result": result.sampling_result,
        "metrics_record": result.metrics_record,
        "figure_paths": result.figure_paths,
        "extra": result.extra,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2, ensure_ascii=False)
    return out_path


__all__ = [
    "ExperimentPaths",
    "save_metrics_json",
    "save_metrics_csv",
    "save_sampling_arrays",
    "save_run_summary",
]

