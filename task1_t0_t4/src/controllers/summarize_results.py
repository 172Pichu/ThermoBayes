from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_GROUP_COLS = [
    "dataset",
    "method",
    "library",
    "backend",
    "optimization",
    "algorithm",
    "update_order",
    "vectorized_chains",
    "posteriors_inplace",
    "compile_requested",
    "compiled_step",
    "device",
    "normalize",
    "hidden_units",
    "hidden_layers",
    "act_fn",
    "batch_size",
    "diffusion_factor",
    "mdecay",
    "seed",
    "num_chains",
    "num_samples_per_chain",
    "burn_in",
    "keep_every",
    "inner_steps",
    "step_size",
    "prior_var",
    "noise_var",
    "eval_scale",
    "func_diag_enabled",
    "func_diag_split",
    "func_diag_max_points",
]

EXCLUDE_NUMERIC_SUMMARY_COLS = {
    "id",
    "fold_id",
    "hidden_units",
    "hidden_layers",
    "batch_size",
    "seed",
    "num_chains",
    "num_samples_per_chain",
    "burn_in",
    "keep_every",
    "inner_steps",
    "step_size",
    "prior_var",
    "noise_var",
    "temperature",
    "reset_prob",
    "grad_noise_var",
    "discarded",
    "print_every",
    "diffusion_factor",
    "mdecay",
    "normalize",
    "func_diag_enabled",
    "func_diag_max_points",
    "vectorized_chains",
    "posteriors_inplace",
    "compile_requested",
    "compiled_step",
}


def _read_csv_one_row(path: Path) -> dict[str, Any] | None:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    row = dict(rows[0])
    row["metrics_path"] = str(path)
    row["run_dir"] = str(path.parents[1])
    if "run_name" not in row or not row.get("run_name"):
        row["run_name"] = path.parents[1].name
    return row


def _maybe_number(x: Any) -> Any:
    if x is None:
        return x
    if isinstance(x, (int, float)):
        return x
    text = str(x).strip()
    if text == "":
        return text
    try:
        val = float(text)
    except ValueError:
        return x
    if np.isfinite(val) and val.is_integer() and not any(c in text.lower() for c in [".", "e"]):
        return int(val)
    return val


def _coerce_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: _maybe_number(v) for k, v in row.items()}


def load_metric_rows(save_root: str | Path, *, dataset: str | None = None, method: str | None = None) -> list[dict[str, Any]]:
    root = Path(save_root)
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.glob("*/records/metrics.csv")):
        row = _read_csv_one_row(metrics_path)
        if row is None:
            continue
        row = _coerce_row(row)
        if dataset is not None and str(row.get("dataset", "")) != str(dataset):
            continue
        if method is not None and str(row.get("method", "")) != str(method):
            continue
        rows.append(row)
    return rows


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(float(x))


def _group_key(row: dict[str, Any], group_cols: list[str]) -> tuple[Any, ...]:
    return tuple(row.get(c, "") for c in group_cols)


def summarize_metric_rows(rows: list[dict[str, Any]], *, group_cols: list[str] | None = None) -> list[dict[str, Any]]:
    if not rows:
        return []
    group_cols = [c for c in (group_cols or DEFAULT_GROUP_COLS) if any(c in row for row in rows)]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_group_key(row, group_cols), []).append(row)

    summaries: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        rec: dict[str, Any] = {col: val for col, val in zip(group_cols, key)}
        fold_vals = []
        for r in group_rows:
            if _is_number(r.get("id")):
                fold_vals.append(int(r["id"]))
            elif _is_number(r.get("fold_id")):
                fold_vals.append(int(r["fold_id"]))
        rec["num_runs"] = len(group_rows)
        rec["num_folds"] = len(set(fold_vals)) if fold_vals else len(group_rows)
        rec["fold_ids"] = ",".join(str(x) for x in sorted(set(fold_vals))) if fold_vals else ""

        all_cols = sorted(set().union(*(r.keys() for r in group_rows)))
        metric_cols = []
        for col in all_cols:
            if col in group_cols or col in EXCLUDE_NUMERIC_SUMMARY_COLS:
                continue
            vals = [r.get(col) for r in group_rows]
            if any(_is_number(v) for v in vals):
                metric_cols.append(col)

        for col in metric_cols:
            vals = np.asarray([float(r[col]) for r in group_rows if _is_number(r.get(col))], dtype=np.float64)
            if vals.size == 0:
                continue
            rec[f"{col}_mean"] = float(np.mean(vals))
            rec[f"{col}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan")
        summaries.append(rec)
    return summaries


def _write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_markdown(rows: list[dict[str, Any]], path: Path, *, max_cols: int = 40) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No rows found.\n", encoding="utf-8")
        return path
    cols = list(rows[0].keys())[:max_cols]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        vals = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def summarize_result_root(
    *,
    save_root: str | Path = "./result",
    dataset: str | None = None,
    method: str | None = None,
    out_dir: str | Path | None = None,
    summary_name: str | None = None,
) -> dict[str, Path]:
    """Read existing result directories and write mean/std benchmark summaries.

    This function never runs samplers. It only consumes existing
    result/*/records/metrics.csv files.
    """
    root = Path(save_root)
    rows = load_metric_rows(root, dataset=dataset, method=method)
    summaries = summarize_metric_rows(rows)

    out_root = Path(out_dir) if out_dir is not None else root / "_summaries"
    name_parts = [summary_name or "summary"]
    if dataset is not None:
        name_parts.append(str(dataset))
    if method is not None:
        name_parts.append(str(method))
    stem = "_".join(name_parts)

    paths = {
        "runs_csv": _write_csv(rows, out_root / f"{stem}_runs.csv") if rows else out_root / f"{stem}_runs.csv",
        "mean_std_csv": _write_csv(summaries, out_root / f"{stem}_mean_std.csv") if summaries else out_root / f"{stem}_mean_std.csv",
        "mean_std_md": _write_markdown(summaries, out_root / f"{stem}_mean_std.md"),
    }
    if not rows:
        paths["runs_csv"].parent.mkdir(parents=True, exist_ok=True)
        paths["runs_csv"].write_text("", encoding="utf-8")
        paths["mean_std_csv"].parent.mkdir(parents=True, exist_ok=True)
        paths["mean_std_csv"].write_text("", encoding="utf-8")

    print("\n==== Benchmark Summary ====")
    print(f"save_root = {root}")
    print(f"dataset   = {dataset if dataset is not None else 'ALL'}")
    print(f"method    = {method if method is not None else 'ALL'}")
    print(f"runs      = {len(rows)}")
    print(f"groups    = {len(summaries)}")
    print(f"Saved runs CSV     : {paths['runs_csv']}")
    print(f"Saved mean/std CSV : {paths['mean_std_csv']}")
    print(f"Saved mean/std MD  : {paths['mean_std_md']}")
    return paths


__all__ = ["load_metric_rows", "summarize_metric_rows", "summarize_result_root"]
