from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

T0 = "T0_flat_autograd"
T1 = "T1_flat_func_grad_and_value"
T2 = "T2_tree_manual_outplace"
T3 = "T3_posteriors_native_outplace"
T4 = "T4_posteriors_native_inplace"
VARIANTS = (T0, T1, T2, T3, T4)


def load_repeats(path: Path) -> dict[int, dict[str, float]]:
    grouped: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            repeat = int(row["repeat"])
            variant = row["variant"]
            if variant not in VARIANTS:
                continue
            grouped.setdefault(repeat, {})[variant] = float(row["duration_sec"])

    incomplete = {
        repeat: sorted(set(VARIANTS) - set(values))
        for repeat, values in grouped.items()
        if set(values) != set(VARIANTS)
    }
    if incomplete:
        raise ValueError(f"Incomplete repeats: {incomplete}")
    if not grouped:
        raise ValueError(f"No T0-T4 records found in {path}")
    return grouped


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "positive_repeats": sum(value > 0 for value in values),
        "num_repeats": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the paired T0-T4 decomposition from t0_t4_repeats.csv."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory produced by run_t0_t4_ablation.py",
    )
    args = parser.parse_args()

    repeats_path = args.output_dir / "t0_t4_repeats.csv"
    grouped = load_repeats(repeats_path)

    ordered = [grouped[key] for key in sorted(grouped)]
    means = {
        variant: statistics.fmean([row[variant] for row in ordered])
        for variant in VARIANTS
    }

    paired = {
        "T1_minus_T0_functional_gradient_pathway": [row[T1] - row[T0] for row in ordered],
        "T2_minus_T1_state_tree_outplace_layer": [row[T2] - row[T1] for row in ordered],
        "T3_minus_T2_residual_native_wrapper": [row[T3] - row[T2] for row in ordered],
        "T3_minus_T4_inplace_saving": [row[T3] - row[T4] for row in ordered],
        "T3_minus_T0_total_gap": [row[T3] - row[T0] for row in ordered],
    }

    total_gap = statistics.fmean(paired["T3_minus_T0_total_gap"])
    results = {
        "num_repeats": len(ordered),
        "mean_runtime_sec": means,
        "T3_over_T0_ratio_of_means": means[T3] / means[T0],
        "paired_differences": {},
    }

    additive_names = {
        "T1_minus_T0_functional_gradient_pathway",
        "T2_minus_T1_state_tree_outplace_layer",
        "T3_minus_T2_residual_native_wrapper",
    }
    for name, values in paired.items():
        row = summarize(values)
        if name in additive_names:
            row["share_of_T3_minus_T0_percent"] = (
                100.0 * float(row["mean"]) / total_gap if total_gap != 0 else float("nan")
            )
        results["paired_differences"][name] = row

    results["T3_minus_T4_percent_of_T3"] = (
        100.0
        * float(results["paired_differences"]["T3_minus_T4_inplace_saving"]["mean"])
        / means[T3]
    )

    out_json = args.output_dir / "t0_t4_paired_analysis.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== Mean runtime ===")
    for variant in VARIANTS:
        print(f"{variant:<38} {means[variant]:.6f} s")
    print(f"\nT3/T0 ratio of means: {results['T3_over_T0_ratio_of_means']:.3f}x")

    print("\n=== Within-repeat paired differences ===")
    for name, row in results["paired_differences"].items():
        share = row.get("share_of_T3_minus_T0_percent")
        share_text = f", share={share:.1f}%" if isinstance(share, float) else ""
        print(
            f"{name}: mean={row['mean']:.6f}s, median={row['median']:.6f}s, "
            f"positive={row['positive_repeats']}/{row['num_repeats']}{share_text}"
        )

    print(f"\nIn-place saving as % of T3: {results['T3_minus_T4_percent_of_T3']:.1f}%")
    print(f"Saved paired analysis to: {out_json.resolve()}")


if __name__ == "__main__":
    main()
