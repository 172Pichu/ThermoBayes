from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("posteriors")

from benchmarks.t0_t4_ablation import BenchmarkConfig, run_t0_t4_ablation


def test_t0_t4_smoke_and_parity(tmp_path: Path) -> None:
    out_dir = tmp_path / "t0_t4"
    result = run_t0_t4_ablation(
        BenchmarkConfig(
            dataset="yacht",
            fold_id=1,
            data_root="./datasets",
            device="cpu",
            num_chains=1,
            warmup_steps=1,
            measure_steps=2,
            repeats=1,
            parity_steps=1,
            output_dir=str(out_dir),
            shuffle_variant_order=False,
            torch_num_threads=1,
        )
    )

    assert result["parity"]["all_variants_match"]
    assert len(result["summary"]) == 5
    assert {row["variant"] for row in result["summary"]} == {
        "T0_flat_autograd",
        "T1_flat_func_grad_and_value",
        "T2_tree_manual_outplace",
        "T3_posteriors_native_outplace",
        "T4_posteriors_native_inplace",
    }

    required = {
        "t0_t4_repeats.csv",
        "t0_t4_summary.csv",
        "t0_t4_decomposition.json",
        "t0_t4_parity.json",
        "t0_t4_environment.json",
        "t0_t4_config.json",
        "t0_t4_report.md",
    }
    assert required.issubset({path.name for path in out_dir.iterdir()})

    decomposition = json.loads((out_dir / "t0_t4_decomposition.json").read_text())
    assert decomposition["primary_endpoint"].startswith("T3_posteriors")
