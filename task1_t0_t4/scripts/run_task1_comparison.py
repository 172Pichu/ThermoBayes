from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


BASE_METHODS = [
    "sgnht",
    "sgnht_posteriors_order",
    "sgnht_vectorized",
    "sgnht_posteriors_order_vectorized",
    "posteriors_sgnht",
    "posteriors_sgnht_inplace",
]

COMPILE_METHODS = [
    "sgnht_vectorized_compile",
    "sgnht_posteriors_order_vectorized_compile",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the Task-1 SGNHT ablation with one shared data/model/sampling "
            "configuration. Existing ThermoBayes CLI arguments are forwarded."
        )
    )
    p.add_argument("--id", type=int, default=1, help="Fold id; use -1 for all folds.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--data_root", type=str, default="./datasets")
    p.add_argument("--save_root", type=str, default="./result_task1")
    p.add_argument("--include_compile", action="store_true")
    p.add_argument(
        "--skip_posteriors",
        action="store_true",
        help="Run only native ThermoBayes variants, even if posteriors is installed.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional explicit method list; overrides the default ablation set.",
    )
    p.add_argument("--seed", type=int, default=77210)
    p.add_argument("--num_chains", type=int, default=4)
    p.add_argument("--num_samples", type=int, default=80)
    p.add_argument("--burn_in", type=int, default=500)
    p.add_argument("--keep_every", type=int, default=100)
    p.add_argument("--inner_steps", type=int, default=10)
    p.add_argument("--step_size", type=float, default=1e-3)
    p.add_argument("--diffusion_factor", type=float, default=1e-2)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hidden_units", type=int, default=20)
    p.add_argument("--hidden_layers", type=int, default=1)
    p.add_argument("--act_fn", type=str, default="leaky_relu")
    p.add_argument("--prior_var", type=float, default=1.0)
    p.add_argument("--noise_var", type=float, default=1.0)
    p.add_argument("--print_every", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_func_diag", action="store_true")
    return p


def _posteriors_available() -> bool:
    return importlib.util.find_spec("posteriors") is not None


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    cli_path = project_root / "src" / "cli.py"

    methods = list(args.methods) if args.methods else list(BASE_METHODS)
    if args.include_compile and args.methods is None:
        methods.extend(COMPILE_METHODS)

    if args.skip_posteriors or not _posteriors_available():
        native = [m for m in methods if not m.startswith("posteriors")]
        skipped = [m for m in methods if m.startswith("posteriors")]
        methods = native
        if skipped:
            reason = "--skip_posteriors" if args.skip_posteriors else "the posteriors package is not installed"
            print(f"Skipping {skipped}: {reason}.")

    common = [
        sys.executable,
        str(cli_path),
        "--dataset", "yacht",
        "--id", str(args.id),
        "--data_root", str(args.data_root),
        "--device", args.device,
        "--save_root", str(args.save_root),
        "--seed", str(args.seed),
        "--num_chains", str(args.num_chains),
        "--num_samples", str(args.num_samples),
        "--burn_in", str(args.burn_in),
        "--keep_every", str(args.keep_every),
        "--inner_steps", str(args.inner_steps),
        "--step_size", str(args.step_size),
        "--diffusion_factor", str(args.diffusion_factor),
        "--temperature", str(args.temperature),
        "--batch_size", str(args.batch_size),
        "--hidden_units", str(args.hidden_units),
        "--hidden_layers", str(args.hidden_layers),
        "--act_fn", args.act_fn,
        "--prior_var", str(args.prior_var),
        "--noise_var", str(args.noise_var),
        "--print_every", str(args.print_every),
    ]
    if args.overwrite:
        common.append("--overwrite")
    if args.no_func_diag:
        common.append("--no_func_diag")

    print("\nTask-1 methods:")
    for method in methods:
        print(f"  - {method}")

    for method in methods:
        print(f"\n{'=' * 72}\nRunning {method}\n{'=' * 72}")
        subprocess.run(common + ["--method", method], cwd=project_root, check=True)

    print(f"\n{'=' * 72}\nSummarizing all Task-1 runs\n{'=' * 72}")
    subprocess.run(
        [
            sys.executable,
            str(cli_path),
            "--summarize",
            "--dataset", "yacht",
            "--save_root", str(args.save_root),
            "--summary_name", "task1_sgnht_ablation",
        ],
        cwd=project_root,
        check=True,
    )


if __name__ == "__main__":
    main()
