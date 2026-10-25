from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmarks.t0_t4_ablation import BenchmarkConfig, run_t0_t4_ablation


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Controlled T0-T4 decomposition of the per-chain ThermoBayes versus "
            "POSTERIORS SGNHT runtime gap."
        )
    )
    p.add_argument("--dataset", type=str, default="yacht")
    p.add_argument("--id", "--fold_id", dest="fold_id", type=int, default=1)
    p.add_argument("--data_root", type=str, default="./datasets")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no_normalize", action="store_true")

    p.add_argument("--hidden_units", type=int, default=20)
    p.add_argument("--hidden_layers", type=int, default=1)
    p.add_argument("--act_fn", type=str, default="leaky_relu", choices=["tanh", "relu", "leaky_relu"])
    p.add_argument("--prior_var", type=float, default=1.0)
    p.add_argument("--noise_var", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=32)

    p.add_argument("--seed", type=int, default=77210)
    p.add_argument("--num_chains", type=int, default=4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--steps", "--measure_steps", dest="measure_steps", type=int, default=2000)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--parity_steps", type=int, default=10)
    p.add_argument("--step_size", type=float, default=1e-3)
    p.add_argument("--diffusion_factor", type=float, default=1e-2)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--thermostat_init", type=float, default=None)

    p.add_argument("--output_dir", type=str, default="./result_task1_t0_t4")
    p.add_argument(
        "--reference_ratio",
        type=float,
        default=None,
        help=(
            "Optional full-run POSTERIORS/plain runtime ratio, e.g. 2.22. The report checks "
            "whether the controlled T3/T0 benchmark reproduces it within 15 percent."
        ),
    )
    p.add_argument(
        "--torch_num_threads",
        type=int,
        default=None,
        help="Optional torch CPU intra-op thread count for reproducible CPU runs.",
    )
    p.add_argument(
        "--fixed_order",
        action="store_true",
        help="Run T0,T1,T2,T3,T4 in fixed order instead of shuffling order per repeat.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    config = BenchmarkConfig(
        dataset=args.dataset,
        fold_id=args.fold_id,
        data_root=args.data_root,
        device=args.device,
        normalize=not args.no_normalize,
        hidden_units=args.hidden_units,
        hidden_layers=args.hidden_layers,
        act_fn=args.act_fn,
        prior_var=args.prior_var,
        noise_var=args.noise_var,
        batch_size=args.batch_size,
        seed=args.seed,
        num_chains=args.num_chains,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        repeats=args.repeats,
        parity_steps=args.parity_steps,
        step_size=args.step_size,
        diffusion_factor=args.diffusion_factor,
        temperature=args.temperature,
        thermostat_init=args.thermostat_init,
        output_dir=args.output_dir,
        reference_ratio=args.reference_ratio,
        shuffle_variant_order=not args.fixed_order,
        torch_num_threads=args.torch_num_threads,
    )
    run_t0_t4_ablation(config)


if __name__ == "__main__":
    main()
