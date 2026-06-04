from __future__ import annotations

import argparse

from controllers.run_single_fold import run_single_fold
from models.configs import (
    AdaptiveSGHMCConfig,
    BlackJAXSGNHTConfig,
    DataConfig,
    EvalConfig,
    MLPConfig,
    OutputConfig,
    SGNHTConfig,
    SamplingBudget,
    SingleFoldRunConfig,
    ThermoConfig,
)
from samplers import SAMPLER_REGISTRY


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one ThermoBayes true-dataset fold.")

    # data
    p.add_argument("--dataset", type=str, default="yacht")
    p.add_argument("--fold_id", type=int, default=1)
    p.add_argument("--data_root", type=str, default="./datasets")
    p.add_argument("--has_header", action="store_true")
    p.add_argument("--no_normalize", action="store_true")
    p.add_argument("--device", type=str, default="cpu")

    # model / posterior
    p.add_argument("--hidden_units", type=int, default=20)
    p.add_argument("--hidden_layers", type=int, default=1)
    p.add_argument("--act_fn", type=str, default="leaky_relu", choices=["tanh", "relu", "leaky_relu"])
    p.add_argument("--prior_var", type=float, default=1.0)
    p.add_argument("--noise_var", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=32, help="<=0 means fullbatch")

    # sampler
    p.add_argument("--method", type=str, default="sgnht", choices=sorted(SAMPLER_REGISTRY.keys()))
    p.add_argument("--seed", type=int, default=77210)
    p.add_argument("--num_chains", type=int, default=4)
    p.add_argument("--num_samples", type=int, default=80)
    p.add_argument("--burn_in", type=int, default=500)
    p.add_argument("--keep_every", type=int, default=100)
    p.add_argument("--inner_steps", type=int, default=10)
    p.add_argument("--step_size", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--reset_prob", type=float, default=0.0)
    p.add_argument("--grad_noise_var", type=float, default=0.0)

    # SGHMC-specific
    p.add_argument("--mdecay", type=float, default=0.05)
    p.add_argument("--warm_up_frac", type=float, default=0.5)
    p.add_argument("--ema_beta", type=float, default=0.999)

    # SGNHT-specific
    p.add_argument("--diffusion_factor", type=float, default=1e-2)
    p.add_argument("--thermostat_init", type=float, default=None)
    p.add_argument("--no_reset_thermostat", action="store_true")
    p.add_argument("--reset_thermostat_to_A", action="store_true")
    p.add_argument("--no_reset_thermostat_to_midpoint", action="store_true")

    # BlackJAX-specific
    p.add_argument("--bjx_beta", type=float, default=0.0)

    # evaluation / output
    p.add_argument("--eval_raw_scale", action="store_true")
    p.add_argument("--discarded", type=int, default=0)
    p.add_argument("--print_every", type=int, default=0)
    p.add_argument("--online_split", type=str, default="val")
    p.add_argument("--save_root", type=str, default="./result")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--no_save_arrays", action="store_true")
    p.add_argument("--no_save_metrics", action="store_true")
    p.add_argument("--no_print_metrics", action="store_true")
    p.add_argument("--notes", type=str, default=None)
    return p


def make_sampler_config(args: argparse.Namespace):
    budget = SamplingBudget(
        seed=args.seed,
        num_chains=args.num_chains,
        num_samples=args.num_samples,
        burn_in=args.burn_in,
        keep_every=args.keep_every,
        inner_steps=args.inner_steps,
    )
    thermo = ThermoConfig(
        temperature=args.temperature,
        reset_prob=args.reset_prob,
        grad_noise_var=args.grad_noise_var,
    )

    if args.method == "adaptive_sghmc":
        return AdaptiveSGHMCConfig(
            name="adaptive_sghmc",
            step_size=args.step_size,
            budget=budget,
            thermo=thermo,
            mdecay=args.mdecay,
            warm_up_frac=args.warm_up_frac,
            ema_beta=args.ema_beta,
        )
    if args.method in {"sgnht", "sgnht_vectorized", "sgnht_vectorized_compile"}:
        return SGNHTConfig(
            name=args.method,
            step_size=args.step_size,
            budget=budget,
            thermo=thermo,
            diffusion_factor=args.diffusion_factor,
            thermostat_init=args.thermostat_init,
            reset_thermostat=not args.no_reset_thermostat,
            reset_thermostat_to_A=args.reset_thermostat_to_A,
            reset_thermostat_to_midpoint=not args.no_reset_thermostat_to_midpoint,
        )
    if args.method == "sgnht_bjx":
        return BlackJAXSGNHTConfig(
            name="sgnht_bjx",
            step_size=args.step_size,
            budget=budget,
            thermo=thermo,
            diffusion_factor=args.diffusion_factor,
            thermostat_init=args.thermostat_init,
            reset_thermostat=not args.no_reset_thermostat,
            reset_thermostat_to_A=args.reset_thermostat_to_A,
            reset_thermostat_to_midpoint=not args.no_reset_thermostat_to_midpoint,
            bjx_beta=args.bjx_beta,
        )
    raise ValueError(f"Unknown method: {args.method}")


def make_run_config(args: argparse.Namespace) -> SingleFoldRunConfig:
    return SingleFoldRunConfig(
        data=DataConfig(
            dataset=args.dataset,
            fold_id=args.fold_id,
            data_root=args.data_root,
            has_header=args.has_header,
            normalize=not args.no_normalize,
            device=args.device,
        ),
        model=MLPConfig(
            hidden_units=args.hidden_units,
            hidden_layers=args.hidden_layers,
            act_fn=args.act_fn,
            prior_var=args.prior_var,
            noise_var=args.noise_var,
            batch_size=args.batch_size,
        ),
        sampler=make_sampler_config(args),
        method=args.method,
        eval=EvalConfig(
            raw_scale=args.eval_raw_scale,
            discarded=args.discarded,
            print_every=args.print_every,
            online_split=args.online_split,
        ),
        output=OutputConfig(
            save_root=args.save_root,
            run_name=args.run_name,
            save_arrays=not args.no_save_arrays,
            save_metrics=not args.no_save_metrics,
            print_metrics=not args.no_print_metrics,
        ),
        notes=args.notes,
    )


def main() -> None:
    args = build_parser().parse_args()
    run_single_fold(make_run_config(args))


if __name__ == "__main__":
    main()

