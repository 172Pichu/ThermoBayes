from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from controllers.metrics import build_function_chains, final_evaluate, online_evaluate, print_metrics_record
from models.configs import BaseSamplerConfig, SingleFoldRunConfig
from models.results import SingleFoldResult
from models.targets import make_bnn_regression_target_j, make_bnn_regression_target_t
from models.dataset.networks import MLPArchitecture, make_predictive_samples_fn_j, make_predictive_samples_fn_t
from models.dataset.params import load_regression_splits_t
from samplers import SAMPLER_REGISTRY
from controllers.io import (
    ExperimentPaths,
    save_metrics_csv,
    save_metrics_json,
    save_online_metrics_csv,
    save_run_summary,
    save_sampling_arrays,
)


def _fmt_value(value: Any) -> str:
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p").replace("/", "_").replace("\\", "_")


def classify_method(method: str) -> dict[str, str]:
    """Map a dispatch method to benchmark protocol labels."""
    if method == "adaptive_sghmc":
        return {"library": "torch", "backend": "torch", "optimization": "plain", "algorithm": "adaptive_sghmc"}
    if method == "sgnht":
        return {"library": "torch", "backend": "torch", "optimization": "plain", "algorithm": "sgnht"}
    if method == "sgnht_posteriors_order":
        return {"library": "torch", "backend": "torch", "optimization": "plain_posteriors_order", "algorithm": "sgnht"}
    if method == "sgnht_vectorized":
        return {"library": "torch", "backend": "torch", "optimization": "vectorizing", "algorithm": "sgnht"}
    if method == "sgnht_vectorized_compile":
        return {"library": "torch", "backend": "torch", "optimization": "compiling", "algorithm": "sgnht"}
    if method == "sgnht_posteriors_order_vectorized":
        return {"library": "torch", "backend": "torch", "optimization": "vectorizing_posteriors_order", "algorithm": "sgnht"}
    if method == "sgnht_posteriors_order_vectorized_compile":
        return {"library": "torch", "backend": "torch", "optimization": "compiling_posteriors_order", "algorithm": "sgnht"}
    if method == "sgnht_bjx":
        return {"library": "blackjax", "backend": "jax", "optimization": "native_jax", "algorithm": "sgnht"}
    if method in {"posteriors", "posteriors_sgnht"}:
        return {"library": "posteriors", "backend": "torch", "optimization": "native", "algorithm": "sgnht"}
    if method == "posteriors_sgnht_inplace":
        return {"library": "posteriors", "backend": "torch", "optimization": "native_inplace", "algorithm": "sgnht"}
    if method == "posteriors_sghmc":
        return {"library": "posteriors", "backend": "torch", "optimization": "native", "algorithm": "sghmc"}
    return {"library": "unknown", "backend": "unknown", "optimization": "unknown", "algorithm": method}


def _default_run_name(config: SingleFoldRunConfig, eval_note: str) -> str:
    labels = classify_method(config.method)
    bs = "fullbatch" if int(config.model.batch_size) <= 0 else f"mb{int(config.model.batch_size)}"
    seed = config.sampler.budget.seed if config.sampler is not None else "na"
    return (
        f"true_{config.data.dataset}"
        f"_id{config.data.fold_id}"
        f"_{labels['library']}_{labels['algorithm']}_{labels['optimization']}"
        f"_{bs}"
        f"_h{config.model.hidden_units}x{config.model.hidden_layers}_{config.model.act_fn}"
        f"_pvar{_fmt_value(config.model.prior_var)}_nvar{_fmt_value(config.model.noise_var)}"
        f"_seed{seed}"
        f"_C{config.sampler.budget.num_chains}_S{config.sampler.budget.num_samples}"
        f"_bi{config.sampler.budget.burn_in}_ke{config.sampler.budget.keep_every}_is{config.sampler.budget.inner_steps}"
        f"_eps{_fmt_value(config.sampler.step_size)}"
        f"_{eval_note}"
    )


def _unique_run_name(save_root: str | Path, base_name: str, *, overwrite: bool = False) -> str:
    if overwrite:
        return base_name
    root = Path(save_root)
    if not (root / base_name).exists():
        return base_name
    idx = 1
    while True:
        candidate = f"{base_name}_rep{idx:03d}"
        if not (root / candidate).exists():
            return candidate
        idx += 1


def _build_protocol_metadata(config: SingleFoldRunConfig, *, run_name: str, eval_note: str, target: Any) -> dict[str, Any]:
    labels = classify_method(config.method)
    budget = config.sampler.budget
    meta: dict[str, Any] = {
        "dataset": config.data.dataset,
        "id": int(config.data.fold_id),
        "fold_id": int(config.data.fold_id),
        "method": config.method,
        "library": labels["library"],
        "backend": labels["backend"],
        "optimization": labels["optimization"],
        "algorithm": labels["algorithm"],
        "run_name": run_name,
        "device": config.data.device,
        "normalize": bool(config.data.normalize),
        "eval_scale": eval_note,
        "hidden_units": int(config.model.hidden_units),
        "hidden_layers": int(config.model.hidden_layers),
        "act_fn": config.model.act_fn,
        "prior_var": float(config.model.prior_var),
        "noise_var": float(config.model.noise_var),
        "batch_size": int(config.model.batch_size),
        "seed": int(budget.seed),
        "num_chains": int(budget.num_chains),
        "num_samples_per_chain": int(budget.num_samples),
        "burn_in": int(budget.burn_in),
        "keep_every": int(budget.keep_every),
        "inner_steps": int(budget.inner_steps),
        "step_size": float(config.sampler.step_size),
        "temperature": float(config.sampler.thermo.temperature),
        "reset_prob": float(config.sampler.thermo.reset_prob),
        "grad_noise_var": float(config.sampler.thermo.grad_noise_var),
        "discarded": int(config.eval.discarded),
        "online_split": config.eval.online_split,
        "print_every": int(config.eval.print_every),
        "func_diag_enabled": bool(config.eval.enable_func_diag),
        "func_diag_split": config.eval.func_diag_split,
        "func_diag_max_points": int(config.eval.func_diag_max_points),
        "target_backend": target.metadata.get("backend", labels["backend"]),
        "grad_mode": target.metadata.get("grad_mode", "unknown"),
    }
    if hasattr(config.sampler, "diffusion_factor"):
        meta["diffusion_factor"] = float(config.sampler.diffusion_factor)
    if hasattr(config.sampler, "mdecay"):
        meta["mdecay"] = float(config.sampler.mdecay)
    return meta


def _build_target_and_predictor(config: SingleFoldRunConfig, data: Any, arch: MLPArchitecture):
    use_jax_target = config.method == "sgnht_bjx"
    target_name = f"{config.data.dataset}_id{config.data.fold_id}_bnn{'_jax' if use_jax_target else ''}"

    if use_jax_target:
        target = make_bnn_regression_target_j(
            data.train.X,
            data.train.y,
            arch=arch,
            prior_var=config.model.prior_var,
            noise_var=config.model.noise_var,
            batch_size=config.model.batch_size,
            seed=config.sampler.budget.seed if config.sampler is not None else 0,
            name=target_name,
            metadata=data.metadata,
        )
    else:
        target = make_bnn_regression_target_t(
            data.train.X,
            data.train.y,
            arch=arch,
            prior_var=config.model.prior_var,
            noise_var=config.model.noise_var,
            batch_size=config.model.batch_size,
            seed=config.sampler.budget.seed if config.sampler is not None else 0,
            name=target_name,
            metadata=data.metadata,
        )

    if config.eval.raw_scale:
        split_inputs = data.raw_split_inputs()
        predictive_samples_fn = make_predictive_samples_fn_j(arch=arch, y_inverse_transform_fn=data.normalizer.unnormalize_y) if use_jax_target else make_predictive_samples_fn_t(arch=arch, y_inverse_transform_fn=data.normalizer.unnormalize_y)
        likelihood_var = config.model.noise_var * data.normalizer.y_var_scale
        eval_note = "raw_scale"
    else:
        split_inputs = data.normalized_split_inputs()
        predictive_samples_fn = make_predictive_samples_fn_j(arch=arch) if use_jax_target else make_predictive_samples_fn_t(arch=arch)
        likelihood_var = config.model.noise_var
        eval_note = "normalized_scale"

    return target, predictive_samples_fn, split_inputs, likelihood_var, eval_note


def _validate_single_fold_config(config: SingleFoldRunConfig) -> BaseSamplerConfig:
    if config.method not in SAMPLER_REGISTRY:
        raise ValueError(f"Unknown method: {config.method}. Available methods: {sorted(SAMPLER_REGISTRY)}")
    if config.sampler is None:
        raise ValueError("SingleFoldRunConfig.sampler must be provided. The CLI builds it from arguments.")
    if config.sampler.name != config.method:
        config.sampler.name = config.method
    if int(config.data.fold_id) <= 0:
        raise ValueError("run_single_fold expects a positive id/fold_id. Use run_fold_or_all for id=-1.")
    return config.sampler


def _available_fold_ids(config: SingleFoldRunConfig) -> list[int]:
    dataset_dir = Path(config.data.data_root) / config.data.dataset
    ids = []
    if dataset_dir.exists():
        for child in dataset_dir.iterdir():
            if child.is_dir() and child.name.isdigit():
                ids.append(int(child.name))
    return sorted(ids) if ids else list(range(1, 11))


def run_single_fold(config: SingleFoldRunConfig) -> SingleFoldResult:
    """Run exactly one dataset fold."""

    torch.set_default_dtype(torch.float32)
    sampler_config = _validate_single_fold_config(config)

    data = load_regression_splits_t(
        config.data.dataset,
        config.data.fold_id,
        has_header=config.data.has_header,
        data_root=str(config.data.data_root),
        device=config.data.device,
        normalize=config.data.normalize,
    )

    arch = MLPArchitecture(
        in_dim=data.input_dim,
        out_dim=data.output_dim,
        num_units=config.model.hidden_units,
        num_hidden=config.model.hidden_layers,
        act_fn=config.model.act_fn,
    )

    target, predictive_samples_fn, split_inputs, likelihood_var, eval_note = _build_target_and_predictor(config, data, arch)
    sampler_fn = SAMPLER_REGISTRY[config.method]

    base_run_name = config.output.run_name or _default_run_name(config, eval_note)
    run_name = _unique_run_name(config.output.save_root, base_run_name, overwrite=config.output.overwrite)
    paths = ExperimentPaths(save_root=Path(config.output.save_root), run_name=run_name)
    protocol_metadata = _build_protocol_metadata(config, run_name=run_name, eval_note=eval_note, target=target)

    print("\n==== True Dataset Single-Fold Run ====")
    print(f"dataset        = {config.data.dataset}")
    print(f"id             = {config.data.fold_id}")
    print(f"split_sizes    = {data.metadata['split_sizes']}")
    print(f"model          = MLP({data.input_dim} -> {config.model.hidden_units} x {config.model.hidden_layers} -> {data.output_dim}, act={config.model.act_fn})")
    print(f"method         = {config.method}")
    print(f"library        = {protocol_metadata['library']}")
    print(f"optimization   = {protocol_metadata['optimization']}")
    print(f"target dims    = {target.num_dims}")
    print(f"grad_mode      = {target.metadata['grad_mode']}")
    print(f"target_backend = {target.metadata.get('backend', 'torch')}")
    print(f"eval_scale     = {eval_note}")
    print(f"posterior      = prior_var={config.model.prior_var}, noise_var={config.model.noise_var}, batch_size={config.model.batch_size}")
    print(f"run_name       = {run_name}")

    t0 = time.perf_counter()
    sampling_result = sampler_fn(target, sampler_config)
    runtime_sec = time.perf_counter() - t0

    # Promote sampler-implementation labels into the protocol metadata so they
    # appear in metrics.csv and become usable grouping columns in summaries.
    for key in (
        "update_order",
        "vectorized_chains",
        "posteriors_inplace",
        "compile_requested",
        "compiled_step",
        "bridge",
    ):
        if key in sampling_result.metadata:
            protocol_metadata[key] = sampling_result.metadata[key]

    online_metrics = online_evaluate(
        chains=sampling_result.chains,
        split_inputs=split_inputs,
        predictive_samples_fn=predictive_samples_fn,
        likelihood_var=likelihood_var,
        print_every=config.eval.print_every,
        online_split=config.eval.online_split,
        print_metrics=config.output.print_metrics,
    )

    # Final predictive evaluation is deliberately restricted to train/test.
    # Validation remains reserved for online evaluation and function-space diagnostics.
    final_split_inputs = {name: split_inputs[name] for name in ("train", "test") if name in split_inputs}

    func_chains = None
    if config.eval.enable_func_diag:
        func_split = config.eval.func_diag_split
        if func_split not in split_inputs:
            raise ValueError(f"func_diag_split={func_split!r} is not available. Available splits: {sorted(split_inputs)}")
        func_t0 = time.perf_counter()
        X_func, _ = split_inputs[func_split]
        func_chains, func_meta = build_function_chains(
            sampling_result.chains,
            X_func,
            predictive_samples_fn=predictive_samples_fn,
            max_points=config.eval.func_diag_max_points,
        )
        protocol_metadata.update(func_meta)
        protocol_metadata["func_diag_runtime_sec"] = float(time.perf_counter() - func_t0)
        protocol_metadata["func_diag_split_used"] = func_split
    else:
        protocol_metadata["func_diag_runtime_sec"] = float("nan")
        protocol_metadata["func_diag_split_used"] = "disabled"

    metrics_record = final_evaluate(
        chains=sampling_result.chains,
        split_inputs=final_split_inputs,
        predictive_samples_fn=predictive_samples_fn,
        likelihood_var=likelihood_var,
        runtime_sec=runtime_sec,
        discarded=config.eval.discarded,
        metadata=sampling_result.metadata,
        protocol_metadata=protocol_metadata,
        func_chains=func_chains,
    )
    metrics_record.online_metrics = online_metrics

    if config.output.print_metrics:
        print_metrics_record(metrics_record)

    extra: dict[str, Any] = {
        "protocol_metadata": protocol_metadata,
        "notes": config.notes,
        "online_split": config.eval.online_split,
        "func_diag_enabled": bool(config.eval.enable_func_diag),
        "func_diag_split": config.eval.func_diag_split,
        "func_diag_max_points": int(config.eval.func_diag_max_points),
        "final_split_names": list(final_split_inputs.keys()),
        "available_split_names": list(split_inputs.keys()),
        "discarded": int(config.eval.discarded),
        "print_every": int(config.eval.print_every),
        "runtime_sec": float(runtime_sec),
        "eval_scale": eval_note,
        "target_metadata": dict(target.metadata),
    }

    if config.output.save_arrays:
        extra["array_paths"] = save_sampling_arrays(sampling_result, paths)
    if config.output.save_metrics:
        extra["metrics_json"] = str(save_metrics_json(metrics_record, paths.records_dir / "metrics.json"))
        extra["metrics_csv"] = str(save_metrics_csv(metrics_record, paths.records_dir / "metrics.csv"))
        extra["online_metrics_csv"] = str(save_online_metrics_csv(metrics_record, paths.records_dir / "online_metrics.csv"))

    result = SingleFoldResult(
        run_dir=paths.run_dir,
        sampling_result=sampling_result,
        metrics_record=metrics_record,
        figure_paths={},
        extra=extra,
    )
    extra["summary_json"] = str(save_run_summary(result))

    print(f"\nSaved run directory: {result.run_dir}")
    return result


def run_all_folds(config: SingleFoldRunConfig) -> list[SingleFoldResult]:
    """Run all available folds by repeatedly calling run_single_fold.

    This function only runs experiments. It intentionally does not compute
    mean/std summaries; summaries are handled by controllers.summarize_results.
    """
    fold_ids = _available_fold_ids(config)
    print(f"\n==== Running all folds ====")
    print(f"dataset = {config.data.dataset}")
    print(f"ids     = {fold_ids}")
    results: list[SingleFoldResult] = []
    for fold_id in fold_ids:
        data_cfg = replace(config.data, fold_id=int(fold_id))
        # If user provides a run_name for id=-1, keep it as a prefix and append id.
        output_cfg = config.output
        if output_cfg.run_name is not None:
            output_cfg = replace(output_cfg, run_name=f"{output_cfg.run_name}_id{fold_id}")
        fold_config = replace(config, data=data_cfg, output=output_cfg)
        results.append(run_single_fold(fold_config))
    return results


def run_fold_or_all(config: SingleFoldRunConfig):
    if int(config.data.fold_id) == -1:
        return run_all_folds(config)
    return run_single_fold(config)


__all__ = ["classify_method", "run_single_fold", "run_all_folds", "run_fold_or_all"]
