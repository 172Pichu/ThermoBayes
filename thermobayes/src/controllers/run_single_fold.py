from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from controllers.metrics import final_evaluate, online_evaluate, print_metrics_record
from models.configs import (
    AdaptiveSGHMCConfig,
    BaseSamplerConfig,
    BlackJAXSGNHTConfig,
    SGNHTConfig,
    SingleFoldRunConfig,
)
from models.results import SingleFoldResult
from models.targets import make_bnn_regression_target_j, make_bnn_regression_target_t
from models.true_datasets.networks import MLPArchitecture, make_predictive_samples_fn_j, make_predictive_samples_fn_t
from models.true_datasets.params import load_regression_splits_t
from samplers import SAMPLER_REGISTRY
from views.io import ExperimentPaths, save_metrics_csv, save_metrics_json, save_run_summary, save_sampling_arrays


def _default_run_name(config: SingleFoldRunConfig, eval_note: str) -> str:
    bs = "fullbatch" if int(config.model.batch_size) <= 0 else f"mb{int(config.model.batch_size)}"
    seed = config.sampler.budget.seed if config.sampler is not None else "na"
    return f"true_{config.data.dataset}_fold{config.data.fold_id}_{config.method}_{bs}_seed{seed}_{eval_note}"


def _build_target_and_predictor(config: SingleFoldRunConfig, data: Any, arch: MLPArchitecture):
    use_jax_target = config.method == "sgnht_bjx"
    target_name = f"{config.data.dataset}_fold{config.data.fold_id}_bnn{'_jax' if use_jax_target else ''}"

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
        # Keep the config explicit but forgiving: method is the dispatch key.
        config.sampler.name = config.method
    return config.sampler


def run_single_fold(config: SingleFoldRunConfig) -> SingleFoldResult:
    """Run exactly one dataset fold.

    This controller is intentionally one pipeline step. It does not loop over
    folds, methods, or seeds.
    """

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

    run_name = config.output.run_name or _default_run_name(config, eval_note)
    paths = ExperimentPaths(save_root=Path(config.output.save_root), run_name=run_name)

    print("\n==== True Dataset Single-Fold Run ====")
    print(f"dataset        = {config.data.dataset}")
    print(f"fold_id        = {config.data.fold_id}")
    print(f"split_sizes    = {data.metadata['split_sizes']}")
    print(f"model          = MLP({data.input_dim} -> {config.model.hidden_units} x {config.model.hidden_layers} -> {data.output_dim}, act={config.model.act_fn})")
    print(f"method         = {config.method}")
    print(f"target dims    = {target.num_dims}")
    print(f"grad_mode      = {target.metadata['grad_mode']}")
    print(f"target_backend = {target.metadata.get('backend', 'torch')}")
    print(f"eval_scale     = {eval_note}")

    t0 = time.perf_counter()
    sampling_result = sampler_fn(target, sampler_config)
    runtime_sec = time.perf_counter() - t0

    online_metrics = online_evaluate(
        chains=sampling_result.chains,
        split_inputs=split_inputs,
        predictive_samples_fn=predictive_samples_fn,
        likelihood_var=likelihood_var,
        print_every=config.eval.print_every,
        online_split=config.eval.online_split,
        print_metrics=config.output.print_metrics,
    )

    metrics_record = final_evaluate(
        chains=sampling_result.chains,
        split_inputs=split_inputs,
        predictive_samples_fn=predictive_samples_fn,
        likelihood_var=likelihood_var,
        runtime_sec=runtime_sec,
        discarded=config.eval.discarded,
        metadata=sampling_result.metadata,
    )
    metrics_record.online_metrics = online_metrics

    if config.output.print_metrics:
        print_metrics_record(metrics_record)

    extra: dict[str, Any] = {
        "notes": config.notes,
        "split_names": list(split_inputs.keys()),
        "discarded": int(config.eval.discarded),
        "print_every": int(config.eval.print_every),
        "online_split": config.eval.online_split,
        "runtime_sec": float(runtime_sec),
        "eval_scale": eval_note,
        "target_metadata": dict(target.metadata),
    }

    if config.output.save_arrays:
        extra["array_paths"] = save_sampling_arrays(sampling_result, paths)
    if config.output.save_metrics:
        extra["metrics_json"] = str(save_metrics_json(metrics_record, paths.records_dir / "metrics.json"))
        extra["metrics_csv"] = str(save_metrics_csv(metrics_record, paths.records_dir / "metrics.csv"))

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


__all__ = ["run_single_fold"]

