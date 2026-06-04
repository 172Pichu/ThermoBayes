from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import numpy as np

try:
    import arviz as az
except Exception:  # pragma: no cover
    az = None  # type: ignore

try:
    import torch
    from torch import Tensor
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Tensor = Any  # type: ignore

try:
    import jax
    from jax import numpy as jnp
except Exception:  # pragma: no cover
    jax = None  # type: ignore
    jnp = None  # type: ignore

from models.results import MetricsRecord

ArrayLike = Any
TensorFn = Callable[[Any], Any]
PredictiveSamplesFn = Callable[[Any, Any], Any]
MeanPredictFn = Callable[[Any, Any], Any]


def _is_torch(x: Any) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def _is_jax(x: Any) -> bool:
    if jax is None or jnp is None:
        return False
    try:
        return isinstance(x, (jnp.ndarray, getattr(jax, "Array", ())))
    except Exception:
        return False


def to_numpy(x: Any) -> np.ndarray:
    if _is_torch(x):
        return x.detach().cpu().numpy()
    if _is_jax(x):
        return np.asarray(jax.device_get(x))
    return np.asarray(x)


def to_torch(x: Any, *, dtype: "torch.dtype | None" = None) -> "torch.Tensor":
    if torch is None:
        raise RuntimeError("torch is not available")
    if isinstance(x, torch.Tensor):
        return x if dtype is None else x.to(dtype=dtype)
    out = torch.as_tensor(x)
    return out if dtype is None else out.to(dtype=dtype)


def _finite_only(x: Any) -> np.ndarray:
    arr = np.asarray(to_numpy(x), dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _finite_samples(samples: Any) -> np.ndarray:
    arr = np.asarray(to_numpy(samples), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"samples must be 2D or 1D, got shape {arr.shape}")
    mask = np.all(np.isfinite(arr), axis=1)
    return arr[mask]


def _flatten_chains_to_samples(chains: Any) -> np.ndarray:
    arr = np.asarray(to_numpy(chains), dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"chains must be (chains, draws, dims), got {arr.shape}")
    return arr.reshape(-1, arr.shape[-1])


def _discard_saved_draws_from_chains(chains: Any, discarded: int) -> np.ndarray:
    arr = np.asarray(to_numpy(chains), dtype=np.float64)
    d = max(0, int(discarded))
    if d <= 0:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"chains must be (chains, draws, dims), got {arr.shape}")
    if d >= arr.shape[1]:
        raise ValueError(f"discarded={d} removes all saved draws; chains have only {arr.shape[1]} draws")
    return arr[:, d:, :]


def _summarize_vector(x: Any, *, prefix: str = "") -> dict[str, float]:
    arr = _finite_only(x)
    keys = ["min", "max", "mean", "median"]
    vals = [float("nan")] * 4 if arr.size == 0 else [float(np.min(arr)), float(np.max(arr)), float(np.mean(arr)), float(np.median(arr))]
    return {f"{prefix}{k}": v for k, v in zip(keys, vals)}


def summarize_param_norm(samples: Any) -> dict[str, float]:
    arr = _finite_samples(samples)
    if arr.shape[0] == 0:
        return {
            "num_samples": 0.0,
            "param_norm_min": float("nan"),
            "param_norm_max": float("nan"),
            "param_norm_mean": float("nan"),
            "param_norm_std": float("nan"),
        }
    norms = np.linalg.norm(arr, axis=1)
    return {
        "num_samples": float(arr.shape[0]),
        "param_norm_min": float(np.min(norms)),
        "param_norm_max": float(np.max(norms)),
        "param_norm_mean": float(np.mean(norms)),
        "param_norm_std": float(np.std(norms)),
    }


def summarize_l2d(samples: Any) -> dict[str, float]:
    arr = _finite_samples(samples)
    if arr.shape[0] < 2:
        return {"L2D_min": float("nan"), "L2D_max": float("nan"), "L2D_mean": float("nan"), "L2D_std": float("nan")}
    vals = np.linalg.norm(arr[1:] - arr[:-1], axis=1)
    return {"L2D_min": float(np.min(vals)), "L2D_max": float(np.max(vals)), "L2D_mean": float(np.mean(vals)), "L2D_std": float(np.std(vals))}


def summarize_sampler_outputs(samples: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(summarize_param_norm(samples))
    out.update(summarize_l2d(samples))
    return out


def _rmse(pred: Any, y: Any) -> float:
    a = np.asarray(to_numpy(pred), dtype=np.float64)
    b = np.asarray(to_numpy(y), dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _nll(y: Any, mean: Any, var: Any, *, eps: float = 1e-8) -> float:
    y_np = np.asarray(to_numpy(y), dtype=np.float64)
    m_np = np.asarray(to_numpy(mean), dtype=np.float64)
    v_np = np.clip(np.asarray(to_numpy(var), dtype=np.float64), a_min=eps, a_max=None)
    return float(np.mean(0.5 * np.log(2.0 * np.pi * v_np) + 0.5 * ((y_np - m_np) ** 2) / v_np))


def _maybe_unsqueeze_last(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x[:, None]
    return x


def _predictive_mean_and_var(
    samples: Any,
    X: Any,
    *,
    predictive_samples_fn: PredictiveSamplesFn | None = None,
    mean_predict_fn: MeanPredictFn | None = None,
    likelihood_var: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if predictive_samples_fn is not None:
        pred_samples = np.asarray(to_numpy(predictive_samples_fn(samples, X)), dtype=np.float64)
        if pred_samples.ndim == 1:
            pred_samples = pred_samples[:, None]
        mean = np.mean(pred_samples, axis=0)
        var = np.var(pred_samples, axis=0, ddof=0)
        if likelihood_var is not None:
            var = var + float(likelihood_var)
        return mean, var

    if mean_predict_fn is not None:
        mean = np.asarray(to_numpy(mean_predict_fn(samples, X)), dtype=np.float64)
        mean = _maybe_unsqueeze_last(mean)
        if likelihood_var is None:
            raise ValueError("likelihood_var is required when using mean_predict_fn")
        var = np.full_like(mean, float(likelihood_var), dtype=np.float64)
        return mean, var

    raise ValueError("Please provide predictive_samples_fn or mean_predict_fn")


def evaluate_with_samples(
    samples: Any,
    X: Any,
    y: Any,
    *,
    predictive_samples_fn: PredictiveSamplesFn | None = None,
    mean_predict_fn: MeanPredictFn | None = None,
    likelihood_var: float | None = None,
    y_inverse_transform_fn: TensorFn | None = None,
    compact: bool = True,
) -> dict[str, float]:
    mean, var = _predictive_mean_and_var(
        samples,
        X,
        predictive_samples_fn=predictive_samples_fn,
        mean_predict_fn=mean_predict_fn,
        likelihood_var=likelihood_var,
    )
    y_np = np.asarray(to_numpy(y), dtype=np.float64)
    y_np = _maybe_unsqueeze_last(y_np)

    if y_inverse_transform_fn is not None:
        mean = np.asarray(to_numpy(y_inverse_transform_fn(mean)), dtype=np.float64)
        y_np = np.asarray(to_numpy(y_inverse_transform_fn(y_np)), dtype=np.float64)

    sampler_summary = summarize_sampler_outputs(samples)
    out = {"rmse": _rmse(mean, y_np), "nll": _nll(y_np, mean, var)}

    if compact:
        out["param_norm"] = float(sampler_summary["param_norm_mean"])
        out["L2D"] = float(sampler_summary["L2D_mean"])
    else:
        out.update(sampler_summary)
    return out


def evaluate_splits_with_samples(
    samples: Any,
    split_inputs: Mapping[str, tuple[Any, Any]],
    *,
    predictive_samples_fn: PredictiveSamplesFn | None = None,
    mean_predict_fn: MeanPredictFn | None = None,
    likelihood_var: float | None = None,
    y_inverse_transform_fn: TensorFn | None = None,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for split_name, (X, y) in split_inputs.items():
        out[split_name] = evaluate_with_samples(
            samples,
            X,
            y,
            predictive_samples_fn=predictive_samples_fn,
            mean_predict_fn=mean_predict_fn,
            likelihood_var=likelihood_var,
            y_inverse_transform_fn=y_inverse_transform_fn,
            compact=False,
        )
    return out


def _to_theta_any(x: Any) -> np.ndarray:
    arr = np.asarray(to_numpy(x), dtype=np.float64)
    if arr.ndim == 1:
        return arr[None, :, None]
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected 1D/2D/3D chains-like input, got shape {arr.shape}")


def _idata_from_theta(theta: np.ndarray):
    if az is None:
        raise RuntimeError("arviz is not available")
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim != 3:
        raise ValueError(f"theta must be (chains, draws, dims), got {theta.shape}")
    try:
        return az.from_dict(posterior={"theta": theta})
    except TypeError:
        return az.from_dict({"posterior": {"theta": theta}})


def _summary_from_arviz_values(values: np.ndarray, *, metric_name: str) -> dict[str, float]:
    vals = _finite_only(values)
    if vals.size == 0:
        return {f"{metric_name}_min": float("nan"), f"{metric_name}_max": float("nan"), f"{metric_name}_mean": float("nan"), f"{metric_name}_median": float("nan")}
    return {
        f"{metric_name}_min": float(np.min(vals)),
        f"{metric_name}_max": float(np.max(vals)),
        f"{metric_name}_mean": float(np.mean(vals)),
        f"{metric_name}_median": float(np.median(vals)),
    }


def diagnose_theta(
    theta: Any,
    *,
    prefix: str = "",
    ess_method: str = "bulk",
    ess_relative: bool = False,
    rhat_method: str = "rank",
) -> dict[str, float]:
    theta_np = _to_theta_any(theta)
    C, T, _ = theta_np.shape
    if C < 2 or T < 2 or az is None:
        return {
            f"{prefix}n_eff_min": float("nan"),
            f"{prefix}n_eff_max": float("nan"),
            f"{prefix}n_eff_mean": float("nan"),
            f"{prefix}n_eff_median": float("nan"),
            f"{prefix}r_hat_min": float("nan"),
            f"{prefix}r_hat_max": float("nan"),
            f"{prefix}r_hat_mean": float("nan"),
            f"{prefix}r_hat_median": float("nan"),
        }
    idata = _idata_from_theta(theta_np)
    ess_ds = az.ess(idata, var_names=["theta"], method=ess_method, relative=ess_relative)
    rhat_ds = az.rhat(idata, var_names=["theta"], method=rhat_method)
    ess_vals = np.asarray(ess_ds["theta"].values, dtype=np.float64).reshape(-1)
    rhat_vals = np.asarray(rhat_ds["theta"].values, dtype=np.float64).reshape(-1)
    out: dict[str, float] = {}
    for k, v in _summary_from_arviz_values(ess_vals, metric_name="n_eff").items():
        out[f"{prefix}{k}"] = v
    for k, v in _summary_from_arviz_values(rhat_vals, metric_name="r_hat").items():
        out[f"{prefix}{k}"] = v
    return out


def diagnose(chains: Any, *, func_chains: Any | None = None, ess_method: str = "bulk", ess_relative: bool = False, rhat_method: str = "rank") -> dict[str, float]:
    out = diagnose_theta(
        chains,
        prefix="param_" if func_chains is not None else "",
        ess_method=ess_method,
        ess_relative=ess_relative,
        rhat_method=rhat_method,
    )
    if func_chains is not None:
        out.update(diagnose_theta(func_chains, prefix="func_", ess_method=ess_method, ess_relative=ess_relative, rhat_method=rhat_method))
    return out


def build_metrics_record(
    *,
    online_metrics: list[dict[str, float]] | None = None,
    final_sampler_summary: dict[str, float] | None = None,
    final_diagnostics: dict[str, float] | None = None,
    final_evaluation: dict[str, dict[str, float]] | None = None,
) -> MetricsRecord:
    return MetricsRecord(
        online_metrics=[] if online_metrics is None else list(online_metrics),
        final_sampler_summary={} if final_sampler_summary is None else dict(final_sampler_summary),
        final_diagnostics={} if final_diagnostics is None else dict(final_diagnostics),
        final_evaluation={} if final_evaluation is None else dict(final_evaluation),
    )


def final_evaluate(
    *,
    chains: Any,
    split_inputs: Mapping[str, tuple[Any, Any]],
    predictive_samples_fn: PredictiveSamplesFn | None,
    likelihood_var: float | None,
    runtime_sec: float,
    discarded: int = 0,
    metadata: Mapping[str, Any] | None = None,
    func_chains: Any | None = None,
) -> MetricsRecord:
    eval_chains = _discard_saved_draws_from_chains(chains, discarded)
    eval_samples = _flatten_chains_to_samples(eval_chains)

    final_sampler_summary = summarize_sampler_outputs(eval_samples)
    final_sampler_summary.update({
        "runtime_sec": float(runtime_sec),
        "saved_draws_per_chain": float(np.asarray(to_numpy(chains)).shape[1]),
        "used_draws_per_chain": float(eval_chains.shape[1]),
        "discarded_saved_draws_per_chain": float(max(0, int(discarded))),
        "samples_per_sec_used": float(eval_samples.shape[0] / runtime_sec) if runtime_sec > 0 else float("nan"),
    })
    if metadata is not None:
        for key in ["atomic_steps", "outer_steps", "grad_eval_calls"]:
            if key in metadata:
                final_sampler_summary[key] = float(metadata[key])
        atomic_steps = float(metadata.get("atomic_steps", float("nan")))
        grad_eval_calls = float(metadata.get("grad_eval_calls", float("nan")))
        if runtime_sec > 0 and math.isfinite(atomic_steps):
            final_sampler_summary["atomic_steps_per_sec"] = atomic_steps / float(runtime_sec)
            final_sampler_summary["sec_per_atomic_step"] = float(runtime_sec) / atomic_steps if atomic_steps > 0 else float("nan")
        if runtime_sec > 0 and math.isfinite(grad_eval_calls):
            final_sampler_summary["grad_eval_calls_per_sec"] = grad_eval_calls / float(runtime_sec)
            final_sampler_summary["sec_per_grad_eval_call"] = float(runtime_sec) / grad_eval_calls if grad_eval_calls > 0 else float("nan")

    final_diagnostics = diagnose(eval_chains, func_chains=func_chains)
    final_evaluation = evaluate_splits_with_samples(
        eval_samples,
        split_inputs,
        predictive_samples_fn=predictive_samples_fn,
        likelihood_var=likelihood_var,
    )
    return build_metrics_record(
        final_sampler_summary=final_sampler_summary,
        final_diagnostics=final_diagnostics,
        final_evaluation=final_evaluation,
    )


def online_evaluate(
    *,
    chains: Any,
    split_inputs: Mapping[str, tuple[Any, Any]],
    predictive_samples_fn: PredictiveSamplesFn | None,
    likelihood_var: float | None,
    print_every: int,
    online_split: str = "val",
    print_metrics: bool = True,
) -> list[dict[str, float]]:
    arr = np.asarray(to_numpy(chains), dtype=np.float64)
    if int(print_every) <= 0 or arr.ndim != 3 or arr.shape[1] <= 0:
        return []
    step = int(print_every)
    split_name = online_split if online_split in split_inputs else next(iter(split_inputs.keys()))
    X_online, y_online = split_inputs[split_name]
    out: list[dict[str, float]] = []
    for draw_count in range(step, arr.shape[1] + 1, step):
        partial_samples = arr[:, :draw_count, :].reshape(-1, arr.shape[-1])
        m = evaluate_with_samples(partial_samples, X_online, y_online, predictive_samples_fn=predictive_samples_fn, likelihood_var=likelihood_var, compact=True)
        m["draws_per_chain"] = float(draw_count)
        m["flat_num_samples"] = float(partial_samples.shape[0])
        out.append(m)
        if print_metrics:
            print(
                f"[online:{split_name}] draws/chain={draw_count:>4d} "
                f"RMSE={m['rmse']:.6g} NLL={m['nll']:.6g} "
                f"param_norm={m['param_norm']:.6g} L2D={m['L2D']:.6g}"
            )
    return out


def flatten_metrics_record(record: MetricsRecord) -> dict[str, float]:
    row: dict[str, float] = {}
    for k, v in record.final_sampler_summary.items():
        row[k] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
    for k, v in record.final_diagnostics.items():
        row[k] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
    for split_name, split_metrics in record.final_evaluation.items():
        for k, v in split_metrics.items():
            row[f"{split_name}_{k}"] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
    return row


def print_metrics_block(title: str, metrics: Mapping[str, Any], *, sort_keys: bool = True) -> None:
    print(f"\n{title}:")
    items = list(metrics.items())
    if sort_keys:
        items = sorted(items, key=lambda kv: kv[0])
    width = max((len(str(k)) for k, _ in items), default=0)
    for key, value in items:
        if isinstance(value, float):
            print(f"  {str(key):<{width}} : {value:.6g}" if math.isfinite(value) else f"  {str(key):<{width}} : {value}")
        else:
            print(f"  {str(key):<{width}} : {value}")


def print_metrics_record(record: MetricsRecord) -> None:
    if record.final_sampler_summary:
        print_metrics_block("final_sampler_summary", record.final_sampler_summary)
    if record.final_diagnostics:
        print_metrics_block("final_diagnostics", record.final_diagnostics)
    if record.final_evaluation:
        print("\nfinal_evaluation:")
        for split_name, split_metrics in record.final_evaluation.items():
            print_metrics_block(f"  {split_name}", split_metrics)


__all__ = [
    "to_numpy",
    "to_torch",
    "summarize_param_norm",
    "summarize_l2d",
    "summarize_sampler_outputs",
    "evaluate_with_samples",
    "evaluate_splits_with_samples",
    "diagnose_theta",
    "diagnose",
    "build_metrics_record",
    "final_evaluate",
    "online_evaluate",
    "flatten_metrics_record",
    "print_metrics_block",
    "print_metrics_record",
]

