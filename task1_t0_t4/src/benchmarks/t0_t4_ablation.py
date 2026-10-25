from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch.func import grad_and_value

from models.dataset.networks import MLPArchitecture
from models.dataset.params import load_regression_splits_t
from models.targets import DatasetTarget, make_bnn_regression_target_t


VARIANT_ORDER = (
    "T0_flat_autograd",
    "T1_flat_func_grad_and_value",
    "T2_tree_manual_outplace",
    "T3_posteriors_native_outplace",
    "T4_posteriors_native_inplace",
)

VARIANT_DESCRIPTIONS = {
    "T0_flat_autograd": (
        "Flat tensor state; torch.autograd.grad; manual POSTERIORS-order SGNHT; "
        "out-of-place tensors. This is the lightweight reference."
    ),
    "T1_flat_func_grad_and_value": (
        "T0 with only the gradient API changed to torch.func.grad_and_value(has_aux=True)."
    ),
    "T2_tree_manual_outplace": (
        "T1 with POSTERIORS SGNHTState plus tree_ravel/flexi_tree_map and explicit "
        "out-of-place state construction; update equations remain handwritten."
    ),
    "T3_posteriors_native_outplace": (
        "Native POSTERIORS transform.update(..., inplace=False). This is the library default "
        "and the primary endpoint for explaining the runtime gap."
    ),
    "T4_posteriors_native_inplace": (
        "Native POSTERIORS transform.update(..., inplace=True). This is an optimization "
        "control measuring how much default out-of-place allocation costs."
    ),
}


@dataclass(slots=True)
class BenchmarkConfig:
    dataset: str = "yacht"
    fold_id: int = 1
    data_root: str = "./datasets"
    device: str = "cpu"
    normalize: bool = True
    hidden_units: int = 20
    hidden_layers: int = 1
    act_fn: str = "leaky_relu"
    prior_var: float = 1.0
    noise_var: float = 1.0
    batch_size: int = 32
    seed: int = 77210
    num_chains: int = 4
    warmup_steps: int = 100
    measure_steps: int = 2000
    repeats: int = 5
    parity_steps: int = 10
    step_size: float = 1e-3
    diffusion_factor: float = 1e-2
    temperature: float = 1.0
    thermostat_init: float | None = None
    output_dir: str = "./result_task1_t0_t4"
    reference_ratio: float | None = None
    shuffle_variant_order: bool = True
    torch_num_threads: int | None = None


@dataclass(slots=True)
class FlatState:
    params: torch.Tensor
    momenta: torch.Tensor
    xi: torch.Tensor
    log_posterior: torch.Tensor
    step: int


@dataclass(slots=True)
class RepeatRecord:
    repeat: int
    execution_position: int
    variant: str
    duration_sec: float
    calls: int
    sec_per_call: float
    calls_per_sec: float
    peak_memory_bytes: int | None
    final_params_norm: float
    final_momenta_norm: float
    final_xi: float


def _require_posteriors() -> dict[str, Any]:
    try:
        import posteriors
        from optree.integrations.torch import tree_ravel
        from posteriors.sgmcmc import sgnht as post_sgnht
        from posteriors.tree_utils import flexi_tree_map
    except Exception as exc:  # pragma: no cover - depends on user environment
        raise ImportError(
            "T0-T4 requires POSTERIORS. Install it in the active environment with:\n"
            "    python -m pip install posteriors\n"
            "Then rerun scripts/run_t0_t4_ablation.py."
        ) from exc
    return {
        "package": posteriors,
        "sgnht": post_sgnht,
        "tree_ravel": tree_ravel,
        "flexi_tree_map": flexi_tree_map,
    }


def _safe_log_posterior(target: DatasetTarget) -> Callable[[torch.Tensor, Any], tuple[torch.Tensor, Any]]:
    def wrapped(params: torch.Tensor, batch: Any) -> tuple[torch.Tensor, Any]:
        log_post, aux = target.log_posterior_batch(params, batch)
        if aux is None:
            aux = torch.empty(0, device=params.device, dtype=params.dtype)
        elif isinstance(aux, dict):
            aux = {
                key: value
                if isinstance(value, torch.Tensor)
                else torch.as_tensor(value, device=params.device, dtype=params.dtype)
                for key, value in aux.items()
            }
        elif not isinstance(aux, (torch.Tensor, tuple, list)):
            aux = torch.as_tensor(aux, device=params.device, dtype=params.dtype)
        return log_post, aux

    return wrapped


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _seed_noise(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _clone_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.detach().clone()


def _initial_tensors(
    *, num_chains: int, num_dims: int, seed: int, device: torch.device, dtype: torch.dtype, xi: float
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    gen_device: torch.device | str = device if device.type != "cpu" else "cpu"
    xs: list[torch.Tensor] = []
    ps: list[torch.Tensor] = []
    xis: list[torch.Tensor] = []
    for chain_idx in range(num_chains):
        gen = torch.Generator(device=gen_device).manual_seed(seed + chain_idx)
        xs.append(torch.randn(num_dims, generator=gen, device=device, dtype=dtype))
        ps.append(torch.randn(num_dims, generator=gen, device=device, dtype=dtype))
        xis.append(torch.tensor(xi, device=device, dtype=dtype))
    return xs, ps, xis


def _batch_index_plan(
    *, total_steps: int, num_chains: int, num_train: int, batch_size: int, seed: int, device: torch.device
) -> torch.Tensor | None:
    if batch_size <= 0 or batch_size >= num_train:
        return None
    gen_device: torch.device | str = device if device.type != "cpu" else "cpu"
    gen = torch.Generator(device=gen_device).manual_seed(seed)
    return torch.randint(
        0,
        num_train,
        (total_steps, num_chains, batch_size),
        generator=gen,
        device=device,
    )


def _batch_at(
    X: torch.Tensor,
    y: torch.Tensor,
    plan: torch.Tensor | None,
    step_idx: int,
    chain_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if plan is None:
        return X, y
    idx = plan[step_idx, chain_idx]
    return X[idx], y[idx]


def _flat_state_from_initial(x: torch.Tensor, p: torch.Tensor, xi: torch.Tensor) -> FlatState:
    return FlatState(
        params=_clone_tensor(x),
        momenta=_clone_tensor(p),
        xi=_clone_tensor(xi),
        log_posterior=torch.tensor(float("nan"), device=x.device, dtype=x.dtype),
        step=0,
    )


def _post_state_from_initial(post_sgnht: Any, x: torch.Tensor, p: torch.Tensor, xi: torch.Tensor) -> Any:
    return post_sgnht.SGNHTState(
        _clone_tensor(x),
        _clone_tensor(p),
        _clone_tensor(xi),
        torch.tensor(float("nan"), device=x.device, dtype=x.dtype),
        torch.tensor(0, device=x.device, dtype=torch.int64),
    )


def _step_t0(
    state: FlatState,
    batch: Any,
    *,
    log_posterior: Callable[[torch.Tensor, Any], tuple[torch.Tensor, Any]],
    lr: float,
    alpha: float,
    temperature: float,
) -> FlatState:
    # Mirror the existing lightweight target gradient pattern: detach + clone + autograd.grad.
    params_var = state.params.detach().clone().requires_grad_(True)
    log_post, _ = log_posterior(params_var, batch)
    (grad_log_post,) = torch.autograd.grad(log_post, params_var)
    grad_log_post = grad_log_post.detach()

    noise_std = math.sqrt(max(0.0, temperature * lr * 2.0 * alpha))
    noise = noise_std * torch.randn_like(state.momenta)
    params_next = state.params + lr * state.momenta
    xi_next = state.xi + lr * (torch.mean(state.momenta**2) - temperature)
    momenta_next = (
        state.momenta
        + lr * grad_log_post
        - lr * state.xi * state.momenta
        + noise
    )
    return FlatState(params_next, momenta_next, xi_next, log_post.detach(), state.step + 1)


def _step_t1(
    state: FlatState,
    batch: Any,
    *,
    log_posterior: Callable[[torch.Tensor, Any], tuple[torch.Tensor, Any]],
    lr: float,
    alpha: float,
    temperature: float,
) -> FlatState:
    with torch.no_grad():
        grads, (log_post, _) = grad_and_value(log_posterior, has_aux=True)(state.params, batch)

    noise_std = math.sqrt(max(0.0, temperature * lr * 2.0 * alpha))
    noise = noise_std * torch.randn_like(state.momenta)
    params_next = state.params + lr * state.momenta
    xi_next = state.xi + lr * (torch.mean(state.momenta**2) - temperature)
    momenta_next = state.momenta + lr * grads - lr * state.xi * state.momenta + noise
    return FlatState(params_next, momenta_next, xi_next, log_post.detach(), state.step + 1)


def _step_t2(
    state: Any,
    batch: Any,
    *,
    log_posterior: Callable[[torch.Tensor, Any], tuple[torch.Tensor, Any]],
    lr: float,
    alpha: float,
    temperature: float,
    post_sgnht: Any,
    tree_ravel: Callable[..., Any],
    flexi_tree_map: Callable[..., Any],
) -> Any:
    # Deliberately mirrors POSTERIORS update internals but bypasses Transform.update,
    # schedule checks and CatchAuxError. Tree/state overhead remains present.
    with torch.no_grad():
        grads, (log_post, aux) = grad_and_value(log_posterior, has_aux=True)(state.params, batch)
    del aux

    def transform_params(params: torch.Tensor, momenta: torch.Tensor) -> torch.Tensor:
        return params + lr * momenta

    def transform_momenta(momenta: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        return (
            momenta
            + lr * grad
            - lr * state.xi * momenta
            + math.sqrt(max(0.0, temperature * lr * 2.0 * alpha)) * torch.randn_like(momenta)
        )

    m_flat, _ = tree_ravel(state.momenta)
    xi_new = state.xi + lr * (torch.mean(m_flat**2) - temperature)
    params = flexi_tree_map(transform_params, state.params, state.momenta, inplace=False)
    momenta = flexi_tree_map(transform_momenta, state.momenta, grads, inplace=False)
    return post_sgnht.SGNHTState(
        params,
        momenta,
        xi_new,
        log_post.detach(),
        state.step + 1,
    )


def _extract_state(variant: str, state: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if variant.startswith("T0_") or variant.startswith("T1_"):
        return state.params, state.momenta, state.xi
    return state.params, state.momenta, state.xi


def _make_initial_states(
    variant: str,
    *,
    xs: list[torch.Tensor],
    ps: list[torch.Tensor],
    xis: list[torch.Tensor],
    post_sgnht: Any,
) -> list[Any]:
    if variant.startswith("T0_") or variant.startswith("T1_"):
        return [_flat_state_from_initial(x, p, xi) for x, p, xi in zip(xs, ps, xis)]
    return [_post_state_from_initial(post_sgnht, x, p, xi) for x, p, xi in zip(xs, ps, xis)]


def _run_steps(
    variant: str,
    states: list[Any],
    *,
    step_start: int,
    num_steps: int,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_plan: torch.Tensor | None,
    log_posterior: Callable[[torch.Tensor, Any], tuple[torch.Tensor, Any]],
    lr: float,
    alpha: float,
    temperature: float,
    post_sgnht: Any,
    tree_ravel: Callable[..., Any],
    flexi_tree_map: Callable[..., Any],
    transform: Any,
) -> list[Any]:
    for local_step in range(num_steps):
        plan_step = step_start + local_step
        for chain_idx in range(len(states)):
            batch = _batch_at(X, y, batch_plan, plan_step, chain_idx)
            if variant == "T0_flat_autograd":
                states[chain_idx] = _step_t0(
                    states[chain_idx],
                    batch,
                    log_posterior=log_posterior,
                    lr=lr,
                    alpha=alpha,
                    temperature=temperature,
                )
            elif variant == "T1_flat_func_grad_and_value":
                states[chain_idx] = _step_t1(
                    states[chain_idx],
                    batch,
                    log_posterior=log_posterior,
                    lr=lr,
                    alpha=alpha,
                    temperature=temperature,
                )
            elif variant == "T2_tree_manual_outplace":
                states[chain_idx] = _step_t2(
                    states[chain_idx],
                    batch,
                    log_posterior=log_posterior,
                    lr=lr,
                    alpha=alpha,
                    temperature=temperature,
                    post_sgnht=post_sgnht,
                    tree_ravel=tree_ravel,
                    flexi_tree_map=flexi_tree_map,
                )
            elif variant == "T3_posteriors_native_outplace":
                states[chain_idx], _ = transform.update(states[chain_idx], batch, inplace=False)
            elif variant == "T4_posteriors_native_inplace":
                states[chain_idx], _ = transform.update(states[chain_idx], batch, inplace=True)
            else:  # pragma: no cover - protected by VARIANT_ORDER
                raise ValueError(f"Unknown variant: {variant}")
    return states


def _summarize_records(records: Iterable[RepeatRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[RepeatRecord]] = {name: [] for name in VARIANT_ORDER}
    for record in records:
        grouped[record.variant].append(record)

    rows: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        group = grouped[variant]
        durations = [item.duration_sec for item in group]
        per_call = [item.sec_per_call for item in group]
        peaks = [item.peak_memory_bytes for item in group if item.peak_memory_bytes is not None]
        rows.append(
            {
                "variant": variant,
                "description": VARIANT_DESCRIPTIONS[variant],
                "repeats": len(group),
                "median_sec": statistics.median(durations),
                "mean_sec": statistics.fmean(durations),
                "std_sec": statistics.stdev(durations) if len(durations) > 1 else 0.0,
                "min_sec": min(durations),
                "max_sec": max(durations),
                "median_sec_per_call": statistics.median(per_call),
                "median_calls_per_sec": 1.0 / statistics.median(per_call),
                "median_peak_memory_bytes": int(statistics.median(peaks)) if peaks else None,
            }
        )
    t0_median = float(rows[0]["median_sec"])
    previous: dict[str, Any] | None = None
    for row in rows:
        row["median_ratio_to_T0"] = float(row["median_sec"]) / t0_median if t0_median > 0 else None
        if previous is None:
            row["incremental_delta_sec"] = 0.0
            row["incremental_formula"] = "baseline"
        else:
            row["incremental_delta_sec"] = float(row["median_sec"]) - float(previous["median_sec"])
            row["incremental_formula"] = f"{row['variant'].split('_', 1)[0]} - {previous['variant'].split('_', 1)[0]}"
        previous = row
    return rows


def _decompose(summary_rows: list[dict[str, Any]], reference_ratio: float | None) -> dict[str, Any]:
    med = {row["variant"]: float(row["median_sec"]) for row in summary_rows}
    t0 = med["T0_flat_autograd"]
    t1 = med["T1_flat_func_grad_and_value"]
    t2 = med["T2_tree_manual_outplace"]
    t3 = med["T3_posteriors_native_outplace"]
    t4 = med["T4_posteriors_native_inplace"]
    total_gap = t3 - t0

    components = [
        ("gradient_api_and_flat_grad_wrapper", t1 - t0, "T1 - T0"),
        ("state_and_tree_outplace", t2 - t1, "T2 - T1"),
        ("native_transform_dispatch_and_guards", t3 - t2, "T3 - T2"),
    ]
    comp_rows = []
    for name, delta, formula in components:
        comp_rows.append(
            {
                "component": name,
                "formula": formula,
                "delta_sec": delta,
                "share_of_T3_minus_T0_percent": (100.0 * delta / total_gap) if total_gap != 0 else None,
            }
        )

    ratio = t3 / t0 if t0 > 0 else float("nan")
    reference_match = None
    if reference_ratio is not None and reference_ratio > 0 and math.isfinite(ratio):
        relative_error = abs(ratio - reference_ratio) / reference_ratio
        reference_match = {
            "reference_ratio": reference_ratio,
            "microbenchmark_ratio": ratio,
            "relative_error_percent": 100.0 * relative_error,
            "reproduces_reference_within_15_percent": relative_error <= 0.15,
        }

    positive_components = [row for row in comp_rows if row["delta_sec"] > 0]
    largest = max(positive_components or comp_rows, key=lambda row: row["delta_sec"])
    return {
        "primary_endpoint": "T3_posteriors_native_outplace versus T0_flat_autograd",
        "T3_over_T0_runtime_ratio": ratio,
        "T3_minus_T0_sec": total_gap,
        "components": comp_rows,
        "largest_positive_component": largest if positive_components else None,
        "largest_component_by_delta": max(comp_rows, key=lambda row: row["delta_sec"]),
        "inplace_optimization": {
            "formula": "T3 - T4",
            "seconds_saved": t3 - t4,
            "percent_of_T3": 100.0 * (t3 - t4) / t3 if t3 > 0 else None,
            "T4_over_T3_runtime_ratio": t4 / t3 if t3 > 0 else None,
        },
        "reference_comparison": reference_match,
        "interpretation_rule": (
            "Attribute the full sampler gap only if T3/T0 approximately reproduces the full-run ratio. "
            "Otherwise, the missing cause lies outside the controlled update kernel, such as batch "
            "generation, outer Python loops, initialization, collection, or evaluation."
        ),
    }


def _parity_check(
    *,
    config: BenchmarkConfig,
    X: torch.Tensor,
    y: torch.Tensor,
    target: DatasetTarget,
    xs: list[torch.Tensor],
    ps: list[torch.Tensor],
    xis: list[torch.Tensor],
    post_sgnht: Any,
    tree_ravel: Callable[..., Any],
    flexi_tree_map: Callable[..., Any],
    transform: Any,
    device: torch.device,
) -> dict[str, Any]:
    steps = max(1, int(config.parity_steps))
    plan = _batch_index_plan(
        total_steps=steps,
        num_chains=config.num_chains,
        num_train=int(X.shape[0]),
        batch_size=config.batch_size,
        seed=config.seed + 400_000,
        device=device,
    )
    log_posterior = _safe_log_posterior(target)
    outputs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for variant in VARIANT_ORDER:
        states = _make_initial_states(variant, xs=xs, ps=ps, xis=xis, post_sgnht=post_sgnht)
        _seed_noise(config.seed + 500_000, device)
        states = _run_steps(
            variant,
            states,
            step_start=0,
            num_steps=steps,
            X=X,
            y=y,
            batch_plan=plan,
            log_posterior=log_posterior,
            lr=config.step_size,
            alpha=config.diffusion_factor,
            temperature=config.temperature,
            post_sgnht=post_sgnht,
            tree_ravel=tree_ravel,
            flexi_tree_map=flexi_tree_map,
            transform=transform,
        )
        params = torch.stack([_extract_state(variant, st)[0].detach() for st in states])
        momenta = torch.stack([_extract_state(variant, st)[1].detach() for st in states])
        xi = torch.stack([_extract_state(variant, st)[2].detach() for st in states])
        outputs[variant] = (params, momenta, xi)

    reference = outputs["T3_posteriors_native_outplace"]
    comparisons = []
    all_pass = True
    for variant in VARIANT_ORDER:
        current = outputs[variant]
        row: dict[str, Any] = {"variant": variant}
        variant_pass = True
        for label, lhs, rhs in zip(("params", "momenta", "xi"), current, reference):
            max_abs = float(torch.max(torch.abs(lhs - rhs)).detach().cpu())
            close = bool(torch.allclose(lhs, rhs, rtol=1e-5, atol=1e-6, equal_nan=True))
            row[f"{label}_max_abs_diff"] = max_abs
            row[f"{label}_allclose"] = close
            variant_pass = variant_pass and close
        row["allclose"] = variant_pass
        all_pass = all_pass and variant_pass
        comparisons.append(row)
    return {
        "reference": "T3_posteriors_native_outplace",
        "steps": steps,
        "rtol": 1e-5,
        "atol": 1e-6,
        "all_variants_match": all_pass,
        "comparisons": comparisons,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_report(
    *,
    config: BenchmarkConfig,
    summary: list[dict[str, Any]],
    decomposition: dict[str, Any],
    parity: dict[str, Any],
    environment: dict[str, Any],
) -> str:
    lines = [
        "# ThermoBayes Task 1: T0–T4 runtime decomposition",
        "",
        "## Experimental contract",
        "",
        "All variants use identical initial parameters, momenta, thermostat values, minibatch-index plans, "
        "SGNHT hyperparameters, chain count, and random-noise seed. The update order is the POSTERIORS order. "
        "Only one implementation layer is changed at each stage.",
        "",
        f"- Dataset/fold: `{config.dataset}` / `{config.fold_id}`",
        f"- Device: `{environment['device']}`",
        f"- Parameters: `{environment['num_dims']}`",
        f"- Chains: `{config.num_chains}`",
        f"- Measured step rounds: `{config.measure_steps}` ({config.measure_steps * config.num_chains} gradient calls per repeat)",
        f"- Warm-up rounds: `{config.warmup_steps}`",
        f"- Repeats: `{config.repeats}`",
        "",
        "## T0–T4 median runtime",
        "",
        "| Variant | Median seconds | Seconds/call | Calls/s | Peak memory |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        peak = row["median_peak_memory_bytes"]
        peak_text = "N/A" if peak is None else f"{peak / (1024**2):.2f} MiB"
        lines.append(
            f"| {row['variant']} | {_fmt(row['median_sec'])} | "
            f"{_fmt(row['median_sec_per_call'], 9)} | {_fmt(row['median_calls_per_sec'], 2)} | {peak_text} |"
        )

    lines.extend(
        [
            "",
            "## Causal decomposition of the default POSTERIORS gap",
            "",
            f"Primary ratio `T3/T0` = **{_fmt(decomposition['T3_over_T0_runtime_ratio'], 3)}×**.",
            "",
            "| Component | Difference | Added seconds | Share of T3−T0 |",
            "|---|---|---:|---:|",
        ]
    )
    for item in decomposition["components"]:
        share = item["share_of_T3_minus_T0_percent"]
        lines.append(
            f"| {item['component']} | {item['formula']} | {_fmt(item['delta_sec'])} | "
            f"{'N/A' if share is None else f'{share:.1f}%'} |"
        )
    inplace = decomposition["inplace_optimization"]
    lines.extend(
        [
            "",
            f"Using `inplace=True` saves **{_fmt(inplace['seconds_saved'])} s** "
            f"({ _fmt(inplace['percent_of_T3'], 1)}% of T3; positive means faster).",
            "",
            "## Numerical parity",
            "",
            f"All variants match T3: **{parity['all_variants_match']}**.",
            "",
            "| Variant | Params max | Momenta max | Xi max | Pass |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in parity["comparisons"]:
        lines.append(
            f"| {row['variant']} | {row['params_max_abs_diff']:.3e} | "
            f"{row['momenta_max_abs_diff']:.3e} | {row['xi_max_abs_diff']:.3e} | {row['allclose']} |"
        )

    ref = decomposition.get("reference_comparison")
    lines.extend(["", "## Interpretation", ""])
    largest = decomposition["largest_positive_component"]
    if largest is None:
        lines.append(
            "No T0→T3 component added positive median runtime in this run. Increase steps/repeats and "
            "check environment noise before interpreting the decomposition."
        )
    else:
        lines.append(
            f"Within this controlled kernel, the largest positive component is "
            f"`{largest['component']}` ({largest['formula']})."
        )
    if ref is not None:
        lines.append(
            f"The microbenchmark ratio is {ref['microbenchmark_ratio']:.3f}× versus the supplied full-run "
            f"reference {ref['reference_ratio']:.3f}×; relative error is {ref['relative_error_percent']:.1f}%."
        )
        if ref["reproduces_reference_within_15_percent"]:
            lines.append(
                "Because the controlled ratio reproduces the full-run gap within 15%, the component "
                "decomposition is evidence for the main cause of the observed full-run difference."
            )
        else:
            lines.append(
                "Because the controlled ratio does not reproduce the full-run gap within 15%, do not stop "
                "at T0–T4. Profile batch generation, outer loops, initialization, sample collection, and evaluation."
            )
    else:
        lines.append(
            "To connect this kernel result to an existing full-run gap, rerun with `--reference_ratio <ratio>`. "
            "A causal full-run conclusion is strongest when T3/T0 approximately reproduces that ratio."
        )

    lines.extend(
        [
            "",
            "## Variant definitions",
            "",
        ]
    )
    for variant in VARIANT_ORDER:
        lines.append(f"- **{variant}**: {VARIANT_DESCRIPTIONS[variant]}")
    lines.append("")
    return "\n".join(lines)


def run_t0_t4_ablation(config: BenchmarkConfig) -> dict[str, Any]:
    if config.num_chains <= 0 or config.measure_steps <= 0 or config.repeats <= 0:
        raise ValueError("num_chains, measure_steps and repeats must be positive")
    if config.warmup_steps < 0 or config.parity_steps <= 0:
        raise ValueError("warmup_steps must be >= 0 and parity_steps must be positive")
    if config.step_size <= 0 or config.diffusion_factor < 0 or config.temperature <= 0:
        raise ValueError("Invalid SGNHT hyperparameters")

    deps = _require_posteriors()
    posteriors_package = deps["package"]
    post_sgnht = deps["sgnht"]
    tree_ravel = deps["tree_ravel"]
    flexi_tree_map = deps["flexi_tree_map"]

    device = torch.device(config.device)
    torch.set_default_dtype(torch.float32)
    if config.torch_num_threads is not None:
        if config.torch_num_threads <= 0:
            raise ValueError("torch_num_threads must be positive when specified")
        torch.set_num_threads(config.torch_num_threads)
    data = load_regression_splits_t(
        config.dataset,
        config.fold_id,
        data_root=config.data_root,
        device=device,
        normalize=config.normalize,
    )
    arch = MLPArchitecture(
        in_dim=data.input_dim,
        out_dim=data.output_dim,
        num_units=config.hidden_units,
        num_hidden=config.hidden_layers,
        act_fn=config.act_fn,
    )
    target = make_bnn_regression_target_t(
        data.train.X,
        data.train.y,
        arch=arch,
        prior_var=config.prior_var,
        noise_var=config.noise_var,
        batch_size=config.batch_size,
        seed=config.seed,
        name=f"{config.dataset}_id{config.fold_id}_t0_t4",
        metadata=data.metadata,
    )
    log_posterior = _safe_log_posterior(target)
    xi0 = config.diffusion_factor if config.thermostat_init is None else config.thermostat_init
    xs, ps, xis = _initial_tensors(
        num_chains=config.num_chains,
        num_dims=target.num_dims,
        seed=config.seed,
        device=device,
        dtype=data.train.X.dtype,
        xi=float(xi0),
    )
    transform = post_sgnht.build(
        log_posterior,
        lr=config.step_size,
        alpha=config.diffusion_factor,
        beta=0.0,
        sigma=1.0,
        temperature=config.temperature,
        xi=float(xi0),
    )

    environment: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "posteriors_version": getattr(posteriors_package, "__version__", "unknown"),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "num_dims": int(target.num_dims),
        "num_train": int(data.train.X.shape[0]),
        "pid": os.getpid(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }

    parity = _parity_check(
        config=config,
        X=data.train.X,
        y=data.train.y,
        target=target,
        xs=xs,
        ps=ps,
        xis=xis,
        post_sgnht=post_sgnht,
        tree_ravel=tree_ravel,
        flexi_tree_map=flexi_tree_map,
        transform=transform,
        device=device,
    )

    records: list[RepeatRecord] = []
    for repeat in range(config.repeats):
        total_steps = config.warmup_steps + config.measure_steps
        plan = _batch_index_plan(
            total_steps=total_steps,
            num_chains=config.num_chains,
            num_train=int(data.train.X.shape[0]),
            batch_size=config.batch_size,
            seed=config.seed + 10_000 + repeat,
            device=device,
        )
        variants = list(VARIANT_ORDER)
        if config.shuffle_variant_order:
            random.Random(config.seed + 20_000 + repeat).shuffle(variants)

        for execution_position, variant in enumerate(variants):
            states = _make_initial_states(variant, xs=xs, ps=ps, xis=xis, post_sgnht=post_sgnht)
            _seed_noise(config.seed + 30_000 + repeat, device)
            if config.warmup_steps:
                states = _run_steps(
                    variant,
                    states,
                    step_start=0,
                    num_steps=config.warmup_steps,
                    X=data.train.X,
                    y=data.train.y,
                    batch_plan=plan,
                    log_posterior=log_posterior,
                    lr=config.step_size,
                    alpha=config.diffusion_factor,
                    temperature=config.temperature,
                    post_sgnht=post_sgnht,
                    tree_ravel=tree_ravel,
                    flexi_tree_map=flexi_tree_map,
                    transform=transform,
                )

            _sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            states = _run_steps(
                variant,
                states,
                step_start=config.warmup_steps,
                num_steps=config.measure_steps,
                X=data.train.X,
                y=data.train.y,
                batch_plan=plan,
                log_posterior=log_posterior,
                lr=config.step_size,
                alpha=config.diffusion_factor,
                temperature=config.temperature,
                post_sgnht=post_sgnht,
                tree_ravel=tree_ravel,
                flexi_tree_map=flexi_tree_map,
                transform=transform,
            )
            _sync(device)
            duration = time.perf_counter() - started
            peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            calls = config.measure_steps * config.num_chains
            params = torch.stack([_extract_state(variant, state)[0].detach() for state in states])
            momenta = torch.stack([_extract_state(variant, state)[1].detach() for state in states])
            xi = torch.stack([_extract_state(variant, state)[2].detach() for state in states])
            records.append(
                RepeatRecord(
                    repeat=repeat,
                    execution_position=execution_position,
                    variant=variant,
                    duration_sec=duration,
                    calls=calls,
                    sec_per_call=duration / calls,
                    calls_per_sec=calls / duration,
                    peak_memory_bytes=peak,
                    final_params_norm=float(torch.linalg.vector_norm(params).cpu()),
                    final_momenta_norm=float(torch.linalg.vector_norm(momenta).cpu()),
                    final_xi=float(torch.mean(xi).cpu()),
                )
            )
            print(
                f"repeat={repeat + 1}/{config.repeats} variant={variant:<38} "
                f"time={duration:.6f}s calls/s={calls / duration:.2f}"
            )

    repeat_rows = [asdict(record) for record in records]
    summary = _summarize_records(records)
    decomposition = _decompose(summary, config.reference_ratio)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "t0_t4_repeats.csv", repeat_rows)
    _write_csv(out_dir / "t0_t4_summary.csv", summary)
    with (out_dir / "t0_t4_decomposition.json").open("w", encoding="utf-8") as handle:
        json.dump(decomposition, handle, indent=2, ensure_ascii=False)
    with (out_dir / "t0_t4_parity.json").open("w", encoding="utf-8") as handle:
        json.dump(parity, handle, indent=2, ensure_ascii=False)
    with (out_dir / "t0_t4_environment.json").open("w", encoding="utf-8") as handle:
        json.dump(environment, handle, indent=2, ensure_ascii=False)
    with (out_dir / "t0_t4_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, ensure_ascii=False)
    report = _markdown_report(
        config=config,
        summary=summary,
        decomposition=decomposition,
        parity=parity,
        environment=environment,
    )
    (out_dir / "t0_t4_report.md").write_text(report, encoding="utf-8")

    print("\n==== T0-T4 median summary ====")
    for row in summary:
        print(
            f"{row['variant']:<38} median={row['median_sec']:.6f}s "
            f"ratio/T0={row['median_sec'] / summary[0]['median_sec']:.3f}x"
        )
    print(f"\nT3/T0 = {decomposition['T3_over_T0_runtime_ratio']:.3f}x")
    print(f"Parity passed = {parity['all_variants_match']}")
    print(f"Saved to: {out_dir.resolve()}")

    return {
        "config": asdict(config),
        "environment": environment,
        "repeat_records": repeat_rows,
        "summary": summary,
        "decomposition": decomposition,
        "parity": parity,
        "output_dir": str(out_dir),
    }
