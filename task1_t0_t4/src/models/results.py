from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:  # torch is needed for samplers, but keeping conversion guarded makes tests easier.
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from models.configs import BaseSamplerConfig


@dataclass(slots=True)
class SamplingResult:
    sampler_name: str
    chains: Any
    samples: Any
    final_states: dict[str, Any]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(slots=True)
class MetricsRecord:
    """Unified metrics container for one run.

    metadata stores the experiment protocol fields used later by the
    benchmark summarizer, e.g. dataset/id/method/backend/optimization.
    """

    metadata: dict[str, Any] = field(default_factory=dict)
    online_metrics: list[dict[str, float]] = field(default_factory=list)
    final_sampler_summary: dict[str, float] = field(default_factory=dict)
    final_diagnostics: dict[str, float] = field(default_factory=dict)
    final_evaluation: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass(slots=True)
class SingleFoldResult:
    run_dir: Path
    sampling_result: SamplingResult
    metrics_record: MetricsRecord
    figure_paths: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def _to_numpy(x: Any) -> Any:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: _to_numpy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_numpy(v) for v in x)
    return x


def finalize_result(
    *,
    sampler_name: str,
    sampler_config: BaseSamplerConfig,
    chains: Any,
    samples: Any,
    final_states: dict[str, Any],
    diagnostics: dict[str, Any],
    metadata: dict[str, Any],
) -> SamplingResult:
    """Apply the common result convention after a sampler has finished."""

    if sampler_config.store_as_numpy:
        chains_out = _to_numpy(chains)
        samples_out = _to_numpy(samples)
        final_states_out = _to_numpy(final_states)
    else:
        chains_out = chains
        samples_out = samples
        final_states_out = final_states

    return SamplingResult(
        sampler_name=sampler_name,
        chains=chains_out,
        samples=samples_out,
        final_states=final_states_out,
        diagnostics=diagnostics,
        metadata=metadata,
    )


__all__ = ["SamplingResult", "MetricsRecord", "SingleFoldResult", "finalize_result"]

