from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SamplingBudget:
    """Shared experimental budget used for fair sampler comparison."""

    seed: int = 77210
    num_chains: int = 4
    num_samples: int = 80
    burn_in: int = 500
    keep_every: int = 100
    inner_steps: int = 10


@dataclass(slots=True)
class ThermoConfig:
    """Shared temperature / reset / artificial gradient-noise knobs."""

    temperature: float = 1.0
    reset_prob: float = 0.0
    grad_noise_var: float = 0.0


@dataclass(slots=True)
class BaseSamplerConfig:
    """Public sampler config shared by all sampler implementations."""

    name: str
    step_size: float
    budget: SamplingBudget
    thermo: ThermoConfig = field(default_factory=ThermoConfig)
    store_as_numpy: bool = True


@dataclass(slots=True)
class AdaptiveSGHMCConfig(BaseSamplerConfig):
    """Algorithm-specific config for adaptive SGHMC."""

    mdecay: float = 0.05
    warm_up_frac: float = 0.5
    ema_beta: float = 0.999


@dataclass(slots=True)
class SGNHTConfig(BaseSamplerConfig):
    """Algorithm-specific config for SGNHT."""

    diffusion_factor: float = 1e-2
    thermostat_init: float | None = None
    reset_thermostat: bool = True
    reset_thermostat_to_A: bool = False
    reset_thermostat_to_midpoint: bool = True


@dataclass(slots=True)
class BlackJAXSGNHTConfig(SGNHTConfig):
    """Config for the BlackJAX SGNHT adapter."""

    bjx_beta: float = 0.0


@dataclass(slots=True)
class DataConfig:
    dataset: str = "yacht"
    fold_id: int = 1
    data_root: str | Path = "./datasets"
    has_header: bool = False
    normalize: bool = True
    device: str = "cpu"


@dataclass(slots=True)
class MLPConfig:
    hidden_units: int = 20
    hidden_layers: int = 1
    act_fn: str = "tanh"
    prior_var: float = 1.0
    noise_var: float = 1.0
    batch_size: int = 32


@dataclass(slots=True)
class EvalConfig:
    raw_scale: bool = False
    discarded: int = 0
    print_every: int = 0
    online_split: str = "val"


@dataclass(slots=True)
class OutputConfig:
    save_root: str | Path = "./result"
    run_name: str | None = None
    save_arrays: bool = True
    save_metrics: bool = True
    print_metrics: bool = True


@dataclass(slots=True)
class SingleFoldRunConfig:
    """One row on the experiment table: one dataset, one fold, one sampler."""

    data: DataConfig = field(default_factory=DataConfig)
    model: MLPConfig = field(default_factory=MLPConfig)
    sampler: BaseSamplerConfig | None = None
    method: str = "sgnht"
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    notes: str | None = None


__all__ = [
    "SamplingBudget",
    "ThermoConfig",
    "BaseSamplerConfig",
    "AdaptiveSGHMCConfig",
    "SGNHTConfig",
    "BlackJAXSGNHTConfig",
    "DataConfig",
    "MLPConfig",
    "EvalConfig",
    "OutputConfig",
    "SingleFoldRunConfig",
]

