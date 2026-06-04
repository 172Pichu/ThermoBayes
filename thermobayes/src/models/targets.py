from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch, jax
from torch import Tensor
from jax import numpy as jnp, random

from models.dataset.networks import (
    MLPArchitecture,
    _to_jnp,
    mlp_forward_from_theta_j,
    mlp_forward_from_theta_t,
    mlp_forward_from_theta_batch_t,
    num_mlp_params,
    unflatten_params_j,
    unflatten_params_t,
)

TensorFn = Callable[[torch.Tensor], torch.Tensor]
PredictFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

class SamplerTarget:
    """
    Small sampler-facing interface.
    Samplers only require:
      - num_dims
      - grad_estimator(theta)
    """
    num_dims: int
    def grad_estimator(self, _: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

@dataclass(slots=True)
class ToyTarget(SamplerTarget):

    num_dims: int
    grad_fn: TensorFn
    name: str = "toy"
    potential_fn: TensorFn | None = None
    density_fn: TensorFn | None = None
    sample_init_fn: Callable[[int, torch.Generator | None], torch.Tensor] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def grad_estimator(self, x: torch.Tensor) -> torch.Tensor:
        return self.grad_fn(x)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        if self.potential_fn is None:
            raise NotImplementedError("potential_fn is not defined for this ToyTarget")
        return self.potential_fn(x)

    def density(self, x: torch.Tensor) -> torch.Tensor:
        if self.density_fn is None:
            raise NotImplementedError("density_fn is not defined for this ToyTarget")
        return self.density_fn(x)

    def sample_init(self, num_samples: int, generator: torch.Generator | None = None) -> torch.Tensor:
        if self.sample_init_fn is None:
            return torch.randn(num_samples, self.num_dims, generator=generator)
        return self.sample_init_fn(num_samples, generator)

@dataclass(slots=True)
class DatasetTarget(SamplerTarget):

    num_dims: int
    grad_fn: TensorFn
    batch_grad_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
    name: str = "dataset"
    log_prob_fn: TensorFn | None = None
    predict_fn: PredictFn | None = None
    unpack_params_fn: Callable[[torch.Tensor], Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def grad_estimator(self, x: torch.Tensor) -> torch.Tensor:
        return self.grad_fn(x)

    def batch_grad_estimator(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_grad_fn is None:
            grads = [self.grad_fn(theta) for theta in x.reshape(-1, x.shape[-1])]
            return torch.stack(grads, dim=0).reshape_as(x)
        return self.batch_grad_fn(x)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        if self.log_prob_fn is None:
            raise NotImplementedError("log_prob_fn is not defined for this DatasetTarget")
        return self.log_prob_fn(x)

    def predict(self, theta: torch.Tensor, x_input: torch.Tensor) -> torch.Tensor:
        if self.predict_fn is None:
            raise NotImplementedError("predict_fn is not defined for this DatasetTarget")
        return self.predict_fn(theta, x_input)

    def unpack_params(self, theta: torch.Tensor) -> Any:
        if self.unpack_params_fn is None:
            raise NotImplementedError("unpack_params_fn is not defined for this DatasetTarget")
        return self.unpack_params_fn(theta)

@dataclass(slots=True)
class JAXDatasetTarget:

    num_dims: int
    grad_fn: Callable[[Any], Any]
    name: str = "jax_dataset"
    log_prob_fn: Callable[[Any], Any] | None = None
    predict_fn: Callable[[Any, Any], Any] | None = None
    unpack_params_fn: Callable[[Any], dict[str, Any]] | None = None
    grad_fn_keyed: Callable[[Any, Any], Any] | None = None
    logdensity_grad_fn_keyed: Callable[[Any, Any], Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def grad_estimator(self, theta: Any) -> Any:
        return self.grad_fn(theta)

    def grad_estimator_keyed(self, theta: Any, key: Any) -> Any:
        if self.grad_fn_keyed is None:
            return self.grad_fn(theta)
        return self.grad_fn_keyed(theta, key)

    def logdensity_grad_keyed(self, theta: Any, key: Any) -> Any:
        if self.logdensity_grad_fn_keyed is not None:
            return self.logdensity_grad_fn_keyed(theta, key)
        return -self.grad_estimator_keyed(theta, key)

    def log_prob(self, theta: Any) -> Any:
        if self.log_prob_fn is None:
            raise NotImplementedError("log_prob_fn is not defined")
        return self.log_prob_fn(theta)

    def predict(self, theta: Any, X: Any) -> Any:
        if self.predict_fn is None:
            raise NotImplementedError("predict_fn is not defined")
        return self.predict_fn(theta, X)

    def unpack_params(self, theta: Any) -> dict[str, Any]:
        if self.unpack_params_fn is None:
            raise NotImplementedError("unpack_params_fn is not defined")
        return self.unpack_params_fn(theta)

@dataclass(slots=True)
class TargetBundle:
    target: SamplerTarget
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    true_density_fn: TensorFn | None = None
    potential_fn: TensorFn | None = None
    predict_fn: PredictFn | None = None

def make_toy_target(
    *,
    num_dims: int,
    grad_fn: TensorFn,
    name: str = "toy",
    potential_fn: TensorFn | None = None,
    density_fn: TensorFn | None = None,
    sample_init_fn: Callable[[int, torch.Generator | None], torch.Tensor] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ToyTarget:
    return ToyTarget(
        num_dims=int(num_dims),
        grad_fn=grad_fn,
        batch_grad_fn=batch_grad_fn,
        name=name,
        potential_fn=potential_fn,
        density_fn=density_fn,
        sample_init_fn=sample_init_fn,
        metadata={} if metadata is None else dict(metadata),
    )

def make_toy_target_from_density(spec: Any, *, name: str | None = None, extra_metadata: dict[str, Any] | None = None) -> ToyTarget:
    if spec.grad_torch is None:
        raise ValueError(f"DensitySpec '{spec.name}' does not provide grad_torch")
    metadata = dict(spec.metadata)
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    return ToyTarget(
        num_dims=int(spec.num_dims),
        grad_fn=spec.grad_torch,
        name=spec.name if name is None else name,
        potential_fn=spec.potential_torch,
        density_fn=spec.density_torch,
        sample_init_fn=spec.sample_init_fn,
        metadata=metadata,
    )

def make_toy_bundle_from_density(spec: Any, *, name: str | None = None, extra_metadata: dict[str, Any] | None = None) -> TargetBundle:
    target = make_toy_target_from_density(spec, name=name, extra_metadata=extra_metadata)
    return TargetBundle(
        target=target,
        name=target.name,
        metadata=dict(target.metadata),
        true_density_fn=target.density_fn,
        potential_fn=target.potential_fn,
    )

def make_dataset_target(
        *,
        num_dims: int,
        grad_fn: TensorFn,
        batch_grad_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        name: str = "dataset",
        log_prob_fn: TensorFn | None = None,
        predict_fn: PredictFn | None = None,
        unpack_params_fn: Callable[[torch.Tensor], Any] | None = None,
        metadata: dict[str, Any] | None = None) -> DatasetTarget:
    
    return DatasetTarget(
        num_dims=int(num_dims),
        grad_fn=grad_fn,
        batch_grad_fn=batch_grad_fn,
        name=name,
        log_prob_fn=log_prob_fn,
        predict_fn=predict_fn,
        unpack_params_fn=unpack_params_fn,
        metadata={} if metadata is None else dict(metadata),
    )

def make_bnn_regression_target_t(
    X_train: Tensor,
    y_train: Tensor,
    *,
    arch: MLPArchitecture,
    prior_var: float = 1.0,
    noise_var: float = 1.0,
    batch_size: int | None = 32,
    seed: int = 0,
    name: str = "bnn_regression",
    metadata: dict[str, Any] | None = None) -> DatasetTarget:

    if y_train.ndim == 1:
        y_train = y_train[:, None]
    N = int(X_train.shape[0])
    D = num_mlp_params(arch)
    prior_var_f = float(prior_var)
    noise_var_f = float(noise_var)
    if prior_var_f <= 0.0 or noise_var_f <= 0.0:
        raise ValueError("prior_var and noise_var must be positive")
    if batch_size is None or int(batch_size) <= 0 or int(batch_size) >= N:
        B = N
        use_fullbatch = True
    else:
        B = int(batch_size)
        use_fullbatch = False
    gen = torch.Generator(device=X_train.device if X_train.device.type != "cpu" else "cpu").manual_seed(int(seed))

    def _choose_batch() -> tuple[Tensor, Tensor]:
        if use_fullbatch:
            return X_train, y_train
        idx = torch.randint(0, N, (B,), generator=gen, device=X_train.device)
        return X_train[idx], y_train[idx]

    def potential(theta: Tensor) -> Tensor:
        pred = mlp_forward_from_theta_t(theta, X_train, arch)
        if pred.ndim == 1:
            pred = pred[:, None]
        prior = 0.5 * torch.sum(theta * theta) / prior_var_f
        nll = 0.5 * torch.sum((y_train - pred) ** 2) / noise_var_f
        return prior + nll

    def log_prob(theta: Tensor) -> Tensor:
        return -potential(theta)

    def grad_estimator(theta: Tensor) -> Tensor:
        theta_var = theta.detach().clone().requires_grad_(True)
        Xb, yb = _choose_batch()
        pred = mlp_forward_from_theta_t(theta_var, Xb, arch)
        if pred.ndim == 1:
            pred = pred[:, None]
        prior = 0.5 * torch.sum(theta_var * theta_var) / prior_var_f
        data_term = (float(N) / float(Xb.shape[0])) * 0.5 * torch.sum((yb - pred) ** 2) / noise_var_f
        U = prior + data_term
        grad, = torch.autograd.grad(U, theta_var)
        return grad.detach()

    def batch_grad_estimator(theta_batch: Tensor) -> Tensor:
        theta_var = theta_batch.detach().clone().requires_grad_(True)
        if theta_var.ndim != 2:
            raise ValueError(f"theta_batch must be (chains, dims), got shape {tuple(theta_var.shape)}")
        C = int(theta_var.shape[0])
        if use_fullbatch:
            Xb, yb = X_train, y_train
            B_eff = int(Xb.shape[0])
        else:
            idx = torch.randint(0, N, (C, B), generator=gen, device=X_train.device)
            Xb = X_train[idx]
            yb = y_train[idx]
            B_eff = B
        pred = mlp_forward_from_theta_batch_t(theta_var, Xb, arch)
        if pred.ndim == 2:
            pred = pred[:, :, None]
        if yb.ndim == 2:
            yb_eval = yb.unsqueeze(0).expand(C, -1, -1)
        else:
            yb_eval = yb
        prior_per_chain = 0.5 * torch.sum(theta_var * theta_var, dim=1) / prior_var_f
        data_per_chain = (float(N) / float(B_eff)) * 0.5 * torch.sum((yb_eval - pred) ** 2, dim=(1, 2)) / noise_var_f
        U = torch.sum(prior_per_chain + data_per_chain)
        grad, = torch.autograd.grad(U, theta_var)
        return grad.detach()

    def predict(theta: Tensor, X_input: Tensor) -> Tensor:
        return mlp_forward_from_theta_t(theta, X_input, arch)

    meta = {} if metadata is None else dict(metadata)
    meta.update({
        "backend": "torch",
        "model": "mlp_regression",
        "in_dim": int(arch.in_dim),
        "out_dim": int(arch.out_dim),
        "num_units": int(arch.num_units),
        "num_hidden": int(arch.num_hidden),
        "act_fn": arch.act_fn,
        "prior_var": prior_var_f,
        "noise_var": noise_var_f,
        "batch_size": B,
        "grad_mode": "fullbatch" if use_fullbatch else f"minibatch[{B}]",
        "num_train": N,
        "num_dims": D,
        "torch_device": str(X_train.device),
        "torch_dtype": str(X_train.dtype),
    })

    return make_dataset_target(
        num_dims=D,
        grad_fn=grad_estimator,
        batch_grad_fn=batch_grad_estimator,
        name=name,
        log_prob_fn=log_prob,
        predict_fn=predict,
        unpack_params_fn=lambda theta: unflatten_params_t(theta, arch),
        metadata=meta,
    )

def make_bnn_regression_target_j(
    X_train: Any,
    y_train: Any,
    *,
    arch: MLPArchitecture,
    prior_var: float = 1.0,
    noise_var: float = 1.0,
    batch_size: int | None = 32,
    seed: int = 0,
    name: str = "bnn_regression_jax",
    metadata: dict[str, Any] | None = None) -> JAXDatasetTarget:

    X_train_j = _to_jnp(X_train)
    y_train_j = _to_jnp(y_train)
    if y_train_j.ndim == 1:
        y_train_j = y_train_j[:, None]
    N = int(X_train_j.shape[0])
    D = num_mlp_params(arch)
    prior_var_f = float(prior_var)
    noise_var_f = float(noise_var)
    if prior_var_f <= 0.0 or noise_var_f <= 0.0:
        raise ValueError("prior_var and noise_var must be positive")
    if batch_size is None or int(batch_size) <= 0 or int(batch_size) >= N:
        B = N
        use_fullbatch = True
    else:
        B = int(batch_size)
        use_fullbatch = False
    key_box = {"key": random.PRNGKey(int(seed))}

    def _choose_batch() -> tuple[Any, Any]:
        if use_fullbatch:
            return X_train_j, y_train_j
        key, subkey = random.split(key_box["key"])
        key_box["key"] = key
        idx = random.randint(subkey, shape=(B,), minval=0, maxval=N)
        return X_train_j[idx], y_train_j[idx]

    def _choose_batch_keyed(key: Any) -> tuple[Any, Any]:
        if use_fullbatch:
            return X_train_j, y_train_j
        idx = random.randint(key, shape=(B,), minval=0, maxval=N)
        return X_train_j[idx], y_train_j[idx]

    def potential_full(theta: Any) -> Any:
        pred = mlp_forward_from_theta_j(theta, X_train_j, arch)
        if pred.ndim == 1:
            pred = pred[:, None]
        prior = 0.5 * jnp.sum(theta * theta) / prior_var_f
        nll = 0.5 * jnp.sum((y_train_j - pred) ** 2) / noise_var_f
        return prior + nll

    def potential_batch(theta: Any, Xb: Any, yb: Any) -> Any:
        pred = mlp_forward_from_theta_j(theta, Xb, arch)
        if pred.ndim == 1:
            pred = pred[:, None]
        prior = 0.5 * jnp.sum(theta * theta) / prior_var_f
        data_term = (float(N) / float(Xb.shape[0])) * 0.5 * jnp.sum((yb - pred) ** 2) / noise_var_f
        return prior + data_term

    grad_potential_batch = jax.grad(potential_batch)

    def grad_estimator(theta: Any) -> Any:
        Xb, yb = _choose_batch()
        theta_j = jnp.asarray(theta, dtype=jnp.float32)
        return grad_potential_batch(theta_j, Xb, yb)

    def grad_estimator_keyed(theta: Any, key: Any) -> Any:
        Xb, yb = _choose_batch_keyed(key)
        theta_j = jnp.asarray(theta, dtype=jnp.float32)
        return grad_potential_batch(theta_j, Xb, yb)

    def logdensity_grad_keyed(theta: Any, key: Any) -> Any:
        return -grad_estimator_keyed(theta, key)

    def log_prob(theta: Any) -> Any:
        return -potential_full(jnp.asarray(theta, dtype=jnp.float32))

    def predict(theta: Any, X_input: Any) -> Any:
        return mlp_forward_from_theta_j(jnp.asarray(theta, dtype=jnp.float32), _to_jnp(X_input), arch)

    meta = {} if metadata is None else dict(metadata)
    meta.update({
        "backend": "jax",
        "model": "mlp_regression",
        "in_dim": int(arch.in_dim),
        "out_dim": int(arch.out_dim),
        "num_units": int(arch.num_units),
        "num_hidden": int(arch.num_hidden),
        "act_fn": arch.act_fn,
        "prior_var": prior_var_f,
        "noise_var": noise_var_f,
        "batch_size": B,
        "grad_mode": "fullbatch" if use_fullbatch else f"minibatch[{B}]",
        "num_train": N,
        "num_dims": D,
    })

    return JAXDatasetTarget(
        num_dims=D,
        grad_fn=grad_estimator,
        name=name,
        log_prob_fn=log_prob,
        predict_fn=predict,
        unpack_params_fn=lambda theta: unflatten_params_j(jnp.asarray(theta, dtype=jnp.float32), arch),
        grad_fn_keyed=grad_estimator_keyed,
        logdensity_grad_fn_keyed=logdensity_grad_keyed,
        metadata=meta,
    )

__all__ = [
    "TensorFn",
    "PredictFn",
    "SamplerTarget",
    "ToyTarget",
    "DatasetTarget",
    "JAXDatasetTarget",
    "TargetBundle",
    "make_toy_target",
    "make_toy_target_from_density",
    "make_toy_bundle_from_density",
    "make_dataset_target",
    "make_bnn_regression_target_t",
    "make_bnn_regression_target_j",
]