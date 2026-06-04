from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import torch, jax
from torch import Tensor
from jax import numpy as jnp, random

@dataclass(slots=True)
class MLPArchitecture:

    in_dim: int
    out_dim: int = 1
    num_units: int = 20
    num_hidden: int = 1
    act_fn: str = "leaky_relu"

    @property
    def dims(self) -> list[int]:
        return [int(self.in_dim)] + [int(self.num_units)] * int(self.num_hidden) + [int(self.out_dim)]

def glorot_init_t(gen: torch.Generator, in_dim: int, out_dim: int, *, dtype: torch.dtype = torch.float32, device: torch.device | None = None) -> tuple[torch.Generator, Tensor, Tensor]:
    limit = math.sqrt(6.0 / (in_dim + out_dim))
    W = (torch.rand((in_dim, out_dim), generator=gen, dtype=dtype, device=device) * 2 - 1) * limit
    b = torch.zeros((out_dim,), dtype=dtype, device=device)
    return gen, W, b

def glorot_init_j(rng: Any, in_dim: int, out_dim: int, *, dtype: Any = None):
    if jnp is None or random is None:
        raise RuntimeError("jax is not available")
    dtype = jnp.float32 if dtype is None else dtype
    limit = jnp.sqrt(6.0 / (in_dim + out_dim)).astype(dtype)
    rng, wk = random.split(rng, 2)
    W = random.uniform(wk, (in_dim, out_dim), minval=-limit, maxval=limit, dtype=dtype)
    b = jnp.zeros((out_dim,), dtype=dtype)
    return rng, W, b

def init_mlp_params_t(gen: torch.Generator, in_dim: int, out_dim: int, num_units: int, num_hidden: int, *, dtype: torch.dtype = torch.float32, device: torch.device | None = None) -> tuple[torch.Generator, dict[str, Tensor]]:
    params: dict[str, Tensor] = {}
    dims = [in_dim] + [num_units] * num_hidden + [out_dim]
    for i in range(len(dims) - 1):
        gen, W, b = glorot_init_t(gen, dims[i], dims[i + 1], dtype=dtype, device=device)
        params[f"W{i}"] = W
        params[f"b{i}"] = b
    return gen, params

def init_mlp_params_j(rng: Any, in_dim: int, out_dim: int, num_units: int, num_hidden: int, *, dtype: Any = None):
    if jnp is None:
        raise RuntimeError("jax is not available")
    dtype = jnp.float32 if dtype is None else dtype
    params: dict[str, Any] = {}
    dims = [in_dim] + [num_units] * num_hidden + [out_dim]
    for i in range(len(dims) - 1):
        rng, W, b = glorot_init_j(rng, dims[i], dims[i + 1], dtype=dtype)
        params[f"W{i}"] = W
        params[f"b{i}"] = b
    return rng, params

def _apply_activation_t(X: Tensor, act_fn: str) -> Tensor:
    if act_fn == "relu":
        return torch.relu(X)
    if act_fn == "tanh":
        return torch.tanh(X)
    if act_fn == "leaky_relu":
        return torch.nn.functional.leaky_relu(X, negative_slope=0.01)
    raise ValueError(f"Unknown act_fn: {act_fn}")

def _apply_activation_j(X: Any, act_fn: str) -> Any:
    if jnp is None:
        raise RuntimeError("jax is not available")
    if act_fn == "relu":
        return jnp.maximum(X, 0)
    if act_fn == "tanh":
        return jnp.tanh(X)
    if act_fn == "leaky_relu":
        return jnp.where(X >= 0, X, 0.01 * X)
    raise ValueError(f"Unknown act_fn: {act_fn}")

def mlp_forward_t(params: dict[str, Tensor], X: Tensor, act_fn: str) -> Tensor:
    h = X
    num_layers = len(params) // 2
    for i in range(num_layers - 1):
        h = h @ params[f"W{i}"] + params[f"b{i}"]
        h = _apply_activation_t(h, act_fn)
    return h @ params[f"W{num_layers - 1}"] + params[f"b{num_layers - 1}"]

def mlp_forward_j(params: Mapping[str, Any], X: Any, act_fn: str) -> Any:
    if jnp is None:
        raise RuntimeError("jax is not available")
    h = X
    num_layers = len(params) // 2
    for i in range(num_layers - 1):
        h = h @ params[f"W{i}"] + params[f"b{i}"]
        h = _apply_activation_j(h, act_fn)
    return h @ params[f"W{num_layers - 1}"] + params[f"b{num_layers - 1}"]

def mlp_param_shapes(arch: MLPArchitecture) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    dims = arch.dims
    for i in range(len(dims) - 1):
        shapes[f"W{i}"] = (dims[i], dims[i + 1])
        shapes[f"b{i}"] = (dims[i + 1],)
    return shapes

def num_mlp_params(arch: MLPArchitecture) -> int:
    return int(sum(math.prod(shape) for shape in mlp_param_shapes(arch).values()))

def flatten_params_t(params: dict[str, Tensor]) -> Tensor:
    keys = sorted(params.keys(), key=lambda k: (int(k[1:]), k[0]))
    return torch.cat([params[k].reshape(-1) for k in keys], dim=0)

def unflatten_params_t(theta: Tensor, arch: MLPArchitecture) -> dict[str, Tensor]:
    theta = theta.reshape(-1)
    params: dict[str, Tensor] = {}
    start = 0
    for name, shape in mlp_param_shapes(arch).items():
        size = math.prod(shape)
        params[name] = theta[start:start + size].reshape(shape)
        start += size
    if start != theta.numel():
        raise ValueError(f"theta has {theta.numel()} values, but architecture requires {start}")
    return params

def unflatten_params_j(theta: Any, arch: MLPArchitecture) -> dict[str, Any]:
    if jnp is None:
        raise RuntimeError("jax is not available")
    theta = jnp.reshape(theta, (-1,))
    params: dict[str, Any] = {}
    start = 0
    for name, shape in mlp_param_shapes(arch).items():
        size = int(math.prod(shape))
        params[name] = jnp.reshape(theta[start:start + size], shape)
        start += size
    if int(theta.shape[0]) != start:
        raise ValueError(f"theta has {theta.shape[0]} values, but architecture requires {start}")
    return params

def init_mlp_theta_t(gen: torch.Generator, arch: MLPArchitecture, *, dtype: torch.dtype = torch.float32, device: torch.device | None = None) -> tuple[torch.Generator, Tensor]:
    gen, params = init_mlp_params_t(gen, arch.in_dim, arch.out_dim, arch.num_units, arch.num_hidden, dtype=dtype, device=device)
    return gen, flatten_params_t(params)

def mlp_forward_from_theta_t(theta: Tensor, X: Tensor, arch: MLPArchitecture) -> Tensor:
    return mlp_forward_t(unflatten_params_t(theta, arch), X, arch.act_fn)

def mlp_forward_from_theta_j(theta: Any, X: Any, arch: MLPArchitecture) -> Any:
    return mlp_forward_j(unflatten_params_j(theta, arch), X, arch.act_fn)

def unflatten_params_batch_t(theta_batch: Tensor, arch: MLPArchitecture) -> dict[str, Tensor]:
    """Unflatten a batch of parameter vectors.
    Args:
        theta_batch: tensor with shape (num_chains, num_params) or (num_params,).
    Returns:
        Dictionary whose values have shape (num_chains, *param_shape).
    """
    if theta_batch.ndim == 1:
        theta_batch = theta_batch[None, :]
    if theta_batch.ndim != 2:
        raise ValueError(f"theta_batch must be 1D or 2D, got shape {tuple(theta_batch.shape)}")
    params: dict[str, Tensor] = {}
    start = 0
    for name, shape in mlp_param_shapes(arch).items():
        size = math.prod(shape)
        params[name] = theta_batch[:, start:start + size].reshape(theta_batch.shape[0], *shape)
        start += size
    if start != theta_batch.shape[1]:
        raise ValueError(f"theta_batch has {theta_batch.shape[1]} values, but architecture requires {start}")
    return params

def mlp_forward_batch_t(params: dict[str, Tensor], X: Tensor, act_fn: str) -> Tensor:
    """Batched MLP forward pass for many parameter vectors.
    Args:
        params: values are shaped (C, in_dim, out_dim) for weights and (C, out_dim) for biases.
        X: either shared inputs with shape (B, in_dim) or per-chain inputs with shape (C, B, in_dim).
    Returns:
        Predictions with shape (C, B, out_dim).
    """
    num_layers = len(params) // 2
    C = params["W0"].shape[0]
    if X.ndim == 2:
        h = X.unsqueeze(0).expand(C, -1, -1)
    elif X.ndim == 3:
        if X.shape[0] != C:
            raise ValueError(f"per-chain X has {X.shape[0]} chains, but params have {C}")
        h = X
    else:
        raise ValueError(f"X must be 2D or 3D, got shape {tuple(X.shape)}")

    for i in range(num_layers - 1):
        h = torch.einsum("cbi,cio->cbo", h, params[f"W{i}"]) + params[f"b{i}"].unsqueeze(1)
        h = _apply_activation_t(h, act_fn)
    return torch.einsum("cbi,cio->cbo", h, params[f"W{num_layers - 1}"]) + params[f"b{num_layers - 1}"].unsqueeze(1)

def mlp_forward_from_theta_batch_t(theta_batch: Tensor, X: Tensor, arch: MLPArchitecture) -> Tensor:
    return mlp_forward_batch_t(unflatten_params_batch_t(theta_batch, arch), X, arch.act_fn)

def _to_jnp(x: Any, *, dtype: Any = None):
    if jnp is None:
        raise RuntimeError("jax is not available")
    dtype = jnp.float32 if dtype is None else dtype
    if isinstance(x, torch.Tensor):
        return jnp.asarray(x.detach().cpu().numpy(), dtype=dtype)
    return jnp.asarray(x, dtype=dtype)

def make_predictive_samples_fn_t(*, arch: MLPArchitecture, y_inverse_transform_fn: Callable[[Tensor | Any], Tensor | Any] | None = None) -> Callable[[Any, Tensor], Any]:
    def predictive_samples(samples: Any, X: Tensor) -> Any:
        if not isinstance(samples, torch.Tensor):
            sample_tensor = torch.as_tensor(samples, dtype=X.dtype, device=X.device)
        else:
            sample_tensor = samples.to(device=X.device, dtype=X.dtype)
        preds = []
        for theta in sample_tensor.reshape(-1, sample_tensor.shape[-1]):
            pred = mlp_forward_from_theta_t(theta, X, arch)
            if pred.ndim == 1:
                pred = pred[:, None]
            preds.append(pred.detach())
        out = torch.stack(preds, dim=0)
        if y_inverse_transform_fn is not None:
            out = y_inverse_transform_fn(out)
        return out
    return predictive_samples

def make_predictive_samples_fn_j(*, arch: MLPArchitecture, y_inverse_transform_fn: Callable[[Any], Any] | None = None) -> Callable[[Any, Any], Any]:
    def predictive_samples(samples: Any, X: Any) -> Any:
        if jax is None or jnp is None:
            raise RuntimeError("jax is not available")
        sample_j = _to_jnp(samples)
        X_j = _to_jnp(X)
        sample_j = jnp.reshape(sample_j, (-1, sample_j.shape[-1]))
        preds = jax.vmap(lambda theta: mlp_forward_from_theta_j(theta, X_j, arch))(sample_j)
        if preds.ndim == 2:
            preds = preds[:, :, None]
        if y_inverse_transform_fn is not None:
            preds_np = np.asarray(jax.device_get(preds))
            return y_inverse_transform_fn(preds_np)
        return preds
    return predictive_samples

__all__ = [
    "MLPArchitecture",
    "glorot_init_t",
    "glorot_init_j",
    "init_mlp_params_t",
    "init_mlp_params_j",
    "mlp_forward_t",
    "mlp_forward_j",
    "mlp_param_shapes",
    "num_mlp_params",
    "flatten_params_t",
    "unflatten_params_t",
    "unflatten_params_batch_t",
    "unflatten_params_j",
    "init_mlp_theta_t",
    "mlp_forward_from_theta_t",
    "mlp_forward_batch_t",
    "mlp_forward_from_theta_batch_t",
    "mlp_forward_from_theta_j",
    "make_predictive_samples_fn_t",
    "make_predictive_samples_fn_j",
    "_to_jnp",
]