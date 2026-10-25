from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor
try:
    from jax import numpy as jnp
except Exception:  # pragma: no cover
    jnp = None  # type: ignore

@dataclass
class TensorNormalizer:

    X_mean: Tensor
    X_std: Tensor
    y_mean: Tensor
    y_std: Tensor

    @classmethod
    def fit(cls, X: Tensor, y: Tensor) -> "TensorNormalizer":
        if y.ndim == 1:
            y = y[:, None]
        X_mean = X.mean(dim=0)
        X_std = X.std(dim=0, unbiased=False) + 1e-8
        y_mean = y.mean(dim=0)
        y_std = y.std(dim=0, unbiased=False) + 1e-8
        return cls(X_mean, X_std, y_mean, y_std)

    def normalize_X(self, X: Tensor) -> Tensor:
        return (X - self.X_mean.to(device=X.device, dtype=X.dtype)) / self.X_std.to(device=X.device, dtype=X.dtype)

    def normalize_y(self, y: Tensor) -> Tensor:
        if y.ndim == 1:
            y = y[:, None]
        return (y - self.y_mean.to(device=y.device, dtype=y.dtype)) / self.y_std.to(device=y.device, dtype=y.dtype)

    def unnormalize_y(self, y_normed: Tensor | np.ndarray) -> Tensor | np.ndarray:
        if isinstance(y_normed, np.ndarray):
            mean = self.y_mean.detach().cpu().numpy()
            std = self.y_std.detach().cpu().numpy()
            return y_normed * std + mean
        return y_normed * self.y_std.to(device=y_normed.device, dtype=y_normed.dtype) + self.y_mean.to(device=y_normed.device, dtype=y_normed.dtype)

    @property
    def y_var_scale(self) -> float:
        return float((self.y_std.detach().cpu().reshape(-1)[0] ** 2).item())

@dataclass(slots=True)
class JNPNormalizer:

    X_mean: Any
    X_std: Any
    y_mean: Any
    y_std: Any

    @classmethod
    def fit(cls, X: Any, y: Any) -> "JNPNormalizer":
        if jnp is None:
            raise RuntimeError("jax is not available")
        if y.ndim == 1:
            y = y[:, None]
        return cls(jnp.mean(X, axis=0), jnp.std(X, axis=0) + 1e-8, jnp.mean(y, axis=0), jnp.std(y, axis=0) + 1e-8)

    def normalize_X(self, X: Any) -> Any:
        return (X - self.X_mean) / self.X_std

    def normalize_y(self, y: Any) -> Any:
        if y.ndim == 1:
            y = y[:, None]
        return (y - self.y_mean) / self.y_std

    def unnormalize_y(self, y_normed: Any) -> Any:
        return y_normed * self.y_std + self.y_mean

@dataclass(slots=True)
class RegressionSplit:
    X: Tensor
    y: Tensor
    X_raw: Tensor
    y_raw: Tensor

@dataclass(slots=True)
class RegressionDataBundle:

    dataset: str
    fold_id: int
    normalizer: TensorNormalizer
    train: RegressionSplit
    val: RegressionSplit
    test: RegressionSplit
    metadata: dict[str, Any]

    @property
    def input_dim(self) -> int:
        return int(self.train.X.shape[1])

    @property
    def output_dim(self) -> int:
        return int(self.train.y.shape[1])

    @property
    def train_size(self) -> int:
        return int(self.train.X.shape[0])

    def normalized_split_inputs(self) -> Mapping[str, tuple[Tensor, Tensor]]:
        return {"train": (self.train.X, self.train.y), "val": (self.val.X, self.val.y), "test": (self.test.X, self.test.y)}

    def raw_split_inputs(self) -> Mapping[str, tuple[Tensor, Tensor]]:
        # X remains normalized because the target/model is trained in normalized X space.
        return {"train": (self.train.X, self.train.y_raw), "val": (self.val.X, self.val.y_raw), "test": (self.test.X, self.test.y_raw)}

def round_list(xs: Iterable[float], ndigits: int = 5) -> list[float]:
    return [round(float(x), ndigits) for x in xs]

def load_csv(path: str, has_header: bool = False) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1 if has_header else 0, ndmin=2)

def _ensure_y_2d(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y[:, None]
    if y.shape[0] == 1 and y.shape[1] > 1:
        return y.T
    return y

def load_split_np(dataset: str, fold_id: int, usage: str, has_header: bool = False, data_root: str = "./datasets") -> tuple[np.ndarray, np.ndarray]:
    base_path = os.path.join(str(data_root), dataset, str(fold_id))
    X_path = os.path.join(base_path, f"{usage}_X.csv")
    y_path = os.path.join(base_path, f"{usage}_y.csv")
    X_np = load_csv(X_path, has_header=has_header).astype(np.float32)
    y_np = _ensure_y_2d(load_csv(y_path, has_header=has_header).astype(np.float32))
    if X_np.shape[0] != y_np.shape[0]:
        if X_np.T.shape[0] == y_np.shape[0]:
            X_np = X_np.T
        else:
            raise ValueError(f"[{usage}] X/Y size mismatch: X={X_np.shape}, y={y_np.shape}")
    return X_np, y_np

def load_split_t(dataset: str, fold_id: int, usage: str, has_header: bool = False, data_root: str = "./datasets") -> tuple[Tensor, Tensor]:
    X_np, y_np = load_split_np(dataset, fold_id, usage, has_header, data_root=str(data_root))
    return torch.from_numpy(X_np), torch.from_numpy(y_np)

def load_split_j(dataset: str, fold_id: int, usage: str, has_header: bool = False, data_root: str = "./datasets"):
    if jnp is None:
        raise RuntimeError("jax is not available")
    X_np, y_np = load_split_np(dataset, fold_id, usage, has_header, data_root=str(data_root))
    return jnp.array(X_np, dtype=jnp.float32), jnp.array(y_np, dtype=jnp.float32)

def load_regression_splits_t(
    dataset: str,
    fold_id: int,
    *,
    has_header: bool = False,
    data_root: str = "./datasets",
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    normalize: bool = True,
) -> RegressionDataBundle:
    
    dev = torch.device("cpu") if device is None else torch.device(device)

    X_train_raw, y_train_raw = load_split_t(dataset, fold_id, "train", has_header, data_root=str(data_root))
    X_val_raw, y_val_raw = load_split_t(dataset, fold_id, "val", has_header, data_root=str(data_root))
    X_test_raw, y_test_raw = load_split_t(dataset, fold_id, "test", has_header, data_root=str(data_root))

    X_train_raw = X_train_raw.to(device=dev, dtype=dtype)
    y_train_raw = y_train_raw.to(device=dev, dtype=dtype)
    X_val_raw = X_val_raw.to(device=dev, dtype=dtype)
    y_val_raw = y_val_raw.to(device=dev, dtype=dtype)
    X_test_raw = X_test_raw.to(device=dev, dtype=dtype)
    y_test_raw = y_test_raw.to(device=dev, dtype=dtype)

    normalizer = TensorNormalizer.fit(X_train_raw, y_train_raw)

    if normalize:
        train = RegressionSplit(normalizer.normalize_X(X_train_raw), normalizer.normalize_y(y_train_raw), X_train_raw, y_train_raw)
        val = RegressionSplit(normalizer.normalize_X(X_val_raw), normalizer.normalize_y(y_val_raw), X_val_raw, y_val_raw)
        test = RegressionSplit(normalizer.normalize_X(X_test_raw), normalizer.normalize_y(y_test_raw), X_test_raw, y_test_raw)
    else:
        train = RegressionSplit(X_train_raw, y_train_raw, X_train_raw, y_train_raw)
        val = RegressionSplit(X_val_raw, y_val_raw, X_val_raw, y_val_raw)
        test = RegressionSplit(X_test_raw, y_test_raw, X_test_raw, y_test_raw)

    return RegressionDataBundle(
        dataset=dataset,
        fold_id=int(fold_id),
        normalizer=normalizer,
        train=train,
        val=val,
        test=test,
        metadata={
            "dataset": dataset,
            "fold_id": int(fold_id),
            "data_root": str(data_root),
            "has_header": bool(has_header),
            "normalize": bool(normalize),
            "split_sizes": {"train": int(train.X.shape[0]), "val": int(val.X.shape[0]), "test": int(test.X.shape[0])},
            "input_dim": int(train.X.shape[1]),
            "output_dim": int(train.y.shape[1]),
        },
    )

def batchify_t(X: Tensor, Y: Tensor, batch_size: int):
    n = X.shape[0]
    for start in range(0, n, batch_size):
        s = slice(start, min(start + batch_size, n))
        yield X[s], Y[s]

def batchify_j(X: Any, Y: Any, batch_size: int):
    n = X.shape[0]
    for start in range(0, n, batch_size):
        s = slice(start, min(start + batch_size, n))
        yield X[s], Y[s]

__all__ = [
    "TensorNormalizer",
    "JNPNormalizer",
    "RegressionSplit",
    "RegressionDataBundle",
    "round_list",
    "load_csv",
    "load_split_np",
    "load_split_t",
    "load_split_j",
    "load_regression_splits_t",
    "batchify_t",
    "batchify_j",
]