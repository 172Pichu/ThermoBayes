from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np, torch

# Default parameters for the standard toy figures
A = 1.0
B = 0.2
PHI = np.pi / 6.0

# Public bundle returned by the new outer API
ArrayFn = Callable[[np.ndarray], np.ndarray]
TorchFn = Callable[[torch.Tensor], torch.Tensor]

@dataclass(slots=True)
class DensitySpec:
    """
    Human-friendly density bundle returned by the outer API.

    The goal is to provide one understandable object that can serve:
      - samplers / targets: grad_torch, potential_torch, density_torch
      - plotting / numerical checks: grad_np, potential_np, density_np
      - experiment layer: metadata, sample_init_fn
    """
    name: str
    num_dims: int
    potential_np: ArrayFn
    grad_np: ArrayFn
    density_np: ArrayFn | None = None
    potential_torch: TorchFn | None = None
    grad_torch: TorchFn | None = None
    density_torch: TorchFn | None = None
    sample_init_fn: Callable[[int, torch.Generator | None], torch.Tensor] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Small helpers

def _to_numpy_1d(x: np.ndarray | list[float] | float) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return arr

def _to_numpy_lastdim2(xy: np.ndarray) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float64)
    if arr.shape[-1] != 2:
        raise ValueError(f"expected last dim 2, got shape {arr.shape}")
    return arr


# Standard fig1 (standard bimodal) : legacy math functions kept

def U_fig1(x: np.ndarray) -> np.ndarray:
    x = _to_numpy_1d(x)
    return (x ** 2) * (x ** 2 - 2.0)

def dU_dx_fig1(x: np.ndarray) -> np.ndarray:
    x = _to_numpy_1d(x)
    return 4.0 * x * (x + 1.0) * (x - 1.0)

def unnormed_density_fig1(x: np.ndarray) -> np.ndarray:
    x = _to_numpy_1d(x)
    return np.exp(-U_fig1(x))

def integral_even(f: Callable[[np.ndarray], np.ndarray], num_integral_grids_per_unit: int, integral_max: float) -> float:
    integral_domain = np.linspace(0.0, float(integral_max), int(num_integral_grids_per_unit * float(integral_max)) + 1)
    y = f(integral_domain)
    return float(np.trapz(y, integral_domain) * 2.0)

def true_density_fig1_standard(
    *,
    plot_max: float,
    plot_grids_per_unit: int,
    integral_max: float,
    integral_grids_per_unit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    New explicit interface replacing the old args-based interface.

    Returns:
        plot_domain, true_density
    """
    plot_domain = np.linspace(-float(plot_max), float(plot_max), int(plot_grids_per_unit * float(plot_max) * 2.0) + 1)
    Z = integral_even(
        unnormed_density_fig1,
        num_integral_grids_per_unit=int(integral_grids_per_unit),
        integral_max=float(integral_max),
    )
    return plot_domain, unnormed_density_fig1(plot_domain) / Z

def true_density_fig1_from_args(args) -> tuple[np.ndarray, np.ndarray]:
    """
    Compatibility wrapper for old experiments that still pass an args object.
    """
    return true_density_fig1_standard(
        plot_max=float(args.plot_max),
        plot_grids_per_unit=int(args.plot_grids_per_unit),
        integral_max=float(args.integral_max),
        integral_grids_per_unit=int(args.integral_grids_per_unit),
    )


# Parameterized fig1 family

def U_bimodal_family(
    x: np.ndarray,
    *,
    well_distance: float = 1.0,
    x_scale: float = 1.0,
    x_shift: float = 0.0,
    energy_scale: float = 1.0,
) -> np.ndarray:
    """
    Parameterized double-well family.

    Standard fig1 is recovered by:
        well_distance=1.0, x_scale=1.0, x_shift=0.0, energy_scale=1.0
    """
    x = _to_numpy_1d(x)
    z = (x - x_shift) / x_scale
    return energy_scale * 4.0 * (0.25 * z**4 - 0.5 * (well_distance**2) * z**2)

def grad_U_bimodal_family(
    x: np.ndarray,
    *,
    well_distance: float = 1.0,
    x_scale: float = 1.0,
    x_shift: float = 0.0,
    energy_scale: float = 1.0,
) -> np.ndarray:
    x = _to_numpy_1d(x)
    z = (x - x_shift) / x_scale
    grad_z = energy_scale * 4.0 * (z**3 - (well_distance**2) * z)
    return grad_z / x_scale

def density_bimodal_family(
    x: np.ndarray,
    *,
    well_distance: float = 1.0,
    x_scale: float = 1.0,
    x_shift: float = 0.0,
    energy_scale: float = 1.0,
    normalize: bool = False,
    integral_max: float = 8.0,
    integral_grids_per_unit: int = 4000,
) -> np.ndarray:
    x = _to_numpy_1d(x)
    unnorm = np.exp(
        -U_bimodal_family(
            x,
            well_distance=well_distance,
            x_scale=x_scale,
            x_shift=x_shift,
            energy_scale=energy_scale,
        )
    )
    if not normalize:
        return unnorm

    # Numerical normalization for the general family
    integral_domain = np.linspace(
        x_shift - float(integral_max),
        x_shift + float(integral_max),
        int(2.0 * integral_grids_per_unit * float(integral_max)) + 1,
    )
    y = np.exp(
        -U_bimodal_family(
            integral_domain,
            well_distance=well_distance,
            x_scale=x_scale,
            x_shift=x_shift,
            energy_scale=energy_scale,
        )
    )
    Z = np.trapz(y, integral_domain)
    return unnorm / Z


# Torch counterparts for fig1 family

def U_bimodal_family_torch(
    x: torch.Tensor,
    *,
    well_distance: float = 1.0,
    x_scale: float = 1.0,
    x_shift: float = 0.0,
    energy_scale: float = 1.0,
) -> torch.Tensor:
    z = (x - float(x_shift)) / float(x_scale)
    return float(energy_scale) * 4.0 * (0.25 * z**4 - 0.5 * (float(well_distance) ** 2) * z**2)

def grad_U_bimodal_family_torch(
    x: torch.Tensor,
    *,
    well_distance: float = 1.0,
    x_scale: float = 1.0,
    x_shift: float = 0.0,
    energy_scale: float = 1.0,
) -> torch.Tensor:
    z = (x - float(x_shift)) / float(x_scale)
    grad_z = float(energy_scale) * 4.0 * (z**3 - (float(well_distance) ** 2) * z)
    return grad_z / float(x_scale)

def density_bimodal_family_torch(
    x: torch.Tensor,
    *,
    well_distance: float = 1.0,
    x_scale: float = 1.0,
    x_shift: float = 0.0,
    energy_scale: float = 1.0,
) -> torch.Tensor:
    return torch.exp(
        -U_bimodal_family_torch(
            x,
            well_distance=well_distance,
            x_scale=x_scale,
            x_shift=x_shift,
            energy_scale=energy_scale,
        )
    )


# Standard fig3 / elliptical Gaussian : legacy math functions kept

def rotation_matrix(phi: float) -> np.ndarray:
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    return np.array([[cos_phi, -sin_phi], [sin_phi, cos_phi]], dtype=np.float64)

def covariance_matrix(a: float = A, b: float = B, phi: float = PHI) -> tuple[np.ndarray, np.ndarray]:
    R = rotation_matrix(phi)
    Sigma = R @ np.diag([a * a, b * b]) @ R.T
    inv_Sigma = np.linalg.inv(Sigma)
    return Sigma, inv_Sigma

def U_fig3_xy(xy: np.ndarray, *, a: float = A, b: float = B, phi: float = PHI) -> np.ndarray:
    xy = _to_numpy_lastdim2(xy)
    _, inv_Sigma = covariance_matrix(a, b, phi)
    return 0.5 * np.einsum("...i,ij,...j->...", xy, inv_Sigma, xy)

def grad_U_fig3_xy(xy: np.ndarray, *, a: float = A, b: float = B, phi: float = PHI) -> np.ndarray:
    xy = _to_numpy_lastdim2(xy)
    _, inv_Sigma = covariance_matrix(a, b, phi)
    return np.einsum("ij,...j->...i", inv_Sigma, xy)

def density_fig3_xy(xy: np.ndarray, *, a: float = A, b: float = B, phi: float = PHI) -> np.ndarray:
    xy = _to_numpy_lastdim2(xy)
    return np.exp(-U_fig3_xy(xy, a=a, b=b, phi=phi)) / (2.0 * np.pi * a * b)


# Torch counterparts for standard fig3

def covariance_matrix_torch(
    *,
    a: float = A,
    b: float = B,
    phi: float = PHI,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    c = math.cos(float(phi))
    s = math.sin(float(phi))
    R = torch.tensor([[c, -s], [s, c]], dtype=dtype, device=device)
    Sigma = R @ torch.diag(torch.tensor([float(a) ** 2, float(b) ** 2], dtype=dtype, device=device)) @ R.T
    inv_Sigma = torch.linalg.inv(Sigma)
    return Sigma, inv_Sigma

def U_fig3_xy_torch(xy: torch.Tensor, *, a: float = A, b: float = B, phi: float = PHI) -> torch.Tensor:
    if xy.shape[-1] != 2:
        raise ValueError(f"xy must have last dim 2, got shape {tuple(xy.shape)}")
    _, inv_Sigma = covariance_matrix_torch(a=a, b=b, phi=phi, dtype=xy.dtype, device=xy.device)
    return 0.5 * torch.einsum("...i,ij,...j->...", xy, inv_Sigma, xy)

def grad_U_fig3_xy_torch(xy: torch.Tensor, *, a: float = A, b: float = B, phi: float = PHI) -> torch.Tensor:
    if xy.shape[-1] != 2:
        raise ValueError(f"xy must have last dim 2, got shape {tuple(xy.shape)}")
    _, inv_Sigma = covariance_matrix_torch(a=a, b=b, phi=phi, dtype=xy.dtype, device=xy.device)
    return torch.einsum("ij,...j->...i", inv_Sigma, xy)

def density_fig3_xy_torch(xy: torch.Tensor, *, a: float = A, b: float = B, phi: float = PHI) -> torch.Tensor:
    return torch.exp(-U_fig3_xy_torch(xy, a=a, b=b, phi=phi)) / (2.0 * math.pi * float(a) * float(b))


# New outer API: easy-to-read "give me a distribution" functions

def get_standard_bimodal(*, name: str = "standard_bimodal", init_std: float = 1.0) -> DensitySpec:
    """
    Green channel for the classic standard bimodal / fig1 distribution.
    """
    def sample_init_fn(num_samples: int, generator: torch.Generator | None = None) -> torch.Tensor:
        return float(init_std) * torch.randn(num_samples, 1, generator=generator)
    return DensitySpec(
        name=name,
        num_dims=1,
        potential_np=U_fig1,
        grad_np=dU_dx_fig1,
        density_np=unnormed_density_fig1,
        potential_torch=lambda x: U_bimodal_family_torch(x, well_distance=1.0, x_scale=1.0, x_shift=0.0, energy_scale=1.0),
        grad_torch=lambda x: grad_U_bimodal_family_torch(x, well_distance=1.0, x_scale=1.0, x_shift=0.0, energy_scale=1.0),
        density_torch=lambda x: density_bimodal_family_torch(x, well_distance=1.0, x_scale=1.0, x_shift=0.0, energy_scale=1.0),
        sample_init_fn=sample_init_fn,
        metadata={"family": "fig1", "variant": "standard_bimodal"},
    )

def get_bimodal_family(
    *,
    well_distance: float = 1.0,
    x_scale: float = 1.0,
    x_shift: float = 0.0,
    energy_scale: float = 1.0,
    name: str = "bimodal_family",
    init_std: float | None = None,
) -> DensitySpec:
    """
    Parameterized fig1-like family built on top of the classic double-well.
    """
    if init_std is None:
        init_std = max(1.0, abs(float(x_scale)) * 1.5)

    def sample_init_fn(num_samples: int, generator: torch.Generator | None = None) -> torch.Tensor:
        return float(init_std) * torch.randn(num_samples, 1, generator=generator) + float(x_shift)

    return DensitySpec(
        name=name,
        num_dims=1,
        potential_np=lambda x: U_bimodal_family(
            x,
            well_distance=well_distance,
            x_scale=x_scale,
            x_shift=x_shift,
            energy_scale=energy_scale,
        ),
        grad_np=lambda x: grad_U_bimodal_family(
            x,
            well_distance=well_distance,
            x_scale=x_scale,
            x_shift=x_shift,
            energy_scale=energy_scale,
        ),
        density_np=lambda x: density_bimodal_family(
            x,
            well_distance=well_distance,
            x_scale=x_scale,
            x_shift=x_shift,
            energy_scale=energy_scale,
            normalize=False,
        ),
        potential_torch=lambda x: U_bimodal_family_torch(
            x,
            well_distance=well_distance,
            x_scale=x_scale,
            x_shift=x_shift,
            energy_scale=energy_scale,
        ),
        grad_torch=lambda x: grad_U_bimodal_family_torch(
            x,
            well_distance=well_distance,
            x_scale=x_scale,
            x_shift=x_shift,
            energy_scale=energy_scale,
        ),
        density_torch=lambda x: density_bimodal_family_torch(
            x,
            well_distance=well_distance,
            x_scale=x_scale,
            x_shift=x_shift,
            energy_scale=energy_scale,
        ),
        sample_init_fn=sample_init_fn,
        metadata={
            "family": "fig1",
            "variant": "bimodal_family",
            "well_distance": float(well_distance),
            "x_scale": float(x_scale),
            "x_shift": float(x_shift),
            "energy_scale": float(energy_scale),
        },
    )

def get_standard_fig3(
    *,
    a: float = A,
    b: float = B,
    phi: float = PHI,
    name: str = "standard_fig3",
    init_scale: float = 1.0,
) -> DensitySpec:
    """
    Green channel for the classic standard fig3 elliptical Gaussian.
    """
    Sigma, _ = covariance_matrix(a=a, b=b, phi=phi)
    def sample_init_fn(num_samples: int, generator: torch.Generator | None = None) -> torch.Tensor:
        z = torch.randn(num_samples, 2, generator=generator)
        L = torch.tensor(np.linalg.cholesky(Sigma), dtype=z.dtype)
        return float(init_scale) * (z @ L.T)
    return DensitySpec(
        name=name,
        num_dims=2,
        potential_np=lambda xy: U_fig3_xy(xy, a=a, b=b, phi=phi),
        grad_np=lambda xy: grad_U_fig3_xy(xy, a=a, b=b, phi=phi),
        density_np=lambda xy: density_fig3_xy(xy, a=a, b=b, phi=phi),
        potential_torch=lambda xy: U_fig3_xy_torch(xy, a=a, b=b, phi=phi),
        grad_torch=lambda xy: grad_U_fig3_xy_torch(xy, a=a, b=b, phi=phi),
        density_torch=lambda xy: density_fig3_xy_torch(xy, a=a, b=b, phi=phi),
        sample_init_fn=sample_init_fn,
        metadata={
            "family": "fig3",
            "variant": "standard_fig3",
            "a": float(a),
            "b": float(b),
            "phi": float(phi),
        },
    )

def get_gaussian_1d(*, mean: float = 0.0, std: float = 1.0, name: str = "gaussian_1d") -> DensitySpec:

    var = float(std) ** 2

    def potential_np(x: np.ndarray) -> np.ndarray:
        x = _to_numpy_1d(x)
        return 0.5 * ((x - mean) ** 2) / var

    def grad_np(x: np.ndarray) -> np.ndarray:
        x = _to_numpy_1d(x)
        return (x - mean) / var

    def density_np(x: np.ndarray) -> np.ndarray:
        x = _to_numpy_1d(x)
        return np.exp(-0.5 * ((x - mean) ** 2) / var) / math.sqrt(2.0 * math.pi * var)

    def potential_torch(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * ((x - float(mean)) ** 2) / var

    def grad_torch(x: torch.Tensor) -> torch.Tensor:
        return (x - float(mean)) / var

    def density_torch(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * ((x - float(mean)) ** 2) / var) / math.sqrt(2.0 * math.pi * var)

    def sample_init_fn(num_samples: int, generator: torch.Generator | None = None) -> torch.Tensor:
        return float(std) * torch.randn(num_samples, 1, generator=generator) + float(mean)

    return DensitySpec(
        name=name,
        num_dims=1,
        potential_np=potential_np,
        grad_np=grad_np,
        density_np=density_np,
        potential_torch=potential_torch,
        grad_torch=grad_torch,
        density_torch=density_torch,
        sample_init_fn=sample_init_fn,
        metadata={"family": "gaussian", "mean": float(mean), "std": float(std)},
    )

def get_gaussian_2d(
    *,
    mean: tuple[float, float] = (0.0, 0.0),
    a: float = 1.0,
    b: float = 1.0,
    phi: float = 0.0,
    name: str = "gaussian_2d",
) -> DensitySpec:
    return get_standard_fig3(a=a, b=b, phi=phi, name=name, init_scale=1.0)


# Legacy fig3 polar helpers kept near the end; no new compatibility layer

def xy_from_rtheta(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=-1)

def rayleigh_quotient_xy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    a: float = A,
    b: float = B,
    phi: float = PHI,
    eps: float = 1e-12,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xy = np.stack([x, y], axis=-1)
    return 2.0 * U_fig3_xy(xy, a=a, b=b, phi=phi) / np.maximum((x * x + y * y), eps)

def rayleigh_quotient_theta(theta: np.ndarray, *, a: float = A, b: float = B, phi: float = PHI) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    t = theta - phi
    cos2t = np.cos(t) ** 2
    sin2t = np.sin(t) ** 2
    return cos2t / (a * a) + sin2t / (b * b)

def U_fig3_rtheta_plain(
    r: np.ndarray,
    theta: np.ndarray,
    *,
    a: float = A,
    b: float = B,
    phi: float = PHI,
) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    return U_fig3_xy(xy_from_rtheta(r, theta), a=a, b=b, phi=phi)

def grad_U_fig3_rtheta_plain(
    r: np.ndarray,
    theta: np.ndarray,
    *,
    a: float = A,
    b: float = B,
    phi: float = PHI,
) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    return grad_U_fig3_xy(xy_from_rtheta(r, theta), a=a, b=b, phi=phi)

def density_fig3_rtheta_plain(
    r: np.ndarray,
    theta: np.ndarray,
    *,
    a: float = A,
    b: float = B,
    phi: float = PHI,
) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    return density_fig3_xy(xy_from_rtheta(r, theta), a=a, b=b, phi=phi)

def U_fig3_rtheta_joint(
    r: np.ndarray,
    theta: np.ndarray,
    *,
    a: float = A,
    b: float = B,
    phi: float = PHI,
    eps: float = 1e-12,
) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    return U_fig3_rtheta_plain(r, theta, a=a, b=b, phi=phi) - np.log(np.maximum(r, eps))

def grad_U_fig3_rtheta_joint(
    r: np.ndarray,
    theta: np.ndarray,
    *,
    a: float = A,
    b: float = B,
    phi: float = PHI,
    eps: float = 1e-12,
) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    r_safe = np.maximum(r, eps)
    q = rayleigh_quotient_theta(theta, a=a, b=b, phi=phi)
    dU_dr = r_safe * q - 1.0 / r_safe
    dq_dtheta = np.sin((theta - phi) * 2.0) * (1.0 / (b * b) - 1.0 / (a * a))
    dU_dtheta = 0.5 * (r_safe * r_safe) * dq_dtheta
    return np.stack([dU_dr, dU_dtheta], axis=-1)

__all__ = [
    "A",
    "B",
    "PHI",
    "DensitySpec",
    "integral_even",
    "U_fig1",
    "dU_dx_fig1",
    "unnormed_density_fig1",
    "true_density_fig1_standard",
    "true_density_fig1_from_args",
    "U_bimodal_family",
    "grad_U_bimodal_family",
    "density_bimodal_family",
    "U_bimodal_family_torch",
    "grad_U_bimodal_family_torch",
    "density_bimodal_family_torch",
    "rotation_matrix",
    "covariance_matrix",
    "U_fig3_xy",
    "grad_U_fig3_xy",
    "density_fig3_xy",
    "U_fig3_xy_torch",
    "grad_U_fig3_xy_torch",
    "density_fig3_xy_torch",
    "get_standard_bimodal",
    "get_bimodal_family",
    "get_standard_fig3",
    "get_gaussian_1d",
    "get_gaussian_2d",
    "xy_from_rtheta",
    "rayleigh_quotient_xy",
    "rayleigh_quotient_theta",
    "U_fig3_rtheta_plain",
    "grad_U_fig3_rtheta_plain",
    "density_fig3_rtheta_plain",
    "U_fig3_rtheta_joint",
    "grad_U_fig3_rtheta_joint",
]