from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from matplotlib import pyplot as plt
from models.toys.densities import density_fig3_xy, covariance_matrix


# basic array helpers

def _to_numpy(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)

def _flatten_samples(samples: Any) -> np.ndarray:
    arr = _to_numpy(samples)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr.reshape(-1, arr.shape[-1])
    raise ValueError(f"Unsupported sample shape: {arr.shape}")

def _flatten_1d(samples: Any) -> np.ndarray:
    arr = _flatten_samples(samples)
    if arr.shape[1] != 1:
        raise ValueError(f"Expected 1D samples, got shape {arr.shape}")
    x = arr[:, 0]
    return x[np.isfinite(x)]

def _flatten_2d(samples: Any) -> np.ndarray:
    arr = _flatten_samples(samples)
    if arr.shape[1] != 2:
        raise ValueError(f"Expected 2D samples, got shape {arr.shape}")
    mask = np.isfinite(arr).all(axis=1)
    return arr[mask]

def _ensure_xy(samples: Any, *, name: str = "samples") -> np.ndarray:
    arr = _to_numpy(samples)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    if arr.shape[0] == 2 and arr.shape[1] != 2:
        arr = arr.T
    if arr.shape[1] != 2:
        raise ValueError(f"{name} must have 2 columns, got shape {arr.shape}")
    mask = np.isfinite(arr).all(axis=1)
    return arr[mask]

# saving config
@dataclass(slots=True)
class FigureSaveConfig:
    save_dir: str | Path
    filename: str
    dpi: int = 150
    def full_path(self) -> Path:
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / self.filename


# fig1 helpers and plots

def kde_1d_gaussian(x: np.ndarray, grid: np.ndarray, bandwidth: float | None = None) -> np.ndarray:

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    grid = np.asarray(grid, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.full_like(grid, np.nan, dtype=np.float64)
    if x.size == 1:
        bw = 1e-2 if bandwidth is None else max(float(bandwidth), 1e-6)
        z = (grid - x[0]) / bw
        return np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * bw)

    if bandwidth is None:
        std = np.std(x, ddof=1)
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        sigma = min(std, iqr / 1.34) if iqr > 0 else std
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = max(np.std(x), 1e-3)
        bandwidth = max(0.9 * sigma * (x.size ** (-1.0 / 5.0)), 1e-3)
    else:
        bandwidth = max(float(bandwidth), 1e-6)

    z = (grid[:, None] - x[None, :]) / bandwidth
    return np.mean(np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * bandwidth), axis=1)

def plot_fig1_hist_kde(
    samples: Any,
    *,
    true_density: tuple[np.ndarray, np.ndarray] | None = None,
    bins: int = 120,
    kde_bandwidth: float | None = None,
    title: str = "fig1 histogram and KDE",
    x_label: str = "x",
    y_label: str = "density",
    save: FigureSaveConfig | None = None,
):
    x = _flatten_1d(samples)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(x, bins=bins, density=True, alpha=0.45, label="samples")

    if x.size > 0:
        grid = np.linspace(np.min(x), np.max(x), 800)
        kde = kde_1d_gaussian(x, grid, bandwidth=kde_bandwidth)
        ax.plot(grid, kde, linewidth=2.0, label="kde")

    if true_density is not None:
        domain, density = true_density
        ax.plot(_to_numpy(domain), _to_numpy(density), linewidth=2.0, label="true_density")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend()

    if save is not None:
        fig.savefig(save.full_path(), dpi=save.dpi, bbox_inches="tight")
    return fig, ax

def plot_fig1_density_vs_samples(
    samples: Any,
    *,
    true_density: tuple[np.ndarray, np.ndarray],
    bins: int = 60,
    title: str = "fig1 density vs samples",
    save: FigureSaveConfig | None = None,
):
    return plot_fig1_hist_kde(samples, true_density=true_density, bins=bins, title=title, save=save)

def plot_fig1_curves(
    *,
    true_density: tuple[np.ndarray, np.ndarray] | None = None,
    curves: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
    title: str = "fig1 curves",
    x_label: str = "x",
    y_label: str = "density",
    save: FigureSaveConfig | None = None,
):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    if true_density is not None:
        domain, density = true_density
        ax.plot(_to_numpy(domain), _to_numpy(density), linewidth=2.0, label="true_density")

    if curves is not None:
        for label, (domain, density) in curves.items():
            ax.plot(_to_numpy(domain), _to_numpy(density), linewidth=1.4, label=label)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend()

    if save is not None:
        fig.savefig(save.full_path(), dpi=save.dpi, bbox_inches="tight")
    return fig, ax


# fig3 basic plots

def plot_fig3_scatter(
    samples: Any,
    *,
    title: str = "fig3 scatter",
    x_label: str = "x",
    y_label: str = "y",
    alpha: float = 0.5,
    marker_size: float = 8.0,
    save: FigureSaveConfig | None = None,
):
    xy = _flatten_2d(samples)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(xy[:, 0], xy[:, 1], s=marker_size, alpha=alpha)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)

    if save is not None:
        fig.savefig(save.full_path(), dpi=save.dpi, bbox_inches="tight")
    return fig, ax

def sample_covariance_2d(samples: Any) -> np.ndarray:
    xy = _flatten_2d(samples)
    if xy.shape[0] < 2:
        return np.full((2, 2), np.nan, dtype=np.float64)
    return np.cov(xy.T, ddof=1)

def covariance_absolute_error(sample_cov: Any, true_cov: Any) -> Mapping[str, float]:
    sample_cov = _to_numpy(sample_cov)
    true_cov = _to_numpy(true_cov)
    err = np.abs(sample_cov - true_cov)
    return {
        "cov_abs_err_00": float(err[0, 0]),
        "cov_abs_err_01": float(err[0, 1]),
        "cov_abs_err_10": float(err[1, 0]),
        "cov_abs_err_11": float(err[1, 1]),
        "cov_abs_err_mean": float(np.mean(err)),
        "cov_abs_err_max": float(np.max(err)),
    }

def plot_fig3_cov_compare(
    samples: Any,
    *,
    true_cov: Any,
    title: str = "fig3 covariance compare",
    save: FigureSaveConfig | None = None,
):
    sample_cov = sample_covariance_2d(samples)
    true_cov_np = _to_numpy(true_cov)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    for ax, mat, sub_title in zip(axes, [sample_cov, true_cov_np], ["sample covariance", "true covariance"]):
        im = ax.imshow(mat)
        ax.set_title(sub_title)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    if save is not None:
        fig.savefig(save.full_path(), dpi=save.dpi, bbox_inches="tight")
    return fig, axes

def plot_trace(
    chains: Any,
    *,
    dims: Sequence[int] | None = None,
    title: str = "trace plot",
    save: FigureSaveConfig | None = None,
):
    arr = _to_numpy(chains)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"chains must be (C,T,D), got shape {arr.shape}")

    C, T, D = arr.shape
    if dims is None:
        dims = list(range(min(D, 3)))

    fig, axes = plt.subplots(len(dims), 1, figsize=(8, 2.4 * len(dims)), squeeze=False)
    for row_idx, dim in enumerate(dims):
        ax = axes[row_idx, 0]
        for c in range(C):
            ax.plot(arr[c, :, dim], linewidth=1.0)
        ax.set_title(f"{title} | dim={dim}")
        ax.set_xlabel("draw")
        ax.set_ylabel("value")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    if save is not None:
        fig.savefig(save.full_path(), dpi=save.dpi, bbox_inches="tight")
    return fig, axes


# fig3 advanced helpers

def autocorr_1d_fft(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return np.array([1.0], dtype=np.float64)
    x = x - x.mean()
    m = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=m)
    acf = np.fft.irfft(f * np.conjugate(f), n=m)[:n]
    acf = acf / acf[0]
    return acf

def iat_initial_positive_sequence(x: np.ndarray, max_lag: int | None = None) -> float:
    acf = autocorr_1d_fft(x)
    n = acf.size
    if max_lag is None:
        max_lag = min(n - 1, 2000)
    max_lag = int(max_lag)
    s = 0.0
    k = 1
    while (2 * k) <= max_lag:
        pair = float(acf[2 * k - 1] + acf[2 * k])
        if pair <= 0.0:
            break
        s += pair
        k += 1
    return float(max(1.0 + 2.0 * s, 1.0))

def iat_multidim_mean(samples_xy: np.ndarray) -> float:
    samples_xy = _ensure_xy(samples_xy, name="samples_xy")
    return 0.5 * (
        iat_initial_positive_sequence(samples_xy[:, 0]) +
        iat_initial_positive_sequence(samples_xy[:, 1])
    )

def covariance_mae(sample_cov: np.ndarray, true_cov: np.ndarray) -> float:
    return float(np.mean(np.abs(_to_numpy(sample_cov) - _to_numpy(true_cov))))

def evaluate_sampler_xy(samples_xy: np.ndarray, *, a: float, b: float, phi: float) -> tuple[float, float]:
    samples_xy = _ensure_xy(samples_xy, name="samples_xy")
    true_cov, _ = covariance_matrix(a=a, b=b, phi=phi)
    sample_cov = np.cov(samples_xy.T, ddof=1)
    err = covariance_mae(sample_cov, true_cov)
    tau = iat_multidim_mean(samples_xy)
    return err, tau

def plot_gaussian_ellipses(ax, *, a: float, b: float, phi: float, levels=(0.1, 0.3, 0.6)):
    L = 3.5 * max(a, b)
    grid = np.linspace(-L, L, 200)
    X, Y = np.meshgrid(grid, grid, indexing="xy")
    xy = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = density_fig3_xy(xy, a=a, b=b, phi=phi).reshape(X.shape)
    zmax = float(Z.max())
    lv = [zmax * float(t) for t in levels]
    ax.contour(X, Y, Z, levels=lv, linewidths=1.0)

def draw_fig3_left(ax, *, left_series: Mapping[str, list[tuple[float, float, str]]]):
    for label, pts in left_series.items():
        taus = [p[0] for p in pts]
        errs = [p[1] for p in pts]
        ax.plot(taus, errs, marker="o", linewidth=1.2, label=label)
        for tau, err, step_label in pts:
            ax.annotate(step_label, (tau, err), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("Autocorrelation time (IAT)")
    ax.set_ylabel("Average Absolute Error of Sample Covariance")
    ax.set_title("Covariance error vs autocorrelation")
    ax.legend()

def draw_fig3_right(
    ax,
    *,
    a: float,
    b: float,
    phi: float,
    scatter_inputs: Mapping[str, np.ndarray],
    scatter_first_n: int = 50,
    ellipse_levels=(0.1, 0.3, 0.6),
):
    plot_gaussian_ellipses(ax, a=a, b=b, phi=phi, levels=ellipse_levels)
    for label, samples in scatter_inputs.items():
        sxy = _ensure_xy(samples, name=f"scatter:{label}")[: int(scatter_first_n)]
        ax.scatter(sxy[:, 0], sxy[:, 1], s=18, alpha=0.8, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"First {scatter_first_n} samples")
    ax.legend()

def plot_fig3_compare(
    *,
    a: float,
    b: float,
    phi: float,
    metrics_inputs: Mapping[str, list[tuple[str, np.ndarray]]],
    scatter_inputs: Mapping[str, np.ndarray],
    num_scatter_first: int = 50,
    ellipse_levels=(0.1, 0.3, 0.6),
    title: str | None = None,
    save: FigureSaveConfig | None = None,
):
    left_series: dict[str, list[tuple[float, float, str]]] = {}
    for label, runs in metrics_inputs.items():
        pts: list[tuple[float, float, str]] = []
        for step_label, samples in runs:
            sxy = _ensure_xy(samples, name=f"{label}:{step_label}")
            err, tau = evaluate_sampler_xy(sxy, a=a, b=b, phi=phi)
            pts.append((tau, err, step_label))
        pts.sort(key=lambda t: t[0])
        left_series[label] = pts

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=120)
    draw_fig3_left(axL, left_series=left_series)
    draw_fig3_right(axR, a=a, b=b, phi=phi, scatter_inputs=scatter_inputs, scatter_first_n=num_scatter_first, ellipse_levels=ellipse_levels)

    fig.suptitle(title or f"Fig3 compare (a={a}, b={b}, phi={phi:.3f})", y=1.02)
    fig.tight_layout()
    if save is not None:
        fig.savefig(save.full_path(), dpi=save.dpi, bbox_inches="tight")
    return fig, (axL, axR), left_series

def plot_fig3_surface_density(
    *,
    plot_max: float,
    plot_grids_per_unit: int,
    a: float,
    b: float,
    phi: float,
    surface_density: np.ndarray | None = None,
    density_fn=None,
    scatter_inputs: Mapping[str, np.ndarray] | None = None,
    scatter_max_points: int = 5000,
    surface_alpha: float = 0.85,
    scatter_alpha: float = 0.35,
    title: str | None = None,
    save: FigureSaveConfig | None = None,
):
    plot_domain = np.linspace(-plot_max, plot_max, int(plot_grids_per_unit * plot_max * 2.0) + 1)
    X, Y = np.meshgrid(plot_domain, plot_domain, indexing="xy")

    if surface_density is not None:
        Z = _to_numpy(surface_density)
        if Z.shape != X.shape:
            raise ValueError(f"surface_density shape {Z.shape} != grid shape {X.shape}")
    elif density_fn is not None:
        xy = np.stack([X.ravel(), Y.ravel()], axis=1)
        Z = _to_numpy(density_fn(xy)).reshape(X.shape)
    else:
        xy = np.stack([X.ravel(), Y.ravel()], axis=1)
        Z = _to_numpy(density_fig3_xy(xy, a=a, b=b, phi=phi)).reshape(X.shape)

    fig = plt.figure(figsize=(8.0, 6.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, rstride=4, cstride=4, alpha=surface_alpha)

    if scatter_inputs is not None:
        for label, samples in scatter_inputs.items():
            pts = _ensure_xy(samples, name=f"scatter:{label}")
            if pts.shape[0] > scatter_max_points:
                idx = np.random.choice(pts.shape[0], size=scatter_max_points, replace=False)
                pts = pts[idx]
            ax.scatter(pts[:, 0], pts[:, 1], np.zeros((pts.shape[0],), dtype=np.float64), s=6, alpha=scatter_alpha, label=label)

    ax.set_title(title or f"Fig3 density surface (a={a}, b={b}, phi={phi:.3f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("density")
    if scatter_inputs:
        ax.legend(loc="upper right")

    plt.tight_layout()
    if save is not None:
        fig.savefig(save.full_path(), dpi=save.dpi, bbox_inches="tight")
    return fig, ax

__all__ = [
    "FigureSaveConfig",
    "plot_fig1_hist_kde",
    "plot_fig1_density_vs_samples",
    "plot_fig1_curves",
    "plot_fig3_scatter",
    "sample_covariance_2d",
    "covariance_absolute_error",
    "plot_fig3_cov_compare",
    "plot_trace",
    "autocorr_1d_fft",
    "iat_initial_positive_sequence",
    "iat_multidim_mean",
    "covariance_mae",
    "evaluate_sampler_xy",
    "plot_gaussian_ellipses",
    "draw_fig3_left",
    "draw_fig3_right",
    "plot_fig3_compare",
    "plot_fig3_surface_density",
]
