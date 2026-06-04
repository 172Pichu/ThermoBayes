from __future__ import annotations

import math
from functools import partial
from typing import Any

import torch

from models.configs import AdaptiveSGHMCConfig, SGNHTConfig, BlackJAXSGNHTConfig
from models.results import SamplingResult, finalize_result
from models.targets import SamplerTarget

def sample_sghmc(target: SamplerTarget, config: AdaptiveSGHMCConfig) -> SamplingResult:

    num_dims = int(target.num_dims)
    grad_fn = target.grad_estimator

    seed = int(config.budget.seed)
    num_chains = int(config.budget.num_chains)
    num_samples = int(config.budget.num_samples)
    burn_in = int(config.budget.burn_in)
    keep_every = int(config.budget.keep_every)
    inner_steps = int(config.budget.inner_steps)

    reset_prob = float(config.thermo.reset_prob)
    grad_noise_std = math.sqrt(max(0.0, float(config.thermo.grad_noise_var)))

    eps = float(config.step_size)
    eps2 = eps * eps
    eps4 = eps2 * eps2
    mdecay = float(config.mdecay)
    warm_up_frac = float(config.warm_up_frac)
    adapt_start = int(burn_in * warm_up_frac)
    ema = float(config.ema_beta)
    one_minus_ema = 1.0 - ema

    gens = [torch.Generator().manual_seed(seed + chain_idx) for chain_idx in range(num_chains)]
    x = [torch.randn(num_dims, generator=gens[chain_idx]) for chain_idx in range(num_chains)]
    v = [torch.zeros_like(x[chain_idx]) for chain_idx in range(num_chains)]
    V_hat = [torch.ones_like(x[chain_idx]) for chain_idx in range(num_chains)]

    atomic_steps = 0
    outer_steps = 0

    for _ in range(burn_in):

        outer_steps += 1

        if reset_prob > 0.0:
            for chain_idx in range(num_chains):
                if reset_prob >= 1.0 or torch.rand((), generator=gens[chain_idx]).item() < reset_prob:
                    prec = 1.0 / (torch.sqrt(V_hat[chain_idx]) + 1e-8)
                    v[chain_idx] = eps * torch.sqrt(prec) * torch.randn(num_dims, generator=gens[chain_idx])

        for _ in range(inner_steps):

            atomic_steps += 1

            for chain_idx in range(num_chains):

                g_det = grad_fn(x[chain_idx])
                g = g_det if grad_noise_std == 0.0 else g_det + grad_noise_std * torch.randn(num_dims, generator=gens[chain_idx])

                if outer_steps >= adapt_start:
                    V_hat[chain_idx] = ema * V_hat[chain_idx] + one_minus_ema * (g_det * g_det)

                prec = 1.0 / (torch.sqrt(V_hat[chain_idx]) + 1e-8)
                mom_noise_var = torch.clamp(2.0 * eps2 * mdecay * prec - eps4, min=0.0)
                mom_noise = torch.sqrt(mom_noise_var) * torch.randn(num_dims, generator=gens[chain_idx])

                x[chain_idx] = x[chain_idx] + v[chain_idx]
                v[chain_idx] = v[chain_idx] - eps2 * (prec * g) - mdecay * v[chain_idx] + mom_noise

    precs = []
    mom_noise_stds = []

    for chain_idx in range(num_chains):
        prec = 1.0 / (torch.sqrt(V_hat[chain_idx]) + 1e-8)
        precs.append(prec)
        mom_noise_var = torch.clamp(2.0 * eps2 * mdecay * prec - eps4, min=0.0)
        mom_noise_stds.append(torch.sqrt(mom_noise_var))

    collected = [0] * num_chains
    all_chains: list[list[torch.Tensor]] = [[] for _ in range(num_chains)]

    while min(collected) < num_samples:

        outer_steps += 1

        if reset_prob > 0.0:
            for chain_idx in range(num_chains):
                if reset_prob >= 1.0 or torch.rand((), generator=gens[chain_idx]).item() < reset_prob:
                    v[chain_idx] = eps * torch.sqrt(precs[chain_idx]) * torch.randn(num_dims, generator=gens[chain_idx])

        for _ in range(inner_steps):

            atomic_steps += 1

            for chain_idx in range(num_chains):

                g_det = grad_fn(x[chain_idx])
                g = g_det if grad_noise_std == 0.0 else g_det + grad_noise_std * torch.randn(num_dims, generator=gens[chain_idx])

                prec = precs[chain_idx]
                mom_noise = mom_noise_stds[chain_idx] * torch.randn(num_dims, generator=gens[chain_idx])

                x[chain_idx] = x[chain_idx] + v[chain_idx]
                v[chain_idx] = v[chain_idx] - eps2 * (prec * g) - mdecay * v[chain_idx] + mom_noise

        if (outer_steps - burn_in) % keep_every == 0:
            for chain_idx in range(num_chains):
                if collected[chain_idx] < num_samples:
                    all_chains[chain_idx].append(x[chain_idx].detach().clone())
                    collected[chain_idx] += 1

    chains = torch.stack([torch.stack(chain, dim=0) for chain in all_chains], dim=0)
    samples = chains.reshape(-1, num_dims)

    metadata = {
        "atomic_steps": atomic_steps,
        "outer_steps": outer_steps,
        "temperature": float(config.thermo.temperature),
        "reset_prob": reset_prob,
        "grad_noise_var": float(config.thermo.grad_noise_var),
        "step_size": eps,
        "inner_steps": inner_steps,
        "keep_every": keep_every,
        "burn_in": burn_in,
    }

    diagnostics: dict[str, Any] = {}

    final_states = {
        "x": torch.stack(x, dim=0),
        "v": torch.stack(v, dim=0),
        "V_hat": torch.stack(V_hat, dim=0),
        "precs": torch.stack(precs, dim=0),
    }

    return finalize_result(
        sampler_name=config.name,
        sampler_config=config,
        chains=chains,
        samples=samples,
        final_states=final_states,
        diagnostics=diagnostics,
        metadata=metadata,
    )

def sample_sgnht(target: SamplerTarget, config: SGNHTConfig) -> SamplingResult:

    num_dims = int(target.num_dims)
    grad_fn = target.grad_estimator

    seed = int(config.budget.seed)
    num_chains = int(config.budget.num_chains)
    num_samples = int(config.budget.num_samples)
    burn_in = int(config.budget.burn_in)
    keep_every = int(config.budget.keep_every)
    inner_steps = int(config.budget.inner_steps)

    temperature = float(config.thermo.temperature)
    reset_prob = float(config.thermo.reset_prob)
    grad_noise_std = math.sqrt(max(0.0, float(config.thermo.grad_noise_var)))

    eps = float(config.step_size)
    diffusion_factor = float(config.diffusion_factor)

    if diffusion_factor < 0.0:
        raise ValueError("diffusion_factor must be non-negative")
    if config.reset_thermostat_to_A and config.reset_thermostat_to_midpoint:
        raise ValueError("please select one of thermostat reset methods")

    thermostat_init = diffusion_factor if config.thermostat_init is None else float(config.thermostat_init)
    momentum_noise_std = math.sqrt(max(0.0, 2.0 * eps * diffusion_factor))

    gens = [torch.Generator().manual_seed(seed + chain_idx) for chain_idx in range(num_chains)]
    x = [torch.randn(num_dims, generator=gens[chain_idx]) for chain_idx in range(num_chains)]
    p = [torch.randn(num_dims, generator=gens[chain_idx]) for chain_idx in range(num_chains)]
    xi = [torch.tensor(thermostat_init, dtype=x[chain_idx].dtype) for chain_idx in range(num_chains)]

    atomic_steps = 0
    outer_steps = 0

    for _ in range(burn_in):

        outer_steps += 1

        if reset_prob > 0.0:
            for chain_idx in range(num_chains):
                if reset_prob >= 1.0 or torch.rand((), generator=gens[chain_idx]).item() < reset_prob:
                    p[chain_idx] = torch.randn(num_dims, generator=gens[chain_idx])
                    if config.reset_thermostat:
                        if config.reset_thermostat_to_A:
                            xi[chain_idx] = torch.tensor(diffusion_factor, dtype=x[chain_idx].dtype)
                        elif config.reset_thermostat_to_midpoint:
                            xi[chain_idx] = 0.5 * (xi[chain_idx] + diffusion_factor)
                        else:
                            xi[chain_idx] = torch.tensor(thermostat_init, dtype=x[chain_idx].dtype)

        for _ in range(inner_steps):

            atomic_steps += 1

            for chain_idx in range(num_chains):

                g_det = grad_fn(x[chain_idx])
                g = g_det if grad_noise_std == 0.0 else g_det + grad_noise_std * torch.randn(num_dims, generator=gens[chain_idx])

                mom_noise = momentum_noise_std * torch.randn(num_dims, generator=gens[chain_idx])

                p[chain_idx] = p[chain_idx] - eps * g - eps * xi[chain_idx] * p[chain_idx] + mom_noise
                x[chain_idx] = x[chain_idx] + eps * p[chain_idx]
                xi[chain_idx] = xi[chain_idx] + eps * (p[chain_idx].dot(p[chain_idx]) / num_dims - temperature)

    collected = [0] * num_chains
    all_chains: list[list[torch.Tensor]] = [[] for _ in range(num_chains)]

    while min(collected) < num_samples:

        outer_steps += 1

        if reset_prob > 0.0:
            for chain_idx in range(num_chains):
                if reset_prob >= 1.0 or torch.rand((), generator=gens[chain_idx]).item() < reset_prob:
                    p[chain_idx] = torch.randn(num_dims, generator=gens[chain_idx])
                    if config.reset_thermostat:
                        if config.reset_thermostat_to_A:
                            xi[chain_idx] = torch.tensor(diffusion_factor, dtype=x[chain_idx].dtype)
                        elif config.reset_thermostat_to_midpoint:
                            xi[chain_idx] = 0.5 * (xi[chain_idx] + diffusion_factor)
                        else:
                            xi[chain_idx] = torch.tensor(thermostat_init, dtype=x[chain_idx].dtype)

        for _ in range(inner_steps):

            atomic_steps += 1

            for chain_idx in range(num_chains):

                g_det = grad_fn(x[chain_idx])
                g = g_det if grad_noise_std == 0.0 else g_det + grad_noise_std * torch.randn(num_dims, generator=gens[chain_idx])

                mom_noise = momentum_noise_std * torch.randn(num_dims, generator=gens[chain_idx])

                p[chain_idx] = p[chain_idx] - eps * g - eps * xi[chain_idx] * p[chain_idx] + mom_noise
                x[chain_idx] = x[chain_idx] + eps * p[chain_idx]
                xi[chain_idx] = xi[chain_idx] + eps * (p[chain_idx].dot(p[chain_idx]) / num_dims - temperature)

        if (outer_steps - burn_in) % keep_every == 0:
            for chain_idx in range(num_chains):
                if collected[chain_idx] < num_samples:
                    all_chains[chain_idx].append(x[chain_idx].detach().clone())
                    collected[chain_idx] += 1

    chains = torch.stack([torch.stack(chain, dim=0) for chain in all_chains], dim=0)
    samples = chains.reshape(-1, num_dims)

    metadata = {
        "atomic_steps": atomic_steps,
        "outer_steps": outer_steps,
        "temperature": temperature,
        "reset_prob": reset_prob,
        "grad_noise_var": float(config.thermo.grad_noise_var),
        "step_size": eps,
        "diffusion_factor": diffusion_factor,
        "inner_steps": inner_steps,
        "keep_every": keep_every,
        "burn_in": burn_in,
    }

    diagnostics: dict[str, Any] = {}

    final_states = {
        "x": torch.stack(x, dim=0),
        "p": torch.stack(p, dim=0),
        "xi": torch.stack(xi, dim=0),
    }

    return finalize_result(
        sampler_name=config.name,
        sampler_config=config,
        chains=chains,
        samples=samples,
        final_states=final_states,
        diagnostics=diagnostics,
        metadata=metadata,
    )


def sample_sgnht_bjx(target: SamplerTarget, config: BlackJAXSGNHTConfig) -> SamplingResult:
    """Run BlackJAX's SGNHT diffusion step through the ThermoBayes sampler API.

    Fast path:
        If the target provides ``logdensity_grad_keyed(theta, key)``, the loop is
        executed in coarse JAX blocks. This is the intended path for the new
        model_jax.py target and avoids Python dispatch at every atomic step.

    Fallback path:
        If only the old Torch-style ``grad_estimator(theta)`` is available, the
        adapter converts ``logdensity_grad = -grad_U`` step by step. This is kept
        for compatibility but is not suitable for speed comparison.
    """

    try:
        import jax
        import jax.numpy as jnp
        from blackjax.sgmcmc.diffusions import sgnht as bjx_sgnht
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise ImportError(
            "sample_sgnht_bjx requires jax and blackjax. Install them in this "
            "environment before running --method sgnht_bjx."
        ) from exc

    import numpy as np

    num_dims = int(target.num_dims)
    grad_fn = target.grad_estimator
    logdensity_grad_keyed = getattr(target, "logdensity_grad_keyed", None)
    target_backend = getattr(target, "metadata", {}).get("backend", "torch")
    use_jax_fast_path = callable(logdensity_grad_keyed) and str(target_backend).lower() == "jax"

    seed = int(config.budget.seed)
    num_chains = int(config.budget.num_chains)
    num_samples = int(config.budget.num_samples)
    burn_in = int(config.budget.burn_in)
    keep_every = int(config.budget.keep_every)
    inner_steps = int(config.budget.inner_steps)

    temperature = float(config.thermo.temperature)
    reset_prob = float(config.thermo.reset_prob)
    grad_noise_std = math.sqrt(max(0.0, float(config.thermo.grad_noise_var)))

    eps = float(config.step_size)
    alpha = float(config.diffusion_factor)
    beta = float(config.bjx_beta)

    if alpha < 0.0:
        raise ValueError("diffusion_factor / BlackJAX alpha must be non-negative")
    if eps <= 0.0:
        raise ValueError("step_size must be positive")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if (2.0 * alpha - eps * temperature * beta) < 0.0:
        raise ValueError("BlackJAX SGNHT noise variance is negative: 2*alpha - step_size*temperature*beta < 0")
    if config.reset_thermostat_to_A and config.reset_thermostat_to_midpoint:
        raise ValueError("please select one of thermostat reset methods")

    thermostat_init = alpha if config.thermostat_init is None else float(config.thermostat_init)
    one_step = bjx_sgnht(alpha=alpha, beta=beta)

    # ------------------------------------------------------------------
    # Fast pure-JAX path. This is what we want for fair BlackJAX timing.
    # It currently assumes reset_prob == 0.0, which is the default setup in
    # the true-dataset 10-fold experiments. Reset support can be added later.
    # ------------------------------------------------------------------
    if use_jax_fast_path:
        if reset_prob != 0.0:
            raise NotImplementedError(
                "The fast pure-JAX sgnht_bjx path currently supports reset_prob=0.0 only. "
                "Use reset_prob=0.0 for the fair BlackJAX comparison."
            )

        key = jax.random.PRNGKey(seed)
        key, kx, kp = jax.random.split(key, 3)
        x = jax.random.normal(kx, (num_chains, num_dims), dtype=jnp.float32)
        p = jax.random.normal(kp, (num_chains, num_dims), dtype=jnp.float32)
        xi = jnp.full((num_chains,), thermostat_init, dtype=jnp.float32)

        def _single_chain_step(x_i, p_i, xi_i, key_step_i, key_grad_i, key_grad_noise_i):
            log_g = logdensity_grad_keyed(x_i, key_grad_i)
            log_g = jnp.asarray(log_g, dtype=jnp.float32)
            if grad_noise_std > 0.0:
                # grad_noise_var is defined on grad_U.  Since BlackJAX consumes
                # logdensity_grad = -grad_U, adding zero-mean Gaussian noise is
                # distributionally equivalent after the sign flip.
                log_g = log_g + float(grad_noise_std) * jax.random.normal(
                    key_grad_noise_i, (num_dims,), dtype=jnp.float32
                )
            new_x, new_p, new_xi = one_step(
                key_step_i,
                x_i,
                p_i,
                xi_i,
                log_g,
                eps,
                temperature,
            )
            return new_x, new_p, new_xi

        def _one_atomic_all(carry, _):
            x_c, p_c, xi_c, key_c = carry
            key_c, key_step, key_grad, key_grad_noise = jax.random.split(key_c, 4)
            step_keys = jax.random.split(key_step, num_chains)
            grad_keys = jax.random.split(key_grad, num_chains)
            grad_noise_keys = jax.random.split(key_grad_noise, num_chains)
            x_next, p_next, xi_next = jax.vmap(_single_chain_step)(
                x_c, p_c, xi_c, step_keys, grad_keys, grad_noise_keys
            )
            return (x_next, p_next, xi_next, key_c), None

        @partial(jax.jit, static_argnames=("n_steps",))
        def _run_steps(x_in, p_in, xi_in, key_in, *, n_steps: int):
            # jax.lax.scan requires length to be a concrete Python int.
            # Mark n_steps as static so burn-in and sampling blocks compile safely.
            return jax.lax.scan(
                _one_atomic_all,
                (x_in, p_in, xi_in, key_in),
                xs=None,
                length=int(n_steps),
            )[0]

        atomic_steps = 0
        outer_steps = 0

        burnin_atomic_steps = int(burn_in * inner_steps)
        if burnin_atomic_steps > 0:
            x, p, xi, key = _run_steps(x, p, xi, key, n_steps=burnin_atomic_steps)
            # Force completion here so runtime includes compilation + execution.
            x.block_until_ready()
            atomic_steps += burnin_atomic_steps
            outer_steps += burn_in

        saved = []
        sample_block_steps = int(max(1, keep_every) * inner_steps)
        for _ in range(num_samples):
            x, p, xi, key = _run_steps(x, p, xi, key, n_steps=sample_block_steps)
            x.block_until_ready()
            atomic_steps += sample_block_steps
            outer_steps += max(1, keep_every)
            saved.append(np.asarray(jax.device_get(x), dtype=np.float32))

        # saved: list of (C,D) -> chains: (C,T,D)
        chains_np = np.stack(saved, axis=1)
        chains = torch.from_numpy(chains_np.copy())
        samples = chains.reshape(-1, num_dims)

        metadata = {
            "atomic_steps": atomic_steps,
            "outer_steps": outer_steps,
            "temperature": temperature,
            "reset_prob": reset_prob,
            "grad_noise_var": float(config.thermo.grad_noise_var),
            "step_size": eps,
            "diffusion_factor": alpha,
            "blackjax_alpha": alpha,
            "blackjax_beta": beta,
            "inner_steps": inner_steps,
            "keep_every": keep_every,
            "burn_in": burn_in,
            "backend": "blackjax.sgmcmc.diffusions.sgnht",
            "bridge": "pure_jax_target_gradients_jitted_blocks",
            "gradient_convention": "logdensity_grad = - target.grad_estimator(theta)",
            "update_order": "blackjax: position_then_momentum_then_xi",
        }

        final_states = {
            "x": torch.from_numpy(np.asarray(jax.device_get(x), dtype=np.float32).copy()),
            "p": torch.from_numpy(np.asarray(jax.device_get(p), dtype=np.float32).copy()),
            "xi": torch.from_numpy(np.asarray(jax.device_get(xi), dtype=np.float32).copy()),
        }

        return finalize_result(
            sampler_name=config.name,
            sampler_config=config,
            chains=chains,
            samples=samples,
            final_states=final_states,
            diagnostics={},
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Compatibility fallback: BlackJAX step + Torch/Python target gradient.
    # ------------------------------------------------------------------
    key = jax.random.PRNGKey(seed)
    chain_keys = list(jax.random.split(key, num_chains))

    x = []
    p = []
    xi = []
    for chain_idx in range(num_chains):
        kx, kp, knext = jax.random.split(chain_keys[chain_idx], 3)
        chain_keys[chain_idx] = knext
        x.append(jax.random.normal(kx, (num_dims,), dtype=jnp.float32))
        p.append(jax.random.normal(kp, (num_dims,), dtype=jnp.float32))
        xi.append(jnp.asarray(thermostat_init, dtype=jnp.float32))

    torch_gens = [torch.Generator().manual_seed(seed + 100_000 + chain_idx) for chain_idx in range(num_chains)]

    def _logdensity_grad(position_jax, chain_idx: int):
        position_np = np.asarray(position_jax, dtype=np.float32)
        theta = torch.from_numpy(position_np.copy())
        g_det = grad_fn(theta)
        if grad_noise_std > 0.0:
            g_det = g_det + grad_noise_std * torch.randn(num_dims, generator=torch_gens[chain_idx])
        return jnp.asarray((-g_det.detach().cpu().numpy()).astype(np.float32))

    def _maybe_reset(chain_idx: int):
        nonlocal x, p, xi, chain_keys
        if reset_prob <= 0.0:
            return
        ku, kp, knext = jax.random.split(chain_keys[chain_idx], 3)
        chain_keys[chain_idx] = knext
        do_reset = reset_prob >= 1.0 or bool(jax.random.uniform(ku, ()).item() < reset_prob)
        if not do_reset:
            return
        p[chain_idx] = jax.random.normal(kp, (num_dims,), dtype=jnp.float32)
        if config.reset_thermostat:
            if config.reset_thermostat_to_A:
                xi[chain_idx] = jnp.asarray(alpha, dtype=jnp.float32)
            elif config.reset_thermostat_to_midpoint:
                xi[chain_idx] = jnp.asarray(0.5 * (float(xi[chain_idx]) + alpha), dtype=jnp.float32)
            else:
                xi[chain_idx] = jnp.asarray(thermostat_init, dtype=jnp.float32)

    def _one_atomic_step(chain_idx: int):
        nonlocal x, p, xi, chain_keys
        kg, knext = jax.random.split(chain_keys[chain_idx], 2)
        chain_keys[chain_idx] = knext
        log_g = _logdensity_grad(x[chain_idx], chain_idx)
        x[chain_idx], p[chain_idx], xi[chain_idx] = one_step(
            kg,
            x[chain_idx],
            p[chain_idx],
            xi[chain_idx],
            log_g,
            eps,
            temperature,
        )

    atomic_steps = 0
    outer_steps = 0

    for _ in range(burn_in):
        outer_steps += 1
        for chain_idx in range(num_chains):
            _maybe_reset(chain_idx)
        for _ in range(inner_steps):
            atomic_steps += 1
            for chain_idx in range(num_chains):
                _one_atomic_step(chain_idx)

    collected = [0] * num_chains
    all_chains: list[list[torch.Tensor]] = [[] for _ in range(num_chains)]

    while min(collected) < num_samples:
        outer_steps += 1
        for chain_idx in range(num_chains):
            _maybe_reset(chain_idx)
        for _ in range(inner_steps):
            atomic_steps += 1
            for chain_idx in range(num_chains):
                _one_atomic_step(chain_idx)

        if (outer_steps - burn_in) % keep_every == 0:
            for chain_idx in range(num_chains):
                if collected[chain_idx] < num_samples:
                    all_chains[chain_idx].append(torch.from_numpy(np.asarray(x[chain_idx], dtype=np.float32).copy()))
                    collected[chain_idx] += 1

    chains = torch.stack([torch.stack(chain, dim=0) for chain in all_chains], dim=0)
    samples = chains.reshape(-1, num_dims)

    metadata = {
        "atomic_steps": atomic_steps,
        "outer_steps": outer_steps,
        "temperature": temperature,
        "reset_prob": reset_prob,
        "grad_noise_var": float(config.thermo.grad_noise_var),
        "step_size": eps,
        "diffusion_factor": alpha,
        "blackjax_alpha": alpha,
        "blackjax_beta": beta,
        "inner_steps": inner_steps,
        "keep_every": keep_every,
        "burn_in": burn_in,
        "backend": "blackjax.sgmcmc.diffusions.sgnht",
        "bridge": "blackjax_step_with_torch_target_gradients",
        "gradient_convention": "logdensity_grad = - target.grad_estimator(theta)",
        "update_order": "blackjax: position_then_momentum_then_xi",
    }

    final_states = {
        "x": torch.stack([torch.from_numpy(np.asarray(v, dtype=np.float32).copy()) for v in x], dim=0),
        "p": torch.stack([torch.from_numpy(np.asarray(v, dtype=np.float32).copy()) for v in p], dim=0),
        "xi": torch.tensor([float(v) for v in xi], dtype=torch.float32),
    }

    return finalize_result(
        sampler_name=config.name,
        sampler_config=config,
        chains=chains,
        samples=samples,
        final_states=final_states,
        diagnostics={},
        metadata=metadata,
    )


def _torch_device_from_target(target: SamplerTarget) -> torch.device:
    meta = getattr(target, "metadata", {}) or {}
    device_name = meta.get("torch_device")
    if device_name is None:
        return torch.device("cpu")
    return torch.device(str(device_name))


def _torch_dtype_from_target(target: SamplerTarget) -> torch.dtype:
    meta = getattr(target, "metadata", {}) or {}
    dtype_name = str(meta.get("torch_dtype", "torch.float32"))
    if dtype_name.endswith("float64"):
        return torch.float64
    if dtype_name.endswith("float16"):
        return torch.float16
    if dtype_name.endswith("bfloat16"):
        return torch.bfloat16
    return torch.float32


def _batch_grad_or_loop(target: SamplerTarget, x: torch.Tensor) -> torch.Tensor:
    batch_grad_fn = getattr(target, "batch_grad_estimator", None)
    if callable(batch_grad_fn):
        return batch_grad_fn(x)
    grads = [target.grad_estimator(theta) for theta in x.reshape(-1, x.shape[-1])]
    return torch.stack(grads, dim=0).reshape_as(x)


def _make_sgnht_step_fn(*, compile_step: bool, device: torch.device | None = None):
    def _step(x: torch.Tensor, p: torch.Tensor, xi: torch.Tensor, grad: torch.Tensor, mom_noise: torch.Tensor, eps: float, temperature: float):
        p_next = p - eps * grad - eps * xi[:, None] * p + mom_noise
        x_next = x + eps * p_next
        xi_next = xi + eps * (torch.mean(p_next * p_next, dim=1) - temperature)
        return x_next, p_next, xi_next

    if not compile_step:
        return _step, {"compiled_step": False}
    if device is not None and device.type == "cpu":
        return _step, {"compiled_step": False, "compile_skipped": "cpu_device"}
    try:
        compiled = torch.compile(_step)  # type: ignore[attr-defined]
        return compiled, {"compiled_step": True}
    except Exception as exc:  # pragma: no cover - depends on torch version/backend
        return _step, {"compiled_step": False, "compile_error": repr(exc)}


def _sample_sgnht_vectorized_impl(target: SamplerTarget, config: SGNHTConfig, *, compile_step: bool) -> SamplingResult:
    """Vectorized SGNHT implementation.

    This keeps the public sampler contract unchanged:
        sampler(target, config) -> SamplingResult

    The main difference from ``sample_sgnht`` is that all chains live in batched
    tensors ``x: (C,D)``, ``p: (C,D)``, and ``xi: (C,)``.  If the target exposes
    ``batch_grad_estimator(x)``, one autograd call computes gradients for all
    chains; otherwise the function safely falls back to the old per-chain loop.
    """

    num_dims = int(target.num_dims)
    seed = int(config.budget.seed)
    num_chains = int(config.budget.num_chains)
    num_samples = int(config.budget.num_samples)
    burn_in = int(config.budget.burn_in)
    keep_every = max(1, int(config.budget.keep_every))
    inner_steps = int(config.budget.inner_steps)

    temperature = float(config.thermo.temperature)
    reset_prob = float(config.thermo.reset_prob)
    grad_noise_std = math.sqrt(max(0.0, float(config.thermo.grad_noise_var)))

    eps = float(config.step_size)
    diffusion_factor = float(config.diffusion_factor)
    if diffusion_factor < 0.0:
        raise ValueError("diffusion_factor must be non-negative")
    if eps <= 0.0:
        raise ValueError("step_size must be positive")
    if config.reset_thermostat_to_A and config.reset_thermostat_to_midpoint:
        raise ValueError("please select one of thermostat reset methods")

    thermostat_init = diffusion_factor if config.thermostat_init is None else float(config.thermostat_init)
    momentum_noise_std = math.sqrt(max(0.0, 2.0 * eps * diffusion_factor))

    device = _torch_device_from_target(target)
    dtype = _torch_dtype_from_target(target)
    gen = torch.Generator(device=device if device.type != "cpu" else "cpu").manual_seed(seed)

    x = torch.randn((num_chains, num_dims), generator=gen, device=device, dtype=dtype)
    p = torch.randn((num_chains, num_dims), generator=gen, device=device, dtype=dtype)
    xi = torch.full((num_chains,), thermostat_init, device=device, dtype=dtype)

    step_fn, compile_meta = _make_sgnht_step_fn(compile_step=compile_step, device=device)
    use_batch_grad = callable(getattr(target, "batch_grad_estimator", None))

    atomic_steps = 0
    outer_steps = 0
    grad_eval_calls = 0

    def _maybe_reset() -> None:
        nonlocal p, xi
        if reset_prob <= 0.0:
            return
        if reset_prob >= 1.0:
            mask = torch.ones((num_chains,), device=device, dtype=torch.bool)
        else:
            mask = torch.rand((num_chains,), generator=gen, device=device, dtype=dtype) < reset_prob
        if not bool(torch.any(mask).item()):
            return
        new_p = torch.randn((num_chains, num_dims), generator=gen, device=device, dtype=dtype)
        p = torch.where(mask[:, None], new_p, p)
        if config.reset_thermostat:
            if config.reset_thermostat_to_A:
                new_xi = torch.full_like(xi, diffusion_factor)
            elif config.reset_thermostat_to_midpoint:
                new_xi = 0.5 * (xi + diffusion_factor)
            else:
                new_xi = torch.full_like(xi, thermostat_init)
            xi = torch.where(mask, new_xi, xi)

    def _one_outer() -> None:
        nonlocal x, p, xi, atomic_steps, grad_eval_calls
        _maybe_reset()
        for _ in range(inner_steps):
            g_det = _batch_grad_or_loop(target, x)
            grad_eval_calls += 1 if use_batch_grad else num_chains
            g = g_det if grad_noise_std == 0.0 else g_det + grad_noise_std * torch.randn((num_chains, num_dims), generator=gen, device=device, dtype=dtype)
            mom_noise = momentum_noise_std * torch.randn((num_chains, num_dims), generator=gen, device=device, dtype=dtype)
            x, p, xi = step_fn(x, p, xi, g, mom_noise, eps, temperature)
            atomic_steps += 1

    for _ in range(burn_in):
        outer_steps += 1
        _one_outer()

    saved: list[torch.Tensor] = []
    for _ in range(num_samples):
        for _ in range(keep_every):
            outer_steps += 1
            _one_outer()
        saved.append(x.detach().clone())

    chains = torch.stack(saved, dim=1)  # (C,T,D)
    samples = chains.reshape(-1, num_dims)

    metadata = {
        "atomic_steps": atomic_steps,
        "outer_steps": outer_steps,
        "temperature": temperature,
        "reset_prob": reset_prob,
        "grad_noise_var": float(config.thermo.grad_noise_var),
        "step_size": eps,
        "diffusion_factor": diffusion_factor,
        "inner_steps": inner_steps,
        "keep_every": keep_every,
        "burn_in": burn_in,
        "backend": "torch",
        "bridge": "batched_torch_target_gradients" if use_batch_grad else "batched_sampler_with_looped_target_gradients",
        "vectorized_chains": True,
        "compile_requested": bool(compile_step),
        "grad_eval_calls": float(grad_eval_calls),
        **compile_meta,
    }

    final_states = {"x": x, "p": p, "xi": xi}

    return finalize_result(
        sampler_name=config.name,
        sampler_config=config,
        chains=chains,
        samples=samples,
        final_states=final_states,
        diagnostics={},
        metadata=metadata,
    )


def sample_sgnht_vectorized(target: SamplerTarget, config: SGNHTConfig) -> SamplingResult:
    return _sample_sgnht_vectorized_impl(target, config, compile_step=False)


def sample_sgnht_vectorized_compile(target: SamplerTarget, config: SGNHTConfig) -> SamplingResult:
    return _sample_sgnht_vectorized_impl(target, config, compile_step=True)


SAMPLER_REGISTRY = {
    "adaptive_sghmc": sample_sghmc,
    "sgnht": sample_sgnht,
    "sgnht_bjx": sample_sgnht_bjx,
    "sgnht_vectorized": sample_sgnht_vectorized,
    "sgnht_vectorized_compile": sample_sgnht_vectorized_compile,
}
