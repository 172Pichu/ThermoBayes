from __future__ import annotations

import torch

from samplers import (
    _sgnht_step_momentum_first,
    _sgnht_step_posteriors_order,
)


def _state(dtype: torch.dtype = torch.float64):
    x = torch.tensor([0.3, -0.2, 0.5], dtype=dtype)
    p = torch.tensor([0.4, 0.1, -0.3], dtype=dtype)
    xi = torch.tensor(0.07, dtype=dtype)
    grad = torch.tensor([0.2, -0.4, 0.6], dtype=dtype)
    noise = torch.tensor([0.01, -0.02, 0.03], dtype=dtype)
    return x, p, xi, grad, noise


def test_original_momentum_first_formula() -> None:
    x, p, xi, grad, noise = _state()
    eps = 0.005
    temperature = 1.0

    x_next, p_next, xi_next = _sgnht_step_momentum_first(
        x, p, xi, grad, noise, eps, temperature
    )

    p_expected = p - eps * grad - eps * xi * p + noise
    x_expected = x + eps * p_expected
    xi_expected = xi + eps * (torch.mean(p_expected**2) - temperature)

    torch.testing.assert_close(p_next, p_expected)
    torch.testing.assert_close(x_next, x_expected)
    torch.testing.assert_close(xi_next, xi_expected)


def test_posteriors_order_formula() -> None:
    x, p, xi, grad, noise = _state()
    eps = 0.005
    temperature = 1.0

    x_next, p_next, xi_next = _sgnht_step_posteriors_order(
        x, p, xi, grad, noise, eps, temperature
    )

    x_expected = x + eps * p
    xi_expected = xi + eps * (torch.mean(p**2) - temperature)
    p_expected = p - eps * grad - eps * xi * p + noise

    torch.testing.assert_close(x_next, x_expected)
    torch.testing.assert_close(xi_next, xi_expected)
    torch.testing.assert_close(p_next, p_expected)


def test_posteriors_order_batched_matches_chain_loop() -> None:
    x, p, xi, grad, noise = _state()
    xb = torch.stack([x, x + 0.1], dim=0)
    pb = torch.stack([p, p - 0.1], dim=0)
    xib = torch.stack([xi, xi + 0.02], dim=0)
    gb = torch.stack([grad, grad * 0.5], dim=0)
    nb = torch.stack([noise, -noise], dim=0)

    batched = _sgnht_step_posteriors_order(xb, pb, xib, gb, nb, 0.005, 1.0)
    looped = [
        _sgnht_step_posteriors_order(xb[i], pb[i], xib[i], gb[i], nb[i], 0.005, 1.0)
        for i in range(xb.shape[0])
    ]

    for batched_value, value_index in zip(batched, range(3)):
        expected = torch.stack([row[value_index] for row in looped], dim=0)
        torch.testing.assert_close(batched_value, expected)


def test_posteriors_library_one_step_parity_without_noise() -> None:
    posteriors = __import__("pytest").importorskip("posteriors")
    del posteriors
    from posteriors.sgmcmc import sgnht as post_sgnht

    x, p, xi, _, _ = _state(dtype=torch.float64)
    eps = 0.005
    temperature = 1.0

    def log_posterior(theta, batch):
        del batch
        return -0.5 * torch.sum(theta**2), torch.empty(0, dtype=theta.dtype)

    transform = post_sgnht.build(
        log_posterior,
        lr=eps,
        alpha=0.0,
        beta=0.0,
        sigma=1.0,
        temperature=temperature,
        momenta=p,
        xi=xi,
    )
    state = transform.init(x)
    state_next, _ = transform.update(state, None)

    ours = _sgnht_step_posteriors_order(
        x=x,
        p=p,
        xi=xi,
        grad=x,  # grad U for U(theta)=0.5*||theta||^2
        mom_noise=torch.zeros_like(p),
        eps=eps,
        temperature=temperature,
    )

    torch.testing.assert_close(ours[0], state_next.params)
    torch.testing.assert_close(ours[1], state_next.momenta)
    torch.testing.assert_close(ours[2], state_next.xi)
