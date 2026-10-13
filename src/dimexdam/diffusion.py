"""Короткая диффузионная цепочка (T=8, косинусное расписание), как в DDGAN.

Индексация: x_0 — чистые данные, x_T — шум. На шаге t (t = T-1 … 0) генератор получает x_{t+1},
предсказывает x̂_0, после чего x_t сэмплируется из апостериорного q(x_t | x_{t+1}, x̂_0)."""
import math

import torch


def _bc(v, t, like):  # коэффициент для каждого элемента батча, расширенный до формы like
    return v[t].view(-1, *[1] * (like.dim() - 1)).to(like.dtype)


class Diffusion:
    def __init__(self, T: int = 8, device="cpu", s: float = 0.008):
        self.T = T
        f = torch.cos((torch.arange(T + 1) / T + s) / (1 + s) * math.pi / 2) ** 2
        abar = (f / f[0]).clamp(1e-5, 1.0)                    # ᾱ_0 = 1 … ᾱ_T ≈ 0
        alpha = torch.cat([torch.ones(1), abar[1:] / abar[:-1]])
        beta = 1 - alpha
        # коэффициенты апостериорного q(x_t | x_{t+1}, x_0) (стандартная формула DDPM)
        a1, ab1, ab0 = alpha[1:], abar[1:], abar[:-1]
        self.c0 = beta[1:] * ab0.sqrt() / (1 - ab1)           # при x_0
        self.c1 = (1 - ab0) * a1.sqrt() / (1 - ab1)           # при x_{t+1}
        self.std = (beta[1:] * (1 - ab0) / (1 - ab1)).sqrt()
        self.abar, self.alpha = abar, alpha
        for k in ("c0", "c1", "std", "abar", "alpha"):
            setattr(self, k, getattr(self, k).to(device))

    def q_pair(self, x0, t):
        """Реальные данные: зашумляем x0 и получаем пару (x_t, x_{t+1})."""
        xt = _bc(self.abar.sqrt(), t, x0) * x0 + _bc((1 - self.abar).sqrt(), t, x0) * torch.randn_like(x0)
        a = _bc(self.alpha, t + 1, x0)
        return xt, a.sqrt() * xt + (1 - a).sqrt() * torch.randn_like(x0)

    def posterior(self, x0, x_next, t):
        """x_t ~ q(x_t | x_{t+1}, x̂_0). На последнем шаге (t=0) шум не добавляется."""
        mean = _bc(self.c0, t, x0) * x0 + _bc(self.c1, t, x0) * x_next
        return mean + _bc(self.std, t, x0) * (t > 0).float().view(-1, *[1] * (x0.dim() - 1)) * torch.randn_like(x0)

    @torch.no_grad()
    def sample(self, G, n: int, size: int, nz: int, device, z=None):
        """Инференс: x_T ~ N(0, I), затем T шагов генератора. z фиксирован на всю траекторию."""
        x = torch.randn(n, 1, size, size, size, device=device)
        z = torch.randn(n, nz, device=device) if z is None else z
        for t in reversed(range(self.T)):
            tt = torch.full((n,), t, device=device, dtype=torch.long)
            x = self.posterior(G(x, tt, z), x, tt)
        return x
