"""3D-генератор (U-Net, условие на t и z) и 2D-дискриминатор (один на все плоскости, spectral norm).

Авторы используют тяжёлый NCSN++. Здесь облегчённая версия той же идеи: свёрточный U-Net,
который через AdaGN-модуляцию получает эмбеддинг шага t и латентного вектора z."""
import torch
import torch.nn as nn
import torch.nn.functional as F

SN = nn.utils.spectral_norm


class ResBlock3D(nn.Module):
    def __init__(self, cin, cout, emb):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv3d(cin, cout, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv3d(cout, cout, 3, padding=1)
        self.mod = nn.Linear(emb, 2 * cout)                    # AdaGN: scale и shift вычисляются из (t, z)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.mod(e)[:, :, None, None, None].chunk(2, 1)
        h = self.c2(F.silu(self.n2(h) * (1 + scale) + shift))
        return h + self.skip(x)


class Generator3D(nn.Module):
    """G(x_{t+1}, t, z) -> x̂_0 в диапазоне [-1, 1]. Полностью свёрточный, поэтому на инференсе
    можно генерировать объёмы любого размера, кратного 2^(len(ch)-1)."""

    def __init__(self, ch=(32, 64, 128, 128), nz=100, emb=128, T=8):
        super().__init__()
        self.t_emb = nn.Embedding(T, emb)
        self.z_mlp = nn.Sequential(nn.Linear(nz, emb), nn.SiLU(), nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.inp = nn.Conv3d(1, ch[0], 3, padding=1)
        self.enc = nn.ModuleList(ResBlock3D(ci, co, emb) for ci, co in zip((ch[0],) + ch[:-1], ch))
        self.mid = ResBlock3D(ch[-1], ch[-1], emb)
        self.dec = nn.ModuleList(ResBlock3D(ci + co, co, emb) for ci, co in zip(ch[:0:-1], ch[-2::-1]))
        self.out = nn.Sequential(nn.GroupNorm(8, ch[0]), nn.SiLU(), nn.Conv3d(ch[0], 1, 3, padding=1))

    def forward(self, x, t, z, logits=False):
        e = self.t_emb(t) + self.z_mlp(z)
        h, skips = self.inp(x), []
        for i, blk in enumerate(self.enc):                     # энкодер: 64 -> 32 -> 16 -> 8
            h = blk(F.avg_pool3d(h, 2) if i else h, e)
            skips.append(h)
        h = self.mid(h, e)
        for blk, s in zip(self.dec, reversed(skips[:-1])):     # декодер со skip-соединениями
            h = blk(torch.cat([F.interpolate(h, scale_factor=2), s], 1), e)
        out = self.out(h)
        return out if logits else torch.tanh(out)              # logits=True: выход до tanh (не насыщается)


class Discriminator2D(nn.Module):
    """D(x_t, x_{t+1}, t, plane): пара 2D-срезов, эмбеддинг шага и эмбеддинг плоскости.
    Spectral norm стоит на всех слоях. На выходе tanh(logit / τ), τ = 0.4 (раздел 3.3.1)."""

    def __init__(self, ch=64, emb=128, T=8, size=64, tau=0.4):
        super().__init__()
        self.tau = tau
        self.t_emb, self.p_emb = nn.Embedding(T, emb), nn.Embedding(3, emb)
        chs = [ch, ch * 2, ch * 4, ch * 8, ch * 8][: 1 + (size // 4).bit_length() - 1]  # downsample до 4×4
        self.convs = nn.ModuleList([SN(nn.Conv2d(2, chs[0], 3, 1, 1))] +
                                   [SN(nn.Conv2d(a, b, 4, 2, 1)) for a, b in zip(chs, chs[1:])])
        self.cond = nn.ModuleList(SN(nn.Linear(2 * emb, c)) for c in chs)  # условие добавляется в каждый слой
        self.head = SN(nn.Linear(chs[-1] * 16, 1))

    def forward(self, x_t, x_next, t, plane, return_raw=False):
        e = torch.cat([self.t_emb(t), self.p_emb(plane)], 1)
        h = torch.cat([x_t, x_next], 1)
        for conv, c in zip(self.convs, self.cond):
            h = F.leaky_relu(conv(h) + c(e)[:, :, None, None], 0.2)
        raw = self.head(h.flatten(1)).squeeze(1).clamp(-10, 10)   # logit clipping
        return raw if return_raw else torch.tanh(raw / self.tau)
