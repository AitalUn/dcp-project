"""Загрузка кернов (.mhd + .raw.gz) и нарезка 2D-срезов для обучения."""
import gzip
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def load_mhd(path) -> np.ndarray:
    """Читает MetaImage. Возвращает массив uint8 формы (Z, Y, X)."""
    path = Path(path)
    hdr = dict(l.split(" = ", 1) for l in path.read_text().splitlines() if " = " in l)
    dims = [int(d) for d in hdr["DimSize"].split()]          # X Y Z (X меняется быстрее всего)
    raw = path.parent / hdr["ElementDataFile"]
    buf = gzip.open(raw).read() if raw.suffix == ".gz" else raw.read_bytes()
    return np.frombuffer(buf, np.uint8).reshape(dims[::-1])


def ortho_slices(vol: np.ndarray, idx=None) -> list[torch.Tensor]:
    """Три ортогональных среза: срез p перпендикулярен оси p. Это единственные данные для обучения
    (как в статье: по одному изображению на плоскость). Значения переводятся из {0,1} в {-1,1}."""
    idx = idx or [s // 2 for s in vol.shape]
    sl = [vol[idx[0]], vol[:, idx[1]], vol[:, :, idx[2]]]
    return [torch.from_numpy(s.astype(np.float32) * 2 - 1) for s in sl]


def gauss_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Гауссово размытие батча (N,1,H,W): нужно для low-pass curriculum."""
    r = max(1, int(3 * sigma))
    k = torch.exp(-torch.arange(-r, r + 1, device=x.device) ** 2 / (2 * sigma**2))
    k = (k / k.sum()).view(1, 1, -1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k[..., None, :])
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k[..., :, None])


class SliceSampler:
    """Случайные кропы L×L из трёх обучающих срезов вместе с меткой плоскости."""

    def __init__(self, slices, crop: int, device):
        self.imgs = [s.to(device) for s in slices]
        self.L, self.device = crop, device

    def __call__(self, n: int, blur=(0.0, 0.0)):
        planes = torch.randint(0, 3, (n,))
        crops = []
        for p in planes.tolist():
            img = self.imgs[p]
            i, j = (np.random.randint(0, s - self.L + 1) for s in img.shape)
            crops.append(img[i:i + self.L, j:j + self.L])
        x = torch.stack(crops)[:, None]
        sigma, prob = blur                                    # curriculum: размываем долю prob кропов
        if sigma > 0:
            m = torch.rand(n, device=self.device) < prob
            if m.any():
                x[m] = gauss_blur(x[m], sigma)
        return x, planes.to(self.device)


def stratified_slices(a: torch.Tensor, b: torch.Tensor, S: int):
    """Вырезает одни и те же стратифицированные срезы из двух объёмов (x_t, x_{t+1}) по всем 3 осям.
    i_k = floor((k + rand) * L / S), где rand выбирается отдельно для каждой оси и каждого объёма."""
    B, L = a.shape[0], a.shape[-1]
    outs_a, outs_b, planes = [], [], []
    for p in range(3):
        idx = ((torch.arange(S, device=a.device) + torch.rand(B, 1, device=a.device)) * L / S).long()
        bi = torch.arange(B, device=a.device)[:, None]
        for v, out in ((a, outs_a), (b, outs_b)):
            out.append(v[:, 0].movedim(1 + p, 1)[bi, idx].reshape(B * S, 1, L, L))
        planes.append(torch.full((B * S,), p, device=a.device))
    return torch.cat(outs_a), torch.cat(outs_b), torch.cat(planes)
