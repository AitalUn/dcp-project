"""Цикл обучения DimExDAM. Логика повторяет train_ddgan.py авторов: на каждом шаге t цепочки
сначала обновляется D, потом G, после чего цепочка отсоединяется (x = x_t.detach())."""
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .data import SliceSampler, stratified_slices
from .diffusion import Diffusion
from .models import Discriminator2D, Generator3D


@dataclass
class Config:
    size: int = 64            # сторона генерируемого куба и 2D-кропа
    T: int = 8                # шагов диффузии
    nz: int = 100
    n_vol: int = 2            # 3D-объёмов на итерацию
    S: int = 8                # стратифицированных срезов на ось (итого 3*S*n_vol фейковых срезов)
    n_real: int = 32          # реальных кропов на шаг D
    lr_g: float = 2e-4        # TTUR: lr генератора больше, чем у D
    lr_d: float = 1e-5
    r1_gamma: float = 10.0    # lazy R1
    r1_every: int = 16
    g_clip: float = 3.0
    iters_per_epoch: int = 100
    amp: bool = True          # bf16 autocast


def blur_schedule(epoch: int):
    """Low-pass curriculum из статьи. Возвращает (sigma, вероятность размыть кроп)."""
    return (0.8, 0.65) if epoch < 4 else (0.4, 0.45) if epoch < 7 else (0.0, 0.0)


class Trainer:
    def __init__(self, slices, cfg: Config, device="cuda"):
        self.cfg, self.dev = cfg, device
        self.G = Generator3D(nz=cfg.nz, T=cfg.T).to(device)
        self.D = Discriminator2D(T=cfg.T, size=cfg.size).to(device)
        self.diff = Diffusion(cfg.T, device)
        self.real = SliceSampler(slices, cfg.size, device)
        self.optG = torch.optim.Adam(self.G.parameters(), cfg.lr_g, betas=(0.0, 0.9))
        self.optD = torch.optim.Adam(self.D.parameters(), cfg.lr_d, betas=(0.0, 0.9))
        self.step, self.hist = 0, {"lossD": [], "lossG": []}

    def _ac(self):
        return torch.autocast("cuda", torch.bfloat16, enabled=self.cfg.amp and self.dev == "cuda")

    def train_iter(self):
        c, G, D = self.cfg, self.G, self.D
        blur = blur_schedule(self.step // c.iters_per_epoch)
        x = torch.randn(c.n_vol, 1, c.size, c.size, c.size, device=self.dev)   # x_T
        z = torch.randn(c.n_vol, c.nz, device=self.dev)
        lD = lG = 0.0
        for t in reversed(range(c.T)):
            tv = lambda n: torch.full((n,), t, device=self.dev, dtype=torch.long)
            with self._ac():                                   # 1) один шаг генератора по 3D-цепочке
                x_t = self.diff.posterior(G(x, tv(c.n_vol), z), x, tv(c.n_vol))
            f_t, f_next, f_p = stratified_slices(x_t, x, c.S)

            # 2) шаг D: реальные пары (x_t, x_{t+1}) против срезов сгенерированного объёма
            D.requires_grad_(True)
            real, r_p = self.real(c.n_real, blur)
            r_t, r_next = self.diff.q_pair(real, tv(c.n_real))
            with self._ac():
                loss_d = F.softplus(-D(r_t, r_next, tv(c.n_real), r_p)).mean() + \
                         F.softplus(D(f_t.detach(), f_next, tv(len(f_p)), f_p)).mean()
            if self.step % c.r1_every == 0:                    # lazy R1 на реальных данных (в fp32)
                r_t = r_t.detach().requires_grad_(True)
                g, = torch.autograd.grad(D(r_t, r_next, tv(c.n_real), r_p).sum(), r_t, create_graph=True)
                loss_d = loss_d + c.r1_gamma / 2 * g.pow(2).flatten(1).sum(1).mean()
            self.optD.zero_grad(set_to_none=True); loss_d.backward(); self.optD.step()

            # 3) шаг G: non-saturating loss по тем же срезам
            D.requires_grad_(False)
            with self._ac():
                loss_g = F.softplus(-D(f_t, f_next, tv(len(f_p)), f_p)).mean()
            self.optG.zero_grad(set_to_none=True); loss_g.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), c.g_clip)
            self.optG.step()

            x = x_t.detach().float()                           # переходим к следующему шагу цепочки
            lD += loss_d.item() / c.T; lG += loss_g.item() / c.T
        self.step += 1
        self.hist["lossD"].append(lD); self.hist["lossG"].append(lG)
        return lD, lG

    def fit(self, n_iters: int, ckpt=None, save_every=200):
        pbar = tqdm(range(n_iters))
        for i in pbar:
            lD, lG = self.train_iter()
            pbar.set_postfix(D=f"{lD:.3f}", G=f"{lG:.3f}", epoch=self.step // self.cfg.iters_per_epoch)
            if ckpt and (i + 1) % save_every == 0:
                self.save(ckpt)
        if ckpt:
            self.save(ckpt)

    def save(self, path):
        torch.save({"G": self.G.state_dict(), "D": self.D.state_dict(), "optG": self.optG.state_dict(),
                    "optD": self.optD.state_dict(), "cfg": asdict(self.cfg), "step": self.step, "hist": self.hist}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.dev)
        for k in ("G", "D", "optG", "optD"):
            getattr(self, k).load_state_dict(ck[k])
        self.step, self.hist = ck["step"], ck["hist"]


def load_generator(path, device="cuda"):
    """Для инференса нужен только генератор и конфиг."""
    ck = torch.load(path, map_location=device)
    cfg = Config(**ck["cfg"])
    G = Generator3D(nz=cfg.nz, T=cfg.T).to(device).eval()
    G.load_state_dict(ck["G"])
    return G, cfg
