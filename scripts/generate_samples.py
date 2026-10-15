"""Обучает модель на вариант (если чекпойнта нет), генерирует сэмплы и сохраняет картинку со срезами.

    python scripts/generate_samples.py Variant_01 Variant_04 --n 8 --iters 3000
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dimexdam.data import load_mhd, ortho_slices
from dimexdam.diffusion import Diffusion
from dimexdam.metrics import phase_fraction, random_subvolumes
from dimexdam.train import Config, Trainer, load_generator

ROOT = Path(__file__).resolve().parents[1]


def gpu_mem(prefix):
    """Пиковая память GPU: allocated — тензоры PyTorch, reserved — примерно то, что видно в nvidia-smi."""
    return {f"{prefix}_peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            f"{prefix}_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2)}


p = argparse.ArgumentParser()
p.add_argument("variants", nargs="+")
p.add_argument("--n", type=int, default=8, help="сколько объёмов сгенерировать")
p.add_argument("--size", type=int, default=64)
p.add_argument("--iters", type=int, default=3000)
args = p.parse_args()

for var in args.variants:
    vol = load_mhd(ROOT / "data" / var / "model.mhd")
    ckpt = ROOT / "checkpoints" / f"{var}.pt"
    ckpt.parent.mkdir(exist_ok=True)
    out = ROOT / "outputs" / var; out.mkdir(parents=True, exist_ok=True)
    stats_path = out / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    if not ckpt.exists():                                  # обучаем на 3 центральных срезах
        torch.manual_seed(0); np.random.seed(0)
        torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        Trainer(ortho_slices(vol), Config()).fit(args.iters, ckpt=ckpt)
        stats.update(train_iters=args.iters, train_min=round((time.time() - t0) / 60, 1), **gpu_mem("train"))

    G, cfg = load_generator(ckpt)
    diff = Diffusion(cfg.T, "cuda")
    torch.manual_seed(1)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    gen = torch.cat([diff.sample(G, 4, args.size, cfg.nz, "cuda") for _ in range(-(-args.n // 4))])[: args.n]
    torch.cuda.synchronize()
    stats.update(n_samples=args.n, size=args.size, sample_sec_per_rock=round((time.time() - t0) / args.n, 3),
                 **gpu_mem("sample"))
    gen = (gen[:, 0].cpu().numpy() > 0).astype(np.uint8)    # 0 = поры, 1 = твёрдая фаза
    np.save(out / "samples.npy", gen)
    stats_path.write_text(json.dumps(stats, indent=1))
    print(var, stats)

    # Картинка: первая строка — куб из керна, дальше первые 8 сэмплов. Колонки — центральные срезы XY/XZ/YZ.
    rows = [("керн", random_subvolumes(vol, 1, args.size)[0])] + [(f"сэмпл {i}", g) for i, g in enumerate(gen[:8])]
    fig, ax = plt.subplots(len(rows), 3, figsize=(6, 2 * len(rows)))
    for r, (name, v) in enumerate(rows):
        for a in range(3):
            ax[r, a].imshow(np.take(v, v.shape[a] // 2, axis=a), cmap="gray", vmin=0, vmax=1)
            ax[r, a].set_xticks([]); ax[r, a].set_yticks([])
        ax[r, 0].set_ylabel(f"{name}\nφ0={phase_fraction(v):.3f}", fontsize=8)
    for a, n in enumerate(["XY", "XZ", "YZ"]): ax[0, a].set_title(n)
    fig.suptitle(f"{var}: доля фазы 0 в керне {phase_fraction(vol):.3f}")
    fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(out / "samples.png", dpi=110); plt.close(fig)
    print(var, "готово:", out / "samples.png")
