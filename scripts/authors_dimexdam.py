"""Оригинальный код DimExDAM (third_party/DimExDAM, коммит 1681693) на кернах из data/.

Модели (NCSNpp, Discriminator_large) и цикл обучения train_ddgan.train() авторские, без изменений.
Обёртка делает только четыре вещи:
  1. готовит обучающие картинки: 3 центральных среза керна в PNG (по одному на плоскость, как в статье);
  2. задаёт параметры из статьи там, где значения по умолчанию в их коде другие (см. PAPER_FLAGS);
  3. глушит Comet (в их скрипте зашит API-ключ их аккаунта) и тяжёлое логирование в конце эпохи;
  4. генерирует так же, как их test_ddgan.py, и сохраняет all_samples.pt (серые объёмы) + samples.npy.

    python scripts/authors_dimexdam.py train Variant_11 --epochs 60
    python scripts/authors_dimexdam.py sample Variant_11 --n 20
"""
import argparse
import json
import os
import sys
import textwrap
import time
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "DimExDAM"
RUNS = ROOT / "runs" / "authors"                     # их скрипт пишет в ./saved_info/... относительно cwd

# --- 3. заглушки: comet_ml (не отправляем ничего в их аккаунт) и porespy (нужен только их логированию) ---
class _Noop:
    def __getattr__(self, _):
        return lambda *a, **k: None

_comet = types.ModuleType("comet_ml"); _comet.Experiment = lambda *a, **k: _Noop()
_pt = types.ModuleType("comet_ml.integration.pytorch"); _pt.watch = lambda *a, **k: None
sys.modules.update({"comet_ml": _comet, "comet_ml.integration": types.ModuleType("comet_ml.integration"),
                    "comet_ml.integration.pytorch": _pt, "porespy": types.ModuleType("porespy")})
sys.path.insert(0, str(REPO))
import train_ddgan as tdg  # noqa: E402

for fn in ("log_comet_grid_samples", "log_preview_real_images", "log_intensity_stats_and_plot",
           "log_forward_vs_reverse_trajectory", "structural_descriptors_compare",
           "log_comet_metrics", "log_comet_gradients"):
    setattr(tdg, fn, lambda *a, **k: None)

from dimexdam.data import load_mhd, ortho_slices  # noqa: E402

# --- 2. параметры из статьи (раздел 3.3), которые отличаются от значений по умолчанию в их train_ddgan.py ---
PAPER_FLAGS = ["--num_timesteps", "8",                # в коде 2
               "--r1_gamma", "10",                    # в коде 1 (и R1 вообще выключен, см. lazy_reg_every)
               "--g_grad_clip", "3",                  # в коде 12
               "--low_pass",                          # в коде выключено
               "--low_pass_ch1", "4", "--low_pass_sigma_lvl2", "0.8", "--low_pass_p_clean_pre", "0.35",
               "--low_pass_ch2", "7", "--low_pass_sigma_lvl1", "0.4", "--low_pass_p_clean_mid", "0.55"]


def authors_parser():
    """Их argparse (build_argparser) из блока __main__ train_ddgan.py, без изменений."""
    src = (REPO / "train_ddgan.py").read_text(encoding="utf-8")
    main = textwrap.dedent(src.split("if __name__ == '__main__':", 1)[1]).split("args = build_argparser()")[0]
    ns = {}
    exec(main, ns)
    return ns["build_argparser"]()


def make_args(var, epochs, seed=0, extra=()):
    data = RUNS / "data" / var
    if not data.exists():                                 # 1. три центральных среза -> PNG (0 = поры, 255 = твёрдая)
        data.mkdir(parents=True)
        for name, s in zip("xyz", ortho_slices(load_mhd(ROOT / "data" / var / "model.mhd"))):
            Image.fromarray(((s.numpy() + 1) * 127.5).astype(np.uint8)).save(data / f"{name}.png")
    args = authors_parser().parse_args(PAPER_FLAGS + ["--exp", var, "--dataset", "hw2", "--seed", str(seed),
                                                      "--num_epoch", str(epochs - 1), *extra])
    # срез "x" перпендикулярен первой оси объёма (D), как в их extract_plane_slices_controlled
    args.data_paths = {k: [str(data / f"{k}.png")] for k in "xyz"}
    args.lazy_reg_every = 16                              # lazy R1 каждые 16 шагов (в коде нет флага, по умолчанию 0)
    # Их LR-расписание шагает раз в эпоху: разогрев 200 эпох + косинус на num_epoch (рассчитано на 1200 эпох).
    # Сохраняем ту же форму расписания для нашего числа эпох.
    args.warmup_steps = max(1, round(epochs * 200 / 1200))
    return args


def exp_dir(var):
    return RUNS / "saved_info" / "dd_gan" / "hw2" / var


def train(var, epochs, seed):
    args = make_args(var, epochs, seed)
    RUNS.mkdir(parents=True, exist_ok=True)
    os.chdir(RUNS)
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    tdg.train(args)
    stats = {"epochs": epochs, "iters_per_epoch": 1500 * 1 // (args.generated_images * args.num_slices_per_direction),
             "train_min": round((time.time() - t0) / 60, 1),
             "train_peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
             "train_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
             "settings": {k: v for k, v in vars(args).items() if k != "data_paths"}}
    (exp_dir(var) / "train_stats.json").write_text(json.dumps(stats, indent=1, default=str))
    print(var, {k: v for k, v in stats.items() if k != "settings"})


@torch.no_grad()
def sample(var, n, seed, bs=4):
    """Генерация как в их test_ddgan.py (_sample_volumes): свежий z на каждом шаге, fp32."""
    args = make_args(var, epochs=1)
    args.num_epoch = 1
    dev = torch.device("cuda")
    netG = tdg.get_models(args, dev)[0].eval()
    netG.load_state_dict(torch.load(exp_dir(var) / "content.pth", map_location=dev)["netG_dict"])  # финальная эпоха
    pos = tdg.PosteriorCoefficients(args, dev)
    torch.manual_seed(seed)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    vols = []
    for i in range(0, n, bs):
        b = min(bs, n - i)
        x = tdg.create_random_3d_noise(b, 1, args.image_size, args.image_size, args.image_size, dev)
        for t in reversed(range(args.num_timesteps)):
            tt = torch.full((b,), t, dtype=torch.int64, device=dev)
            x = tdg.sample_posterior(pos, netG(x, tt, torch.randn(b, args.nz, device=dev)), x, tt).detach()
        vols.append(x)
    torch.cuda.synchronize()
    vols = torch.cat(vols).cpu()                          # (N, 1, 64, 64, 64), серые значения в [-1, 1]
    stats = {"n_samples": n, "sample_sec_per_rock": round((time.time() - t0) / n, 3),
             "sample_peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}

    out = ROOT / "outputs" / "authors" / var
    out.mkdir(parents=True, exist_ok=True)
    torch.save(vols, out / "all_samples.pt")              # как в их test_ddgan.py
    grey = vols[:, 0].numpy()
    np.save(out / "samples.npy", (grey >= 0).astype(np.uint8))   # порог 0: 0 = поры, 1 = твёрдая
    # Порог по пористости 2D-обучающих срезов (только 2D-информация), один на все объёмы
    target = float(np.mean([(s < 0).float().mean() for s in ortho_slices(load_mhd(ROOT / "data" / var / "model.mhd"))]))
    c = float(np.quantile(grey, target))
    np.save(out / "samples_cal.npy", (grey >= c).astype(np.uint8))
    p0, p1 = (grey < 0).mean(axis=(1, 2, 3)), (grey < c).mean(axis=(1, 2, 3))
    stats.update(porosity_cut0=[round(p0.mean(), 4), round(p0.std(), 4)], target_2d=round(target, 4),
                 cal_threshold=round(c, 4), porosity_cal=[round(p1.mean(), 4), round(p1.std(), 4)],
                 frac_grey_voxels=round(float((np.abs(grey) < 0.9).mean()), 4))
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(var, stats)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "sample"])
    p.add_argument("variants", nargs="+")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    for v in a.variants:
        train(v, a.epochs, a.seed) if a.cmd == "train" else sample(v, a.n, a.seed + 1)
