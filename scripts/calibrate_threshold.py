"""Порог бинаризации по пористости 2D-обучающих срезов (используется только 2D-информация).

1. Повторяет генерацию из generate_samples.py (тот же seed) и сохраняет «серые» объёмы.
2. Проверяет, что порог 0 даёт ровно тот же samples.npy, что был отправлен.
3. Подбирает один общий на вариант порог c: доля вокселей (grey < c) по всем сэмплам
   равна пористости трёх обучающих срезов. Порог один на все 20 объёмов, поэтому
   разброс между ними не подгоняется искусственно.

    python scripts/calibrate_threshold.py Variant_11 Variant_02
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from dimexdam.data import load_mhd, ortho_slices
from dimexdam.diffusion import Diffusion
from dimexdam.train import load_generator

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("variants", nargs="+")
p.add_argument("--n", type=int, default=20)
p.add_argument("--size", type=int, default=64)
p.add_argument("--pad", type=int, default=0,
               help="генерировать (size+2*pad)³ и вырезать центр size³: убирает краевой артефакт (pad кратен 4)")
args = p.parse_args()
sfx = f"_crop{args.pad}" if args.pad else ""           # файлы с обрезкой не перезаписывают основные

for var in args.variants:
    out = ROOT / "outputs" / var
    out.mkdir(parents=True, exist_ok=True)
    G, cfg = load_generator(ROOT / "checkpoints" / f"{var}.pt")
    diff = Diffusion(cfg.T, "cuda")
    torch.manual_seed(1)                                   # тот же seed, что в generate_samples.py
    L, q = args.size + 2 * args.pad, slice(args.pad, args.pad + args.size)
    logit = torch.cat([diff.sample(G, 4, L, cfg.nz, "cuda", logits=True) for _ in range(-(-args.n // 4))])
    logit = logit[: args.n, 0, q, q, q].float().cpu().numpy()   # (N, 64, 64, 64), выход до tanh
    grey = np.tanh(logit)                                  # «серый» объём в [-1, 1], 99% вокселей насыщены
    np.save(out / f"samples_grey{sfx}.npy", grey.astype(np.float16))
    np.save(out / f"samples_logit{sfx}.npy", logit.astype(np.float16))

    # отправленный ранее файл: samples.npy или переименованный var_XX.npy (может лежать в другой папке)
    ref = [out / "samples.npy"] + sorted((ROOT / "outputs").rglob(f"var_{var[-2:]}.npy"))
    ref = next(f for f in ref if f.exists())
    old = np.load(ref)
    same = np.array_equal((grey > 0).astype(np.uint8), old)   # при --pad объёмы другие, будет False
    if args.pad:
        np.save(out / f"samples{sfx}.npy", (grey > 0).astype(np.uint8))

    # Целевая пористость: пиксели фазы 0 в трёх обучающих срезах (в них фаза 0 закодирована как -1)
    target = float(np.mean([(s < 0).float().mean() for s in ortho_slices(load_mhd(ROOT / "data" / var / "model.mhd"))]))
    # Порог по логиту: tanh монотонен, так что это тот же «серый» порог, но без потолка насыщения
    c = float(np.quantile(logit, target))                  # поры = logit < c
    cal = (logit >= c).astype(np.uint8)                    # 0 = поры, 1 = твёрдая фаза
    np.save(out / f"samples_cal{sfx}.npy", cal)

    base = (grey > 0).astype(np.uint8)
    por0, por1 = (base == 0).mean(axis=(1, 2, 3)), (cal == 0).mean(axis=(1, 2, 3))
    faces = lambda a: np.mean([np.take(a == 0, i, axis=k).mean() for k in (1, 2, 3) for i in (0, -1)])
    print(f"{var}: совпадает с {ref.relative_to(ROOT)} при пороге 0: {same} | цель (2D-срезы) {target:.3f} | порог по логиту {c:+.2f} (tanh = {np.tanh(c):+.4f})\n"
          f"   пористость при пороге 0: {por0.mean():.3f} ± {por0.std():.3f} -> после калибровки: {por1.mean():.3f} ± {por1.std():.3f}\n"
          f"   пористость на гранях: порог 0 {faces(base):.3f}, калибровка {faces(cal):.3f}"
          f" | доля «серых» вокселей (|grey| < 0.9): {(np.abs(grey) < 0.9).mean():.4f}")
