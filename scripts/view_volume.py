"""Просмотр 3D-объёма: сгенерированного (.npy) или исходного керна (.mhd).

    python scripts/view_volume.py outputs/Variant_01/samples.npy              # 3D-рендер сэмпла 0
    python scripts/view_volume.py outputs/Variant_01/samples.npy -i 3 --slices  # листать срезы
    python scripts/view_volume.py data/Variant_01/model.mhd --crop 64         # кусок керна для сравнения
    python scripts/view_volume.py outputs/Variant_01/samples.npy --mhd        # экспорт для ParaView

3D-рендер сохраняется как HTML и открывается в браузере (мышью крутить, колесом зум).
"""
import argparse
import gzip
from pathlib import Path

import numpy as np

from dimexdam.data import load_mhd

p = argparse.ArgumentParser()
p.add_argument("path", type=Path)
p.add_argument("-i", "--index", type=int, default=0, help="номер объёма в .npy")
p.add_argument("--phase", type=int, default=0, help="какую фазу рисовать поверхностью (0 = поры)")
p.add_argument("--crop", type=int, default=None, help="вырезать куб crop³ из начала объёма (для больших кернов)")
p.add_argument("--slices", action="store_true", help="вместо 3D листать срезы слайдером")
p.add_argument("--mhd", action="store_true", help="только сохранить .mhd/.raw.gz для ParaView")
args = p.parse_args()

# --- загрузка: .npy содержит (N, D, H, W), .mhd — один объём ---
vol = load_mhd(args.path) if args.path.suffix == ".mhd" else np.load(args.path)[args.index]
if args.crop:
    vol = vol[: args.crop, : args.crop, : args.crop]
name = f"{args.path.stem}_{args.index}" if args.path.suffix == ".npy" else "core"
out_dir = Path("outputs") / args.path.parent.name          # всё сохраняется в outputs/<вариант>/
out_dir.mkdir(parents=True, exist_ok=True)
print(f"{name}: форма {vol.shape}, доля фазы {args.phase} = {(vol == args.phase).mean():.3f}")

if args.mhd:  # --- экспорт в формат исходных данных ---
    out = out_dir / f"{name}.mhd"
    gzip.open(out.with_suffix(".raw.gz"), "wb").write(vol.astype(np.uint8).tobytes())
    D, H, W = vol.shape
    out.write_text("ObjectType = Image\nNDims = 3\nElementType = MET_UCHAR\nCompressedData = True\n"
                   f"DimSize = {W} {H} {D}\nElementSpacing = 1 1 1\nElementDataFile = {out.stem}.raw.gz\n")
    print("сохранено:", out)

elif args.slices:  # --- листалка срезов по трём осям ---
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.5))
    ims = [a.imshow(np.take(vol, vol.shape[k] // 2, axis=k), cmap="gray", vmin=0, vmax=1) for k, a in enumerate(ax)]
    for a, t in zip(ax, ["XY", "XZ", "YZ"]): a.set_title(t); a.axis("off")
    s = Slider(fig.add_axes([0.2, 0.05, 0.6, 0.03]), "срез", 0, min(vol.shape) - 1, valinit=min(vol.shape) // 2, valstep=1)
    s.on_changed(lambda v: [im.set_data(np.take(vol, int(v), axis=k)) for k, im in enumerate(ims)] and fig.canvas.draw_idle())
    plt.show()

else:  # --- 3D: поверхность выбранной фазы (marching cubes) в plotly ---
    import plotly.graph_objects as go
    from skimage.measure import marching_cubes
    # Обкладываем объём рамкой из другой фазы, чтобы поверхность закрывалась на гранях куба
    mask = np.pad((vol == args.phase).astype(np.float32), 1)
    v, f, _, _ = marching_cubes(mask, 0.5)
    fig = go.Figure(go.Mesh3d(x=v[:, 2], y=v[:, 1], z=v[:, 0], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                              color="steelblue", flatshading=True,
                              lighting=dict(ambient=0.4, diffuse=0.8), opacity=1.0))
    fig.update_layout(title=f"{name}: фаза {args.phase}", scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=40, b=0))
    out = out_dir / f"{name}_3d.html"
    fig.write_html(out, auto_open=True)
    print("3D-рендер:", out)
