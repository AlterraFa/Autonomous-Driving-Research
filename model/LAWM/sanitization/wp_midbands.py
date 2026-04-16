import glob
import os
import numpy as np

base = './../Autonomous_Dataset/carla/LAWM'
recs = [
    'recording_20251025_142727_best_spatial',
    'recording_20260204_010805_spatial',
    'recording_20260308_212005_spatial',
    'recording_20260317_214033_best_spatial',
    'recording_20260317_233603_spatial',
    'recording_20260318_083409_best_spatial',
    'recording_20260323_200940_best_spatial',
    'recording_20260329_233141_best_spatial',
    'recording_20260323_204100_best_spatial',
    'recording_20260323_210357_best_spatial',
    'recording_20260329_164940_best_spatial',
    'recording_20260410_152712_best_spatial',
    'recording_20260410_154404_best_spatial',
    'recording_20260410_160255_best_spatial',
]

rows = []
for r in recs:
    for p in glob.glob(os.path.join(base, r, '**', '*.npy'), recursive=True):
        try:
            d = np.load(p, allow_pickle=True).item()
            wp = np.asarray(d.get('metadata', {}).get('gt_data', {}).get('midlane_wp'), dtype=np.float32)
            if wp.ndim != 2 or wp.shape[1] != 2:
                continue
            x_max = float(np.abs(wp[:, 0]).max())
            y_max = float(np.abs(wp[:, 1]).max())
            rows.append((p, x_max, y_max))
        except Exception:
            pass

N = len(rows)
print(f'Total valid samples: {N}')

x = np.array([r[1] for r in rows], dtype=np.float64)
y = np.array([r[2] for r in rows], dtype=np.float64)

print('\nX-band counts (by per-sample |x|_max):')
bands = [(0, 15), (15, 20), (20, 30), (30, 50), (50, 100), (100, 500), (500, 2000), (2000, 10000)]
for lo, hi in bands:
    c = int(((x > lo) & (x <= hi)).sum())
    print(f'  ({lo:>4},{hi:>5}] : {c:>4} ({100.0*c/N:6.3f}%)')

print('\nY-band counts (by per-sample |y|_max):')
y_bands = [(0, 8), (8, 10), (10, 20), (20, 50), (50, 100), (100, 500), (500, 2000)]
for lo, hi in y_bands:
    c = int(((y > lo) & (y <= hi)).sum())
    print(f'  ({lo:>4},{hi:>5}] : {c:>4} ({100.0*c/N:6.3f}%)')

mid_x = [r for r in rows if 30 < r[1] <= 100]
mid_y = [r for r in rows if 20 < r[2] <= 100]
near_50 = [r for r in rows if 45 <= r[1] <= 55 or 45 <= r[2] <= 55]

print(f"\nSamples with 30m < |x|_max <= 100m: {len(mid_x)}")
for p, x_max, y_max in sorted(mid_x, key=lambda t: t[1], reverse=True):
    print(f'  x={x_max:8.3f} y={y_max:8.3f}  {p}')

print(f"\nSamples with 20m < |y|_max <= 100m: {len(mid_y)}")
for p, x_max, y_max in sorted(mid_y, key=lambda t: t[2], reverse=True):
    print(f'  x={x_max:8.3f} y={y_max:8.3f}  {p}')

print(f"\nSamples near 50m (45-55 on either axis): {len(near_50)}")
for p, x_max, y_max in sorted(near_50, key=lambda t: max(t[1], t[2]), reverse=True):
    print(f'  x={x_max:8.3f} y={y_max:8.3f}  {p}')
