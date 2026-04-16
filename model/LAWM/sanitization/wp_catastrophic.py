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
            step = np.sqrt(np.sum(np.diff(wp, axis=0) ** 2, axis=1))
            rows.append((
                p,
                float(np.abs(wp[:, 0]).max()),
                float(np.abs(wp[:, 1]).max()),
                float(step.max() if len(step) else 0.0),
            ))
        except Exception:
            pass

cat = [x for x in rows if x[1] > 50 or x[2] > 50 or x[3] > 20]
cat.sort(key=lambda t: max(t[1], t[2], t[3]), reverse=True)

print(f'Catastrophic extremes: {len(cat)}')
for i, (p, xm, ym, sm) in enumerate(cat, 1):
    print(f'{i:>2}. x_max={xm:8.3f} y_max={ym:8.3f} step_max={sm:8.3f}  {p}')
