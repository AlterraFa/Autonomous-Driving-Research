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
            dx = np.diff(wp[:, 0])
            neg_mask = dx < 0
            n_neg = int(neg_mask.sum())
            if n_neg > 0:
                rows.append({
                    'path': p,
                    'recording': r,
                    'frame': os.path.basename(p).replace('.npy', ''),
                    'n_neg': n_neg,
                    'min_dx': float(dx.min()),
                    'dx': dx.tolist(),
                    'wp': wp,
                })
        except Exception:
            pass

total_npy = sum(
    len(glob.glob(os.path.join(base, r, '**', '*.npy'), recursive=True))
    for r in recs
)

print(f'Samples with any dx < 0: {len(rows)} / {total_npy} ({100*len(rows)/total_npy:.2f}%)')

# Sort by most negative dx first
rows.sort(key=lambda s: s['min_dx'])

print(f"\n{'#':>4}  {'n_neg':>5}  {'min_dx':>9}  recording + frame")
print('-' * 90)
for i, s in enumerate(rows, 1):
    print(f"{i:>4}  {s['n_neg']:>5}  {s['min_dx']:>9.3f}  {s['recording']}/{s['frame']}")

print(f"\nTop 10 worst dx arrays:")
for s in rows[:10]:
    print(f"\n{s['recording']}/{s['frame']}.npy  n_neg={s['n_neg']}  min_dx={s['min_dx']:.3f}")
    print(f"  {'wp_idx':>6}  {'x':>10}  {'y':>10}  {'dx':>10}")
    wp = s['wp']
    dxs = [None] + [wp[i+1,0]-wp[i,0] for i in range(len(wp)-1)]
    for i, (x, y) in enumerate(wp):
        dx_str = f'{dxs[i]:>10.3f}' if dxs[i] is not None else f'{"---":>10}'
        flag = ' <' if dxs[i] is not None and dxs[i] < 0 else ''
        print(f"  {i:>6}  {x:>10.3f}  {y:>10.3f}  {dx_str}{flag}")

# Histogram of n_neg
from collections import Counter
cnt = Counter(s['n_neg'] for s in rows)
print('\nHistogram of how many negative dx steps per sample:')
for k in sorted(cnt):
    print(f'  n_neg={k}: {cnt[k]} samples')

# Per-recording breakdown
from collections import defaultdict
per_rec = defaultdict(int)
for s in rows:
    per_rec[s['recording']] += 1
print('\nPer-recording samples with dx<0:')
for r, c in sorted(per_rec.items(), key=lambda kv: kv[1], reverse=True):
    total = len(glob.glob(os.path.join(base, r, '**', '*.npy'), recursive=True))
    print(f'  {r}: {c}/{total} ({100*c/total:.1f}%)')
