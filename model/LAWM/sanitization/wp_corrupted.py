import glob, os
import numpy as np

base = "./../Autonomous_Dataset/carla/LAWM"
recordings = [
    "recording_20251025_142727_best_spatial",
    "recording_20260204_010805_spatial",
    "recording_20260308_212005_spatial",
    "recording_20260317_214033_best_spatial",
    "recording_20260317_233603_spatial",
    "recording_20260318_083409_best_spatial",
    "recording_20260323_200940_best_spatial",
    "recording_20260329_233141_best_spatial",
    "recording_20260323_204100_best_spatial",
    "recording_20260323_210357_best_spatial",
    "recording_20260329_164940_best_spatial",
    "recording_20260410_152712_best_spatial",
    "recording_20260410_154404_best_spatial",
    "recording_20260410_160255_best_spatial",
]

# Thresholds beyond which a sample is considered corrupted
X_THRESH = 20.0  # longitudinal, meters
Y_THRESH = 8.0   # lateral, meters

all_npy = []
for rec in recordings:
    path = os.path.join(base, rec)
    all_npy.extend(glob.glob(os.path.join(path, "**", "*.npy"), recursive=True))

print(f"Scanning {len(all_npy)} .npy files ...\n")

corrupted = []

for path in all_npy:
    try:
        data = np.load(path, allow_pickle=True).item()
        meta = data.get("metadata")
        if meta is None:
            continue
        gt_data = meta.get("gt_data", {})
        wp = gt_data.get("midlane_wp")
        if wp is None:
            continue
        wp = np.array(wp, dtype=np.float32)
        if wp.ndim != 2 or wp.shape[1] != 2:
            continue
        x_max = np.abs(wp[:, 0]).max()
        y_max = np.abs(wp[:, 1]).max()
        if x_max > X_THRESH or y_max > Y_THRESH:
            corrupted.append({
                "path": path,
                "x_max": float(x_max),
                "y_max": float(y_max),
                "wp": wp,
            })
    except Exception as e:
        print(f"  ERROR reading {path}: {e}")

print(f"Found {len(corrupted)} corrupted samples out of {len(all_npy)} total "
      f"({100*len(corrupted)/len(all_npy):.2f}%)\n")

# Sort by worst offender
corrupted.sort(key=lambda s: s["x_max"] + s["y_max"], reverse=True)

print(f"{'#':>4}  {'x_max(m)':>10}  {'y_max(m)':>10}  path")
print("-" * 100)
for i, s in enumerate(corrupted):
    print(f"{i+1:>4}  {s['x_max']:>10.2f}  {s['y_max']:>10.2f}  {s['path']}")

if corrupted:
    print(f"\n--- Waypoint arrays of top 5 worst samples ---")
    for s in corrupted[:5]:
        print(f"\n{s['path']}")
        print(f"  shape: {s['wp'].shape}")
        print(f"  {'idx':>4}  {'x(m)':>10}  {'y(m)':>10}")
        for i, (x, y) in enumerate(s['wp']):
            flag = " <-- OUTLIER" if abs(x) > X_THRESH or abs(y) > Y_THRESH else ""
            print(f"  {i:>4}  {x:>10.3f}  {y:>10.3f}{flag}")

# Per-recording breakdown
print(f"\n--- Corrupted sample count per recording ---")
per_rec = {}
for s in corrupted:
    rec = s["path"].split(os.sep)
    # find which recording it belongs to
    for r in recordings:
        if r in s["path"]:
            per_rec[r] = per_rec.get(r, 0) + 1
            break
for r in recordings:
    count = per_rec.get(r, 0)
    total = len(glob.glob(os.path.join(base, r, "**", "*.npy"), recursive=True))
    if count > 0:
        print(f"  {r}: {count}/{total} ({100*count/total:.1f}%)")
