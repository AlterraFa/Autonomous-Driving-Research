import glob, os, random
import numpy as np

base = "../../../Autonomous_Dataset"
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

all_npy = []
for rec in recordings:
    path = os.path.join(base, rec)
    all_npy.extend(glob.glob(os.path.join(path, "**", "*.npy"), recursive=True))

print(f"Total .npy files: {len(all_npy)}")
random.seed(42)
sample = random.sample(all_npy, min(2000, len(all_npy)))

wps_raw, missing_meta, missing_key, wrong_shape, nan_files = [], 0, 0, 0, 0
road_types, wp_counts = {}, {}

for path in sample:
    try:
        data = np.load(path, allow_pickle=True).item()
        meta = data.get("metadata")
        if meta is None:
            missing_meta += 1
            continue
        gt_data = meta.get("gt_data", {})
        cond = meta.get("condition", {})
        rt = cond.get("road_type", "unknown")
        road_types[rt] = road_types.get(rt, 0) + 1
        wp = gt_data.get("pixel_wp")
        if wp is None:
            missing_key += 1
            continue
        wp = np.array(wp, dtype=np.float32)
        if wp.ndim != 2 or wp.shape[1] != 2:
            wrong_shape += 1
            continue
        if not np.isfinite(wp).all():
            nan_files += 1
            wp = wp[np.isfinite(wp).all(axis=1)]
            if wp.size == 0:
                continue
        wp_counts[wp.shape[0]] = wp_counts.get(wp.shape[0], 0) + 1
        wps_raw.append(wp)
    except Exception as e:
        pass

print(f"Loaded: {len(wps_raw)}, missing_meta: {missing_meta}, missing_key: {missing_key}, "
      f"wrong_shape: {wrong_shape}, nan_files: {nan_files}")
print(f"Road types: {road_types}")
print(f"WP count distribution: {wp_counts}")

# ── full-dataset pass (all files) ───────────────────────────────────────────
all_npy_full = []
for rec in recordings:
    all_npy_full.extend(glob.glob(os.path.join(base, rec, "metadata", "*.npy")))

wps_all = []
nan_total = 0
for path in all_npy_full:
    data = np.load(path, allow_pickle=True).item()
    wp = np.asarray(data.get("metadata", {}).get("gt_data", {}).get("pixel_wp", []), dtype=np.float32)
    if wp.ndim != 2 or wp.shape[1] != 2:
        continue
    if not np.isfinite(wp).all():
        nan_total += 1
        wp = wp[np.isfinite(wp).all(axis=1)]
    if wp.size:
        wps_all.append(wp)

if wps_raw:
    # sample-based per-waypoint-index stats (keeps variable length shapes separate)
    common_count = max(wp_counts, key=wp_counts.get)
    uniform_wps = np.stack([w for w in wps_raw if w.shape[0] == common_count])

    all_wp = np.concatenate(wps_all, axis=0)
    X, Y = all_wp[:, 0], all_wp[:, 1]

    print(f"\n=== Global stats — full dataset ({len(all_wp):,} finite points from {len(wps_all)} files) ===")
    print(f"  nan_files_skipped: {nan_total}")
    print(f"  x : mean={X.mean():8.2f}  std={X.std():7.2f}  min={X.min():9.2f}  max={X.max():9.2f}")
    print(f"  y : mean={Y.mean():8.2f}  std={Y.std():7.2f}  min={Y.min():9.2f}  max={Y.max():9.2f}")
    for q in [50, 95, 99, 99.5, 99.9]:
        px, py = np.percentile(np.abs(X), q), np.percentile(np.abs(Y), q)
        print(f"  abs p{q:5.1f}: x={px:8.2f}  y={py:8.2f}")

    # out-of-image-boundary counts
    IMG_W, IMG_H = 1280.0, 720.0
    print(f"\n=== Out-of-image-boundary checks (image={int(IMG_W)}x{int(IMG_H)}) ===")
    checks = [
        ("x < 0",       X < 0),
        ("x > 1280",    X > IMG_W),
        ("y < 0",       Y < 0),
        ("y > 720",     Y > IMG_H),      # below frame — expected for near waypoints
        ("y > 1440",    Y > 2 * IMG_H),  # 2x height — unusual
        ("x < -200",    X < -200),       # far off left edge
        ("x > 1480",    X > IMG_W + 200), # far off right edge
        ("x > 2000",    X > 2000),       # catastrophic
        ("y > 2000",    Y > 2000),       # catastrophic
        ("x > 4000",    X > 4000),
        ("y > 4000",    Y > 4000),
    ]
    for name, mask in checks:
        n = int(mask.sum())
        print(f"  {name:<12}  {n:6d} pts  ({100*n/len(X):.4f}%)")

    # Statistics centered at image center
    all_wp_c = all_wp - np.array([640.0, 360.0])
    print(f"\n=== Centered at image centre (640, 360) ===")
    print(f"  x : mean={all_wp_c[:,0].mean():8.2f}  std={all_wp_c[:,0].std():7.2f}  "
          f"min={all_wp_c[:,0].min():9.2f}  max={all_wp_c[:,0].max():9.2f}")
    print(f"  y : mean={all_wp_c[:,1].mean():8.2f}  std={all_wp_c[:,1].std():7.2f}  "
          f"min={all_wp_c[:,1].min():9.2f}  max={all_wp_c[:,1].max():9.2f}")

    print(f"\n=== Per-waypoint-index stats ({common_count} wps, {len(uniform_wps)} samples from 2 k-sample) ===")
    print(f"{'idx':>4}  {'mean_x':>8}  {'std_x':>8}  {'mean_y':>8}  {'std_y':>8}  "
          f"{'p99|x|':>8}  {'p99|y|':>8}  {'max|x|':>8}  {'max|y|':>8}")
    for i in range(common_count):
        xi = uniform_wps[:, i, 0]
        yi = uniform_wps[:, i, 1]
        print(f"{i:>4}  {xi.mean():>8.1f}  {xi.std():>8.1f}  {yi.mean():>8.1f}  {yi.std():>8.1f}  "
              f"{np.percentile(np.abs(xi),99):>8.1f}  {np.percentile(np.abs(yi),99):>8.1f}  "
              f"{np.abs(xi).max():>8.1f}  {np.abs(yi).max():>8.1f}")
