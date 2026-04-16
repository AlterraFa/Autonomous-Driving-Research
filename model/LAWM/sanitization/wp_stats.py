import glob, os, random
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

all_npy = []
for rec in recordings:
    path = os.path.join(base, rec)
    all_npy.extend(glob.glob(os.path.join(path, "**", "*.npy"), recursive=True))

print(f"Total .npy files: {len(all_npy)}")
random.seed(42)
sample = random.sample(all_npy, min(2000, len(all_npy)))

wps, missing_meta, missing_key, wrong_shape = [], 0, 0, 0
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
        wp = gt_data.get("midlane_wp")
        if wp is None:
            missing_key += 1
            continue
        wp = np.array(wp, dtype=np.float32)
        if wp.ndim != 2 or wp.shape[1] != 2:
            wrong_shape += 1
            continue
        wp_counts[wp.shape[0]] = wp_counts.get(wp.shape[0], 0) + 1
        wps.append(wp)
    except Exception:
        pass

print(f"Loaded: {len(wps)}, missing_meta: {missing_meta}, missing_key: {missing_key}, wrong_shape: {wrong_shape}")
print(f"Road types: {road_types}")
print(f"WP count distribution: {wp_counts}")

if wps:
    all_wp = np.concatenate(wps, axis=0)
    print("\n=== Global stats ===")
    print(f"x: mean={all_wp[:,0].mean():.3f}  std={all_wp[:,0].std():.3f}  min={all_wp[:,0].min():.3f}  max={all_wp[:,0].max():.3f}")
    print(f"y: mean={all_wp[:,1].mean():.3f}  std={all_wp[:,1].std():.3f}  min={all_wp[:,1].min():.3f}  max={all_wp[:,1].max():.3f}")
    p95 = np.percentile(np.abs(all_wp), 95, axis=0)
    p99 = np.percentile(np.abs(all_wp), 99, axis=0)
    print(f"95th |wp|: x={p95[0]:.3f}  y={p95[1]:.3f}")
    print(f"99th |wp|: x={p99[0]:.3f}  y={p99[1]:.3f}")
    print(f"abs-max:   x={np.abs(all_wp[:,0]).max():.3f}  y={np.abs(all_wp[:,1]).max():.3f}")

    common_count = max(wp_counts, key=wp_counts.get)
    uniform_wps = np.stack([w for w in wps if w.shape[0] == common_count])
    print(f"\n=== Per-waypoint-index stats ({common_count} wps, {len(uniform_wps)} samples) ===")
    print(f"{'idx':>4}  {'mean_x':>8}  {'std_x':>8}  {'mean_y':>8}  {'std_y':>8}  {'max|x|':>8}  {'max|y|':>8}")
    for i in range(common_count):
        print(f"{i:>4}  {uniform_wps[:,i,0].mean():>8.3f}  {uniform_wps[:,i,0].std():>8.3f}  {uniform_wps[:,i,1].mean():>8.3f}  {uniform_wps[:,i,1].std():>8.3f}  {np.abs(uniform_wps[:,i,0]).max():>8.3f}  {np.abs(uniform_wps[:,i,1]).max():>8.3f}")
