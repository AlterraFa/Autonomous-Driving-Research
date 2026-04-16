import glob
import os
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
    all_npy.extend(glob.glob(os.path.join(base, rec, "**", "*.npy"), recursive=True))

samples = []
all_wp_points = []

for p in all_npy:
    try:
        data = np.load(p, allow_pickle=True).item()
        meta = data.get("metadata", {})
        gt = meta.get("gt_data", {})
        wp = gt.get("midlane_wp", None)
        if wp is None:
            continue
        wp = np.asarray(wp, dtype=np.float32)
        if wp.ndim != 2 or wp.shape[1] != 2:
            continue

        x = wp[:, 0]
        y = wp[:, 1]
        dx = np.diff(x)
        dy = np.diff(y)
        step = np.sqrt(dx * dx + dy * dy)

        rec_name = next((r for r in recordings if r in p), "unknown")
        idx_name = os.path.basename(p).replace(".npy", "")

        samples.append({
            "path": p,
            "recording": rec_name,
            "frame": idx_name,
            "x_max_abs": float(np.max(np.abs(x))),
            "y_max_abs": float(np.max(np.abs(y))),
            "xy_max_abs": float(np.max(np.abs(wp))),
            "step_max": float(np.max(step)) if len(step) > 0 else 0.0,
            "step_mean": float(np.mean(step)) if len(step) > 0 else 0.0,
            "wp": wp,
        })

        all_wp_points.append(wp)
    except Exception:
        continue

print(f"Scanned samples: {len(samples)}")
if not samples:
    raise SystemExit(0)

all_wp = np.concatenate(all_wp_points, axis=0)
abs_x = np.abs(all_wp[:, 0])
abs_y = np.abs(all_wp[:, 1])

print("\nGlobal absolute waypoint stats")
print(f"|x| p95={np.percentile(abs_x,95):.3f}, p99={np.percentile(abs_x,99):.3f}, p99.9={np.percentile(abs_x,99.9):.3f}, max={np.max(abs_x):.3f}")
print(f"|y| p95={np.percentile(abs_y,95):.3f}, p99={np.percentile(abs_y,99):.3f}, p99.9={np.percentile(abs_y,99.9):.3f}, max={np.max(abs_y):.3f}")

# Dynamic extreme thresholds from distribution
x_thr = np.percentile(abs_x, 99.9)
y_thr = np.percentile(abs_y, 99.9)
step_vals = np.array([s["step_max"] for s in samples], dtype=np.float32)
step_thr = np.percentile(step_vals, 99.9)

print("\nDynamic extreme thresholds")
print(f"x_thr (99.9% |x|): {x_thr:.3f}")
print(f"y_thr (99.9% |y|): {y_thr:.3f}")
print(f"step_thr (99.9% max step): {step_thr:.3f}")

extreme = [
    s for s in samples
    if s["x_max_abs"] > x_thr or s["y_max_abs"] > y_thr or s["step_max"] > step_thr
]

# Rank by strongest anomaly signal
extreme.sort(
    key=lambda s: max(
        s["x_max_abs"] / max(x_thr, 1e-6),
        s["y_max_abs"] / max(y_thr, 1e-6),
        s["step_max"] / max(step_thr, 1e-6),
    ),
    reverse=True,
)

print(f"\nExtreme samples found: {len(extreme)}")
print("Top 30 extremes")
print(f"{'#':>3}  {'x_max':>10}  {'y_max':>10}  {'step_max':>10}  {'recording':<40}  {'frame':>7}")
print("-" * 95)
for i, s in enumerate(extreme[:30], start=1):
    print(f"{i:>3}  {s['x_max_abs']:>10.3f}  {s['y_max_abs']:>10.3f}  {s['step_max']:>10.3f}  {s['recording']:<40}  {s['frame']:>7}")

print("\nTop 5 extreme waypoint arrays")
for s in extreme[:5]:
    print(f"\n{s['path']}")
    print(f"x_max={s['x_max_abs']:.3f}, y_max={s['y_max_abs']:.3f}, step_max={s['step_max']:.3f}")
    print(f"{'idx':>4}  {'x':>11}  {'y':>11}")
    for i, (x, y) in enumerate(s["wp"]):
        print(f"{i:>4}  {x:>11.3f}  {y:>11.3f}")

print("\nExtreme count by recording")
counts = {}
for s in extreme:
    counts[s["recording"]] = counts.get(s["recording"], 0) + 1
for rec, c in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
    print(f"{rec}: {c}")
