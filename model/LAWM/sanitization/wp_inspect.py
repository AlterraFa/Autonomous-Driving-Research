import numpy as np

base = "./../Autonomous_Dataset/carla/LAWM"

# Spread-throughout samples (flagged mostly by y > 8m, not catastrophic x)
samples = [
    ("recording_20251025_142727_best_spatial", "000000"),
    ("recording_20251025_142727_best_spatial", "000097"),
    ("recording_20251025_142727_best_spatial", "000276"),
    ("recording_20260329_233141_best_spatial", "000002"),
    ("recording_20260318_083409_best_spatial", "000078"),
    ("recording_20260317_233603_spatial",      "000022"),
    ("recording_20260317_233603_spatial",      "000386"),
    ("recording_20260317_233603_spatial",      "000387"),
]

for rec, idx in samples:
    path = f"{base}/{rec}/metadata/{idx}.npy"
    data = np.load(path, allow_pickle=True).item()
    meta = data.get("metadata", {})
    wp = np.array(meta["gt_data"]["midlane_wp"], dtype=np.float32)
    cond = meta.get("condition", {})
    rt = cond.get("road_type", "?")
    x_max = np.abs(wp[:, 0]).max()
    y_max = np.abs(wp[:, 1]).max()
    print(f"\n{rec}  [{idx}]  road_type={rt}  x_max={x_max:.2f}m  y_max={y_max:.2f}m")
    print(f"  {'idx':>4}  {'x(m)':>9}  {'y(m)':>9}  {'delta_x':>9}  note")
    prev_x = None
    for i, (x, y) in enumerate(wp):
        dx = x - prev_x if prev_x is not None else 0.0
        flag = " <-- OVER THRESH" if abs(x) > 20 or abs(y) > 8 else ""
        print(f"  {i:>4}  {x:>9.3f}  {y:>9.3f}  {dx:>9.3f}{flag}")
        prev_x = x
