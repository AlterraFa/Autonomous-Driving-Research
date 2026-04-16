import glob, os
from pathlib import Path

import cv2
import numpy as np

base = "../Autonomous_Dataset/carla/LAWM1"
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

# pixel_wp is in raw image pixel coordinates for a 1280x720 image.
# y > 720 is EXPECTED (near waypoints projected below the image bottom frame).
# Tiered anomaly thresholds:
#   off-image: x outside [0, 1280]  (horizontal)  – moderate
#   extreme:   x < -200 or x > 1480, or y > 1440 (2× height)
#   catastrophic: x < -400 or x > 2000 or y > 2000
#   huge:      |x| > 4000 or |y| > 4000
IMG_W, IMG_H = 1280.0, 720.0
X_LO_OFF, X_HI_OFF = 0.0,        IMG_W           # off-image x band
X_LO_EXT, X_HI_EXT = -200.0,     IMG_W + 200     # extended x band
Y_HI_EXT = 2 * IMG_H                              # 2× height — unusual
X_LO_CAT, X_HI_CAT = -400.0,     2000.0          # catastrophic x
Y_HI_CAT = 2000.0                                 # catastrophic y
HUGE_ABS = 4000.0


def _clip_line_to_frame(p0, p1, w, h):
    """Clip a segment to image bounds [0, w-1] x [0, h-1] using Liang-Barsky."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx, dy = x1 - x0, y1 - y0

    xmin, xmax = 0.0, float(w - 1)
    ymin, ymax = 0.0, float(h - 1)

    p = [-dx, dx, -dy, dy]
    q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]

    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return None
            continue
        t = qi / pi
        if pi < 0:
            if t > u2:
                return None
            if t > u1:
                u1 = t
        else:
            if t < u1:
                return None
            if t < u2:
                u2 = t

    cx0, cy0 = x0 + u1 * dx, y0 + u1 * dy
    cx1, cy1 = x0 + u2 * dx, y0 + u2 * dy
    return (int(round(cx0)), int(round(cy0))), (int(round(cx1)), int(round(cy1)))


def _draw_pixel_waypoints(frame, pixel_wp,
                          line_color=(0, 255, 0),
                          edge_color=(0, 200, 255),
                          thickness=2):
    """Draw projected waypoints and keep off-screen segments visible via frame-edge clipping."""
    if frame is None or pixel_wp is None:
        return frame

    # Sensor buffers can be read-only; OpenCV drawing requires writable memory.
    if not frame.flags.writeable:
        frame = frame.copy()

    pts = np.asarray(pixel_wp, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 1:
        return frame

    h, w = frame.shape[:2]
    xy = pts[:, :2]
    valid = np.isfinite(xy).all(axis=1)

    # Draw polyline segments with robust frame clipping.
    for i in range(len(xy) - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        seg = _clip_line_to_frame(xy[i], xy[i + 1], w, h)
        if seg is None:
            continue
        cv2.line(frame, seg[0], seg[1], line_color, thickness, cv2.LINE_AA)

    # Draw point markers; clamp off-screen points to nearest edge.
    for i in range(len(xy)):
        if not valid[i]:
            continue
        x, y = xy[i]
        in_frame = (0 <= x < w) and (0 <= y < h)
        cx = int(round(x if in_frame else np.clip(x, 0, w - 1)))
        cy = int(round(y if in_frame else np.clip(y, 0, h - 1)))
        color = line_color if in_frame else edge_color
        radius = 3 if in_frame else 4
        cv2.circle(frame, (cx, cy), radius, color, -1, cv2.LINE_AA)

    return frame


def _resolve_first_person_image_path(meta_npy_path):
    """Resolve the image path from a metadata .npy file (prefers FIRST_PERSON)."""
    try:
        data = np.load(meta_npy_path, allow_pickle=True).item()
    except Exception:
        return None

    img_file = data.get("img_file", None)
    if img_file is None:
        return None

    rec_dir = os.path.dirname(os.path.dirname(meta_npy_path))
    rel = None

    if isinstance(img_file, dict):
        rel = img_file.get("FIRST_PERSON", None)
        if rel is None and len(img_file) > 0:
            rel = next(iter(img_file.values()))
    elif isinstance(img_file, str):
        rel = img_file

    if not isinstance(rel, str):
        return None

    candidate = os.path.join(rec_dir, rel)
    return candidate if os.path.exists(candidate) else None


def _save_detected_overlay_images(output_root, groups):
    """Render and save overlay images for all detected samples."""
    saved = 0
    missing_img = 0
    read_fail = 0

    # Keep one image per metadata path, with highest-priority tier first.
    selected = {}
    for tier_name, items in groups:
        for s in items:
            if s["path"] not in selected:
                selected[s["path"]] = (tier_name, s)

    for meta_path, (tier_name, s) in selected.items():
        img_path = _resolve_first_person_image_path(meta_path)
        if img_path is None:
            missing_img += 1
            continue

        frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if frame is None:
            read_fail += 1
            continue

        overlay = _draw_pixel_waypoints(frame, s["wp"])

        rec = next((r for r in recordings if r in meta_path), "unknown_recording")
        frame_name = os.path.basename(meta_path).replace(".npy", "")
        out_dir = output_root / tier_name / rec
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{frame_name}.jpg"
        cv2.imwrite(str(out_path), overlay)
        saved += 1

    print("\n=== Overlay image export ===")
    print(f"  output_dir : {output_root}")
    print(f"  saved      : {saved}")
    print(f"  missing_img: {missing_img}")
    print(f"  read_fail  : {read_fail}")

all_npy = []
for rec in recordings:
    path = os.path.join(base, rec)
    all_npy.extend(glob.glob(os.path.join(path, "**", "*.npy"), recursive=True))

print(f"Scanning {len(all_npy)} .npy files for pixel_wp...\n")

nan_files = []
off_image   = []   # x outside [0,1280]
extreme     = []   # x outside [-200,1480] or y > 1440
catastrophic = []  # x outside [-400,2000] or y > 2000
huge = []          # |x| > 4000 or |y| > 4000
normal = []
frame_stats = {}   # (recording, frame_idx) -> stats for temporal spike analysis

for path in all_npy:
    try:
        data = np.load(path, allow_pickle=True).item()
        meta = data.get("metadata")
        if meta is None:
            continue
        gt_data = meta.get("gt_data", {})
        wp = gt_data.get("pixel_wp")
        if wp is None:
            continue
        wp = np.array(wp, dtype=np.float32)
        if wp.ndim != 2 or wp.shape[1] != 2:
            continue
        if not np.isfinite(wp).all():
            nan_files.append(path)
            wp = wp[np.isfinite(wp).all(axis=1)]
            if wp.size == 0:
                continue
        x, y = wp[:, 0], wp[:, 1]
        x_min_v, x_max_v = float(x.min()), float(x.max())
        y_max_v          = float(y.max())
        x_abs_max_v      = float(np.max(np.abs(x)))
        y_abs_max_v      = float(np.max(np.abs(y)))
        is_normal= True
        is_off  = x_min_v < X_LO_OFF or x_max_v > X_HI_OFF
        is_ext  = x_min_v < X_LO_EXT or x_max_v > X_HI_EXT or y_max_v > Y_HI_EXT
        is_cat  = x_min_v < X_LO_CAT or x_max_v > X_HI_CAT or y_max_v > Y_HI_CAT
        is_huge = x_abs_max_v > HUGE_ABS or y_abs_max_v > HUGE_ABS
        entry = {
            "path":    path,
            "x_min":   x_min_v,
            "x_max":   x_max_v,
            "y_max":   y_max_v,
            "x_abs_max": x_abs_max_v,
            "y_abs_max": y_abs_max_v,
            "n_off_x": int((x < X_LO_OFF).sum() + (x > X_HI_OFF).sum()),
            "n_cat":   int((x < X_LO_CAT).sum() + (x > X_HI_CAT).sum() + (y > Y_HI_CAT).sum()),
            "wp":      wp,
        }

        # Store per-frame stats for temporal continuity checks.
        rec_name = next((r for r in recordings if r in path), None)
        frame_name = os.path.basename(path).replace(".npy", "")
        try:
            frame_idx = int(frame_name)
        except ValueError:
            frame_idx = None
        if rec_name is not None and frame_idx is not None:
            frame_stats[(rec_name, frame_idx)] = {
                "path": path,
                "x_abs_max": x_abs_max_v,
                "y_abs_max": y_abs_max_v,
                "x_max": x_max_v,
                "y_max": y_max_v,
                "is_huge": is_huge,
                "is_cat": is_cat,
            }

        if is_huge: huge.append(entry)
        elif is_cat:  catastrophic.append(entry)
        elif is_ext: extreme.append(entry)
        elif is_off: off_image.append(entry)
        else : normal.append(entry)
    except Exception as e:
        print(f"  ERROR reading {path}: {e}")

total = len(all_npy)
print(f"Scanned {total} files")
print(f"  NaN files            : {len(nan_files)}")
print(f"  HUGE  (|x|>4000 or |y|>4000) : {len(huge)} ({100*len(huge)/total:.2f}%)")
print(f"  Off-image x (mild)   : {len(off_image)} ({100*len(off_image)/total:.2f}%)")
print(f"  Extreme (x<-200 | x>1480 | y>1440) : {len(extreme)} ({100*len(extreme)/total:.2f}%)")
print(f"  Catastrophic (x<-400 | x>2000 | y>2000) : {len(catastrophic)} ({100*len(catastrophic)/total:.2f}%)")

if nan_files:
    print(f"\n--- NaN files ---")
    for p in nan_files:
        print(f"  {p}")

# sort by worst exceedance
def _severity(s):
    return max(0, -s["x_min"] - X_LO_CAT, s["x_max"] - X_HI_CAT, s["y_max"] - Y_HI_CAT)

huge.sort(key=lambda s: max(s["x_abs_max"], s["y_abs_max"]), reverse=True)
catastrophic.sort(key=_severity, reverse=True)
extreme.sort(key=lambda s: max(0, -s["x_min"] - X_LO_EXT, s["x_max"] - X_HI_EXT, s["y_max"] - Y_HI_EXT), reverse=True)
off_image.sort(key=lambda s: s["n_off_x"], reverse=True)

if huge:
    print(f"\n=== HUGE outliers ({len(huge)}) (|x|>4000 or |y|>4000) ===")
    print(f"{'#':>3}  {'max|x|':>9}  {'max|y|':>9}  {'x_min':>9}  {'x_max':>9}  {'y_max':>9}  path")
    print("-" * 120)
    for i, s in enumerate(huge):
        print(f"{i+1:>3}  {s['x_abs_max']:>9.1f}  {s['y_abs_max']:>9.1f}  {s['x_min']:>9.1f}  {s['x_max']:>9.1f}  {s['y_max']:>9.1f}  {s['path']}")

    print(f"\n--- Full waypoint arrays for HUGE samples ---")
    for si, s in enumerate(huge):
        print(f"\n[{si+1}] {s['path']}")
        print(f"  {'idx':>4}  {'x':>9}  {'y':>9}  flags")
        for i, (xv, yv) in enumerate(s["wp"]):
            flags = []
            if abs(xv) > HUGE_ABS: flags.append("HUGE-X")
            if abs(yv) > HUGE_ABS: flags.append("HUGE-Y")
            if xv < X_LO_CAT or xv > X_HI_CAT: flags.append("CAT-X")
            if yv > Y_HI_CAT: flags.append("CAT-Y")
            print(f"  {i:>4}  {xv:>9.1f}  {yv:>9.1f}  {' '.join(flags)}")

# Temporal spike-recovery checks.
def _is_normal(e):
    return (not e["is_huge"]) and (e["x_abs_max"] <= X_HI_CAT) and (e["y_abs_max"] <= Y_HI_CAT)

strict_spike_recovery = []
post_spike_recovery = []

for key, cur in frame_stats.items():
    rec_name, idx = key
    if not cur["is_huge"]:
        continue

    prev = frame_stats.get((rec_name, idx - 1))
    nxt = frame_stats.get((rec_name, idx + 1))

    # Strict: immediate prev and next are normal.
    if prev is not None and nxt is not None and _is_normal(prev) and _is_normal(nxt):
        strict_spike_recovery.append({
            "recording": rec_name,
            "frame": idx,
            "huge": cur,
            "prev": prev,
            "next": nxt,
        })

    # Practical: huge frame recovers to normal within next 1..3 frames.
    recovery = None
    for step in [1, 2, 3]:
        cand = frame_stats.get((rec_name, idx + step))
        if cand is not None and _is_normal(cand):
            recovery = (step, cand)
            break
    if recovery is not None:
        post_spike_recovery.append({
            "recording": rec_name,
            "frame": idx,
            "huge": cur,
            "recover_step": recovery[0],
            "recover": recovery[1],
            "prev": prev,
        })

if strict_spike_recovery:
    print(f"\n=== STRICT SPIKE-RECOVERY huge events ({len(strict_spike_recovery)}) ===")
    print("(huge frame with immediate previous and next frames already back in normal range)")
    print(f"{'#':>3}  {'recording':<40}  {'frame':>7}  {'prev_max(x,y)':>22}  {'huge_max(x,y)':>22}  {'next_max(x,y)':>22}")
    print("-" * 130)
    for i, s in enumerate(sorted(strict_spike_recovery, key=lambda z: z["huge"]["x_abs_max"], reverse=True), start=1):
        p = s["prev"]
        h = s["huge"]
        n = s["next"]
        print(
            f"{i:>3}  {s['recording']:<40}  {s['frame']:>7d}  "
            f"({p['x_abs_max']:>7.1f},{p['y_abs_max']:>7.1f})  "
            f"({h['x_abs_max']:>7.1f},{h['y_abs_max']:>7.1f})  "
            f"({n['x_abs_max']:>7.1f},{n['y_abs_max']:>7.1f})"
        )

if post_spike_recovery:
    print(f"\n=== POST-SPIKE RECOVERY huge events ({len(post_spike_recovery)}) ===")
    print("(huge frame that returns to normal within next 1-3 frames)")
    print(f"{'#':>3}  {'recording':<40}  {'frame':>7}  {'huge_max(x,y)':>22}  {'recover_in':>10}  {'recover_max(x,y)':>22}")
    print("-" * 125)
    for i, s in enumerate(sorted(post_spike_recovery, key=lambda z: z["huge"]["x_abs_max"], reverse=True), start=1):
        h = s["huge"]
        r = s["recover"]
        print(
            f"{i:>3}  {s['recording']:<40}  {s['frame']:>7d}  "
            f"({h['x_abs_max']:>7.1f},{h['y_abs_max']:>7.1f})  "
            f"+{s['recover_step']} frame(s)  "
            f"({r['x_abs_max']:>7.1f},{r['y_abs_max']:>7.1f})"
        )

    print("\n--- Post-spike frame paths ---")
    for s in post_spike_recovery:
        idx = s["frame"]
        rec = s["recording"]
        step = s["recover_step"]
        rec_p = frame_stats.get((rec, idx - 1))
        rec_n = frame_stats.get((rec, idx + step))
        if rec_p is not None:
            print(f"  prev   : {rec_p['path']}")
        print(f"  huge   : {s['huge']['path']}")
        if rec_n is not None:
            print(f"  recover: {rec_n['path']}")

if catastrophic:
    print(f"\n=== CATASTROPHIC outliers ({len(catastrophic)}) ===")
    print(f"{'#':>3}  {'x_min':>9}  {'x_max':>9}  {'y_max':>9}  path")
    print("-" * 100)
    for i, s in enumerate(catastrophic):
        print(f"{i+1:>3}  {s['x_min']:>9.1f}  {s['x_max']:>9.1f}  {s['y_max']:>9.1f}  {s['path']}")

    print(f"\n--- Full waypoint arrays for catastrophic samples ---")
    for si, s in enumerate(catastrophic):
        print(f"\n[{si+1}] {s['path']}")
        print(f"  {'idx':>4}  {'x':>9}  {'y':>9}  flags")
        for i, (xv, yv) in enumerate(s["wp"]):
            flags = []
            if xv < X_LO_CAT or xv > X_HI_CAT: flags.append("CAT-X")
            elif xv < X_LO_EXT or xv > X_HI_EXT: flags.append("EXT-X")
            elif xv < X_LO_OFF or xv > X_HI_OFF: flags.append("off-X")
            if yv > Y_HI_CAT: flags.append("CAT-Y")
            elif yv > Y_HI_EXT: flags.append("EXT-Y")
            elif yv > IMG_H: flags.append("below-frame")
            print(f"  {i:>4}  {xv:>9.1f}  {yv:>9.1f}  {' '.join(flags)}")

if extreme:
    print(f"\n=== EXTREME outliers ({len(extreme)}) (x<-200 | x>1480 | y>1440) ===")
    print(f"{'#':>3}  {'x_min':>9}  {'x_max':>9}  {'y_max':>9}  path")
    print("-" * 100)
    for i, s in enumerate(extreme[:30]):
        print(f"{i+1:>3}  {s['x_min']:>9.1f}  {s['x_max']:>9.1f}  {s['y_max']:>9.1f}  {s['path']}")
    if len(extreme) > 30:
        print(f"  ... and {len(extreme)-30} more")

if off_image:
    print(f"\n=== OFF-IMAGE-X (mild, x outside [0,1280]) ({len(off_image)}) ===")
    print(f"{'#':>3}  {'x_min':>9}  {'x_max':>9}  {'y_max':>9}  {'n_off':>6}  path")
    print("-" * 110)
    for i, s in enumerate(off_image[:30]):
        print(f"{i+1:>3}  {s['x_min']:>9.1f}  {s['x_max']:>9.1f}  {s['y_max']:>9.1f}  {s['n_off_x']:>6}  {s['path']}")
    if len(off_image) > 30:
        print(f"  ... and {len(off_image)-30} more")

# Per-recording breakdown (all tiers)
print(f"\n=== Per-recording breakdown ===")
all_bad = {s["path"]: "catastrophic" for s in catastrophic}
all_bad.update({s["path"]: "extreme" for s in extreme})
all_bad.update({s["path"]: "off-image" for s in off_image})
all_bad.update({s["path"]: "huge" for s in huge})
print(f"{'recording':<45}  {'total':>6}  {'huge':>5}  {'cat':>5}  {'ext':>5}  {'off':>5}")
print("-" * 75)
for r in recordings:
    total_r  = len(glob.glob(os.path.join(base, r, "metadata", "*.npy")))
    huge_r   = sum(1 for s in huge        if r in s["path"])
    cat_r    = sum(1 for s in catastrophic if r in s["path"])
    ext_r    = sum(1 for s in extreme     if r in s["path"])
    off_r    = sum(1 for s in off_image   if r in s["path"])
    print(f"{r:<45}  {total_r:>6}  {huge_r:>5}  {cat_r:>5}  {ext_r:>5}  {off_r:>5}")

# Draw all detected waypoints onto images.
draw_groups = [
    ("normal", normal),
    ("huge", huge),
    ("catastrophic", catastrophic),
    ("extreme", extreme),
    ("off_image", off_image),
]
draw_output = Path(__file__).resolve().parent / "pixel_wp_draws"
_save_detected_overlay_images(draw_output, draw_groups)
