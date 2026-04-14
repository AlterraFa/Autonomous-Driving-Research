#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from config.enum import CameraView


def _load_coordinate_transform_funcs(repo_root: Path):
    module_path = repo_root / "src" / "math" / "coordinate_transform.py"
    spec = importlib.util.spec_from_file_location("coordinate_transform_standalone", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.camera_extrinsic, module.camera_intrinsic, module.ego_to_pixel


def _find_image_size(sample: dict, dataset_dir: Path, size_cache: Dict[Path, Tuple[int, int]]) -> Tuple[int, int]:
    if dataset_dir in size_cache:
        return size_cache[dataset_dir]

    img_file = sample.get("img_file", {})
    if not isinstance(img_file, dict) or len(img_file) == 0:
        raise ValueError("Missing or invalid img_file map")

    rel_path = next(iter(img_file.values()))
    img_path = dataset_dir / rel_path
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    h, w = img.shape[:2]
    size_cache[dataset_dir] = (w, h)
    return w, h


def _backfill_file(meta_path: Path, size_cache: Dict[Path, Tuple[int, int]], fov_deg: float,
                   camera_extrinsic, camera_intrinsic, ego_to_pixel) -> bool:
    sample = np.load(meta_path, allow_pickle=True).item()
    gt = sample["metadata"]["gt_data"]

    if "pixel_wp" in gt and "pixel_wp_temporal" in gt:
        return False

    dataset_dir = meta_path.parents[1]
    w, h = _find_image_size(sample, dataset_dir, size_cache)

    midlane_wp = np.asarray(gt["midlane_wp"], dtype=np.float64)
    midlane_wp_temporal = np.asarray(gt["midlane_wp_temporal"], dtype=np.float64)

    K = camera_intrinsic(w, h, fov_deg)
    E = camera_extrinsic(CameraView.FIRST_PERSON.value)

    gt["pixel_wp"] = ego_to_pixel(midlane_wp, K, E, w, h, clip=False)
    gt["pixel_wp_temporal"] = ego_to_pixel(midlane_wp_temporal, K, E, w, h, clip=False)

    np.save(meta_path, sample, allow_pickle=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill pixel waypoint arrays into CARLA metadata npy files.")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root, e.g. model/Autonomous_Dataset/carla/LAWM")
    parser.add_argument("--fov", type=float, default=90.0, help="Horizontal FOV in degrees (default: 90)")
    args = parser.parse_args()

    root = args.root.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    camera_extrinsic, camera_intrinsic, ego_to_pixel = _load_coordinate_transform_funcs(repo_root)
    meta_files = sorted(root.rglob("metadata/*.npy"))
    if not meta_files:
        print(f"No metadata npy files found under: {root}")
        return

    size_cache: Dict[Path, Tuple[int, int]] = {}
    updated = 0
    skipped = 0
    failed = 0

    for p in meta_files:
        try:
            if _backfill_file(p, size_cache, args.fov, camera_extrinsic, camera_intrinsic, ego_to_pixel):
                updated += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {p}: {exc}")

    print(f"Scanned: {len(meta_files)}")
    print(f"Updated: {updated}")
    print(f"Skipped: {skipped}")
    print(f"Failed : {failed}")


if __name__ == "__main__":
    main()
