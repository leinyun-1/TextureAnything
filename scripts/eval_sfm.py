from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from colmap_model import (
    pick_largest_model,
    read_images_txt,
    read_points3d_txt,
    rotation_error_deg,
    umeyama,
)
from config_utils import ROOT, load_configs, load_yaml, pattern_folder
from geometry import load_camera_gt


def convert_model_to_txt(colmap_exe: str, model_dir: Path):
    if (model_dir / "images.txt").exists():
        return
    subprocess.run(
        [
            colmap_exe,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(model_dir),
            "--output_type",
            "TXT",
        ],
        check=True,
    )


def stats(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(values.mean()), float(np.median(values)), float(np.percentile(values, 90))


def evaluate_scene(colmap_exe: str, scene_dir: Path, exp: dict) -> dict:
    cam = load_camera_gt(scene_dir)
    n_frames = len(cam["frames"])
    gt_by_name = {}
    for fr in cam["frames"]:
        T_wc = np.array(fr["T_wc"], dtype=np.float64)
        name = Path(fr["file"]).name
        gt_by_name[name] = {"r_wc": T_wc[:3, :3], "center": T_wc[:3, 3]}

    row = {
        "scene_id": cam.get("object", scene_dir.parent.name),
        "trajectory_id": "orbit300",
        "pattern_type": cam.get("pattern"),
        "pattern_seed": cam.get("seed", 0),
        "pattern_scale": cam.get("scale"),
        "alpha": cam.get("alpha"),
        "num_frames": n_frames,
        "num_registered": 0,
        "registration_ratio": 0.0,
        "rotation_error_mean": float("nan"),
        "rotation_error_median": float("nan"),
        "rotation_error_p90": float("nan"),
        "camera_center_error_mean": float("nan"),
        "camera_center_error_median": float("nan"),
        "camera_center_error_p90": float("nan"),
        "num_sparse_points": 0,
        "mean_track_length": float("nan"),
        "mean_reprojection_error": float("nan"),
        "num_models": 0,
    }

    sparse = scene_dir / "colmap" / "sparse"
    model = pick_largest_model(sparse)
    if model is None:
        return row
    convert_model_to_txt(colmap_exe, model)
    images = read_images_txt(model / "images.txt")
    n_pts, mean_track, mean_repr = read_points3d_txt(model / "points3D.txt")
    row["num_models"] = sum(1 for p in sparse.iterdir() if p.is_dir()) if sparse.exists() else 0
    row["num_registered"] = len(images)
    row["registration_ratio"] = len(images) / max(n_frames, 1)
    row["num_sparse_points"] = n_pts
    row["mean_track_length"] = mean_track
    row["mean_reprojection_error"] = mean_repr

    names = [name for name in images if name in gt_by_name]
    if len(names) < 3:
        return row
    src = np.stack([images[n]["center"] for n in names])
    dst = np.stack([gt_by_name[n]["center"] for n in names])
    scale, rot, trans = umeyama(src, dst)
    rot_errs = []
    cen_errs = []
    for name in names:
        c_align = scale * rot @ images[name]["center"] + trans
        r_align = rot @ images[name]["r_wc"]
        rot_errs.append(rotation_error_deg(gt_by_name[name]["r_wc"], r_align))
        cen_errs.append(float(np.linalg.norm(c_align - gt_by_name[name]["center"])))
    r_mean, r_med, r_p90 = stats(np.array(rot_errs))
    c_mean, c_med, c_p90 = stats(np.array(cen_errs))
    row.update(
        {
            "rotation_error_mean": r_mean,
            "rotation_error_median": r_med,
            "rotation_error_p90": r_p90,
            "camera_center_error_mean": c_mean,
            "camera_center_error_median": c_med,
            "camera_center_error_p90": c_p90,
        }
    )
    return row


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="configs/smoke.yaml")
    p.add_argument("--objects", nargs="*")
    p.add_argument("--patterns", nargs="*")
    p.add_argument("--data_root")
    p.add_argument("--results_root")
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    cfgs = load_configs()
    exp = load_yaml(ROOT / args.experiment)
    objects = args.objects or exp["objects"]
    patterns = args.patterns or exp["patterns"]
    data_root = ROOT / (args.data_root or exp["data_root"])
    results_root = ROOT / (args.results_root or exp["results_root"])
    results_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for obj_name in objects:
        for pattern_id in patterns:
            scene_dir = data_root / obj_name / pattern_folder(pattern_id, cfgs["patterns"])
            row = evaluate_scene(cfgs["colmap"]["colmap_exe"], scene_dir, exp)
            row["scene_id"] = obj_name
            row["pattern_type"] = pattern_id
            rows.append(row)
            print(
                f"[eval-sfm] {obj_name} {pattern_id} "
                f"reg={row['num_registered']}/{row['num_frames']} "
                f"eR={row['rotation_error_median']} "
                f"eC={row['camera_center_error_median']}",
                flush=True,
            )
    fieldnames = list(rows[0].keys()) if rows else []
    csv_path = results_root / "sfm_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("[eval-sfm] wrote", csv_path)


if __name__ == "__main__":
    main()
