from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from colmap_io import image_ids_to_pair_id, read_colmap_features
from config_utils import ROOT, load_configs, load_yaml, pattern_folder
from geometry import (
    erode_mask,
    invert_T,
    load_camera_gt,
    load_depth,
    load_mask,
    project_opencv,
    sample_nearest,
    tile_coverage,
    transform_points,
    unproject_opencv,
)


def frame_index_from_name(name: str) -> int:
    return int(Path(name).stem)


def evaluate_pair(
    scene_dir: Path,
    cam: dict,
    feats: dict,
    i: int,
    j: int,
    cfg: dict,
):
    K = np.array(cam["K"], dtype=np.float64)
    T_wc_i = np.array(cam["frames"][i]["T_wc"], dtype=np.float64)
    T_wc_j = np.array(cam["frames"][j]["T_wc"], dtype=np.float64)
    T_cw_j = invert_T(T_wc_j)

    name_i = f"{i:04d}.png"
    name_j = f"{j:04d}.png"
    id_i = feats["name_to_id"][name_i]
    id_j = feats["name_to_id"][name_j]
    kp_i = feats["keypoints"].get(id_i, np.zeros((0, 2)))
    kp_j = feats["keypoints"].get(id_j, np.zeros((0, 2)))

    pair_id = image_ids_to_pair_id(id_i, id_j)
    raw = feats["matches"].get(pair_id, np.zeros((0, 2), dtype=np.uint32))
    inl = feats["inliers"].get(pair_id, np.zeros((0, 2), dtype=np.uint32))
    if raw.size and id_i > id_j:
        raw = raw[:, ::-1]
    if inl.size and id_i > id_j:
        inl = inl[:, ::-1]

    depth_i = load_depth(scene_dir, i)
    depth_j = load_depth(scene_dir, j)
    mask_i = load_mask(scene_dir, i)
    mask_j = load_mask(scene_dir, j)
    interior_i = erode_mask(mask_i, int(cfg["interior_erode_px"]))
    px_thr = float(cfg["correct_match_px"])
    abs_tol = float(cfg["depth_abs_tol"])
    rel_tol = float(cfg["depth_rel_tol"])

    def classify(matches):
        if matches.size == 0 or kp_i.size == 0 or kp_j.size == 0:
            return {
                "n": 0,
                "correct": 0,
                "interior_correct": 0,
                "correct_uv": np.zeros((0, 2)),
                "interior_uv": np.zeros((0, 2)),
            }
        u_i = kp_i[matches[:, 0]]
        u_j = kp_j[matches[:, 1]]
        z_i, valid_i = sample_nearest(depth_i, u_i)
        m_i, _ = sample_nearest(mask_i, u_i)
        interior, _ = sample_nearest(interior_i, u_i)
        valid = valid_i & (z_i > 1e-6) & (m_i > 0)
        X_cam = unproject_opencv(u_i, z_i, K)
        X_w = transform_points(T_wc_i, X_cam)
        X_cam_j = transform_points(T_cw_j, X_w)
        u_hat, z_proj = project_opencv(X_cam_j, K)
        z_j, valid_j = sample_nearest(depth_j, u_hat)
        m_j, _ = sample_nearest(mask_j, u_hat)
        visible = (
            valid
            & valid_j
            & (z_proj > 1e-6)
            & (m_j > 0)
            & (np.abs(z_proj - z_j) < (abs_tol + rel_tol * np.maximum(z_j, 1e-6)))
        )
        dist = np.linalg.norm(u_j - u_hat, axis=1)
        correct = visible & (dist < px_thr)
        interior_correct = correct & (interior > 0)
        return {
            "n": int(len(matches)),
            "correct": int(correct.sum()),
            "interior_correct": int(interior_correct.sum()),
            "correct_uv": u_i[correct],
            "interior_uv": u_i[interior_correct],
        }

    putative = classify(raw)
    geom = classify(inl)

    # Repeatability from GT projection of keypoints, independent of matcher.
    repeatable = 0
    visible_kp = 0
    if kp_i.size:
        z_i, valid_i = sample_nearest(depth_i, kp_i)
        m_i, _ = sample_nearest(mask_i, kp_i)
        valid = valid_i & (z_i > 1e-6) & (m_i > 0)
        X_cam = unproject_opencv(kp_i, z_i, K)
        X_w = transform_points(T_wc_i, X_cam)
        X_cam_j = transform_points(T_cw_j, X_w)
        u_hat, z_proj = project_opencv(X_cam_j, K)
        z_j, valid_j = sample_nearest(depth_j, u_hat)
        m_j, _ = sample_nearest(mask_j, u_hat)
        visible = (
            valid
            & valid_j
            & (z_proj > 1e-6)
            & (m_j > 0)
            & (np.abs(z_proj - z_j) < (abs_tol + rel_tol * np.maximum(z_j, 1e-6)))
        )
        visible_kp = int(visible.sum())
        if visible_kp and kp_j.size:
            vis_uv = u_hat[visible]
            # For each projected point, nearest neighbor in kp_j.
            d2 = ((vis_uv[:, None, :] - kp_j[None, :, :]) ** 2).sum(axis=2)
            nn = np.sqrt(d2.min(axis=1))
            repeatable = int((nn < px_thr).sum())

    precision = putative["correct"] / putative["n"] if putative["n"] else float("nan")
    coverage = tile_coverage(mask_i, putative["correct_uv"], int(cfg["coverage_tiles"]))
    interior_coverage = tile_coverage(interior_i, putative["interior_uv"], int(cfg["coverage_tiles"]))
    return {
        "frame_i": i,
        "frame_j": j,
        "view_gap": j - i,
        "num_keypoints_i": int(len(kp_i)),
        "num_keypoints_j": int(len(kp_j)),
        "num_matches": putative["n"],
        "num_correct_matches": putative["correct"],
        "match_precision": precision,
        "num_geometric_inliers": geom["n"],
        "repeatability": repeatable / visible_kp if visible_kp else 0.0,
        "surface_coverage": coverage,
        "interior_correct_matches": putative["interior_correct"],
        "interior_coverage": interior_coverage,
        "visible_keypoints": visible_kp,
    }


def self_check_reprojection(scene_dir: Path, cam: dict, n_samples: int = 200) -> dict:
    K = np.array(cam["K"], dtype=np.float64)
    i = 0
    depth = load_depth(scene_dir, i)
    mask = load_mask(scene_dir, i)
    ys, xs = np.where((mask > 0) & (depth > 1e-6))
    if len(xs) == 0:
        return {"n": 0, "mean_px": None}
    rng = np.random.default_rng(0)
    idx = rng.choice(len(xs), size=min(n_samples, len(xs)), replace=False)
    uv = np.stack([xs[idx] + 0.5, ys[idx] + 0.5], axis=1)
    z, _ = sample_nearest(depth, uv)
    X_cam = unproject_opencv(uv, z, K)
    u_hat, _ = project_opencv(X_cam, K)
    err = np.linalg.norm(u_hat - uv, axis=1)
    return {"n": int(len(err)), "mean_px": float(err.mean()), "max_px": float(err.max())}


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
    cfg = cfgs["colmap"]
    rows = []
    checks = []
    for obj_name in objects:
        for pattern_id in patterns:
            scene_dir = data_root / obj_name / pattern_folder(pattern_id, cfgs["patterns"])
            cam = load_camera_gt(scene_dir)
            check = self_check_reprojection(scene_dir, cam)
            check.update({"scene_id": obj_name, "pattern_type": pattern_id})
            checks.append(check)
            feats = read_colmap_features(scene_dir / "colmap" / "database.db")
            n = len(cam["frames"])
            for gap in cfg["view_gaps"]:
                for i in range(0, n - gap):
                    row = evaluate_pair(scene_dir, cam, feats, i, i + gap, cfg)
                    row.update(
                        {
                            "scene_id": obj_name,
                            "trajectory_id": "orbit300",
                            "pattern_type": pattern_id,
                            "pattern_seed": cam.get("seed", 0),
                            "pattern_scale": cam.get("scale", None),
                            "alpha": cam.get("alpha", None),
                        }
                    )
                    rows.append(row)
            print(f"[eval] {obj_name} {pattern_id} pairs={sum(1 for r in rows if r['scene_id']==obj_name and r['pattern_type']==pattern_id)}", flush=True)

    fieldnames = [
        "scene_id",
        "trajectory_id",
        "pattern_type",
        "pattern_seed",
        "pattern_scale",
        "alpha",
        "frame_i",
        "frame_j",
        "view_gap",
        "num_keypoints_i",
        "num_keypoints_j",
        "num_matches",
        "num_correct_matches",
        "match_precision",
        "num_geometric_inliers",
        "repeatability",
        "surface_coverage",
        "interior_correct_matches",
        "interior_coverage",
        "visible_keypoints",
    ]
    csv_path = results_root / "pair_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(results_root / "reprojection_selfcheck.json", "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2)
    print("[eval] wrote", csv_path)


if __name__ == "__main__":
    main()
