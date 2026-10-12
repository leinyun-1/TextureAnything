from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from config_utils import ROOT, load_configs, load_yaml, pattern_folder
from geometry import load_camera_gt, load_depth, load_mask, sample_nearest, transform_points, unproject_opencv


_PLY_DTYPE = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "i2",
    "ushort": "u2",
    "int": "i4",
    "uint": "u4",
    "int32": "i4",
    "uint32": "u4",
    "float": "f4",
    "double": "f8",
    "float32": "f4",
    "float64": "f8",
}


def load_ply_xyz(path: Path) -> np.ndarray:
    if not path.exists() or path.stat().st_size < 80:
        return np.zeros((0, 3), dtype=np.float64)
    raw = path.read_bytes()
    header_end = raw.find(b"end_header")
    if header_end < 0:
        return np.zeros((0, 3), dtype=np.float64)
    nl = raw.find(b"\n", header_end)
    header = raw[:header_end].decode("ascii", errors="ignore").splitlines()
    n_vert = 0
    fmt = "ascii"
    endian = "<"
    props: list[tuple[str, str]] = []
    in_vertex = False
    for ln in header:
        parts = ln.split()
        if not parts:
            continue
        if ln.startswith("format"):
            fmt = parts[1]
            endian = ">" if "big" in fmt else "<"
        elif ln.startswith("element vertex"):
            n_vert = int(parts[-1])
            in_vertex = True
        elif ln.startswith("element"):
            in_vertex = False
        elif in_vertex and ln.startswith("property") and parts[1] != "list":
            props.append((parts[2], parts[1]))
    if n_vert <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    body = raw[nl + 1 :] if nl >= 0 else raw[header_end:]
    if fmt.startswith("ascii"):
        rows = []
        for ln in body.decode("ascii", errors="ignore").splitlines():
            toks = ln.split()
            if len(toks) >= 3:
                rows.append([float(toks[0]), float(toks[1]), float(toks[2])])
            if len(rows) >= n_vert:
                break
        return np.asarray(rows, dtype=np.float64)
    names = [p[0] for p in props]
    dtype = np.dtype([(p[0], endian + _PLY_DTYPE[p[1]]) for p in props])
    verts = np.frombuffer(body, dtype=dtype, count=n_vert)
    if "x" not in names:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64)


def gt_cloud_from_depth(scene_dir: Path, stride: int) -> np.ndarray:
    cam = load_camera_gt(scene_dir)
    k = np.array(cam["K"], dtype=np.float64)
    chunks = []
    for fr in cam["frames"]:
        i = int(fr["id"])
        depth = load_depth(scene_dir, i)
        mask = load_mask(scene_dir, i)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        ys = ys[::stride]
        xs = xs[::stride]
        uv = np.stack([xs + 0.5, ys + 0.5], axis=1)
        z, valid = sample_nearest(depth, uv)
        keep = valid & (z > 1e-6)
        if not np.any(keep):
            continue
        x_cam = unproject_opencv(uv[keep], z[keep], k)
        x_w = transform_points(np.array(fr["T_wc"], dtype=np.float64), x_cam)
        chunks.append(x_w)
    if not chunks:
        return np.zeros((0, 3), dtype=np.float64)
    pts = np.concatenate(chunks, axis=0)
    # Voxel downsample
    voxel = 0.01
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[idx]


def mean_nn(src: np.ndarray, dst: np.ndarray, cap: float) -> float:
    if src.size == 0 or dst.size == 0:
        return float("nan")
    tree = cKDTree(dst)
    dist, _ = tree.query(src, k=1, workers=-1)
    return float(np.minimum(dist, cap).mean())


def evaluate_one(fused_ply: Path, gt_pts: np.ndarray, cap: float, max_points: int) -> dict:
    est = load_ply_xyz(fused_ply)
    if len(est) > max_points:
        rng = np.random.default_rng(0)
        est = est[rng.choice(len(est), size=max_points, replace=False)]
    gt_eval = gt_pts
    if len(gt_eval) > max_points:
        rng = np.random.default_rng(1)
        gt_eval = gt_eval[rng.choice(len(gt_eval), size=max_points, replace=False)]
    if len(est) and len(gt_pts):
        lo = gt_pts.min(axis=0) - 0.2
        hi = gt_pts.max(axis=0) + 0.2
        inside = np.all((est >= lo) & (est <= hi), axis=1)
        est_obj = est[inside]
    else:
        est_obj = est
        inside = np.zeros((len(est),), dtype=bool)
    acc = mean_nn(est_obj, gt_eval, cap) if len(est_obj) else float("nan")
    comp = mean_nn(gt_eval, est_obj, cap) if len(est_obj) else cap
    chamfer = float("nan") if (np.isnan(acc) or np.isnan(comp)) else 0.5 * (acc + comp)
    chamfer_sq = float("nan")
    if len(est_obj) and len(gt_eval):
        tree_gt = cKDTree(gt_eval)
        tree_est = cKDTree(est_obj)
        d1, _ = tree_gt.query(est_obj, k=1, workers=-1)
        d2, _ = tree_est.query(gt_eval, k=1, workers=-1)
        chamfer_sq = float(np.mean(np.minimum(d1, cap) ** 2) + np.mean(np.minimum(d2, cap) ** 2))
    return {
        "num_fused_points": int(len(est)),
        "num_fused_near_object": int(inside.sum()) if len(est) else 0,
        "accuracy": acc,
        "completeness": comp,
        "chamfer_distance": chamfer,
        "chamfer_l2": chamfer_sq,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="configs/smoke.yaml")
    p.add_argument("--objects", nargs="*")
    p.add_argument("--patterns", nargs="*")
    p.add_argument("--data_root")
    p.add_argument("--results_root")
    p.add_argument("--cap", type=float, default=0.2)
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
    stride = int(cfgs["colmap"].get("dense_gt_stride", 4))
    max_points = int(cfgs["colmap"].get("dense_max_points", 200000))
    rows = []
    gt_cache = {}
    for obj_name in objects:
        p0 = data_root / obj_name / pattern_folder("P0", cfgs["patterns"])
        gt_pts = gt_cloud_from_depth(p0, stride)
        gt_cache[obj_name] = gt_pts
        np.save(results_root / f"gt_cloud_{obj_name}.npy", gt_pts)
        for pattern_id in patterns:
            scene_dir = data_root / obj_name / pattern_folder(pattern_id, cfgs["patterns"])
            cam = load_camera_gt(scene_dir)
            fused = scene_dir / "colmap" / "mvs_gtpose" / "fused.ply"
            metrics = evaluate_one(fused, gt_pts, args.cap, max_points)
            row = {
                "scene_id": obj_name,
                "trajectory_id": "orbit300",
                "pattern_type": pattern_id,
                "pattern_seed": cam.get("seed", 0),
                "pattern_scale": cam.get("scale"),
                "alpha": cam.get("alpha"),
                "pose_source": "gt",
                **metrics,
            }
            rows.append(row)
            print(
                f"[eval-dense] {obj_name} {pattern_id} n={row['num_fused_points']} "
                f"acc={row['accuracy']} comp={row['completeness']}",
                flush=True,
            )
    csv_path = results_root / "dense_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("[eval-dense] wrote", csv_path)


if __name__ == "__main__":
    main()
