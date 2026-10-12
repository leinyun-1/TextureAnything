from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from config_utils import ROOT, load_configs, load_yaml, pattern_folder
from geometry import load_camera_gt


def run_colmap(cmd):
    print("[colmap]", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def process_scene(colmap_exe: str, scene_dir: Path, cfg: dict, use_gpu: int):
    images = scene_dir / "rgb"
    work = scene_dir / "colmap"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    db = work / "database.db"
    cam = load_camera_gt(scene_dir)
    K = cam["K"]
    params = f"{K[0][0]},{K[1][1]},{K[0][2]},{K[1][2]}"

    extract = [
        colmap_exe,
        "feature_extractor",
        "--database_path",
        str(db),
        "--image_path",
        str(images),
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_model",
        cfg.get("camera_model", "PINHOLE"),
        "--ImageReader.camera_params",
        params,
        "--FeatureExtraction.type",
        cfg.get("feature_type", "SIFT"),
        "--FeatureExtraction.use_gpu",
        str(use_gpu),
        "--SiftExtraction.max_num_features",
        str(cfg.get("max_num_features", 8192)),
    ]
    match = [
        colmap_exe,
        "exhaustive_matcher",
        "--database_path",
        str(db),
        "--FeatureMatching.type",
        cfg.get("matching_type", "SIFT_BRUTEFORCE"),
        "--FeatureMatching.use_gpu",
        str(use_gpu),
        "--SiftMatching.max_ratio",
        str(cfg.get("max_ratio", 0.8)),
        "--SiftMatching.max_distance",
        str(cfg.get("max_distance", 0.7)),
        "--SiftMatching.cross_check",
        str(cfg.get("cross_check", 1)),
    ]
    try:
        run_colmap(extract)
        run_colmap(match)
    except subprocess.CalledProcessError:
        if use_gpu:
            print("[colmap] GPU failed, retrying on CPU", flush=True)
            process_scene(colmap_exe, scene_dir, cfg, use_gpu=0)
            return
        raise
    meta = {"database": str(db), "use_gpu": use_gpu, "camera_params": params}
    with open(work / "run.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="configs/smoke.yaml")
    p.add_argument("--objects", nargs="*")
    p.add_argument("--patterns", nargs="*")
    p.add_argument("--data_root")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    cfgs = load_configs()
    exp = load_yaml(ROOT / args.experiment)
    objects = args.objects or exp["objects"]
    patterns = args.patterns or exp["patterns"]
    data_root = ROOT / (args.data_root or exp["data_root"])
    use_gpu = 0 if args.cpu else int(cfgs["colmap"].get("use_gpu", 1))
    for obj_name in objects:
        for pattern_id in patterns:
            scene_dir = data_root / obj_name / pattern_folder(pattern_id, cfgs["patterns"])
            print(f"[features] {scene_dir}", flush=True)
            db = scene_dir / "colmap" / "database.db"
            if args.skip_existing and db.exists() and db.stat().st_size > 1024:
                print(f"[features] skip existing {db}", flush=True)
                continue
            process_scene(cfgs["colmap"]["colmap_exe"], scene_dir, cfgs["colmap"], use_gpu)
    print("[features] done")


if __name__ == "__main__":
    main()
