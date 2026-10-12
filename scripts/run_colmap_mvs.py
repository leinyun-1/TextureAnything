from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from config_utils import ROOT, load_configs, load_yaml, pattern_folder
from gt_colmap_model import write_gt_colmap_model, write_patch_match_cfg


def run_colmap(cmd):
    print("[colmap]", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_bin_model(colmap_exe: str, model_dir: Path):
    if (model_dir / "images.bin").exists() and (model_dir / "cameras.bin").exists():
        return
    run_colmap(
        [
            colmap_exe,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(model_dir),
            "--output_type",
            "BIN",
        ]
    )


def run_mvs_for_pattern(colmap_exe: str, cfg: dict, gt_model: Path, sources: dict, rgb_dir: Path, dense_dir: Path):
    if dense_dir.exists():
        shutil.rmtree(dense_dir)
    dense_dir.mkdir(parents=True, exist_ok=True)
    run_colmap(
        [
            colmap_exe,
            "image_undistorter",
            "--image_path",
            str(rgb_dir),
            "--input_path",
            str(gt_model),
            "--output_path",
            str(dense_dir),
            "--output_type",
            "COLMAP",
            "--copy_policy",
            "COPY",
            "--num_patch_match_src_images",
            str(int(cfg.get("mvs_num_src_images", 20))),
        ]
    )
    write_patch_match_cfg(dense_dir / "stereo" / "patch-match.cfg", sources)
    run_colmap(
        [
            colmap_exe,
            "patch_match_stereo",
            "--workspace_path",
            str(dense_dir),
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.gpu_index",
            str(int(cfg.get("mvs_gpu_index", 0))),
            "--PatchMatchStereo.cache_size",
            str(int(cfg.get("mvs_cache_size", 8))),
            "--PatchMatchStereo.geom_consistency",
            "1",
        ]
    )
    fused = dense_dir / "fused.ply"
    run_colmap(
        [
            colmap_exe,
            "stereo_fusion",
            "--workspace_path",
            str(dense_dir),
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric",
            "--output_path",
            str(fused),
            "--StereoFusion.cache_size",
            str(int(cfg.get("mvs_cache_size", 8))),
            "--StereoFusion.max_depth_error",
            str(float(cfg.get("mvs_max_depth_error", 0.01))),
            "--StereoFusion.min_num_pixels",
            str(int(cfg.get("mvs_min_num_pixels", 5))),
        ]
    )
    return fused


def parse_args():
    p = argparse.ArgumentParser(description="Dense MVS with identical GT poses for every pattern.")
    p.add_argument("--experiment", default="configs/smoke.yaml")
    p.add_argument("--objects", nargs="*")
    p.add_argument("--patterns", nargs="*")
    p.add_argument("--data_root")
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    cfgs = load_configs()
    exp = load_yaml(ROOT / args.experiment)
    objects = args.objects or exp["objects"]
    patterns = args.patterns or exp["patterns"]
    data_root = ROOT / (args.data_root or exp["data_root"])
    colmap_exe = cfgs["colmap"]["colmap_exe"]
    cfg = cfgs["colmap"]

    for obj_name in objects:
        p0_dir = data_root / obj_name / pattern_folder("P0", cfgs["patterns"])
        gt_model = data_root / obj_name / "_gt_colmap_model"
        print(f"[mvs] writing GT model from {p0_dir}", flush=True)
        sources = write_gt_colmap_model(p0_dir, gt_model, cfg)
        ensure_bin_model(colmap_exe, gt_model)
        write_patch_match_cfg(gt_model / "patch-match.cfg", sources)
        (gt_model / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
        for pattern_id in patterns:
            scene_dir = data_root / obj_name / pattern_folder(pattern_id, cfgs["patterns"])
            dense_dir = scene_dir / "colmap" / "mvs_gtpose"
            fused = dense_dir / "fused.ply"
            if args.skip_existing and fused.exists() and fused.stat().st_size > 80:
                print(f"[mvs] skip existing {dense_dir}", flush=True)
                continue
            print(f"[mvs] {obj_name} {pattern_id}  (GT poses, only RGB changes)", flush=True)
            try:
                run_mvs_for_pattern(colmap_exe, cfg, gt_model, sources, scene_dir / "rgb", dense_dir)
                status = "ok" if fused.exists() else "missing_fused"
            except subprocess.CalledProcessError as exc:
                status = f"failed:{exc.returncode}"
                print(f"[mvs] FAILED {obj_name} {pattern_id}: {exc}", flush=True)
            meta = {
                "pose_source": "gt",
                "neighbor_source": "gt_camera_center_distance",
                "rgb": str(scene_dir / "rgb"),
                "gt_model": str(gt_model),
                "fused": str(fused),
                "status": status,
            }
            dense_dir.mkdir(parents=True, exist_ok=True)
            (dense_dir / "mvs.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[mvs] done")


if __name__ == "__main__":
    main()
