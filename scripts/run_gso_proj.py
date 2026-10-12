from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from config_utils import ROOT, load_configs, load_yaml, pattern_folder


def run_py(script: str, extra: list[str]):
    cmd = [sys.executable, str(ROOT / "scripts" / script), *extra]
    print("[gso-proj]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def _copy_colmap_reuse(src: Path, dst: Path):
    """Copy SIFT db, SfM model, and fused cloud. Skip dense depth/normal maps."""
    if not (src / "database.db").exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "database.db", dst / "database.db")
    for name in ("run.json", "mapper.json"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    if (src / "sparse").exists():
        shutil.copytree(src / "sparse", dst / "sparse")
    fused = src / "mvs_gtpose" / "fused.ply"
    if fused.exists():
        dense = dst / "mvs_gtpose"
        dense.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fused, dense / "fused.ply")
        meta = src / "mvs_gtpose" / "mvs.json"
        if meta.exists():
            shutil.copy2(meta, dense / "mvs.json")


def copy_reuse_renders(exp: dict, patterns_cfg: dict):
    reuse_root = ROOT / exp["reuse_from"]
    data_root = ROOT / exp["data_root"]
    copied = []
    for obj_name in exp["objects"]:
        for pattern_id in ("P0", "P4"):
            folder = pattern_folder(pattern_id, patterns_cfg)
            src = reuse_root / obj_name / folder
            dst = data_root / obj_name / folder
            rgb0 = dst / "rgb" / "0000.png"
            if rgb0.exists() and (dst / "camera_gt.json").exists():
                if not (dst / "colmap" / "database.db").exists():
                    _copy_colmap_reuse(src / "colmap", dst / "colmap")
                print(f"[gso-proj] reuse already present {dst}", flush=True)
                continue
            if not (src / "rgb" / "0000.png").exists():
                print(f"[gso-proj] no reuse source {src}, will render", flush=True)
                continue
            if dst.exists():
                shutil.rmtree(dst)
            dst.mkdir(parents=True, exist_ok=True)
            for name in ("rgb", "depth", "mask"):
                shutil.copytree(src / name, dst / name)
            shutil.copy2(src / "camera_gt.json", dst / "camera_gt.json")
            _copy_colmap_reuse(src / "colmap", dst / "colmap")
            copied.append(f"{obj_name}/{folder}")
            print(f"[gso-proj] copied {src} -> {dst}", flush=True)
    return copied


def parse_args():
    p = argparse.ArgumentParser(
        description="Virtual projector ablation: P0 vs P4-albedo vs P4-projector on hard GSO objects."
    )
    p.add_argument("--experiment", default="configs/gso_proj.yaml")
    p.add_argument("--skip_render", action="store_true")
    p.add_argument("--skip_sfm", action="store_true")
    p.add_argument("--skip_mvs", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    cfgs = load_configs()
    exp = load_yaml(ROOT / args.experiment)
    common = ["--experiment", args.experiment]
    results_root = ROOT / exp["results_root"]
    results_root.mkdir(parents=True, exist_ok=True)

    copied = []
    if not args.skip_render:
        copied = copy_reuse_renders(exp, cfgs["patterns"])
        run_py("render_dataset.py", common + ["--skip_existing"])

    colmap_extra = common + (["--cpu"] if args.cpu else [])
    if args.skip_existing:
        colmap_extra.append("--skip_existing")
    run_py("run_colmap_features.py", colmap_extra)
    run_py("eval_correspondence.py", common)
    if not args.skip_sfm:
        run_py("run_colmap_sfm.py", colmap_extra if args.skip_existing else common)
        run_py("eval_sfm.py", common)
    if not args.skip_mvs:
        run_py("run_colmap_mvs.py", common + ["--skip_existing"])
        run_py("eval_dense.py", common)
    run_py("aggregate_results.py", common)

    env = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment,
        "note": exp.get(
            "note",
            "P0 = original albedo; P4 = albedo + object-space speckle; P4P = speckle projectors.",
        ),
        "reused_p0_p4": copied,
        "blender_exe": cfgs["render"]["blender_exe"],
        "colmap_exe": cfgs["colmap"]["colmap_exe"],
    }
    (results_root / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print("[gso-proj] complete", results_root)


if __name__ == "__main__":
    main()
