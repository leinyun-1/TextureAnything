from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from config_utils import ROOT, load_configs, load_yaml, pattern_folder


def run_colmap(cmd):
    print("[colmap]", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def process_scene(colmap_exe: str, scene_dir: Path, cfg: dict):
    db = scene_dir / "colmap" / "database.db"
    images = scene_dir / "rgb"
    if not db.exists():
        raise FileNotFoundError(f"missing feature database: {db}")
    sparse = scene_dir / "colmap" / "sparse"
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True, exist_ok=True)
    cmd = [
        colmap_exe,
        "mapper",
        "--database_path",
        str(db),
        "--image_path",
        str(images),
        "--output_path",
        str(sparse),
        "--Mapper.ba_refine_focal_length",
        str(int(cfg.get("ba_refine_focal_length", 0))),
        "--Mapper.ba_refine_principal_point",
        str(int(cfg.get("ba_refine_principal_point", 0))),
        "--Mapper.ba_refine_extra_params",
        str(int(cfg.get("ba_refine_extra_params", 0))),
    ]
    try:
        run_colmap(cmd)
    except subprocess.CalledProcessError as exc:
        print(f"[sfm] mapper failed ({exc.returncode}): {scene_dir}", flush=True)
    models = [p for p in sparse.iterdir() if p.is_dir()]
    meta = {"num_models": len(models), "models": [p.name for p in models]}
    with open(scene_dir / "colmap" / "mapper.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[sfm] {scene_dir} models={len(models)}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
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
    for obj_name in objects:
        for pattern_id in patterns:
            scene_dir = data_root / obj_name / pattern_folder(pattern_id, cfgs["patterns"])
            if args.skip_existing and (scene_dir / "colmap" / "sparse").exists() and any(
                (scene_dir / "colmap" / "sparse").iterdir()
            ):
                print(f"[sfm] skip existing {scene_dir}", flush=True)
                continue
            process_scene(cfgs["colmap"]["colmap_exe"], scene_dir, cfgs["colmap"])
    print("[sfm] done")


if __name__ == "__main__":
    main()
