from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from config_utils import ROOT, load_configs, load_yaml


def run_py(script: str, extra: list[str]):
    cmd = [sys.executable, str(ROOT / "scripts" / script), *extra]
    print("[gso-p4]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="GSO weak-texture objects: original albedo vs P4 overlay.")
    p.add_argument("--experiment", default="configs/gso_p4.yaml")
    p.add_argument("--skip_download", action="store_true")
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

    if not args.skip_download:
        run_py("download_gso.py", common)
    if not args.skip_render:
        render_extra = common + (["--skip_existing"] if args.skip_existing else [])
        run_py("render_dataset.py", render_extra)
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
        "note": "P0 = original GSO albedo; P4 = original albedo + object-space speckle. Same GT poses for MVS.",
        "blender_exe": cfgs["render"]["blender_exe"],
        "colmap_exe": cfgs["colmap"]["colmap_exe"],
    }
    (results_root / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print("[gso-p4] complete", results_root)


if __name__ == "__main__":
    main()
