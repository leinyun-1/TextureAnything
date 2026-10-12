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
    print("[smoke]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Stage A/B smoke test: render, SIFT match, GT correspondence.")
    p.add_argument("--experiment", default="configs/smoke.yaml")
    p.add_argument("--quick", action="store_true", help="1 object, 8 views, P0/P1/P4 only")
    p.add_argument("--skip_render", action="store_true")
    p.add_argument("--skip_colmap", action="store_true")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    cfgs = load_configs()
    exp = load_yaml(ROOT / args.experiment)

    if args.quick:
        data_root = "data/smoke_quick"
        results_root = "results/smoke_quick"
        objects = ["box"]
        patterns = ["P0", "P1", "P4"]
        render_extra = [
            "--experiment",
            args.experiment,
            "--objects",
            *objects,
            "--patterns",
            *patterns,
            "--num_views",
            "8",
            "--resolution",
            "384",
            "--samples",
            "8",
            "--data_root",
            data_root,
        ]
        common = [
            "--experiment",
            args.experiment,
            "--objects",
            *objects,
            "--patterns",
            *patterns,
            "--data_root",
            data_root,
        ]
        eval_extra = common + ["--results_root", results_root]
        agg_extra = ["--experiment", args.experiment, "--results_root", results_root]
        env_dir = ROOT / results_root
    else:
        data_root = exp["data_root"]
        results_root = exp["results_root"]
        render_extra = ["--experiment", args.experiment]
        common = ["--experiment", args.experiment]
        eval_extra = common
        agg_extra = common
        env_dir = ROOT / results_root

    if not args.skip_render:
        run_py("render_dataset.py", render_extra)
    if not args.skip_colmap:
        colmap_extra = common + (["--cpu"] if args.cpu else [])
        run_py("run_colmap_features.py", colmap_extra)
    run_py("eval_correspondence.py", eval_extra)
    run_py("aggregate_results.py", agg_extra)

    env = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "blender_exe": cfgs["render"]["blender_exe"],
        "colmap_exe": cfgs["colmap"]["colmap_exe"],
        "quick": args.quick,
        "experiment": args.experiment,
        "data_root": data_root,
    }
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print("[smoke] complete", env_dir)


if __name__ == "__main__":
    main()
