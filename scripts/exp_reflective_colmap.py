"""How well does COLMAP reconstruct reflective objects? EEVEE with screen-space ray tracing.

Every material of one object shares the same camera trajectory and HDRI, so the
only thing that changes between rows is the surface.
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from argparse import Namespace
from pathlib import Path

from config_utils import ABO_MODELS, ABO_ROOT, ROOT, load_configs
from eval_sfm import evaluate_scene
from render_abo_black import (
    HDRI_DIR, HEIGHT, NUM_VIEWS, QUEUE, WIDTH, hdri_paths, ready, resolve_glb, sample_pair,
)
from render_dataset import render_variant
from run_colmap_features import process_scene as extract_and_match
from run_colmap_sfm import process_scene as run_mapper

OUT_ROOT = ABO_ROOT / "renders_reflect"
RESULTS = ROOT / "results" / "reflect_colmap"
DEFAULT_IDS = ("B075HWK9QT", "B07BLB6644", "B07BG69MRW", "B0754KHKCX")
MATERIALS = ("matte", "glossy", "abo_pbr", "white_matte", "white_glaze", "chrome", "brushed_metal")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", nargs="*", default=list(DEFAULT_IDS))
    p.add_argument("--materials", nargs="*", default=list(MATERIALS))
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    p.add_argument("--skip-render", action="store_true")
    p.add_argument("--skip-colmap", action="store_true")
    p.add_argument("--redo-colmap", action="store_true")
    p.add_argument("--speckle", action="store_true", help="print matte dots over each material")
    p.add_argument("--speckle-scale", type=float, default=28.0)
    p.add_argument("--speckle-threshold", type=float, default=0.08)
    p.add_argument("--results-name", default="")
    return p.parse_args()


def mean_keypoints(db: Path) -> float:
    if not db.exists():
        return float("nan")
    with sqlite3.connect(db) as con:
        value = con.execute("SELECT AVG(rows) FROM keypoints").fetchone()[0]
    return float(value) if value is not None else 0.0


def main():
    args = parse_args()
    cfgs = load_configs()
    items = {i["id"]: i for i in json.loads(QUEUE.read_text(encoding="utf-8"))["items"]}
    hdris = hdri_paths(HDRI_DIR)
    render_args = Namespace(
        num_views=NUM_VIEWS, resolution=WIDTH, samples=32, alpha=0.4, scale="medium", seed=0
    )
    colmap_exe = cfgs["colmap"]["colmap_exe"]
    rows = []
    for object_id in args.ids:
        item = items[object_id]
        spec = sample_pair(object_id, hdris)
        obj_root = args.out_root / object_id
        obj_root.mkdir(parents=True, exist_ok=True)
        (obj_root / "pair.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        for material in args.materials:
            variant = f"{material}_speckle" if args.speckle else material
            scene = obj_root / variant
            if not args.skip_render and not ready(scene):
                extra = {
                    key: spec[key]
                    for key in (
                        "lens_mm", "radius", "azimuth_span_deg", "azimuth_start_deg", "elevation_deg",
                        "elevation_amp_deg", "hdri", "hdri_rotation_deg", "hdri_strength", "frame_fill",
                    )
                }
                extra.update(
                    rgb_only=True, engine="BLENDER_EEVEE", raytrace=True, taa_samples=32,
                    material=material, resolution_x=WIDTH, resolution_y=HEIGHT,
                )
                if args.speckle:
                    extra.update(
                        speckle=True,
                        speckle_scale=args.speckle_scale,
                        speckle_threshold=args.speckle_threshold,
                    )
                render_variant(
                    render_args, cfgs["render"], cfgs["patterns"], object_id, "P0", scene,
                    mesh=resolve_glb(item, ABO_MODELS), extra=extra,
                )
            if not args.skip_colmap and ready(scene):
                if args.redo_colmap or not (scene / "colmap" / "mapper.json").exists():
                    extract_and_match(colmap_exe, scene, cfgs["colmap"], int(cfgs["colmap"].get("use_gpu", 1)))
                    run_mapper(colmap_exe, scene, cfgs["colmap"])
            if not (scene / "colmap" / "mapper.json").exists():
                continue
            row = evaluate_scene(colmap_exe, scene, {})
            row.update(
                scene_id=object_id,
                product_type=item.get("product_type", ""),
                material=material,
                speckle=bool(args.speckle),
                mean_keypoints=mean_keypoints(scene / "colmap" / "database.db"),
            )
            rows.append(row)
            print(
                f"[reflect] {object_id} {variant:21s} kp={row['mean_keypoints']:.0f} "
                f"reg={row['num_registered']}/{row['num_frames']} pts={row['num_sparse_points']} "
                f"eR={row['rotation_error_median']:.2f} eC={row['camera_center_error_median']:.3f}",
                flush=True,
            )
    if rows:
        RESULTS.mkdir(parents=True, exist_ok=True)
        name = args.results_name or ("sfm_metrics_speckle.csv" if args.speckle else "sfm_metrics.csv")
        keys = ["scene_id", "product_type", "material", "speckle", "mean_keypoints"]
        keys += [k for k in rows[0] if k not in keys]
        with open(RESULTS / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print("[reflect] wrote", RESULTS / name, flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    main()
