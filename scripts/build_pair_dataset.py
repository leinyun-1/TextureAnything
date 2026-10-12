"""Package a small (x, y) video-pair set from renders that already passed the COLMAP gate.

x is the original albedo. y is the object-space P4 overlay. Same camera trajectory.
An object is kept only when P0 registered no views and P4 registered at least
p4_min_registered views. Projector renders are not copied.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config_utils import ROOT, load_configs, load_yaml, pattern_folder


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def cameras_match(a: dict, b: dict, atol: float = 1e-8) -> tuple[bool, str]:
    if len(a["frames"]) != len(b["frames"]):
        return False, f"frame count {len(a['frames'])} vs {len(b['frames'])}"
    if not np.allclose(np.array(a["K"], dtype=np.float64), np.array(b["K"], dtype=np.float64), atol=atol):
        return False, "K differs"
    for fa, fb in zip(a["frames"], b["frames"]):
        if Path(fa["file"]).name != Path(fb["file"]).name:
            return False, f"filename {fa['file']} vs {fb['file']}"
        if not np.allclose(np.array(fa["T_wc"], dtype=np.float64), np.array(fb["T_wc"], dtype=np.float64), atol=atol):
            return False, f"T_wc differs at frame {fa['id']}"
    return True, ""


def select_objects(sfm: pd.DataFrame, exp: dict, cfg: dict) -> list[dict]:
    held = set(exp.get("held_out") or [])
    assets = exp.get("assets") or {}
    rows = []
    for scene_id, group in sfm.groupby("scene_id"):
        by = {r.pattern_type: r for r in group.itertuples(index=False)}
        p0 = by.get("P0")
        p4 = by.get("P4")
        if p0 is None or p4 is None:
            continue
        if int(p0.num_registered) > int(cfg["p0_max_registered"]):
            continue
        if int(p4.num_registered) < int(cfg["p4_min_registered"]):
            continue
        ratio = float(p4.registration_ratio)
        rows.append(
            {
                "id": scene_id,
                "split": "held_out" if scene_id in held else "train",
                "strength": "strong" if ratio >= float(cfg["strong_min_registration_ratio"]) else "partial",
                "family": (assets.get(scene_id) or {}).get("family", ""),
                "fuel_name": (assets.get(scene_id) or {}).get("fuel_name", ""),
                "p0_registered": int(p0.num_registered),
                "p4_registered": int(p4.num_registered),
                "p0_registration_ratio": float(p0.registration_ratio),
                "p4_registration_ratio": ratio,
                "p0_sparse_points": int(p0.num_sparse_points),
                "p4_sparse_points": int(p4.num_sparse_points),
                "p4_rotation_error_median": None if pd.isna(p4.rotation_error_median) else float(p4.rotation_error_median),
                "p4_camera_center_error_median": None
                if pd.isna(p4.camera_center_error_median)
                else float(p4.camera_center_error_median),
            }
        )
    rows.sort(key=lambda r: (r["split"] != "train", -r["p4_registration_ratio"], r["id"]))
    return rows


def dense_lookup(path: Path) -> dict:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for row in df.itertuples(index=False):
        out[(row.scene_id, row.pattern_type)] = {
            "chamfer_distance": float(row.chamfer_distance),
            "accuracy": float(row.accuracy),
            "completeness": float(row.completeness),
            "num_fused_points": int(row.num_fused_points),
        }
    return out


def package_one(item: dict, src_root: Path, out_root: Path, x_folder: str, y_folder: str, dense: dict) -> dict:
    src_x = src_root / item["id"] / x_folder
    src_y = src_root / item["id"] / y_folder
    cam_x = json.loads((src_x / "camera_gt.json").read_text(encoding="utf-8"))
    cam_y = json.loads((src_y / "camera_gt.json").read_text(encoding="utf-8"))
    ok, reason = cameras_match(cam_x, cam_y)
    if not ok:
        raise RuntimeError(f"{item['id']}: x/y cameras differ ({reason})")
    rgb_x = sorted((src_x / "rgb").glob("*.png"))
    rgb_y = sorted((src_y / "rgb").glob("*.png"))
    if [p.name for p in rgb_x] != [p.name for p in rgb_y]:
        raise RuntimeError(f"{item['id']}: rgb filenames differ")
    if len(rgb_x) != len(cam_x["frames"]):
        raise RuntimeError(f"{item['id']}: rgb count {len(rgb_x)} != camera frames {len(cam_x['frames'])}")

    dst = out_root / item["split"] / item["id"]
    if dst.exists():
        shutil.rmtree(dst)
    methods = set()
    for src, sub in ((rgb_x, "x"), (rgb_y, "y")):
        for png in src:
            methods.add(link_or_copy(png, dst / sub / png.name))
    for npy in sorted((src_x / "depth").glob("*.npy")):
        methods.add(link_or_copy(npy, dst / "depth" / npy.name))
    for png in sorted((src_x / "mask").glob("*.png")):
        methods.add(link_or_copy(png, dst / "mask" / png.name))

    depth0_x = np.load(src_x / "depth" / rgb_x[0].with_suffix(".npy").name)
    depth0_y = np.load(src_y / "depth" / rgb_y[0].with_suffix(".npy").name)
    depth_diff = float(np.max(np.abs(depth0_x.astype(np.float64) - depth0_y.astype(np.float64))))

    camera = {
        "convention": cam_x["convention"],
        "depth_type": cam_x["depth_type"],
        "width": cam_x["width"],
        "height": cam_x["height"],
        "K": cam_x["K"],
        "frames": [{"id": fr["id"], "file": Path(fr["file"]).name, "T_wc": fr["T_wc"], "T_cw": fr["T_cw"]} for fr in cam_x["frames"]],
    }
    (dst / "camera_gt.json").write_text(json.dumps(camera, indent=2), encoding="utf-8")

    record = {
        **item,
        "num_frames": len(rgb_x),
        "width": int(cam_x["width"]),
        "height": int(cam_x["height"]),
        "alpha": cam_y.get("alpha"),
        "scale": cam_y.get("scale"),
        "seed": cam_y.get("seed"),
        "x": f"{item['split']}/{item['id']}/x",
        "y": f"{item['split']}/{item['id']}/y",
        "depth": f"{item['split']}/{item['id']}/depth",
        "mask": f"{item['split']}/{item['id']}/mask",
        "camera": f"{item['split']}/{item['id']}/camera_gt.json",
        "link_method": sorted(methods)[0] if len(methods) == 1 else sorted(methods),
        "depth_frame0_max_abs_diff": depth_diff,
        "mvs_x": dense.get((item["id"], "P0")),
        "mvs_y": dense.get((item["id"], "P4")),
        "source_x": str(src_x.relative_to(ROOT)).replace("\\", "/"),
        "source_y": str(src_y.relative_to(ROOT)).replace("\\", "/"),
    }
    (dst / "meta.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def parse_args():
    p = argparse.ArgumentParser(description="Build the small P0/P4 pair dataset.")
    p.add_argument("--config", default="configs/pairs_p4.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    cfg = load_yaml(ROOT / args.config)
    exp = load_yaml(ROOT / cfg["source_experiment"])
    patterns = load_configs()["patterns"]
    x_folder = pattern_folder(cfg["x_pattern"], patterns)
    y_folder = pattern_folder(cfg["y_pattern"], patterns)
    sfm = pd.read_csv(ROOT / cfg["sfm_metrics"])
    selected = select_objects(sfm, exp, cfg)
    if not selected:
        raise RuntimeError("no objects passed the pair gate")
    dense = dense_lookup(ROOT / cfg["dense_metrics"]) if cfg.get("dense_metrics") else {}
    out_root = ROOT / cfg["output_root"]
    src_root = ROOT / exp["data_root"]
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    records = []
    for item in selected:
        record = package_one(item, src_root, out_root, x_folder, y_folder, dense)
        records.append(record)
        print(
            f"[pairs] {record['split']:8} {record['strength']:7} {record['id']} "
            f"P0={record['p0_registered']} P4={record['p4_registered']}",
            flush=True,
        )

    manifest = {
        "name": cfg["name"],
        "note": cfg.get("note", ""),
        "x": "original albedo video (P0)",
        "y": "same cameras, albedo plus object-space P4 speckle",
        "projector": "excluded",
        "gate": {
            "p0_max_registered": int(cfg["p0_max_registered"]),
            "p4_min_registered": int(cfg["p4_min_registered"]),
            "strong_min_registration_ratio": float(cfg["strong_min_registration_ratio"]),
            "source_sfm": cfg["sfm_metrics"],
        },
        "num_pairs": len(records),
        "num_train": sum(r["split"] == "train" for r in records),
        "num_held_out": sum(r["split"] == "held_out" for r in records),
        "pairs": records,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    fields = [
        "id",
        "split",
        "strength",
        "family",
        "num_frames",
        "p0_registered",
        "p4_registered",
        "p4_registration_ratio",
        "p0_sparse_points",
        "p4_sparse_points",
        "x",
        "y",
    ]
    with open(out_root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(
        f"[pairs] wrote {out_root}  n={len(records)} "
        f"train={manifest['num_train']} held_out={manifest['num_held_out']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
