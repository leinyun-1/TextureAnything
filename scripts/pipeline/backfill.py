"""Load the already-rendered GSO scale run into the ledger.

This does not download or render. It marks each existing object at the furthest
stage the files and SfM tables already support.
"""
from __future__ import annotations

import pandas as pd

from config_utils import ROOT, load_yaml
from select_gso_weak import family_of

from pipeline.common import rel, render_ready, scene_dir, sfm_ready
from pipeline.ledger import Ledger


def _sfm_table(path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for row in df.itertuples(index=False):
        out[(row.scene_id, row.pattern_type)] = {
            "num_registered": int(row.num_registered),
            "registration_ratio": float(row.registration_ratio),
            "num_sparse_points": int(row.num_sparse_points),
            "rotation_error_median": None if pd.isna(row.rotation_error_median) else float(row.rotation_error_median),
            "camera_center_error_median": None
            if pd.isna(row.camera_center_error_median)
            else float(row.camera_center_error_median),
        }
    return out


def run(ledger: Ledger, cfg: dict, limit: int | None = None) -> dict:
    exp = load_yaml(ROOT / cfg["backfill_experiment"])
    table = _sfm_table(ROOT / cfg["backfill_sfm"])
    render_root = ROOT / cfg["backfill_render_root"]
    x_pat = cfg["x_pattern"]
    y_pat = cfg["y_pattern"]
    objects = list(exp["objects"])
    if limit:
        objects = objects[:limit]
    n = 0
    for obj_id in objects:
        asset = exp["assets"][obj_id]
        rec = ledger.records.get(obj_id) or {
            "id": obj_id,
            "fuel_name": asset["fuel_name"],
            "source": "gso_fuel",
            "description": "",
        }
        rec["family"] = asset.get("family") or family_of(asset["fuel_name"])
        rec["l2"] = {"weakness": asset.get("weakness")}
        rec["paths"] = {
            "mesh": asset["mesh"],
            "texture": asset["texture"],
        }
        rec["render_root"] = rel(render_root)
        p0_dir = scene_dir(render_root, obj_id, x_pat)
        p4_dir = scene_dir(render_root, obj_id, y_pat)
        if render_ready(p0_dir):
            rec["paths"][x_pat] = rel(p0_dir)
        if render_ready(p4_dir):
            rec["paths"][y_pat] = rel(p4_dir)
        p0 = table.get((obj_id, x_pat))
        p4 = table.get((obj_id, y_pat))
        rec["sfm"] = {}
        if p0:
            rec["sfm"][x_pat] = p0
        if p4:
            rec["sfm"][y_pat] = p4

        if p0 and p0["num_registered"] <= int(cfg["p0_max_registered"]) and p4:
            registered = int(p4["num_registered"])
            ratio = float(p4["registration_ratio"])
            if registered >= int(cfg["p4_min_registered"]):
                rec["stage"] = "gated"
                rec["reject_reason"] = ""
                rec["strength"] = "strong" if ratio >= float(cfg["strong_min_registration_ratio"]) else "partial"
                held = set(exp.get("held_out") or [])
                rec["split"] = "held_out" if obj_id in held else "train"
            else:
                rec["stage"] = "gate_fail"
                rec["reject_reason"] = "p4_did_not_lift"
        elif p0 and p0["num_registered"] > int(cfg["p0_max_registered"]):
            rec["stage"] = "l3_easy"
            rec["reject_reason"] = "p0_already_registers"
        elif render_ready(p4_dir) and sfm_ready(p0_dir):
            rec["stage"] = "rendered_p4"
            rec["reject_reason"] = ""
        elif render_ready(p0_dir):
            rec["stage"] = "rendered_p0"
            rec["reject_reason"] = ""
        elif (ROOT / asset["mesh"]).exists() and (ROOT / asset["texture"]).exists():
            rec["stage"] = "acquired"
            rec["reject_reason"] = ""
        else:
            rec["stage"] = "l2_pass"
            rec["reject_reason"] = ""
        ledger.upsert(rec)
        n += 1
        print(f"[pipe] backfill {obj_id} {rec['stage']}", flush=True)
    ledger.save()
    return {"backfilled": n}
