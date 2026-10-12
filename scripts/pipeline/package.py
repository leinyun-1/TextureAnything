from __future__ import annotations

import csv
import json

from config_utils import ROOT, load_configs, pattern_folder
from build_pair_dataset import package_one

from pipeline.common import rel, take
from pipeline.ledger import Ledger


def run(ledger: Ledger, cfg: dict, limit: int | None = None) -> dict:
    out_root = ROOT / cfg["pair_root"]
    out_root.mkdir(parents=True, exist_ok=True)
    patterns = load_configs()["patterns"]
    x_folder = pattern_folder(cfg["x_pattern"], patterns)
    y_folder = pattern_folder(cfg["y_pattern"], patterns)
    records = []
    for rec in take(ledger.at("gated"), limit):
        item = {
            "id": rec["id"],
            "split": rec["split"],
            "strength": rec["strength"],
            "family": rec.get("family", ""),
            "fuel_name": rec.get("fuel_name", ""),
            "p0_registered": int(rec["sfm"][cfg["x_pattern"]]["num_registered"]),
            "p4_registered": int(rec["sfm"][cfg["y_pattern"]]["num_registered"]),
            "p0_registration_ratio": float(rec["sfm"][cfg["x_pattern"]]["registration_ratio"]),
            "p4_registration_ratio": float(rec["sfm"][cfg["y_pattern"]]["registration_ratio"]),
            "p0_sparse_points": int(rec["sfm"][cfg["x_pattern"]]["num_sparse_points"]),
            "p4_sparse_points": int(rec["sfm"][cfg["y_pattern"]]["num_sparse_points"]),
            "p4_rotation_error_median": rec["sfm"][cfg["y_pattern"]].get("rotation_error_median"),
            "p4_camera_center_error_median": rec["sfm"][cfg["y_pattern"]].get("camera_center_error_median"),
        }
        record = package_one(item, ROOT / rec["render_root"], out_root, x_folder, y_folder, {})
        records.append(record)
        rec["stage"] = "packaged"
        rec["reject_reason"] = ""
        rec.setdefault("paths", {})
        rec["paths"]["pair"] = f"{rel(out_root)}/{rec['split']}/{rec['id']}"
        ledger.upsert(rec)
        print(f"[pipe] package {rec['split']} {rec['id']}", flush=True)
    ledger.save()
    _write_manifest(out_root, cfg, records)
    return {"packaged": len(records)}


def _write_manifest(out_root, cfg: dict, records: list[dict]):
    previous = []
    manifest_path = out_root / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8")).get("pairs") or []
    by_id = {row["id"]: row for row in previous}
    for row in records:
        by_id[row["id"]] = row
    pairs = sorted(by_id.values(), key=lambda r: (r["split"] != "train", r["id"]))
    manifest = {
        "name": cfg["name"],
        "x": "original albedo video (P0)",
        "y": "same cameras, albedo plus object-space P4 speckle",
        "projector": "excluded",
        "gate": {
            "p0_max_registered": int(cfg["p0_max_registered"]),
            "p4_min_registered": int(cfg["p4_min_registered"]),
            "strong_min_registration_ratio": float(cfg["strong_min_registration_ratio"]),
        },
        "num_pairs": len(pairs),
        "num_train": sum(r["split"] == "train" for r in pairs),
        "num_held_out": sum(r["split"] == "held_out" for r in pairs),
        "pairs": pairs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    fields = [
        "id",
        "split",
        "strength",
        "family",
        "num_frames",
        "p0_registered",
        "p4_registered",
        "p4_registration_ratio",
        "x",
        "y",
    ]
    with open(out_root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(pairs)
