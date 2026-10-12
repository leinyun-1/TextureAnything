from __future__ import annotations

from pathlib import Path

from config_utils import ROOT, load_configs
from eval_sfm import evaluate_scene
from run_colmap_features import process_scene as extract_features
from run_colmap_sfm import process_scene as run_mapper

from pipeline.common import scene_dir, sfm_ready, take
from pipeline.ledger import Ledger


def _metrics(row: dict) -> dict:
    def num(key: str):
        value = row.get(key)
        if value is None:
            return None
        try:
            if value != value:
                return None
        except TypeError:
            return None
        return value

    return {
        "num_registered": int(row.get("num_registered") or 0),
        "registration_ratio": float(row.get("registration_ratio") or 0.0),
        "num_sparse_points": int(row.get("num_sparse_points") or 0),
        "rotation_error_median": num("rotation_error_median"),
        "camera_center_error_median": num("camera_center_error_median"),
    }


def ensure_sfm(scene: Path, colmap_cfg: dict):
    db = scene / "colmap" / "database.db"
    if not (db.exists() and db.stat().st_size > 1024):
        extract_features(colmap_cfg["colmap_exe"], scene, colmap_cfg, int(colmap_cfg.get("use_gpu", 1)))
    if not sfm_ready(scene):
        run_mapper(colmap_cfg["colmap_exe"], scene, colmap_cfg)
    return evaluate_scene(colmap_cfg["colmap_exe"], scene, {})


def sfm_pattern(
    ledger: Ledger,
    cfg: dict,
    pattern_id: str,
    source_stage: str,
    finish,
    limit: int | None,
) -> dict:
    colmap_cfg = load_configs()["colmap"]
    done = 0
    for rec in take(ledger.at(source_stage), limit):
        render_root = ROOT / rec["render_root"]
        scene = scene_dir(render_root, rec["id"], pattern_id)
        row = ensure_sfm(scene, colmap_cfg)
        metrics = _metrics(row)
        rec.setdefault("sfm", {})
        rec["sfm"][pattern_id] = metrics
        finish(rec, metrics)
        ledger.upsert(rec)
        done += 1
        print(
            f"[pipe] sfm {pattern_id} {rec['id']} reg={metrics['num_registered']} -> {rec['stage']}",
            flush=True,
        )
    ledger.save()
    return {"evaluated": done}


def _finish_p0(cfg: dict):
    def finish(rec: dict, metrics: dict):
        if metrics["num_registered"] <= int(cfg["p0_max_registered"]):
            rec["stage"] = "l3_hard"
            rec["reject_reason"] = ""
        else:
            rec["stage"] = "l3_easy"
            rec["reject_reason"] = "p0_already_registers"
    return finish


def _finish_p4(cfg: dict):
    del cfg

    def finish(rec: dict, metrics: dict):
        del metrics
        rec["stage"] = "sfm_p4"
        rec["reject_reason"] = ""

    return finish


def sfm_p0(ledger: Ledger, cfg: dict, limit: int | None = None) -> dict:
    return sfm_pattern(ledger, cfg, cfg["x_pattern"], "rendered_p0", _finish_p0(cfg), limit)


def sfm_p4(ledger: Ledger, cfg: dict, limit: int | None = None) -> dict:
    return sfm_pattern(ledger, cfg, cfg["y_pattern"], "rendered_p4", _finish_p4(cfg), limit)
