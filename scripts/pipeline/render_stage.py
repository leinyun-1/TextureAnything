from __future__ import annotations

import argparse

from config_utils import ROOT, load_configs
from render_dataset import render_variant

from pipeline.common import rel, render_ready, scene_dir, take
from pipeline.ledger import Ledger


def _args(cfg: dict) -> argparse.Namespace:
    return argparse.Namespace(
        num_views=int(cfg["num_views"]),
        resolution=int(cfg["resolution"]),
        samples=int(cfg["samples"]),
        alpha=float(cfg["alpha"]),
        scale=cfg["scale"],
        seed=int(cfg["seed"]),
    )


def render_pattern(ledger: Ledger, cfg: dict, pattern_id: str, source_stage: str, done_stage: str, limit: int | None) -> dict:
    cfgs = load_configs()
    args = _args(cfg)
    rendered = 0
    skipped = 0
    for rec in take(ledger.at(source_stage), limit):
        render_root = ROOT / rec.get("render_root", cfg["render_root"])
        mesh = ROOT / rec["paths"]["mesh"]
        texture = ROOT / rec["paths"]["texture"]
        out_dir = scene_dir(render_root, rec["id"], pattern_id)
        if render_ready(out_dir):
            skipped += 1
            print(f"[pipe] render {pattern_id} skip {rec['id']}", flush=True)
        else:
            if out_dir.exists():
                import shutil

                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            render_variant(
                args,
                cfgs["render"],
                cfgs["patterns"],
                rec["id"],
                pattern_id,
                out_dir,
                mesh=str(mesh),
                texture=str(texture),
            )
            rendered += 1
            print(f"[pipe] render {pattern_id} {rec['id']}", flush=True)
        if not render_ready(out_dir):
            rec["stage"] = "rejected"
            rec["reject_reason"] = f"render_{pattern_id}_failed"
        else:
            rec["stage"] = done_stage
            rec["reject_reason"] = ""
            rec["render_root"] = rel(render_root)
            rec.setdefault("paths", {})
            rec["paths"][pattern_id] = rel(out_dir)
        ledger.upsert(rec)
    ledger.save()
    return {"rendered": rendered, "skipped": skipped}


def render_p0(ledger: Ledger, cfg: dict, limit: int | None = None) -> dict:
    return render_pattern(ledger, cfg, cfg["x_pattern"], "acquired", "rendered_p0", limit)


def render_p4(ledger: Ledger, cfg: dict, limit: int | None = None) -> dict:
    return render_pattern(ledger, cfg, cfg["y_pattern"], "l3_hard", "rendered_p4", limit)
