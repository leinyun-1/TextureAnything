from __future__ import annotations

from config_utils import ROOT
from download_gso import download_object, strip_obj_materials

from pipeline.common import rel, take
from pipeline.ledger import Ledger


def run(ledger: Ledger, cfg: dict, limit: int | None = None) -> dict:
    out_dir = ROOT / cfg["asset_root"]
    cache = ROOT / cfg["texture_cache"]
    done = 0
    for rec in take(ledger.at("l2_pass"), limit):
        meta = download_object(rec["id"], rec["fuel_name"], out_dir)
        mesh = out_dir / rec["id"] / "model.obj"
        tex = out_dir / rec["id"] / "texture.png"
        cached = cache / f"{rec['fuel_name']}.png"
        if cached.exists() and (not tex.exists() or tex.stat().st_size < 64):
            tex.write_bytes(cached.read_bytes())
        if mesh.exists():
            strip_obj_materials(mesh)
        if not mesh.exists() or not tex.exists():
            rec["stage"] = "rejected"
            rec["reject_reason"] = "acquire_missing_files"
        else:
            rec["stage"] = "acquired"
            rec["reject_reason"] = ""
            rec["paths"] = {
                "mesh": rel(mesh),
                "texture": rel(tex),
                "source": meta.get("downloaded_from", ""),
            }
            done += 1
        ledger.upsert(rec)
        print(f"[pipe] acquire {rec['id']} {rec['stage']}", flush=True)
    ledger.save()
    return {"acquired": done}
