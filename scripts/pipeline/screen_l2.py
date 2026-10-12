from __future__ import annotations

from pathlib import Path

from config_utils import ROOT
from select_gso_weak import ensure_texture, score_texture_used, weakness_score

from pipeline.common import take
from pipeline.ledger import Ledger


def run(ledger: Ledger, cfg: dict, limit: int | None = None) -> dict:
    cache = ROOT / cfg["texture_cache"]
    passed = 0
    rejected = 0
    for rec in take(ledger.at("l1_pass"), limit):
        dest = cache / f"{rec['fuel_name']}.png"
        ok = ensure_texture(rec["fuel_name"], dest)
        stats = score_texture_used(dest) if ok else None
        if not stats:
            rec["stage"] = "rejected"
            rec["reject_reason"] = "l2_texture_unreadable"
            rejected += 1
        else:
            score = weakness_score(stats)
            stats = {**stats, "weakness": score}
            rec["l2"] = stats
            if stats["mean_sat_used"] > float(cfg["l2_max_sat"]) or stats["mean_grad_used"] > float(cfg["l2_max_grad"]):
                rec["stage"] = "rejected"
                rec["reject_reason"] = "l2_too_textured"
                rejected += 1
            else:
                rec["stage"] = "l2_pass"
                rec["reject_reason"] = ""
                passed += 1
        ledger.upsert(rec)
        print(f"[pipe] l2 {rec['id']} {rec['stage']} {rec.get('reject_reason', '')}", flush=True)
    ledger.save()
    return {"passed": passed, "rejected": rejected}
