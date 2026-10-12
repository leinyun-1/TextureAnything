from __future__ import annotations

from select_gso_weak import prefilter

from pipeline.common import take
from pipeline.ledger import Ledger


def run(ledger: Ledger, cfg: dict, limit: int | None = None) -> dict:
    del cfg
    rows = take(ledger.at("catalog"), limit)
    passed = 0
    rejected = 0
    if not rows:
        ledger.save()
        return {"passed": 0, "rejected": 0}
    models = [{"name": r["fuel_name"], "description": r.get("description") or ""} for r in rows]
    kept = {m["name"] for m in prefilter(models)}
    for rec in rows:
        if rec["fuel_name"] in kept:
            rec["stage"] = "l1_pass"
            rec["reject_reason"] = ""
            passed += 1
        else:
            rec["stage"] = "rejected"
            rec["reject_reason"] = "l1_keyword"
            rejected += 1
        ledger.upsert(rec)
        print(f"[pipe] l1 {rec['id']} {rec['stage']}", flush=True)
    ledger.save()
    return {"passed": passed, "rejected": rejected}
