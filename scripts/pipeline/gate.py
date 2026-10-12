from __future__ import annotations

from pipeline.common import split_for, take
from pipeline.ledger import Ledger


def run(ledger: Ledger, cfg: dict, limit: int | None = None) -> dict:
    passed = 0
    failed = 0
    y_pat = cfg["y_pattern"]
    minimum = int(cfg["p4_min_registered"])
    strong_at = float(cfg["strong_min_registration_ratio"])
    for rec in take(ledger.at("sfm_p4"), limit):
        metrics = (rec.get("sfm") or {}).get(y_pat) or {}
        registered = int(metrics.get("num_registered") or 0)
        ratio = float(metrics.get("registration_ratio") or 0.0)
        if registered >= minimum:
            rec["stage"] = "gated"
            rec["reject_reason"] = ""
            rec["strength"] = "strong" if ratio >= strong_at else "partial"
            rec["split"] = split_for(rec["id"], float(cfg["held_out_fraction"]), int(cfg["held_out_seed"]))
            passed += 1
        else:
            rec["stage"] = "gate_fail"
            rec["reject_reason"] = "p4_did_not_lift"
            failed += 1
        ledger.upsert(rec)
        print(f"[pipe] gate {rec['id']} {rec['stage']} P4={registered}", flush=True)
    ledger.save()
    return {"passed": passed, "failed": failed}
