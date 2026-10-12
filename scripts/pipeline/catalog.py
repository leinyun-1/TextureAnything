from __future__ import annotations

from select_gso_weak import family_of, list_gso_models, slug

from pipeline.common import take
from pipeline.ledger import Ledger


def run(ledger: Ledger, cfg: dict, limit: int | None = None) -> dict:
    models = list_gso_models()
    used = {rec["id"] for rec in ledger.records.values()}
    fresh = [m for m in models if ledger.by_fuel(m["name"]) is None]
    added = 0
    for model in take(fresh, limit):
        obj_id = slug(model["name"], used)
        ledger.upsert(
            {
                "id": obj_id,
                "fuel_name": model["name"],
                "source": cfg.get("source", "gso_fuel"),
                "family": family_of(model["name"]),
                "description": model.get("description") or "",
                "stage": "catalog",
                "reject_reason": "",
            }
        )
        added += 1
        print(f"[pipe] catalog {obj_id} <= {model['name']}", flush=True)
    ledger.save()
    return {"added": added, "catalog_size": len(models), "already_known": len(models) - len(fresh)}
