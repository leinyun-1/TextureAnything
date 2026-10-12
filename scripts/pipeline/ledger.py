from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Ledger:
    def __init__(self, path: Path):
        self.path = path
        self.records: dict[str, dict] = {}
        if path.exists():
            self.records = json.loads(path.read_text(encoding="utf-8"))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.records, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def upsert(self, rec: dict):
        rec["updated"] = now()
        self.records[rec["id"]] = rec

    def by_fuel(self, fuel_name: str) -> dict | None:
        for rec in self.records.values():
            if rec.get("fuel_name") == fuel_name:
                return rec
        return None

    def at(self, stage: str) -> list[dict]:
        return sorted((r for r in self.records.values() if r.get("stage") == stage), key=lambda r: r["id"])

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rec in self.records.values():
            stage = rec.get("stage", "unknown")
            counts[stage] = counts.get(stage, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
