"""Extract ABO models that passed the albedo screen and are rigid household objects.

Rebuilds manifests/abo_queue.json from the full archive. Other machines only need
scripts/fetch_data.py, which downloads the models already listed there.
"""
from __future__ import annotations

import gzip
import json
import tarfile
from pathlib import Path

from config_utils import ABO_MODELS, ABO_QUEUE, ABO_ROOT

SCORES = ABO_ROOT / "albedo_scores.jsonl"
LISTINGS = ABO_ROOT / "listings" / "metadata"
ARCHIVE = ABO_ROOT / "abo-3dmodels.tar"
MODEL_ROOT = ABO_MODELS
QUEUE = ABO_QUEUE

DENY_TYPES = {
    "PILLOW",
    "FURNITURE_COVER",
    "ELECTRONIC_CABLE",
    "BUILDING_MATERIAL",
    "MULTIPORT_HUB",
    "SPORTING_GOODS",
    "AUTO_ACCESSORY",
    "TOOLS",
    "AIR_CONDITIONER",
    "FLAT_SCREEN_DISPLAY_MOUNT",
    "COMPUTER_ADD_ON",
    "SOUND_AND_RECORDING_EQUIPMENT",
    "OFFICE_PRODUCTS",
    "FREESTANDING_SHELTER",
}


def passed_rows() -> dict[str, dict]:
    rows = {}
    for line in SCORES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "pass":
            rows[row["id"]] = row
    return rows


def product_types(want: set[str]) -> dict[str, str]:
    found = {}
    for path in LISTINGS.glob("*.json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                item_id = row.get("item_id")
                if item_id not in want or item_id in found:
                    continue
                pt = row.get("product_type") or []
                found[item_id] = pt[0].get("value", "") if pt else ""
    return found


def main():
    rows = passed_rows()
    types = product_types(set(rows))
    keep = {}
    denied = 0
    for item_id, row in rows.items():
        product = types.get(item_id, "")
        if product in DENY_TYPES:
            denied += 1
            continue
        keep[item_id] = {**row, "product_type": product}
    print(f"[abo] albedo_pass={len(rows)} denied_type={denied} queue={len(keep)}", flush=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with tarfile.open(ARCHIVE, "r|") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".glb"):
                continue
            item_id = Path(member.name).stem
            if item_id not in keep:
                continue
            dest = MODEL_ROOT / f"{item_id}.glb"
            if not dest.exists() or dest.stat().st_size < 64:
                src = tar.extractfile(member)
                dest.write_bytes(src.read())
                extracted += 1
            keep[item_id]["member"] = member.name
            if extracted and extracted % 100 == 0:
                print(f"[abo] extracted {extracted}", flush=True)
    missing = [item_id for item_id, row in keep.items() if "member" not in row]
    queue = [row for row in keep.values() if "member" in row]
    queue.sort(key=lambda row: row.get("weakness", 99))
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE.write_text(
        json.dumps(
            {
                "source": "ABO 3D models, CC-BY-4.0",
                "url_base": "https://amazon-berkeley-objects.s3.amazonaws.com/",
                "count": len(queue),
                "missing": missing,
                "items": queue,
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[abo] queue={len(queue)} missing={len(missing)} -> {QUEUE}", flush=True)


if __name__ == "__main__":
    main()
