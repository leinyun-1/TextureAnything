"""Extract ABO models that passed the albedo screen and are rigid household objects."""
from __future__ import annotations

import gzip
import json
import tarfile
from pathlib import Path

SCORES = Path(r"D:\TextureAnything\datasets\abo\albedo_scores.jsonl")
LISTINGS = Path(r"D:\TextureAnything\datasets\abo\listings\metadata")
ARCHIVE = Path(r"D:\TextureAnything\datasets\abo\abo-3dmodels.tar")
MODEL_ROOT = Path(r"D:\TextureAnything\datasets\abo\models")
QUEUE = Path(r"D:\TextureAnything\datasets\abo\p0_queue.json")

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
            keep[item_id]["glb"] = str(dest)
            if extracted and extracted % 100 == 0:
                print(f"[abo] extracted {extracted}", flush=True)
    missing = [item_id for item_id, row in keep.items() if "glb" not in row]
    queue = [row for row in keep.values() if "glb" in row]
    queue.sort(key=lambda row: row.get("weakness", 99))
    QUEUE.write_text(json.dumps({"count": len(queue), "missing": missing, "items": queue}, indent=2), encoding="utf-8")
    print(f"[abo] queue={len(queue)} missing={len(missing)} -> {QUEUE}", flush=True)


if __name__ == "__main__":
    main()
