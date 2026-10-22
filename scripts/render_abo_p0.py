"""Render P0 orbit videos for the ABO queue. Safe to resume."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from argparse import Namespace
from pathlib import Path

from config_utils import ABO_MODELS, ABO_QUEUE, ABO_ROOT, ROOT, load_configs
from render_dataset import render_variant

QUEUE = ABO_QUEUE
RENDER_ROOT = ABO_ROOT / "renders"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    doc = json.loads(QUEUE.read_text(encoding="utf-8"))
    items = doc["items"]
    if args.limit:
        items = items[: args.limit]
    cfgs = load_configs()
    render_args = Namespace(num_views=24, resolution=512, samples=24, alpha=0.4, scale="medium", seed=0)
    done = 0
    skipped = 0
    failed = 0
    for item in items:
        out_dir = RENDER_ROOT / item["id"] / "P0_plain"
        if (out_dir / "rgb" / "0023.png").exists() and (out_dir / "camera_gt.json").exists():
            skipped += 1
            continue
        try:
            render_variant(
                render_args,
                cfgs["render"],
                cfgs["patterns"],
                item["id"],
                "P0",
                out_dir,
                mesh=str(ABO_MODELS / f"{item['id']}.glb"),
            )
            done += 1
            print(f"[abo-p0] rendered {item['id']} ({done} new, {skipped} skipped)", flush=True)
        except Exception as exc:
            failed += 1
            print(f"[abo-p0] FAIL {item['id']}: {exc}", flush=True)
            traceback.print_exc()
    print(f"[abo-p0] done new={done} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
