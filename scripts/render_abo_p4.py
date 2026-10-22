"""Render P4 videos for ABO objects that already have a finished P0. Safe to resume.

Same camera trajectory as P0, so each finished object is one (x, y) pair.
No COLMAP gate.
"""
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


def ready(scene: Path) -> bool:
    return (scene / "rgb" / "0023.png").exists() and (scene / "camera_gt.json").exists()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    items = json.loads(QUEUE.read_text(encoding="utf-8"))["items"]
    if args.limit:
        items = items[: args.limit]
    cfgs = load_configs()
    render_args = Namespace(num_views=24, resolution=512, samples=24, alpha=0.4, scale="medium", seed=0)
    done = 0
    skipped = 0
    waiting = 0
    failed = 0
    for item in items:
        p0 = RENDER_ROOT / item["id"] / "P0_plain"
        p4 = RENDER_ROOT / item["id"] / "P4_speckle"
        if ready(p4):
            skipped += 1
            continue
        if not ready(p0):
            waiting += 1
            continue
        try:
            render_variant(
                render_args,
                cfgs["render"],
                cfgs["patterns"],
                item["id"],
                "P4",
                p4,
                mesh=str(ABO_MODELS / f"{item['id']}.glb"),
            )
            done += 1
            print(f"[abo-p4] rendered {item['id']} ({done} new, {skipped} skipped)", flush=True)
        except Exception as exc:
            failed += 1
            print(f"[abo-p4] FAIL {item['id']}: {exc}", flush=True)
            traceback.print_exc()
    print(f"[abo-p4] done new={done} skipped={skipped} waiting_for_p0={waiting} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
