"""Render ABO (x, y) pairs on a pure black background with real HDRI lighting.

Each object gets its own camera trajectory and HDRI rotation. P0 and P4 share
that pair spec. Gray-studio renders are left untouched.

Each video is 832×480 and 81 frames. Safe to resume. A pair spec from an older
resolution is rewritten before that object is rendered again.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from argparse import Namespace
from pathlib import Path

import numpy as np

from config_utils import ABO_MODELS, ABO_QUEUE, ABO_ROOT, HDRI_DIR, ROOT, load_configs
from render_dataset import render_variant

QUEUE = ABO_QUEUE
RENDER_ROOT = ABO_ROOT / "renders_black"

HDRI_NAMES = (
    "hotel_room",
    "wooden_lounge",
    "lythwood_room",
    "lythwood_lounge",
    "comfy_cafe",
    "reading_room",
    "pine_attic",
    "kiara_interior",
    "fireplace",
    "cayley_interior",
    "artist_workshop",
    "veranda",
    "blinds",
    "cabin",
    "warm_restaurant",
    "warm_bar",
    "empty_play_room",
    "small_empty_house",
    "modern_bathroom",
    "paul_lobe_haus",
)

WIDTH = 832
HEIGHT = 480
NUM_VIEWS = 81
SAMPLES = 32


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--queue", type=Path, default=QUEUE)
    p.add_argument("--render-root", type=Path, default=RENDER_ROOT)
    p.add_argument("--hdri-dir", type=Path, default=HDRI_DIR)
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument("--blender", default="")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--ids", nargs="*")
    return p.parse_args()


def ready(scene: Path) -> bool:
    last = scene / "rgb" / f"{NUM_VIEWS - 1:04d}.png"
    cam_path = scene / "camera_gt.json"
    if not last.exists() or not cam_path.exists():
        return False
    cam = json.loads(cam_path.read_text(encoding="utf-8"))
    return (
        int(cam.get("width", 0)) == WIDTH
        and int(cam.get("height", 0)) == HEIGHT
        and len(cam.get("frames", [])) == NUM_VIEWS
    )


def hdri_paths(hdri_dir: Path) -> list[Path]:
    paths = [hdri_dir / f"{name}_2k.hdr" for name in HDRI_NAMES]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing HDRIs in {hdri_dir}: {missing}")
    return paths


def resolve_glb(item: dict, models_dir: Path) -> str:
    candidate = models_dir / f"{item['id']}.glb"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"missing glb for {item['id']}: {candidate} (run scripts/fetch_data.py abo)")


def resolve_hdri(spec: dict, hdri_dir: Path) -> str:
    stored = Path(spec["hdri"])
    if stored.exists():
        return str(stored)
    local = hdri_dir / stored.name
    if local.exists():
        return str(local)
    raise FileNotFoundError(f"missing hdri {stored.name} in {hdri_dir}")


def sample_pair(object_id: str, hdris: list[Path]) -> dict:
    seed = int(hashlib.sha256(object_id.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    lens = float(rng.uniform(34.0, 52.0))
    frame_fill = float(rng.uniform(0.80, 0.90))
    return {
        "id": object_id,
        "lens_mm": round(lens, 2),
        "radius": 3.0,
        "frame_fill": round(frame_fill, 3),
        "framing": "longest_edge_vs_height_0.8_0.9",
        "azimuth_span_deg": round(float(rng.uniform(140.0, 300.0)), 2),
        "azimuth_start_deg": round(float(rng.uniform(0.0, 360.0)), 2),
        "elevation_deg": round(float(rng.uniform(10.0, 32.0)), 2),
        "elevation_amp_deg": round(float(rng.uniform(4.0, 12.0)), 2),
        "hdri": str(hdris[int(rng.integers(0, len(hdris)))]),
        "hdri_rotation_deg": round(float(rng.uniform(0.0, 360.0)), 2),
        "hdri_strength": 1.0,
        "resolution_x": WIDTH,
        "resolution_y": HEIGHT,
        "num_views": NUM_VIEWS,
        "samples": SAMPLES,
        "denoise": True,
        "background": "black",
    }


def spec_current(spec: dict) -> bool:
    return (
        int(spec.get("resolution_x", 0)) == WIDTH
        and int(spec.get("resolution_y", 0)) == HEIGHT
        and int(spec.get("num_views", 0)) == NUM_VIEWS
        and spec.get("framing") == "longest_edge_vs_height_0.8_0.9"
    )


def load_or_make_spec(render_root: Path, object_id: str, hdris: list[Path]) -> dict:
    path = render_root / object_id / "pair.json"
    if path.exists():
        spec = json.loads(path.read_text(encoding="utf-8"))
        if spec_current(spec):
            return spec
    spec = sample_pair(object_id, hdris)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return spec


def main():
    args = parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index must be in [0, --shard-count)")
    sys.path.insert(0, str(ROOT / "scripts"))
    queue = args.queue
    render_root = args.render_root
    hdri_dir = args.hdri_dir
    models_dir = args.models_dir or ABO_MODELS
    items = json.loads(queue.read_text(encoding="utf-8"))["items"]
    if args.ids:
        wanted = set(args.ids)
        items = [item for item in items if item["id"] in wanted]
    items = [item for i, item in enumerate(items) if i % args.shard_count == args.shard_index]
    if args.limit:
        items = items[: args.limit]
    print(
        f"[abo-black] shard {args.shard_index}/{args.shard_count} objects={len(items)} root={render_root}",
        flush=True,
    )
    hdris = hdri_paths(hdri_dir)
    cfgs = load_configs()
    if args.blender:
        cfgs["render"]["blender_exe"] = args.blender
    render_args = Namespace(
        num_views=NUM_VIEWS,
        resolution=WIDTH,
        resolution_x=WIDTH,
        resolution_y=HEIGHT,
        samples=SAMPLES,
        alpha=0.4,
        scale="medium",
        seed=0,
    )
    done = 0
    skipped = 0
    failed = 0
    for item in items:
        root = render_root / item["id"]
        p0 = root / "P0_plain"
        p4 = root / "P4_speckle"
        if ready(p0) and ready(p4):
            skipped += 1
            continue
        spec = load_or_make_spec(render_root, item["id"], hdris)
        spec["hdri"] = resolve_hdri(spec, hdri_dir)
        glb = resolve_glb(item, models_dir)
        extra = {
            "lens_mm": spec["lens_mm"],
            "radius": spec["radius"],
            "azimuth_span_deg": spec["azimuth_span_deg"],
            "azimuth_start_deg": spec["azimuth_start_deg"],
            "elevation_deg": spec["elevation_deg"],
            "elevation_amp_deg": spec["elevation_amp_deg"],
            "hdri": spec["hdri"],
            "hdri_rotation_deg": spec["hdri_rotation_deg"],
            "hdri_strength": spec["hdri_strength"],
            "denoise": True,
            "rgb_only": True,
            "engine": "BLENDER_EEVEE",
            "frame_fill": spec["frame_fill"],
            "resolution_x": WIDTH,
            "resolution_y": HEIGHT,
        }
        try:
            if not ready(p0):
                render_variant(
                    render_args,
                    cfgs["render"],
                    cfgs["patterns"],
                    item["id"],
                    "P0",
                    p0,
                    mesh=glb,
                    extra=extra,
                )
            if not ready(p4):
                render_variant(
                    render_args,
                    cfgs["render"],
                    cfgs["patterns"],
                    item["id"],
                    "P4",
                    p4,
                    mesh=glb,
                    extra=extra,
                )
            done += 1
            print(f"[abo-black] pair {item['id']} ({done} new, {skipped} skipped)", flush=True)
        except Exception as exc:
            failed += 1
            print(f"[abo-black] FAIL {item['id']}: {exc}", flush=True)
            traceback.print_exc()
    print(f"[abo-black] done new={done} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
