"""Render one indoor (x, y) pair with a sampled orbit."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

BLENDER = r"D:\blender\blender-5.2.2-windows-x64\blender.exe"
PLACE = r"D:\TextureAnything\indoors\place_pair.py"


def sample_orbit(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "radius": float(rng.uniform(1.8, 3.2)),
        "elevation_deg": float(rng.uniform(8.0, 28.0)),
        "elevation_amp_deg": float(rng.uniform(4.0, 12.0)),
        "arc_deg": float(rng.uniform(120.0, 300.0)),
        "start_deg": float(rng.uniform(0.0, 360.0)),
    }


def render_pair(blend: Path, glb: Path, out_root: Path, seed: int, object_extent: float = 0.28):
    cfg = {
        "glb": str(glb),
        "out_root": str(out_root),
        "object_extent": object_extent,
        **sample_orbit(seed),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    cfg_path = out_root / "orbit.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    for pattern in ("P0", "P4"):
        if (out_root / pattern / "0023.png").exists():
            print(f"[indoor] skip {pattern} {out_root}", flush=True)
            continue
        cmd = [
            BLENDER,
            "-b",
            str(blend),
            "--python-exit-code",
            "1",
            "--python",
            PLACE,
            "--",
            pattern,
            "24",
            str(cfg_path),
        ]
        print("[indoor]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    print("[indoor] pair", out_root, cfg, flush=True)


if __name__ == "__main__":
    render_pair(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]))
