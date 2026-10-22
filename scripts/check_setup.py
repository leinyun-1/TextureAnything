"""Check that Python packages, Blender, COLMAP and the data are where the scripts expect."""
from __future__ import annotations

import importlib
import json
import subprocess

from config_utils import ABO_MODELS, ABO_QUEUE, DATA_ROOT, HDRI_DIR, load_configs

PACKAGES = {"numpy": "numpy", "cv2": "opencv-python", "yaml": "pyyaml", "imageio_ffmpeg": "imageio-ffmpeg"}


def first_line(cmd: list[str]) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    lines = (out.stdout or out.stderr).strip().splitlines()
    return lines[0] if lines else f"exit {out.returncode}"


def main():
    problems = []
    for module, pip_name in PACKAGES.items():
        try:
            importlib.import_module(module)
            print(f"[ok]   python {pip_name}")
        except ImportError:
            problems.append(f"pip install {pip_name}")
            print(f"[miss] python {pip_name}")

    cfgs = load_configs()
    for label, cmd in (
        ("blender", [cfgs["render"]["blender_exe"], "--version"]),
        ("colmap", [cfgs["colmap"]["colmap_exe"], "-h"]),
    ):
        try:
            print(f"[ok]   {label}: {first_line(cmd)}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            problems.append(f"{label} not runnable at {cmd[0]} (set TA_{label.upper()})")
            print(f"[miss] {label}: {cmd[0]} ({exc})")

    print(f"[info] TA_DATA_ROOT={DATA_ROOT}")
    items = json.loads(ABO_QUEUE.read_text(encoding="utf-8"))["items"]
    have = sum((ABO_MODELS / f"{item['id']}.glb").exists() for item in items)
    status = "ok" if have == len(items) else "miss"
    print(f"[{status:4s}] ABO models {have}/{len(items)} in {ABO_MODELS}")
    if have < len(items):
        problems.append("python scripts/fetch_data.py abo")

    from render_abo_black import HDRI_NAMES

    hdris = sum((HDRI_DIR / f"{name}_2k.hdr").exists() for name in HDRI_NAMES)
    status = "ok" if hdris == len(HDRI_NAMES) else "miss"
    print(f"[{status:4s}] HDRI {hdris}/{len(HDRI_NAMES)} in {HDRI_DIR}")
    if hdris < len(HDRI_NAMES):
        problems.append("python scripts/fetch_data.py hdri")

    if problems:
        print("\nTo fix:")
        for p in dict.fromkeys(problems):
            print("  " + p)
        raise SystemExit(1)
    print("\nAll set.")


if __name__ == "__main__":
    main()
