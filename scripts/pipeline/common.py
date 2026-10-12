from __future__ import annotations

import hashlib
from pathlib import Path

from config_utils import ROOT, load_configs, load_yaml, pattern_folder


def load_pipeline(path: str | Path = "configs/pipeline.yaml") -> dict:
    return load_yaml(ROOT / path)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def scene_dir(render_root: Path, obj_id: str, pattern_id: str) -> Path:
    folder = pattern_folder(pattern_id, load_configs()["patterns"])
    return render_root / obj_id / folder


def render_ready(scene: Path) -> bool:
    return (scene / "rgb" / "0000.png").exists() and (scene / "camera_gt.json").exists()


def sfm_ready(scene: Path) -> bool:
    return (scene / "colmap" / "mapper.json").exists()


def split_for(obj_id: str, fraction: float, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{obj_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % 10000
    return "held_out" if bucket < int(round(fraction * 10000)) else "train"


def take(rows: list[dict], limit: int | None) -> list[dict]:
    if limit is None or limit <= 0:
        return rows
    return rows[:limit]
