from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_configs(render="configs/render.yaml", patterns="configs/patterns.yaml", colmap="configs/colmap.yaml"):
    return {
        "render": load_yaml(ROOT / render),
        "patterns": load_yaml(ROOT / patterns),
        "colmap": load_yaml(ROOT / colmap),
    }


def pattern_folder(pattern_id: str, patterns_cfg: dict) -> str:
    for item in patterns_cfg["variants"]:
        if item["id"] == pattern_id:
            return item["name"]
    raise KeyError(pattern_id)


def scale_value(patterns_cfg: dict, scale_name: str) -> float:
    return float(patterns_cfg["scales"][scale_name])
