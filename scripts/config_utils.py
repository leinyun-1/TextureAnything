from __future__ import annotations

import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

# Data lives outside the repo. Set TA_DATA_ROOT on other machines.
DATA_ROOT = Path(os.environ.get("TA_DATA_ROOT", r"D:\TextureAnything\datasets"))
ABO_ROOT = DATA_ROOT / "abo"
ABO_MODELS = ABO_ROOT / "models"
HDRI_DIR = DATA_ROOT / "hdri"
ABO_QUEUE = ROOT / "manifests" / "abo_queue.json"


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_configs(render="configs/render.yaml", patterns="configs/patterns.yaml", colmap="configs/colmap.yaml"):
    cfgs = {
        "render": load_yaml(ROOT / render),
        "patterns": load_yaml(ROOT / patterns),
        "colmap": load_yaml(ROOT / colmap),
    }
    if os.environ.get("TA_BLENDER"):
        cfgs["render"]["blender_exe"] = os.environ["TA_BLENDER"]
    if os.environ.get("TA_COLMAP"):
        cfgs["colmap"]["colmap_exe"] = os.environ["TA_COLMAP"]
    return cfgs


def pattern_folder(pattern_id: str, patterns_cfg: dict) -> str:
    for item in patterns_cfg["variants"]:
        if item["id"] == pattern_id:
            return item["name"]
    raise KeyError(pattern_id)


def scale_value(patterns_cfg: dict, scale_name: str) -> float:
    return float(patterns_cfg["scales"][scale_name])
