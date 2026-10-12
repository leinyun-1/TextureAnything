from __future__ import annotations

import argparse
import json
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

from config_utils import ROOT, load_yaml

FUEL_BASE = "https://fuel.gazebosim.org/1.0/GoogleResearch/models"
GITHUB_RAW = "https://raw.githubusercontent.com/kevinzakka/mujoco_scanned_objects/main/models"
USER_AGENT = "TextureAnything/gso-downloader"


def fetch(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
    )
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
        data = resp.read()
    if len(data) < 64:
        raise RuntimeError(f"too small ({len(data)} bytes): {url}")
    dest.write_bytes(data)
    return len(data)


def strip_obj_materials(obj_path: Path):
    text = obj_path.read_text(encoding="utf-8", errors="ignore")
    lines = []
    for ln in text.splitlines():
        if ln.startswith("mtllib") or ln.startswith("usemtl"):
            lines.append("# " + ln)
        else:
            lines.append(ln)
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def download_object(obj_id: str, fuel_name: str, out_dir: Path) -> dict:
    obj_dir = out_dir / obj_id
    mesh = obj_dir / "model.obj"
    tex = obj_dir / "texture.png"
    meta = {
        "id": obj_id,
        "fuel_name": fuel_name,
        "source": "Google Scanned Objects",
        "owner": "GoogleResearch",
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Downs, L. et al. Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household Items. ICRA 2022.",
    }
    if mesh.exists() and tex.exists() and mesh.stat().st_size > 64 and tex.stat().st_size > 64:
        print(f"[gso] skip existing {obj_id}", flush=True)
        strip_obj_materials(mesh)
        (obj_dir / "source.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    fuel_mesh = f"{FUEL_BASE}/{fuel_name}/tip/files/meshes/model.obj"
    fuel_tex = f"{FUEL_BASE}/{fuel_name}/tip/files/materials/textures/texture.png"
    gh_mesh = f"{GITHUB_RAW}/{fuel_name}/model.obj"
    gh_tex = f"{GITHUB_RAW}/{fuel_name}/texture.png"
    try:
        n_m = fetch(fuel_mesh, mesh)
        n_t = fetch(fuel_tex, tex)
        meta["downloaded_from"] = "gazebo_fuel"
        print(f"[gso] fuel {obj_id} mesh={n_m} tex={n_t}", flush=True)
    except (urllib.error.URLError, RuntimeError, TimeoutError) as exc:
        print(f"[gso] fuel failed for {obj_id}: {exc}; trying GitHub mirror", flush=True)
        n_m = fetch(gh_mesh, mesh)
        n_t = fetch(gh_tex, tex)
        meta["downloaded_from"] = "kevinzakka/mujoco_scanned_objects"
        print(f"[gso] github {obj_id} mesh={n_m} tex={n_t}", flush=True)
    strip_obj_materials(mesh)
    (obj_dir / "source.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def score_texture(path: Path) -> dict:
    import cv2
    import numpy as np

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "unreadable"}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "mean_luma": float(gray.mean() / 255.0),
        "std_luma": float(gray.std() / 255.0),
        "mean_grad": float(mag.mean()),
        "p90_grad": float(np.percentile(mag, 90)),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Download weakly textured GSO meshes + albedo maps.")
    p.add_argument("--experiment", default="configs/gso_p4.yaml")
    p.add_argument("--out_dir", default="data/gso_assets")
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    exp = load_yaml(ROOT / args.experiment)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = exp["assets"]
    rows = []
    for obj_id in exp["objects"]:
        item = assets[obj_id]
        meta = download_object(obj_id, item["fuel_name"], out_dir)
        tex = out_dir / obj_id / "texture.png"
        scores = score_texture(tex)
        meta.update(scores)
        (out_dir / obj_id / "texture_stats.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
        rows.append(meta)
        print(f"[gso] {obj_id} std_luma={scores.get('std_luma')} mean_grad={scores.get('mean_grad')}", flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    readme = [
        "# Google Scanned Objects subset",
        "",
        "Downloaded from Gazebo Fuel (`GoogleResearch`) for Virtual Texture Oracle experiments.",
        "License: Creative Commons Attribution 4.0 International (CC-BY-4.0).",
        "Citation: Downs, L. et al., Google Scanned Objects, ICRA 2022.",
        "",
        "Objects were chosen as household items with existing diffuse texture maps that are",
        "visually weak (near-uniform porcelain / glaze). P0 renders the original albedo;",
        "P4 adds an object-space speckle overlay A' = clip(A + αF, 0, 1).",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print("[gso] wrote", out_dir / "manifest.json")


if __name__ == "__main__":
    main()
