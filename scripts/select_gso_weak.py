from __future__ import annotations

import argparse
import json
import re
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import yaml

from config_utils import ROOT, load_yaml
from download_gso import FUEL_BASE, USER_AGENT, download_object, strip_obj_materials

CTX = ssl.create_default_context()

EXCLUDE_PILOT = {
    "Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White",
    "Threshold_Porcelain_Serving_Bowl_Coupe_White",
    "Threshold_Dinner_Plate_Square_Rim_White_Porcelain",
    "Threshold_Porcelain_Pitcher_White",
    "Threshold_Porcelain_Teapot_White",
}

INCLUDE_RE = re.compile(
    r"porcelain|ceramic|white|bowl|plate|mug|cup|pitcher|teapot|ramekin|"
    r"utensil|tray|bin|canister|container|planter|vase|colander|dish|saucer|"
    r"spoon|bucket|crock|casserole|salad|dinner|serving|mortar|mixing|"
    r"plastic|lid|holder|jar|tumbler|beaker|basin|sink|pot\b|pan\b",
    re.I,
)
EXCLUDE_RE = re.compile(
    r"shark|turtle|truck|robot|lego|crayola|cereal|chocolate|shoe|clock|"
    r"keyboard|gaffer|vtech|pokemon|nintendo|barbie|dinosaur|unicorn|"
    r"superhero|floral|camouflage|\bcamo\b|striped|stuffed|plush|animal|"
    r"puppy|kitten|bear\b|dog\b|cat\b|car\b|train\b|airplane|helicopter|"
    r"hammer|wrench|drill|phone|laptop|tablet|headphone|speaker|remote|"
    r"bookend|book\b|magazine|newspaper|keyboard|mousepad|sneaker|boot\b|"
    r"sandal|slipper|sock|shirt|pant|dress|hat\b|helmet|glove|"
    r"remington|straightener|antislip|surfacing|bahama|ballet|flats|"
    r"sequin|dollhouse|sorter|frother|jarrodophilus|allergen|hair|"
    r"croc\b|canvas|file_sorter|tape\b|adhesive|loafer|penny|capetown|mens_|womens_",
    re.I,
)

FAMILY_RE = [
    ("mug", re.compile(r"mug|cup|tumbler", re.I)),
    ("bowl", re.compile(r"bowl|ramekin|casserole|mixing", re.I)),
    ("plate", re.compile(r"plate|saucer|dish", re.I)),
    ("pitcher", re.compile(r"pitcher|teapot|pot\b", re.I)),
    ("tray", re.compile(r"tray|platter", re.I)),
    ("bin", re.compile(r"bin|canister|container|bucket|holder|jar", re.I)),
    ("planter", re.compile(r"planter|vase|pot\b", re.I)),
    ("utensil", re.compile(r"utensil|spoon|rest|colander", re.I)),
]


def http_get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urllib.request.urlopen(req, context=CTX, timeout=timeout) as resp:
        return resp.read()


def http_json(url: str):
    return json.loads(http_get(url).decode("utf-8"))


def list_gso_models() -> list[dict]:
    models = []
    for page in range(1, 50):
        url = f"https://fuel.gazebosim.org/1.0/GoogleResearch/models?page={page}&per_page=50"
        batch = http_json(url)
        if not batch:
            break
        models.extend(batch)
        print(f"[select] listed page {page} n={len(models)}", flush=True)
        if len(batch) < 50:
            break
    uniq = {}
    for m in models:
        uniq[m["name"]] = m
    return list(uniq.values())


def slug(fuel_name: str, used: set[str]) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", fuel_name).strip("_").lower()
    parts = [p for p in s.split("_") if p and p not in {"the", "and", "with", "oz", "in"}]
    if len(parts) > 6:
        parts = parts[:2] + parts[-4:]
    base = "_".join(parts)[:48]
    out = base or "obj"
    i = 2
    while out in used:
        out = f"{base}_{i}"
        i += 1
    used.add(out)
    return out


def family_of(name: str) -> str:
    for fam, rgx in FAMILY_RE:
        if rgx.search(name):
            return fam
    return "other"


def prefilter(models: list[dict]) -> list[dict]:
    out = []
    for m in models:
        name = m["name"]
        if name in EXCLUDE_PILOT:
            continue
        text = f"{name} {m.get('description') or ''}"
        if EXCLUDE_RE.search(text):
            continue
        if not INCLUDE_RE.search(text):
            continue
        out.append(m)
    return out


def score_texture_used(path: Path) -> dict | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    used = gray > 18.0
    if float(used.mean()) < 0.08:
        return None
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    g = gray[used]
    return {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "used_frac": float(used.mean()),
        "mean_luma_used": float(g.mean() / 255.0),
        "std_luma_used": float(g.std() / 255.0),
        "mean_grad_used": float(mag[used].mean()),
        "p90_grad_used": float(np.percentile(mag[used], 90)),
        "mean_sat_used": float(sat[used].mean() / 255.0),
    }


def download_texture(fuel_name: str, dest: Path) -> bool:
    url = f"{FUEL_BASE}/{urllib.parse.quote(fuel_name)}/tip/files/materials/textures/texture.png"
    try:
        data = http_get(url)
        if len(data) < 64:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True
    except (urllib.error.URLError, TimeoutError, RuntimeError):
        return False


def ensure_texture(fuel_name: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 64:
        return True
    return download_texture(fuel_name, dest)


def weakness_score(stats: dict) -> float:
    """Lower is weaker / better candidate."""
    return (
        float(stats["mean_grad_used"])
        + 12.0 * float(stats["mean_sat_used"])
        + 4.0 * max(0.0, float(stats["std_luma_used"]) - 0.12)
    )


def pick_diverse(scored: list[dict], n: int, max_per_family: int) -> list[dict]:
    scored = sorted(scored, key=lambda r: r["weakness"])
    picked = []
    counts: dict[str, int] = {}
    for row in scored:
        fam = row["family"]
        if counts.get(fam, 0) >= max_per_family:
            continue
        picked.append(row)
        counts[fam] = counts.get(fam, 0) + 1
        if len(picked) >= n:
            break
    if len(picked) < n:
        have = {r["fuel_name"] for r in picked}
        for row in scored:
            if row["fuel_name"] in have:
                continue
            picked.append(row)
            if len(picked) >= n:
                break
    return picked


def write_experiment_yaml(path: Path, rows: list[dict], held_out: list[str]):
    assets = {}
    objects = []
    for r in rows:
        oid = r["id"]
        objects.append(oid)
        assets[oid] = {
            "fuel_name": r["fuel_name"],
            "mesh": f"data/gso_assets/{oid}/model.obj",
            "texture": f"data/gso_assets/{oid}/texture.png",
            "family": r["family"],
            "weakness": round(float(r["weakness"]), 4),
        }
    doc = {
        "name": "gso_scale",
        "data_root": "data/gso_scale",
        "results_root": "results/gso_scale",
        "resolution": 512,
        "num_views": 24,
        "samples": 24,
        "objects": objects,
        "held_out": held_out,
        "patterns": ["P0", "P4"],
        "alpha": 0.40,
        "scale": "medium",
        "seed": 0,
        "assets": assets,
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Select weakly textured GSO objects and write experiment config.")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--max_per_family", type=int, default=5)
    p.add_argument("--max_score_candidates", type=int, default=200)
    p.add_argument("--cache", default="data/gso_select")
    p.add_argument("--out_yaml", default="configs/gso_scale.yaml")
    p.add_argument("--held_out", type=int, default=8)
    p.add_argument("--download_meshes", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    cache = ROOT / args.cache
    cache.mkdir(parents=True, exist_ok=True)
    models = list_gso_models()
    cands = prefilter(models)
    cands.sort(
        key=lambda m: (
            0 if re.search(r"porcelain|ceramic", m["name"], re.I) else 1,
            0 if re.search(r"white", m["name"], re.I) else 1,
            m["name"],
        )
    )
    print(f"[select] catalog={len(models)} keyword_candidates={len(cands)}", flush=True)
    (cache / "keyword_candidates.json").write_text(
        json.dumps([{"name": m["name"], "description": m.get("description")} for m in cands], indent=2),
        encoding="utf-8",
    )

    cands = cands[: args.max_score_candidates]
    scored = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {
            ex.submit(ensure_texture, m["name"], cache / "textures" / f"{m['name']}.png"): m
            for m in cands
        }
        for fut in as_completed(futs):
            m = futs[fut]
            tex = cache / "textures" / f"{m['name']}.png"
            try:
                ok = bool(fut.result())
            except Exception as exc:
                print(f"[select] texture fail {m['name']}: {exc}", flush=True)
                continue
            if not ok:
                continue
            stats = score_texture_used(tex)
            if not stats:
                continue
            if stats["mean_sat_used"] > 0.28:
                continue
            if stats["mean_grad_used"] > 14.0:
                continue
            row = {
                "fuel_name": m["name"],
                "family": family_of(m["name"]),
                "weakness": weakness_score(stats),
                **stats,
            }
            scored.append(row)
            print(
                f"[select] {m['name']} fam={row['family']} grad={stats['mean_grad_used']:.2f} "
                f"sat={stats['mean_sat_used']:.3f} weak={row['weakness']:.2f}",
                flush=True,
            )

    scored.sort(key=lambda r: r["weakness"])
    (cache / "scored.json").write_text(json.dumps(scored, indent=2), encoding="utf-8")
    picked = pick_diverse(scored, args.n, args.max_per_family)
    used_ids: set[str] = set()
    for r in picked:
        r["id"] = slug(r["fuel_name"], used_ids)

    rng = np.random.default_rng(0)
    order = np.arange(len(picked))
    rng.shuffle(order)
    n_hold = min(args.held_out, max(1, len(picked) // 3))
    hold_idx = set(int(i) for i in order[:n_hold])
    held_out = [picked[i]["id"] for i in range(len(picked)) if i in hold_idx]

    out_yaml = ROOT / args.out_yaml
    write_experiment_yaml(out_yaml, picked, held_out)
    print(f"[select] picked={len(picked)} held_out={held_out} -> {out_yaml}", flush=True)

    if args.download_meshes:
        exp = load_yaml(out_yaml)
        out_dir = ROOT / "data/gso_assets"
        for oid in exp["objects"]:
            download_object(oid, exp["assets"][oid]["fuel_name"], out_dir)
            src_tex = cache / "textures" / f"{exp['assets'][oid]['fuel_name']}.png"
            dst_tex = out_dir / oid / "texture.png"
            if src_tex.exists() and (not dst_tex.exists() or dst_tex.stat().st_size < 64):
                dst_tex.write_bytes(src_tex.read_bytes())
            strip_obj_materials(out_dir / oid / "model.obj")
            print(f"[select] mesh ready {oid}", flush=True)
    print("[select] done")


if __name__ == "__main__":
    main()
