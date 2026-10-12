"""Score ABO base-color textures inside abo-3dmodels.tar.

Reads the archive once. A model passes when its base-color map meets the same
albedo thresholds as the GSO L2 screen.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np

from select_gso_weak import weakness_score

JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


def glb_parts(data: bytes):
    if data[:4] != b"glTF":
        raise ValueError("not a glb")
    offset = 12
    js = None
    blob = b""
    while offset + 8 <= len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8 : offset + 8 + length]
        if kind == JSON_CHUNK:
            js = json.loads(chunk)
        elif kind == BIN_CHUNK:
            blob = chunk
        offset += 8 + length
    if js is None:
        raise ValueError("glb has no json chunk")
    return js, blob


def image_bytes(js: dict, blob: bytes, index: int) -> bytes:
    image = js["images"][index]
    view = js["bufferViews"][image["bufferView"]]
    start = int(view.get("byteOffset", 0))
    end = start + int(view["byteLength"])
    return blob[start:end]


def base_color_index(js: dict) -> int | None:
    for mat in js.get("materials") or []:
        pbr = mat.get("pbrMetallicRoughness") or {}
        tex = (pbr.get("baseColorTexture") or {}).get("index")
        if tex is None:
            continue
        source = js["textures"][tex].get("source")
        if source is not None:
            return int(source)
    return None


def score_bgr(img: np.ndarray) -> dict | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    used = gray > 18.0
    if float(used.mean()) < 0.08:
        return None
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    stats = {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "used_frac": float(used.mean()),
        "mean_luma_used": float(gray[used].mean() / 255.0),
        "std_luma_used": float(gray[used].std() / 255.0),
        "mean_grad_used": float(mag[used].mean()),
        "p90_grad_used": float(np.percentile(mag[used], 90)),
        "mean_sat_used": float(sat[used].mean() / 255.0),
    }
    stats["weakness"] = weakness_score(stats)
    return stats


def load_done(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        done.add(json.loads(line)["member"])
    return done


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tar", default=r"D:\TextureAnything\datasets\abo\abo-3dmodels.tar")
    p.add_argument("--out", default=r"D:\TextureAnything\datasets\abo\albedo_scores.jsonl")
    p.add_argument("--max_sat", type=float, default=0.28)
    p.add_argument("--max_grad", type=float, default=14.0)
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    passed = 0
    seen = 0
    with tarfile.open(args.tar, "r|") as tar, out.open("a", encoding="utf-8") as sink:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".glb"):
                continue
            if member.name in done:
                continue
            seen += 1
            row = {"member": member.name, "id": Path(member.name).stem}
            try:
                data = tar.extractfile(member).read()
                js, blob = glb_parts(data)
                index = base_color_index(js)
                if index is None:
                    row["status"] = "no_base_color"
                else:
                    encoded = np.frombuffer(image_bytes(js, blob, index), dtype=np.uint8)
                    img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    stats = score_bgr(img) if img is not None else None
                    if stats is None:
                        row["status"] = "unreadable"
                    else:
                        row.update(stats)
                        keep = stats["mean_sat_used"] <= args.max_sat and stats["mean_grad_used"] <= args.max_grad
                        row["status"] = "pass" if keep else "too_textured"
                        passed += int(keep)
            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if seen % 100 == 0:
                print(f"[abo] scored={seen} new_pass={passed} last={row['id']} {row['status']}", flush=True)
    print(f"[abo] done scored={seen} new_pass={passed} -> {out}", flush=True)


if __name__ == "__main__":
    main()
