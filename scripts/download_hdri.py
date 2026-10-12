"""Download the CC0 Poly Haven HDRIs used to light black-background pairs."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from render_abo_black import HDRI_DIR, HDRI_NAMES

UA = {"User-Agent": "TextureAnything/1.0 (research; local dataset)"}


def get_json(url: str):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def main():
    HDRI_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in HDRI_NAMES:
        dest = HDRI_DIR / f"{name}_2k.hdr"
        meta = get_json(f"https://api.polyhaven.com/files/{name}")
        url = meta["hdri"]["2k"]["hdr"]["url"]
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"[hdri] have {dest.name}", flush=True)
        else:
            print(f"[hdri] get {name}", flush=True)
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=180) as resp:
                dest.write_bytes(resp.read())
            print(f"[hdri] wrote {dest.name} {dest.stat().st_size}", flush=True)
        manifest.append({"name": name, "file": dest.name, "url": url, "license": "CC0"})
    (HDRI_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[hdri] {len(manifest)} ready", flush=True)


if __name__ == "__main__":
    main()
