"""Download the data the current experiments need into TA_DATA_ROOT.

  abo   the ABO models listed in manifests/abo_queue.json, one glb per object (~20.6 GB)
  hdri  the 20 Poly Haven HDRIs used for lighting (~121 MB)

Finished files are skipped, so an interrupted run can simply be started again.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from config_utils import ABO_MODELS, ABO_QUEUE, DATA_ROOT

UA = {"User-Agent": "TextureAnything/1.0 (research; dataset fetch)"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("targets", nargs="*", choices=["abo", "hdri"], default=["abo", "hdri"])
    p.add_argument("--ids", nargs="*", help="only these ABO ids")
    p.add_argument("--limit", type=int, default=0, help="only the first N ABO objects in queue order")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def download(url: str, dest, retries: int = 3) -> int:
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as resp, open(part, "wb") as f:
                expected = int(resp.headers.get("Content-Length") or 0)
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
            size = part.stat().st_size
            if expected and size != expected:
                raise IOError(f"short read {size}/{expected}")
            part.replace(dest)
            return size
        except Exception:
            part.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(2 * attempt)
    return 0


def fetch_abo(args):
    manifest = json.loads(ABO_QUEUE.read_text(encoding="utf-8"))
    items = manifest["items"]
    if args.ids:
        wanted = set(args.ids)
        items = [item for item in items if item["id"] in wanted]
    if args.limit:
        items = items[: args.limit]
    ABO_MODELS.mkdir(parents=True, exist_ok=True)
    todo = [item for item in items if not (ABO_MODELS / f"{item['id']}.glb").exists()]
    print(f"[abo] {len(items)} in queue, {len(items) - len(todo)} present, {len(todo)} to download -> {ABO_MODELS}", flush=True)
    done = failed = 0
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download, manifest["url_base"] + item["member"], ABO_MODELS / f"{item['id']}.glb"): item["id"]
            for item in todo
        }
        for future in as_completed(futures):
            try:
                total_bytes += future.result()
                done += 1
            except Exception as exc:
                failed += 1
                print(f"[abo] FAIL {futures[future]}: {exc}", flush=True)
            if (done + failed) % 50 == 0:
                print(f"[abo] {done + failed}/{len(todo)}  {total_bytes / 1e9:.1f} GB", flush=True)
    print(f"[abo] downloaded={done} failed={failed} ({total_bytes / 1e9:.1f} GB)", flush=True)
    return failed


def main():
    args = parse_args()
    print(f"[fetch] TA_DATA_ROOT={DATA_ROOT}", flush=True)
    failed = 0
    if "hdri" in args.targets:
        import download_hdri

        download_hdri.main()
    if "abo" in args.targets:
        failed += fetch_abo(args)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
