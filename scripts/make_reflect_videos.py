"""One MP4 per object: every material next to its speckled version, with COLMAP results."""
from __future__ import annotations

import csv
import math
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from config_utils import ABO_ROOT, ROOT

RENDER_ROOT = ABO_ROOT / "renders_reflect"
RESULTS = ROOT / "results" / "reflect_colmap"
OUT_DIR = RENDER_ROOT / "videos"
NAMES = {
    "B075HWK9QT": "vase",
    "B07BLB6644": "saute pan",
    "B07BG69MRW": "bunk bed",
    "B0754KHKCX": "shelf",
}
MATERIALS = ("matte", "glossy", "abo_pbr", "white_matte", "white_glaze", "chrome", "brushed_metal")
TILE_W, TILE_H = 416, 240
COLS, ROWS = 4, 4
TITLE_H = 40
NUM_FRAMES = 81
FPS = 15


def load_metrics() -> dict:
    metrics = {}
    for name, speckle in (("sfm_metrics.csv", False), ("sfm_metrics_speckle.csv", True)):
        with open(RESULTS / name, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                metrics[(row["scene_id"], row["material"], speckle)] = row
    return metrics


def caption(row: dict | None) -> tuple[str, tuple[int, int, int]]:
    if row is None:
        return "no result", (160, 160, 160)
    reg = int(row["num_registered"])
    total = int(row["num_frames"])
    err = float(row["rotation_error_median"])
    if math.isnan(err):
        return f"reg {reg}/{total}  FAILED", (60, 60, 255)
    good = reg >= 0.9 * total and err < 2.0
    return f"reg {reg}/{total}  rot err {err:.2f} deg", (80, 220, 80) if good else (60, 60, 255)


def label(img, lines):
    y = 6
    for text, color, scale in lines:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(img, (4, y), (12 + w, y + h + 8), (30, 30, 30), -1)
        cv2.putText(img, text, (8, y + h + 3), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += h + 12


def legend_tile() -> np.ndarray:
    tile = np.zeros((TILE_H * 1, TILE_W * 2, 3), np.uint8)
    lines = [
        "Each pair: left = original surface,",
        "           right = + matte speckle dots",
        "reg = frames registered by COLMAP",
        "rot err = median rotation error after Sim(3)",
        "green: reg >= 90% and rot err < 2 deg",
        "red: failed or wrong poses",
    ]
    for i, text in enumerate(lines):
        cv2.putText(tile, text, (16, 40 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    return tile


def open_writer(path: Path, width: int, height: int):
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def make_video(object_id: str, metrics: dict):
    width, height = COLS * TILE_W, TITLE_H + ROWS * TILE_H
    slots = []
    for material in MATERIALS:
        for speckle in (False, True):
            folder = RENDER_ROOT / object_id / (f"{material}_speckle" if speckle else material) / "rgb"
            text, color = caption(metrics.get((object_id, material, speckle)))
            name = f"{material} + speckle" if speckle else material
            slots.append((folder, [(name, (255, 255, 255), 0.55), (text, color, 0.5)]))
    legend = legend_tile()
    out = OUT_DIR / f"{object_id}_{NAMES.get(object_id, '').replace(' ', '_')}.mp4"
    proc = open_writer(out, width, height)
    for t in range(NUM_FRAMES):
        canvas = np.zeros((height, width, 3), np.uint8)
        title = f"{object_id}  ({NAMES.get(object_id, '')})   frame {t:02d}/{NUM_FRAMES - 1}"
        cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        for i, (folder, lines) in enumerate(slots):
            img = cv2.imread(str(folder / f"{t:04d}.png"))
            tile = np.zeros((TILE_H, TILE_W, 3), np.uint8) if img is None else cv2.resize(
                img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA
            )
            label(tile, lines)
            r, c = divmod(i, COLS)
            y0, x0 = TITLE_H + r * TILE_H, c * TILE_W
            canvas[y0 : y0 + TILE_H, x0 : x0 + TILE_W] = tile
            if c % 2 == 1:
                cv2.line(canvas, (x0 + TILE_W - 1, y0), (x0 + TILE_W - 1, y0 + TILE_H), (90, 90, 90), 2)
        r, c = divmod(len(slots), COLS)
        y0 = TITLE_H + r * TILE_H
        canvas[y0 : y0 + TILE_H, c * TILE_W : c * TILE_W + legend.shape[1]] = legend
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    proc.wait()
    print("[video]", out, flush=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    for object_id in NAMES:
        make_video(object_id, metrics)


if __name__ == "__main__":
    main()
