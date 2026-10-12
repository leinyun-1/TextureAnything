from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def load_camera_gt(scene_dir: Path) -> dict:
    with open(scene_dir / "camera_gt.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_depth(scene_dir: Path, frame_id: int) -> np.ndarray:
    return np.load(scene_dir / "depth" / f"{frame_id:04d}.npy")


def load_mask(scene_dir: Path, frame_id: int) -> np.ndarray:
    npy = scene_dir / "mask" / f"{frame_id:04d}.npy"
    png = scene_dir / "mask" / f"{frame_id:04d}.png"
    if npy.exists():
        mask = np.load(npy)
    else:
        mask = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(png)
    return (mask > 127).astype(np.uint8)


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def sample_nearest(img: np.ndarray, uv: np.ndarray):
    h, w = img.shape[:2]
    x = np.floor(uv[:, 0]).astype(np.int32)
    y = np.floor(uv[:, 1]).astype(np.int32)
    valid = (x >= 0) & (y >= 0) & (x < w) & (y < h)
    x_clip = np.clip(x, 0, w - 1)
    y_clip = np.clip(y, 0, h - 1)
    vals = img[y_clip, x_clip]
    return vals, valid


def unproject_opencv(uv: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (uv[:, 0] - cx) / fx * z
    y = (uv[:, 1] - cy) / fy * z
    return np.stack([x, y, z], axis=1)


def transform_points(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return X @ R.T + t


def project_opencv(X_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = X_cam[:, 2]
    uv = np.stack(
        [
            K[0, 0] * X_cam[:, 0] / np.clip(z, 1e-8, None) + K[0, 2],
            K[1, 1] * X_cam[:, 1] / np.clip(z, 1e-8, None) + K[1, 2],
        ],
        axis=1,
    )
    return uv, z


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask.astype(np.uint8), kernel)


def tile_coverage(mask: np.ndarray, points_uv: np.ndarray, tiles: int) -> float:
    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    tile_w = w / tiles
    tile_h = h / tiles
    obj_tiles = set(zip((xs / tile_w).astype(int), (ys / tile_h).astype(int)))
    if points_uv.size == 0:
        return 0.0
    hit = set(zip((points_uv[:, 0] / tile_w).astype(int), (points_uv[:, 1] / tile_h).astype(int)))
    return len(hit & obj_tiles) / max(len(obj_tiles), 1)
