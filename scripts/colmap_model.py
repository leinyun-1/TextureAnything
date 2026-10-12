from __future__ import annotations

from pathlib import Path

import numpy as np


def rotmat_to_qvec(rot: np.ndarray) -> np.ndarray:
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rot.flat
    k = (
        np.array(
            [
                [rxx - ryy - rzz, 0, 0, 0],
                [ryx + rxy, ryy - rxx - rzz, 0, 0],
                [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0],
                [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
            ],
            dtype=np.float64,
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(k)
    qvec = eigvecs[[3, 0, 1, 2], int(np.argmax(eigvals))]
    if qvec[0] < 0:
        qvec = -qvec
    return qvec


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Sim(3): dst ≈ s * R @ src + t."""
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    cov = (dst_c.T @ src_c) / n
    u, d, vt = np.linalg.svd(cov)
    s_mat = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[2, 2] = -1.0
    rot = u @ s_mat @ vt
    var_s = float((src_c**2).sum() / n)
    scale = float(np.trace(np.diag(d) @ s_mat) / max(var_s, 1e-12))
    trans = mu_d - scale * rot @ mu_s
    return scale, rot, trans


def rotation_error_deg(r_gt: np.ndarray, r_est: np.ndarray) -> float:
    delta = r_gt.T @ r_est
    tr = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


def read_images_txt(path: Path) -> dict[str, dict]:
    images = {}
    if not path.exists():
        return images
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        qvec = np.array([float(x) for x in parts[1:5]], dtype=np.float64)
        tvec = np.array([float(x) for x in parts[5:8]], dtype=np.float64)
        name = parts[9]
        r_cw = qvec_to_rotmat(qvec)
        center = -r_cw.T @ tvec
        r_wc = r_cw.T
        images[name] = {
            "r_wc": r_wc,
            "center": center,
            "num_points2d": len(lines[i + 1].split()) // 3 if i + 1 < len(lines) else 0,
        }
    return images


def read_points3d_txt(path: Path) -> tuple[int, float, float]:
    if not path.exists():
        return 0, float("nan"), float("nan")
    errors = []
    tracks = []
    n = 0
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        parts = ln.split()
        n += 1
        errors.append(float(parts[7]))
        tracks.append((len(parts) - 8) // 2)
    if n == 0:
        return 0, float("nan"), float("nan")
    return n, float(np.mean(tracks)), float(np.mean(errors))


def pick_largest_model(sparse_dir: Path) -> Path | None:
    if not sparse_dir.exists():
        return None
    candidates = []
    for child in sparse_dir.iterdir():
        if child.is_dir() and ((child / "images.bin").exists() or (child / "images.txt").exists()):
            candidates.append(child)
        elif child.name in {"images.bin", "images.txt"}:
            return sparse_dir
    if not candidates:
        return None

    def n_images(model: Path) -> int:
        txt = model / "images.txt"
        if txt.exists():
            return sum(1 for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")) // 2
        bin_path = model / "images.bin"
        return bin_path.stat().st_size if bin_path.exists() else 0

    return max(candidates, key=n_images)
