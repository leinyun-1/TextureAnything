from __future__ import annotations

from pathlib import Path

import numpy as np

from colmap_model import rotmat_to_qvec
from geometry import invert_T, load_camera_gt, load_depth, load_mask, project_opencv, sample_nearest, transform_points, unproject_opencv


def _visible(uv, z_proj, depth, mask, abs_tol, rel_tol):
    z_gt, valid = sample_nearest(depth, uv)
    m, _ = sample_nearest(mask, uv)
    return (
        valid
        & (z_proj > 1e-6)
        & (m > 0)
        & (z_gt > 1e-6)
        & (np.abs(z_proj - z_gt) < (abs_tol + rel_tol * np.maximum(z_gt, 1e-6)))
    )


def build_gt_tracks(scene_dir: Path, stride: int, min_track: int, abs_tol: float, rel_tol: float):
    cam = load_camera_gt(scene_dir)
    k = np.array(cam["K"], dtype=np.float64)
    frames = cam["frames"]
    depths = [load_depth(scene_dir, fr["id"]) for fr in frames]
    masks = [load_mask(scene_dir, fr["id"]) for fr in frames]
    t_wc = [np.array(fr["T_wc"], dtype=np.float64) for fr in frames]
    t_cw = [np.array(fr["T_cw"], dtype=np.float64) if "T_cw" in fr else invert_T(t) for fr, t in zip(frames, t_wc)]

    seed_i = 0
    ys, xs = np.where(masks[seed_i] > 0)
    if len(xs) == 0:
        return [], []
    ys = ys[::stride]
    xs = xs[::stride]
    uv0 = np.stack([xs + 0.5, ys + 0.5], axis=1)
    z0, valid0 = sample_nearest(depths[seed_i], uv0)
    keep = valid0 & (z0 > 1e-6)
    x_cam = unproject_opencv(uv0[keep], z0[keep], k)
    x_w = transform_points(t_wc[seed_i], x_cam)

    obs_lists: list[list[tuple[int, float, float]]] = [[] for _ in range(len(x_w))]
    for i, _fr in enumerate(frames):
        x_c = transform_points(t_cw[i], x_w)
        uv, z_proj = project_opencv(x_c, k)
        vis = _visible(uv, z_proj, depths[i], masks[i], abs_tol, rel_tol)
        for j in np.flatnonzero(vis):
            obs_lists[j].append((i, float(uv[j, 0]), float(uv[j, 1])))

    points = []
    tracks = []
    for xyz, obs in zip(x_w, obs_lists):
        if len(obs) >= min_track:
            points.append(xyz)
            tracks.append(obs)
    return points, tracks


def source_names_by_camera_distance(centers: np.ndarray, names: list[str], num_src: int) -> dict[str, list[str]]:
    n = len(names)
    out = {}
    for i, name in enumerate(names):
        dist = np.linalg.norm(centers - centers[i], axis=1)
        dist[i] = np.inf
        order = np.argsort(dist)[: min(num_src, n - 1)]
        out[name] = [names[j] for j in order]
    return out


def write_gt_colmap_model(scene_dir: Path, out_dir: Path, cfg: dict) -> dict[str, list[str]]:
    cam = load_camera_gt(scene_dir)
    k = np.array(cam["K"], dtype=np.float64)
    width = int(cam["width"])
    height = int(cam["height"])
    frames = cam["frames"]
    names = [Path(fr["file"]).name for fr in frames]
    centers = np.stack([np.array(fr["T_wc"], dtype=np.float64)[:3, 3] for fr in frames])
    points, tracks = build_gt_tracks(
        scene_dir,
        stride=int(cfg.get("mvs_gt_sample_stride", 12)),
        min_track=int(cfg.get("mvs_min_track_length", 3)),
        abs_tol=float(cfg.get("depth_abs_tol", 0.04)),
        rel_tol=float(cfg.get("depth_rel_tol", 0.02)),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("cameras.bin", "images.bin", "points3D.bin"):
        p = out_dir / stale
        if p.exists():
            p.unlink()

    cameras_txt = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        "# Number of cameras: 1",
        f"1 PINHOLE {width} {height} {k[0, 0]} {k[1, 1]} {k[0, 2]} {k[1, 2]}",
        "",
    ]
    (out_dir / "cameras.txt").write_text("\n".join(cameras_txt), encoding="utf-8")

    image_obs: list[list[tuple[float, float, int]]] = [[] for _ in frames]
    points_lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
        f"# Number of points: {len(points)}",
    ]
    for pid, (xyz, obs) in enumerate(zip(points, tracks), start=1):
        track_tokens = []
        for img_idx, u, v in obs:
            point2d_idx = len(image_obs[img_idx])
            image_obs[img_idx].append((u, v, pid))
            track_tokens.extend([str(img_idx + 1), str(point2d_idx)])
        points_lines.append(
            f"{pid} {xyz[0]} {xyz[1]} {xyz[2]} 128 128 128 0.5 " + " ".join(track_tokens)
        )
    points_lines.append("")
    (out_dir / "points3D.txt").write_text("\n".join(points_lines), encoding="utf-8")

    image_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(frames)}",
    ]
    for i, fr in enumerate(frames):
        if "T_cw" in fr:
            t_cw = np.array(fr["T_cw"], dtype=np.float64)
        else:
            t_cw = invert_T(np.array(fr["T_wc"], dtype=np.float64))
        rot = t_cw[:3, :3]
        tvec = t_cw[:3, 3]
        center = -rot.T @ tvec
        u, _, vt = np.linalg.svd(rot)
        rot = u @ vt
        if np.linalg.det(rot) < 0:
            u[:, -1] *= -1
            rot = u @ vt
        qvec = rotmat_to_qvec(rot)
        tvec = -rot @ center
        name = names[i]
        image_lines.append(
            f"{i + 1} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]} 1 {name}"
        )
        if image_obs[i]:
            image_lines.append(" ".join(f"{u} {v} {pid}" for u, v, pid in image_obs[i]))
        else:
            image_lines.append("")
    (out_dir / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    print(f"[gt-model] {len(frames)} images, {len(points)} GT sparse points -> {out_dir}", flush=True)
    return source_names_by_camera_distance(centers, names, int(cfg.get("mvs_num_src_images", 20)))


def write_patch_match_cfg(path: Path, sources: dict[str, list[str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for name, src in sources.items():
        lines.append(name)
        lines.append(", ".join(src))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
