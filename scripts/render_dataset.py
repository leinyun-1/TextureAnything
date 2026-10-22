from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from config_utils import ROOT, load_configs, load_yaml, pattern_folder, scale_value


def run(cmd, cwd=None):
    print("[cmd]", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def write_mask_pngs(scene_dir: Path):
    mask_dir = scene_dir / "mask"
    for npy_path in sorted(mask_dir.glob("*.npy")):
        mask = np.load(npy_path)
        cv2.imwrite(str(npy_path.with_suffix(".png")), mask)


def apply_p1_noise(src_dir: Path, dst_dir: Path, sigma: float, seed: int):
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    rng = np.random.default_rng(seed)
    rgb_dir = dst_dir / "rgb"
    for png in sorted(rgb_dir.glob("*.png")):
        img = cv2.imread(str(png), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(png)
        noise = rng.normal(0.0, sigma * 255.0, size=img.shape)
        out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(png), out)
    cam_path = dst_dir / "camera_gt.json"
    with open(cam_path, "r", encoding="utf-8") as f:
        cam = json.load(f)
    cam["pattern"] = "P1"
    cam["p1_sigma"] = sigma
    with open(cam_path, "w", encoding="utf-8") as f:
        json.dump(cam, f, indent=2)


def render_variant(args, render_cfg, patterns_cfg, obj_name: str, pattern_id: str, out_dir: Path, mesh: str = "", texture: str = "", extra: dict | None = None):
    blender = render_cfg["blender_exe"]
    script = ROOT / "scripts" / "blender_render.py"
    scale = scale_value(patterns_cfg, args.scale)
    extra = dict(extra or {})

    def pick(key, default):
        if key in extra:
            return extra[key]
        return render_cfg.get(key, default)
    cmd = [
        blender,
        "-b",
        "--python-exit-code",
        "1",
        "--python",
        str(script),
        "--",
        "--output_dir",
        str(out_dir),
        "--object",
        obj_name,
        "--pattern",
        pattern_id,
        "--num_views",
        str(args.num_views),
        "--resolution",
        str(int(extra.get("resolution_x") or args.resolution)),
        "--resolution_x",
        str(int(extra.get("resolution_x") or args.resolution)),
        "--resolution_y",
        str(int(extra.get("resolution_y") or extra.get("resolution_x") or args.resolution)),
        "--samples",
        str(args.samples),
        "--alpha",
        str(args.alpha),
        "--scale",
        str(scale),
        "--seed",
        str(args.seed),
        "--engine",
        str(extra.get("engine") or render_cfg.get("engine", "CYCLES")),
        "--device",
        str(render_cfg.get("device", "GPU")),
        "--lens_mm",
        str(pick("lens_mm", 50.0)),
        "--sensor_width_mm",
        str(render_cfg.get("sensor_width_mm", 36.0)),
        "--radius",
        str(pick("radius", 2.55)),
        "--azimuth_span_deg",
        str(pick("azimuth_span_deg", 300.0)),
        "--azimuth_start_deg",
        str(pick("azimuth_start_deg", 0.0)),
        "--elevation_deg",
        str(pick("elevation_deg", 18.0)),
        "--elevation_amp_deg",
        str(pick("elevation_amp_deg", 10.0)),
        "--hdri",
        str(extra.get("hdri", "")),
        "--hdri_rotation_deg",
        str(extra.get("hdri_rotation_deg", 0.0)),
        "--hdri_strength",
        str(extra.get("hdri_strength", 1.0)),
        "--frame_fill",
        str(extra.get("frame_fill", 0.0)),
        "--world_gray",
        str(render_cfg.get("world_gray", 0.22)),
        "--key_light_energy",
        str(render_cfg.get("key_light_energy", 180.0)),
        "--fill_light_energy",
        str(render_cfg.get("fill_light_energy", 70.0)),
        "--projector_count",
        str(render_cfg.get("projector_count", 3)),
        "--projector_energy",
        str(render_cfg.get("projector_energy", 90.0)),
        "--projector_spot_deg",
        str(render_cfg.get("projector_spot_deg", 55.0)),
        "--projector_radius",
        str(render_cfg.get("projector_radius", 2.2)),
        "--projector_elevation_deg",
        str(render_cfg.get("projector_elevation_deg", 50.0)),
        "--projector_elevation_low_deg",
        str(render_cfg.get("projector_elevation_low_deg", 6.0)),
        "--projector_scale",
        str(render_cfg.get("projector_scale", 0.0)),
        "--projector_env_scale",
        str(render_cfg.get("projector_env_scale", 0.45)),
        "--projector_contrast_lo",
        str(render_cfg.get("projector_contrast_lo", 0.40)),
        "--projector_contrast_hi",
        str(render_cfg.get("projector_contrast_hi", 0.60)),
        "--projector_gain",
        str(render_cfg.get("projector_gain", 0.85)),
    ]
    if extra.get("denoise"):
        cmd.append("--denoise")
    if extra.get("rgb_only"):
        cmd.append("--rgb_only")
    if extra.get("material"):
        cmd.extend(["--material", str(extra["material"])])
    if extra.get("raytrace"):
        cmd.append("--raytrace")
    if extra.get("speckle"):
        cmd.append("--speckle")
        for key in ("speckle_scale", "speckle_threshold"):
            if key in extra:
                cmd.extend([f"--{key}", str(extra[key])])
    if extra.get("taa_samples"):
        cmd.extend(["--taa_samples", str(int(extra["taa_samples"]))])
    if pattern_id == "P0":
        cmd.append("--write_geometry")
    if mesh:
        cmd.extend(["--mesh", str(mesh)])
    if texture:
        cmd.extend(["--texture", str(texture)])
    run(cmd)
    if not extra.get("rgb_only"):
        write_mask_pngs(out_dir)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="configs/smoke.yaml")
    p.add_argument("--objects", nargs="*")
    p.add_argument("--patterns", nargs="*")
    p.add_argument("--num_views", type=int)
    p.add_argument("--resolution", type=int)
    p.add_argument("--samples", type=int)
    p.add_argument("--alpha", type=float)
    p.add_argument("--scale")
    p.add_argument("--seed", type=int)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--data_root")
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    cfgs = load_configs()
    exp = load_yaml(ROOT / args.experiment)
    render_cfg = dict(cfgs["render"])
    patterns_cfg = cfgs["patterns"]
    for key in (
        "projector_count",
        "projector_energy",
        "projector_spot_deg",
        "projector_radius",
        "projector_elevation_deg",
        "projector_elevation_low_deg",
        "projector_scale",
        "projector_env_scale",
        "projector_contrast_lo",
        "projector_contrast_hi",
        "projector_gain",
    ):
        if exp.get(key) is not None:
            render_cfg[key] = exp[key]

    objects = args.objects or exp["objects"]
    patterns = args.patterns or exp["patterns"]
    args.num_views = args.num_views or exp.get("num_views", render_cfg["num_views"])
    args.resolution = args.resolution or exp.get("resolution", render_cfg["resolution"])
    args.samples = args.samples or exp.get("samples", render_cfg["samples"])
    args.alpha = args.alpha if args.alpha is not None else exp.get("alpha", patterns_cfg["alpha"])
    args.scale = args.scale or exp.get("scale", patterns_cfg["scale"])
    args.seed = args.seed if args.seed is not None else exp.get("seed", patterns_cfg["seed"])
    data_root = ROOT / (args.data_root or exp["data_root"])

    if "P0" in patterns:
        patterns = ["P0"] + [p for p in patterns if p != "P0"]

    for obj_name in objects:
        item = (exp.get("assets") or {}).get(obj_name, {})
        p0_dir = None
        for pattern_id in patterns:
            folder = pattern_folder(pattern_id, patterns_cfg)
            out_dir = data_root / obj_name / folder
            print(f"[dataset] {obj_name} {pattern_id} -> {out_dir}", flush=True)
            if pattern_id == "P1":
                if p0_dir is None or not (p0_dir / "rgb").exists():
                    raise RuntimeError("P1 requires P0 to be rendered first")
                apply_p1_noise(p0_dir, out_dir, float(patterns_cfg["p1_sigma"]), int(args.seed))
                write_mask_pngs(out_dir)
            else:
                rgb0 = out_dir / "rgb" / "0000.png"
                if args.skip_existing and rgb0.exists() and (out_dir / "camera_gt.json").exists():
                    print(f"[dataset] skip existing {out_dir}", flush=True)
                else:
                    if out_dir.exists():
                        shutil.rmtree(out_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    render_variant(
                        args,
                        render_cfg,
                        patterns_cfg,
                        obj_name,
                        pattern_id,
                        out_dir,
                        mesh=str(ROOT / item["mesh"]) if item.get("mesh") else "",
                        texture=str(ROOT / item["texture"]) if item.get("texture") else "",
                    )
            if pattern_id == "P0":
                p0_dir = out_dir
    print("[dataset] done", data_root)


if __name__ == "__main__":
    main()
