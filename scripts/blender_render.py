"""Headless Blender 5.2 renderer for Virtual Texture Oracle smoke tests.

Invoked as:
  blender -b --python-exit-code 1 --python scripts/blender_render.py -- [args]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


PATTERN_NAMES = {
    "P0": "P0_plain",
    "P1": "P1_frame_noise",
    "P2": "P2_grid",
    "P3": "P3_random_dots",
    "P4": "P4_speckle",
    "P5": "P5_multiscale",
    "P6": "P6_hash",
    "P4P": "P4_projector",
}

# base=None keeps the asset's own albedo. abo_pbr keeps the glTF material untouched.
MATERIAL_PRESETS = {
    "matte": {"base": None, "metallic": 0.0, "roughness": 0.88, "specular": 0.05, "coat": 0.0},
    "glossy": {"base": None, "metallic": 0.0, "roughness": 0.08, "specular": 0.5, "coat": 1.0},
    "abo_pbr": None,
    "white_matte": {"base": (0.82, 0.82, 0.80), "metallic": 0.0, "roughness": 0.88, "specular": 0.05, "coat": 0.0},
    "white_glaze": {"base": (0.82, 0.82, 0.80), "metallic": 0.0, "roughness": 0.05, "specular": 0.5, "coat": 1.0},
    "chrome": {"base": (0.95, 0.95, 0.95), "metallic": 1.0, "roughness": 0.03, "specular": 0.5, "coat": 0.0},
    "brushed_metal": {"base": (0.85, 0.84, 0.82), "metallic": 1.0, "roughness": 0.25, "specular": 0.5, "coat": 0.0},
}


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--object", required=True)
    p.add_argument("--pattern", required=True, choices=list(PATTERN_NAMES))
    p.add_argument("--mesh", default="")
    p.add_argument("--texture", default="")
    p.add_argument("--num_views", type=int, default=24)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--resolution_x", type=int, default=0)
    p.add_argument("--resolution_y", type=int, default=0)
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--denoise", action="store_true")
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--scale", type=float, default=24.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--engine", default="CYCLES")
    p.add_argument("--device", default="GPU")
    p.add_argument("--lens_mm", type=float, default=50.0)
    p.add_argument("--sensor_width_mm", type=float, default=36.0)
    p.add_argument("--radius", type=float, default=2.55)
    p.add_argument("--azimuth_span_deg", type=float, default=300.0)
    p.add_argument("--azimuth_start_deg", type=float, default=0.0)
    p.add_argument("--elevation_deg", type=float, default=18.0)
    p.add_argument("--elevation_amp_deg", type=float, default=10.0)
    p.add_argument("--world_gray", type=float, default=0.22)
    p.add_argument("--hdri", default="")
    p.add_argument("--hdri_rotation_deg", type=float, default=0.0)
    p.add_argument("--hdri_strength", type=float, default=1.0)
    p.add_argument("--key_light_energy", type=float, default=180.0)
    p.add_argument("--fill_light_energy", type=float, default=70.0)
    p.add_argument("--clip_start", type=float, default=0.15)
    p.add_argument("--clip_end", type=float, default=20.0)
    p.add_argument("--projector_count", type=int, default=8)
    p.add_argument("--projector_energy", type=float, default=90.0)
    p.add_argument("--projector_spot_deg", type=float, default=55.0)
    p.add_argument("--projector_radius", type=float, default=2.2)
    p.add_argument("--projector_elevation_deg", type=float, default=50.0)
    p.add_argument("--projector_elevation_low_deg", type=float, default=6.0)
    p.add_argument("--projector_scale", type=float, default=0.0)
    p.add_argument("--projector_env_scale", type=float, default=0.45)
    p.add_argument("--projector_contrast_lo", type=float, default=0.40)
    p.add_argument("--projector_contrast_hi", type=float, default=0.60)
    p.add_argument("--projector_gain", type=float, default=0.85)
    p.add_argument("--write_geometry", action="store_true")
    p.add_argument("--rgb_only", action="store_true")
    p.add_argument("--frame_fill", type=float, default=0.0)
    p.add_argument("--material", default="", choices=["", *MATERIAL_PRESETS])
    p.add_argument("--raytrace", action="store_true")
    p.add_argument("--speckle", action="store_true")
    p.add_argument("--speckle_scale", type=float, default=28.0)
    p.add_argument("--speckle_threshold", type=float, default=0.08)
    p.add_argument("--taa_samples", type=int, default=16)
    p.add_argument("--probe_resolution", default="2048", choices=["512", "1024", "2048", "4096"])
    return p.parse_args(argv)


def link(nt, src, dst):
    nt.links.new(src, dst)


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in list(bpy.data.meshes) + list(bpy.data.cameras) + list(bpy.data.lights) + list(bpy.data.materials):
        if block.users == 0:
            if isinstance(block, bpy.types.Mesh):
                bpy.data.meshes.remove(block)
            elif isinstance(block, bpy.types.Camera):
                bpy.data.cameras.remove(block)
            elif isinstance(block, bpy.types.Light):
                bpy.data.lights.remove(block)
            elif isinstance(block, bpy.types.Material):
                bpy.data.materials.remove(block)


def center_active():
    obj = bpy.context.active_object
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    obj.location = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    coords = np.array([obj.matrix_world @ v.co for v in obj.data.vertices])
    center = coords.mean(axis=0)
    obj.location -= Vector(center.tolist())
    bpy.context.view_layer.update()
    obj.pass_index = 1
    obj.name = "target"


def make_box():
    bpy.ops.mesh.primitive_cube_add(size=1.05, location=(0, 0, 0))
    obj = bpy.context.active_object
    bevel = obj.modifiers.new("bevel", "BEVEL")
    bevel.width = 0.035
    bevel.segments = 3
    bpy.ops.object.modifier_apply(modifier="bevel")
    center_active()
    return bpy.context.active_object


def make_bottle():
    bpy.ops.mesh.primitive_cylinder_add(radius=0.28, depth=0.85, vertices=48, location=(0, 0, 0.05))
    body = bpy.context.active_object
    bpy.ops.mesh.primitive_cylinder_add(radius=0.12, depth=0.32, vertices=32, location=(0, 0, 0.52))
    neck = bpy.context.active_object
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.13, segments=32, ring_count=16, location=(0, 0, 0.70))
    cap = bpy.context.active_object
    for o in (body, neck, cap):
        o.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()
    center_active()
    return bpy.context.active_object


def make_bowl():
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.58, segments=48, ring_count=24, location=(0, 0, 0))
    obj = bpy.context.active_object
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for v in obj.data.vertices:
        v.select = v.co.z > 0.08
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.delete(type="VERT")
    bpy.ops.object.mode_set(mode="OBJECT")
    solid = obj.modifiers.new("solid", "SOLIDIFY")
    solid.thickness = 0.045
    bpy.ops.object.modifier_apply(modifier="solid")
    center_active()
    return bpy.context.active_object


OBJECT_BUILDERS = {"box": make_box, "bottle": make_bottle, "bowl": make_bowl}


def import_mesh(path: str, target_extent: float = 1.0):
    mesh_path = Path(path)
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)
    before = set(bpy.data.objects)
    suffix = mesh_path.suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    else:
        try:
            bpy.ops.wm.obj_import(filepath=str(mesh_path))
        except Exception:
            bpy.ops.import_scene.obj(filepath=str(mesh_path))
    new_objects = [o for o in bpy.data.objects if o not in before]
    for extra in new_objects:
        if extra.type != "MESH":
            bpy.data.objects.remove(extra, do_unlink=True)
    new_meshes = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not new_meshes:
        raise RuntimeError(f"OBJ import produced no mesh: {mesh_path}")
    for o in bpy.data.objects:
        o.select_set(False)
    for o in new_meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = new_meshes[0]
    if len(new_meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.active_object
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    coords = np.array([obj.matrix_world @ v.co for v in obj.data.vertices], dtype=np.float64)
    center = coords.mean(axis=0)
    obj.location -= Vector(center.tolist())
    bpy.context.view_layer.update()
    coords = np.array([obj.matrix_world @ v.co for v in obj.data.vertices], dtype=np.float64)
    extent = float((coords.max(axis=0) - coords.min(axis=0)).max())
    if extent < 1e-8:
        raise RuntimeError(f"degenerate mesh extent: {mesh_path}")
    obj.scale *= target_extent / extent
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    coords = np.array([obj.matrix_world @ v.co for v in obj.data.vertices], dtype=np.float64)
    obj.location -= Vector(coords.mean(axis=0).tolist())
    bpy.context.view_layer.update()
    obj.pass_index = 1
    obj.name = "target"
    print(f"[render] imported {mesh_path.name} verts={len(obj.data.vertices)} extent={target_extent}", flush=True)
    return obj


def _basecolor_source(mat):
    nt = mat.node_tree if mat is not None else None
    if nt is None:
        return None, (0.5, 0.5, 0.5, 1.0)
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None or "Base Color" not in bsdf.inputs:
        return None, (0.5, 0.5, 0.5, 1.0)
    sock = bsdf.inputs["Base Color"]
    factor = tuple(sock.default_value)
    image = None
    if sock.is_linked:
        src = sock.links[0].from_node
        if src.type == "TEX_IMAGE" and src.image is not None:
            image = src.image
    return image, factor


def simplify_to_matte_albedo(obj):
    """Keep each material's base color and drop metal, normal, and roughness maps."""
    for mat in obj.data.materials:
        if mat is None:
            continue
        mat.use_nodes = True
        image, factor = _basecolor_source(mat)
        nt = mat.node_tree
        for node in list(nt.nodes):
            nt.nodes.remove(node)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = 0.88
        bsdf.inputs["Metallic"].default_value = 0.0
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.05
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        if image is not None:
            tex = nt.nodes.new("ShaderNodeTexImage")
            tex.image = image
            coord = nt.nodes.new("ShaderNodeTexCoord")
            nt.links.new(coord.outputs["UV"], tex.inputs["Vector"])
            nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        else:
            bsdf.inputs["Base Color"].default_value = factor


def apply_material_preset(obj, name: str):
    preset = MATERIAL_PRESETS[name]
    if preset is None:
        print(f"[render] material={name} (glTF materials kept)", flush=True)
        return
    if not obj.data.materials:
        obj.data.materials.append(bpy.data.materials.new("preset"))
    for mat in obj.data.materials:
        if mat is None:
            continue
        mat.use_nodes = True
        image, factor = _basecolor_source(mat)
        nt = mat.node_tree
        for node in list(nt.nodes):
            nt.nodes.remove(node)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Metallic"].default_value = preset["metallic"]
        bsdf.inputs["Roughness"].default_value = preset["roughness"]
        bsdf.inputs["Specular IOR Level"].default_value = preset["specular"]
        bsdf.inputs["Coat Weight"].default_value = preset["coat"]
        bsdf.inputs["Coat Roughness"].default_value = 0.03
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        if preset["base"] is not None:
            bsdf.inputs["Base Color"].default_value = (*preset["base"], 1.0)
        elif image is not None:
            tex = nt.nodes.new("ShaderNodeTexImage")
            tex.image = image
            coord = nt.nodes.new("ShaderNodeTexCoord")
            nt.links.new(coord.outputs["UV"], tex.inputs["Vector"])
            nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        else:
            bsdf.inputs["Base Color"].default_value = factor
    print(f"[render] material={name}", flush=True)


def _paint_bsdf(nt, value: float):
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (value, value, value, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.9
    bsdf.inputs["Specular IOR Level"].default_value = 0.05
    return bsdf


def add_speckle_paint(obj, scale: float, seed: int, threshold: float):
    """Print matte dark and light dots on top of whatever surface the object has.

    Between the dots the original shader is untouched, so reflections stay.
    """
    if not obj.data.materials:
        obj.data.materials.append(bpy.data.materials.new("speckle_base"))
    for mat in obj.data.materials:
        if mat is None:
            continue
        mat.use_nodes = True
        nt = mat.node_tree
        out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None)
        out = out or next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
        if out is None:
            out = nt.nodes.new("ShaderNodeOutputMaterial")
        surface = out.inputs["Surface"]
        if surface.is_linked:
            original = surface.links[0].from_socket
        else:
            original = _paint_bsdf(nt, 0.5).outputs["BSDF"]
        coord = nt.nodes.new("ShaderNodeTexCoord")
        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.inputs["Location"].default_value = (0.17 * (seed + 1), 0.31 * (seed + 2), 0.47 * (seed + 3))
        mapping.inputs["Scale"].default_value = (scale, scale, scale)
        link(nt, coord.outputs["Object"], mapping.inputs["Vector"])
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.noise_dimensions = "3D"
        noise.noise_type = "FBM"
        noise.inputs["Scale"].default_value = 1.0
        noise.inputs["Detail"].default_value = 2.0
        noise.inputs["Roughness"].default_value = 0.5
        link(nt, mapping.outputs["Vector"], noise.inputs["Vector"])
        edge = 0.01
        dark = map_range(nt, noise.outputs["Factor"], 0.5 - threshold, 0.5 - threshold - edge, 0.0, 1.0)
        light = map_range(nt, noise.outputs["Factor"], 0.5 + threshold, 0.5 + threshold + edge, 0.0, 1.0)
        mix_dark = nt.nodes.new("ShaderNodeMixShader")
        link(nt, dark, mix_dark.inputs["Fac"])
        link(nt, original, mix_dark.inputs[1])
        link(nt, _paint_bsdf(nt, 0.03).outputs["BSDF"], mix_dark.inputs[2])
        mix_light = nt.nodes.new("ShaderNodeMixShader")
        link(nt, light, mix_light.inputs["Fac"])
        link(nt, mix_dark.outputs["Shader"], mix_light.inputs[1])
        link(nt, _paint_bsdf(nt, 0.85).outputs["BSDF"], mix_light.inputs[2])
        link(nt, mix_light.outputs["Shader"], surface)
    print(f"[render] speckle paint scale={scale} threshold={threshold} seed={seed}", flush=True)


def _object_space_fbm(nt, scale: float, seed: int):
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Location"].default_value = (
        0.17 * (seed + 1),
        0.31 * (seed + 2),
        0.47 * (seed + 3),
    )
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    link(nt, coord.outputs["Object"], mapping.inputs["Vector"])
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.noise_dimensions = "3D"
    noise.noise_type = "FBM"
    noise.inputs["Scale"].default_value = 1.0
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.55
    noise.inputs["Distortion"].default_value = 0.15
    link(nt, mapping.outputs["Vector"], noise.inputs["Vector"])
    return map_range(nt, noise.outputs["Factor"], 0.0, 1.0, -1.0, 1.0)


def overlay_p4_on_materials(obj, alpha: float, scale: float, seed: int):
    """A' = clip(A + alpha * F, 0, 1) on each glTF material. F is object-space."""
    for mat in obj.data.materials:
        if mat is None:
            continue
        mat.use_nodes = True
        image, factor = _basecolor_source(mat)
        nt = mat.node_tree
        for node in list(nt.nodes):
            nt.nodes.remove(node)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Roughness"].default_value = 0.88
        bsdf.inputs["Metallic"].default_value = 0.0
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.05
        nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        if image is not None:
            tex = nt.nodes.new("ShaderNodeTexImage")
            tex.image = image
            coord = nt.nodes.new("ShaderNodeTexCoord")
            nt.links.new(coord.outputs["UV"], tex.inputs["Vector"])
            albedo = tex.outputs["Color"]
        else:
            solid = nt.nodes.new("ShaderNodeCombineColor")
            solid.inputs["Red"].default_value = float(factor[0])
            solid.inputs["Green"].default_value = float(factor[1])
            solid.inputs["Blue"].default_value = float(factor[2])
            albedo = solid.outputs["Color"]
        fac = _object_space_fbm(nt, scale, seed)
        nt.links.new(add_pattern_to_albedo(nt, albedo, fac, alpha), bsdf.inputs["Base Color"])


def new_math(nt, op):
    node = nt.nodes.new("ShaderNodeMath")
    node.operation = op
    node.use_clamp = False
    return node


def map_range(nt, src, from_min, from_max, to_min, to_max):
    node = nt.nodes.new("ShaderNodeMapRange")
    node.clamp = True
    node.inputs["From Min"].default_value = float(from_min)
    node.inputs["From Max"].default_value = float(from_max)
    node.inputs["To Min"].default_value = float(to_min)
    node.inputs["To Max"].default_value = float(to_max)
    link(nt, src, node.inputs["Value"])
    return node.outputs["Result"]


def add_pattern_to_base(nt, f_socket, alpha: float):
    """A' = clip(0.5 + alpha * F, 0, 1) for uniform-gray primitives."""
    return add_pattern_to_albedo(nt, None, f_socket, alpha, base_value=0.5)


def add_pattern_to_albedo(nt, albedo_socket, f_socket, alpha: float, base_value: float = 0.5):
    """A' = clip(A + alpha * F, 0, 1). A is a color socket or constant gray."""
    mul = new_math(nt, "MULTIPLY")
    mul.inputs[1].default_value = float(alpha)
    link(nt, f_socket, mul.inputs[0])
    if albedo_socket is None:
        add = new_math(nt, "ADD")
        add.inputs[1].default_value = float(base_value)
        link(nt, mul.outputs[0], add.inputs[0])
        return map_range(nt, add.outputs[0], 0.0, 1.0, 0.0, 1.0)
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    link(nt, albedo_socket, sep.inputs["Color"])
    channels = []
    for name in ("Red", "Green", "Blue"):
        add = new_math(nt, "ADD")
        link(nt, sep.outputs[name], add.inputs[0])
        link(nt, mul.outputs[0], add.inputs[1])
        channels.append(map_range(nt, add.outputs[0], 0.0, 1.0, 0.0, 1.0))
    comb = nt.nodes.new("ShaderNodeCombineColor")
    link(nt, channels[0], comb.inputs["Red"])
    link(nt, channels[1], comb.inputs["Green"])
    link(nt, channels[2], comb.inputs["Blue"])
    return comb.outputs["Color"]


def load_uv_albedo(nt, texture_path: str):
    img = bpy.data.images.load(str(texture_path))
    img.colorspace_settings.name = "sRGB"
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Linear"
    tex.extension = "REPEAT"
    coord = nt.nodes.new("ShaderNodeTexCoord")
    link(nt, coord.outputs["UV"], tex.inputs["Vector"])
    return tex.outputs["Color"]


def grayscale_color(nt, value_socket):
    comb = nt.nodes.new("ShaderNodeCombineColor")
    link(nt, value_socket, comb.inputs["Red"])
    link(nt, value_socket, comb.inputs["Green"])
    link(nt, value_socket, comb.inputs["Blue"])
    return comb.outputs["Color"]


def make_material(pattern: str, alpha: float, scale: float, seed: int, texture_path: str = ""):
    mat = bpy.data.materials.new(f"mat_{pattern}")
    mat.use_nodes = True
    nt = mat.node_tree
    for node in list(nt.nodes):
        nt.nodes.remove(node)

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (900, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (600, 0)
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.88
    bsdf.inputs["Specular IOR Level"].default_value = 0.05
    link(nt, bsdf.outputs["BSDF"], out.inputs["Surface"])

    gray = (0.5, 0.5, 0.5, 1.0)
    base_albedo = load_uv_albedo(nt, texture_path) if texture_path else None
    if pattern in ("P0", "P4P"):
        if base_albedo is not None:
            link(nt, base_albedo, bsdf.inputs["Base Color"])
        else:
            bsdf.inputs["Base Color"].default_value = gray
        return mat

    coord = nt.nodes.new("ShaderNodeTexCoord")
    coord.location = (-1100, 0)
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.location = (-900, 0)
    mapping.inputs["Location"].default_value = (
        0.17 * (seed + 1),
        0.31 * (seed + 2),
        0.47 * (seed + 3),
    )
    mapping_scale = scale * 0.35 if pattern == "P3" else scale
    mapping.inputs["Scale"].default_value = (mapping_scale, mapping_scale, mapping_scale)
    link(nt, coord.outputs["Object"], mapping.inputs["Vector"])

    fac_socket = None
    color_socket = None

    if pattern == "P2":
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        link(nt, mapping.outputs["Vector"], sep.inputs["Vector"])
        sx = new_math(nt, "SINE")
        sy = new_math(nt, "SINE")
        sz = new_math(nt, "SINE")
        link(nt, sep.outputs["X"], sx.inputs[0])
        link(nt, sep.outputs["Y"], sy.inputs[0])
        link(nt, sep.outputs["Z"], sz.inputs[0])
        mul1 = new_math(nt, "MULTIPLY")
        mul2 = new_math(nt, "MULTIPLY")
        link(nt, sx.outputs[0], mul1.inputs[0])
        link(nt, sy.outputs[0], mul1.inputs[1])
        link(nt, mul1.outputs[0], mul2.inputs[0])
        link(nt, sz.outputs[0], mul2.inputs[1])
        fac_socket = mul2.outputs[0]

    elif pattern == "P3":
        vor = nt.nodes.new("ShaderNodeTexVoronoi")
        vor.voronoi_dimensions = "3D"
        vor.feature = "F1"
        vor.distance = "EUCLIDEAN"
        vor.inputs["Randomness"].default_value = 1.0
        vor.inputs["Scale"].default_value = 1.0
        link(nt, mapping.outputs["Vector"], vor.inputs["Vector"])
        to01 = nt.nodes.new("ShaderNodeMapRange")
        to01.clamp = True
        to01.inputs["From Min"].default_value = 0.0
        to01.inputs["From Max"].default_value = 0.22
        to01.inputs["To Min"].default_value = 1.0
        to01.inputs["To Max"].default_value = 0.0
        link(nt, vor.outputs["Distance"], to01.inputs["Value"])
        fac_socket = to01.outputs["Result"]

    elif pattern == "P4":
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.noise_dimensions = "3D"
        noise.noise_type = "FBM"
        noise.inputs["Scale"].default_value = 1.0
        noise.inputs["Detail"].default_value = 6.0
        noise.inputs["Roughness"].default_value = 0.55
        noise.inputs["Distortion"].default_value = 0.15
        link(nt, mapping.outputs["Vector"], noise.inputs["Vector"])
        fac_socket = map_range(nt, noise.outputs["Factor"], 0.0, 1.0, -1.0, 1.0)

    elif pattern == "P5":
        weights = (0.5, 0.3, 0.2)
        scale_mul = (1.0, 4.0, 16.0)
        factors = []
        for i, sm in enumerate(scale_mul):
            nmap = nt.nodes.new("ShaderNodeMapping")
            nmap.inputs["Scale"].default_value = (sm, sm, sm)
            link(nt, mapping.outputs["Vector"], nmap.inputs["Vector"])
            noise = nt.nodes.new("ShaderNodeTexNoise")
            noise.noise_dimensions = "3D"
            noise.noise_type = "FBM"
            noise.inputs["Scale"].default_value = 1.0
            noise.inputs["Detail"].default_value = 4.0
            noise.inputs["Roughness"].default_value = 0.5
            link(nt, nmap.outputs["Vector"], noise.inputs["Vector"])
            mul = new_math(nt, "MULTIPLY")
            mul.inputs[1].default_value = weights[i]
            link(nt, noise.outputs["Factor"], mul.inputs[0])
            factors.append(mul.outputs[0])
        add1 = new_math(nt, "ADD")
        add2 = new_math(nt, "ADD")
        link(nt, factors[0], add1.inputs[0])
        link(nt, factors[1], add1.inputs[1])
        link(nt, add1.outputs[0], add2.inputs[0])
        link(nt, factors[2], add2.inputs[1])
        fac_socket = map_range(nt, add2.outputs[0], 0.0, 1.0, -1.0, 1.0)

    elif pattern == "P6":
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        link(nt, mapping.outputs["Vector"], sep.inputs["Vector"])
        axes = []
        coeffs = [(1.7, 2.9, 1.1), (2.3, 0.7, 3.1), (0.9, 3.3, 1.9)]
        for a, b, c in coeffs:
            mx = new_math(nt, "MULTIPLY")
            my = new_math(nt, "MULTIPLY")
            mz = new_math(nt, "MULTIPLY")
            mx.inputs[1].default_value = a
            my.inputs[1].default_value = b
            mz.inputs[1].default_value = c
            link(nt, sep.outputs["X"], mx.inputs[0])
            link(nt, sep.outputs["Y"], my.inputs[0])
            link(nt, sep.outputs["Z"], mz.inputs[0])
            add1 = new_math(nt, "ADD")
            add2 = new_math(nt, "ADD")
            link(nt, mx.outputs[0], add1.inputs[0])
            link(nt, my.outputs[0], add1.inputs[1])
            link(nt, add1.outputs[0], add2.inputs[0])
            link(nt, mz.outputs[0], add2.inputs[1])
            sine = new_math(nt, "SINE")
            link(nt, add2.outputs[0], sine.inputs[0])
            axes.append(sine.outputs[0])
        comb = nt.nodes.new("ShaderNodeCombineColor")
        link(nt, add_pattern_to_base(nt, axes[0], alpha), comb.inputs["Red"])
        link(nt, add_pattern_to_base(nt, axes[1], alpha), comb.inputs["Green"])
        link(nt, add_pattern_to_base(nt, axes[2], alpha), comb.inputs["Blue"])
        color_socket = comb.outputs["Color"]
    else:
        raise ValueError(pattern)

    if color_socket is not None:
        albedo = color_socket
    elif base_albedo is not None:
        albedo = add_pattern_to_albedo(nt, base_albedo, fac_socket, alpha)
    else:
        albedo = grayscale_color(nt, add_pattern_to_base(nt, fac_socket, alpha))
    link(nt, albedo, bsdf.inputs["Base Color"])
    return mat


def setup_world(gray: float):
    world = bpy.data.worlds[0]
    nt = world.node_tree
    bg = next(n for n in nt.nodes if n.type == "BACKGROUND")
    bg.inputs[0].default_value = (gray, gray, gray, 1.0)
    bg.inputs[1].default_value = 1.0


def setup_world_black_hdri(path: str, rotation_deg: float, strength: float):
    """Light the object with a captured HDRI. Camera rays see pure black."""
    hdri_path = Path(path)
    if not hdri_path.exists():
        raise FileNotFoundError(hdri_path)
    world = bpy.data.worlds[0]
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value = (0.0, 0.0, math.radians(float(rotation_deg)))
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    image = bpy.data.images.load(str(hdri_path), check_existing=True)
    for colorspace in ("Linear Rec.709", "Linear"):
        try:
            image.colorspace_settings.name = colorspace
            break
        except TypeError:
            continue
    env.image = image
    bg_lit = nt.nodes.new("ShaderNodeBackground")
    bg_lit.inputs["Strength"].default_value = float(strength)
    bg_black = nt.nodes.new("ShaderNodeBackground")
    bg_black.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    bg_black.inputs["Strength"].default_value = 1.0
    light = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    link(nt, coord.outputs["Generated"], mapping.inputs["Vector"])
    link(nt, mapping.outputs["Vector"], env.inputs["Vector"])
    link(nt, env.outputs["Color"], bg_lit.inputs["Color"])
    link(nt, light.outputs["Is Camera Ray"], mix.inputs["Fac"])
    link(nt, bg_lit.outputs["Background"], mix.inputs[1])
    link(nt, bg_black.outputs["Background"], mix.inputs[2])
    link(nt, mix.outputs["Shader"], out.inputs["Surface"])
    print(
        f"[render] black background hdri={hdri_path.name} rot={rotation_deg:.1f} strength={strength}",
        flush=True,
    )


def setup_lights(key_energy: float, fill_energy: float):
    bpy.ops.object.light_add(type="AREA", location=(1.9, -1.7, 2.4))
    key = bpy.context.active_object
    key.data.energy = key_energy
    key.data.size = 3.2
    key.rotation_euler = (0.7, 0.2, 0.9)
    bpy.ops.object.light_add(type="AREA", location=(-2.1, -1.1, 1.7))
    fill = bpy.context.active_object
    fill.data.energy = fill_energy
    fill.data.size = 4.0
    fill.rotation_euler = (1.1, -0.4, -0.8)


def _wire_spot_gobo(nt, gobo_scale: float, seed: int, alpha: float, contrast_lo: float, contrast_hi: float, gain: float):
    """Project FBM speckle through a spot cone. I = clip(0.5 + gain * F, 0, 1).

    Cycles spot Texture Coordinate > Normal is the projector-plane mapping
    (verified: checkerboard is perspective-correct across faces).
    """
    emission = next(n for n in nt.nodes if n.type == "EMISSION")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (gobo_scale, gobo_scale, 1.0)
    mapping.inputs["Location"].default_value = (
        0.17 * (seed + 1),
        0.31 * (seed + 2),
        0.0,
    )
    link(nt, coord.outputs["Normal"], mapping.inputs["Vector"])

    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.noise_dimensions = "2D"
    noise.noise_type = "FBM"
    noise.inputs["Scale"].default_value = 1.0
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.55
    noise.inputs["Distortion"].default_value = 0.15
    link(nt, mapping.outputs["Vector"], noise.inputs["Vector"])
    contrast = map_range(nt, noise.outputs["Factor"], float(contrast_lo), float(contrast_hi), 0.0, 1.0)
    fac = map_range(nt, contrast, 0.0, 1.0, -1.0, 1.0)
    mul = new_math(nt, "MULTIPLY")
    mul.inputs[1].default_value = float(gain if gain > 0 else max(alpha, 0.45))
    link(nt, fac, mul.inputs[0])
    add = new_math(nt, "ADD")
    add.inputs[1].default_value = 0.5
    link(nt, mul.outputs[0], add.inputs[0])
    intensity = map_range(nt, add.outputs[0], 0.0, 1.0, 0.0, 1.0)
    link(nt, grayscale_color(nt, intensity), emission.inputs["Color"])
    emission.inputs["Strength"].default_value = 1.0


def setup_projectors(args):
    """Fixed world-space speckle spots. Albedo unchanged; pattern is arriving light.

    Six or more lamps use two rings: a low ring for vertical walls and a high ring
    for upward faces. Fewer lamps stay on one ring at projector_elevation_deg.
    """
    count = max(int(args.projector_count), 1)
    radius = float(args.projector_radius)
    spot = math.radians(args.projector_spot_deg)
    gobo_scale = float(args.projector_scale) if args.projector_scale > 0 else float(args.scale)
    look_at = Vector((0.0, 0.0, 0.05))
    if count >= 6:
        n_high = max(2, count // 4)
        n_low = count - n_high
        rings = [
            ("low", n_low, math.radians(args.projector_elevation_low_deg), 0.0),
            ("high", n_high, math.radians(args.projector_elevation_deg), math.pi / n_high),
        ]
    else:
        rings = [("mid", count, math.radians(args.projector_elevation_deg), 0.0)]
    infos = []
    index = 0
    for ring_name, n, elev, azim0 in rings:
        for k in range(n):
            azim = azim0 + (2.0 * math.pi * k) / n + math.radians(20.0)
            loc = (
                radius * math.cos(elev) * math.cos(azim),
                radius * math.cos(elev) * math.sin(azim),
                radius * math.sin(elev),
            )
            bpy.ops.object.light_add(type="SPOT", location=loc)
            lamp = bpy.context.active_object
            lamp.name = f"projector_{index}"
            lamp.rotation_euler = (look_at - lamp.location).to_track_quat("-Z", "Y").to_euler()
            lamp.data.energy = float(args.projector_energy)
            lamp.data.spot_size = spot
            lamp.data.spot_blend = 0.12
            lamp.data.shadow_soft_size = 0.03
            lamp.data.use_nodes = True
            _wire_spot_gobo(
                lamp.data.node_tree,
                gobo_scale,
                int(args.seed) + index * 17,
                args.alpha,
                args.projector_contrast_lo,
                args.projector_contrast_hi,
                args.projector_gain,
            )
            bpy.context.view_layer.update()
            infos.append(
                {
                    "name": lamp.name,
                    "ring": ring_name,
                    "location": list(loc),
                    "elevation_deg": math.degrees(elev),
                    "energy": float(args.projector_energy),
                    "spot_deg": float(args.projector_spot_deg),
                    "gobo_scale": gobo_scale,
                    "seed": int(args.seed) + index * 17,
                }
            )
            print(
                f"[render] projector {index} ring={ring_name} azim={math.degrees(azim):.1f} "
                f"elev={math.degrees(elev):.1f} energy={args.projector_energy}",
                flush=True,
            )
            index += 1
    return infos


def frame_size(args):
    width = int(args.resolution_x) or int(args.resolution)
    height = int(args.resolution_y) or int(args.resolution)
    return width, height


def use_eevee(args) -> bool:
    return str(args.engine).upper() in {"BLENDER_EEVEE", "EEVEE", "BLENDER_EEVEE_NEXT"}


def setup_cycles(args):
    scene = bpy.context.scene
    if use_eevee(args):
        scene.render.engine = "BLENDER_EEVEE"
        ee = scene.eevee
        ee.taa_render_samples = int(args.taa_samples)
        ee.use_raytracing = bool(args.raytrace)
        if args.raytrace:
            ee.ray_tracing_method = "SCREEN"
            rt = ee.ray_tracing_options
            rt.resolution_scale = "1"
            rt.screen_trace_quality = 1.0
            rt.trace_max_roughness = 0.5
            ee.use_fast_gi = True
            ee.use_shadows = True
            world = scene.world or bpy.data.worlds[0]
            world.use_sun_shadow = True
            world.probe_resolution = str(args.probe_resolution)
        scene.render.film_transparent = bool(args.hdri)
        print(f"[render] engine=EEVEE raytracing={'on' if args.raytrace else 'off'} taa={args.taa_samples}", flush=True)
    else:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = int(args.samples)
        scene.cycles.use_denoising = bool(args.denoise)
        if args.denoise:
            scene.cycles.denoiser = "OPENIMAGEDENOISE"
        scene.cycles.use_adaptive_sampling = True
    width, height = frame_size(args)
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.media_type = "IMAGE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    if not use_eevee(args):
        scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.display_settings.display_device = "sRGB"

    prefs = bpy.context.preferences.addons["cycles"].preferences
    used_gpu = use_eevee(args)
    if (not use_eevee(args)) and args.device.upper() == "GPU":
        for compute in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = compute
                prefs.get_devices()
                enabled = False
                for d in prefs.devices:
                    d.use = d.type == compute
                    enabled = enabled or d.use
                if enabled:
                    scene.cycles.device = "GPU"
                    used_gpu = True
                    print(f"[render] Cycles device={compute}", flush=True)
                    break
            except Exception as exc:
                print(f"[render] {compute} failed: {exc}", flush=True)
    if not used_gpu:
        scene.cycles.device = "CPU"
        print("[render] Cycles device=CPU", flush=True)

    vl = scene.view_layers[0]
    if args.rgb_only:
        scene.render.use_compositing = False
    else:
        vl.use_pass_z = True
        vl.use_pass_object_index = True
        scene.render.use_compositing = True
    scene.render.use_sequencer = False
    return scene


def setup_compositor(scene, tmp_dir: Path):
    tree = bpy.data.node_groups.new("VTO_Comp", "CompositorNodeTree")
    scene.compositing_node_group = tree
    rlayers = tree.nodes.new("CompositorNodeRLayers")
    g_out = tree.nodes.new("NodeGroupOutput")
    tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    tree.links.new(rlayers.outputs["Image"], g_out.inputs["Image"])

    fout = tree.nodes.new("CompositorNodeOutputFile")
    fout.directory = str(tmp_dir) + os.sep
    fout.file_name = "pass"
    fout.format.media_type = "IMAGE"
    fout.format.file_format = "OPEN_EXR"
    fout.format.color_depth = "32"
    fout.file_output_items.new("FLOAT", "depth")
    fout.file_output_items.new("FLOAT", "mask")
    tree.links.new(rlayers.outputs["Depth"], fout.inputs["depth"])
    tree.links.new(rlayers.outputs["Object Index"], fout.inputs["mask"])
    return fout


def camera_intrinsics(cam, scene):
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    fx = cam.data.lens / cam.data.sensor_width * w
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def blender_c2w_opencv(cam):
    W = np.array(cam.matrix_world, dtype=np.float64)
    R_bcam2cv = np.diag([1.0, -1.0, -1.0])
    R_wc = W[:3, :3] @ R_bcam2cv
    t_wc = W[:3, 3]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_wc
    T[:3, 3] = t_wc
    return T


def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def set_camera_on_trajectory(cam, t, args, look_at=(0.0, 0.0, 0.05)):
    azim = math.radians(args.azimuth_start_deg) + (t / args.num_views) * math.radians(args.azimuth_span_deg)
    elev = math.radians(args.elevation_deg + args.elevation_amp_deg * math.sin(2.0 * math.pi * t / args.num_views))
    x = args.radius * math.cos(elev) * math.cos(azim)
    y = args.radius * math.cos(elev) * math.sin(azim)
    z = args.radius * math.sin(elev)
    cam.location = (x, y, z)
    direction = Vector(look_at) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()


def load_exr_channel(path: Path) -> np.ndarray:
    img = bpy.data.images.load(str(path))
    w, h = img.size
    pix = np.array(img.pixels[:], dtype=np.float32)
    ch = int(pix.size / (w * h))
    arr = pix.reshape(h, w, ch)[:, :, 0]
    arr = np.flipud(arr).copy()
    bpy.data.images.remove(img)
    return arr


def _project_corners(cam, scene, corners):
    bpy.context.view_layer.update()
    world = np.array(cam.matrix_world, dtype=np.float64)
    rotation = world[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    origin = world[:3, 3]
    cam_pts = (corners - origin) @ rotation
    depth = cam_pts[:, 2]
    if np.any(depth <= 1e-4):
        return None
    intrinsics = camera_intrinsics(cam, scene)
    x = intrinsics[0][0] * cam_pts[:, 0] / depth + intrinsics[0][2]
    y = intrinsics[1][1] * cam_pts[:, 1] / depth + intrinsics[1][2]
    return x, y


def fit_radius_to_frame(cam, obj, args, scene):
    """Place the longest projected side at frame_fill of the image height, without cropping."""
    corners = np.array([obj.matrix_world @ Vector(corner) for corner in obj.bound_box], dtype=np.float64)
    width = int(scene.render.resolution_x)
    height = int(scene.render.resolution_y)
    margin = 0.02 * min(width, height)
    target = float(args.frame_fill)

    def measure(radius):
        args.radius = float(radius)
        longest = 0.0
        for frame in range(int(args.num_views)):
            set_camera_on_trajectory(cam, frame, args)
            projected = _project_corners(cam, scene, corners)
            if projected is None:
                return False, None
            x, y = projected
            inside = (
                np.all(x >= margin)
                and np.all(x <= width - margin)
                and np.all(y >= margin)
                and np.all(y <= height - margin)
            )
            if not inside:
                return False, None
            longest = max(longest, float(x.max() - x.min()), float(y.max() - y.min()))
        return True, longest / height

    near, far = 0.8, 40.0
    fits, _fill = measure(far)
    if not fits:
        args.radius = far
        print("[render] frame fit failed at far radius", flush=True)
        return
    for _ in range(22):
        mid = 0.5 * (near + far)
        fits, fill = measure(mid)
        if fits and fill <= target:
            far = mid
        else:
            near = mid
    args.radius = far
    fits, fill = measure(args.radius)
    print(
        f"[render] frame fill target={target:.3f} achieved={0.0 if fill is None else fill:.3f} "
        f"radius={args.radius:.3f}",
        flush=True,
    )


def export_mesh(obj, path: Path):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    try:
        bpy.ops.wm.obj_export(filepath=str(path), export_selected_objects=True)
    except Exception:
        bpy.ops.export_scene.obj(filepath=str(path), use_selection=True)


def main():
    args = parse_args()
    if args.mesh:
        args.mesh = str(Path(args.mesh).resolve())
    if args.texture:
        args.texture = str(Path(args.texture).resolve())
    out_dir = Path(args.output_dir)
    rgb_dir = out_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = out_dir / "depth"
    mask_dir = out_dir / "mask"
    tmp_dir = out_dir / "_tmp_passes"
    if not args.rgb_only:
        for d in (depth_dir, mask_dir, tmp_dir):
            d.mkdir(parents=True, exist_ok=True)

    reset_scene()
    glb = bool(args.mesh) and Path(args.mesh).suffix.lower() in {".glb", ".gltf"}
    if args.mesh:
        obj = import_mesh(args.mesh)
    else:
        if args.object not in OBJECT_BUILDERS:
            raise KeyError(f"unknown primitive object {args.object}; pass --mesh for scanned assets")
        obj = OBJECT_BUILDERS[args.object]()
    if args.material:
        apply_material_preset(obj, args.material)
    elif glb and args.pattern == "P0":
        simplify_to_matte_albedo(obj)
    elif glb and args.pattern == "P4":
        overlay_p4_on_materials(obj, args.alpha, args.scale, args.seed)
    else:
        mat = make_material(args.pattern, args.alpha, args.scale, args.seed, texture_path=args.texture)
        obj.data.materials.clear()
        obj.data.materials.append(mat)
    if args.speckle:
        add_speckle_paint(obj, args.speckle_scale, args.seed, args.speckle_threshold)

    if args.hdri:
        setup_world_black_hdri(args.hdri, args.hdri_rotation_deg, args.hdri_strength)
    else:
        setup_world(args.world_gray)
        env_scale = float(args.projector_env_scale) if args.pattern == "P4P" else 1.0
        setup_lights(args.key_light_energy * env_scale, args.fill_light_energy * env_scale)
    projectors = setup_projectors(args) if args.pattern == "P4P" else []
    scene = setup_cycles(args)
    if args.hdri:
        scene.cycles.sample_clamp_indirect = 10.0

    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.data.lens = args.lens_mm
    cam.data.sensor_width = args.sensor_width_mm
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.clip_start = args.clip_start
    cam.data.clip_end = args.clip_end
    scene.camera = cam
    if args.frame_fill > 0:
        fit_radius_to_frame(cam, obj, args, scene)
    if not args.rgb_only:
        setup_compositor(scene, tmp_dir)

    frames = []
    K = None
    for t in range(args.num_views):
        scene.frame_set(t)
        set_camera_on_trajectory(cam, t, args)
        if K is None:
            K = camera_intrinsics(cam, scene)
        T_wc = blender_c2w_opencv(cam)
        T_cw = invert_T(T_wc)
        rgb_path = rgb_dir / f"{t:04d}.png"
        scene.render.filepath = str(rgb_path.with_suffix(""))
        bpy.ops.render.render(write_still=True)

        if not args.rgb_only:
            depth_exr = tmp_dir / "passdepth.exr"
            mask_exr = tmp_dir / "passmask.exr"
            if not depth_exr.exists() or not mask_exr.exists():
                raise FileNotFoundError(f"missing compositor EXR for frame {t}: {list(tmp_dir.iterdir())}")
            depth = load_exr_channel(depth_exr)
            mask = load_exr_channel(mask_exr)
            depth[~np.isfinite(depth)] = 0.0
            depth[depth > 1.0e8] = 0.0
            mask_u8 = (mask > 0.5).astype(np.uint8) * 255
            np.save(depth_dir / f"{t:04d}.npy", depth.astype(np.float32))
            np.save(mask_dir / f"{t:04d}.npy", mask_u8)

        frames.append(
            {
                "id": t,
                "file": f"rgb/{t:04d}.png",
                "T_wc": T_wc.tolist(),
                "T_cw": T_cw.tolist(),
            }
        )
        print(f"[render] {args.object} {args.pattern} frame {t:02d}/{args.num_views}", flush=True)

    camera_gt = {
        "convention": "opencv",
        "depth_type": "camera_plane_z",
        "width": int(scene.render.resolution_x),
        "height": int(scene.render.resolution_y),
        "K": K,
        "object": args.object,
        "pattern": args.pattern,
        "material": args.material,
        "speckle": bool(args.speckle),
        "speckle_scale": args.speckle_scale,
        "speckle_threshold": args.speckle_threshold,
        "engine": scene.render.engine,
        "raytrace": bool(args.raytrace),
        "alpha": args.alpha,
        "scale": args.scale,
        "seed": args.seed,
        "radius": args.radius,
        "lens_mm": args.lens_mm,
        "azimuth_span_deg": args.azimuth_span_deg,
        "azimuth_start_deg": args.azimuth_start_deg,
        "elevation_deg": args.elevation_deg,
        "elevation_amp_deg": args.elevation_amp_deg,
        "background": "black" if args.hdri else "gray",
        "hdri": args.hdri,
        "hdri_rotation_deg": args.hdri_rotation_deg,
        "hdri_strength": args.hdri_strength,
        "projectors": projectors,
        "projector_env_scale": float(args.projector_env_scale) if args.pattern == "P4P" else 1.0,
        "frames": frames,
    }
    with open(out_dir / "camera_gt.json", "w", encoding="utf-8") as f:
        json.dump(camera_gt, f, indent=2)

    if args.write_geometry and not args.rgb_only:
        export_mesh(obj, out_dir / "mesh.obj")

    if not args.rgb_only:
        for p in tmp_dir.glob("*"):
            p.unlink()
        tmp_dir.rmdir()
    print("[render] done", out_dir, flush=True)


if __name__ == "__main__":
    main()
    sys.exit(0)
