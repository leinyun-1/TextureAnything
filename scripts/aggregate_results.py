from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from config_utils import ROOT, load_yaml


QUESTIONS = [
    "Surface-attached pattern 是否显著提升 correct correspondence？",
    "提升是否明显超过 per-frame 2D random noise（P1）？",
    "Correspondence 提升是否真正转化成 COLMAP registration / pose improvement？",
    "哪种 pattern 最有效？",
    "是否值得进入生成模型阶段？",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="configs/smoke.yaml")
    p.add_argument("--results_root")
    return p.parse_args()


def yn(cond: bool) -> str:
    return "yes" if cond else "no / not yet"


def main():
    args = parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    exp = load_yaml(ROOT / args.experiment)
    results_root = ROOT / (args.results_root or exp["results_root"])
    df = pd.read_csv(results_root / "pair_metrics.csv")
    summary = (
        df.groupby(["pattern_type"], as_index=False)
        .agg(
            num_pairs=("num_correct_matches", "count"),
            keypoints=("num_keypoints_i", "mean"),
            matches=("num_matches", "mean"),
            correct_matches=("num_correct_matches", "mean"),
            precision=("match_precision", "mean"),
            geometric_inliers=("num_geometric_inliers", "mean"),
            repeatability=("repeatability", "mean"),
            surface_coverage=("surface_coverage", "mean"),
            interior_correct=("interior_correct_matches", "mean"),
            interior_coverage=("interior_coverage", "mean"),
        )
        .sort_values("correct_matches", ascending=False)
    )
    summary_path = results_root / "summary.csv"
    summary.to_csv(summary_path, index=False)

    by = {row.pattern_type: row for row in summary.itertuples(index=False)}
    p0 = by.get("P0")
    p1 = by.get("P1")
    attached = [p for p in ("P3", "P4", "P5") if p in by]
    best_attached = max(attached, key=lambda p: by[p].correct_matches) if attached else None
    best_all = summary.iloc[0].pattern_type if len(summary) else None

    q1 = bool(best_attached and p0 and by[best_attached].correct_matches > 1.5 * max(p0.correct_matches, 1e-6))
    if p1 is None:
        q2 = None
        q2_text = "n/a — P1 not in this experiment"
    else:
        q2 = bool(best_attached and by[best_attached].correct_matches > 1.3 * max(p1.correct_matches, 1e-6))
        q2_text = yn(q2)

    sfm_path = results_root / "sfm_metrics.csv"
    sfm_summary = None
    q3 = False
    q3_text = "not evaluated (mapper has not been run)"
    if sfm_path.exists():
        sfm = pd.read_csv(sfm_path)
        sfm_summary = (
            sfm.groupby(["pattern_type"], as_index=False)
            .agg(
                scenes=("scene_id", "count"),
                registration_ratio=("registration_ratio", "mean"),
                rotation_error_median=("rotation_error_median", "median"),
                camera_center_error_median=("camera_center_error_median", "median"),
                num_sparse_points=("num_sparse_points", "mean"),
                mean_track_length=("mean_track_length", "mean"),
                mean_reprojection_error=("mean_reprojection_error", "mean"),
            )
            .sort_values("registration_ratio", ascending=False)
        )
        sfm_by = {row.pattern_type: row for row in sfm_summary.itertuples(index=False)}
        best_sfm = best_attached if best_attached in sfm_by else (sfm_summary.iloc[0].pattern_type if len(sfm_summary) else None)
        p0s = sfm_by.get("P0")
        p1s = sfm_by.get("P1")
        bs = sfm_by.get(best_sfm) if best_sfm else None
        if bs is not None:
            base_reg = max((p0s.registration_ratio if p0s is not None else 0.0), (p1s.registration_ratio if p1s is not None else 0.0))
            reg_gain = bs.registration_ratio - base_reg >= 0.20
            rot_drop = False
            if p0s is not None and pd.notna(bs.rotation_error_median) and pd.notna(p0s.rotation_error_median) and p0s.rotation_error_median > 0:
                rot_drop = bs.rotation_error_median <= 0.7 * p0s.rotation_error_median
            q3 = bool(reg_gain or rot_drop)
            q3_text = yn(q3)
        else:
            q3_text = "no / mapper produced no usable comparison"
        sfm_summary.to_csv(results_root / "sfm_summary.csv", index=False)

    per_obj_sfm = None
    held_sfm = None
    held_dense = None
    if sfm_path.exists():
        sfm_full = pd.read_csv(sfm_path)
        if {"scene_id", "pattern_type", "registration_ratio"}.issubset(sfm_full.columns):
            per_obj_sfm = (
                sfm_full.pivot_table(
                    index="scene_id",
                    columns="pattern_type",
                    values=["registration_ratio", "num_registered", "num_sparse_points"],
                    aggfunc="mean",
                )
            )
        held = exp.get("held_out") or []
        if held:
            held_sfm = (
                sfm_full[sfm_full["scene_id"].isin(held)]
                .groupby(["pattern_type"], as_index=False)
                .agg(
                    scenes=("scene_id", "count"),
                    registration_ratio=("registration_ratio", "mean"),
                    rotation_error_median=("rotation_error_median", "median"),
                    camera_center_error_median=("camera_center_error_median", "median"),
                    num_sparse_points=("num_sparse_points", "mean"),
                )
                .sort_values("registration_ratio", ascending=False)
            )

    p4p = by.get("P4P")
    q_proj = False
    q_proj_text = "n/a — P4P not in this experiment"
    q_retain = None
    q_retain_text = "n/a — P4P not in this experiment"
    if p4p is not None and p0 is not None:
        q_proj = bool(p4p.correct_matches > 1.5 * max(p0.correct_matches, 1e-6))
        q_proj_text = yn(q_proj)
        if "P4" in by:
            gain_oracle = by["P4"].correct_matches - p0.correct_matches
            gain_proj = p4p.correct_matches - p0.correct_matches
            q_retain = bool(gain_oracle <= 1e-6 or gain_proj >= 0.4 * gain_oracle)
            q_retain_text = yn(q_retain)

    q_proj_sfm = None
    q_proj_sfm_text = "n/a — P4P not in this experiment"
    if p4p is not None and sfm_summary is not None:
        sfm_by_local = {row.pattern_type: row for row in sfm_summary.itertuples(index=False)}
        p0s_local = sfm_by_local.get("P0")
        p4ps_local = sfm_by_local.get("P4P")
        if p0s_local is not None and p4ps_local is not None:
            q_proj_sfm = bool(p4ps_local.registration_ratio - p0s_local.registration_ratio >= 0.15)
            q_proj_sfm_text = yn(q_proj_sfm)

    q5 = q1 and q3 and (True if q2 is None else q2)
    if p4p is not None:
        q5 = q5 and q_proj and (q_retain is not False) and (True if q_proj_sfm is None else q_proj_sfm)
    q5_text = yn(q5) if q3_text.startswith("yes") or q3_text.startswith("no") else "not yet — Stage C SfM is required before any generative-model decision"
    if q1 and (q2 is not False) and not q3:
        q5_text = "not yet — correspondence improved, but SfM gate is not clearly passed"
    if p4p is not None and not q_proj:
        q5_text = "not yet — albedo oracle helped, but virtual projector has not clearly beaten P0"
    elif p4p is not None and q_retain is False:
        q5_text = "not yet — projector beats P0, but keeps under 40% of the albedo-oracle correspondence gain"

    dense_summary = None
    dense_path = results_root / "dense_metrics.csv"
    if dense_path.exists():
        dense = pd.read_csv(dense_path)
        dense_summary = (
            dense.groupby(["pattern_type"], as_index=False)
            .agg(
                scenes=("scene_id", "count"),
                num_fused_points=("num_fused_points", "mean"),
                num_fused_near_object=("num_fused_near_object", "mean"),
                accuracy=("accuracy", "mean"),
                completeness=("completeness", "mean"),
                chamfer_distance=("chamfer_distance", "mean"),
            )
            .sort_values("chamfer_distance", ascending=True)
        )
        dense_summary.to_csv(results_root / "dense_summary.csv", index=False)
        held = exp.get("held_out") or []
        if held:
            held_dense = (
                dense[dense["scene_id"].isin(held)]
                .groupby(["pattern_type"], as_index=False)
                .agg(
                    scenes=("scene_id", "count"),
                    num_fused_points=("num_fused_points", "mean"),
                    accuracy=("accuracy", "mean"),
                    completeness=("completeness", "mean"),
                    chamfer_distance=("chamfer_distance", "mean"),
                )
                .sort_values("chamfer_distance", ascending=True)
            )

    lines = [
        f"# {exp.get('name', 'experiment')} report",
        "",
        "## Correspondence summary",
        "",
        summary.to_string(index=False),
        "",
    ]
    if sfm_summary is not None:
        lines += ["## SfM summary", "", sfm_summary.to_string(index=False), ""]
    if per_obj_sfm is not None and len(per_obj_sfm):
        lines += ["## SfM per object (registration)", "", per_obj_sfm.to_string(), ""]
    if held_sfm is not None and len(held_sfm):
        lines += ["## Held-out SfM", "", held_sfm.to_string(index=False), ""]
    if dense_summary is not None:
        lines += [
            "## Dense MVS (GT poses, identical source views)",
            "",
            "Every pattern uses the same GT camera poses and the same neighbor graph. Only RGB texture changes.",
            "Accuracy / completeness / Chamfer are distances to the GT depth-unprojected cloud (capped at 0.2).",
            "",
            dense_summary.to_string(index=False),
            "",
        ]
        if held_dense is not None and len(held_dense):
            lines += ["## Held-out dense MVS", "", held_dense.to_string(index=False), ""]
    lines += [
        "## Gate questions",
        "",
        f"1. {QUESTIONS[0]} **{yn(q1)}**",
        f"2. {QUESTIONS[1]} **{q2_text}**",
        f"3. {QUESTIONS[2]} **{q3_text}**",
        f"4. {QUESTIONS[3]} **{best_all}**"
        + (f" (best surface-attached among P3/P4/P5: {best_attached})" if best_attached else ""),
        f"5. {QUESTIONS[4]} **{q5_text}**",
        "",
    ]
    if p4p is not None:
        lines += [
            "## Projector gates",
            "",
            "P4 = albedo rewrite (oracle). P4P = original albedo + fixed speckle projectors.",
            "",
            f"6. Virtual projector 是否仍优于 P0 correspondence？ **{q_proj_text}**",
            f"7. Projector 是否保住 albedo-oracle 至少 40% 的 correspondence 增益？ **{q_retain_text}**",
            f"8. Virtual projector 是否仍提升 SfM registration（Δreg ≥ 0.15）？ **{q_proj_sfm_text}**",
            "",
        ]
    if (results_root / "reprojection_selfcheck.json").exists():
        checks = json.loads((results_root / "reprojection_selfcheck.json").read_text(encoding="utf-8"))
        mean_err = [c["mean_px"] for c in checks if c.get("mean_px") is not None]
        lines += [
            "## GT reprojection self-check",
            "",
            f"Same-frame unproject/project mean error: {sum(mean_err)/len(mean_err):.4f} px" if mean_err else "no valid pixels",
            "",
        ]
    report_path = results_root / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[aggregate] wrote", summary_path)
    print("[aggregate] wrote", report_path)
    print(summary.to_string(index=False))
    if sfm_summary is not None:
        print(sfm_summary.to_string(index=False))
    if dense_summary is not None:
        print(dense_summary.to_string(index=False))


if __name__ == "__main__":
    main()
