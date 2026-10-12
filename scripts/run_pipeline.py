"""Run one stage of the pair factory.

Stages, in order:

  catalog -> l1 -> l2 -> acquire -> render_p0 -> sfm_p0
          -> render_p4 -> sfm_p4 -> gate -> package

backfill loads the existing GSO scale run into the ledger without new downloads.
status prints how many records sit at each stage.
"""
from __future__ import annotations

import argparse
import sys

from config_utils import ROOT
from pipeline.backfill import run as backfill_run
from pipeline.common import load_pipeline
from pipeline.ledger import Ledger


STAGE_FUNCS = {
    "catalog": ("pipeline.catalog", "run"),
    "l1": ("pipeline.screen_l1", "run"),
    "l2": ("pipeline.screen_l2", "run"),
    "acquire": ("pipeline.acquire", "run"),
    "render_p0": ("pipeline.render_stage", "render_p0"),
    "sfm_p0": ("pipeline.sfm_stage", "sfm_p0"),
    "render_p4": ("pipeline.render_stage", "render_p4"),
    "sfm_p4": ("pipeline.sfm_stage", "sfm_p4"),
    "gate": ("pipeline.gate", "run"),
    "package": ("pipeline.package", "run"),
}

ORDER = list(STAGE_FUNCS)


def _load_func(stage: str):
    import importlib

    module_name, func_name = STAGE_FUNCS[stage]
    return getattr(importlib.import_module(module_name), func_name)


def _print_status(ledger: Ledger):
    counts = ledger.summary()
    total = sum(counts.values())
    print(f"[pipe] ledger {ledger.path} records={total}")
    for stage, n in counts.items():
        print(f"  {stage:16} {n}")


def parse_args():
    p = argparse.ArgumentParser(description="Pair-factory data pipeline.")
    p.add_argument("--config", default="configs/pipeline.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("backfill")
    run_p = sub.add_parser("run")
    run_p.add_argument("--stage", required=True, choices=ORDER)
    run_p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    cfg = load_pipeline(args.config)
    ledger = Ledger(ROOT / cfg["ledger"])
    if args.cmd == "status":
        _print_status(ledger)
        return
    if args.cmd == "backfill":
        result = backfill_run(ledger, cfg)
        print("[pipe] backfill", result, flush=True)
        _print_status(ledger)
        return
    limit = args.limit or None
    result = _load_func(args.stage)(ledger, cfg, limit)
    print(f"[pipe] {args.stage}", result, flush=True)
    _print_status(ledger)


if __name__ == "__main__":
    main()
