"""Run Phase 3: corrected comparisons and calibration.

Full supervisor-aligned Phase-3 run includes:
- primary historical within-region split (seed 42);
- five strict complete-block spatial folds from Phase 2;
- corrected EcoTreeFuseNet-Plus calibration;
- equally calibrated RF, XGBoost, ExtraTrees;
- same-learner ExtraTrees modality comparison;
- class-support diagnostics under strict spatial holdout;
- aggregate mean/sample-SD tables.

This stage does NOT perform Phase-4 bootstrap/McNemar claim adjudication.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from spatial_data import load_and_prepare_data
from comparisons import (
    run_calibrated_baselines,
    run_modality_comparison,
    collect_proposed_results,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "data" / "vegetation_reference.csv"
DEFAULT_OUT = BASE_DIR / "outputs_phase3"


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 corrected comparisons and calibration")
    p.add_argument("--data-path", default=str(DEFAULT_DATA))
    p.add_argument("--output-dir", default=str(DEFAULT_OUT))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--protocol", choices=["primary", "spatial", "both"], default="both")
    p.add_argument("--n-spatial-folds", type=int, default=5)
    p.add_argument("--spatial-grid", type=int, default=5)
    p.add_argument("--stack-folds", type=int, default=5)
    p.add_argument("--fast", action="store_true", help="Smoke test only; not manuscript reporting")
    p.add_argument("--skip-proposed", action="store_true", help="Debug only: skip expensive proposed pipeline")
    return p.parse_args()


def _run_proposed(args, run_dir: Path, split_mode: str, spatial_fold=None):
    if args.skip_proposed:
        return
    cmd = [
        sys.executable, str(BASE_DIR / "train.py"),
        "--data-path", str(Path(args.data_path).resolve()),
        "--output-dir", str(run_dir),
        "--seed", str(args.seed),
        "--split-mode", split_mode,
        "--n-spatial-folds", str(args.n_spatial_folds),
        "--spatial-grid", str(args.spatial_grid),
        "--stack-folds", str(args.stack_folds),
        "--models", "ecofusenet", "ecotreefusenet_plus",
        "--skip-figures",
    ]
    if spatial_fold is not None:
        cmd += ["--spatial-fold", str(spatial_fold)]
    if args.fast:
        cmd.append("--fast")
    print("\n[Phase 3] proposed pipeline:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(BASE_DIR), check=True)


def _one_run(args, run_name: str, split_mode: str, spatial_fold=None):
    run_dir = Path(args.output_dir).resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    _run_proposed(args, run_dir, split_mode, spatial_fold)

    data = load_and_prepare_data(
        csv_path=str(Path(args.data_path).resolve()),
        output_dir=str(run_dir),
        random_state=args.seed,
        split_mode=split_mode,
        spatial_fold=0 if spatial_fold is None else int(spatial_fold),
        n_spatial_folds=args.n_spatial_folds,
        n_grid=args.spatial_grid,
    )

    baseline_df = run_calibrated_baselines(data, str(run_dir), seed=args.seed, fast=args.fast)
    modality_df = run_modality_comparison(data, str(run_dir), seed=args.seed, fast=args.fast, calibrate=True)

    proposed_df = pd.DataFrame()
    if not args.skip_proposed:
        proposed_df = collect_proposed_results(str(run_dir), data=data)

    comparison = pd.concat([baseline_df, proposed_df], ignore_index=True, sort=False)
    comparison.insert(0, "run", run_name)
    comparison.insert(1, "split_mode", split_mode)
    comparison.insert(2, "spatial_fold", -1 if spatial_fold is None else int(spatial_fold))
    comparison.to_csv(run_dir / "phase3_main_comparison.csv", index=False)

    modality_df.insert(0, "run", run_name)
    modality_df.insert(1, "split_mode", split_mode)
    modality_df.insert(2, "spatial_fold", -1 if spatial_fold is None else int(spatial_fold))
    modality_df.to_csv(run_dir / "phase3_modality_comparison.csv", index=False)

    split = data["split_audit"]
    train_labels = set(map(int, np.unique(data["y_fine_train"])))
    test_labels = set(map(int, np.unique(data["y_fine_test"])))
    supported = np.isin(data["y_fine_test"], list(train_labels))
    missing = sorted(test_labels - train_labels)
    class_names = list(map(str, data["fine_enc"].classes_))
    support = {
        "run": run_name,
        "split_mode": split_mode,
        "spatial_fold": None if spatial_fold is None else int(spatial_fold),
        "train_classes_present": int(len(np.unique(data["y_fine_train"]))),
        "validation_classes_present": int(len(np.unique(data["y_fine_val"]))),
        "test_classes_present": int(len(np.unique(data["y_fine_test"]))),
        "test_classes_absent_from_train": missing,
        "test_class_names_absent_from_train": [class_names[i] for i in missing],
        "n_test": int(len(data["y_fine_test"])),
        "n_test_supported": int(supported.sum()),
        "test_supported_fraction": float(supported.mean()),
        "train_test_block_overlap": split.get("block_overlap_train_test", []),
        "validation_test_block_overlap": split.get("block_overlap_val_test", []),
    }
    with open(run_dir / "phase3_class_support_audit.json", "w", encoding="utf-8") as f:
        json.dump(support, f, indent=2)
    return comparison, modality_df, support


def _aggregate(df: pd.DataFrame, group_cols, metrics):
    rows = []
    if df.empty:
        return pd.DataFrame()
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_runs"] = int(len(g))
        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{metric}_sample_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    all_comp = []
    all_mod = []
    supports = []

    if args.protocol in {"primary", "both"}:
        c, m, s = _one_run(args, "primary_seed42", "mixed", None)
        all_comp.append(c); all_mod.append(m); supports.append(s)

    if args.protocol in {"spatial", "both"}:
        for fold in range(args.n_spatial_folds):
            c, m, s = _one_run(args, f"spatial_fold_{fold}", "spatial_block", fold)
            all_comp.append(c); all_mod.append(m); supports.append(s)

    comp = pd.concat(all_comp, ignore_index=True, sort=False) if all_comp else pd.DataFrame()
    mod = pd.concat(all_mod, ignore_index=True, sort=False) if all_mod else pd.DataFrame()
    comp.to_csv(out / "phase3_all_main_comparisons.csv", index=False)
    mod.to_csv(out / "phase3_all_modality_comparisons.csv", index=False)
    pd.DataFrame(supports).to_json(out / "phase3_class_support_summary.json", orient="records", indent=2)

    spatial_comp = comp[comp["split_mode"] == "spatial_block"].copy() if not comp.empty else comp
    spatial_mod = mod[mod["split_mode"] == "spatial_block"].copy() if not mod.empty else mod
    metrics = ["accuracy", "macro_f1", "balanced_accuracy", "mcc", "ece", "brier", "nll"]
    # Keep the historical all-test summary columns unchanged. New support-only
    # columns are explicitly labelled and MUST NOT replace all-test scores.
    support_metrics = [f"supported_only_{name}" for name in metrics]
    comp_summary = _aggregate(spatial_comp, ["model", "state"], metrics + support_metrics)
    if not spatial_comp.empty:
        counts = spatial_comp.groupby(["model", "state"], dropna=False).agg(
            n_test_total=("n_test", "sum"),
            n_test_supported_total=("n_test_supported", "sum"),
            n_unsupported_test_classes_max=("n_unsupported_test_classes", "max"),
        ).reset_index()
        for col in ("n_test_total", "n_test_supported_total", "n_unsupported_test_classes_max"):
            counts[col] = counts[col].astype("int64")
        counts["test_supported_fraction_pooled"] = (
            counts["n_test_supported_total"] / counts["n_test_total"]
        )
        comp_summary = comp_summary.merge(counts, on=["model", "state"], how="left", validate="one_to_one")
    mod_summary = _aggregate(spatial_mod, ["learner", "modality", "state"], metrics)
    comp_summary.to_csv(out / "phase3_spatial_main_summary.csv", index=False)
    mod_summary.to_csv(out / "phase3_spatial_modality_summary.csv", index=False)

    audit = {
        "phase": 3,
        "purpose": "corrected calibrated-baseline and fair modality comparisons",
        "protocol": args.protocol,
        "seed": int(args.seed),
        "n_spatial_folds": int(args.n_spatial_folds),
        "spatial_grid": int(args.spatial_grid),
        "stack_folds": int(args.stack_folds),
        "fast_mode": bool(args.fast),
        "skip_proposed": bool(args.skip_proposed),
        "calibration": {
            "selection_data": "validation only",
            "criterion": "NLL",
            "method": "continuous bounded optimization in log-temperature",
            "temperature_bounds": [0.05, 10.0],
            "ece_final_bin_includes_confidence_1": True,
        },
        "calibrated_baselines": ["Random Forest", "XGBoost", "ExtraTrees"],
        "modality_learner": "ExtraTrees",
        "modality_groups": ["LiDAR-derived (6)", "spectral NDVI/NDWI (2)", "fused (8)"],
        "phase4_statistics_intentionally_deferred": True,
    }
    with open(out / "PHASE3_RUN_AUDIT.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print("\n[Phase 3] COMPLETE")
    if not comp_summary.empty:
        print("\nStrict-spatial main comparison summary:\n", comp_summary.to_string(index=False))
    if not mod_summary.empty:
        print("\nStrict-spatial modality summary:\n", mod_summary.to_string(index=False))
    print(f"\nOutputs: {out}")


if __name__ == "__main__":
    main()
