"""Freeze a COMPLETE, existing Phase-3 run without retraining (Python 3.10).

Copy the original results to a NEW destination, validate split IDs and the saved
raw/calibrated test probabilities, and fill supported-only metrics for the
proposed fusion using EXACTLY the baseline metric implementation. Update the
per-run and global CSVs, class-support audit, and spatial aggregate summary.
The original source directory remains untouched. This program does not perform
Phase-4 statistical significance testing or tune any model.

Usage:
    python freeze_results.py \\
      --source outputs_phase3 --destination outputs_phase3_frozen \\
      --data-path data/vegetation_reference.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from evaluation import compute_all_metrics
from comparisons import collect_proposed_results
from run_experiments import _aggregate

RUN_NAMES = ["primary_seed42"] + [f"spatial_fold_{n}" for n in range(5)]
METRICS = ["accuracy", "macro_f1", "balanced_accuracy", "mcc", "ece", "brier", "nll"]
SUPPORT_COLS = ["supported_only_" + metric for metric in METRICS]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_saved_splits(run_dir: Path, y: np.ndarray, names: list[str], ids: np.ndarray,
                        blocks: np.ndarray):
    paths = [run_dir / f"{part}_indices.npy" for part in ("train", "validation", "test")]
    train, val, test = [np.load(p, allow_pickle=False).astype(int) for p in paths]
    sets = [set(v.tolist()) for v in (train, val, test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), run_dir
    assert set.union(*sets) == set(range(len(y))), f"Partition does not cover dataset: {run_dir}"
    saved_y = np.load(run_dir / "predictions" / "y_true.npy", allow_pickle=False)
    assert np.array_equal(y[test], saved_y), f"Saved labels/test ordering mismatch: {run_dir}"
    saved_names = json.loads((run_dir / "class_names.json").read_text(encoding="utf-8"))
    assert saved_names == names, f"Label encoder mismatch: {run_dir}"
    # The historical sample-ID .npy files are object/pickle arrays. Do NOT
    # unpickle uploaded artefacts. Derive sample IDs from validated numeric
    # split indices and the trusted, explicit CSV instead; check saved shapes
    # via the .npy header without executing any pickle payload.
    for part, idx in (("train", train), ("validation", val), ("test", test)):
        sid_path = run_dir / f"{part}_sample_ids.npy"
        with sid_path.open("rb") as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, _fortran, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version in ((2, 0), (3, 0)):
                shape, _fortran, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise ValueError(f"Unknown sample-ID .npy format: {version}")
        assert shape == (len(idx),) and dtype.hasobject, (run_dir, part, shape, dtype)
        assert len(set(ids[idx])) == len(idx), f"Duplicate derived sample IDs: {run_dir} {part}"
    split = json.loads((run_dir / "phase2_split_audit.json").read_text(encoding="utf-8"))
    if run_dir.name.startswith("spatial_fold_"):
        assert not split["block_overlap_train_test"], run_dir
        assert not split["block_overlap_val_test"], run_dir
        assert not split["block_overlap_train_val"], run_dir
        # all_spatial_block_ids.npy is an object array in historical outputs;
        # rebuild deterministic block IDs from numeric coordinates instead.
        assert not (set(blocks[train]) & set(blocks[test])), run_dir
        assert not (set(blocks[val]) & set(blocks[test])), run_dir
        assert not (set(blocks[train]) & set(blocks[val])), run_dir
        for part, idx in (("train", train), ("validation", val), ("test", test)):
            assert sorted(set(blocks[idx])) == split[part + "_blocks"], (run_dir, part)
    return train, test


def _baseline_crosscheck(run_dir: Path, comp: pd.DataFrame, y_train: np.ndarray,
                         y_test: np.ndarray, K: int):
    mask = np.isin(y_test, np.unique(y_train))
    for _, row in comp[comp.model.isin(["rf", "xgboost", "extra_trees"])].iterrows():
        filename = run_dir / "predictions" / f"phase3_{row['model']}_test_probs_{row['state']}.npy"
        probs = np.load(filename, allow_pickle=False)
        assert probs.shape == (len(y_test), K)
        assert int(row["n_test_supported"]) == int(mask.sum())
        if not mask.any():
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
            m = compute_all_metrics(y_test[mask], probs.argmax(axis=1)[mask], probs[mask], K)
        for key in METRICS:
            mk = "balanced_acc" if key == "balanced_accuracy" else key
            assert np.isclose(float(row["supported_only_" + key]), m[mk], atol=1e-6, rtol=1e-6), (
                run_dir.name, row["model"], row["state"], key
            )


def _repair_run(run_dir: Path, y: np.ndarray, names: list[str], ids: np.ndarray,
                blocks: np.ndarray) -> dict:
    train_idx, test_idx = verify_saved_splits(run_dir, y, names, ids, blocks)
    y_train, y_test = y[train_idx], y[test_idx]
    data = {"y_fine_train": y_train, "y_fine_test": y_test, "K": len(names)}
    existing = pd.read_csv(run_dir / "phase3_main_comparison.csv")
    if (existing.model == "proposed_fusion").sum() != 2:
        raise ValueError(f"Expected raw and calibrated proposed rows: {run_dir}")
    _baseline_crosscheck(run_dir, existing, y_train, y_test, len(names))
    repaired = collect_proposed_results(str(run_dir), data=data)
    existing["unsupported_test_class_indices"] = existing["unsupported_test_class_indices"].fillna("").astype(str)
    # Only add support diagnostics. Historical *all-test* metrics, temperatures,
    # prediction arrays and model artefacts are not modified.
    new_cols = ["n_test", "n_test_supported", "test_supported_fraction",
                "n_unsupported_test_classes", "unsupported_test_class_indices"] + SUPPORT_COLS
    for state in ("raw", "calibrated"):
        mask = (existing.model == "proposed_fusion") & (existing.state == state)
        row = repaired.loc[repaired.state == state].iloc[0]
        assert mask.sum() == 1
        for metric in METRICS + ["temperature"]:
            assert np.isclose(float(existing.loc[mask, metric].iloc[0]), float(row[metric]),
                              atol=1e-6, rtol=1e-6), (run_dir.name, state, metric)
        for key in new_cols:
            existing.loc[mask, key] = row[key]
    required_numeric = ["n_test", "n_test_supported", "test_supported_fraction",
                        "n_unsupported_test_classes"] + SUPPORT_COLS
    if existing.loc[existing.model == "proposed_fusion", required_numeric].isna().any().any():
        raise ValueError(f"Incomplete support diagnostics: {run_dir}")
    existing.to_csv(run_dir / "phase3_main_comparison.csv", index=False)
    repaired.to_csv(run_dir / "phase3_proposed_supported_only.csv", index=False)

    audit_path = run_dir / "phase3_class_support_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    missing = sorted(set(map(int, np.unique(y_test))) - set(map(int, np.unique(y_train))))
    supp = np.isin(y_test, np.unique(y_train))
    assert audit["test_classes_absent_from_train"] == missing
    audit.update({
        "test_class_names_absent_from_train": [names[i] for i in missing],
        "n_test": int(len(y_test)),
        "n_test_supported": int(supp.sum()),
        "test_supported_fraction": float(supp.mean()),
        "support_definition": "test observations whose true fine class occurs in the training split",
        "supported_only_metric_policy": "identical to Phase-3 baseline compute_all_metrics on the supported test subset",
    })
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return {"run": run_dir.name, "n_test": len(y_test), "n_supported": int(supp.sum()),
            "unsupported_class_names": [names[i] for i in missing],
            "proposed_raw_supported_f1": float(repaired.loc[repaired.state == "raw", "supported_only_macro_f1"].iloc[0]),
            "proposed_calibrated_supported_f1": float(repaired.loc[repaired.state == "calibrated", "supported_only_macro_f1"].iloc[0])}


def freeze(source: Path, destination: Path, data_path: Path) -> dict:
    source, destination, data_path = source.resolve(), destination.resolve(), data_path.resolve()
    if not source.is_dir() or not (source / "phase3_all_main_comparisons.csv").is_file():
        raise FileNotFoundError("Source must be an existing complete outputs_phase3 folder")
    original_run = json.loads((source / "PHASE3_RUN_AUDIT.json").read_text(encoding="utf-8"))
    if (original_run.get("protocol") != "both" or original_run.get("n_spatial_folds") != 5
            or original_run.get("spatial_grid") != 5 or original_run.get("seed") != 42
            or original_run.get("fast_mode") or original_run.get("skip_proposed")):
        raise ValueError("This freeze applies only to the completed primary + five spatial-fold Phase-3 seed-42 run")
    if destination.exists() or source == destination or source in destination.parents:
        raise ValueError("Destination must be NEW, different from and outside the source")
    df = pd.read_csv(data_path)
    enc = LabelEncoder().fit(df["class"].astype(str))
    names = list(map(str, enc.classes_))
    y = enc.transform(df["class"].astype(str))
    ids = df["sample_id"].astype(str).to_numpy()
    # Reconstruct the code's deterministic 5x5 quantile block ID definitions.
    from spatial_data import build_spatial_blocks
    blocks = build_spatial_blocks(df, n_grid=5).to_numpy(dtype=str)
    assert len(df) == 1833 and len(names) == 29 and len(set(ids)) == len(ids)
    for run in RUN_NAMES:
        assert (source / run / "phase3_main_comparison.csv").exists(), f"Missing full run: {run}"
    # Source evidence stays byte-for-byte unchanged; copy then patch only the copy.
    shutil.copytree(source, destination)
    try:
        details = [_repair_run(destination / run, y, names, ids, blocks) for run in RUN_NAMES]
        main = pd.concat([pd.read_csv(destination / run / "phase3_main_comparison.csv")
                          for run in RUN_NAMES], ignore_index=True)
        historical = pd.read_csv(source / "phase3_all_main_comparisons.csv")
        if len(main) != len(historical):
            raise ValueError("Row-count mismatch with original global comparison")
        for key in ("run", "model", "state"):
            assert np.array_equal(main[key].to_numpy(), historical[key].to_numpy()), key
        for metric in METRICS + ["temperature"]:
            assert np.allclose(main[metric].to_numpy(float), historical[metric].to_numpy(float),
                               atol=1e-9, rtol=1e-9), f"Historical all-test metric changed: {metric}"
        main.to_csv(destination / "phase3_all_main_comparisons.csv", index=False)
        (destination / "phase3_class_support_summary.json").write_text(
            json.dumps([json.loads((destination / run / "phase3_class_support_audit.json").read_text(encoding="utf-8"))
                        for run in RUN_NAMES], indent=2) + "\n", encoding="utf-8")
        spatial = main.loc[main.split_mode == "spatial_block"]
        expanded = METRICS + SUPPORT_COLS
        aggregate = _aggregate(spatial, ["model", "state"], expanded)
        count = spatial.groupby(["model", "state"], dropna=False).agg(
            n_test_total=("n_test", "sum"),
            n_test_supported_total=("n_test_supported", "sum"),
            n_unsupported_test_classes_max=("n_unsupported_test_classes", "max"),
        ).reset_index()
        for col in ("n_test_total", "n_test_supported_total", "n_unsupported_test_classes_max"):
            count[col] = count[col].astype("int64")
        count["test_supported_fraction_pooled"] = count.n_test_supported_total / count.n_test_total
        aggregate = aggregate.merge(count, on=["model", "state"], how="left", validate="one_to_one")
        old_summary = pd.read_csv(source / "phase3_spatial_main_summary.csv")
        for _, row in old_summary.iterrows():
            new = aggregate[(aggregate.model == row.model) & (aggregate.state == row.state)]
            assert len(new) == 1
            for metric in METRICS:
                for suffix in ("_mean", "_sample_sd"):
                    assert np.isclose(float(row[metric + suffix]), float(new.iloc[0][metric + suffix]),
                                      atol=1e-9, rtol=1e-9), (row.model, row.state, metric)
        aggregate.to_csv(destination / "phase3_spatial_main_summary.csv", index=False)
        # Immutable controls for all raw outputs not intentionally repaired.
        allowed = {"phase3_all_main_comparisons.csv", "phase3_spatial_main_summary.csv",
                   "phase3_class_support_summary.json"}
        for run in RUN_NAMES:
            allowed.add(run + "/phase3_main_comparison.csv")
            allowed.add(run + "/phase3_class_support_audit.json")
            allowed.add(run + "/phase3_proposed_supported_only.csv")
        source_files = {p.relative_to(source).as_posix(): p for p in source.rglob("*") if p.is_file()}
        dest_files = {p.relative_to(destination).as_posix(): p for p in destination.rglob("*") if p.is_file()}
        unexpected = []
        for name, path in source_files.items():
            if name not in allowed and sha256(path) != sha256(dest_files[name]):
                unexpected.append(name)
        if unexpected:
            raise ValueError(f"Unintended file modifications: {unexpected}")
        assert not any(path.name.endswith(".npy") and sha256(path) != sha256(dest_files[name])
                       for name, path in source_files.items()), "Prediction array changed"
        report = {
            "status": "PHASE_3_FROZEN_AFTER_SUPPORT_ONLY_CLEANUP",
            "source_evidence_preserved": True,
            "data_sha256": sha256(data_path),
            "run_count": len(details), "spatial_fold_count": 5,
            "correction": "Fill proposed_fusion raw/calibrated supported-only fields to match existing tree baseline protocol",
            "historical_all_test_metrics_unchanged": True,
            "all_existing_prediction_arrays_unchanged": True,
            "strict_spatial_split_indices_unchanged": True,
            "sample_id_validation": "derived from numeric split indices + unique CSV sample_id; historical object-dtype NPY headers checked but pickle payload deliberately not loaded",
            "spatial_block_validation": "regenerated deterministic 5x5 quantile block IDs from the CSV and compared with saved split-audit block lists; historical object NPY not unpickled",
            "no_retraining": True, "no_calibration_refitting": True,
            "no_phase4_significance_tests": True,
            "details": details,
            "supported_only_spatial_summary": aggregate.loc[aggregate.model == "proposed_fusion", [
                "state", "supported_only_macro_f1_mean", "supported_only_macro_f1_sample_sd",
                "n_test_total", "n_test_supported_total", "test_supported_fraction_pooled"]].to_dict(orient="records"),
            "changed_or_added_files": sorted(allowed),
        }
        (destination / "PHASE3_FREEZE_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    except Exception:
        # A failed repair must not leave an apparently valid frozen directory.
        shutil.rmtree(destination, ignore_errors=True)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True, help="Original completed outputs_phase3 folder")
    ap.add_argument("--destination", type=Path, required=True, help="NEW location for repaired, frozen outputs")
    ap.add_argument("--data-path", type=Path, default=Path(__file__).parent / "data" / "vegetation_reference.csv")
    args = ap.parse_args()
    report = freeze(args.source, args.destination, args.data_path)
    print("PHASE 3 FREEZE: PASS")
    for d in report["details"]:
        print("{run}: {n_supported}/{n_test} supported; missing classes: {unsupported_class_names}; "
              "proposed supported Macro-F1={proposed_calibrated_supported_f1:.6f}".format(**d))


if __name__ == "__main__":
    main()
