"""Phase 3 corrected comparisons and calibration utilities.

Implements the supervisor-requested Phase-3 evidence:
1) equally calibrated Random Forest, XGBoost, and ExtraTrees;
2) corrected temperature calibration (continuous log-temperature search);
3) corrected ECE with confidence==1.0 included in the final bin;
4) fair same-learner modality comparison using ExtraTrees;
5) explicit class-support diagnostics for strict spatial folds.

All calibration parameters are selected from validation probabilities only.
No test label is used for calibration or model selection.
Python 3.10+.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from baselines import make_xgb_model, _align_proba, _class_sample_weights
from calibration import fit_temperature_scaling, apply_temperature_scaling
from evaluation import compute_all_metrics

FEATURE_NAMES = [
    "elevation", "slope", "curvature", "tpi",
    "solar_insolation", "chm", "ndvi", "ndwi",
]
MODALITY_INDICES = {
    "lidar_derived": np.array([0, 1, 2, 3, 4, 5], dtype=int),
    "spectral": np.array([6, 7], dtype=int),
    "fused": np.arange(8, dtype=int),
}


def _fit_predict_aligned(model, X_train, y_train, X_val, X_test, K, sample_weight=None):
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        model.fit(X_train, y_train)

    classes = getattr(model, "classes_", np.arange(K))
    p_val = _align_proba(model.predict_proba(X_val), classes, K)
    p_test = _align_proba(model.predict_proba(X_test), classes, K)
    return p_val.astype(np.float32), p_test.astype(np.float32)


def _make_model(name: str, K: int, seed: int, fast: bool = False):
    name = str(name).lower()
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=120 if fast else 500,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=120 if fast else 800,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=1,
            bootstrap=False,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "xgboost":
        model = make_xgb_model(
            K=K,
            random_state=seed,
            n_estimators=120 if fast else 500,
            max_depth=5,
        )
        if model is None:
            raise RuntimeError("XGBoost is required for Phase 3. Install requirements.txt")
        return model
    raise ValueError(f"Unsupported Phase-3 baseline: {name}")


def _metric_rows(model_name: str, y_test: np.ndarray, p_raw: np.ndarray,
                 p_cal: np.ndarray, K: int, temperature_info: Dict,
                 train_classes: Iterable[int]) -> Tuple[Dict, Dict]:
    train_classes = np.array(sorted(set(map(int, train_classes))), dtype=int)
    supported_mask = np.isin(y_test, train_classes)
    unsupported_classes = sorted(set(map(int, np.unique(y_test))) - set(map(int, train_classes)))

    rows = []
    for state, probs in [("raw", p_raw), ("calibrated", p_cal)]:
        pred = probs.argmax(axis=1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
            m = compute_all_metrics(y_test, pred, probs, K)
        row = {
            "model": model_name,
            "state": state,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "balanced_accuracy": m["balanced_acc"],
            "mcc": m["mcc"],
            "ece": m["ece"],
            "brier": m["brier"],
            "nll": m["nll"],
            "n_test": int(len(y_test)),
            "n_test_supported": int(supported_mask.sum()),
            "test_supported_fraction": float(supported_mask.mean()) if len(y_test) else np.nan,
            "n_unsupported_test_classes": int(len(unsupported_classes)),
            "unsupported_test_class_indices": ";".join(map(str, unsupported_classes)),
            "temperature": float(temperature_info["temperature"]) if state == "calibrated" else 1.0,
        }
        if supported_mask.any():
            # A classifier may legitimately predict a class not present in
            # this spatial fold's supported subset. sklearn issues a warning
            # for balanced accuracy, but still returns the intended score.
            # This is the identical definition used for every model.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
                sm = compute_all_metrics(
                    y_test[supported_mask], pred[supported_mask], probs[supported_mask], K
                )
            row.update({
                "supported_only_accuracy": sm["accuracy"],
                "supported_only_macro_f1": sm["macro_f1"],
                "supported_only_balanced_accuracy": sm["balanced_acc"],
                "supported_only_mcc": sm["mcc"],
                "supported_only_ece": sm["ece"],
                "supported_only_brier": sm["brier"],
                "supported_only_nll": sm["nll"],
            })
        rows.append(row)

    audit = {
        "model": model_name,
        "temperature_scaling": {k: v for k, v in temperature_info.items() if k != "all_temperatures"},
        "unsupported_test_class_indices": unsupported_classes,
        "n_test_supported": int(supported_mask.sum()),
        "n_test_total": int(len(y_test)),
    }
    return rows, audit


def run_calibrated_baselines(data: dict, output_dir: str, seed: int = 42,
                             fast: bool = False,
                             models=("rf", "xgboost", "extra_trees")) -> pd.DataFrame:
    """Fit raw and equally calibrated strong baselines on one fixed split."""
    out = Path(output_dir)
    pred_dir = out / "predictions"
    audit_dir = out / "phase3_audit"
    pred_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Use the exact outer-training-scaled arrays used by the historical
    # standalone baselines. Scaling is fitted on training data only and this
    # preserves direct comparability with the frozen Phase-1/Phase-2 evidence.
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    y_train = data["y_fine_train"]
    y_val = data["y_fine_val"]
    y_test = data["y_fine_test"]
    K = int(data["K"])

    rows = []
    audits = {}
    for name in models:
        # Use the same predefined run seed for each baseline family. Different
        # model classes do not need artificial seed offsets, and keeping seed=42
        # reproduces the historical baseline configuration more directly.
        model = _make_model(name, K=K, seed=seed, fast=fast)
        # Match the frozen baseline fitting protocol exactly: XGBoost and the
        # generic ExtraTrees baseline receive the training-only inverse-frequency
        # sample weights; Random Forest uses its built-in class_weight only.
        sw = _class_sample_weights(y_train) if name in {"xgboost", "extra_trees"} else None
        p_val, p_test = _fit_predict_aligned(
            model, X_train, y_train, X_val, X_test, K, sample_weight=sw
        )

        temp = fit_temperature_scaling(
            p_val, y_val, criterion="nll", temperature_bounds=(0.05, 10.0)
        )
        p_cal = apply_temperature_scaling(p_test, temp["temperature"])
        p_val_cal = apply_temperature_scaling(p_val, temp["temperature"])

        np.save(pred_dir / f"phase3_{name}_val_probs_raw.npy", p_val)
        np.save(pred_dir / f"phase3_{name}_test_probs_raw.npy", p_test)
        np.save(pred_dir / f"phase3_{name}_val_probs_calibrated.npy", p_val_cal)
        np.save(pred_dir / f"phase3_{name}_test_probs_calibrated.npy", p_cal)
        np.save(pred_dir / f"phase3_{name}_test_pred_raw.npy", p_test.argmax(axis=1))
        np.save(pred_dir / f"phase3_{name}_test_pred_calibrated.npy", p_cal.argmax(axis=1))

        model_rows, audit = _metric_rows(
            name, y_test, p_test, p_cal, K, temp, np.unique(y_train)
        )
        rows.extend(model_rows)
        audits[name] = audit

    with open(audit_dir / "calibrated_baseline_audit.json", "w", encoding="utf-8") as f:
        json.dump(audits, f, indent=2)
    df = pd.DataFrame(rows)
    df.to_csv(out / "phase3_calibrated_baselines.csv", index=False)
    return df


def run_modality_comparison(data: dict, output_dir: str, seed: int = 42,
                            fast: bool = False, calibrate: bool = True) -> pd.DataFrame:
    """Fair modality experiment: same ExtraTrees learner, only predictors vary."""
    out = Path(output_dir)
    pred_dir = out / "predictions"
    audit_dir = out / "phase3_audit"
    pred_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Same training-only standardized matrix for every modality; only columns
    # differ. This keeps the modality experiment controlled.
    X_train_all = data["X_train"]
    X_val_all = data["X_val"]
    X_test_all = data["X_test"]
    y_train = data["y_fine_train"]
    y_val = data["y_fine_val"]
    y_test = data["y_fine_test"]
    K = int(data["K"])

    rows = []
    audit = {}
    for offset, (modality, idx) in enumerate(MODALITY_INDICES.items()):
        model = _make_model("extra_trees", K=K, seed=seed, fast=fast)
        # Same seed/hyperparameters across modalities by design; only X columns differ.
        p_val, p_test = _fit_predict_aligned(
            model,
            X_train_all[:, idx], y_train,
            X_val_all[:, idx], X_test_all[:, idx],
            K, sample_weight=_class_sample_weights(y_train),
        )
        temp = fit_temperature_scaling(
            p_val, y_val, criterion="nll", temperature_bounds=(0.05, 10.0)
        ) if calibrate else {
            "temperature": 1.0, "val_nll": np.nan, "val_ece": np.nan,
            "val_macro_f1": np.nan, "score": np.nan,
            "search_method": "disabled", "at_lower_bound": False,
            "at_upper_bound": False,
        }
        p_cal = apply_temperature_scaling(p_test, temp["temperature"])

        for state, probs in [("raw", p_test), ("calibrated", p_cal)]:
            pred = probs.argmax(axis=1)
            m = compute_all_metrics(y_test, pred, probs, K)
            rows.append({
                "learner": "extra_trees",
                "modality": modality,
                "state": state,
                "n_features": int(len(idx)),
                "features": ";".join(FEATURE_NAMES[i] for i in idx),
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "balanced_accuracy": m["balanced_acc"],
                "mcc": m["mcc"],
                "ece": m["ece"],
                "brier": m["brier"],
                "nll": m["nll"],
                "temperature": float(temp["temperature"]) if state == "calibrated" else 1.0,
            })
        np.save(pred_dir / f"phase3_modality_{modality}_extra_trees_test_raw.npy", p_test)
        np.save(pred_dir / f"phase3_modality_{modality}_extra_trees_test_calibrated.npy", p_cal)
        audit[modality] = {
            "feature_indices": idx.tolist(),
            "features": [FEATURE_NAMES[i] for i in idx],
            "temperature_scaling": {k: v for k, v in temp.items() if k != "all_temperatures"},
        }

    with open(audit_dir / "modality_comparison_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    df = pd.DataFrame(rows)
    df.to_csv(out / "phase3_modality_extra_trees.csv", index=False)
    return df


def collect_proposed_results(run_dir: str, data: dict) -> pd.DataFrame:
    """Read *and independently validate* proposed-model results.

    Phase-3 freeze repair: calculate exactly the same training-class-supported
    test diagnostics already calculated for RF/XGBoost/ExtraTrees. The held-out
    labels are read only for post-training evaluation. No model is refitted and
    neither temperature nor model selection is changed.

    ``data`` must refer to the *same saved split* used by the completed run.
    An explicit test-label equality check prevents silently merging metrics
    from differently ordered predictions and labels.
    """
    run_dir = Path(run_dir)
    stats_path = run_dir / "stats" / "statistical_report.json"
    report_path = run_dir / "ecotreefusenet_plus" / "ecotreefusenet_plus_report.json"
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)
    stats = json.load(open(stats_path, encoding="utf-8"))
    temp_info = {}
    if report_path.exists():
        report = json.load(open(report_path, encoding="utf-8"))
        temp_info = report.get("temperature_scaling", {})

    pred_dir = run_dir / "predictions"
    y_train = np.asarray(data["y_fine_train"], dtype=int)
    y_test = np.asarray(data["y_fine_test"], dtype=int)
    K = int(data["K"])
    saved_y = np.load(pred_dir / "y_true.npy", allow_pickle=False)
    if not np.array_equal(saved_y, y_test):
        raise ValueError(f"Saved test labels disagree with the Phase-3 split: {run_dir}")

    p_raw = np.load(pred_dir / "probabilities_ecotreefusenet_plus.npy", allow_pickle=False)
    p_cal = np.load(pred_dir / "probabilities_calibrated_ecotreefusenet_plus.npy", allow_pickle=False)
    for name, probs in (("raw", p_raw), ("calibrated", p_cal)):
        if probs.shape != (len(y_test), K) or not np.all(np.isfinite(probs)):
            raise ValueError(f"Invalid {name} proposed probabilities in {run_dir}")
        if np.any(probs < -1e-7) or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"Non-normalized {name} proposed probabilities in {run_dir}")

    rows, _ = _metric_rows(
        "proposed_fusion", y_test, p_raw, p_cal, K,
        {"temperature": float(temp_info.get("temperature", 1.0))},
        np.unique(y_train),
    )
    # Assert that the original, immutable all-test metrics are reproducible
    # from saved probabilities before filling the missing supported-only fields.
    keys = {"balanced_accuracy": "balanced_acc"}
    for row, name in zip(rows, ("ecotreefusenet_plus", "calibrated_ecotreefusenet_plus")):
        old = stats.get("exp_metrics", {}).get(name)
        if old is None:
            raise KeyError(f"Missing original results for {name} in {stats_path}")
        for metric in ("accuracy", "macro_f1", "balanced_accuracy", "mcc", "ece", "brier", "nll"):
            previous = float(old[keys.get(metric, metric)])
            if not np.isclose(row[metric], previous, atol=1e-6, rtol=1e-6):
                raise ValueError(
                    f"Historical metric mismatch: {run_dir} {name} {metric}: "
                    f"saved={previous}, recomputed={row[metric]}"
                )
            row[metric] = previous
    return pd.DataFrame(rows)
