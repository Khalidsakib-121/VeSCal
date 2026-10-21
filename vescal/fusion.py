"""
EcoTreeFuseNet-Plus
===================
Professional-strength extension of EcoTreeFuseNet.

Adds the seven requested upgrades:
1) stronger tree probability branches (RF, XGB, ExtraTrees, HGB, optional LGBM/CatBoost),
2) serious validation-selected meta-classifier,
3) validation-weighted soft voting,
4) post-hoc calibration,
5) class-bias tuning for Macro-F1,
6) hierarchy-aware coarse-to-fine correction,
7) artifacts needed for multi-seed ensemble scripts.

No test-set tuning is performed: all model/meta/calibration/bias/hierarchy choices
are selected using validation data only, then applied once to D_test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler as SkStandardScaler

from baselines import generate_oof_tree_probabilities_plus
from calibration import (
    normalize_probs,
    fit_temperature_scaling,
    apply_temperature_scaling,
    tune_class_bias,
    apply_class_bias,
    optimize_weighted_soft_voting,
    tune_hierarchy_alpha,
    save_calibration_report,
    make_json_safe,
)
from oof import generate_ecofusenet_oof_features


def _safe_predict_proba(model, X: np.ndarray, K: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.shape[1] == K:
            return normalize_probs(p)
        out = np.zeros((X.shape[0], K), dtype=np.float32)
        classes = getattr(model, "classes_", np.arange(p.shape[1]))
        for j, c in enumerate(classes):
            if 0 <= int(c) < K:
                out[:, int(c)] = p[:, j]
        return normalize_probs(out)
    pred = model.predict(X).astype(int)
    out = np.zeros((X.shape[0], K), dtype=np.float32)
    out[np.arange(len(pred)), pred] = 1.0
    return out


def build_plus_matrix(X: np.ndarray,
                      branch_probs: Dict[str, np.ndarray],
                      eco_probs: np.ndarray,
                      eco_embeddings: np.ndarray,
                      eco_gates: np.ndarray,
                      eco_distances: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Create variable-dimensional EcoTreeFuseNet-Plus meta-feature matrix."""
    blocks = [("original_features", X.astype(np.float32))]
    for name in sorted(branch_probs.keys()):
        blocks.append((f"{name}_probabilities", branch_probs[name].astype(np.float32)))
    blocks.extend([
        ("ecofusenet_probabilities", eco_probs.astype(np.float32)),
        ("ecofusenet_embeddings", eco_embeddings.astype(np.float32)),
        ("ecofusenet_gate_values", eco_gates.astype(np.float32)),
        ("ecofusenet_prototype_distances", eco_distances.astype(np.float32)),
    ])
    Z = np.concatenate([b for _, b in blocks], axis=1).astype(np.float32)
    meta_blocks = {name: int(arr.shape[1]) for name, arr in blocks}
    return Z, {"meta_feature_dim": int(Z.shape[1]), "meta_blocks": meta_blocks, "branch_order": sorted(branch_probs.keys())}


def train_plus_meta_candidates(Z_train: np.ndarray,
                               y_train: np.ndarray,
                               Z_val: np.ndarray,
                               y_val: np.ndarray,
                               K: int,
                               random_state: int = 42) -> Tuple[object, Dict]:
    """Validation-select a robust meta-classifier."""
    candidates = []
    for C in [0.001, 0.003, 0.01, 0.03, 0.1]:
        candidates.append((
            f"logistic_l2_C{C}",
            LogisticRegression(
                C=C,
                penalty="l2",
                class_weight="balanced",
                max_iter=6000,
                solver="lbfgs",
                random_state=random_state,
                n_jobs=-1,
            ),
        ))
    for C in [0.003, 0.01, 0.03]:
        candidates.append((
            f"logistic_elasticnet_C{C}",
            LogisticRegression(
                C=C,
                penalty="elasticnet",
                l1_ratio=0.15,
                class_weight="balanced",
                max_iter=8000,
                solver="saga",
                random_state=random_state,
                n_jobs=-1,
            ),
        ))
    candidates.append((
        "hist_gradient_boosting_meta_plus",
        HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=350,
            max_leaf_nodes=15,
            l2_regularization=0.08,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=random_state,
        ),
    ))
    candidates.append((
        "extra_trees_meta_regularized",
        ExtraTreesClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
    ))
    candidates.append((
        "rf_meta_regularized_plus",
        RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
    ))

    best_model = None
    best_info = None
    rows = []
    for name, clf in candidates:
        try:
            clf.fit(Z_train, y_train)
            val_prob = _safe_predict_proba(clf, Z_val, K)
            val_pred = val_prob.argmax(axis=1)
            val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
            rows.append({"candidate": name, "val_macro_f1": float(val_f1)})
            print(f"  [Meta-Plus] {name:<34s} val_macro_F1={val_f1:.4f}")
            if best_info is None or val_f1 > best_info["val_macro_f1"]:
                best_info = {"candidate": name, "val_macro_f1": float(val_f1), "val_prob": val_prob}
                best_model = clf
        except Exception as exc:
            rows.append({"candidate": name, "error": repr(exc)})
            print(f"  [Meta-Plus] {name:<34s} skipped: {exc}")
    if best_model is None:
        raise RuntimeError("All EcoTreeFuseNet-Plus meta-candidates failed.")
    best_info["all_candidates"] = rows
    return best_model, best_info


def _make_result(name: str, y_true: np.ndarray, probs: np.ndarray) -> dict:
    probs = normalize_probs(probs)
    preds = probs.argmax(axis=1).astype(np.int64)
    return {
        "exp_name": name,
        "test_macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "preds": preds,
        "probs": probs,
        "labels": y_true,
    }


def train_ecotreefusenet_plus(data: dict,
                              output_dir: str,
                              device,
                              final_ecofuse_model,
                              phi: np.ndarray,
                              random_state: int = 42,
                              n_splits: int = 5,
                              tree_estimators: int = 800,
                              xgb_max_depth: int = 5,
                              fast: bool = False) -> Dict[str, dict]:
    """Train EcoTreeFuseNet-Plus and return multiple proposed variants."""
    print("\n[Exp] EcoTreeFuseNet-Plus — stronger branches + calibration + hierarchy")
    out_dir = Path(output_dir)
    pred_dir = out_dir / "predictions"
    stats_dir = out_dir / "stats"
    plus_dir = out_dir / "ecotreefusenet_plus"
    for d in [pred_dir, stats_dir, plus_dir, out_dir / "stacking_plus"]:
        d.mkdir(parents=True, exist_ok=True)

    K = data["K"]
    y_val = data["y_fine_val"]
    y_test = data["y_fine_test"]

    # 1) Stronger tree OOF branches.
    tree_plus = generate_oof_tree_probabilities_plus(
        data=data,
        output_dir=output_dir,
        random_state=random_state,
        n_splits=n_splits,
        n_estimators=tree_estimators,
        xgb_max_depth=xgb_max_depth,
        fast=fast,
    )
    branches = tree_plus["branches"]

    # 2) Leakage-safe EcoFuseNet OOF internals.
    fold_hparams = dict(
        warmup_epochs=3 if fast else 15,
        max_epochs=10 if fast else 90,
        patience=4 if fast else 15,
        proto_weight=0.005 if fast else 0.01,
        coarse_weight=0.002 if fast else 0.005,
        sep_weight=0.0001 if fast else 0.0005,
    )
    eco = generate_ecofusenet_oof_features(
        data=data,
        output_dir=output_dir,
        final_model=final_ecofuse_model,
        device=device,
        phi=phi,
        random_state=random_state,
        n_splits=n_splits,
        batch_size=64,
        fold_hparams=fold_hparams,
    )

    # 3) Collect branch probabilities for soft voting and hybrid features.
    branch_train = {name: obj["oof_train"] for name, obj in branches.items()}
    branch_val = {name: obj["val"] for name, obj in branches.items()}
    branch_test = {name: obj["test"] for name, obj in branches.items()}
    branch_train["ecofusenet"] = eco["eco_oof_train_probs"]
    branch_val["ecofusenet"] = eco["eco_val_probs"]
    branch_test["ecofusenet"] = eco["eco_test_probs"]

    # 4) Variable-dimensional Plus hybrid matrix.
    Z_train, meta_report = build_plus_matrix(
        data["X_train"],
        {k: v for k, v in branch_train.items() if k != "ecofusenet"},
        eco["eco_oof_train_probs"], eco["eco_oof_train_embeddings"], eco["eco_oof_train_gates"], eco["eco_oof_train_distances"],
    )
    Z_val, _ = build_plus_matrix(
        data["X_val"],
        {k: v for k, v in branch_val.items() if k != "ecofusenet"},
        eco["eco_val_probs"], eco["eco_val_embeddings"], eco["eco_val_gates"], eco["eco_val_distances"],
    )
    Z_test, _ = build_plus_matrix(
        data["X_test"],
        {k: v for k, v in branch_test.items() if k != "ecofusenet"},
        eco["eco_test_probs"], eco["eco_test_embeddings"], eco["eco_test_gates"], eco["eco_test_distances"],
    )
    np.save(pred_dir / "hybrid_plus_features_train.npy", Z_train)
    np.save(pred_dir / "hybrid_plus_features_val.npy", Z_val)
    np.save(pred_dir / "hybrid_plus_features_test.npy", Z_test)

    scaler = SkStandardScaler()
    Z_train_s = scaler.fit_transform(Z_train)
    Z_val_s = scaler.transform(Z_val)
    Z_test_s = scaler.transform(Z_test)
    np.savez(pred_dir / "hybrid_plus_meta_scaler.npz", mean=scaler.mean_, scale=scaler.scale_)

    # 5) Validation-selected meta-classifier.
    meta_model, meta_info = train_plus_meta_candidates(
        Z_train_s, data["y_fine_train"], Z_val_s, y_val, K, random_state=random_state
    )
    meta_val_probs = _safe_predict_proba(meta_model, Z_val_s, K)
    meta_test_probs = _safe_predict_proba(meta_model, Z_test_s, K)
    meta_test_pred = meta_test_probs.argmax(axis=1)
    print(f"[EcoTreeFuseNet-Plus] Meta-classifier Test Macro F1 = {f1_score(y_test, meta_test_pred, average='macro', zero_division=0):.4f}")

    # 6) Validation-weighted soft voting over all branch probabilities.
    soft_vote = optimize_weighted_soft_voting(
        branch_val_probs=branch_val,
        y_val=y_val,
        branch_test_probs=branch_test,
        random_state=random_state,
        n_random=500 if fast else 4000,
    )
    sv_val_probs = soft_vote["val_probs"]
    sv_test_probs = soft_vote["test_probs"]
    print(f"[EcoTreeFuseNet-Plus] Soft-vote val Macro F1 = {soft_vote['best']['val_macro_f1']:.4f}")

    # 7) Select base Plus probability source using validation Macro F1.
    meta_val_f1 = f1_score(y_val, meta_val_probs.argmax(axis=1), average="macro", zero_division=0)
    sv_val_f1 = f1_score(y_val, sv_val_probs.argmax(axis=1), average="macro", zero_division=0)
    if sv_val_f1 > meta_val_f1:
        base_source = "validation_weighted_soft_vote"
        plus_val_probs = sv_val_probs
        plus_test_probs = sv_test_probs
    else:
        base_source = "meta_classifier"
        plus_val_probs = meta_val_probs
        plus_test_probs = meta_test_probs
    print(f"[EcoTreeFuseNet-Plus] Selected base source: {base_source}")

    # 8) Temperature calibration.
    temp_info = fit_temperature_scaling(plus_val_probs, y_val)
    cal_val_probs = apply_temperature_scaling(plus_val_probs, temp_info["temperature"])
    cal_test_probs = apply_temperature_scaling(plus_test_probs, temp_info["temperature"])

    # 9) Class-bias tuning for Macro F1.
    bias_info = tune_class_bias(cal_val_probs, y_val, passes=3 if not fast else 1)
    bias_val_probs, _ = apply_class_bias(cal_val_probs, bias_info["bias"])
    bias_test_probs, _ = apply_class_bias(cal_test_probs, bias_info["bias"])

    # 10) Hierarchy-aware correction on top of bias-calibrated probabilities.
    hier = tune_hierarchy_alpha(bias_val_probs, y_val, bias_test_probs, data["phi"], data["C"])
    hier_test_probs = hier["test_probs"]

    # Results to include in final report.
    results = {
        "ecotreefusenet_plus": _make_result("ecotreefusenet_plus", y_test, plus_test_probs),
        "calibrated_ecotreefusenet_plus": _make_result("calibrated_ecotreefusenet_plus", y_test, cal_test_probs),
        "bias_tuned_ecotreefusenet_plus": _make_result("bias_tuned_ecotreefusenet_plus", y_test, bias_test_probs),
        "hierarchical_ecotreefusenet_plus": _make_result("hierarchical_ecotreefusenet_plus", y_test, hier_test_probs),
    }

    # Save prediction arrays.
    for name, res in results.items():
        np.save(pred_dir / f"predictions_{name}.npy", res["preds"])
        np.save(pred_dir / f"probabilities_{name}.npy", res["probs"])

    # Phase 3 audit artifacts: save validation probabilities used for source
    # selection and temperature fitting so calibration can be reproduced without
    # retraining and without touching test labels.
    np.save(pred_dir / "probabilities_ecotreefusenet_plus_val.npy", plus_val_probs)
    np.save(pred_dir / "probabilities_calibrated_ecotreefusenet_plus_val.npy", cal_val_probs)
    np.save(pred_dir / "y_validation.npy", y_val)
    np.save(pred_dir / "embeddings_ecotreefusenet_plus.npy", eco["eco_test_embeddings"])
    np.save(pred_dir / "gate_values_ecotreefusenet_plus.npy", eco["eco_test_gates"])
    np.save(pred_dir / "proto_distances_ecotreefusenet_plus.npy", eco["eco_test_distances"])

    # Save branch/meta reports. Keep the selected candidate summary, but do not
    # dump large validation probability matrices inside JSON reports.
    meta_info_report = {k: v for k, v in meta_info.items() if k != "val_prob"}
    report = {
        "selected_base_source": base_source,
        "meta_feature_report": meta_report,
        "tree_plus_diagnostics": tree_plus["diagnostics"],
        "ecofusenet_oof_diagnostics": eco["diagnostics"],
        "meta_info": meta_info_report,
        "soft_voting": soft_vote["best"],
        "temperature_scaling": {k: v for k, v in temp_info.items() if k != "all_temperatures"},
        "class_bias": {"val_macro_f1": bias_info["val_macro_f1"], "bias": bias_info["bias"]},
        "hierarchy": hier["best"],
        "result_macro_f1": {name: float(res["test_macro_f1"]) for name, res in results.items()},
    }
    save_calibration_report(report, output_dir, filename="ecotreefusenet_plus_report.json")
    with open(plus_dir / "ecotreefusenet_plus_report.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe(report), f, indent=2)

    # Meta feature importance.
    try:
        if hasattr(meta_model, "coef_"):
            imp = np.mean(np.abs(meta_model.coef_), axis=0)
        elif hasattr(meta_model, "feature_importances_"):
            imp = meta_model.feature_importances_
        else:
            imp = np.zeros(Z_train.shape[1], dtype=np.float32)
        pd.DataFrame({"feature_index": np.arange(len(imp)), "importance": imp}).to_csv(
            stats_dir / "table_ecotreefusenet_plus_meta_feature_importance.csv", index=False
        )
    except Exception as exc:
        print(f"  [Meta-Plus] Feature-importance export skipped: {exc}")

    print("[EcoTreeFuseNet-Plus] Final variants:")
    for name, res in results.items():
        print(f"  {name:<38s} Test Macro F1 = {res['test_macro_f1']:.4f}")
    return results
