"""
EcoTreeFuseNet hybrid tree-neural stacking module.

This file implements the leakage-safe hybrid layer described in the updated
EcoTreeFuseNet mathematics and algorithm documents.

Hybrid meta-feature dimension:
  8 original standardized predictors
+ 29 RF probability features
+ 29 XGBoost probability features
+ 29 EcoFuseNet-V2 class probability features
+ 128 EcoFuseNet-V2 embedding features
+ 64 EcoFuseNet-V2 gate features
+ 29 EcoFuseNet-V2 prototype-distance features
= 316 dimensions.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler as SkStandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from baselines import generate_oof_tree_probabilities
from data_utils import make_dataloaders
from losses import compute_class_weights
from network import EcoFuseNetV2
from training import train_ecofusenet_v2, evaluate_model


META_BLOCKS = {
    "original_features": 8,
    "rf_probabilities": 29,
    "xgb_probabilities": 29,
    "eco_probabilities": 29,
    "eco_embeddings": 128,
    "eco_gates": 64,
    "eco_proto_distances": 29,
}
EXPECTED_META_DIM = sum(META_BLOCKS.values())


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_fold_data(data: dict, train_indices: np.ndarray, holdout_indices: np.ndarray) -> dict:
    """Create a mini data dictionary for an inner OOF fold on D_train."""
    y_train = data["y_fine_train"]
    yc_train = data["y_coarse_train"]
    K, C = data["K"], data["C"]
    class_counts = np.bincount(y_train[train_indices], minlength=K)
    return {
        "X_train": data["X_train"][train_indices],
        "X_val": data["X_train"][holdout_indices],
        "X_test": data["X_train"][holdout_indices],
        "y_fine_train": y_train[train_indices],
        "y_fine_val": y_train[holdout_indices],
        "y_fine_test": y_train[holdout_indices],
        "y_coarse_train": yc_train[train_indices],
        "y_coarse_val": yc_train[holdout_indices],
        "y_coarse_test": yc_train[holdout_indices],
        "K": K,
        "C": C,
        "class_weights": compute_class_weights(class_counts, mode="sqrt"),
    }


@torch.no_grad()
def extract_ecofusenet_features(model: EcoFuseNetV2,
                                X: np.ndarray,
                                y_fine: np.ndarray,
                                y_coarse: np.ndarray,
                                device: torch.device,
                                batch_size: int = 256) -> Dict[str, np.ndarray]:
    """Extract probabilities, embeddings, gates and prototype distances."""
    fold_data = {
        "X_train": X, "X_val": X, "X_test": X,
        "y_fine_train": y_fine, "y_fine_val": y_fine, "y_fine_test": y_fine,
        "y_coarse_train": y_coarse, "y_coarse_val": y_coarse, "y_coarse_test": y_coarse,
    }
    loaders = make_dataloaders(fold_data, batch_size=batch_size)
    res = evaluate_model(model, loaders["test"], device, K=model.K, classifier_head=None, collect_internal=True)
    return {
        "probs": res["probs"].astype(np.float32),
        "embeddings": res["embeddings"].astype(np.float32),
        "gates": res["gate_values"].astype(np.float32),
        "distances": res["proto_distances"].astype(np.float32),
    }


def generate_ecofusenet_oof_features(data: dict,
                                      output_dir: str,
                                      final_model: EcoFuseNetV2,
                                      device: torch.device,
                                      phi: np.ndarray,
                                      random_state: int = 42,
                                      n_splits: int = 5,
                                      batch_size: int = 64,
                                      fold_hparams: Optional[dict] = None) -> Dict[str, np.ndarray]:
    """
    Generate leakage-safe OOF EcoFuseNet features for D_train.

    For each fold, an EcoFuseNet-V2 model is trained on K-1 folds and evaluated
    on the held-out fold. The final full EcoFuseNet model is used only for
    validation/test features.
    """
    print(f"\n[Stacking] Generating leakage-safe EcoFuseNet-V2 OOF features ({n_splits} folds)...")
    out_dir = Path(output_dir)
    stack_dir = out_dir / "stacking" / "ecofusenet_oof"
    pred_dir = out_dir / "predictions"
    stack_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    K, C = data["K"], data["C"]
    y_train = data["y_fine_train"]
    n_train = len(y_train)
    d_e = final_model.d_e
    d_f = final_model.d_f

    oof_probs = np.zeros((n_train, K), dtype=np.float32)
    oof_emb = np.zeros((n_train, d_e), dtype=np.float32)
    oof_gate = np.zeros((n_train, d_f), dtype=np.float32)
    oof_dist = np.zeros((n_train, K), dtype=np.float32)
    fold_scores = []

    default_fold_hparams = dict(
        warmup_epochs=15,
        max_epochs=90,
        patience=15,
        lr_warmup=1e-3,
        lr_main=8e-4,
        weight_decay=5e-4,
        grad_clip=1.0,
        proto_weight=0.01,
        coarse_weight=0.005,
        sep_weight=0.0005,
        sep_margin=1.0,
        label_smoothing=0.03,
    )
    if fold_hparams:
        default_fold_hparams.update(fold_hparams)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (tr, ho) in enumerate(skf.split(data["X_train"], y_train), start=1):
        print(f"  [Stacking] Fold {fold}/{n_splits}: EcoFuseNet-V2 branch")
        set_seed(random_state + fold)
        fold_data = _make_fold_data(data, tr, ho)
        fold_loaders = make_dataloaders(fold_data, batch_size=batch_size)
        fold_model = EcoFuseNetV2(
            d_f=final_model.d_f,
            d_e=final_model.d_e,
            K=K,
            C=C,
            tau_init=1.0,
            dropout=0.10,
            alpha_chm_init=0.05,
        ).to(device)
        fold_out = stack_dir / f"fold_{fold}"
        res = train_ecofusenet_v2(
            model=fold_model,
            loaders=fold_loaders,
            data=fold_data,
            phi=phi,
            class_weights=fold_data["class_weights"],
            device=device,
            output_dir=str(fold_out),
            exp_name=f"ecofusenet_oof_fold{fold}",
            batch_size=batch_size if "batch_size" in train_ecofusenet_v2.__code__.co_varnames else batch_size,
            **default_fold_hparams,
        ) if False else train_ecofusenet_v2(
            fold_model, fold_loaders, fold_data, phi, fold_data["class_weights"], device,
            str(fold_out), exp_name=f"ecofusenet_oof_fold{fold}", **default_fold_hparams
        )
        # The fold 'test' split is the held-out fold, so returned features are OOF.
        oof_probs[ho] = res["probs"].astype(np.float32)
        oof_emb[ho] = res["embeddings"].astype(np.float32)
        oof_gate[ho] = res["gate_values"].astype(np.float32)
        oof_dist[ho] = res["proto_distances"].astype(np.float32)
        fold_scores.append(float(res["test_macro_f1"]))

    # Final full model produces validation/test branch features.
    val_feat = extract_ecofusenet_features(final_model, data["X_val"], data["y_fine_val"], data["y_coarse_val"], device)
    test_feat = extract_ecofusenet_features(final_model, data["X_test"], data["y_fine_test"], data["y_coarse_test"], device)

    np.save(pred_dir / "ecofuse_oof_train_probs.npy", oof_probs)
    np.save(pred_dir / "ecofuse_oof_train_embeddings.npy", oof_emb)
    np.save(pred_dir / "ecofuse_oof_train_gate_values.npy", oof_gate)
    np.save(pred_dir / "ecofuse_oof_train_proto_distances.npy", oof_dist)
    np.save(pred_dir / "ecofuse_val_probs.npy", val_feat["probs"])
    np.save(pred_dir / "ecofuse_val_embeddings.npy", val_feat["embeddings"])
    np.save(pred_dir / "ecofuse_val_gate_values.npy", val_feat["gates"])
    np.save(pred_dir / "ecofuse_val_proto_distances.npy", val_feat["distances"])
    np.save(pred_dir / "ecofuse_test_probs.npy", test_feat["probs"])
    np.save(pred_dir / "ecofuse_test_embeddings.npy", test_feat["embeddings"])
    np.save(pred_dir / "ecofuse_test_gate_values.npy", test_feat["gates"])
    np.save(pred_dir / "ecofuse_test_proto_distances.npy", test_feat["distances"])

    diag = {
        "n_splits": int(n_splits),
        "fold_macro_f1": fold_scores,
        "mean_oof_fold_macro_f1": float(np.mean(fold_scores)),
        "std_oof_fold_macro_f1": float(np.std(fold_scores)),
        "d_e": int(d_e),
        "d_f": int(d_f),
    }
    with open(stack_dir / "ecofusenet_oof_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)
    print(f"  [Stacking] EcoFuseNet OOF fold macro-F1 mean={diag['mean_oof_fold_macro_f1']:.4f}")

    return {
        "eco_oof_train_probs": oof_probs,
        "eco_oof_train_embeddings": oof_emb,
        "eco_oof_train_gates": oof_gate,
        "eco_oof_train_distances": oof_dist,
        "eco_val_probs": val_feat["probs"],
        "eco_val_embeddings": val_feat["embeddings"],
        "eco_val_gates": val_feat["gates"],
        "eco_val_distances": val_feat["distances"],
        "eco_test_probs": test_feat["probs"],
        "eco_test_embeddings": test_feat["embeddings"],
        "eco_test_gates": test_feat["gates"],
        "eco_test_distances": test_feat["distances"],
        "diagnostics": diag,
    }


def build_hybrid_matrix(X: np.ndarray,
                        rf_probs: np.ndarray,
                        xgb_probs: np.ndarray,
                        eco_probs: np.ndarray,
                        eco_embeddings: np.ndarray,
                        eco_gates: np.ndarray,
                        eco_distances: np.ndarray) -> np.ndarray:
    """Concatenate the 316-dimensional EcoTreeFuseNet meta-feature vector."""
    Z = np.concatenate([
        X.astype(np.float32),
        rf_probs.astype(np.float32),
        xgb_probs.astype(np.float32),
        eco_probs.astype(np.float32),
        eco_embeddings.astype(np.float32),
        eco_gates.astype(np.float32),
        eco_distances.astype(np.float32),
    ], axis=1)
    if Z.shape[1] != EXPECTED_META_DIM:
        raise ValueError(f"EcoTreeFuseNet meta-feature dim mismatch: got {Z.shape[1]}, expected {EXPECTED_META_DIM}")
    return Z.astype(np.float32)


def _safe_predict_proba(model, X: np.ndarray, K: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.shape[1] == K:
            return p.astype(np.float32)
        # Align when a classifier dropped rare classes.
        out = np.zeros((X.shape[0], K), dtype=np.float32)
        classes = getattr(model, "classes_", np.arange(p.shape[1]))
        for j, c in enumerate(classes):
            out[:, int(c)] = p[:, j]
        return out
    pred = model.predict(X)
    out = np.zeros((X.shape[0], K), dtype=np.float32)
    out[np.arange(X.shape[0]), pred.astype(int)] = 1.0
    return out


def train_meta_classifier(Z_train: np.ndarray,
                          y_train: np.ndarray,
                          Z_val: np.ndarray,
                          y_val: np.ndarray,
                          K: int,
                          random_state: int = 42) -> Tuple[object, dict]:
    """Validation-select the meta-classifier without touching D_test."""
    candidates = []
    for C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        candidates.append((
            f"balanced_logistic_C{C}",
            LogisticRegression(
                C=C,
                penalty="l2",
                class_weight="balanced",
                multi_class="auto",
                max_iter=5000,
                solver="lbfgs",
                random_state=random_state,
                n_jobs=-1,
            ),
        ))
    candidates.append((
        "hist_gradient_boosting_meta",
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=random_state,
        ),
    ))
    candidates.append((
        "rf_meta_regularized",
        RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=4,
            class_weight="balanced",
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        ),
    ))

    best_model, best_info = None, None
    rows = []
    for name, clf in candidates:
        clf.fit(Z_train, y_train)
        val_pred = clf.predict(Z_val)
        val_prob = _safe_predict_proba(clf, Z_val, K)
        val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        rows.append({"candidate": name, "val_macro_f1": float(val_f1)})
        print(f"  [Meta] {name:<28s} val_macro_F1={val_f1:.4f}")
        if best_info is None or val_f1 > best_info["val_macro_f1"]:
            best_info = {"candidate": name, "val_macro_f1": float(val_f1), "val_prob": val_prob}
            best_model = clf
    best_info["all_candidates"] = rows
    return best_model, best_info


def train_ecotreefusenet(data: dict,
                         output_dir: str,
                         device: torch.device,
                         final_ecofuse_model: EcoFuseNetV2,
                         phi: np.ndarray,
                         random_state: int = 42,
                         n_splits: int = 5,
                         tree_estimators: int = 500,
                         xgb_max_depth: int = 6,
                         fast: bool = False) -> dict:
    """
    Train EcoTreeFuseNet: leakage-safe hybrid RF/XGB + EcoFuseNet-V2 stacking.

    This is the new proposed model. It uses OOF train branches and validation
    selection for the meta-classifier; D_test is used only once for final audit.
    """
    print("\n[Exp] EcoTreeFuseNet — leakage-safe hybrid tree-neural stacking")
    out_dir = Path(output_dir)
    pred_dir = out_dir / "predictions"
    stats_dir = out_dir / "stats"
    stack_dir = out_dir / "stacking"
    for d in [pred_dir, stats_dir, stack_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1) Tree OOF branches.
    tree_branches = generate_oof_tree_probabilities(
        data=data,
        output_dir=output_dir,
        random_state=random_state,
        n_splits=n_splits,
        n_estimators=tree_estimators,
        xgb_max_depth=xgb_max_depth,
    )

    # 2) EcoFuseNet OOF branch.
    fold_hparams = dict(
        warmup_epochs=3 if fast else 15,
        max_epochs=10 if fast else 90,
        patience=4 if fast else 15,
        proto_weight=0.005 if fast else 0.01,
        coarse_weight=0.002 if fast else 0.005,
        sep_weight=0.0001 if fast else 0.0005,
    )
    eco_branches = generate_ecofusenet_oof_features(
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

    # 3) Build 316-dimensional hybrid matrices.
    Z_train = build_hybrid_matrix(
        data["X_train"],
        tree_branches["rf_oof_train"],
        tree_branches["xgb_oof_train"],
        eco_branches["eco_oof_train_probs"],
        eco_branches["eco_oof_train_embeddings"],
        eco_branches["eco_oof_train_gates"],
        eco_branches["eco_oof_train_distances"],
    )
    Z_val = build_hybrid_matrix(
        data["X_val"],
        tree_branches["rf_val"],
        tree_branches["xgb_val"],
        eco_branches["eco_val_probs"],
        eco_branches["eco_val_embeddings"],
        eco_branches["eco_val_gates"],
        eco_branches["eco_val_distances"],
    )
    Z_test = build_hybrid_matrix(
        data["X_test"],
        tree_branches["rf_test"],
        tree_branches["xgb_test"],
        eco_branches["eco_test_probs"],
        eco_branches["eco_test_embeddings"],
        eco_branches["eco_test_gates"],
        eco_branches["eco_test_distances"],
    )

    np.save(pred_dir / "hybrid_features_train.npy", Z_train)
    np.save(pred_dir / "hybrid_features_val.npy", Z_val)
    np.save(pred_dir / "hybrid_features_test.npy", Z_test)

    # 4) Meta-feature scaling is fitted on OOF training features only.
    scaler = SkStandardScaler()
    Z_train_s = scaler.fit_transform(Z_train)
    Z_val_s = scaler.transform(Z_val)
    Z_test_s = scaler.transform(Z_test)
    np.savez(pred_dir / "hybrid_meta_scaler.npz", mean=scaler.mean_, scale=scaler.scale_)

    # 5) Validation-selected meta-classifier.
    meta_model, meta_info = train_meta_classifier(
        Z_train_s, data["y_fine_train"], Z_val_s, data["y_fine_val"], data["K"], random_state=random_state
    )
    y_pred = meta_model.predict(Z_test_s).astype(np.int64)
    y_prob = _safe_predict_proba(meta_model, Z_test_s, data["K"])
    macro_f1 = f1_score(data["y_fine_test"], y_pred, average="macro", zero_division=0)
    print(f"[EcoTreeFuseNet] Test Macro F1 = {macro_f1:.4f}")

    # 6) Save outputs.
    np.save(pred_dir / "predictions_ecotreefusenet.npy", y_pred)
    np.save(pred_dir / "probabilities_ecotreefusenet.npy", y_prob)
    # Save the interpretable neural branch internals under the proposed model name too.
    np.save(pred_dir / "embeddings_ecotreefusenet.npy", eco_branches["eco_test_embeddings"])
    np.save(pred_dir / "gate_values_ecotreefusenet.npy", eco_branches["eco_test_gates"])
    np.save(pred_dir / "proto_distances_ecotreefusenet.npy", eco_branches["eco_test_distances"])

    meta_info["test_macro_f1"] = float(macro_f1)
    meta_info["meta_feature_dim"] = int(Z_train.shape[1])
    meta_info["meta_blocks"] = META_BLOCKS
    meta_info["tree_diagnostics"] = tree_branches["diagnostics"]
    meta_info["eco_oof_diagnostics"] = eco_branches["diagnostics"]
    with open(stack_dir / "ecotreefusenet_meta_report.json", "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2, default=str)

    # Meta feature importance for logistic models.
    try:
        if hasattr(meta_model, "coef_"):
            imp = np.mean(np.abs(meta_model.coef_), axis=0)
        elif hasattr(meta_model, "feature_importances_"):
            imp = meta_model.feature_importances_
        else:
            imp = np.zeros(Z_train.shape[1], dtype=np.float32)
        pd.DataFrame({"feature_index": np.arange(len(imp)), "importance": imp}).to_csv(
            stats_dir / "table_ecotreefusenet_meta_feature_importance.csv", index=False
        )
    except Exception as exc:
        print(f"  [Meta] Feature-importance export skipped: {exc}")

    return {
        "exp_name": "ecotreefusenet",
        "test_macro_f1": float(macro_f1),
        "preds": y_pred,
        "probs": y_prob,
        "labels": data["y_fine_test"],
        "meta_model": meta_model,
        "meta_info": meta_info,
    }

# ---------------------------------------------------------------------------
# PHASE 2 OVERRIDE
# ---------------------------------------------------------------------------
# train_ecotreefusenet() resolves this global function at runtime. Rebinding it
# here ensures the legacy hybrid path also uses the corrected fold-specific OOF
# neural procedure instead of the historical holdout-as-validation routine.
from oof import generate_ecofusenet_oof_features  # noqa: E402,F401
