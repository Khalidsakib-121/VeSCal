"""Leakage-corrected OOF neural feature generation for Phase 2.

The original pipeline trained each fold-specific EcoFuseNet-V2 without gradient
updates from the OOF holdout, but reused the OOF holdout as the validation set for
checkpoint selection and reused the outer-training scaler. Phase 2 fixes both:

* fold-training data are split again into INNER-TRAIN and INNER-VALIDATION;
* a new scaler is fitted on INNER-TRAIN only;
* checkpoint selection/early stopping use INNER-VALIDATION only;
* the OOF holdout is transformed with that fold scaler and is touched only once,
  after the checkpoint is frozen;
* explicit index/sample-ID audit files prove the three sets are disjoint.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from spatial_data import StandardScaler, make_dataloaders
from losses import compute_class_weights
from network import EcoFuseNetV2
from training import train_ecofusenet_v2, evaluate_model


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _robust_inner_train_val_split(
    y: np.ndarray,
    val_fraction: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split fold-development samples into inner-train/inner-val.

    StratifiedShuffleSplit is used when feasible. If rare classes make exact
    stratification impossible, a deterministic class-aware fallback places at
    most one sample from each sufficiently represented class into validation,
    then fills toward the requested size without emptying a class from training.
    """
    n = len(y)
    if n < 3:
        raise ValueError("OOF fold development set is too small for inner validation")
    target_val = max(1, int(round(n * val_fraction)))
    target_val = min(target_val, n - 1)

    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=target_val, random_state=random_state)
        tr, va = next(sss.split(np.zeros((n, 1)), y))
        return tr.astype(int), va.astype(int)
    except ValueError:
        rng = np.random.default_rng(random_state)
        val = []
        train = set(range(n))
        for cls in rng.permutation(np.unique(y)):
            idx = np.where(y == cls)[0]
            if len(idx) >= 2 and len(val) < target_val:
                chosen = int(rng.choice(idx))
                val.append(chosen)
                train.remove(chosen)
        if len(val) < target_val:
            candidates = np.array(sorted(train), dtype=int)
            rng.shuffle(candidates)
            for chosen in candidates:
                cls = y[chosen]
                # Never remove the last training example of a class.
                remaining_same = sum((i in train) and y[i] == cls for i in range(n))
                if remaining_same <= 1:
                    continue
                val.append(int(chosen))
                train.remove(int(chosen))
                if len(val) >= target_val:
                    break
        if not val:
            chosen = int(rng.integers(0, n))
            val = [chosen]
            train.remove(chosen)
        return np.array(sorted(train), dtype=int), np.array(sorted(val), dtype=int)


def build_fold_safe_oof_data(
    data: dict,
    development_indices: np.ndarray,
    oof_holdout_indices: np.ndarray,
    inner_val_fraction: float = 0.15,
    random_state: int = 42,
) -> Tuple[dict, Dict[str, object]]:
    """Build one fold-specific data dictionary with zero checkpoint-label leakage.

    ``development_indices`` and ``oof_holdout_indices`` are positions within the
    OUTER training partition. The OOF holdout is never used for scaler fitting,
    class weights, gradient training, scheduling, early stopping, or checkpoint
    selection.
    """
    X_raw = data.get("X_train_raw")
    if X_raw is None:
        raise KeyError(
            "Phase 2 requires data['X_train_raw']. Use phase2_data_utils.load_and_prepare_data()."
        )
    y = data["y_fine_train"]
    yc = data["y_coarse_train"]
    K, C = data["K"], data["C"]

    development_indices = np.asarray(development_indices, dtype=int)
    oof_holdout_indices = np.asarray(oof_holdout_indices, dtype=int)
    if set(development_indices.tolist()) & set(oof_holdout_indices.tolist()):
        raise RuntimeError("Development and OOF holdout indices overlap")

    inner_tr_local, inner_va_local = _robust_inner_train_val_split(
        y[development_indices], val_fraction=inner_val_fraction, random_state=random_state
    )
    inner_train_idx = development_indices[inner_tr_local]
    inner_val_idx = development_indices[inner_va_local]

    sets = [set(inner_train_idx.tolist()), set(inner_val_idx.tolist()), set(oof_holdout_indices.tolist())]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("Inner train/validation/OOF holdout are not disjoint")

    # Critical Phase-2 correction: fold-specific scaler fit ONLY on INNER-TRAIN.
    scaler = StandardScaler().fit(X_raw[inner_train_idx])
    X_inner_train = scaler.transform(X_raw[inner_train_idx]).astype(np.float32)
    X_inner_val = scaler.transform(X_raw[inner_val_idx]).astype(np.float32)
    X_oof_holdout = scaler.transform(X_raw[oof_holdout_indices]).astype(np.float32)

    counts = np.bincount(y[inner_train_idx], minlength=K)
    class_weights = compute_class_weights(counts, mode="sqrt")

    fold_data = {
        "X_train": X_inner_train,
        "X_val": X_inner_val,
        "X_test": X_oof_holdout,
        "y_fine_train": y[inner_train_idx],
        "y_fine_val": y[inner_val_idx],
        "y_fine_test": y[oof_holdout_indices],
        "y_coarse_train": yc[inner_train_idx],
        "y_coarse_val": yc[inner_val_idx],
        "y_coarse_test": yc[oof_holdout_indices],
        "K": K,
        "C": C,
        "class_weights": class_weights,
    }

    outer_global_idx = np.asarray(data["train_idx"], dtype=int)
    sample_ids_all = np.asarray(data.get("sample_ids_all", np.arange(len(data.get("X_all_raw", []))).astype(str)))
    audit = {
        "random_state": int(random_state),
        "inner_val_fraction_requested": float(inner_val_fraction),
        "n_inner_train": int(len(inner_train_idx)),
        "n_inner_validation": int(len(inner_val_idx)),
        "n_oof_holdout": int(len(oof_holdout_indices)),
        "inner_train_outer_train_positions": inner_train_idx.tolist(),
        "inner_validation_outer_train_positions": inner_val_idx.tolist(),
        "oof_holdout_outer_train_positions": oof_holdout_indices.tolist(),
        "inner_train_global_indices": outer_global_idx[inner_train_idx].tolist(),
        "inner_validation_global_indices": outer_global_idx[inner_val_idx].tolist(),
        "oof_holdout_global_indices": outer_global_idx[oof_holdout_indices].tolist(),
        "inner_train_sample_ids": sample_ids_all[outer_global_idx[inner_train_idx]].astype(str).tolist(),
        "inner_validation_sample_ids": sample_ids_all[outer_global_idx[inner_val_idx]].astype(str).tolist(),
        "oof_holdout_sample_ids": sample_ids_all[outer_global_idx[oof_holdout_indices]].astype(str).tolist(),
        "overlap_inner_train_validation": [],
        "overlap_inner_train_oof": [],
        "overlap_inner_validation_oof": [],
        "scaler_fit_scope": "inner_train_only",
        "checkpoint_selection_scope": "inner_validation_only",
        "oof_holdout_role": "final_feature_extraction_only_after_checkpoint_freeze",
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.std_.tolist(),
        "n_train_classes": int(len(np.unique(y[inner_train_idx]))),
        "n_validation_classes": int(len(np.unique(y[inner_val_idx]))),
        "n_oof_classes": int(len(np.unique(y[oof_holdout_indices]))),
    }
    return fold_data, audit


@torch.no_grad()
def extract_ecofusenet_features(
    model: EcoFuseNetV2,
    X: np.ndarray,
    y_fine: np.ndarray,
    y_coarse: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    fold_data = {
        "X_train": X,
        "X_val": X,
        "X_test": X,
        "y_fine_train": y_fine,
        "y_fine_val": y_fine,
        "y_fine_test": y_fine,
        "y_coarse_train": y_coarse,
        "y_coarse_val": y_coarse,
        "y_coarse_test": y_coarse,
    }
    loaders = make_dataloaders(fold_data, batch_size=batch_size)
    res = evaluate_model(
        model, loaders["test"], device, K=model.K, classifier_head=None, collect_internal=True
    )
    return {
        "probs": res["probs"].astype(np.float32),
        "embeddings": res["embeddings"].astype(np.float32),
        "gates": res["gate_values"].astype(np.float32),
        "distances": res["proto_distances"].astype(np.float32),
    }


def generate_ecofusenet_oof_features(
    data: dict,
    output_dir: str,
    final_model: EcoFuseNetV2,
    device: torch.device,
    phi: np.ndarray,
    random_state: int = 42,
    n_splits: int = 5,
    batch_size: int = 64,
    fold_hparams: Optional[dict] = None,
    inner_val_fraction: float = 0.15,
) -> Dict[str, np.ndarray]:
    """Generate fully fold-specific EcoFuseNet OOF features.

    This is the Phase-2 replacement for the original OOF neural routine.
    """
    print(
        f"\n[Phase 2 OOF] Generating fold-safe EcoFuseNet-V2 OOF features "
        f"({n_splits} outer-training folds, inner_val={inner_val_fraction:.2f})..."
    )
    out_dir = Path(output_dir)
    stack_dir = out_dir / "stacking" / "phase2_ecofusenet_oof"
    pred_dir = out_dir / "predictions"
    stack_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    K, C = data["K"], data["C"]
    y_train = data["y_fine_train"]
    n_train = len(y_train)
    d_e, d_f = final_model.d_e, final_model.d_f

    oof_probs = np.zeros((n_train, K), dtype=np.float32)
    oof_emb = np.zeros((n_train, d_e), dtype=np.float32)
    oof_gate = np.zeros((n_train, d_f), dtype=np.float32)
    oof_dist = np.zeros((n_train, K), dtype=np.float32)
    fold_scores = []
    fold_audits = []

    hparams = dict(
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
        hparams.update(fold_hparams)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (development, holdout) in enumerate(skf.split(data["X_train_raw"], y_train), start=1):
        fold_seed = random_state + 10000 + fold
        set_seed(fold_seed)
        print(f"  [Phase 2 OOF] Fold {fold}/{n_splits}")

        fold_data, audit = build_fold_safe_oof_data(
            data=data,
            development_indices=development,
            oof_holdout_indices=holdout,
            inner_val_fraction=inner_val_fraction,
            random_state=fold_seed,
        )
        audit["fold"] = int(fold)
        fold_audits.append(audit)

        fold_out = stack_dir / f"fold_{fold}"
        fold_out.mkdir(parents=True, exist_ok=True)
        with open(fold_out / "phase2_oof_fold_audit.json", "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)

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

        # Trainer sees INNER-VALIDATION as loaders['val']; OOF holdout is only
        # loaders['test'], evaluated after the best checkpoint has been restored.
        res = train_ecofusenet_v2(
            fold_model,
            fold_loaders,
            fold_data,
            phi,
            fold_data["class_weights"],
            device,
            str(fold_out),
            exp_name=f"phase2_ecofusenet_oof_fold{fold}",
            **hparams,
        )
        oof_probs[holdout] = res["probs"].astype(np.float32)
        oof_emb[holdout] = res["embeddings"].astype(np.float32)
        oof_gate[holdout] = res["gate_values"].astype(np.float32)
        oof_dist[holdout] = res["proto_distances"].astype(np.float32)
        fold_scores.append(float(res["test_macro_f1"]))

    if np.any(np.isclose(oof_probs.sum(axis=1), 0.0)):
        raise RuntimeError("At least one outer-training sample did not receive an OOF neural probability vector")

    # Final full neural model remains trained on OUTER train with OUTER validation;
    # it produces the features used for outer validation/test inference.
    val_feat = extract_ecofusenet_features(
        final_model, data["X_val"], data["y_fine_val"], data["y_coarse_val"], device
    )
    test_feat = extract_ecofusenet_features(
        final_model, data["X_test"], data["y_fine_test"], data["y_coarse_test"], device
    )

    arrays = {
        "ecofuse_oof_train_probs": oof_probs,
        "ecofuse_oof_train_embeddings": oof_emb,
        "ecofuse_oof_train_gate_values": oof_gate,
        "ecofuse_oof_train_proto_distances": oof_dist,
        "ecofuse_val_probs": val_feat["probs"],
        "ecofuse_val_embeddings": val_feat["embeddings"],
        "ecofuse_val_gate_values": val_feat["gates"],
        "ecofuse_val_proto_distances": val_feat["distances"],
        "ecofuse_test_probs": test_feat["probs"],
        "ecofuse_test_embeddings": test_feat["embeddings"],
        "ecofuse_test_gate_values": test_feat["gates"],
        "ecofuse_test_proto_distances": test_feat["distances"],
    }
    for name, arr in arrays.items():
        np.save(pred_dir / f"{name}.npy", arr)

    diag = {
        "phase": 2,
        "correction": "fold_specific_scaler_plus_inner_validation_checkpoint_selection",
        "n_splits": int(n_splits),
        "inner_val_fraction": float(inner_val_fraction),
        "fold_macro_f1": fold_scores,
        "mean_oof_fold_macro_f1": float(np.mean(fold_scores)),
        "sample_sd_oof_fold_macro_f1": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
        "d_e": int(d_e),
        "d_f": int(d_f),
        "all_folds_index_disjoint": True,
        "outer_holdout_used_for_checkpoint_selection": False,
        "scaler_fit_within_each_fold": True,
    }
    with open(stack_dir / "phase2_ecofusenet_oof_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)
    with open(stack_dir / "phase2_all_fold_audits.json", "w", encoding="utf-8") as f:
        json.dump(fold_audits, f, indent=2)

    print(
        f"[Phase 2 OOF] ✓ fold-safe OOF complete; mean holdout Macro-F1="
        f"{diag['mean_oof_fold_macro_f1']:.4f}"
    )
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


__all__ = ["build_fold_safe_oof_data", "generate_ecofusenet_oof_features"]
