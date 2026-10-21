"""Phase 2 data preparation for EcoTreeFuseNet-Plus.

Corrections implemented in this module
--------------------------------------
1. Preserves the original within-region split for backward-compatible primary runs.
2. Adds *strict spatial block holdout* evaluation. Complete geographic blocks are
   assigned to train/validation/test; no test block is allowed in train or validation.
3. Exposes the unscaled, CHM-transformed predictor arrays needed by fold-specific
   OOF preprocessing. This prevents the outer-training scaler from leaking OOF-fold
   distribution information into fold-specific neural models.
4. Saves explicit split/block/sample-ID audit artifacts for reviewer verification.

Python: 3.10+
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

# Reuse stable utilities/definitions from the original implementation.
from data_utils import (
    PREDICTOR_COLS,
    LABEL_FINE,
    LABEL_COARSE,
    SPATIAL_COLS,
    COARSE_TO_FINE,
    StandardScaler,
    spatial_stratified_split,
    spatial_leakage_diagnostic,
    VegetationDataset,
    make_dataloaders,
)


def build_spatial_blocks(df: pd.DataFrame, n_grid: int = 5) -> pd.Series:
    """Create quantile-grid spatial block IDs from projected x/y coordinates.

    Blocks are deterministic for a given dataset and n_grid. Each block is atomic:
    strict spatial validation never splits a block across train/val/test.
    """
    if n_grid < 2:
        raise ValueError("n_grid must be >= 2")
    x_bins = pd.qcut(df[SPATIAL_COLS[0]], q=n_grid, labels=False, duplicates="drop")
    y_bins = pd.qcut(df[SPATIAL_COLS[1]], q=n_grid, labels=False, duplicates="drop")
    return (x_bins.astype(str) + "_" + y_bins.astype(str)).rename("spatial_block")


def _class_coverage(y: np.ndarray, idx: np.ndarray, K: int) -> Dict[str, object]:
    present = np.unique(y[idx]) if len(idx) else np.array([], dtype=int)
    missing = sorted(set(range(K)) - set(map(int, present.tolist())))
    return {
        "n_samples": int(len(idx)),
        "n_classes_present": int(len(present)),
        "missing_class_indices": [int(v) for v in missing],
    }


def strict_spatial_block_split(
    df: pd.DataFrame,
    y_encoded: np.ndarray,
    spatial_fold: int = 0,
    n_splits: int = 5,
    n_grid: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Create a strict geographic block train/validation/test split.

    Outer test fold
      Selected with StratifiedGroupKFold using spatial block IDs as groups.
      Complete test blocks are held out.

    Inner validation fold
      Chosen from the remaining blocks with a second StratifiedGroupKFold.
      Among candidate inner folds, the split maximizing training class coverage,
      then validation class coverage, is selected. No sample is moved between
      blocks to repair class coverage.

    This yields approximately 60-65% train, 15-20% validation, 20% test depending
    on block sizes. The exact proportions are intentionally secondary to strict
    geographic separation.
    """
    if n_splits < 3:
        raise ValueError("n_splits must be >= 3 for train/validation/test block holdout")
    spatial_fold = int(spatial_fold)
    if spatial_fold < 0 or spatial_fold >= n_splits:
        raise ValueError(f"spatial_fold must be in [0, {n_splits-1}]")

    block_ids = build_spatial_blocks(df, n_grid=n_grid).to_numpy()
    all_idx = np.arange(len(df), dtype=int)
    K = int(np.max(y_encoded)) + 1

    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    outer_splits = list(outer.split(np.zeros((len(df), 1)), y_encoded, groups=block_ids))
    remain_idx, test_idx = outer_splits[spatial_fold]

    # Strict check: complete groups only.
    test_blocks = set(block_ids[test_idx].tolist())
    remain_blocks = set(block_ids[remain_idx].tolist())
    if test_blocks & remain_blocks:
        raise RuntimeError("Spatial block leakage detected while creating outer test fold")

    # Inner validation from remaining complete blocks. Try all candidate folds and
    # select the most class-preserving training split without touching test blocks.
    remain_y = y_encoded[remain_idx]
    remain_groups = block_ids[remain_idx]
    n_unique_remain = len(np.unique(remain_groups))
    inner_splits = min(max(2, n_splits), n_unique_remain)
    inner = StratifiedGroupKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=random_state + 1000 + spatial_fold,
    )

    candidates = []
    for inner_no, (tr_local, va_local) in enumerate(
        inner.split(np.zeros((len(remain_idx), 1)), remain_y, groups=remain_groups)
    ):
        tr_idx = remain_idx[tr_local]
        va_idx = remain_idx[va_local]
        tr_cov = len(np.unique(y_encoded[tr_idx]))
        va_cov = len(np.unique(y_encoded[va_idx]))
        # Prefer all training classes, then maximum validation coverage, then
        # validation size closest to ~20% of non-test samples.
        target_val = len(remain_idx) / inner_splits
        size_penalty = abs(len(va_idx) - target_val)
        score = (tr_cov, va_cov, -size_penalty, -inner_no)
        candidates.append((score, tr_idx, va_idx, inner_no))

    if not candidates:
        raise RuntimeError("Unable to construct a strict spatial validation fold")
    _, train_idx, val_idx, selected_inner = max(candidates, key=lambda x: x[0])

    train_blocks = set(block_ids[train_idx].tolist())
    val_blocks = set(block_ids[val_idx].tolist())
    if train_blocks & val_blocks or train_blocks & test_blocks or val_blocks & test_blocks:
        raise RuntimeError("Strict block separation assertion failed")

    union = np.sort(np.concatenate([train_idx, val_idx, test_idx]))
    if not np.array_equal(union, all_idx):
        raise RuntimeError("Strict spatial split is not collectively exhaustive")

    audit = {
        "split_mode": "strict_spatial_block",
        "n_grid": int(n_grid),
        "n_spatial_folds": int(n_splits),
        "spatial_fold": int(spatial_fold),
        "selected_inner_validation_fold": int(selected_inner),
        "n_total_blocks": int(len(np.unique(block_ids))),
        "train_blocks": sorted(train_blocks),
        "validation_blocks": sorted(val_blocks),
        "test_blocks": sorted(test_blocks),
        "n_train_blocks": int(len(train_blocks)),
        "n_validation_blocks": int(len(val_blocks)),
        "n_test_blocks": int(len(test_blocks)),
        "block_overlap_train_val": sorted(train_blocks & val_blocks),
        "block_overlap_train_test": sorted(train_blocks & test_blocks),
        "block_overlap_val_test": sorted(val_blocks & test_blocks),
        "class_coverage": {
            "train": _class_coverage(y_encoded, train_idx, K),
            "validation": _class_coverage(y_encoded, val_idx, K),
            "test": _class_coverage(y_encoded, test_idx, K),
        },
    }
    return train_idx.astype(int), val_idx.astype(int), test_idx.astype(int), audit


def _build_phi(df: pd.DataFrame, fine_enc: LabelEncoder, coarse_enc: LabelEncoder) -> Tuple[dict, np.ndarray]:
    coarse_map = {}
    for coarse_name, fine_list in COARSE_TO_FINE.items():
        for f in fine_list:
            coarse_map[f] = coarse_name
    K = len(fine_enc.classes_)
    phi = np.zeros(K, dtype=np.int64)
    for fi, fname in enumerate(fine_enc.classes_):
        coarse_name = coarse_map.get(fname)
        if coarse_name is None:
            matches = df[df[LABEL_FINE] == fname][LABEL_COARSE].mode()
            coarse_name = matches.iloc[0] if len(matches) else coarse_enc.classes_[0]
        ci = np.where(coarse_enc.classes_ == coarse_name)[0]
        phi[fi] = int(ci[0]) if len(ci) else 0
    return coarse_map, phi


def load_and_prepare_data(
    csv_path: str,
    output_dir: str = "outputs",
    random_state: int = 42,
    apply_chm_log1p: bool = True,
    split_mode: str = "mixed",
    spatial_fold: int = 0,
    n_spatial_folds: int = 5,
    n_grid: int = 5,
) -> dict:
    """Phase-2 data preparation with optional strict spatial block holdout.

    split_mode:
      - ``mixed``: original within-block class-preserving split (historical primary)
      - ``spatial_block``: complete geographic blocks held out from development

    In both modes, the function returns both:
      - X_train/X_val/X_test: outer-training-scaled arrays for final models
      - X_*_raw: CHM-transformed but *unscaled* arrays for fold-specific OOF fitting
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path).reset_index(drop=True)
    required = set(PREDICTOR_COLS + [LABEL_FINE, LABEL_COARSE] + SPATIAL_COLS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "has_any_missing_feature" in df.columns and int(df["has_any_missing_feature"].sum()) != 0:
        raise ValueError("Dataset contains rows flagged with missing predictor values")

    if apply_chm_log1p:
        if (df["chm"] < 0).any():
            raise ValueError("CHM contains negative values; log1p is not safe")
        df["chm"] = np.log1p(df["chm"].astype(float))

    fine_enc = LabelEncoder()
    coarse_enc = LabelEncoder()
    df["y_fine"] = fine_enc.fit_transform(df[LABEL_FINE])
    df["y_coarse"] = coarse_enc.fit_transform(df[LABEL_COARSE])
    K = len(fine_enc.classes_)
    C = len(coarse_enc.classes_)
    y_fine = df["y_fine"].to_numpy(dtype=np.int64)
    y_coarse = df["y_coarse"].to_numpy(dtype=np.int64)

    block_series = build_spatial_blocks(df, n_grid=n_grid)
    df["spatial_block"] = block_series

    split_mode = str(split_mode).lower().strip()
    if split_mode == "mixed":
        train_idx, val_idx, test_idx = spatial_stratified_split(
            df, n_grid=n_grid, random_state=random_state
        )
        split_audit = {
            "split_mode": "mixed_within_block",
            "n_grid": int(n_grid),
            "n_spatial_folds": None,
            "spatial_fold": None,
            "train_blocks": sorted(set(df.loc[train_idx, "spatial_block"].tolist())),
            "validation_blocks": sorted(set(df.loc[val_idx, "spatial_block"].tolist())),
            "test_blocks": sorted(set(df.loc[test_idx, "spatial_block"].tolist())),
        }
        split_audit["block_overlap_train_test"] = sorted(
            set(split_audit["train_blocks"]) & set(split_audit["test_blocks"])
        )
    elif split_mode == "spatial_block":
        train_idx, val_idx, test_idx, split_audit = strict_spatial_block_split(
            df=df,
            y_encoded=y_fine,
            spatial_fold=spatial_fold,
            n_splits=n_spatial_folds,
            n_grid=n_grid,
            random_state=random_state,
        )
    else:
        raise ValueError("split_mode must be 'mixed' or 'spatial_block'")

    X_all_raw = df[PREDICTOR_COLS].to_numpy(dtype=np.float32)
    X_train_raw = X_all_raw[train_idx].copy()
    X_val_raw = X_all_raw[val_idx].copy()
    X_test_raw = X_all_raw[test_idx].copy()

    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)
    scaler.save(str(out_dir / "scaler.npz"))

    yf_train, yf_val, yf_test = y_fine[train_idx], y_fine[val_idx], y_fine[test_idx]
    yc_train, yc_val, yc_test = y_coarse[train_idx], y_coarse[val_idx], y_coarse[test_idx]

    from losses import compute_class_weights
    class_counts = np.bincount(yf_train, minlength=K)
    class_weights = compute_class_weights(class_counts, mode="sqrt")

    coarse_map, phi = _build_phi(df, fine_enc, coarse_enc)
    np.save(str(out_dir / "phi.npy"), phi)

    coords = df[SPATIAL_COLS].to_numpy(dtype=float)
    leakage_diag = spatial_leakage_diagnostic(coords[train_idx], coords[test_idx])

    # Reviewer-auditable split artifacts.
    split_audit.update({
        "seed": int(random_state),
        "n_samples": int(len(df)),
        "sizes": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "fractions": {
            "train": float(len(train_idx) / len(df)),
            "validation": float(len(val_idx) / len(df)),
            "test": float(len(test_idx) / len(df)),
        },
        "nearest_neighbor_test_to_train_m": leakage_diag,
    })

    with open(out_dir / "phase2_split_audit.json", "w", encoding="utf-8") as f:
        json.dump(split_audit, f, indent=2)
    np.save(out_dir / "train_indices.npy", train_idx)
    np.save(out_dir / "validation_indices.npy", val_idx)
    np.save(out_dir / "test_indices.npy", test_idx)
    np.save(out_dir / "all_spatial_block_ids.npy", df["spatial_block"].astype(str).to_numpy())
    if "sample_id" in df.columns:
        np.save(out_dir / "train_sample_ids.npy", df.loc[train_idx, "sample_id"].astype(str).to_numpy())
        np.save(out_dir / "validation_sample_ids.npy", df.loc[val_idx, "sample_id"].astype(str).to_numpy())
        np.save(out_dir / "test_sample_ids.npy", df.loc[test_idx, "sample_id"].astype(str).to_numpy())

    with open(out_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(list(fine_enc.classes_), f, indent=2)
    with open(out_dir / "coarse_map.json", "w", encoding="utf-8") as f:
        json.dump(coarse_map, f, indent=2)
    test_support = {c: int((yf_test == k).sum()) for k, c in enumerate(fine_enc.classes_)}
    with open(out_dir / "test_support.json", "w", encoding="utf-8") as f:
        json.dump(test_support, f, indent=2)

    print(f"[Phase 2 Data] split_mode={split_mode} seed={random_state} ")
    print(f"[Phase 2 Data] train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    if split_mode == "spatial_block":
        assert not split_audit["block_overlap_train_val"]
        assert not split_audit["block_overlap_train_test"]
        assert not split_audit["block_overlap_val_test"]
        print("[Phase 2 Data] ✓ strict geographic block separation verified")
    print(
        "[Phase 2 Data] test→train NN median="
        f"{leakage_diag['median_nn_dist']:.2f} m, min={leakage_diag['min_nn_dist']:.2f} m"
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "X_train_raw": X_train_raw,
        "X_val_raw": X_val_raw,
        "X_test_raw": X_test_raw,
        "X_all_raw": X_all_raw,
        "y_fine_train": yf_train,
        "y_fine_val": yf_val,
        "y_fine_test": yf_test,
        "y_coarse_train": yc_train,
        "y_coarse_val": yc_val,
        "y_coarse_test": yc_test,
        "fine_enc": fine_enc,
        "coarse_enc": coarse_enc,
        "scaler": scaler,
        "class_weights": class_weights,
        "coarse_map": coarse_map,
        "phi": phi,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "leakage_diag": leakage_diag,
        "split_audit": split_audit,
        "spatial_block_ids_all": df["spatial_block"].astype(str).to_numpy(),
        "sample_ids_all": df["sample_id"].astype(str).to_numpy() if "sample_id" in df.columns else np.arange(len(df)).astype(str),
        "coords_all": coords,
        "K": K,
        "C": C,
    }


__all__ = [
    "PREDICTOR_COLS", "LABEL_FINE", "LABEL_COARSE", "SPATIAL_COLS",
    "StandardScaler", "VegetationDataset", "make_dataloaders",
    "build_spatial_blocks", "strict_spatial_block_split", "load_and_prepare_data",
]
