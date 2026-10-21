"""
Data Preparation — Phase 0 & 1 of EcoFuseNet pipeline

Implements:
  - Column selection and label encoding
  - CHM log1p preprocessing
  - Spatially aware stratified split (70/10/20)
  - Z-score normalization (training statistics only)
  - Spatial leakage diagnostics
  - PyTorch Dataset wrappers
"""

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.spatial import cKDTree
from pathlib import Path


# ── Column definitions ────────────────────────────────────────────────────────

PREDICTOR_COLS = [
    'elevation', 'slope', 'curvature', 'tpi',
    'solar_insolation', 'chm', 'ndvi', 'ndwi'
]
LABEL_FINE   = 'class'
LABEL_COARSE = 'ref_Type'
SPATIAL_COLS = ['x_original', 'y_original']

EXCLUDE_COLS = [
    'sample_id', 'reference_crs', 'ref_Longitude', 'ref_Latitude',
    'ref_Classnum', 'ref_Subclass', 'ref_Source',
    'n_missing_features', 'has_any_missing_feature'
]

# Ecological coarse-type → fine-class mapping (from proposal Table 2)
COARSE_TO_FINE = {
    'Community': [
        'IrrigLawnPasture', 'DrySubalpShale', 'AlpineSandstone', 'AlpineGranite',
        'DrySubalpTill', 'AlpineShale', 'WetMeadowTill', 'EmergentWetland',
        'DrySubalpGranite', 'WetAlpine', 'WetMeadowAlluv', 'DrySubalpSandstone',
        'AspenFringe'
    ],
    'Non-vegetated': [
        'GraniteOutcrop', 'Water', 'GraniteTalus', 'StreamSandGravel',
        'GravelRoadTrail', 'ShaleBarren', 'Paved', 'SandstoneOutcrop',
        'Snow', 'SandstoneTalus', 'BareSoil'
    ],
    'Herbs'  : ['Lupinus', 'Wyethia', 'Frasera'],
    'Shrubs' : ['Artemisia'],
    'Grasses': ['Festuca'],
}


# ── Standard scaler ───────────────────────────────────────────────────────────

class StandardScaler:
    """Fits on training data only; transforms any split without leakage."""

    def __init__(self, eps: float = 1e-8):
        self.mean_  = None
        self.std_   = None
        self.eps    = eps

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0) + self.eps
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str):
        np.savez(path, mean=self.mean_, std=self.std_)

    def load(self, path: str):
        d = np.load(path)
        self.mean_ = d['mean']
        self.std_  = d['std']
        return self


# ── Spatial split ─────────────────────────────────────────────────────────────

def spatial_stratified_split(df: pd.DataFrame,
                               train_frac: float = 0.70,
                               val_frac: float   = 0.10,
                               test_frac: float  = 0.20,
                               n_grid: int = 5,
                               random_state: int = 42) -> tuple:
    """
    Spatially aware stratified split (Algorithm 0.2).

    Divides spatial domain into n_grid × n_grid blocks using quantile cuts,
    then performs stratified sampling by fine class within each block.

    Returns:
        (train_idx, val_idx, test_idx) as numpy arrays of integer indices
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    rng = np.random.default_rng(random_state)

    x_bins = pd.qcut(df['x_original'], q=n_grid, labels=False, duplicates='drop')
    y_bins = pd.qcut(df['y_original'], q=n_grid, labels=False, duplicates='drop')
    block_id = x_bins.astype(str) + '_' + y_bins.astype(str)
    df = df.copy()
    df['_block'] = block_id.values

    train_idx, val_idx, test_idx = [], [], []
    fine_label_col = LABEL_FINE

    for block, group in df.groupby('_block'):
        idx = group.index.to_numpy()
        labels = group[fine_label_col].values

        if len(idx) < 3:
            # too small to split — assign all to train
            train_idx.extend(idx.tolist())
            continue

        # First split: extract test set (20%)
        try:
            sss1 = StratifiedShuffleSplit(
                n_splits=1, test_size=test_frac, random_state=random_state)
            tv_idx_local, test_idx_local = next(sss1.split(idx, labels))
            tv_idx   = idx[tv_idx_local]
            test_idx_block = idx[test_idx_local]
        except ValueError:
            # Not enough samples to stratify — do random split
            perm = rng.permutation(len(idx))
            n_test = max(1, int(len(idx) * test_frac))
            test_idx_block = idx[perm[:n_test]]
            tv_idx         = idx[perm[n_test:]]

        # Second split: from remaining, extract val (10 / 80 ≈ 12.5%)
        try:
            val_frac_of_tv = val_frac / (train_frac + val_frac)
            tv_labels = df.loc[tv_idx, fine_label_col].values
            sss2 = StratifiedShuffleSplit(
                n_splits=1, test_size=val_frac_of_tv, random_state=random_state)
            tr_local, va_local = next(sss2.split(tv_idx, tv_labels))
            train_idx_block = tv_idx[tr_local]
            val_idx_block   = tv_idx[va_local]
        except ValueError:
            n_val = max(1, int(len(tv_idx) * val_frac_of_tv))
            perm  = rng.permutation(len(tv_idx))
            val_idx_block   = tv_idx[perm[:n_val]]
            train_idx_block = tv_idx[perm[n_val:]]

        train_idx.extend(train_idx_block.tolist())
        val_idx.extend(val_idx_block.tolist())
        test_idx.extend(test_idx_block.tolist())

    # Guarantee every class has at least 1 training sample
    all_classes = df[fine_label_col].unique()
    train_classes = df.loc[train_idx, fine_label_col].unique()
    missing = set(all_classes) - set(train_classes)

    if missing:
        for cls in missing:
            candidates = df[df[fine_label_col] == cls].index.tolist()
            # Move one sample from val or test to train
            for pool, other in [(val_idx, 'val'), (test_idx, 'test')]:
                cls_in_pool = [i for i in pool if df.loc[i, fine_label_col] == cls]
                if cls_in_pool:
                    chosen = cls_in_pool[0]
                    if other == 'val':
                        val_idx.remove(chosen)
                    else:
                        test_idx.remove(chosen)
                    train_idx.append(chosen)
                    break

    return (np.array(train_idx), np.array(val_idx), np.array(test_idx))


# ── Spatial leakage diagnostic ────────────────────────────────────────────────

def spatial_leakage_diagnostic(coords_train: np.ndarray,
                                 coords_test: np.ndarray) -> dict:
    """
    Compute nearest-neighbor distances from each test point to the training set.
    Returns diagnostic statistics.
    """
    tree = cKDTree(coords_train)
    dists, _ = tree.query(coords_test, k=1)
    return {
        'median_nn_dist' : float(np.median(dists)),
        'mean_nn_dist'   : float(np.mean(dists)),
        'p5_nn_dist'     : float(np.percentile(dists, 5)),
        'p95_nn_dist'    : float(np.percentile(dists, 95)),
        'min_nn_dist'    : float(dists.min()),
        'max_nn_dist'    : float(dists.max()),
        'n_test'         : len(dists),
        'flag_potential_leakage': bool(np.median(dists) < 1.0),
    }


# ── Main data loading function ────────────────────────────────────────────────

def load_and_prepare_data(csv_path: str,
                           output_dir: str = 'outputs',
                           random_state: int = 42,
                           apply_chm_log1p: bool = True) -> dict:
    """
    Full data preparation pipeline.

    Args:
        csv_path       : path to vegetation_reference.csv
        output_dir     : directory to save encoders, scaler, split indices
        random_state   : reproducibility seed
        apply_chm_log1p: apply log1p to chm before standardization (recommended)

    Returns:
        data dict with keys: X_train, X_val, X_test, y_fine_train, y_fine_val,
        y_fine_test, y_coarse_train, y_coarse_val, y_coarse_test,
        fine_enc, coarse_enc, scaler, class_weights, coarse_map,
        train_idx, val_idx, test_idx, leakage_diag
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 0. Load ───────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    print(f"[Data] Loaded {len(df)} rows, {df.shape[1]} columns")

    # Verify no missing features
    assert df['has_any_missing_feature'].sum() == 0, \
        "Some rows have missing features — fix dataset before proceeding"
    print("[Data] ✓ All 1833 samples have complete features")

    # ── 1. Optional CHM log1p transform ──────────────────────────────────
    if apply_chm_log1p:
        df['chm'] = np.log1p(df['chm'])
        print("[Data] ✓ Applied log1p transform to CHM (right-skewed)")

    # ── 2. Encode labels ──────────────────────────────────────────────────
    fine_enc   = LabelEncoder()
    coarse_enc = LabelEncoder()
    df['y_fine']   = fine_enc.fit_transform(df[LABEL_FINE])
    df['y_coarse'] = coarse_enc.fit_transform(df[LABEL_COARSE])

    K = len(fine_enc.classes_)
    C = len(coarse_enc.classes_)
    print(f"[Data] Fine classes: {K}, Coarse types: {C}")
    print(f"[Data] Fine class names: {list(fine_enc.classes_)}")

    # ── 3. Spatial split ──────────────────────────────────────────────────
    train_idx, val_idx, test_idx = spatial_stratified_split(
        df, random_state=random_state)
    print(f"[Data] Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # ── 4. Extract arrays ─────────────────────────────────────────────────
    X       = df[PREDICTOR_COLS].values.astype(np.float32)
    y_fine  = df['y_fine'].values.astype(np.int64)
    y_coarse= df['y_coarse'].values.astype(np.int64)

    X_train   = X[train_idx];   X_val   = X[val_idx];   X_test   = X[test_idx]
    yf_train  = y_fine[train_idx]; yf_val = y_fine[val_idx]; yf_test = y_fine[test_idx]
    yc_train  = y_coarse[train_idx]; yc_val = y_coarse[val_idx]; yc_test = y_coarse[test_idx]

    # ── 5. Standardization (train stats only) ─────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)
    scaler.save(str(out_dir / 'scaler.npz'))
    print("[Data] ✓ Features standardized using training statistics only")

    # ── 6. Class weights ──────────────────────────────────────────────────
    from losses import compute_class_weights
    class_counts = np.bincount(yf_train, minlength=K)
    class_weights = compute_class_weights(class_counts, mode='sqrt')
    print(f"[Data] Class weights min={class_weights.min():.3f}, "
          f"max={class_weights.max():.3f}")

    # ── 7. Build coarse map for each fine class ───────────────────────────
    coarse_map = {}  # fine_class_name → coarse_type_name
    for coarse_name, fine_list in COARSE_TO_FINE.items():
        for f in fine_list:
            coarse_map[f] = coarse_name
    # Build index-based map: fine_idx → coarse_idx
    phi = np.zeros(K, dtype=np.int64)
    for fi, fname in enumerate(fine_enc.classes_):
        coarse_name = coarse_map.get(fname, None)
        if coarse_name is None:
            # Fallback: derive from the dataset ref_Type
            matches = df[df[LABEL_FINE] == fname][LABEL_COARSE].mode()
            coarse_name = matches.iloc[0] if len(matches) > 0 else coarse_enc.classes_[0]
        ci = np.where(coarse_enc.classes_ == coarse_name)[0]
        phi[fi] = int(ci[0]) if len(ci) > 0 else 0
    np.save(str(out_dir / 'phi.npy'), phi)

    # ── 8. Spatial leakage diagnostic ────────────────────────────────────
    coords = df[SPATIAL_COLS].values
    leakage_diag = spatial_leakage_diagnostic(
        coords[train_idx], coords[test_idx])
    print(f"[Data] Spatial leakage check — median NN dist: "
          f"{leakage_diag['median_nn_dist']:.2f} m "
          f"(flag={leakage_diag['flag_potential_leakage']})")

    # ── 9. Source balance check ───────────────────────────────────────────
    for split_name, split_idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        src_dist = df.loc[split_idx, 'ref_Source'].value_counts(normalize=True).to_dict()
        print(f"[Data] {split_name} source dist: {src_dist}")

    # ── 10. Save metadata ─────────────────────────────────────────────────
    with open(str(out_dir / 'class_names.json'), 'w') as f:
        json.dump(list(fine_enc.classes_), f, indent=2)
    with open(str(out_dir / 'coarse_map.json'), 'w') as f:
        json.dump(coarse_map, f, indent=2)

    # Per-class test support
    test_support = {}
    for k, cname in enumerate(fine_enc.classes_):
        n = int((yf_test == k).sum())
        test_support[cname] = n
    with open(str(out_dir / 'test_support.json'), 'w') as f:
        json.dump(test_support, f, indent=2)
    print(f"[Data] ✓ Metadata saved to {out_dir}")

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_fine_train': yf_train, 'y_fine_val': yf_val, 'y_fine_test': yf_test,
        'y_coarse_train': yc_train, 'y_coarse_val': yc_val, 'y_coarse_test': yc_test,
        'fine_enc'    : fine_enc,
        'coarse_enc'  : coarse_enc,
        'scaler'      : scaler,
        'class_weights': class_weights,
        'coarse_map'  : coarse_map,
        'phi'         : phi,
        'train_idx'   : train_idx,
        'val_idx'     : val_idx,
        'test_idx'    : test_idx,
        'leakage_diag': leakage_diag,
        'K'           : K,
        'C'           : C,
    }


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class VegetationDataset(Dataset):
    """PyTorch Dataset for EcoFuseNet vegetation data."""

    def __init__(self, X: np.ndarray, y_fine: np.ndarray,
                 y_coarse: np.ndarray):
        self.X        = torch.tensor(X,        dtype=torch.float32)
        self.y_fine   = torch.tensor(y_fine,   dtype=torch.long)
        self.y_coarse = torch.tensor(y_coarse, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_fine[idx], self.y_coarse[idx]


def make_dataloaders(data: dict,
                     batch_size: int = 64,
                     num_workers: int = 0) -> dict:
    """Create train/val/test DataLoaders from prepared data dict."""
    train_ds = VegetationDataset(
        data['X_train'], data['y_fine_train'], data['y_coarse_train'])
    val_ds   = VegetationDataset(
        data['X_val'],   data['y_fine_val'],   data['y_coarse_val'])
    test_ds  = VegetationDataset(
        data['X_test'],  data['y_fine_test'],  data['y_coarse_test'])

    return {
        'train': DataLoader(train_ds, batch_size=batch_size,
                             shuffle=True,  num_workers=num_workers,
                             pin_memory=False),
        'val'  : DataLoader(val_ds,   batch_size=256,
                             shuffle=False, num_workers=num_workers),
        'test' : DataLoader(test_ds,  batch_size=256,
                             shuffle=False, num_workers=num_workers),
    }