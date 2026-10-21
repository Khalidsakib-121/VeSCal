"""
Baseline Models — Experiments 1 & 2

Experiment 1: Random Forest  (n_estimators=500, class_weight='balanced')
Experiment 2: XGBoost        (n_estimators=500, max_depth=6, lr=0.05)
"""

import numpy as np
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[Baselines] XGBoost not installed — run: pip install xgboost")


def train_random_forest(data: dict,
                         output_dir: str,
                         random_state: int = 42,
                         n_estimators: int = 500) -> dict:
    """
    Experiment 1 — Random Forest.
    n_estimators=500, class_weight='balanced', max_features='sqrt'
    """
    print(f"\n[RF] Training Random Forest (n_estimators={n_estimators})...")
    pred_dir = Path(output_dir) / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)

    X_train = data['X_train']
    X_test  = data['X_test']
    y_train = data['y_fine_train']
    y_test  = data['y_fine_test']

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight='balanced',
        max_features='sqrt',
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"  [RF] Test Macro F1 = {macro_f1:.4f}")

    np.save(str(pred_dir / 'predictions_rf.npy'), y_pred)
    np.save(str(pred_dir / 'probabilities_rf.npy'), y_prob)

    # Feature importances
    feat_imp = dict(zip(
        ['elevation', 'slope', 'curvature', 'tpi',
         'solar_insolation', 'chm', 'ndvi', 'ndwi'],
        rf.feature_importances_.tolist()
    ))
    with open(str(pred_dir / 'rf_feature_importance.json'), 'w') as f:
        json.dump(feat_imp, f, indent=2)

    return {
        'exp_name'     : 'rf',
        'test_macro_f1': macro_f1,
        'preds'        : y_pred,
        'probs'        : y_prob,
        'labels'       : y_test,
        'model'        : rf,
    }


def train_xgboost(data: dict,
                   output_dir: str,
                   random_state: int = 42,
                   n_estimators: int = 500,
                   max_depth: int = 6) -> dict:
    """
    Experiment 2 — XGBoost.
    n_estimators=500, max_depth=6, learning_rate=0.05
    """
    if not XGB_AVAILABLE:
        print("[XGBoost] Skipped — XGBoost not installed.")
        y_test = data['y_fine_test']
        dummy  = np.zeros(len(y_test), dtype=int)
        dummy_prob = np.ones((len(y_test), data['K'])) / data['K']
        return {
            'exp_name': 'xgboost', 'test_macro_f1': 0.0,
            'preds': dummy, 'probs': dummy_prob, 'labels': y_test
        }

    print(f"\n[XGBoost] Training XGBoost (n_estimators={n_estimators}, max_depth={max_depth})...")
    pred_dir = Path(output_dir) / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)

    X_train = data['X_train']
    X_test  = data['X_test']
    y_train = data['y_fine_train']
    y_test  = data['y_fine_test']
    K = data['K']

    # Compute scale_pos_weight as class-frequency inverse
    from collections import Counter
    counts = Counter(y_train.tolist())
    n_majority = max(counts.values())
    sample_weights = np.array([n_majority / counts[int(y)] for y in y_train])

    xgb_clf = make_xgb_model(K, random_state, n_estimators=n_estimators, max_depth=max_depth)
    xgb_clf.fit(X_train, y_train, sample_weight=sample_weights)

    y_prob = _align_proba(xgb_clf.predict_proba(X_test), xgb_clf.classes_, K)
    y_pred = y_prob.argmax(axis=1)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"  [XGBoost] Test Macro F1 = {macro_f1:.4f}")

    np.save(str(pred_dir / 'predictions_xgboost.npy'), y_pred)
    np.save(str(pred_dir / 'probabilities_xgboost.npy'), y_prob)

    # Feature importances for manuscript Figure 11 / tables.
    feat_imp = dict(zip(
        ['elevation', 'slope', 'curvature', 'tpi',
         'solar_insolation', 'chm', 'ndvi', 'ndwi'],
        xgb_clf.feature_importances_.tolist()
    ))
    with open(str(pred_dir / 'xgboost_feature_importance.json'), 'w') as f:
        json.dump(feat_imp, f, indent=2)

    return {
        'exp_name'     : 'xgboost',
        'test_macro_f1': macro_f1,
        'preds'        : y_pred,
        'probs'        : y_prob,
        'labels'       : y_test,
        'model'        : xgb_clf,
    }


def permutation_importance_with_bootstrap(model,
                                            X_test: np.ndarray,
                                            y_test: np.ndarray,
                                            feature_names: list,
                                            n_repeats: int = 20,
                                            random_state: int = 42) -> dict:
    """
    Permutation importance with bootstrap stability estimate.
    Works with any sklearn-compatible model.
    """
    from sklearn.metrics import f1_score as f1

    rng = np.random.default_rng(random_state)
    baseline_f1 = f1(y_test, model.predict(X_test),
                      average='macro', zero_division=0)
    n_features = X_test.shape[1]
    importance_boot = np.zeros((n_repeats, n_features))

    for rep in range(n_repeats):
        for j in range(n_features):
            X_perm = X_test.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_f1 = f1(y_test, model.predict(X_perm),
                          average='macro', zero_division=0)
            importance_boot[rep, j] = baseline_f1 - perm_f1

    mean_imp = importance_boot.mean(axis=0)
    std_imp  = importance_boot.std(axis=0)

    return {
        'features'  : feature_names,
        'mean'      : mean_imp.tolist(),
        'std'       : std_imp.tolist(),
        'baseline_f1': float(baseline_f1),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Leakage-safe out-of-fold probability branches for EcoTreeFuseNet
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.model_selection import StratifiedKFold


def _align_proba(proba: np.ndarray, classes: np.ndarray, K: int) -> np.ndarray:
    """Align sklearn predict_proba columns to fixed class indices 0..K-1."""
    out = np.zeros((proba.shape[0], K), dtype=np.float32)
    for j, c in enumerate(classes):
        if 0 <= int(c) < K:
            out[:, int(c)] = proba[:, j]
    row_sum = out.sum(axis=1, keepdims=True)
    out = np.divide(out, np.clip(row_sum, 1e-12, None), where=row_sum > 0)
    return out.astype(np.float32)



class MissingClassSafeXGB:
    """XGBoost adapter that tolerates globally encoded classes absent from a fold.

    Strict spatial holdout can legitimately leave one or more fine classes absent
    from a training fold. The sklearn XGBoost wrapper requires training labels to
    be contiguous. This adapter remaps only the *present* training classes to
    0..C_present-1 and exposes the original global class IDs through ``classes_``.
    Probability alignment back to K global classes is handled by _align_proba().
    """
    def __init__(self, random_state=42, n_estimators=500, max_depth=6):
        self.random_state = int(random_state)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y, sample_weight=None, **kwargs):
        if not XGB_AVAILABLE:
            raise RuntimeError("XGBoost is not installed")
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        y_local = np.searchsorted(self.classes_, y)
        n_present = int(len(self.classes_))
        if n_present < 2:
            raise ValueError("XGBoost fold contains fewer than two training classes")
        self.model_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=1,
            reg_lambda=3.0,
            reg_alpha=0.05,
            objective='multi:softprob',
            num_class=n_present,
            eval_metric='mlogloss',
            random_state=self.random_state,
            verbosity=0,
            n_jobs=-1,
        )
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs['sample_weight'] = sample_weight
        # Deliberately ignore eval_set passed in global label space; Phase-2 model
        # selection does not depend on XGBoost early stopping.
        self.model_.fit(X, y_local, **fit_kwargs)
        return self

    def predict_proba(self, X):
        if self.model_ is None:
            raise RuntimeError("Model is not fitted")
        return self.model_.predict_proba(X)

    def predict(self, X):
        p = self.predict_proba(X)
        return self.classes_[np.argmax(p, axis=1)]

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_ if self.model_ is not None else np.array([])


def make_rf_model(random_state: int = 42, n_estimators: int = 500) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight='balanced',
        max_features='sqrt',
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=random_state,
    )


def make_xgb_model(K: int, random_state: int = 42, n_estimators: int = 500, max_depth: int = 6):
    if not XGB_AVAILABLE:
        return None
    return MissingClassSafeXGB(
        random_state=random_state, n_estimators=n_estimators, max_depth=max_depth
    )


def _class_sample_weights(y: np.ndarray) -> np.ndarray:
    from collections import Counter
    counts = Counter(y.tolist())
    n_majority = max(counts.values())
    return np.array([n_majority / counts[int(v)] for v in y], dtype=np.float32)


def generate_oof_tree_probabilities(data: dict,
                                    output_dir: str,
                                    random_state: int = 42,
                                    n_splits: int = 5,
                                    n_estimators: int = 500,
                                    xgb_max_depth: int = 6) -> dict:
    """
    Generate leakage-safe OOF train probabilities for RF and XGBoost and
    validation/test probabilities from final models trained on D_train only.

    These are the tree probability branches used by EcoTreeFuseNet.
    """
    print(f"\n[Stacking] Generating leakage-safe RF/XGBoost OOF probabilities ({n_splits} folds)...")
    out_dir = Path(output_dir)
    pred_dir = out_dir / 'predictions'
    stack_dir = out_dir / 'stacking'
    pred_dir.mkdir(parents=True, exist_ok=True)
    stack_dir.mkdir(parents=True, exist_ok=True)

    # Phase 2: tree OOF branches consume the CHM-transformed *unscaled* arrays
    # when available. This removes any dependence on the outer-training scaler
    # during fold-specific probability generation (trees do not require scaling).
    X_train = data.get('X_train_raw', data['X_train'])
    y_train = data['y_fine_train']
    X_val = data.get('X_val_raw', data['X_val'])
    X_test = data.get('X_test_raw', data['X_test'])
    K = data['K']

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rf_oof = np.zeros((len(y_train), K), dtype=np.float32)
    xgb_oof = np.zeros((len(y_train), K), dtype=np.float32)
    rf_fold_f1, xgb_fold_f1 = [], []

    for fold, (tr, ho) in enumerate(skf.split(X_train, y_train), start=1):
        print(f"  [Stacking] Fold {fold}/{n_splits}: tree branches")
        rf = make_rf_model(random_state + fold, n_estimators=n_estimators)
        rf.fit(X_train[tr], y_train[tr])
        rf_p = _align_proba(rf.predict_proba(X_train[ho]), rf.classes_, K)
        rf_oof[ho] = rf_p
        rf_fold_f1.append(float(f1_score(y_train[ho], rf_p.argmax(axis=1), average='macro', zero_division=0)))

        if XGB_AVAILABLE:
            xgb_clf = make_xgb_model(K, random_state + fold, n_estimators=n_estimators, max_depth=xgb_max_depth)
            xgb_clf.fit(X_train[tr], y_train[tr], sample_weight=_class_sample_weights(y_train[tr]))
            xgb_p = xgb_clf.predict_proba(X_train[ho]).astype(np.float32)
            if xgb_p.shape[1] != K:
                xgb_p = _align_proba(xgb_p, np.arange(xgb_p.shape[1]), K)
            xgb_oof[ho] = xgb_p
            xgb_fold_f1.append(float(f1_score(y_train[ho], xgb_p.argmax(axis=1), average='macro', zero_division=0)))
        else:
            xgb_oof[ho] = np.ones((len(ho), K), dtype=np.float32) / K

    # Final tree models trained on full D_train; validation/test remain untouched.
    rf_final = make_rf_model(random_state, n_estimators=n_estimators)
    rf_final.fit(X_train, y_train)
    rf_val = _align_proba(rf_final.predict_proba(X_val), rf_final.classes_, K)
    rf_test = _align_proba(rf_final.predict_proba(X_test), rf_final.classes_, K)

    if XGB_AVAILABLE:
        xgb_final = make_xgb_model(K, random_state, n_estimators=n_estimators, max_depth=xgb_max_depth)
        xgb_final.fit(X_train, y_train, sample_weight=_class_sample_weights(y_train))
        xgb_val = xgb_final.predict_proba(X_val).astype(np.float32)
        xgb_test = xgb_final.predict_proba(X_test).astype(np.float32)
    else:
        xgb_final = None
        xgb_val = np.ones((len(X_val), K), dtype=np.float32) / K
        xgb_test = np.ones((len(X_test), K), dtype=np.float32) / K

    # Save branch outputs.
    np.save(pred_dir / 'rf_oof_train_probs.npy', rf_oof)
    np.save(pred_dir / 'rf_val_probs.npy', rf_val)
    np.save(pred_dir / 'rf_test_probs.npy', rf_test)
    np.save(pred_dir / 'xgb_oof_train_probs.npy', xgb_oof)
    np.save(pred_dir / 'xgb_val_probs.npy', xgb_val)
    np.save(pred_dir / 'xgb_test_probs.npy', xgb_test)

    # Save branch diagnostics and feature importances.
    diag = {
        'n_splits': int(n_splits),
        'rf_oof_macro_f1_mean': float(np.mean(rf_fold_f1)) if rf_fold_f1 else None,
        'rf_oof_macro_f1_std': float(np.std(rf_fold_f1)) if rf_fold_f1 else None,
        'xgb_oof_macro_f1_mean': float(np.mean(xgb_fold_f1)) if xgb_fold_f1 else None,
        'xgb_oof_macro_f1_std': float(np.std(xgb_fold_f1)) if xgb_fold_f1 else None,
    }
    with open(stack_dir / 'tree_oof_diagnostics.json', 'w', encoding='utf-8') as f:
        json.dump(diag, f, indent=2)

    feature_names = ['elevation', 'slope', 'curvature', 'tpi', 'solar_insolation', 'chm', 'ndvi', 'ndwi']
    with open(pred_dir / 'rf_feature_importance.json', 'w', encoding='utf-8') as f:
        json.dump(dict(zip(feature_names, rf_final.feature_importances_.tolist())), f, indent=2)
    if xgb_final is not None and hasattr(xgb_final, 'feature_importances_'):
        with open(pred_dir / 'xgboost_feature_importance.json', 'w', encoding='utf-8') as f:
            json.dump(dict(zip(feature_names, xgb_final.feature_importances_.tolist())), f, indent=2)

    print(f"  [Stacking] RF OOF macro-F1 mean={diag['rf_oof_macro_f1_mean']:.4f}")
    if diag['xgb_oof_macro_f1_mean'] is not None:
        print(f"  [Stacking] XGBoost OOF macro-F1 mean={diag['xgb_oof_macro_f1_mean']:.4f}")

    return {
        'rf_oof_train': rf_oof,
        'rf_val': rf_val,
        'rf_test': rf_test,
        'xgb_oof_train': xgb_oof,
        'xgb_val': xgb_val,
        'xgb_test': xgb_test,
        'rf_final': rf_final,
        'xgb_final': xgb_final,
        'diagnostics': diag,
    }

# ─────────────────────────────────────────────────────────────────────────────
# EcoTreeFuseNet-Plus: stronger tree branches and optional modern boosters
# ─────────────────────────────────────────────────────────────────────────────

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

FEATURE_NAMES = ['elevation', 'slope', 'curvature', 'tpi', 'solar_insolation', 'chm', 'ndvi', 'ndwi']


def make_extra_trees_model(random_state: int = 42, n_estimators: int = 800) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        class_weight='balanced',
        max_features='sqrt',
        min_samples_leaf=1,
        bootstrap=False,
        n_jobs=-1,
        random_state=random_state,
    )


def make_hgb_model(random_state: int = 42, max_iter: int = 350) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=max_iter,
        max_leaf_nodes=15,
        l2_regularization=0.05,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=random_state,
    )


def make_lgbm_model(K: int, random_state: int = 42, n_estimators: int = 800):
    if not LGBM_AVAILABLE:
        return None
    return lgb.LGBMClassifier(
        objective='multiclass',
        num_class=K,
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        reg_alpha=0.05,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def make_catboost_model(K: int, random_state: int = 42, iterations: int = 800):
    if not CATBOOST_AVAILABLE:
        return None
    return CatBoostClassifier(
        loss_function='MultiClass',
        iterations=iterations,
        depth=5,
        learning_rate=0.03,
        l2_leaf_reg=5.0,
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        auto_class_weights='SqrtBalanced',
    )


def _fit_with_optional_weights(model, X, y, sample_weight=None):
    """Fit sklearn/boosting models safely with optional sample weights."""
    try:
        if sample_weight is not None:
            return model.fit(X, y, sample_weight=sample_weight)
        return model.fit(X, y)
    except TypeError:
        return model.fit(X, y)


def _branch_score_row(name, y_true, y_pred, y_prob, K):
    from evaluation import compute_ece, compute_brier, compute_nll
    return {
        'model': name,
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'ece': float(compute_ece(y_true, y_prob)),
        'brier': float(compute_brier(y_true, y_prob, K)),
        'nll': float(compute_nll(y_true, y_prob)),
    }


def _train_generic_tree_baseline(data: dict, output_dir: str, name: str, model, random_state: int = 42) -> dict:
    print(f"\n[{name}] Training {name} baseline...")
    pred_dir = Path(output_dir) / 'predictions'
    pred_dir.mkdir(parents=True, exist_ok=True)
    K = data['K']
    X_train, y_train = data['X_train'], data['y_fine_train']
    X_test, y_test = data['X_test'], data['y_fine_test']
    sw = _class_sample_weights(y_train)
    _fit_with_optional_weights(model, X_train, y_train, sw)
    if hasattr(model, 'predict_proba'):
        y_prob = _align_proba(model.predict_proba(X_test), getattr(model, 'classes_', np.arange(K)), K)
        y_pred = y_prob.argmax(axis=1)
    else:
        y_pred = model.predict(X_test).astype(int)
        y_prob = np.zeros((len(y_pred), K), dtype=np.float32)
        y_prob[np.arange(len(y_pred)), y_pred] = 1.0
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"  [{name}] Test Macro F1 = {macro_f1:.4f}")
    safe_name = name.lower().replace('-', '_')
    np.save(pred_dir / f'predictions_{safe_name}.npy', y_pred)
    np.save(pred_dir / f'probabilities_{safe_name}.npy', y_prob)
    if hasattr(model, 'feature_importances_'):
        with open(pred_dir / f'{safe_name}_feature_importance.json', 'w', encoding='utf-8') as f:
            json.dump(dict(zip(FEATURE_NAMES, model.feature_importances_.tolist())), f, indent=2)
    return {
        'exp_name': safe_name,
        'test_macro_f1': float(macro_f1),
        'preds': y_pred.astype(np.int64),
        'probs': y_prob.astype(np.float32),
        'labels': y_test,
        'model': model,
    }


def train_extra_trees(data: dict, output_dir: str, random_state: int = 42, n_estimators: int = 800) -> dict:
    return _train_generic_tree_baseline(data, output_dir, 'extra_trees', make_extra_trees_model(random_state, n_estimators), random_state)


def train_hist_gradient_boosting(data: dict, output_dir: str, random_state: int = 42, max_iter: int = 350) -> dict:
    return _train_generic_tree_baseline(data, output_dir, 'hist_gradient_boosting', make_hgb_model(random_state, max_iter), random_state)


def train_lightgbm(data: dict, output_dir: str, random_state: int = 42, n_estimators: int = 800) -> dict:
    if not LGBM_AVAILABLE:
        print("\n[LightGBM] Skipped — LightGBM not installed. Run: pip install lightgbm")
        y_test = data['y_fine_test']; K = data['K']
        dummy = np.zeros(len(y_test), dtype=int)
        dummy_prob = np.ones((len(y_test), K), dtype=np.float32) / K
        return {'exp_name': 'lightgbm', 'test_macro_f1': 0.0, 'preds': dummy, 'probs': dummy_prob, 'labels': y_test, 'skipped': True}
    return _train_generic_tree_baseline(data, output_dir, 'lightgbm', make_lgbm_model(data['K'], random_state, n_estimators), random_state)


def train_catboost(data: dict, output_dir: str, random_state: int = 42, iterations: int = 800) -> dict:
    if not CATBOOST_AVAILABLE:
        print("\n[CatBoost] Skipped — CatBoost not installed. Run: pip install catboost")
        y_test = data['y_fine_test']; K = data['K']
        dummy = np.zeros(len(y_test), dtype=int)
        dummy_prob = np.ones((len(y_test), K), dtype=np.float32) / K
        return {'exp_name': 'catboost', 'test_macro_f1': 0.0, 'preds': dummy, 'probs': dummy_prob, 'labels': y_test, 'skipped': True}
    return _train_generic_tree_baseline(data, output_dir, 'catboost', make_catboost_model(data['K'], random_state, iterations), random_state)


def _make_plus_branch_factories(K: int, random_state: int, n_estimators: int, xgb_max_depth: int, fast: bool = False) -> dict:
    """Factories for OOF tree branches. Optional branches are included only when installed."""
    n_tree = 120 if fast else n_estimators
    n_boost = 120 if fast else max(n_estimators, 800)
    factories = {
        'rf': lambda seed: make_rf_model(seed, n_estimators=n_tree),
        'xgb': lambda seed: make_xgb_model(K, seed, n_estimators=n_tree, max_depth=xgb_max_depth),
        'extra_trees': lambda seed: make_extra_trees_model(seed, n_estimators=n_tree),
        'hist_gradient_boosting': lambda seed: make_hgb_model(seed, max_iter=(120 if fast else 350)),
    }
    if LGBM_AVAILABLE:
        factories['lightgbm'] = lambda seed: make_lgbm_model(K, seed, n_estimators=n_boost)
    if CATBOOST_AVAILABLE:
        factories['catboost'] = lambda seed: make_catboost_model(K, seed, iterations=n_boost)
    return factories


def generate_oof_tree_probabilities_plus(data: dict,
                                         output_dir: str,
                                         random_state: int = 42,
                                         n_splits: int = 5,
                                         n_estimators: int = 800,
                                         xgb_max_depth: int = 5,
                                         fast: bool = False,
                                         include_branches: list = None) -> dict:
    """
    Generate leakage-safe OOF probabilities for the stronger EcoTreeFuseNet-Plus branches.

    Branches: RF, XGBoost, ExtraTrees, HistGradientBoosting, plus LightGBM/CatBoost if installed.
    Final validation/test probabilities are generated from models trained only on D_train.
    """
    factories = _make_plus_branch_factories(data['K'], random_state, n_estimators, xgb_max_depth, fast=fast)
    if include_branches is not None:
        factories = {k: v for k, v in factories.items() if k in include_branches}
    print(f"\n[Stacking-Plus] OOF tree branches ({n_splits} folds): {', '.join(factories.keys())}")

    out_dir = Path(output_dir)
    pred_dir = out_dir / 'predictions'
    stack_dir = out_dir / 'stacking_plus'
    pred_dir.mkdir(parents=True, exist_ok=True)
    stack_dir.mkdir(parents=True, exist_ok=True)

    # Phase 2: tree OOF branches consume the CHM-transformed *unscaled* arrays
    # when available. This removes any dependence on the outer-training scaler
    # during fold-specific probability generation (trees do not require scaling).
    X_train = data.get('X_train_raw', data['X_train'])
    y_train = data['y_fine_train']
    X_val = data.get('X_val_raw', data['X_val'])
    X_test = data.get('X_test_raw', data['X_test'])
    K = data['K']
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    branch_outputs = {}
    diagnostics = {'n_splits': int(n_splits), 'branches': {}, 'skipped': []}

    for branch_name, factory in factories.items():
        print(f"  [Stacking-Plus] Branch: {branch_name}")
        oof = np.zeros((len(y_train), K), dtype=np.float32)
        fold_scores = []
        ok = True
        for fold, (tr, ho) in enumerate(skf.split(X_train, y_train), start=1):
            model = factory(random_state + 100 * fold + len(branch_name))
            if model is None:
                ok = False
                break
            try:
                _fit_with_optional_weights(model, X_train[tr], y_train[tr], _class_sample_weights(y_train[tr]))
                p = _align_proba(model.predict_proba(X_train[ho]), getattr(model, 'classes_', np.arange(K)), K)
                oof[ho] = p
                fold_scores.append(float(f1_score(y_train[ho], p.argmax(axis=1), average='macro', zero_division=0)))
            except Exception as exc:
                ok = False
                diagnostics['skipped'].append({'branch': branch_name, 'fold': fold, 'reason': repr(exc)})
                print(f"    [Warning] Skipping branch {branch_name}: {exc}")
                break
        if not ok:
            continue

        final_model = factory(random_state + len(branch_name))
        _fit_with_optional_weights(final_model, X_train, y_train, _class_sample_weights(y_train))
        val_p = _align_proba(final_model.predict_proba(X_val), getattr(final_model, 'classes_', np.arange(K)), K)
        test_p = _align_proba(final_model.predict_proba(X_test), getattr(final_model, 'classes_', np.arange(K)), K)
        branch_outputs[branch_name] = {
            'oof_train': oof,
            'val': val_p,
            'test': test_p,
            'model': final_model,
        }
        diagnostics['branches'][branch_name] = {
            'oof_macro_f1_mean': float(np.mean(fold_scores)),
            'oof_macro_f1_std': float(np.std(fold_scores)),
            'fold_macro_f1': fold_scores,
        }
        np.save(pred_dir / f'{branch_name}_oof_train_probs.npy', oof)
        np.save(pred_dir / f'{branch_name}_val_probs.npy', val_p)
        np.save(pred_dir / f'{branch_name}_test_probs.npy', test_p)
        print(f"    OOF macro-F1 mean={np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
        if hasattr(final_model, 'feature_importances_'):
            try:
                with open(pred_dir / f'{branch_name}_feature_importance.json', 'w', encoding='utf-8') as f:
                    json.dump(dict(zip(FEATURE_NAMES, final_model.feature_importances_.tolist())), f, indent=2)
            except Exception:
                pass

    with open(stack_dir / 'tree_plus_oof_diagnostics.json', 'w', encoding='utf-8') as f:
        json.dump(diagnostics, f, indent=2, default=str)

    return {
        'branches': branch_outputs,
        'diagnostics': diagnostics,
    }
