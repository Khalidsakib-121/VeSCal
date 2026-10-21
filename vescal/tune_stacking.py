"""
Optional meta-classifier tuner for EcoTreeFuseNet.

Run this AFTER a completed EcoTreeFuseNet run has generated:
  outputs/predictions/hybrid_features_train.npy
  outputs/predictions/hybrid_features_val.npy
  outputs/predictions/hybrid_features_test.npy

It does not retrain RF/XGBoost/EcoFuseNet branches. It only retunes the final
meta-classifier using validation macro-F1 and writes candidate results to:
  outputs/tuning/ecotreefusenet_meta_tuning_results.csv
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier

from data_utils import load_and_prepare_data
from stacking import _safe_predict_proba

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "vegetation_reference.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
SEED = 42


def main():
    pred_dir = OUTPUT_DIR / "predictions"
    tune_dir = OUTPUT_DIR / "tuning"
    tune_dir.mkdir(parents=True, exist_ok=True)

    needed = [pred_dir / "hybrid_features_train.npy", pred_dir / "hybrid_features_val.npy", pred_dir / "hybrid_features_test.npy"]
    for p in needed:
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run: python train.py --models ecofusenet ecotreefusenet --stack-folds 5")

    data = load_and_prepare_data(str(DATA_PATH), output_dir=str(OUTPUT_DIR), random_state=SEED, apply_chm_log1p=True)
    Z_train = np.load(pred_dir / "hybrid_features_train.npy")
    Z_val = np.load(pred_dir / "hybrid_features_val.npy")
    Z_test = np.load(pred_dir / "hybrid_features_test.npy")
    y_train, y_val, y_test = data["y_fine_train"], data["y_fine_val"], data["y_fine_test"]
    K = data["K"]

    scaler = StandardScaler()
    Z_train_s = scaler.fit_transform(Z_train)
    Z_val_s = scaler.transform(Z_val)
    Z_test_s = scaler.transform(Z_test)

    candidates = []
    for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
        candidates.append((f"logistic_C{C}", LogisticRegression(C=C, class_weight="balanced", max_iter=7000, solver="lbfgs", n_jobs=-1, random_state=SEED)))
    candidates.extend([
        ("hgb_lr003_l2", HistGradientBoostingClassifier(learning_rate=0.03, max_iter=350, max_leaf_nodes=15, l2_regularization=0.1, random_state=SEED)),
        ("hgb_lr005_l2", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=250, max_leaf_nodes=15, l2_regularization=0.05, random_state=SEED)),
        ("rf_meta_depth6", RandomForestClassifier(n_estimators=500, max_depth=6, min_samples_leaf=4, class_weight="balanced", max_features="sqrt", random_state=SEED, n_jobs=-1)),
        ("extra_trees_meta", ExtraTreesClassifier(n_estimators=500, max_depth=8, min_samples_leaf=3, class_weight="balanced", max_features="sqrt", random_state=SEED, n_jobs=-1)),
    ])

    rows = []
    best = None
    for name, model in candidates:
        model.fit(Z_train_s, y_train)
        val_pred = model.predict(Z_val_s)
        test_pred = model.predict(Z_test_s)
        val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        test_f1_audit = f1_score(y_test, test_pred, average="macro", zero_division=0)
        row = {"candidate": name, "val_macro_f1": float(val_f1), "test_macro_f1_audit_only": float(test_f1_audit)}
        rows.append(row)
        print(f"{name:<24s} val_macro_F1={val_f1:.4f} | test audit={test_f1_audit:.4f}")
        if best is None or val_f1 > best["val_macro_f1"]:
            best = {"candidate": name, "val_macro_f1": float(val_f1), "model": model}

    df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    df.to_csv(tune_dir / "ecotreefusenet_meta_tuning_results.csv", index=False)
    with open(tune_dir / "best_ecotreefusenet_meta_config.json", "w", encoding="utf-8") as f:
        json.dump({k:v for k,v in best.items() if k != "model"}, f, indent=2)

    # Save predictions from validation-selected best model for audit.
    best_model = best["model"]
    y_pred = best_model.predict(Z_test_s).astype(int)
    y_prob = _safe_predict_proba(best_model, Z_test_s, K)
    np.save(pred_dir / "predictions_ecotreefusenet_tuned_meta.npy", y_pred)
    np.save(pred_dir / "probabilities_ecotreefusenet_tuned_meta.npy", y_prob)
    print("\nBest validation-selected meta model:")
    print(json.dumps({k:v for k,v in best.items() if k != "model"}, indent=2))
    print(f"Saved: {tune_dir / 'ecotreefusenet_meta_tuning_results.csv'}")


if __name__ == "__main__":
    main()
