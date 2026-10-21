"""
Validation-only hyperparameter tuner for EcoFuseNet-V2.

Run after package installation:
    python tune_ecofusenet_v2.py

This script does NOT touch the test set for model selection. It selects by validation
macro-F1 only, then writes outputs/tuning/ecofusenet_v2_tuning_results.csv.

Recommended workflow:
    1) python tune_ecofusenet_v2.py
    2) copy the best config printed at the end into train.py v2_hparams if desired
    3) delete outputs/
    4) python train.py
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_utils import load_and_prepare_data, make_dataloaders
from network import EcoFuseNetV2
from training import train_ecofusenet_v2

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "vegetation_reference.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
SEED = 42


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tune_dir = OUTPUT_DIR / "tuning"
    tune_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_prepare_data(
        csv_path=str(DATA_PATH),
        output_dir=str(OUTPUT_DIR),
        random_state=SEED,
        apply_chm_log1p=True,
    )
    loaders = make_dataloaders(data, batch_size=64)
    K, C = data["K"], data["C"]

    # Compact, defensible validation grid for CPU. This is deliberately not huge.
    configs = [
        {"name": "A_ce_only_d64",    "d_f": 64,  "d_e": 128, "dropout": 0.10, "lr_main": 8e-4, "weight_decay": 5e-4, "proto_weight": 0.00, "coarse_weight": 0.00, "sep_weight": 0.00,  "label_smoothing": 0.03},
        {"name": "B_weak_proto",     "d_f": 64,  "d_e": 128, "dropout": 0.10, "lr_main": 8e-4, "weight_decay": 5e-4, "proto_weight": 0.03, "coarse_weight": 0.01, "sep_weight": 0.001, "label_smoothing": 0.03},
        {"name": "C_very_weak_proto","d_f": 64,  "d_e": 128, "dropout": 0.10, "lr_main": 8e-4, "weight_decay": 5e-4, "proto_weight": 0.01, "coarse_weight": 0.005,"sep_weight": 0.0005,"label_smoothing": 0.03},
        {"name": "D_more_regular",   "d_f": 64,  "d_e": 128, "dropout": 0.20, "lr_main": 5e-4, "weight_decay": 1e-3, "proto_weight": 0.03, "coarse_weight": 0.01, "sep_weight": 0.001, "label_smoothing": 0.05},
        {"name": "E_larger_df",      "d_f": 128, "d_e": 128, "dropout": 0.10, "lr_main": 5e-4, "weight_decay": 5e-4, "proto_weight": 0.01, "coarse_weight": 0.005,"sep_weight": 0.0005,"label_smoothing": 0.03},
        {"name": "F_larger_low_drop", "d_f": 128, "d_e": 128, "dropout": 0.05, "lr_main": 5e-4, "weight_decay": 1e-4, "proto_weight": 0.00, "coarse_weight": 0.00, "sep_weight": 0.00,  "label_smoothing": 0.03},
    ]

    rows = []
    for cfg in configs:
        print("\n" + "=" * 80)
        print(f"[Tune] {cfg['name']}")
        print("=" * 80)
        set_seed(SEED)
        model = EcoFuseNetV2(
            d_f=cfg["d_f"],
            d_e=cfg["d_e"],
            K=K,
            C=C,
            tau_init=1.0,
            dropout=cfg["dropout"],
            alpha_chm_init=0.05,
        ).to(device)

        exp_name = "tune_" + cfg["name"]
        res = train_ecofusenet_v2(
            model=model,
            loaders=loaders,
            data=data,
            phi=data["phi"],
            class_weights=data["class_weights"],
            device=device,
            output_dir=str(OUTPUT_DIR),
            exp_name=exp_name,
            warmup_epochs=30,
            max_epochs=180,
            patience=25,
            lr_warmup=1e-3,
            lr_main=cfg["lr_main"],
            weight_decay=cfg["weight_decay"],
            grad_clip=1.0,
            proto_weight=cfg["proto_weight"],
            coarse_weight=cfg["coarse_weight"],
            sep_weight=cfg["sep_weight"],
            sep_margin=1.0,
            label_smoothing=cfg["label_smoothing"],
        )
        row = {**cfg, "best_val_f1": res["best_val_f1"], "test_macro_f1_for_audit_only": res["test_macro_f1"]}
        rows.append(row)
        pd.DataFrame(rows).to_csv(tune_dir / "ecofusenet_v2_tuning_results.csv", index=False)

    df = pd.DataFrame(rows).sort_values("best_val_f1", ascending=False)
    df.to_csv(tune_dir / "ecofusenet_v2_tuning_results.csv", index=False)
    best = df.iloc[0].to_dict()
    with open(tune_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("\nBest validation config:")
    print(json.dumps(best, indent=2))
    print(f"\nSaved: {tune_dir / 'ecofusenet_v2_tuning_results.csv'}")


if __name__ == "__main__":
    main()
