"""
EcoTreeFuseNet-Plus — PyCharm-ready full pipeline.

This version keeps the previous EcoTreeFuseNet experiments and adds the
professional upgrade package requested after the full all-model v1 result:

  1. Stronger tree branches: RF, XGBoost, ExtraTrees, HistGradientBoosting,
     LightGBM and CatBoost when installed.
  2. EcoTreeFuseNet-Plus hybrid stacking with validation-selected meta-model.
  3. Validation-weighted soft voting.
  4. Post-hoc temperature calibration.
  5. Validation-only class-bias tuning for Macro F1.
  6. Hierarchy-aware coarse-to-fine correction.
  7. Multi-seed-ready outputs and script support.

Important: D_test is used only once for final auditing. All tuning/calibration
is selected on D_val or OOF D_train features only.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from spatial_data import load_and_prepare_data, make_dataloaders
from network import (
    EcoFuseNet,
    EcoFuseNetV2,
    NoGateMLP,
    FlatLossEcoFuseNet,
    LiDAROnlyNet,
    SpectralOnlyNet,
    NoCHMGateNet,
)
from training import train_ecofusenet, train_ecofusenet_v2, train_ablation_variant
from baselines import (
    train_random_forest,
    train_xgboost,
    train_extra_trees,
    train_hist_gradient_boosting,
    train_lightgbm,
    train_catboost,
)
from stacking import train_ecotreefusenet
from fusion import train_ecotreefusenet_plus
from evaluation import (
    mc_dropout_uncertainty,
    uncertainty_hypothesis_test,
    gate_interpretability,
    prototype_quality_metrics,
    run_full_statistical_analysis,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "vegetation_reference.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_SEED = 42

DEFAULT_MODELS = [
    "rf",
    "xgboost",
    "extra_trees",
    "hist_gradient_boosting",
    "lightgbm",
    "catboost",
    "ecofusenet",
    "ecotreefusenet",
    "ecotreefusenet_plus",
    "ecofusenet_proto",
    "nogate_mlp",
    "flat_loss",
    "lidar_only",
    "spectral_only",
    "no_chm_gate",
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="EcoTreeFuseNet-Plus full pipeline")
    p.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=DEFAULT_MODELS,
        help=(
            "Models to run. If ecotreefusenet/ecotreefusenet_plus is selected, "
            "EcoFuseNet-V2 is trained automatically as the neural branch."
        ),
    )
    p.add_argument("--fast", action="store_true", help="Fast smoke test; not for manuscript reporting.")
    p.add_argument("--skip-figures", action="store_true", help="Skip manuscript figure generation.")
    p.add_argument("--stack-folds", type=int, default=None, help="OOF folds. Default: 2 in fast mode, 5 otherwise.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducible split/training.")
    p.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH), help="Path to vegetation_reference.csv.")
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    p.add_argument("--split-mode", choices=["mixed", "spatial_block"], default="mixed",
                   help="Phase-2 split: historical mixed within-block split or strict complete-block holdout.")
    p.add_argument("--spatial-fold", type=int, default=0,
                   help="Strict spatial fold index when --split-mode spatial_block is used.")
    p.add_argument("--n-spatial-folds", type=int, default=5,
                   help="Number of strict spatial folds (default 5).")
    p.add_argument("--spatial-grid", type=int, default=5,
                   help="Quantile grid dimension used to define atomic geographic blocks (default 5x5).")
    return p.parse_args()


def ensure_output_dirs(output_dir: Path) -> None:
    for sub in ["predictions", "stats", "figures", "checkpoints", "stacking", "stacking_plus", "ecotreefusenet_plus"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)


def _save_raw_chm_for_test(data_path: Path, data: dict):
    raw_df = np.genfromtxt(data_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return raw_df["chm"][data["test_idx"]]


def main():
    args = parse_args()
    seed = int(args.seed)
    set_seed(seed)

    data_path = Path(args.data_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_output_dirs(output_dir)

    # Hybrid variants require EcoFuseNet-V2 branch features.
    if ("ecotreefusenet" in args.models or "ecotreefusenet_plus" in args.models) and "ecofusenet" not in args.models:
        args.models = ["ecofusenet"] + args.models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stack_folds = args.stack_folds if args.stack_folds is not None else (2 if args.fast else 5)

    legacy_hparams = {
        "batch_size": 64,
        "warmup_epochs": 3 if args.fast else 25,
        "max_epochs": 8 if args.fast else 150,
        "patience": 3 if args.fast else 20,
        "lr_warmup": 1e-3,
        "lr_proto": 5e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "lambda1": 0.3,
        "lambda2": 0.1,
        "tau_init": 1.0,
        "sep_margin": 2.0,
    }
    v2_hparams = {
        "warmup_epochs": 5 if args.fast else 35,
        "max_epochs": 15 if args.fast else 220,
        "patience": 5 if args.fast else 35,
        "lr_warmup": 1e-3,
        "lr_main": 8e-4,
        "weight_decay": 5e-4,
        "grad_clip": 1.0,
        "proto_weight": 0.03,
        "coarse_weight": 0.01,
        "sep_weight": 0.001,
        "sep_margin": 1.0,
        "label_smoothing": 0.03,
    }
    baseline_trees = 80 if args.fast else 500
    plus_trees = 120 if args.fast else 800
    xgb_max_depth = 4 if args.fast else 5

    t_start = time.time()
    print("=" * 78)
    print("EcoTreeFuseNet-Plus — PyCharm-ready hybrid tree-neural full pipeline")
    print(f"Base folder : {BASE_DIR}")
    print(f"Data file   : {data_path}")
    print(f"Output dir  : {output_dir}")
    print(f"Device      : {device}")
    print(f"Seed        : {seed}")
    print(f"Fast mode   : {args.fast}")
    print(f"OOF folds   : {stack_folds}")
    print(f"Split mode  : {args.split_mode}")
    if args.split_mode == "spatial_block":
        print(f"Spatial fold: {args.spatial_fold}/{args.n_spatial_folds - 1} (grid={args.spatial_grid}x{args.spatial_grid})")
    print(f"Models      : {', '.join(args.models)}")
    print("=" * 78)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = load_and_prepare_data(
        csv_path=str(data_path),
        output_dir=str(output_dir),
        random_state=seed,
        apply_chm_log1p=True,
        split_mode=args.split_mode,
        spatial_fold=args.spatial_fold,
        n_spatial_folds=args.n_spatial_folds,
        n_grid=args.spatial_grid,
    )
    loaders = make_dataloaders(data, batch_size=legacy_hparams["batch_size"])

    K = data["K"]
    C = data["C"]
    phi = data["phi"]
    class_weights = data["class_weights"]
    class_names = data["fine_enc"].classes_.tolist()
    np.save(output_dir / "predictions" / "y_true.npy", data["y_fine_test"])

    results = {}
    eco_model = None
    eco_res = None

    if "rf" in args.models:
        results["rf"] = train_random_forest(data, str(output_dir), random_state=seed, n_estimators=baseline_trees)

    if "xgboost" in args.models:
        results["xgboost"] = train_xgboost(
            data, str(output_dir), random_state=seed,
            n_estimators=baseline_trees, max_depth=xgb_max_depth,
        )

    if "extra_trees" in args.models:
        results["extra_trees"] = train_extra_trees(data, str(output_dir), random_state=seed, n_estimators=plus_trees)

    if "hist_gradient_boosting" in args.models:
        results["hist_gradient_boosting"] = train_hist_gradient_boosting(
            data, str(output_dir), random_state=seed, max_iter=(120 if args.fast else 350)
        )

    if "lightgbm" in args.models:
        results["lightgbm"] = train_lightgbm(data, str(output_dir), random_state=seed, n_estimators=plus_trees)

    if "catboost" in args.models:
        results["catboost"] = train_catboost(data, str(output_dir), random_state=seed, iterations=plus_trees)

    if "ecofusenet" in args.models:
        print("\n[Exp] EcoFuseNet-V2 neural branch (CE-dominant + weak prototype regularization)")
        eco_model = EcoFuseNetV2(d_f=64, d_e=128, K=K, C=C, tau_init=1.0, dropout=0.10, alpha_chm_init=0.05).to(device)
        eco_res = train_ecofusenet_v2(
            eco_model, loaders, data, phi, class_weights, device, str(output_dir),
            exp_name="ecofusenet", **v2_hparams,
        )
        results["ecofusenet"] = eco_res

    if "ecotreefusenet" in args.models:
        if eco_model is None:
            raise RuntimeError("EcoTreeFuseNet requires the EcoFuseNet-V2 branch; include 'ecofusenet'.")
        results["ecotreefusenet"] = train_ecotreefusenet(
            data=data,
            output_dir=str(output_dir),
            device=device,
            final_ecofuse_model=eco_model,
            phi=phi,
            random_state=seed,
            n_splits=stack_folds,
            tree_estimators=baseline_trees,
            xgb_max_depth=xgb_max_depth,
            fast=args.fast,
        )

    if "ecotreefusenet_plus" in args.models:
        if eco_model is None:
            raise RuntimeError("EcoTreeFuseNet-Plus requires EcoFuseNet-V2 branch; include 'ecofusenet'.")
        plus_results = train_ecotreefusenet_plus(
            data=data,
            output_dir=str(output_dir),
            device=device,
            final_ecofuse_model=eco_model,
            phi=phi,
            random_state=seed,
            n_splits=stack_folds,
            tree_estimators=plus_trees,
            xgb_max_depth=xgb_max_depth,
            fast=args.fast,
        )
        results.update(plus_results)

    if "ecofusenet_proto" in args.models:
        print("\n[Exp] Legacy prototype-only EcoFuseNet")
        proto_model = EcoFuseNet(d_f=32, d_e=64, K=K, C=C, tau_init=1.0).to(device)
        results["ecofusenet_proto"] = train_ecofusenet(
            proto_model, loaders, data, phi, class_weights, device, str(output_dir),
            exp_name="ecofusenet_proto", use_proto_loss=True, **legacy_hparams,
        )

    if "nogate_mlp" in args.models:
        print("\n[Exp] No-Gate MLP")
        model = NoGateMLP(d_f=32, d_e=64, K=K, C=C, tau_init=1.0).to(device)
        results["nogate_mlp"] = train_ablation_variant(
            model, loaders, data, phi, class_weights, device, str(output_dir),
            exp_name="nogate_mlp", use_proto_loss=True,
            warmup_epochs=legacy_hparams["warmup_epochs"], max_epochs=legacy_hparams["max_epochs"],
            patience=legacy_hparams["patience"], lambda1=legacy_hparams["lambda1"],
            lambda2=legacy_hparams["lambda2"], sep_margin=legacy_hparams["sep_margin"],
        )

    if "flat_loss" in args.models:
        print("\n[Exp] Flat-Loss CE Ablation")
        model = FlatLossEcoFuseNet(d_f=32, d_e=64, K=K, C=C, tau_init=1.0).to(device)
        results["flat_loss"] = train_ecofusenet(
            model, loaders, data, phi, class_weights, device, str(output_dir),
            exp_name="flat_loss", use_proto_loss=False, **legacy_hparams,
        )

    if "lidar_only" in args.models:
        print("\n[Exp] LiDAR-Only")
        model = LiDAROnlyNet(d_f=32, d_e=64, K=K, C=C, tau_init=1.0).to(device)
        results["lidar_only"] = train_ablation_variant(
            model, loaders, data, phi, class_weights, device, str(output_dir),
            exp_name="lidar_only", use_proto_loss=True,
            warmup_epochs=legacy_hparams["warmup_epochs"], max_epochs=legacy_hparams["max_epochs"],
            patience=legacy_hparams["patience"], lambda1=legacy_hparams["lambda1"],
            lambda2=legacy_hparams["lambda2"], sep_margin=legacy_hparams["sep_margin"],
        )

    if "spectral_only" in args.models:
        print("\n[Exp] Spectral-Only")
        model = SpectralOnlyNet(d_f=32, d_e=64, K=K, C=C, tau_init=1.0).to(device)
        results["spectral_only"] = train_ablation_variant(
            model, loaders, data, phi, class_weights, device, str(output_dir),
            exp_name="spectral_only", use_proto_loss=True,
            warmup_epochs=legacy_hparams["warmup_epochs"], max_epochs=legacy_hparams["max_epochs"],
            patience=legacy_hparams["patience"], lambda1=legacy_hparams["lambda1"],
            lambda2=legacy_hparams["lambda2"], sep_margin=legacy_hparams["sep_margin"],
        )

    if "no_chm_gate" in args.models:
        print("\n[Exp] No-CHM-Gate")
        model = NoCHMGateNet(d_f=32, d_e=64, K=K, C=C, tau_init=1.0).to(device)
        results["no_chm_gate"] = train_ablation_variant(
            model, loaders, data, phi, class_weights, device, str(output_dir),
            exp_name="no_chm_gate", use_proto_loss=True,
            warmup_epochs=legacy_hparams["warmup_epochs"], max_epochs=legacy_hparams["max_epochs"],
            patience=legacy_hparams["patience"], lambda1=legacy_hparams["lambda1"],
            lambda2=legacy_hparams["lambda2"], sep_margin=legacy_hparams["sep_margin"],
        )

    # Uncertainty and gate/prototype interpretation are based on EcoFuseNet-V2 branch.
    unc_results = None
    gate_analysis = None
    unc_test = {}
    if eco_model is not None and eco_res is not None:
        print("\n[Analysis] MC dropout uncertainty + gate interpretation")
        T = 10 if args.fast else 50
        unc_results = mc_dropout_uncertainty(eco_model, data["X_test"], T=T, device=device)
        y_true = data["y_fine_test"]
        y_pred = eco_res["preds"]
        unc_test = uncertainty_hypothesis_test(unc_results["entropy"], y_true, y_pred)
        np.save(output_dir / "predictions" / "mc_entropy_ecofusenet.npy", unc_results["entropy"])
        np.save(output_dir / "predictions" / "d_nearest_ecofusenet.npy", unc_results["d_nearest"])
        np.save(output_dir / "predictions" / "embeddings_ecofusenet_mc.npy", unc_results["embeddings"])

        chm_raw_test = _save_raw_chm_for_test(data_path, data)
        gate_values = np.load(output_dir / "predictions" / "gate_values_ecofusenet.npy")
        gate_analysis = gate_interpretability(
            gate_values=gate_values,
            y_true=y_true,
            y_pred=y_pred,
            chm_raw=chm_raw_test,
            class_names=class_names,
            K=K,
        )
        with open(output_dir / "stats" / "gate_analysis.json", "w", encoding="utf-8") as f:
            json.dump(gate_analysis, f, indent=2)

        P_fine_np = eco_model.P_fine.detach().cpu().numpy()
        proto_q = prototype_quality_metrics(unc_results["embeddings"], y_true, P_fine_np, K, class_names)
        with open(output_dir / "stats" / "prototype_quality.json", "w", encoding="utf-8") as f:
            json.dump(proto_q, f, indent=2)

        try:
            import pandas as pd
            pd.DataFrame([{"class": k, **v} for k, v in gate_analysis.get("per_class", {}).items()]).to_csv(
                output_dir / "stats" / "table_gate_contribution_by_class.csv", index=False)
            pd.DataFrame([{"class": k, **v} for k, v in proto_q.get("per_class", {}).items()]).to_csv(
                output_dir / "stats" / "table_prototype_quality_by_class.csv", index=False)
        except Exception as exc:
            print(f"  [Warning] Gate/prototype table export failed: {exc}")

    print("\n[Analysis] Statistical report")
    stats_report = run_full_statistical_analysis(
        results, data["y_fine_test"], K, class_names, str(output_dir),
        B=100 if args.fast else 1000,
    )
    if unc_results is not None:
        stats_report["uncertainty_test"] = unc_test
        with open(output_dir / "stats" / "statistical_report.json", "w", encoding="utf-8") as f:
            json.dump(stats_report, f, indent=2, default=str)

    print("\n" + "=" * 78)
    print(f"{'Model':<40} {'Macro F1':>10} {'Bal Acc':>10} {'MCC':>10}")
    print("-" * 78)
    for name, m in stats_report["exp_metrics"].items():
        print(f"{name:<40} {m['macro_f1']:>10.4f} {m['balanced_acc']:>10.4f} {m['mcc']:>10.4f}")
    print("=" * 78)

    if not args.skip_figures:
        print("\n[Visualization] Generating figures")
        try:
            from visualization import generate_all_figures
            generate_all_figures(data, results, unc_results, gate_analysis, stats_report, str(output_dir))
        except Exception as exc:
            print(f"[Warning] Figure generation failed: {exc}")

    summary = {
        "runtime_minutes": round((time.time() - t_start) / 60.0, 2),
        "device": str(device),
        "seed": seed,
        "fast_mode": args.fast,
        "stack_folds": stack_folds,
        "models": args.models,
        "split_mode": args.split_mode,
        "spatial_fold": int(args.spatial_fold) if args.split_mode == "spatial_block" else None,
        "n_spatial_folds": int(args.n_spatial_folds),
        "spatial_grid": int(args.spatial_grid),
        "hparams": {"legacy": legacy_hparams, "ecofusenet_v2": v2_hparams},
        "results_summary": {k: {"macro_f1": float(v.get("test_macro_f1", 0.0))} for k, v in results.items()},
        "reference_model": stats_report.get("reference_model"),
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Done] Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
