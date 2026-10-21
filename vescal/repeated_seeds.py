"""
Run EcoTreeFuseNet-Plus over multiple seeds and create an ensemble result.

Recommended after one successful full run:
    python run_5seed_ensemble.py --seeds 42 101 202 303 404 --stack-folds 5

This script calls train.py once per seed, then averages the selected Plus variant
probabilities and saves repeated-seed/ensemble reports under outputs_multiseed_plus.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef

from evaluation import compute_all_metrics

BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description="EcoTreeFuseNet-Plus multi-seed ensemble")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 202, 303, 404])
    p.add_argument("--stack-folds", type=int, default=5)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--variant", type=str, default="hierarchical_ecotreefusenet_plus",
                   choices=["ecotreefusenet_plus", "calibrated_ecotreefusenet_plus", "bias_tuned_ecotreefusenet_plus", "hierarchical_ecotreefusenet_plus"])
    p.add_argument("--skip-existing", action="store_true", help="Do not rerun seed folders that already contain the requested probabilities.")
    return p.parse_args()


def run_seed(seed: int, stack_folds: int, fast: bool, out_dir: Path, variant: str, skip_existing: bool):
    prob_path = out_dir / "predictions" / f"probabilities_{variant}.npy"
    if skip_existing and prob_path.exists():
        print(f"[Seed {seed}] Existing output found; skipping run.")
        return
    cmd = [
        sys.executable, "train.py",
        "--models", "ecofusenet", "ecotreefusenet_plus",
        "--seed", str(seed),
        "--stack-folds", str(stack_folds),
        "--output-dir", str(out_dir),
        "--skip-figures",
    ]
    if fast:
        cmd.append("--fast")
    print("[Run]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(BASE_DIR), check=True)


def main():
    args = parse_args()
    root = BASE_DIR / "outputs_multiseed_plus"
    root.mkdir(parents=True, exist_ok=True)

    per_seed_rows = []
    probs = []
    y_true_ref = None
    for seed in args.seeds:
        out = root / f"seed_{seed}"
        run_seed(seed, args.stack_folds, args.fast, out, args.variant, args.skip_existing)
        pred_dir = out / "predictions"
        p = np.load(pred_dir / f"probabilities_{args.variant}.npy")
        y_true = np.load(pred_dir / "y_true.npy")
        y_pred = p.argmax(axis=1)
        if y_true_ref is None:
            y_true_ref = y_true
        else:
            # Splits can differ by seed; ensemble is safest only when y_true length/order match.
            if len(y_true_ref) != len(y_true) or not np.array_equal(y_true_ref, y_true):
                print(f"[Warning] Seed {seed} has a different split/order. It will be included in repeated-seed summary but not probability ensemble.")
                per_seed_rows.append({
                    "seed": seed,
                    **compute_all_metrics(y_true, y_pred, p, K=p.shape[1]),
                })
                continue
        probs.append(p)
        per_seed_rows.append({
            "seed": seed,
            **compute_all_metrics(y_true, y_pred, p, K=p.shape[1]),
        })

    stats_dir = root / "stats"
    pred_dir = root / "predictions"
    stats_dir.mkdir(exist_ok=True)
    pred_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(per_seed_rows)
    df.to_csv(stats_dir / "table_repeated_seed_results.csv", index=False)

    summary = {}
    for col in ["macro_f1", "balanced_acc", "mcc", "ece", "brier", "nll", "accuracy"]:
        if col in df:
            summary[col] = {"mean": float(df[col].mean()), "std": float(df[col].std(ddof=0))}
    with open(stats_dir / "repeated_seed_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if len(probs) >= 2:
        p_ens = np.mean(np.stack(probs, axis=0), axis=0)
        y_pred_ens = p_ens.argmax(axis=1)
        K = p_ens.shape[1]
        ens_metrics = compute_all_metrics(y_true_ref, y_pred_ens, p_ens, K)
        np.save(pred_dir / f"probabilities_{args.variant}_ensemble.npy", p_ens)
        np.save(pred_dir / f"predictions_{args.variant}_ensemble.npy", y_pred_ens)
        np.save(pred_dir / "y_true.npy", y_true_ref)
        with open(stats_dir / "ensemble_metrics.json", "w", encoding="utf-8") as f:
            json.dump(ens_metrics, f, indent=2)
        print("\n[Ensemble result]")
        print(f"  Variant     : {args.variant}")
        print(f"  Macro F1    : {ens_metrics['macro_f1']:.4f}")
        print(f"  Bal. Acc.   : {ens_metrics['balanced_acc']:.4f}")
        print(f"  MCC         : {ens_metrics['mcc']:.4f}")
        print(f"  ECE         : {ens_metrics['ece']:.4f}")

    print(f"\n[Done] Multi-seed outputs saved to: {root}")


if __name__ == "__main__":
    main()
