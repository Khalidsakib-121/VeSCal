"""
Visualization — All 14 Manuscript Figures

Figure 1  : Full workflow diagram
Figure 2  : EcoFuseNet architecture diagram
Figure 3  : Class distribution colored by coarse type
Figure 4  : Feature distribution & correlation heatmap
Figure 5  : Spatial train/val/test distribution map
Figure 6  : Main performance comparison bar chart
Figure 7  : Ablation delta plot
Figure 8  : 29×29 confusion matrix with coarse-type boundaries
Figure 9  : CHM-gate modality contribution per class
Figure 10 : Prototype embedding t-SNE/UMAP
Figure 11 : Feature importance stability plot
Figure 12 : Reliability diagram (calibration) and ECE
Figure 13 : Uncertainty distribution (correct vs. wrong)
Figure 14 : Minority-class F1 improvement plot
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless rendering for PyCharm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance


# ── Style defaults ────────────────────────────────────────────────────────────

COARSE_COLORS = {
    'Community'     : '#2E86AB',
    'Non-vegetated' : '#A8DADC',
    'Herbs'         : '#6BAE75',
    'Shrubs'        : '#E09B3D',
    'Grasses'       : '#C75146',
}

plt.rcParams.update({
    'font.family'    : 'DejaVu Sans',
    'font.size'      : 10,
    'axes.titlesize' : 12,
    'axes.labelsize' : 11,
    'figure.dpi'     : 150,
    'savefig.dpi'    : 300,
    'savefig.bbox'   : 'tight',
})




def choose_reference_model(results_or_metrics: dict) -> str:
    priority = [
        'hierarchical_ecotreefusenet_plus',
        'bias_tuned_ecotreefusenet_plus',
        'calibrated_ecotreefusenet_plus',
        'ecotreefusenet_plus',
        'ecotreefusenet',
        'ecofusenet',
    ]
    for name in priority:
        if name in results_or_metrics:
            return name
    return next(iter(results_or_metrics.keys())) if results_or_metrics else 'ecofusenet'

def save_fig(fig, path: str):
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Fig] Saved: {path}")



# ── Figure 1: Full workflow diagram ──────────────────────────────────────────

def fig1_workflow_diagram(output_dir: str):
    steps = [
        "Task 4 CSV\n1833 samples × 8 predictors",
        "Spatially aware\ntrain/val/test split",
        "Train-only\nstandardization",
        "Hybrid experiments\nRF, XGBoost, EcoFuseNet, EcoTreeFuseNet",
        "Saved predictions\nprobabilities, embeddings, gates",
        "Statistics + figures\nCI, tests, uncertainty, XAI",
    ]
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis("off")
    x_positions = np.linspace(0.06, 0.94, len(steps))
    for i, (x, label) in enumerate(zip(x_positions, steps)):
        box = mpatches.FancyBboxPatch(
            (x - 0.07, 0.42), 0.14, 0.22,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.4, edgecolor="#333333", facecolor="#F4F6F8"
        )
        ax.add_patch(box)
        ax.text(x, 0.53, label, ha="center", va="center", fontsize=9)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x_positions[i+1] - 0.085, 0.53), xytext=(x + 0.085, 0.53),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"))
    ax.set_title("Figure 1: EcoFuseNet End-to-End Experimental Workflow", fontsize=13, pad=12)
    save_fig(fig, str(Path(output_dir) / "fig01_workflow_diagram.png"))


# ── Figure 2: EcoFuseNet architecture diagram ────────────────────────────────

def fig2_architecture_diagram(output_dir: str):
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.axis("off")
    boxes = [
        (0.05, 0.70, "Terrain input\n5 features", "FC→GELU→Dropout\nh_T ∈ R³²"),
        (0.05, 0.43, "CHM input\nlog1p(CHM)", "FC→GELU→FC\nh_C ∈ R³²"),
        (0.05, 0.16, "Spectral input\nNDVI, NDWI", "FC→GELU→Dropout→FC\nh_S ∈ R³²"),
        (0.39, 0.43, "CHM-conditioned gate", "g = sigmoid(W[h_C;h_T;h_S])"),
        (0.63, 0.43, "Gated fusion", "z = g⊙h_T + (1−g)⊙h_S + αh_C"),
        (0.82, 0.43, "Neural branch", "e ∈ R¹²⁸, gates, distances\nfor hybrid meta-features"),
    ]
    for x, y, title, body in boxes:
        rect = mpatches.FancyBboxPatch((x, y), 0.15, 0.16, boxstyle="round,pad=0.02",
                                       linewidth=1.3, edgecolor="#333333", facecolor="#F4F6F8")
        ax.add_patch(rect)
        ax.text(x + 0.075, y + 0.105, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + 0.075, y + 0.045, body, ha="center", va="center", fontsize=8.5)
    # arrows from branches to gate
    for y in [0.78, 0.51, 0.24]:
        ax.annotate("", xy=(0.39, 0.51), xytext=(0.20, y), arrowprops=dict(arrowstyle="->", lw=1.3))
    ax.annotate("", xy=(0.63, 0.51), xytext=(0.54, 0.51), arrowprops=dict(arrowstyle="->", lw=1.3))
    ax.annotate("", xy=(0.82, 0.51), xytext=(0.78, 0.51), arrowprops=dict(arrowstyle="->", lw=1.3))
    ax.text(0.52, 0.18, "Hybrid layer: [8 features + RF/XGB probs + EcoFuseNet probs/embedding/gate/distances] → calibrated meta-classifier",
            ha="center", va="center", fontsize=10)
    ax.set_title("Figure 2: EcoTreeFuseNet Hybrid Tree-Neural Architecture", fontsize=13, pad=12)
    save_fig(fig, str(Path(output_dir) / "fig02_ecofusenet_architecture.png"))


# ── Figure 11: Feature importance summary ────────────────────────────────────

def fig11_feature_importance(predictions_dir: str, output_dir: str):
    feature_names = ['elevation', 'slope', 'curvature', 'tpi', 'solar_insolation', 'chm', 'ndvi', 'ndwi']
    pred_dir = Path(predictions_dir)
    rows = []
    for model_name, file_name in [('Random Forest', 'rf_feature_importance.json'), ('XGBoost', 'xgboost_feature_importance.json')]:
        path = pred_dir / file_name
        if not path.exists():
            continue
        with open(path, 'r', encoding='utf-8') as f:
            vals = json.load(f)
        for feat in feature_names:
            rows.append((model_name, feat, float(vals.get(feat, 0.0))))
    if not rows:
        print("  [Fig11] Skipped: no feature-importance JSON found")
        return
    models = sorted(set(r[0] for r in rows))
    x = np.arange(len(feature_names))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, model_name in enumerate(models):
        vals = [next((v for m, f, v in rows if m == model_name and f == feat), 0.0) for feat in feature_names]
        ax.bar(x + i * width - (len(models)-1)*width/2, vals, width, label=model_name, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=35, ha='right')
    ax.set_ylabel('Importance')
    ax.set_title('Figure 11: Baseline Feature-Importance Summary')
    ax.legend()
    save_fig(fig, str(Path(output_dir) / 'fig11_feature_importance_summary.png'))


# ── Figure 3: Class distribution ─────────────────────────────────────────────

def fig3_class_distribution(class_names: list,
                              class_counts: list,
                              coarse_map: dict,
                              output_dir: str):
    K = len(class_names)
    coarse_types = [coarse_map.get(c, 'Community') for c in class_names]
    colors = [COARSE_COLORS.get(ct, '#999999') for ct in coarse_types]

    # Sort by count descending
    order = np.argsort(class_counts)[::-1]
    sorted_names  = [class_names[i] for i in order]
    sorted_counts = [class_counts[i] for i in order]
    sorted_colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(K), sorted_counts, color=sorted_colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(K))
    ax.set_xticklabels(sorted_names, rotation=75, ha='right', fontsize=8)
    ax.set_ylabel('Sample Count')
    ax.set_title('Figure 3: Fine-Class Distribution (N=1833, K=29, colored by Coarse Type)')

    patches = [mpatches.Patch(color=c, label=t)
               for t, c in COARSE_COLORS.items()]
    ax.legend(handles=patches, loc='upper right', fontsize=9)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, linewidth=1, label='N=50')

    save_fig(fig, str(Path(output_dir) / 'fig03_class_distribution.png'))


# ── Figure 4: Feature distribution & correlation ─────────────────────────────

def fig4_feature_distributions(X_raw: np.ndarray,
                                 feature_names: list,
                                 output_dir: str):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Figure 4: Feature Distributions (raw)', fontsize=13)

    for i, (ax, fname) in enumerate(zip(axes.flatten(), feature_names)):
        ax.hist(X_raw[:, i], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        ax.set_title(fname, fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')

    save_fig(fig, str(Path(output_dir) / 'fig04a_feature_distributions.png'))

    # Correlation heatmap
    import pandas as pd
    df_feat = pd.DataFrame(X_raw, columns=feature_names)
    corr = df_feat.corr()
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax2, square=True, linewidths=0.5)
    ax2.set_title('Figure 4b: Feature Correlation Heatmap')
    save_fig(fig2, str(Path(output_dir) / 'fig04b_correlation_heatmap.png'))


# ── Figure 5: Spatial map ─────────────────────────────────────────────────────

def fig5_spatial_split_map(coords_train: np.ndarray,
                             coords_val: np.ndarray,
                             coords_test: np.ndarray,
                             output_dir: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords_train[:, 0], coords_train[:, 1],
               c='steelblue', s=8, alpha=0.6, label=f'Train (n={len(coords_train)})')
    ax.scatter(coords_val[:, 0],   coords_val[:, 1],
               c='orange',    s=10, alpha=0.8, label=f'Val (n={len(coords_val)})')
    ax.scatter(coords_test[:, 0],  coords_test[:, 1],
               c='red',       s=10, alpha=0.8, label=f'Test (n={len(coords_test)})')
    ax.set_xlabel('X (UTM Easting)')
    ax.set_ylabel('Y (UTM Northing)')
    ax.set_title('Figure 5: Spatial Distribution of Train/Val/Test Splits\n'
                  'East River, CO — NEON AOP 2018')
    ax.legend(markerscale=2, fontsize=10)
    save_fig(fig, str(Path(output_dir) / 'fig05_spatial_split_map.png'))


# ── Figure 6: Performance comparison ─────────────────────────────────────────

def fig6_performance_comparison(metrics_dict: dict,
                                  output_dir: str):
    exp_names = list(metrics_dict.keys())
    macro_f1  = [metrics_dict[n]['macro_f1']   for n in exp_names]
    bal_acc   = [metrics_dict[n]['balanced_acc'] for n in exp_names]

    x = np.arange(len(exp_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, macro_f1, width, label='Macro F1',
                    color='#2E86AB', edgecolor='white')
    bars2 = ax.bar(x + width/2, bal_acc, width, label='Balanced Accuracy',
                    color='#E09B3D', edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.set_title('Figure 6: Performance Comparison — 8 Experiments')
    ax.legend()
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    save_fig(fig, str(Path(output_dir) / 'fig06_performance_comparison.png'))


# ── Figure 7: Ablation delta plot ────────────────────────────────────────────

def fig7_ablation_delta(metrics_dict: dict,
                          output_dir: str):
    reference = choose_reference_model(metrics_dict)
    if reference not in metrics_dict:
        print(f"[Fig7] Skipped — {reference} not in results")
        return

    base_f1 = metrics_dict[reference]['macro_f1']
    ablations = {k: v for k, v in metrics_dict.items() if k != reference}

    names  = list(ablations.keys())
    deltas = [metrics_dict[n]['macro_f1'] - base_f1 for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = ['#C75146' if d < 0 else '#6BAE75' for d in deltas]
    ax.barh(names, deltas, color=colors, edgecolor='white')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel(f'ΔMacro F1 vs. {reference}')
    ax.set_title(f'Figure 7: Ablation/Comparison Delta from {reference}')
    for i, (d, n) in enumerate(zip(deltas, names)):
        ax.text(d + (0.001 if d >= 0 else -0.001), i,
                f'{d:+.4f}', va='center', ha='left' if d >= 0 else 'right',
                fontsize=9)
    save_fig(fig, str(Path(output_dir) / 'fig07_ablation_delta.png'))


# ── Figure 8: Confusion matrix ───────────────────────────────────────────────

def fig8_confusion_matrix(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            class_names: list,
                            coarse_map: dict,
                            output_dir: str):
    K  = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(K)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_norm, annot=False, cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1, linewidths=0.1, linecolor='#eeeeee')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_title('Figure 8: Normalized Confusion Matrix (EcoFuseNet, 29 Classes)')
    ax.tick_params(axis='x', labelsize=7, rotation=75)
    ax.tick_params(axis='y', labelsize=7, rotation=0)
    save_fig(fig, str(Path(output_dir) / 'fig08_confusion_matrix.png'))


# ── Figure 9: Gate contribution per class ────────────────────────────────────

def fig9_gate_contribution(gate_analysis: dict,
                             class_names: list,
                             output_dir: str):
    names = []
    gates = []
    chms  = []
    for c in class_names:
        d = gate_analysis['per_class'].get(c, {})
        if d.get('mean_gate') is not None:
            names.append(c)
            gates.append(d['mean_gate'])
            chms.append(d.get('mean_chm', 0) or 0)

    order  = np.argsort(gates)[::-1]
    names  = [names[i] for i in order]
    gates  = [gates[i] for i in order]

    fig, ax = plt.subplots(figsize=(13, 6))
    colors  = ['#2E86AB' if g > 0.5 else '#C75146' for g in gates]
    ax.bar(range(len(names)), gates, color=colors, edgecolor='white')
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.6, linewidth=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=75, ha='right', fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Mean Gate Value ḡ_k')
    ax.set_title(f'Figure 9: Per-Class Gate Contribution\n'
                  f'(>0.5 = Terrain-dominant, <0.5 = Spectral-dominant)\n'
                  f'Spearman ρ = {gate_analysis["spearman_rho"]:.3f} '
                  f'(p={gate_analysis["spearman_p"]:.3f})')

    patch_t = mpatches.Patch(color='#2E86AB', label='Terrain-dominant (ḡ > 0.5)')
    patch_s = mpatches.Patch(color='#C75146', label='Spectral-dominant (ḡ < 0.5)')
    ax.legend(handles=[patch_t, patch_s], fontsize=9)
    save_fig(fig, str(Path(output_dir) / 'fig09_gate_contribution.png'))


# ── Figure 10: Prototype t-SNE ────────────────────────────────────────────────

def fig10_prototype_tsne(embeddings: np.ndarray,
                          y_true: np.ndarray,
                          prototypes: np.ndarray,
                          class_names: list,
                          coarse_map: dict,
                          output_dir: str,
                          method: str = 'pca'):
    try:
        all_pts = np.vstack([embeddings, prototypes])
        # PCA is used by default because it is deterministic and fast on CPU/PyCharm.
        # Users may later switch to UMAP/t-SNE for supplementary analysis.
        from sklearn.decomposition import PCA
        reduced = PCA(n_components=2, random_state=42).fit_transform(all_pts)
        method = 'pca'
    except Exception as e:
        print(f"  [Fig10] Dimensionality reduction failed: {e}")
        return

    K = len(class_names)
    emb_2d   = reduced[:len(y_true)]
    proto_2d = reduced[len(y_true):]

    coarse_types = list(set(coarse_map.values()))
    coarse_to_idx= {ct: i for i, ct in enumerate(coarse_types)}
    cmap = plt.cm.get_cmap('tab10', len(coarse_types))

    fig, ax = plt.subplots(figsize=(12, 10))

    for k in range(K):
        mask = (y_true == k)
        if mask.sum() == 0:
            continue
        ct    = coarse_map.get(class_names[k], 'Community')
        color = cmap(coarse_to_idx.get(ct, 0))
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                    c=[color], s=8, alpha=0.4)

    for k in range(min(K, len(proto_2d))):
        ct    = coarse_map.get(class_names[k], 'Community')
        color = cmap(coarse_to_idx.get(ct, 0))
        ax.scatter(proto_2d[k, 0], proto_2d[k, 1],
                    c=[color], s=120, marker='*', edgecolors='black',
                    linewidths=0.5, zorder=5)
        ax.annotate(class_names[k], (proto_2d[k, 0], proto_2d[k, 1]),
                     fontsize=5, alpha=0.8,
                     xytext=(3, 3), textcoords='offset points')

    patches = [mpatches.Patch(color=cmap(coarse_to_idx[ct]), label=ct)
               for ct in coarse_types]
    ax.legend(handles=patches, fontsize=9, loc='upper left')
    ax.set_title(f'Figure 10: Prototype Embedding Space ({method.upper()})\n'
                  '★ = Class Prototype | Points colored by Coarse Type')
    ax.set_xlabel(f'{method.upper()} Dim 1')
    ax.set_ylabel(f'{method.upper()} Dim 2')
    save_fig(fig, str(Path(output_dir) / f'fig10_prototype_{method}.png'))


# ── Figure 12: Reliability diagram ───────────────────────────────────────────

def fig12_reliability_diagram(y_true: np.ndarray,
                                y_prob: np.ndarray,
                                ece: float,
                                exp_name: str,
                                output_dir: str,
                                n_bins: int = 10):
    confidences = y_prob.max(axis=1)
    correct     = (y_prob.argmax(axis=1) == y_true).astype(float)
    bin_edges   = np.linspace(0, 1, n_bins + 1)
    bin_acc  = []
    bin_conf = []
    bin_n    = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_acc.append(correct[mask].mean())
        bin_conf.append(confidences[mask].mean())
        bin_n.append(mask.sum())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=1.5)
    ax1.bar(bin_conf, bin_acc, width=0.08, alpha=0.7, color='steelblue',
             edgecolor='white', label='EcoFuseNet')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'Figure 12a: Reliability Diagram\n(ECE = {ece:.4f})')
    ax1.legend()
    ax1.set_xlim(0, 1);  ax1.set_ylim(0, 1)

    # Confidence histogram
    ax2.hist(confidences, bins=30, color='orange', edgecolor='white', alpha=0.8)
    ax2.set_xlabel('Predicted Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Figure 12b: Confidence Distribution ({exp_name})')

    save_fig(fig, str(Path(output_dir) / f'fig12_reliability_{exp_name}.png'))


# ── Figure 13: Uncertainty distribution ──────────────────────────────────────

def fig13_uncertainty_distribution(entropy: np.ndarray,
                                     d_nearest: np.ndarray,
                                     y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     uncertainty_test: dict,
                                     output_dir: str):
    correct_mask = (y_pred == y_true)
    h_correct = entropy[correct_mask]
    h_wrong   = entropy[~correct_mask]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # MC entropy
    ax1.hist(h_correct, bins=30, alpha=0.6, color='steelblue',
              density=True, label=f'Correct (n={len(h_correct)})')
    ax1.hist(h_wrong,   bins=30, alpha=0.6, color='red',
              density=True, label=f'Wrong (n={len(h_wrong)})')
    p = uncertainty_test.get('p_value', 1.0)
    d = uncertainty_test.get('cohens_d', 0.0)
    ax1.set_xlabel('MC Dropout Entropy (nats)')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Figure 13a: MC Entropy (Correct vs. Wrong)\n'
                   f'Mann-Whitney p={p:.4f}, Cohen\'s d={d:.3f}')
    ax1.legend()

    # Prototype distance
    ax2.hist(d_nearest[correct_mask], bins=30, alpha=0.6, color='steelblue',
              density=True, label='Correct')
    ax2.hist(d_nearest[~correct_mask], bins=30, alpha=0.6, color='red',
              density=True, label='Wrong')
    ax2.set_xlabel('Nearest Prototype Distance d_nearest')
    ax2.set_ylabel('Density')
    ax2.set_title('Figure 13b: Prototype Distance Uncertainty')
    ax2.legend()

    save_fig(fig, str(Path(output_dir) / 'fig13_uncertainty_distribution.png'))


# ── Figure 14: Minority-class F1 improvement ─────────────────────────────────

def fig14_minority_class_f1(minority_analysis: dict,
                              minority_classes: list,
                              output_dir: str):
    models = list(minority_analysis.keys())
    n_models = len(models)
    x = np.arange(len(minority_classes))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(13, 6))
    cmap = plt.cm.get_cmap('tab10', n_models)

    for i, model in enumerate(models):
        vals = [minority_analysis[model].get(mc, 0.0) for mc in minority_classes]
        ax.bar(x + i * width - (n_models - 1) * width / 2,
               vals, width, label=model, color=cmap(i), edgecolor='white', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(minority_classes, rotation=20, ha='right')
    ax.set_ylabel('F1-Score')
    ax.set_ylim(0, 1.05)
    ax.set_title('Figure 14: Minority-Class F1 Comparison\n(5 smallest classes)')
    ax.legend(fontsize=8, loc='upper right')
    save_fig(fig, str(Path(output_dir) / 'fig14_minority_class_f1.png'))


# ── Generate all figures from saved results ───────────────────────────────────

def generate_all_figures(data: dict,
                           results: dict,
                           unc_results: dict,
                           gate_analysis: dict,
                           stats_report: dict,
                           output_dir: str):
    """
    Convenience function to generate all 14 figures at once.
    Called from train.py after all experiments are complete.
    """
    fig_dir = Path(output_dir) / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    class_names = data['fine_enc'].classes_.tolist()
    coarse_map  = data['coarse_map']
    K = data['K']

    import pandas as pd
    # Raw feature data for fig 4
    df_raw = pd.read_csv(
        str(Path(output_dir).parent / 'data' / 'vegetation_reference.csv'))
    feature_names = ['elevation', 'slope', 'curvature', 'tpi',
                     'solar_insolation', 'chm', 'ndvi', 'ndwi']
    X_raw = df_raw[feature_names].values

    # Train class counts for fig 3
    y_ft = data['y_fine_train']
    class_counts_train = [int((y_ft == k).sum()) for k in range(K)]

    print("\n[Figures] Generating all 14 manuscript figures...")

    # Figure 1
    try:
        fig1_workflow_diagram(str(fig_dir))
    except Exception as e:
        print(f"  [Fig1] Error: {e}")

    # Figure 2
    try:
        fig2_architecture_diagram(str(fig_dir))
    except Exception as e:
        print(f"  [Fig2] Error: {e}")

    # Figure 3
    try:
        fig3_class_distribution(class_names, class_counts_train,
                                  coarse_map, str(fig_dir))
    except Exception as e:
        print(f"  [Fig3] Error: {e}")

    # Figure 4
    try:
        fig4_feature_distributions(X_raw, feature_names, str(fig_dir))
    except Exception as e:
        print(f"  [Fig4] Error: {e}")

    # Figure 5
    try:
        coords      = df_raw[['x_original', 'y_original']].values
        tr_idx = data['train_idx']
        va_idx = data['val_idx']
        te_idx = data['test_idx']
        fig5_spatial_split_map(
            coords[tr_idx], coords[va_idx], coords[te_idx], str(fig_dir))
    except Exception as e:
        print(f"  [Fig5] Error: {e}")

    # Figure 6
    try:
        exp_metrics = stats_report.get('exp_metrics', {})
        if exp_metrics:
            fig6_performance_comparison(exp_metrics, str(fig_dir))
    except Exception as e:
        print(f"  [Fig6] Error: {e}")

    # Figure 7
    try:
        fig7_ablation_delta(stats_report.get('exp_metrics', {}), str(fig_dir))
    except Exception as e:
        print(f"  [Fig7] Error: {e}")

    # Figure 8
    try:
        reference = choose_reference_model(results)
        if reference in results:
            y_true = results[reference]['labels']
            y_pred = results[reference]['preds']
            fig8_confusion_matrix(y_true, y_pred, class_names,
                                   coarse_map, str(fig_dir))
    except Exception as e:
        print(f"  [Fig8] Error: {e}")

    # Figure 9
    try:
        if gate_analysis:
            fig9_gate_contribution(gate_analysis, class_names, str(fig_dir))
    except Exception as e:
        print(f"  [Fig9] Error: {e}")

    # Figure 10
    try:
        pred_dir = Path(output_dir) / 'predictions'
        emb_path = pred_dir / 'embeddings_ecofusenet.npy'
        proto_path = pred_dir / 'prototypes_fine_ecofusenet.npy'
        if 'ecofusenet' in results and emb_path.exists() and proto_path.exists():
            emb = np.load(emb_path)
            proto = np.load(proto_path)
            y_t = results['ecofusenet']['labels']
            fig10_prototype_tsne(emb, y_t, proto, class_names, coarse_map, str(fig_dir), method='pca')
        else:
            print("  [Fig10] Skipped: embeddings/prototypes not available")
    except Exception as e:
        print(f"  [Fig10] Skipped: {e}")

    # Figure 11
    try:
        fig11_feature_importance(str(Path(output_dir) / 'predictions'), str(fig_dir))
    except Exception as e:
        print(f"  [Fig11] Error: {e}")

    # Figure 12
    try:
        reference = choose_reference_model(results)
        if reference in results:
            m = stats_report['exp_metrics'].get(reference, {})
            y_true = results[reference]['labels']
            y_prob = results[reference]['probs']
            fig12_reliability_diagram(y_true, y_prob,
                                       m.get('ece', 0.0),
                                       reference, str(fig_dir))
    except Exception as e:
        print(f"  [Fig12] Error: {e}")

    # Figure 13
    try:
        if unc_results and 'ecofusenet' in results:
            y_true = results['ecofusenet']['labels']
            y_pred = results['ecofusenet']['preds']
            unc_test = stats_report.get('uncertainty_test', {})
            fig13_uncertainty_distribution(
                unc_results['entropy'], unc_results['d_nearest'],
                y_true, y_pred, unc_test, str(fig_dir))
    except Exception as e:
        print(f"  [Fig13] Error: {e}")

    # Figure 14
    try:
        minority_classes = ['BareSoil', 'SandstoneTalus', 'Frasera',
                             'Snow', 'SandstoneOutcrop']
        ma = stats_report.get('minority_analysis', {})
        if ma:
            fig14_minority_class_f1(ma, minority_classes, str(fig_dir))
    except Exception as e:
        print(f"  [Fig14] Error: {e}")

    print(f"[Figures] All figures saved to {fig_dir}")