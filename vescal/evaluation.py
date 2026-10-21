"""
Evaluation, Statistical Analysis & Uncertainty — Phases 11–13

Implements all metrics, tests, and analyses from the proposal and math doc:
  - Macro F1, Balanced Accuracy, MCC, Cohen's Kappa
  - ECE, Brier Score, NLL (calibration)
  - Bootstrap 95% CIs (B=1000)
  - Paired bootstrap difference tests
  - McNemar test for error comparison
  - Cohen's d effect size
  - Prototype quality metrics (intra/inter/compactness)
  - Monte Carlo Dropout uncertainty (T=50)
  - Prototype distance uncertainty
  - Per-class gate interpretability (Eq. 35–36)
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, matthews_corrcoef,
    cohen_kappa_score, confusion_matrix, classification_report
)
from pathlib import Path
import json
import warnings


# ── Core metrics ──────────────────────────────────────────────────────────────

def compute_all_metrics(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         y_prob: np.ndarray,
                         K: int,
                         n_bins: int = 10) -> dict:
    """
    Compute all evaluation metrics from predictions and probabilities.

    Returns:
        dict with macro_f1, balanced_acc, mcc, kappa,
              ece, brier, nll, per_class_f1
    """
    macro_f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1= f1_score(y_true, y_pred, average='weighted', zero_division=0)
    bal_acc    = balanced_accuracy_score(y_true, y_pred)
    mcc        = matthews_corrcoef(y_true, y_pred)
    kappa      = cohen_kappa_score(y_true, y_pred)
    per_cls_f1 = f1_score(y_true, y_pred, average=None,
                           zero_division=0, labels=list(range(K)))
    accuracy   = (y_true == y_pred).mean()

    # Calibration
    ece   = compute_ece(y_true, y_prob, n_bins=n_bins)
    brier = compute_brier(y_true, y_prob, K)
    nll   = compute_nll(y_true, y_prob)

    return {
        'macro_f1'    : float(macro_f1),
        'weighted_f1' : float(weighted_f1),
        'balanced_acc': float(bal_acc),
        'mcc'         : float(mcc),
        'kappa'       : float(kappa),
        'accuracy'    : float(accuracy),
        'ece'         : float(ece),
        'brier'       : float(brier),
        'nll'         : float(nll),
        'per_class_f1': per_cls_f1.tolist(),
    }


def compute_ece(y_true: np.ndarray,
                 y_prob: np.ndarray,
                 n_bins: int = 10) -> float:
    """Expected Calibration Error with an inclusive final bin.

    Phase 3 correction: confidence exactly equal to 1.0 is included in the
    final interval [0.9, 1.0]. Earlier code used a right-open final interval
    and silently excluded confidence==1.0 observations from ECE.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    confidences = y_prob.max(axis=1)
    correct = (y_prob.argmax(axis=1) == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if not np.any(mask):
            continue
        acc_b = correct[mask].mean()
        conf_b = confidences[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(acc_b - conf_b)
    return float(ece)


def compute_brier(y_true: np.ndarray,
                   y_prob: np.ndarray,
                   K: int) -> float:
    """Brier Score (Eq. 27)."""
    one_hot = np.zeros_like(y_prob)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(((y_prob - one_hot) ** 2).sum(axis=1).mean())


def compute_nll(y_true: np.ndarray,
                 y_prob: np.ndarray,
                 eps: float = 1e-8) -> float:
    """Negative log-likelihood."""
    p_true = y_prob[np.arange(len(y_true)), y_true].clip(eps, 1.0)
    return float(-np.log(p_true).mean())


# ── Bootstrap CI (Eq. 29) ─────────────────────────────────────────────────────

def bootstrap_confidence_interval(y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    metric_fn,
                                    B: int = 1000,
                                    alpha: float = 0.05,
                                    seed: int = 42) -> dict:
    """
    Bootstrap 95% confidence interval for a metric.

    Args:
        metric_fn : callable(y_true_boot, y_pred_boot) → float

    Returns:
        dict with observed, ci_low, ci_high, std
    """
    rng     = np.random.default_rng(seed)
    n       = len(y_true)
    samples = []

    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samples.append(metric_fn(y_true[idx], y_pred[idx]))

    samples   = np.array(samples)
    observed  = metric_fn(y_true, y_pred)
    ci_lo     = np.percentile(samples, 100 * alpha / 2)
    ci_hi     = np.percentile(samples, 100 * (1 - alpha / 2))

    return {
        'observed': float(observed),
        'ci_low'  : float(ci_lo),
        'ci_high' : float(ci_hi),
        'std'     : float(samples.std()),
    }


# ── Paired bootstrap difference test (Eq. 30) ────────────────────────────────

def paired_bootstrap_test(y_true: np.ndarray,
                            y_pred_A: np.ndarray,
                            y_pred_B: np.ndarray,
                            metric_fn,
                            B: int = 1000,
                            seed: int = 42) -> dict:
    """
    Paired bootstrap test: H₀: M(A) = M(B).
    Returns one-sided p-value (A > B), two-sided p-value, observed delta, and CI.
    """
    rng      = np.random.default_rng(seed)
    n        = len(y_true)
    deltas   = []
    m_A_boot = []
    m_B_boot = []

    for _ in range(B):
        idx  = rng.integers(0, n, size=n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mA_b = metric_fn(y_true[idx], y_pred_A[idx])
            mB_b = metric_fn(y_true[idx], y_pred_B[idx])
        deltas.append(mA_b - mB_b)
        m_A_boot.append(mA_b)
        m_B_boot.append(mB_b)

    deltas   = np.array(deltas)
    m_A_boot = np.array(m_A_boot)
    m_B_boot = np.array(m_B_boot)
    delta_obs = metric_fn(y_true, y_pred_A) - metric_fn(y_true, y_pred_B)

    # Small-sample corrected bootstrap p-values.
    p_one_sided = (1.0 + np.sum(deltas <= 0)) / (B + 1.0)  # A > B
    p_two_sided = 2.0 * min(
        (1.0 + np.sum(deltas <= 0)) / (B + 1.0),
        (1.0 + np.sum(deltas >= 0)) / (B + 1.0),
    )
    p_two_sided = min(float(p_two_sided), 1.0)

    # Cohen's d from bootstrap distributions (Eq. 32)
    s_pooled = np.sqrt((m_A_boot.std()**2 + m_B_boot.std()**2) / 2)
    cohens_d = (m_A_boot.mean() - m_B_boot.mean()) / (s_pooled + 1e-8)

    return {
        'delta_obs'   : float(delta_obs),
        'p_one_sided' : float(p_one_sided),
        'p_two_sided' : float(p_two_sided),
        'ci_delta_low': float(np.percentile(deltas, 2.5)),
        'ci_delta_high': float(np.percentile(deltas, 97.5)),
        'cohens_d'    : float(cohens_d),
        'effect_size' : _effect_label(abs(cohens_d)),
    }


def _effect_label(d: float) -> str:
    if d < 0.2: return 'small'
    if d < 0.5: return 'medium'
    if d < 0.8: return 'large'
    return 'very_large'


# ── McNemar test (Eq. 31) ─────────────────────────────────────────────────────

def mcnemar_test(y_true: np.ndarray,
                  y_pred_A: np.ndarray,
                  y_pred_B: np.ndarray) -> dict:
    """
    McNemar continuity-corrected test for paired error comparison.
    n_01 = correct by A, wrong by B
    n_10 = wrong by A, correct by B
    """
    correct_A = (y_pred_A == y_true)
    correct_B = (y_pred_B == y_true)
    n_01 = int((correct_A & ~correct_B).sum())
    n_10 = int((~correct_A & correct_B).sum())
    denom = n_01 + n_10
    if denom == 0:
        return {'chi2': 0.0, 'p_value': 1.0, 'n_01': 0, 'n_10': 0,
                'significant': False}
    if denom < 25:
        # Exact binomial McNemar is safer for small discordant counts.
        try:
            p = stats.binomtest(min(n_01, n_10), n=denom, p=0.5, alternative='two-sided').pvalue
        except AttributeError:  # older scipy fallback
            p = stats.binom_test(min(n_01, n_10), n=denom, p=0.5, alternative='two-sided')
        return {
            'test': 'exact_binomial',
            'chi2': None,
            'p_value': float(p),
            'n_01': n_01,
            'n_10': n_10,
            'significant': bool(p < 0.05),
        }
    chi2 = (abs(n_01 - n_10) - 1.0)**2 / denom
    p    = 1.0 - stats.chi2.cdf(chi2, df=1)
    return {
        'test'       : 'continuity_corrected_chi2',
        'chi2'       : float(chi2),
        'p_value'    : float(p),
        'n_01'       : n_01,
        'n_10'       : n_10,
        'significant': bool(p < 0.05),
    }


# ── Prototype quality metrics (Eq. 28) ───────────────────────────────────────

def prototype_quality_metrics(embeddings: np.ndarray,
                                y_true: np.ndarray,
                                prototypes: np.ndarray,
                                K: int,
                                class_names: list = None) -> dict:
    """
    Compute intra-class radius, nearest inter-class distance, compactness ratio.
    """
    intra = np.zeros(K)
    for k in range(K):
        mask = (y_true == k)
        if mask.sum() == 0:
            continue
        diff = embeddings[mask] - prototypes[k]
        intra[k] = np.linalg.norm(diff, axis=1).mean()

    # Inter: nearest prototype distance
    dists_between = np.linalg.norm(
        prototypes[:, None, :] - prototypes[None, :, :], axis=-1)
    np.fill_diagonal(dists_between, np.inf)
    inter = dists_between.min(axis=1)

    CR = inter / (intra + 1e-8)
    CR_macro = CR.mean()

    result = {
        'CR_macro'   : float(CR_macro),
        'intra_mean' : float(intra.mean()),
        'inter_mean' : float(inter[inter < np.inf].mean()),
        'per_class'  : {}
    }
    for k in range(K):
        name = class_names[k] if class_names else str(k)
        result['per_class'][name] = {
            'intra': float(intra[k]),
            'inter': float(inter[k]) if inter[k] < np.inf else None,
            'CR'   : float(CR[k]),
        }
    return result


# ── MC Dropout Uncertainty (Phase 11) ────────────────────────────────────────

def mc_dropout_uncertainty(model,
                             X_test: np.ndarray,
                             T: int = 50,
                             device: torch.device = None,
                             batch_size: int = 128) -> dict:
    """
    Monte Carlo Dropout uncertainty estimation (Eqs. 19–22).
    Runs T stochastic forward passes with dropout ACTIVE.

    Returns:
        dict with mean_probs, entropy, d_nearest, uncertainty_flag
    """
    if device is None:
        device = torch.device('cpu')

    model.train()   # enable dropout
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    N = len(X_test)
    K = model.K

    all_probs_t = np.zeros((T, N, K), dtype=np.float32)
    all_emb_det = None
    all_d_det   = None

    # T stochastic passes
    for t in range(T):
        preds_t = []
        for i in range(0, N, batch_size):
            xb = X_tensor[i:i+batch_size].to(device)
            with torch.no_grad():
                probs, _, _, _ = model(xb)
            preds_t.append(probs.cpu().numpy())
        all_probs_t[t] = np.concatenate(preds_t, axis=0)

    # Mean prediction
    mean_probs = all_probs_t.mean(axis=0)           # [N, K]
    entropy    = -(mean_probs * np.log(mean_probs + 1e-8)).sum(axis=1)  # [N]

    # Deterministic pass for prototype distance uncertainty
    model.eval()
    embs_all = []
    d_all    = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = X_tensor[i:i+batch_size].to(device)
            _, e, _, d = model(xb)
            embs_all.append(e.cpu().numpy())
            d_all.append(d.cpu().numpy())
    embs_all = np.concatenate(embs_all, axis=0)
    d_all    = np.concatenate(d_all, axis=0)

    d_nearest = np.sqrt(d_all.min(axis=1))         # nearest prototype L2 dist

    return {
        'mean_probs'   : mean_probs,
        'entropy'      : entropy,
        'embeddings'   : embs_all,
        'd_all'        : d_all,
        'd_nearest'    : d_nearest,
    }


def compute_uncertainty_threshold(d_nearest_train: np.ndarray,
                                    pct: float = 95.0) -> float:
    """OOD threshold δ = p95 of training d_nearest (Eq. 21)."""
    return float(np.percentile(d_nearest_train, pct))


def uncertainty_hypothesis_test(entropy: np.ndarray,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray) -> dict:
    """
    H5: E[H | wrong] > E[H | correct].
    Mann-Whitney U test + Cohen's d.
    """
    correct_mask = (y_pred == y_true)
    h_correct = entropy[correct_mask]
    h_wrong   = entropy[~correct_mask]

    if len(h_wrong) == 0 or len(h_correct) == 0:
        return {'error': 'no wrong or correct predictions'}

    stat, p = stats.mannwhitneyu(h_wrong, h_correct, alternative='greater')
    d = (h_wrong.mean() - h_correct.mean()) / (
        np.sqrt((h_wrong.std()**2 + h_correct.std()**2) / 2) + 1e-8)
    r = 1 - 2 * stat / (len(h_wrong) * len(h_correct))

    return {
        'mean_H_wrong'  : float(h_wrong.mean()),
        'mean_H_correct': float(h_correct.mean()),
        'mann_whitney_U': float(stat),
        'p_value'       : float(p),
        'cohens_d'      : float(d),
        'rank_biserial_r': float(r),
        'hypothesis_supported': bool(p < 0.05),
    }


# ── Gate interpretability (Eqs. 35–36) ────────────────────────────────────────

def gate_interpretability(gate_values: np.ndarray,
                         y_true: np.ndarray,
                         chm_raw: np.ndarray,
                         class_names: list,
                         K: int,
                         y_pred: np.ndarray = None) -> dict:
    """
    Per-class mean gate values using TRUE classes for ecological validation.
    Optionally also reports predicted-class gate behavior when y_pred is supplied.
    Includes CHM positive rate because CHM is sparse in this dataset.
    """
    gate_scalar = gate_values.mean(axis=1)

    def _summarize(labels: np.ndarray) -> dict:
        per_class = {}
        mean_gate = np.full(K, np.nan)
        mean_chm = np.full(K, np.nan)
        chm_pos = np.full(K, np.nan)
        for k in range(K):
            mask = labels == k
            n_k = int(mask.sum())
            if n_k == 0:
                per_class[class_names[k]] = {
                    'mean_gate': None, 'mean_chm': None, 'chm_positive_rate': None,
                    'n_test': 0, 'dominant_modality': 'N/A'
                }
                continue
            g_k = float(gate_scalar[mask].mean())
            chm_k = float(chm_raw[mask].mean())
            pos_k = float((chm_raw[mask] > 0).mean())
            mean_gate[k] = g_k
            mean_chm[k] = chm_k
            chm_pos[k] = pos_k
            if g_k > 0.55:
                dom = 'terrain'
            elif g_k < 0.45:
                dom = 'spectral'
            else:
                dom = 'both'
            per_class[class_names[k]] = {
                'mean_gate': round(g_k, 4),
                'mean_chm': round(chm_k, 4),
                'chm_positive_rate': round(pos_k, 4),
                'n_test': n_k,
                'dominant_modality': dom,
            }
        valid = np.where(~np.isnan(mean_gate) & ~np.isnan(mean_chm))[0]
        if len(valid) >= 3:
            rho_mean, p_mean = stats.spearmanr(mean_gate[valid], mean_chm[valid])
            rho_pos, p_pos = stats.spearmanr(mean_gate[valid], chm_pos[valid])
        else:
            rho_mean, p_mean, rho_pos, p_pos = 0.0, 1.0, 0.0, 1.0
        return {
            'per_class': per_class,
            'spearman_rho_gate_vs_chm_mean': float(rho_mean),
            'spearman_p_gate_vs_chm_mean': float(p_mean),
            'spearman_rho_gate_vs_chm_positive_rate': float(rho_pos),
            'spearman_p_gate_vs_chm_positive_rate': float(p_pos),
        }

    true_summary = _summarize(y_true)
    out = {
        'analysis_note': 'Ecological validation uses true labels. Predicted-label summary is optional model-decision behavior.',
        'by_true_class': true_summary,
        # Backward-compatible keys used by visualization.py
        'per_class': true_summary['per_class'],
        'spearman_rho': true_summary['spearman_rho_gate_vs_chm_mean'],
        'spearman_p': true_summary['spearman_p_gate_vs_chm_mean'],
        'ecological_valid': bool(
            true_summary['spearman_rho_gate_vs_chm_mean'] > 0 and
            true_summary['spearman_p_gate_vs_chm_mean'] < 0.05
        ),
    }
    if y_pred is not None:
        out['by_predicted_class'] = _summarize(y_pred)
    return out




def save_manuscript_tables(report: dict, class_names: list, stats_dir: Path) -> None:
    """Save manuscript-ready tables as CSV files under outputs/stats."""
    reference_model = report.get('reference_model', 'ecotreefusenet')
    import pandas as pd

    # Table: main performance comparison.
    perf_rows = []
    for model_name, m in report.get('exp_metrics', {}).items():
        perf_rows.append({
            'model': model_name,
            'accuracy': m.get('accuracy'),
            'macro_f1': m.get('macro_f1'),
            'weighted_f1': m.get('weighted_f1'),
            'balanced_accuracy': m.get('balanced_acc'),
            'mcc': m.get('mcc'),
            'cohens_kappa': m.get('kappa'),
            'ece': m.get('ece'),
            'brier_score': m.get('brier'),
            'nll': m.get('nll'),
        })
    if perf_rows:
        pd.DataFrame(perf_rows).to_csv(stats_dir / 'table_main_performance.csv', index=False)

    # Table: ablation deltas relative to the proposed reference model.
    exp_metrics = report.get('exp_metrics', {})
    if reference_model in exp_metrics:
        eco = exp_metrics[reference_model]
        ablation_rows = []
        for model_name, m in exp_metrics.items():
            ablation_rows.append({
                'model': model_name,
                'macro_f1': m.get('macro_f1'),
                f'delta_macro_f1_vs_{reference_model}': m.get('macro_f1', 0.0) - eco.get('macro_f1', 0.0),
                'balanced_accuracy': m.get('balanced_acc'),
                f'delta_balanced_accuracy_vs_{reference_model}': m.get('balanced_acc', 0.0) - eco.get('balanced_acc', 0.0),
                'mcc': m.get('mcc'),
                f'delta_mcc_vs_{reference_model}': m.get('mcc', 0.0) - eco.get('mcc', 0.0),
            })
        pd.DataFrame(ablation_rows).to_csv(stats_dir / 'table_ablation_deltas.csv', index=False)

    # Table: bootstrap CIs for proposed reference model.
    ci_rows = []
    for metric_name, vals in report.get('bootstrap_reference', report.get('bootstrap_ecofusenet', {})).items():
        row = {'metric': metric_name}
        row.update(vals)
        ci_rows.append(row)
    if ci_rows:
        pd.DataFrame(ci_rows).to_csv(stats_dir / f'table_bootstrap_ci_{reference_model}.csv', index=False)

    # Table: paired bootstrap tests.
    paired_rows = []
    for comparison, metrics in report.get('paired_tests', {}).items():
        for metric_name, vals in metrics.items():
            row = {'comparison': comparison, 'metric': metric_name}
            row.update(vals)
            paired_rows.append(row)
    if paired_rows:
        pd.DataFrame(paired_rows).to_csv(stats_dir / 'table_paired_bootstrap_tests.csv', index=False)

    # Table: McNemar tests.
    mcnemar_rows = []
    for comparison, vals in report.get('mcnemar_tests', {}).items():
        row = {'comparison': comparison}
        row.update(vals)
        mcnemar_rows.append(row)
    if mcnemar_rows:
        pd.DataFrame(mcnemar_rows).to_csv(stats_dir / 'table_mcnemar_tests.csv', index=False)

    # Table: minority-class F1.
    minority = report.get('minority_analysis', {})
    if minority:
        rows = []
        for model_name, vals in minority.items():
            row = {'model': model_name}
            row.update(vals)
            rows.append(row)
        pd.DataFrame(rows).to_csv(stats_dir / 'table_minority_class_f1.csv', index=False)

    # Table: per-class F1 for every model.
    pc_rows = []
    for model_name, m in exp_metrics.items():
        f1s = m.get('per_class_f1', [])
        for k, f1_val in enumerate(f1s):
            pc_rows.append({
                'model': model_name,
                'class_index': k,
                'class_name': class_names[k] if k < len(class_names) else str(k),
                'f1': f1_val,
            })
    if pc_rows:
        pd.DataFrame(pc_rows).to_csv(stats_dir / 'table_per_class_f1_all_models.csv', index=False)

    print(f"  [Stats] CSV tables saved to {stats_dir}")


# ── Full statistical evaluation runner ───────────────────────────────────────

def run_full_statistical_analysis(results: dict,
                                    y_true: np.ndarray,
                                    K: int,
                                    class_names: list,
                                    output_dir: str,
                                    B: int = 1000) -> dict:
    """
    Run all statistical tests comparing the proposed reference model against baselines/ablations.

    Args:
        results : dict mapping exp_name → {'preds': ..., 'probs': ...}
        y_true  : ground truth labels
        K, class_names, output_dir

    Returns:
        Full statistical report dict
    """
    out_dir = Path(output_dir)
    stats_dir = out_dir / 'stats'
    stats_dir.mkdir(parents=True, exist_ok=True)

    macro_f1_fn = lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0)
    bal_acc_fn  = lambda yt, yp: balanced_accuracy_score(yt, yp)

    # Per-experiment metrics
    exp_metrics = {}
    for name, res in results.items():
        m = compute_all_metrics(y_true, res['preds'], res['probs'], K)
        exp_metrics[name] = m
        print(f"  {name:30s} | Macro F1={m['macro_f1']:.4f} | "
              f"BA={m['balanced_acc']:.4f} | MCC={m['mcc']:.4f} | "
              f"ECE={m['ece']:.4f}")

    # Proposed reference model priority: final Plus/hierarchical variants first.
    reference_priority = [
        'hierarchical_ecotreefusenet_plus',
        'bias_tuned_ecotreefusenet_plus',
        'calibrated_ecotreefusenet_plus',
        'ecotreefusenet_plus',
        'ecotreefusenet',
        'ecofusenet',
    ]
    reference_model = next((m for m in reference_priority if m in results), next(iter(results.keys())))
    ref_preds = results.get(reference_model, {}).get('preds')
    bootstrap_results = {}
    if ref_preds is not None:
        print(f"\n  Computing bootstrap CIs (B={B})...")
        bootstrap_results['macro_f1'] = bootstrap_confidence_interval(
            y_true, ref_preds, macro_f1_fn, B=B)
        bootstrap_results['balanced_acc'] = bootstrap_confidence_interval(
            y_true, ref_preds, bal_acc_fn, B=B)

    # Paired bootstrap tests: reference model vs. each other model
    paired_results = {}
    if ref_preds is not None:
        for name, res in results.items():
            if name == reference_model:
                continue
            print(f"  Paired bootstrap: {reference_model} vs {name}...")
            paired_results[f'vs_{name}'] = {
                'macro_f1': paired_bootstrap_test(
                    y_true, ref_preds, res['preds'], macro_f1_fn, B=B),
                'balanced_acc': paired_bootstrap_test(
                    y_true, ref_preds, res['preds'], bal_acc_fn, B=B),
            }

    # McNemar tests: reference model vs key baselines
    mcnemar_results = {}
    if ref_preds is not None:
        for name in ['rf', 'xgboost', 'extra_trees', 'hist_gradient_boosting', 'lightgbm', 'catboost', 'ecofusenet', 'ecotreefusenet', 'nogate_mlp']:
            if name in results and name != reference_model:
                mcnemar_results[f'vs_{name}'] = mcnemar_test(
                    y_true, ref_preds, results[name]['preds'])

    # Minority class analysis
    minority_classes = ['BareSoil', 'SandstoneTalus', 'Frasera', 'Snow', 'SandstoneOutcrop']
    minority_analysis = {}
    for name, res in results.items():
        per_cls = f1_score(y_true, res['preds'], average=None,
                           zero_division=0, labels=list(range(K)))
        minority_f1 = {}
        for mc in minority_classes:
            if mc in class_names:
                k = class_names.index(mc)
                minority_f1[mc] = round(float(per_cls[k]), 4)
        minority_analysis[name] = minority_f1

    report = {
        'exp_metrics'     : exp_metrics,
        'reference_model': reference_model,
        'bootstrap_reference': bootstrap_results,
        'bootstrap_ecofusenet': bootstrap_results,
        'paired_tests'    : paired_results,
        'mcnemar_tests'   : mcnemar_results,
        'minority_analysis': minority_analysis,
    }

    save_manuscript_tables(report, class_names, stats_dir)

    with open(str(stats_dir / 'statistical_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  [Stats] Report saved to {stats_dir}/statistical_report.json")

    return report