"""
Calibration, class-bias tuning, weighted voting, and hierarchy-aware correction
for EcoTreeFuseNet-Plus.

All selection/tuning uses validation data only. Test data is used once after the
validation-selected transformation is fixed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import f1_score

from evaluation import compute_ece, compute_nll


def normalize_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, None)
    p = p / np.clip(p.sum(axis=1, keepdims=True), eps, None)
    return p.astype(np.float32)


def softmax_logits(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)).astype(np.float32)


def probs_to_logits(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(normalize_probs(p, eps), eps, 1.0))


def apply_temperature_scaling(p: np.ndarray, temperature: float) -> np.ndarray:
    logits = probs_to_logits(p) / max(float(temperature), 1e-6)
    return softmax_logits(logits)


def fit_temperature_scaling(p_val: np.ndarray,
                            y_val: np.ndarray,
                            grid: np.ndarray = None,
                            criterion: str = "nll",
                            temperature_bounds=(0.05, 10.0)) -> Dict:
    """Fit a scalar temperature on validation probabilities only.

    Phase 3 correction
    ------------------
    The original search was truncated at T=0.50.  This implementation optimizes
    log(T) continuously over a prespecified positive interval (default 0.05 to
    10.0), with a dense-grid fallback.  Optimizing log(T) keeps positivity and
    gives comparable resolution below and above T=1.

    ``grid`` is retained only for reproducibility experiments; when supplied it
    is evaluated exactly.  The default path is continuous validation-NLL
    minimization.  Test labels are never accessed.
    """
    from scipy.optimize import minimize_scalar

    p_val = normalize_probs(p_val)
    y_val = np.asarray(y_val, dtype=int)
    criterion = str(criterion).lower().strip()
    if criterion not in {"nll", "ece"}:
        raise ValueError("criterion must be 'nll' or 'ece'")

    lo, hi = map(float, temperature_bounds)
    if not (0.0 < lo < hi):
        raise ValueError("temperature_bounds must satisfy 0 < low < high")

    def evaluate(T: float) -> Dict:
        pT = apply_temperature_scaling(p_val, float(T))
        nll = compute_nll(y_val, pT)
        ece = compute_ece(y_val, pT)
        f1 = f1_score(y_val, pT.argmax(axis=1), average="macro", zero_division=0)
        score = nll if criterion == "nll" else ece
        return {
            "temperature": float(T),
            "val_nll": float(nll),
            "val_ece": float(ece),
            "val_macro_f1": float(f1),
            "score": float(score),
        }

    rows = []
    if grid is not None:
        temps = np.asarray(grid, dtype=float)
        temps = temps[np.isfinite(temps) & (temps > 0)]
        if temps.size == 0:
            raise ValueError("grid contains no positive finite temperatures")
        rows = [evaluate(float(T)) for T in temps]
        best = min(rows, key=lambda r: (r["score"], abs(np.log(r["temperature"]))))
        method = "explicit_grid"
        optimizer_success = True
    else:
        log_lo, log_hi = np.log(lo), np.log(hi)

        def objective(logT):
            return evaluate(float(np.exp(logT)))["score"]

        opt = minimize_scalar(
            objective, bounds=(log_lo, log_hi), method="bounded",
            options={"xatol": 1e-7, "maxiter": 300},
        )
        optimizer_success = bool(opt.success and np.isfinite(opt.fun))
        if optimizer_success:
            T_opt = float(np.exp(opt.x))
            # Include exact neutral and bounds in audit candidates.
            candidates = sorted(set([lo, hi, 1.0, T_opt]))
            rows = [evaluate(T) for T in candidates]
            best = min(rows, key=lambda r: (r["score"], abs(np.log(r["temperature"]))))
            method = "bounded_log_temperature"
        else:
            temps = np.geomspace(lo, hi, 401)
            rows = [evaluate(float(T)) for T in temps]
            best = min(rows, key=lambda r: (r["score"], abs(np.log(r["temperature"]))))
            method = "geometric_grid_fallback"

    out = dict(best)
    out.update({
        "criterion": criterion,
        "search_method": method,
        "temperature_lower_bound": lo,
        "temperature_upper_bound": hi,
        "optimizer_success": bool(optimizer_success),
        "at_lower_bound": bool(np.isclose(best["temperature"], lo, rtol=0, atol=max(1e-6, lo*1e-4))),
        "at_upper_bound": bool(np.isclose(best["temperature"], hi, rtol=0, atol=max(1e-6, hi*1e-4))),
        "all_temperatures": rows,
    })
    return out


def tune_class_bias(p_val: np.ndarray,
                    y_val: np.ndarray,
                    max_abs_bias: float = 1.25,
                    n_grid: int = 21,
                    passes: int = 3) -> Dict:
    """
    Greedy validation-only class-bias tuning for Macro F1.

    Prediction rule: argmax_k [log p_k(x) + b_k]. This does not retrain the
    base model and is useful for imbalanced multi-class Macro-F1 optimization.
    """
    logits_val = probs_to_logits(p_val)
    K = logits_val.shape[1]
    bias = np.zeros(K, dtype=np.float32)
    candidates = np.linspace(-max_abs_bias, max_abs_bias, n_grid)

    def score(b):
        pred = (logits_val + b[None, :]).argmax(axis=1)
        return f1_score(y_val, pred, average="macro", zero_division=0)

    best_f1 = score(bias)
    history = [{"pass": 0, "class_index": -1, "bias": 0.0, "val_macro_f1": float(best_f1)}]
    for p in range(1, passes + 1):
        improved = False
        for k in range(K):
            old = bias[k]
            best_local = (best_f1, old)
            for v in candidates:
                trial = bias.copy()
                trial[k] = v
                f1 = score(trial)
                if f1 > best_local[0] + 1e-12:
                    best_local = (f1, float(v))
            bias[k] = best_local[1]
            if best_local[0] > best_f1 + 1e-12:
                best_f1 = best_local[0]
                improved = True
                history.append({"pass": p, "class_index": int(k), "bias": float(bias[k]), "val_macro_f1": float(best_f1)})
        if not improved:
            break
    return {"bias": bias.astype(np.float32), "val_macro_f1": float(best_f1), "history": history}


def apply_class_bias(p: np.ndarray, bias: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    logits = probs_to_logits(p) + np.asarray(bias, dtype=np.float32)[None, :]
    probs = softmax_logits(logits)
    preds = probs.argmax(axis=1).astype(np.int64)
    return probs, preds


def optimize_weighted_soft_voting(branch_val_probs: Dict[str, np.ndarray],
                                  y_val: np.ndarray,
                                  branch_test_probs: Dict[str, np.ndarray],
                                  random_state: int = 42,
                                  n_random: int = 3000) -> Dict:
    """
    Validation-only optimization of soft-voting weights over branch probabilities.
    Includes single-branch, uniform, Dirichlet random search, and sparse mixtures.
    """
    names = list(branch_val_probs.keys())
    if len(names) == 0:
        raise ValueError("No branches supplied for weighted soft voting.")
    rng = np.random.default_rng(random_state)
    M = len(names)

    candidates = []
    candidates.append(("uniform", np.ones(M) / M))
    for i, name in enumerate(names):
        w = np.zeros(M); w[i] = 1.0
        candidates.append((f"single_{name}", w))
    for _ in range(n_random):
        # Sparse-friendly Dirichlet: encourages both individual dominance and mixtures.
        alpha = rng.uniform(0.15, 1.5, size=M)
        w = rng.dirichlet(alpha)
        candidates.append(("dirichlet", w))

    best = None
    rows = []
    val_stack = np.stack([branch_val_probs[n] for n in names], axis=0)  # [M,N,K]
    test_stack = np.stack([branch_test_probs[n] for n in names], axis=0)
    for label, w in candidates:
        pv = normalize_probs(np.tensordot(w, val_stack, axes=(0, 0)))
        pred = pv.argmax(axis=1)
        f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        ece = compute_ece(y_val, pv)
        row = {"candidate": label, "val_macro_f1": float(f1), "val_ece": float(ece), "weights": {n: float(w[i]) for i, n in enumerate(names)}}
        rows.append(dict(row))
        if best is None or (f1 > best["val_macro_f1"] + 1e-12) or (abs(f1 - best["val_macro_f1"]) <= 1e-12 and ece < best["val_ece"]):
            # IMPORTANT: use a copy, not the same dict object held inside rows.
            # Otherwise best["all_candidates_top20"] can contain best itself, which
            # creates a circular JSON reference when the report is saved.
            best = dict(row)
            best_w = w.copy()
    p_val_best = normalize_probs(np.tensordot(best_w, val_stack, axes=(0, 0)))
    p_test_best = normalize_probs(np.tensordot(best_w, test_stack, axes=(0, 0)))
    top20 = sorted(rows, key=lambda r: (-r["val_macro_f1"], r["val_ece"]))[:20]
    best["all_candidates_top20"] = [dict(r) for r in top20]
    return {"best": best, "val_probs": p_val_best, "test_probs": p_test_best, "branch_names": names}


def hierarchy_correct_probs(p: np.ndarray, phi: np.ndarray, C: int, alpha: float = 1.0) -> np.ndarray:
    """Hierarchy-aware correction: p_k ← p_k * P(coarse=phi(k))^alpha, then normalize."""
    p = normalize_probs(p)
    coarse_p = np.zeros((p.shape[0], C), dtype=np.float32)
    for k, c in enumerate(phi):
        coarse_p[:, int(c)] += p[:, k]
    adjusted = p.copy()
    for k, c in enumerate(phi):
        adjusted[:, k] *= np.power(np.clip(coarse_p[:, int(c)], 1e-12, 1.0), alpha)
    return normalize_probs(adjusted)


def tune_hierarchy_alpha(p_val: np.ndarray, y_val: np.ndarray, p_test: np.ndarray,
                         phi: np.ndarray, C: int,
                         alpha_grid=None) -> Dict:
    if alpha_grid is None:
        alpha_grid = [0.0, 0.15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    best = None
    rows = []
    for a in alpha_grid:
        pv = hierarchy_correct_probs(p_val, phi, C, alpha=float(a))
        pred = pv.argmax(axis=1)
        f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        ece = compute_ece(y_val, pv)
        row = {"alpha": float(a), "val_macro_f1": float(f1), "val_ece": float(ece)}
        rows.append(row)
        if best is None or (f1 > best["val_macro_f1"] + 1e-12) or (abs(f1 - best["val_macro_f1"]) <= 1e-12 and ece < best["val_ece"]):
            best = row
    p_test_h = hierarchy_correct_probs(p_test, phi, C, alpha=best["alpha"])
    p_val_h = hierarchy_correct_probs(p_val, phi, C, alpha=best["alpha"])
    best["all_alpha"] = rows
    return {"best": best, "val_probs": p_val_h, "test_probs": p_test_h}


def make_json_safe(obj, _seen=None):
    """Convert nested Python/numpy/sklearn objects to JSON-safe objects and break cycles."""
    if _seen is None:
        _seen = set()
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    oid = id(obj)
    if isinstance(obj, (dict, list, tuple, set)):
        if oid in _seen:
            return "__circular_reference_removed__"
        _seen.add(oid)
        if isinstance(obj, dict):
            out = {str(k): make_json_safe(v, _seen) for k, v in obj.items()}
        else:
            out = [make_json_safe(v, _seen) for v in obj]
        _seen.remove(oid)
        return out
    # sklearn models and other non-serializable objects are summarized, not dumped.
    return str(obj)


def save_calibration_report(report: Dict, output_dir: str, filename: str = "calibration_plus_report.json") -> None:
    out = Path(output_dir) / "stats"
    out.mkdir(parents=True, exist_ok=True)
    safe_report = make_json_safe(report)
    with open(out / filename, "w", encoding="utf-8") as f:
        json.dump(safe_report, f, indent=2)
