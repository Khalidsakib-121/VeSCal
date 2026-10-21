"""
Loss functions for EcoFuseNet.

Important update: prototype separation is now bounded and margin-based, avoiding
the unbounded negative loss that can explode prototype norms.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(class_counts: np.ndarray, mode: str = "sqrt") -> torch.Tensor:
    counts = np.asarray(class_counts, dtype=np.float64)
    n_max = counts.max()
    if mode == "sqrt":
        raw = np.sqrt(n_max / np.clip(counts, 1, None))
    elif mode == "inverse":
        raw = 1.0 / np.clip(counts, 1, None)
    else:
        raise ValueError(f"Unknown weight mode: {mode}")
    raw = raw / (raw.sum() / len(raw))  # mean = 1
    return torch.tensor(raw, dtype=torch.float32)


def proto_contrastive_loss(e: torch.Tensor,
                           y: torch.Tensor,
                           P_fine: torch.Tensor,
                           tau: torch.Tensor,
                           weights: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(e, P_fine).pow(2)
    s = -d / tau
    loss_i = -s[torch.arange(len(y), device=e.device), y] + torch.logsumexp(s, dim=1)
    w = weights.to(e.device)[y]
    return (w * loss_i).mean()


def coarse_consistency_loss(e: torch.Tensor,
                            y_coarse: torch.Tensor,
                            P_coarse: torch.Tensor) -> torch.Tensor:
    pc = P_coarse[y_coarse]
    return ((e - pc) ** 2).sum(dim=1).mean()


def prototype_separation_loss(P_fine: torch.Tensor, margin: float = 2.0) -> torch.Tensor:
    """
    Bounded margin-based separation loss.
    Penalizes only prototype pairs whose L2 distance is below margin.
    """
    K = P_fine.shape[0]
    idx_i, idx_j = torch.triu_indices(K, K, offset=1, device=P_fine.device)
    diff = P_fine[idx_i] - P_fine[idx_j]
    dist = torch.sqrt((diff ** 2).sum(dim=1) + 1e-8)
    return F.relu(margin - dist).pow(2).mean()


def hierarchical_prototype_loss(e: torch.Tensor,
                                y_fine: torch.Tensor,
                                y_coarse: torch.Tensor,
                                P_fine: torch.Tensor,
                                P_coarse: torch.Tensor,
                                tau: torch.Tensor,
                                weights: torch.Tensor,
                                lambda1: float = 0.3,
                                lambda2: float = 0.1,
                                sep_margin: float = 2.0) -> dict:
    L_fine = proto_contrastive_loss(e, y_fine, P_fine, tau, weights)
    L_coarse = coarse_consistency_loss(e, y_coarse, P_coarse)
    L_sep = prototype_separation_loss(P_fine, margin=sep_margin)
    L_total = L_fine + lambda1 * L_coarse + lambda2 * L_sep
    return {
        "total": L_total,
        "fine": float(L_fine.detach().cpu()),
        "coarse": float(L_coarse.detach().cpu()),
        "sep": float(L_sep.detach().cpu()),
    }


def flat_cross_entropy_loss(e: torch.Tensor,
                            y_fine: torch.Tensor,
                            classifier_head: nn.Linear) -> torch.Tensor:
    return F.cross_entropy(classifier_head(e), y_fine)


def ce_with_weak_prototype_regularization(e: torch.Tensor,
                                          y_fine: torch.Tensor,
                                          y_coarse: torch.Tensor,
                                          classifier_head: nn.Module,
                                          P_fine: torch.Tensor,
                                          P_coarse: torch.Tensor,
                                          tau: torch.Tensor,
                                          weights: torch.Tensor,
                                          proto_weight: float = 0.03,
                                          coarse_weight: float = 0.01,
                                          sep_weight: float = 0.001,
                                          sep_margin: float = 1.0,
                                          label_smoothing: float = 0.03) -> dict:
    """
    CE-dominant objective for Option-B EcoFuseNetV2.

    The prediction loss is standard class-balanced CE.
    Prototype terms are weak regularizers only. This avoids the previous failure mode
    where the prototype-distance classifier over-constrained class boundaries.
    """
    logits = classifier_head(e)
    L_ce = F.cross_entropy(
        logits,
        y_fine,
        weight=weights,
        label_smoothing=label_smoothing,
    )

    # Use prototype terms only when their weights are positive.
    if proto_weight > 0:
        L_proto = proto_contrastive_loss(e, y_fine, P_fine, tau, weights)
    else:
        L_proto = torch.zeros((), dtype=e.dtype, device=e.device)

    if coarse_weight > 0:
        L_coarse = coarse_consistency_loss(e, y_coarse, P_coarse)
    else:
        L_coarse = torch.zeros((), dtype=e.dtype, device=e.device)

    if sep_weight > 0:
        L_sep = prototype_separation_loss(P_fine, margin=sep_margin)
    else:
        L_sep = torch.zeros((), dtype=e.dtype, device=e.device)

    L_total = L_ce + proto_weight * L_proto + coarse_weight * L_coarse + sep_weight * L_sep
    return {
        "total": L_total,
        "ce": float(L_ce.detach().cpu()),
        "proto": float(L_proto.detach().cpu()),
        "coarse": float(L_coarse.detach().cpu()),
        "sep": float(L_sep.detach().cpu()),
    }
