"""Training and evaluation loops for EcoFuseNet and ablations."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from losses import flat_cross_entropy_loss, hierarchical_prototype_loss, ce_with_weak_prototype_regularization


@torch.no_grad()
def initialize_prototypes(model: nn.Module,
                          train_loader: DataLoader,
                          phi: np.ndarray,
                          K: int,
                          C: int,
                          device: torch.device) -> None:
    """Initialize fine/coarse prototypes from training embeddings only."""
    model.eval()
    all_e, all_y = [], []
    for X_b, y_fine_b, _ in train_loader:
        X_b = X_b.to(device)
        _, e, _, _ = model(X_b)
        all_e.append(e.detach().cpu())
        all_y.append(y_fine_b.detach().cpu())

    all_e = torch.cat(all_e, dim=0)
    all_y = torch.cat(all_y, dim=0)
    d_e = all_e.shape[1]

    P_fine_init = torch.zeros(K, d_e)
    for k in range(K):
        mask = all_y == k
        n_k = int(mask.sum().item())
        if n_k == 0:
            print(f"  [Warning] Class {k} has 0 training samples; prototype left as zero.")
            continue
        P_fine_init[k] = all_e[mask].mean(dim=0)
        if n_k < 20:
            print(f"  [Minority prototype init] class={k}, N_train={n_k}; using all samples.")

    P_coarse_init = torch.zeros(C, d_e)
    phi_t = torch.tensor(phi, dtype=torch.long)
    for c in range(C):
        fine_in_c = (phi_t == c).nonzero(as_tuple=True)[0]
        if len(fine_in_c) > 0:
            P_coarse_init[c] = P_fine_init[fine_in_c].mean(dim=0)

    with torch.no_grad():
        model.P_fine.copy_(P_fine_init.to(device))
        model.P_coarse.copy_(P_coarse_init.to(device))
    model.train()
    print(f"[Proto Init] Initialized {K} fine and {C} coarse prototypes from D_train.")


@torch.no_grad()
def evaluate_model(model: nn.Module,
                   loader: DataLoader,
                   device: torch.device,
                   K: int,
                   classifier_head: Optional[nn.Module] = None,
                   collect_internal: bool = False) -> dict:
    """Evaluate either prototype classifier or CE classifier head."""
    model.eval()
    if classifier_head is not None:
        classifier_head.eval()

    all_preds, all_labels, all_probs = [], [], []
    all_embeddings, all_gates, all_distances = [], [], []

    for X_b, y_fine_b, _ in loader:
        X_b = X_b.to(device)
        probs_proto, e, g, d = model(X_b)
        if classifier_head is not None:
            logits = classifier_head(e)
            probs = torch.softmax(logits, dim=1)
        else:
            probs = probs_proto

        preds = probs.argmax(dim=1).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y_fine_b.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())
        if collect_internal:
            all_embeddings.append(e.detach().cpu().numpy())
            all_gates.append(g.detach().cpu().numpy())
            all_distances.append(d.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs, axis=0)
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0,
                         labels=list(range(K)))
    out = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "preds": y_pred,
        "labels": y_true,
        "probs": y_prob,
        "per_class_f1": per_class,
    }
    if collect_internal:
        out["embeddings"] = np.concatenate(all_embeddings, axis=0)
        out["gate_values"] = np.concatenate(all_gates, axis=0)
        out["proto_distances"] = np.concatenate(all_distances, axis=0)
    model.train()
    if classifier_head is not None:
        classifier_head.train()
    return out


def _save_checkpoint(path: Path, model: nn.Module, epoch: int, val_f1: float,
                     classifier_head: Optional[nn.Module] = None) -> None:
    payload = {"model_state": model.state_dict(), "epoch": epoch, "val_macro_f1": val_f1}
    if classifier_head is not None:
        payload["classifier_head_state"] = classifier_head.state_dict()
    torch.save(payload, str(path))


def _load_checkpoint(path: Path, model: nn.Module, device: torch.device,
                     classifier_head: Optional[nn.Module] = None) -> None:
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if classifier_head is not None and "classifier_head_state" in ckpt:
        classifier_head.load_state_dict(ckpt["classifier_head_state"])


def train_ecofusenet(model: nn.Module,
                     loaders: dict,
                     data: dict,
                     phi: np.ndarray,
                     class_weights: torch.Tensor,
                     device: torch.device,
                     output_dir: str,
                     exp_name: str = "ecofusenet",
                     warmup_epochs: int = 25,
                     max_epochs: int = 150,
                     patience: int = 20,
                     lr_warmup: float = 1e-3,
                     lr_proto: float = 5e-4,
                     weight_decay: float = 1e-4,
                     batch_size: int = 64,
                     grad_clip: float = 1.0,
                     lambda1: float = 0.3,
                     lambda2: float = 0.1,
                     tau_init: float = 1.0,
                     use_proto_loss: bool = True,
                     sep_margin: float = 2.0) -> dict:
    """Train full EcoFuseNet or flat-loss CE ablation."""
    out_dir = Path(output_dir)
    ckpt_dir = out_dir / "checkpoints"
    pred_dir = out_dir / "predictions"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    K, C = data["K"], data["C"]
    weights = class_weights.to(device)

    # Warm-up / flat-loss head. For FlatLossEcoFuseNet, use registered classifier_head.
    classifier_head = getattr(model, "classifier_head", None)
    classifier_head_registered = classifier_head is not None
    if classifier_head is None:
        classifier_head = nn.Linear(model.d_e, K).to(device)
    else:
        classifier_head = classifier_head.to(device)

    def params_with_head():
        # If the classifier head is registered inside model, model.parameters() already includes it.
        if classifier_head_registered:
            return list(model.parameters())
        return list(model.parameters()) + list(classifier_head.parameters())

    print(f"\n[{exp_name}] Warm-up / CE phase ({warmup_epochs} epochs)")
    optimizer_warmup = torch.optim.Adam(
        params_with_head(),
        lr=lr_warmup,
        weight_decay=weight_decay,
    )
    warmup_history = []
    for epoch in range(1, warmup_epochs + 1):
        model.train(); classifier_head.train()
        total_loss, n_batches = 0.0, 0
        for X_b, y_fine_b, _ in loaders["train"]:
            X_b, y_fine_b = X_b.to(device), y_fine_b.to(device)
            _, e, _, _ = model(X_b)
            loss = flat_cross_entropy_loss(e, y_fine_b, classifier_head)
            optimizer_warmup.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_with_head(), grad_clip)
            optimizer_warmup.step()
            total_loss += float(loss.detach().cpu())
            n_batches += 1
        val_res = evaluate_model(model, loaders["val"], device, K, classifier_head=classifier_head)
        warmup_history.append({"epoch": epoch, "loss": total_loss / max(n_batches, 1), "val_macro_f1": val_res["macro_f1"]})
        if epoch == 1 or epoch % 5 == 0 or epoch == warmup_epochs:
            print(f"  epoch {epoch:03d}/{warmup_epochs} | CE={total_loss/max(n_batches,1):.4f} | val_F1={val_res['macro_f1']:.4f}")

    if use_proto_loss:
        print(f"[{exp_name}] Prototype initialization")
        initialize_prototypes(model, loaders["train"], phi, K, C, device)
        classifier_head_for_eval = None
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_proto, weight_decay=weight_decay)
    else:
        # Real flat CE ablation: continue using the CE classifier head, no random prototype classifier.
        classifier_head_for_eval = classifier_head
        optimizer = torch.optim.Adam(params_with_head(), lr=lr_proto, weight_decay=weight_decay)

    print(f"[{exp_name}] Main training phase (epochs {warmup_epochs+1}..{max_epochs}, patience={patience})")
    best_val_f1 = -float("inf")
    patience_counter = 0
    train_history = []
    ckpt_path = ckpt_dir / f"{exp_name}_best.pt"

    for epoch in range(warmup_epochs + 1, max_epochs + 1):
        model.train()
        if classifier_head_for_eval is not None:
            classifier_head_for_eval.train()
        epoch_stats = {"total": 0.0, "fine": 0.0, "coarse": 0.0, "sep": 0.0}
        n_batches = 0

        for X_b, y_fine_b, y_coarse_b in loaders["train"]:
            X_b = X_b.to(device)
            y_fine_b = y_fine_b.to(device)
            y_coarse_b = y_coarse_b.to(device)
            _, e, _, _ = model(X_b)
            if use_proto_loss:
                loss_dict = hierarchical_prototype_loss(
                    e, y_fine_b, y_coarse_b, model.P_fine, model.P_coarse, model.tau,
                    weights, lambda1=lambda1, lambda2=lambda2, sep_margin=sep_margin,
                )
                loss = loss_dict["total"]
            else:
                loss = flat_cross_entropy_loss(e, y_fine_b, classifier_head_for_eval)
                loss_dict = {"total": loss, "fine": 0.0, "coarse": 0.0, "sep": 0.0}

            optimizer.zero_grad()
            loss.backward()
            if use_proto_loss:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(params_with_head(), grad_clip)
            optimizer.step()

            epoch_stats["total"] += float(loss.detach().cpu())
            epoch_stats["fine"] += float(loss_dict.get("fine", 0.0))
            epoch_stats["coarse"] += float(loss_dict.get("coarse", 0.0))
            epoch_stats["sep"] += float(loss_dict.get("sep", 0.0))
            n_batches += 1

        for key in epoch_stats:
            epoch_stats[key] /= max(n_batches, 1)

        val_res = evaluate_model(model, loaders["val"], device, K, classifier_head=classifier_head_for_eval)
        val_f1 = val_res["macro_f1"]
        train_history.append({"epoch": epoch, "val_macro_f1": val_f1, **{f"loss_{k}": v for k, v in epoch_stats.items()}})

        improved = val_f1 > best_val_f1
        if epoch == warmup_epochs + 1 or epoch % 10 == 0 or improved:
            print(f"  epoch {epoch:03d}/{max_epochs} | L={epoch_stats['total']:.4f} | val_F1={val_f1:.4f}{' <- best' if improved else ''}")

        if improved:
            best_val_f1 = val_f1
            patience_counter = 0
            _save_checkpoint(ckpt_path, model, epoch, val_f1, classifier_head_for_eval)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if not ckpt_path.exists():
        _save_checkpoint(ckpt_path, model, max_epochs, best_val_f1, classifier_head_for_eval)

    _load_checkpoint(ckpt_path, model, device, classifier_head_for_eval)
    test_res = evaluate_model(model, loaders["test"], device, K, classifier_head=classifier_head_for_eval, collect_internal=True)
    print(f"[{exp_name}] Test Macro F1 = {test_res['macro_f1']:.4f}")

    np.save(pred_dir / f"predictions_{exp_name}.npy", test_res["preds"])
    np.save(pred_dir / f"probabilities_{exp_name}.npy", test_res["probs"])
    np.save(pred_dir / f"embeddings_{exp_name}.npy", test_res["embeddings"])
    np.save(pred_dir / f"gate_values_{exp_name}.npy", test_res["gate_values"])
    np.save(pred_dir / f"proto_distances_{exp_name}.npy", test_res["proto_distances"])
    # Save learned prototypes for prototype-space plots and compactness tables.
    if hasattr(model, "P_fine"):
        np.save(pred_dir / f"prototypes_fine_{exp_name}.npy", model.P_fine.detach().cpu().numpy())
    if hasattr(model, "P_coarse"):
        np.save(pred_dir / f"prototypes_coarse_{exp_name}.npy", model.P_coarse.detach().cpu().numpy())
    np.save(pred_dir / "y_true.npy", test_res["labels"])

    with open(out_dir / f"history_{exp_name}.json", "w", encoding="utf-8") as f:
        json.dump({"warmup": warmup_history, "training": train_history}, f, indent=2)

    return {
        "exp_name": exp_name,
        "best_val_f1": float(best_val_f1),
        "test_macro_f1": float(test_res["macro_f1"]),
        "preds": test_res["preds"],
        "probs": test_res["probs"],
        "labels": test_res["labels"],
        "embeddings": test_res.get("embeddings"),
        "gate_values": test_res.get("gate_values"),
        "proto_distances": test_res.get("proto_distances"),
    }


def train_ecofusenet_v2(model: nn.Module,
                        loaders: dict,
                        data: dict,
                        phi: np.ndarray,
                        class_weights: torch.Tensor,
                        device: torch.device,
                        output_dir: str,
                        exp_name: str = "ecofusenet",
                        warmup_epochs: int = 35,
                        max_epochs: int = 220,
                        patience: int = 35,
                        lr_warmup: float = 1e-3,
                        lr_main: float = 8e-4,
                        weight_decay: float = 5e-4,
                        grad_clip: float = 1.0,
                        proto_weight: float = 0.03,
                        coarse_weight: float = 0.01,
                        sep_weight: float = 0.001,
                        sep_margin: float = 1.0,
                        label_smoothing: float = 0.03) -> dict:
    """
    Option-B CE-dominant EcoFuseNetV2 training.

    This is the recommended main neural training loop after the first full run
    showed that pure prototype-distance classification was weaker than CE.
    """
    out_dir = Path(output_dir)
    ckpt_dir = out_dir / "checkpoints"
    pred_dir = out_dir / "predictions"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    K, C = data["K"], data["C"]
    weights = class_weights.to(device)

    print(f"\n[{exp_name}] Warm-up CE phase ({warmup_epochs} epochs)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_warmup, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8
    )

    warmup_history = []
    best_warmup_f1 = -float("inf")
    for epoch in range(1, warmup_epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for X_b, y_fine_b, _ in loaders["train"]:
            X_b, y_fine_b = X_b.to(device), y_fine_b.to(device)
            _, e, _, _ = model(X_b)
            logits = model.classifier_head(e)
            loss = torch.nn.functional.cross_entropy(
                logits,
                y_fine_b,
                weight=weights,
                label_smoothing=label_smoothing,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            n_batches += 1

        val_res = evaluate_model(model, loaders["val"], device, K, classifier_head=None)
        val_f1 = val_res["macro_f1"]
        scheduler.step(val_f1)
        best_warmup_f1 = max(best_warmup_f1, val_f1)
        warmup_history.append({"epoch": epoch, "loss": total_loss / max(n_batches, 1), "val_macro_f1": val_f1})
        if epoch == 1 or epoch % 5 == 0 or epoch == warmup_epochs:
            print(f"  epoch {epoch:03d}/{warmup_epochs} | CE={total_loss/max(n_batches,1):.4f} | val_F1={val_f1:.4f}")

    print(f"[{exp_name}] Prototype initialization after CE warm-up")
    initialize_prototypes(model, loaders["train"], phi, K, C, device)

    print(f"[{exp_name}] CE + weak prototype regularization phase "
          f"(epochs {warmup_epochs+1}..{max_epochs}, patience={patience})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_main, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8
    )

    best_val_f1 = -float("inf")
    patience_counter = 0
    ckpt_path = ckpt_dir / f"{exp_name}_best.pt"
    train_history = []

    for epoch in range(warmup_epochs + 1, max_epochs + 1):
        model.train()
        epoch_stats = {"total": 0.0, "ce": 0.0, "proto": 0.0, "coarse": 0.0, "sep": 0.0}
        n_batches = 0

        for X_b, y_fine_b, y_coarse_b in loaders["train"]:
            X_b = X_b.to(device)
            y_fine_b = y_fine_b.to(device)
            y_coarse_b = y_coarse_b.to(device)

            _, e, _, _ = model(X_b)
            loss_dict = ce_with_weak_prototype_regularization(
                e=e,
                y_fine=y_fine_b,
                y_coarse=y_coarse_b,
                classifier_head=model.classifier_head,
                P_fine=model.P_fine,
                P_coarse=model.P_coarse,
                tau=model.tau,
                weights=weights,
                proto_weight=proto_weight,
                coarse_weight=coarse_weight,
                sep_weight=sep_weight,
                sep_margin=sep_margin,
                label_smoothing=label_smoothing,
            )
            loss = loss_dict["total"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            for key in epoch_stats:
                epoch_stats[key] += float(loss_dict.get(key, 0.0) if key != "total" else loss.detach().cpu())
            n_batches += 1

        for key in epoch_stats:
            epoch_stats[key] /= max(n_batches, 1)

        val_res = evaluate_model(model, loaders["val"], device, K, classifier_head=None)
        val_f1 = val_res["macro_f1"]
        scheduler.step(val_f1)

        improved = val_f1 > best_val_f1
        train_history.append({"epoch": epoch, "val_macro_f1": val_f1,
                              "lr": float(optimizer.param_groups[0]["lr"]),
                              **{f"loss_{k}": v for k, v in epoch_stats.items()}})
        if epoch == warmup_epochs + 1 or epoch % 10 == 0 or improved:
            print(f"  epoch {epoch:03d}/{max_epochs} | L={epoch_stats['total']:.4f} "
                  f"| CE={epoch_stats['ce']:.4f} | val_F1={val_f1:.4f}{' <- best' if improved else ''}")

        if improved:
            best_val_f1 = val_f1
            patience_counter = 0
            _save_checkpoint(ckpt_path, model, epoch, val_f1, classifier_head=None)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if not ckpt_path.exists():
        _save_checkpoint(ckpt_path, model, max_epochs, best_val_f1, classifier_head=None)

    _load_checkpoint(ckpt_path, model, device, classifier_head=None)
    test_res = evaluate_model(model, loaders["test"], device, K, classifier_head=None, collect_internal=True)
    print(f"[{exp_name}] Test Macro F1 = {test_res['macro_f1']:.4f}")

    np.save(pred_dir / f"predictions_{exp_name}.npy", test_res["preds"])
    np.save(pred_dir / f"probabilities_{exp_name}.npy", test_res["probs"])
    np.save(pred_dir / f"embeddings_{exp_name}.npy", test_res["embeddings"])
    np.save(pred_dir / f"gate_values_{exp_name}.npy", test_res["gate_values"])
    np.save(pred_dir / f"proto_distances_{exp_name}.npy", test_res["proto_distances"])
    np.save(pred_dir / f"prototypes_fine_{exp_name}.npy", model.P_fine.detach().cpu().numpy())
    np.save(pred_dir / f"prototypes_coarse_{exp_name}.npy", model.P_coarse.detach().cpu().numpy())
    np.save(pred_dir / "y_true.npy", test_res["labels"])

    with open(out_dir / f"history_{exp_name}.json", "w", encoding="utf-8") as f:
        json.dump({"warmup": warmup_history, "training": train_history}, f, indent=2)

    return {
        "exp_name": exp_name,
        "best_val_f1": float(best_val_f1),
        "test_macro_f1": float(test_res["macro_f1"]),
        "preds": test_res["preds"],
        "probs": test_res["probs"],
        "labels": test_res["labels"],
        "embeddings": test_res.get("embeddings"),
        "gate_values": test_res.get("gate_values"),
        "proto_distances": test_res.get("proto_distances"),
    }


def train_ablation_variant(model: nn.Module,
                           loaders: dict,
                           data: dict,
                           phi: np.ndarray,
                           class_weights: torch.Tensor,
                           device: torch.device,
                           output_dir: str,
                           exp_name: str,
                           use_proto_loss: bool = True,
                           warmup_epochs: int = 25,
                           max_epochs: int = 150,
                           patience: int = 20,
                           lr: float = 5e-4,
                           weight_decay: float = 1e-4,
                           grad_clip: float = 1.0,
                           lambda1: float = 0.3,
                           lambda2: float = 0.1,
                           sep_margin: float = 2.0) -> dict:
    """Train prototype-based ablation variants."""
    # For prototype ablations, reuse the full training function but with equal warm-up/proto LRs.
    return train_ecofusenet(
        model=model,
        loaders=loaders,
        data=data,
        phi=phi,
        class_weights=class_weights,
        device=device,
        output_dir=output_dir,
        exp_name=exp_name,
        warmup_epochs=warmup_epochs,
        max_epochs=max_epochs,
        patience=patience,
        lr_warmup=1e-3,
        lr_proto=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        lambda1=lambda1,
        lambda2=lambda2,
        use_proto_loss=use_proto_loss,
        sep_margin=sep_margin,
    )
