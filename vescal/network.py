"""
EcoFuseNet model definitions.

Fixes included in this updated version:
- Safe positive temperature: tau = 0.1 + softplus(raw_tau).
- Optional scaled CHM residual alpha_chm instead of unbounded additive CHM residual.
- All ablation models expose d_e, P_fine, P_coarse and tau consistently.
- FlatLossEcoFuseNet includes a real linear classifier head for standard CE ablation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _raw_tau_from_init(tau_init: float, tau_min: float = 0.1) -> torch.Tensor:
    """Inverse-softplus initialization for tau = tau_min + softplus(raw_tau)."""
    value = max(float(tau_init) - tau_min, 1e-4)
    return torch.tensor(math.log(math.expm1(value)), dtype=torch.float32)


class TemperatureMixin:
    tau_min: float = 0.1

    @property
    def tau(self) -> torch.Tensor:
        return self.tau_min + F.softplus(self.raw_tau)


class TerrainEncoder(nn.Module):
    def __init__(self, d_f: int = 32, dropout: float = 0.15):
        super().__init__()
        self.fc = nn.Linear(5, d_f)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_T: torch.Tensor) -> torch.Tensor:
        return self.drop(F.gelu(self.fc(x_T)))


class CHMEncoder(nn.Module):
    """CHM branch: deterministic scalar structural signal, no dropout."""
    def __init__(self, d_c: int = 8, d_f: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(1, d_c)
        self.proj = nn.Linear(d_c, d_f)

    def forward(self, x_C: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc1(x_C)))


class SpectralEncoder(nn.Module):
    def __init__(self, d_s: int = 16, d_f: int = 32, dropout: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(2, d_s)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(d_s, d_f)

    def forward(self, x_S: torch.Tensor) -> torch.Tensor:
        return self.proj(self.drop(F.gelu(self.fc1(x_S))))


class EcoFuseNet(nn.Module, TemperatureMixin):
    """Full EcoFuseNet with CHM-conditioned cross-modal gated fusion."""
    def __init__(self, d_f: int = 32, d_e: int = 64, K: int = 29,
                 C: int = 5, tau_init: float = 1.0,
                 learnable_chm_residual: bool = True):
        super().__init__()
        self.d_f = d_f
        self.d_e = d_e
        self.K = K
        self.C = C

        self.terrain_enc = TerrainEncoder(d_f=d_f, dropout=0.15)
        self.chm_enc = CHMEncoder(d_c=8, d_f=d_f)
        self.spectral_enc = SpectralEncoder(d_s=16, d_f=d_f, dropout=0.15)

        self.gate_fc = nn.Linear(d_f * 3, d_f)
        self.proto_head = nn.Linear(d_f, d_e)

        self.P_fine = nn.Parameter(torch.randn(K, d_e) * 0.01)
        self.P_coarse = nn.Parameter(torch.randn(C, d_e) * 0.01)
        self.raw_tau = nn.Parameter(_raw_tau_from_init(tau_init))

        # CHM is very sparse in this dataset, so this residual is scaled.
        # alpha starts near 0.1 and is learned if enabled.
        if learnable_chm_residual:
            self.alpha_chm = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        else:
            self.register_buffer("alpha_chm", torch.tensor(0.1, dtype=torch.float32))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor):
        x_T = x[:, :5]
        x_C = x[:, 5:6]
        x_S = x[:, 6:8]
        h_T = self.terrain_enc(x_T)
        h_C = self.chm_enc(x_C)
        h_S = self.spectral_enc(x_S)
        return h_T, h_C, h_S

    def fuse(self, h_T: torch.Tensor, h_C: torch.Tensor, h_S: torch.Tensor):
        cat = torch.cat([h_C, h_T, h_S], dim=1)
        g = torch.sigmoid(self.gate_fc(cat))
        z = g * h_T + (1.0 - g) * h_S + self.alpha_chm * h_C
        return z, g

    def embed(self, z: torch.Tensor) -> torch.Tensor:
        return self.proto_head(z)

    def classify(self, e: torch.Tensor):
        d = torch.cdist(e, self.P_fine).pow(2)
        logits = -d / self.tau
        logits = logits - logits.max(dim=1, keepdim=True).values
        probs = F.softmax(logits, dim=1)
        return probs, d

    def forward(self, x: torch.Tensor):
        h_T, h_C, h_S = self.encode(x)
        z, g = self.fuse(h_T, h_C, h_S)
        e = self.embed(z)
        probs, d = self.classify(e)
        return probs, e, g, d


# ─────────────────────────────────────────────────────────────────────────────
# Option-B upgraded model: CE-dominant EcoFuseNet with weak prototype regularizers
# ─────────────────────────────────────────────────────────────────────────────

class MLPBlock(nn.Module):
    """Linear → LayerNorm → GELU → Dropout block for small tabular data."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EcoFuseNetV2(nn.Module, TemperatureMixin):
    """
    Option-B main model.

    Key change from the prototype-only model:
    - Final class prediction uses a CE classifier head, which is more stable for
      this small tabular dataset.
    - Fine/coarse prototypes remain learnable and are used as WEAK auxiliary
      geometric regularizers, not as the only classifier.
    - A deeper normalized fusion network is used to close the gap with tree models.
    """
    def __init__(self,
                 d_f: int = 64,
                 d_e: int = 128,
                 K: int = 29,
                 C: int = 5,
                 tau_init: float = 1.0,
                 dropout: float = 0.10,
                 alpha_chm_init: float = 0.05):
        super().__init__()
        self.d_f = d_f
        self.d_e = d_e
        self.K = K
        self.C = C

        self.terrain_enc = nn.Sequential(
            MLPBlock(5, d_f, dropout),
            MLPBlock(d_f, d_f, dropout),
        )

        # CHM is sparse; use low dropout-free capacity and small residual scale.
        self.chm_enc = nn.Sequential(
            nn.Linear(1, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, d_f),
            nn.LayerNorm(d_f),
        )

        self.spectral_enc = nn.Sequential(
            MLPBlock(2, 32, dropout),
            MLPBlock(32, d_f, dropout),
        )

        self.gate_net = nn.Sequential(
            nn.Linear(d_f * 3, d_f),
            nn.LayerNorm(d_f),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_f, d_f),
        )

        self.fusion_norm = nn.LayerNorm(d_f)
        self.fusion_refine = nn.Sequential(
            MLPBlock(d_f, d_f, dropout),
            nn.Linear(d_f, d_f),
        )

        self.proto_head = nn.Sequential(
            nn.Linear(d_f, d_e),
            nn.LayerNorm(d_e),
        )

        self.classifier_head = nn.Linear(d_e, K)

        self.P_fine = nn.Parameter(torch.randn(K, d_e) * 0.01)
        self.P_coarse = nn.Parameter(torch.randn(C, d_e) * 0.01)
        self.raw_tau = nn.Parameter(_raw_tau_from_init(tau_init))

        # Learned scalar; initialized small because CHM is mostly zero.
        self.alpha_chm = nn.Parameter(torch.tensor(float(alpha_chm_init), dtype=torch.float32))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor):
        h_T = self.terrain_enc(x[:, :5])
        h_C = self.chm_enc(x[:, 5:6])
        h_S = self.spectral_enc(x[:, 6:8])
        return h_T, h_C, h_S

    def fuse(self, h_T: torch.Tensor, h_C: torch.Tensor, h_S: torch.Tensor):
        g = torch.sigmoid(self.gate_net(torch.cat([h_C, h_T, h_S], dim=1)))
        z0 = g * h_T + (1.0 - g) * h_S + self.alpha_chm * h_C
        z0 = self.fusion_norm(z0)
        z = self.fusion_norm(z0 + self.fusion_refine(z0))
        return z, g

    def embed(self, z: torch.Tensor) -> torch.Tensor:
        return self.proto_head(z)

    def classify_proto(self, e: torch.Tensor):
        d = torch.cdist(e, self.P_fine).pow(2)
        logits = -d / self.tau
        logits = logits - logits.max(dim=1, keepdim=True).values
        return torch.softmax(logits, dim=1), d

    def forward(self, x: torch.Tensor):
        h_T, h_C, h_S = self.encode(x)
        z, g = self.fuse(h_T, h_C, h_S)
        e = self.embed(z)
        logits = self.classifier_head(e)
        probs = torch.softmax(logits, dim=1)
        _, d = self.classify_proto(e)
        return probs, e, g, d


class FlatLossEcoFuseNet(EcoFuseNet):
    """Variant B: same gate/encoder architecture, but trained with a real CE head."""
    def __init__(self, d_f: int = 32, d_e: int = 64, K: int = 29,
                 C: int = 5, tau_init: float = 1.0):
        super().__init__(d_f=d_f, d_e=d_e, K=K, C=C, tau_init=tau_init)
        self.classifier_head = nn.Linear(d_e, K)

    def flat_logits(self, e: torch.Tensor) -> torch.Tensor:
        return self.classifier_head(e)


class NoGateMLP(nn.Module, TemperatureMixin):
    """Variant A: flat concatenation of terrain, spectral and CHM embeddings."""
    def __init__(self, d_f: int = 32, d_e: int = 64, K: int = 29,
                 C: int = 5, tau_init: float = 1.0):
        super().__init__()
        self.d_f = d_f
        self.d_e = d_e
        self.K = K
        self.C = C
        self.terrain_enc = TerrainEncoder(d_f=d_f, dropout=0.15)
        self.chm_enc = CHMEncoder(d_c=8, d_f=d_f)
        self.spectral_enc = SpectralEncoder(d_s=16, d_f=d_f, dropout=0.15)
        self.proto_head = nn.Linear(d_f * 3, d_e)
        self.P_fine = nn.Parameter(torch.randn(K, d_e) * 0.01)
        self.P_coarse = nn.Parameter(torch.randn(C, d_e) * 0.01)
        self.raw_tau = nn.Parameter(_raw_tau_from_init(tau_init))

    def forward(self, x: torch.Tensor):
        h_T = self.terrain_enc(x[:, :5])
        h_C = self.chm_enc(x[:, 5:6])
        h_S = self.spectral_enc(x[:, 6:8])
        z = torch.cat([h_T, h_S, h_C], dim=1)
        e = self.proto_head(z)
        d = torch.cdist(e, self.P_fine).pow(2)
        logits = -d / self.tau
        logits = logits - logits.max(dim=1, keepdim=True).values
        probs = F.softmax(logits, dim=1)
        g = torch.full((x.size(0), self.d_f), 0.5, device=x.device)
        return probs, e, g, d


class LiDAROnlyNet(nn.Module, TemperatureMixin):
    """Variant C: terrain + CHM only, no spectral branch."""
    def __init__(self, d_f: int = 32, d_e: int = 64, K: int = 29,
                 C: int = 5, tau_init: float = 1.0):
        super().__init__()
        self.d_f = d_f
        self.d_e = d_e
        self.K = K
        self.C = C
        self.fc1 = nn.Linear(6, d_f)
        self.drop = nn.Dropout(0.15)
        self.proto_head = nn.Linear(d_f, d_e)
        self.P_fine = nn.Parameter(torch.randn(K, d_e) * 0.01)
        self.P_coarse = nn.Parameter(torch.randn(C, d_e) * 0.01)
        self.raw_tau = nn.Parameter(_raw_tau_from_init(tau_init))

    def forward(self, x: torch.Tensor):
        h = self.drop(F.gelu(self.fc1(x[:, :6])))
        e = self.proto_head(h)
        d = torch.cdist(e, self.P_fine).pow(2)
        logits = -d / self.tau
        logits = logits - logits.max(dim=1, keepdim=True).values
        probs = F.softmax(logits, dim=1)
        g = torch.full((x.size(0), self.d_f), 1.0, device=x.device)
        return probs, e, g, d


class SpectralOnlyNet(nn.Module, TemperatureMixin):
    """Variant D: NDVI + NDWI only."""
    def __init__(self, d_f: int = 32, d_e: int = 64, K: int = 29,
                 C: int = 5, tau_init: float = 1.0):
        super().__init__()
        self.d_f = d_f
        self.d_e = d_e
        self.K = K
        self.C = C
        self.fc1 = nn.Linear(2, d_f)
        self.drop = nn.Dropout(0.15)
        self.proto_head = nn.Linear(d_f, d_e)
        self.P_fine = nn.Parameter(torch.randn(K, d_e) * 0.01)
        self.P_coarse = nn.Parameter(torch.randn(C, d_e) * 0.01)
        self.raw_tau = nn.Parameter(_raw_tau_from_init(tau_init))

    def forward(self, x: torch.Tensor):
        h = self.drop(F.gelu(self.fc1(x[:, 6:8])))
        e = self.proto_head(h)
        d = torch.cdist(e, self.P_fine).pow(2)
        logits = -d / self.tau
        logits = logits - logits.max(dim=1, keepdim=True).values
        probs = F.softmax(logits, dim=1)
        g = torch.full((x.size(0), self.d_f), 0.0, device=x.device)
        return probs, e, g, d


class NoCHMGateNet(nn.Module, TemperatureMixin):
    """Variant E: CHM is merged into terrain branch as a plain feature."""
    def __init__(self, d_f: int = 32, d_e: int = 64, K: int = 29,
                 C: int = 5, tau_init: float = 1.0):
        super().__init__()
        self.d_f = d_f
        self.d_e = d_e
        self.K = K
        self.C = C
        self.terrain_fc = nn.Linear(6, d_f)
        self.terrain_drop = nn.Dropout(0.15)
        self.spectral_fc1 = nn.Linear(2, 16)
        self.spectral_drop = nn.Dropout(0.15)
        self.spectral_proj = nn.Linear(16, d_f)
        self.gate_fc = nn.Linear(d_f * 2, d_f)
        self.proto_head = nn.Linear(d_f, d_e)
        self.P_fine = nn.Parameter(torch.randn(K, d_e) * 0.01)
        self.P_coarse = nn.Parameter(torch.randn(C, d_e) * 0.01)
        self.raw_tau = nn.Parameter(_raw_tau_from_init(tau_init))

    def forward(self, x: torch.Tensor):
        h_T = self.terrain_drop(F.gelu(self.terrain_fc(x[:, :6])))
        h_S = self.spectral_proj(self.spectral_drop(F.gelu(self.spectral_fc1(x[:, 6:8]))))
        g = torch.sigmoid(self.gate_fc(torch.cat([h_T, h_S], dim=1)))
        z = g * h_T + (1.0 - g) * h_S
        e = self.proto_head(z)
        d = torch.cdist(e, self.P_fine).pow(2)
        logits = -d / self.tau
        logits = logits - logits.max(dim=1, keepdim=True).values
        probs = F.softmax(logits, dim=1)
        return probs, e, g, d
