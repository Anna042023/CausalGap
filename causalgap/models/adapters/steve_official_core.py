from __future__ import annotations

"""Released-STEVE-core traffic-only port for the unified benchmark.

This module ports the graph-temporal dual-branch prediction core from the
released STEVE/StableST implementation supplied with this benchmark. The
released repository additionally trains temporal/spatial/congestion auxiliary
classifiers and MI regularizers using labels that are not available under the
CausalGap common-input protocol. Those unavailable labels are deliberately not
fabricated here.

Key fidelity points retained from the released code:
  * two independent STGCN encoders (invariant / variant),
  * an adaptive 3-basis graph for the variant branch,
  * a non-trainable confounder bank built from variant representations,
  * MLP attention over the bank,
  * additive invariant prediction plus confounder-weighted variant prediction.

For reproducible paired clean/intervention evaluation, the bank is updated by
EMA only while the module is in training mode and is frozen in eval mode. The
released code mutates the bank in every forward call, which would otherwise
make DMIS depend on evaluation order.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalized_laplacian(graph: torch.Tensor) -> torch.Tensor:
    n = graph.size(0)
    eye = torch.eye(n, device=graph.device, dtype=graph.dtype)
    g = graph + eye
    d = torch.sum(g, dim=-1).clamp_min(1e-6).pow(-0.5)
    D = torch.diag(d)
    return eye - D @ g @ D


def _cheb(graph: torch.Tensor, k: int = 3) -> torch.Tensor:
    L = _normalized_laplacian(graph)
    n = L.size(0)
    out = [torch.eye(n, device=L.device, dtype=L.dtype)]
    if k == 1:
        return torch.stack(out)
    out.append(L)
    for _ in range(2, k):
        out.append(2 * L @ out[-1] - out[-2])
    return torch.stack(out)


class Align(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out
        self.proj = nn.Conv2d(c_in, c_out, 1) if c_in > c_out else None

    def forward(self, x):
        if self.proj is not None:
            return self.proj(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x


class TemporalConv(nn.Module):
    def __init__(self, kt: int, c_in: int, c_out: int, glu: bool = False):
        super().__init__()
        self.kt, self.c_out, self.glu = kt, c_out, glu
        self.align = Align(c_in, c_out)
        self.conv = nn.Conv2d(c_in, c_out * (2 if glu else 1), (kt, 1))

    def forward(self, x):
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        y = self.conv(x)
        if self.glu:
            p, q = y[:, :self.c_out], y[:, self.c_out:]
            return (p + x_in) * torch.sigmoid(q)
        return torch.relu(y + x_in)


class SpatialConv(nn.Module):
    def __init__(self, ks: int, c_in: int, c_out: int):
        super().__init__()
        self.theta = nn.Parameter(torch.empty(c_in, c_out, ks))
        self.bias = nn.Parameter(torch.empty(1, c_out, 1, 1))
        self.align = Align(c_in, c_out)
        nn.init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x, Lk):
        # Lk [K,N,N], x [B,C,T,N]
        xc = torch.einsum('knm,bitm->bitkn', Lk, x)
        y = torch.einsum('iok,bitkn->botn', self.theta, xc) + self.bias
        return torch.relu(y + self.align(x))


class STBlock(nn.Module):
    def __init__(self, n: int, c: int, ks: int = 3, kt: int = 3, dropout: float = 0.1):
        super().__init__()
        mid = max(1, c // 2)
        self.t1 = TemporalConv(kt, c, mid, glu=True)
        self.s = SpatialConv(ks, mid, mid)
        self.t2 = TemporalConv(kt, mid, c, glu=False)
        self.norm = nn.LayerNorm([n, c])
        self.drop = nn.Dropout(dropout)

    def forward(self, x, Lk):
        x = self.t2(self.s(self.t1(x), Lk))
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.drop(x)


class STEncoder(nn.Module):
    def __init__(self, n: int, d_model: int, history_len: int, dropout: float):
        super().__init__()
        if history_len <= 8:
            raise ValueError('STEVE released STGCN core requires history_len > 8.')
        self.in_proj = nn.Conv2d(1, d_model, 1)
        self.b1 = STBlock(n, d_model, 3, 3, dropout)
        self.b2 = STBlock(n, d_model, 3, 3, dropout)

    def forward(self, x, Lk):
        return self.b2(self.b1(self.in_proj(x), Lk), Lk)


class MLPAttention(nn.Module):
    def __init__(self, d_model: int, tau: float = 4.0):
        super().__init__()
        self.tau = float(tau)
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

    def forward(self, q, k, v):
        # q [B,N,D], k/v [B,K,D]
        B, N, D = q.shape
        K = k.size(1)
        qq = q.unsqueeze(2).expand(B, N, K, D)
        kk = k.unsqueeze(1).expand(B, N, K, D)
        score = self.mlp(torch.cat([qq, kk], dim=-1)).squeeze(-1)
        att = torch.softmax(score / self.tau, dim=-1)
        return torch.bmm(att, v), att


class ReleasedSTEVECore(nn.Module):
    def __init__(self, num_nodes: int, history_len: int, pred_len: int,
                 d_model: int = 64, dropout: float = 0.1,
                 bank_multiplier: float = 2.0, bank_gamma: float = 0.5):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.d_model = int(d_model)
        self.bank_gamma = float(bank_gamma)
        self.temporal_out = self.history_len - 8

        self.invariant_encoder = STEncoder(self.num_nodes, self.d_model, self.history_len, dropout)
        self.variant_encoder = STEncoder(self.num_nodes, self.d_model, self.history_len, dropout)

        # Released STEVE uses three learned adjacency bases.
        self.node_embeddings_1 = nn.Parameter(torch.randn(3, self.num_nodes, self.d_model))
        self.node_embeddings_2 = nn.Parameter(torch.randn(3, self.d_model, self.num_nodes))

        K = max(1, int(round(self.d_model * float(bank_multiplier))))
        bank = torch.randn(K, self.d_model)
        bank = (bank - bank.mean(0, keepdim=True)) / bank.std(0, keepdim=True).clamp_min(1e-5)
        self.register_buffer('Bank', bank)
        self.bank_projector = nn.Linear(self.temporal_out * self.num_nodes, K)
        self.bank_attention = MLPAttention(self.d_model, tau=4.0)

        self.inv_time_collapse = nn.Conv2d(self.temporal_out, 1, 1)
        self.var_time_collapse = nn.Conv2d(self.temporal_out, 1, 1)
        self.inv_head = nn.Linear(self.d_model, self.pred_len)
        self.var_head = nn.Linear(self.d_model, self.pred_len)
        self.weight_head = nn.Linear(self.d_model, self.pred_len)

    def _adaptive_cheb(self):
        # [3,N,N], preserving released code's 3-basis adaptive graph semantics.
        return torch.softmax(torch.relu(torch.bmm(self.node_embeddings_1, self.node_embeddings_2)), dim=1)

    def _confounder(self, z):
        # z [B,T,N,D]
        B, T, N, D = z.shape
        flat = z.reshape(B, T * N, D).permute(0, 2, 1)  # [B,D,TN]
        proposal = self.bank_projector(flat).permute(0, 2, 1)  # [B,K,D]
        if self.training:
            with torch.no_grad():
                # Order-independent batch-mean EMA for deterministic training/eval separation.
                batch_bank = proposal.mean(dim=0)
                self.Bank.mul_(self.bank_gamma).add_(batch_bank * (1.0 - self.bank_gamma))
        bank = self.Bank.unsqueeze(0).expand(B, -1, -1)
        q = z.mean(dim=1)
        c, att = self.bank_attention(q, bank, bank)
        return c, att

    def forward(self, x, adj):
        # unified x [B,L,N,1] -> released STGCN NCLV [B,1,L,N]
        x4 = x.permute(0, 3, 1, 2)
        ext = _cheb(adj, 3)
        h_raw = self.invariant_encoder(x4, ext)  # [B,D,T,N]
        z_raw = self.variant_encoder(x4, self._adaptive_cheb())
        h = h_raw.permute(0, 2, 3, 1)  # [B,T,N,D]
        z = z_raw.permute(0, 2, 3, 1)
        c, att = self._confounder(z)

        inv_feat = self.inv_time_collapse(h).squeeze(1)  # [B,N,D]
        var_feat = self.var_time_collapse(z).squeeze(1) + c
        y_h = self.inv_head(inv_feat)
        y_c = self.var_head(var_feat)
        c_weight = torch.relu(self.weight_head(c))
        y = y_h + c_weight * y_c  # released STEVE additive invariant + weighted variant form
        pred = y.permute(0, 2, 1).unsqueeze(-1)  # [B,H,N,1]
        return pred, inv_feat, {'variant': var_feat, 'attention': att}
