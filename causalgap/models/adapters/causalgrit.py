"""Paper-faithful scalable PyTorch reimplementation of CausalGRIT (Liu et al., 2025).

The publication does not provide source code and leaves several numerical
hyperparameters unspecified.  This implementation follows Eqs. (10)-(25) and
uses an auditable training-only RF-CATE prior constructed in paper_graphs.py.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import UnifiedModel, ModelOutput


def _row_norm(a: torch.Tensor) -> torch.Tensor:
    return a / a.sum(-1, keepdim=True).clamp_min(1e-8)


class PlanarFlow(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.u = nn.Parameter(torch.empty(d))
        self.w = nn.Parameter(torch.empty(d))
        self.b = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.u, std=0.02); nn.init.normal_(self.w, std=0.02)
    def forward(self, z):
        # Paper Eq.(18), intentionally without an extra u-hat correction not stated in the paper.
        return z + self.u * torch.tanh(torch.einsum('...d,d->...', z, self.w) + self.b).unsqueeze(-1)


class CausalGRITAdapter(UnifiedModel):
    model_name = 'CausalGRIT'
    uses_graph = True
    model_protocol = 'v3.3-causalgrit-paper-faithful-scalable-reimplementation-masked-targets'
    representation_hook_name = 'final_gated_belief_GRU_state_before_prediction_head'

    def __init__(self, num_nodes, history_len, pred_len, causal_prior, baseline_adj,
                 d_model=64, flow_steps=4, refine_rank=16, dropout=0.1,
                 egc_eps=0.10, kl_beta_max=1e-3, kl_steepness=10.0,
                 lambda_main=1.0, lambda_consistency=0.1, graph_reg_weight=1e-4):
        super().__init__()
        self.num_nodes = int(num_nodes); self.history_len = int(history_len); self.pred_len = int(pred_len)
        self.d_model = int(d_model); self.flow_steps = int(flow_steps)
        self.egc_eps = float(egc_eps); self.kl_beta_max = float(kl_beta_max); self.kl_steepness = float(kl_steepness)
        self.lambda_main = float(lambda_main); self.lambda_consistency = float(lambda_consistency)
        self.graph_reg_weight = float(graph_reg_weight)
        self._epoch = 1; self._max_epochs = 1
        self.register_buffer('causal_prior', torch.as_tensor(causal_prior, dtype=torch.float32), persistent=True)
        self.register_buffer('baseline_adj', torch.as_tensor(baseline_adj, dtype=torch.float32), persistent=True)

        # Eq.(10)-(12). Full N->N MLP dimensions are not specified in the paper;
        # a low-rank factorization is used to make the same N x N mask scalable to N=1024.
        r = int(min(refine_rank, d_model))
        self.refine_node = nn.Sequential(nn.Linear(1, d_model), nn.ReLU(), nn.Linear(d_model, 2 * r))
        self.base_mask_logits = nn.Parameter(torch.full((self.num_nodes, self.num_nodes), 2.0))

        # EGC Eq.(14)-(15): two-layer input-conditioned residual, also low-rank for scalability.
        self.egc_node = nn.Sequential(nn.Linear(1, d_model), nn.ReLU(), nn.Linear(d_model, 2 * r))

        # Belief encoder Eq.(16)-(18).
        self.belief_base = nn.Sequential(nn.Linear(1, d_model), nn.ReLU(), nn.Linear(d_model, 2 * d_model))
        self.flows = nn.ModuleList([PlanarFlow(d_model) for _ in range(self.flow_steps)])
        # Temporal belief gate Eq.(19).
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.gru = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(d_model, self.pred_len)

    def set_training_progress(self, epoch: int, max_epochs: int):
        self._epoch = max(1, int(epoch)); self._max_epochs = max(1, int(max_epochs))

    def _beta(self):
        p = self._epoch / max(1.0, float(self._max_epochs))
        return self.kl_beta_max / (1.0 + math.exp(-self.kl_steepness * (p - 0.5)))

    def _physical_factor(self, adj, x):
        A = adj if torch.is_tensor(adj) else torch.as_tensor(adj)
        A = A.to(x.device, x.dtype).clamp_min(0.0)
        A0 = self.baseline_adj.to(x.device, x.dtype).clamp_min(0.0)
        # Preserve paper causal prior on baseline support while making CausalGap
        # edge removals/weight changes observable through the native Track-A path.
        ratio = torch.where(A0 > 1e-12, A / A0.clamp_min(1e-12), torch.zeros_like(A))
        return ratio.clamp(0.0, 2.0)

    def _low_rank_mask(self, xt, module):
        e = module(xt)                       # [B,N,2r]
        q, k = torch.chunk(e, 2, dim=-1)
        score = torch.einsum('bnr,bmr->bnm', q, k) / math.sqrt(max(1, q.size(-1)))
        return torch.sigmoid(score)

    def _refined_graph(self, xt, adj):
        prior = self.causal_prior.to(xt.device, xt.dtype) * self._physical_factor(adj, xt)
        prior = _row_norm(prior)
        R = self._low_rank_mask(xt, self.refine_node)
        base = torch.sigmoid(self.base_mask_logits).to(xt.dtype)
        G = prior.unsqueeze(0) * base.unsqueeze(0) * R
        return _row_norm(G), prior

    def _counterfactual_graph(self, G, xt):
        e = self.egc_node(xt)
        q, k = torch.chunk(e, 2, dim=-1)
        delta = self.egc_eps * torch.tanh(torch.einsum('bnr,bmr->bnm', q, k) / math.sqrt(max(1, q.size(-1))))
        return torch.clamp(G + delta * (G > 0).to(G.dtype), 0.0, 1.0)

    def _belief(self, agg):
        pars = self.belief_base(agg)
        mu, logvar = torch.chunk(pars, 2, dim=-1)
        logvar = logvar.clamp(-8.0, 8.0)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        for flow in self.flows:
            z = flow(z)
        return z, mu, logvar

    def _encode(self, x, adj, counterfactual=False):
        beliefs, mus, logvars = [], [], []
        graph_dev = x.new_tensor(0.0)
        for t in range(x.size(1)):
            xt = x[:, t]                     # [B,N,1]
            G, prior = self._refined_graph(xt, adj)
            graph_dev = graph_dev + torch.mean(torch.abs(G - prior.unsqueeze(0)))
            if counterfactual:
                G = self._counterfactual_graph(G, xt)
            agg = torch.einsum('bnm,bmc->bnc', G, xt)
            z, mu, lv = self._belief(agg)
            alpha = self.gate(z).sigmoid()
            beliefs.append(alpha * z); mus.append(mu); logvars.append(lv)
        Z = torch.stack(beliefs, dim=1)       # [B,T,N,D]
        MU = torch.stack(mus, dim=1); LV = torch.stack(logvars, dim=1)
        B, T, N, D = Z.shape
        seq = Z.permute(0, 2, 1, 3).reshape(B * N, T, D)
        _, h = self.gru(seq)
        rep = h[-1].reshape(B, N, D)
        pred = self.head(self.dropout(rep)).permute(0, 2, 1).unsqueeze(-1)
        pred = pred + x[:, -1:].expand(-1, self.pred_len, -1, -1)  # Eq.(21) residual x_last
        kl = -0.5 * torch.mean(1.0 + LV - MU.pow(2) - LV.exp())
        return pred, rep, kl, graph_dev / max(1, x.size(1))

    @staticmethod
    def _masked_mse(pred, target, target_mask=None):
        if target_mask is None:
            return F.mse_loss(pred, target)
        m = target_mask.to(device=pred.device, dtype=torch.bool)
        if m.shape != target.shape:
            raise ValueError(f'CausalGRIT target mask shape {m.shape} != target {target.shape}')
        m = m & torch.isfinite(target) & torch.isfinite(pred)
        if not torch.any(m):
            raise RuntimeError('CausalGRIT batch contains no valid targets after masking.')
        return torch.mean((pred[m] - target[m]) ** 2)

    def _training_forward_impl(self, x, adj, y=None, target_mask=None):
        pred, rep, kl, greg = self._encode(x, adj, counterfactual=False)
        if y is None:
            return ModelOutput(pred=pred, representation=rep)
        target = y[:, :self.pred_len]
        if target_mask is not None:
            target_mask = target_mask[:, :self.pred_len]
        pred_cf, _, kl_cf, greg_cf = self._encode(x, adj, counterfactual=True)
        beta = float(self._beta())
        mse_clean = self._masked_mse(pred, target, target_mask)
        mse_cf = self._masked_mse(pred_cf, target, target_mask)
        main = mse_clean + beta * kl
        main_cf = mse_cf + beta * kl_cf
        consistency = F.mse_loss(pred, pred_cf)
        graph_reg = 0.5 * (greg + greg_cf)
        total = self.lambda_main * (main + main_cf) + self.lambda_consistency * consistency + self.graph_reg_weight * graph_reg
        terms = {
            'mse_clean': float(mse_clean.detach().item()),
            'mse_cf': float(mse_cf.detach().item()),
            'kl': float(kl.detach().item()), 'kl_beta': float(beta),
            'consistency': float(consistency.detach().item()), 'graph_reg': float(graph_reg.detach().item()),
        }
        return ModelOutput(pred=pred, representation=rep, loss=total, loss_terms=terms)

    def training_forward(self, x, adj, y=None):
        return self._training_forward_impl(x, adj, y, target_mask=None)

    def training_forward_with_mask(self, x, adj, y, target_mask):
        return self._training_forward_impl(x, adj, y, target_mask=target_mask)

    def inference_forward(self, x, adj, pred_len):
        if int(pred_len) != self.pred_len:
            raise ValueError((pred_len, self.pred_len))
        was = self.training
        self.eval()
        pred, rep, _, _ = self._encode(x, adj, counterfactual=False)
        self.train(was)
        return ModelOutput(pred=pred, representation=rep)
