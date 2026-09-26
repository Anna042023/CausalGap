"""Paper-faithful PyTorch reimplementation of TF-MVGNN (Cheng et al., 2025)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import UnifiedModel, ModelOutput


def _row_norm(a: torch.Tensor) -> torch.Tensor:
    return a / a.sum(-1, keepdim=True).clamp_min(1e-8)


class TFMVGNNAdapter(UnifiedModel):
    model_name = 'TF-MVGNN'
    uses_graph = True
    model_protocol = 'v3.1-tfmvgnn-paper-reimplementation'
    representation_hook_name = 'final_DL-GNN_GRU_hidden_before_horizon_head'

    def __init__(self, num_nodes, history_len, pred_len, te_graph,
                 hidden_dim=64, k=3, dropout=0.0):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)
        self.hidden_dim = int(hidden_dim)
        self.k = int(k)
        self.register_buffer('te_graph', torch.as_tensor(te_graph, dtype=torch.float32), persistent=True)

        # Paper Eq.(4): A_ad = softmax(ReLU(E1 E2^T)).
        self.E1 = nn.Parameter(torch.empty(self.num_nodes, self.hidden_dim))
        self.E2 = nn.Parameter(torch.empty(self.num_nodes, self.hidden_dim))
        nn.init.xavier_uniform_(self.E1); nn.init.xavier_uniform_(self.E2)

        # Paper Eq.(5)-(8): independent weights for in/out/adaptive powers 0..K.
        self.w_in = nn.ModuleList([nn.Linear(1, self.hidden_dim, bias=False) for _ in range(self.k + 1)])
        self.w_out = nn.ModuleList([nn.Linear(1, self.hidden_dim, bias=False) for _ in range(self.k + 1)])
        self.w_ad = nn.ModuleList([nn.Linear(1, self.hidden_dim, bias=False) for _ in range(self.k + 1)])
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(self.hidden_dim, self.pred_len)

    def _graphs(self, adj, x):
        A = adj if torch.is_tensor(adj) else torch.as_tensor(adj)
        A = A.to(device=x.device, dtype=x.dtype)
        # Paper A_C = A_R ⊙ A_TE. CausalGap interventions modify A_R here.
        AC = torch.clamp(A, min=0.0) * self.te_graph.to(x.device, x.dtype)
        PI = _row_norm(AC)
        PO = _row_norm(AC.transpose(0, 1))
        Aad = F.softmax(F.relu(self.E1 @ self.E2.transpose(0, 1)), dim=-1).to(x.dtype)
        return PI, PO, Aad

    @staticmethod
    def _propagate_power(P, x, k):
        if k == 0:
            return x
        z = x
        for _ in range(k):
            z = torch.einsum('nm,bmc->bnc', P, z)
        return z

    def _dl_gnn(self, xt, PI, PO, Aad):
        z = 0.0
        for k in range(self.k + 1):
            zi = self._propagate_power(PI, xt, k)
            zo = self._propagate_power(PO, xt, k)
            za = self._propagate_power(Aad, xt, k)
            z = z + self.w_in[k](zi) + self.w_out[k](zo) + self.w_ad[k](za)
        return F.relu(z)

    def _forward(self, x, adj):
        PI, PO, Aad = self._graphs(adj, x)
        spatial = [self._dl_gnn(x[:, t], PI, PO, Aad) for t in range(x.size(1))]
        s = torch.stack(spatial, dim=1)                 # [B,T,N,D]
        B, T, N, D = s.shape
        seq = s.permute(0, 2, 1, 3).reshape(B * N, T, D)
        _, h = self.gru(seq)
        rep = h[-1].reshape(B, N, D)
        pred = self.head(self.dropout(rep)).permute(0, 2, 1).unsqueeze(-1)
        return ModelOutput(pred=pred, representation=rep)

    def training_forward(self, x, adj, y=None):
        return self._forward(x, adj)

    def inference_forward(self, x, adj, pred_len):
        if int(pred_len) != self.pred_len:
            raise ValueError((pred_len, self.pred_len))
        return self._forward(x, adj)
