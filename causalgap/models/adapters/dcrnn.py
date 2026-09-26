import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import UnifiedModel, ModelOutput


def random_walk(adj):
    a = np.asarray(adj, dtype=np.float32)
    d = a.sum(axis=1)
    out = np.zeros_like(a)
    nz = d > 1e-8
    out[nz] = a[nz] / d[nz, None]
    return out


class DiffusionGraphLinear(nn.Module):
    def __init__(self, in_dim, out_dim, max_diffusion_step=2, support_count=2, bias=0.0):
        super().__init__()
        self.k = int(max_diffusion_step)
        self.support_count = int(support_count)
        m = 1 + self.k * self.support_count
        self.linear = nn.Linear(in_dim * m, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, bias)

    def forward(self, x, supports):
        # x: [B,N,C]
        feats = [x]
        for A in supports:
            x0 = x
            x1 = torch.einsum('nm,bmc->bnc', A, x0)
            feats.append(x1)
            for _ in range(2, self.k + 1):
                x2 = 2.0 * torch.einsum('nm,bmc->bnc', A, x1) - x0
                feats.append(x2)
                x0, x1 = x1, x2
        return self.linear(torch.cat(feats, dim=-1))


class DCGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_diffusion_step=2):
        super().__init__()
        total = input_dim + hidden_dim
        self.hidden_dim = hidden_dim
        self.gates = DiffusionGraphLinear(total, 2*hidden_dim, max_diffusion_step, 2, bias=1.0)
        self.cand = DiffusionGraphLinear(total, hidden_dim, max_diffusion_step, 2, bias=0.0)

    def forward(self, x, h, supports):
        z_r = torch.sigmoid(self.gates(torch.cat([x,h], dim=-1), supports))
        r, u = torch.chunk(z_r, 2, dim=-1)
        c = torch.tanh(self.cand(torch.cat([x, r*h], dim=-1), supports))
        return u*h + (1-u)*c


class DCRNNAdapter(UnifiedModel):
    model_name = 'DCRNN'
    uses_graph = True
    model_protocol = 'v3.0-dcrnn-pytorch-faithful-port'
    representation_hook_name = 'final_decoder_hidden_state'

    def __init__(self, num_nodes, history_len, pred_len, hidden_dim=64, num_layers=2,
                 max_diffusion_step=2, curriculum_decay=1000):
        super().__init__()
        self.num_nodes=num_nodes; self.history_len=history_len; self.pred_len=pred_len
        self.hidden_dim=hidden_dim; self.num_layers=num_layers
        self.curriculum_decay=float(curriculum_decay)
        self.encoder = nn.ModuleList([
            DCGRUCell(1 if i==0 else hidden_dim, hidden_dim, max_diffusion_step)
            for i in range(num_layers)
        ])
        self.decoder = nn.ModuleList([
            DCGRUCell(1 if i==0 else hidden_dim, hidden_dim, max_diffusion_step)
            for i in range(num_layers)
        ])
        self.proj = nn.Linear(hidden_dim, 1)
        self.register_buffer('_global_step', torch.zeros((), dtype=torch.long), persistent=True)

    def _supports(self, adj, device):
        a = adj.detach().cpu().numpy() if torch.is_tensor(adj) else np.asarray(adj)
        f = torch.from_numpy(random_walk(a).T.copy()).float().to(device)
        b = torch.from_numpy(random_walk(a.T).T.copy()).float().to(device)
        return [f,b]

    def _encode(self, x, supports):
        B,N=x.size(0),x.size(2)
        hs=[torch.zeros(B,N,self.hidden_dim,device=x.device,dtype=x.dtype) for _ in range(self.num_layers)]
        for t in range(x.size(1)):
            z=x[:,t]
            for i,cell in enumerate(self.encoder):
                hs[i]=cell(z,hs[i],supports); z=hs[i]
        return hs

    def _teacher_ratio(self):
        k=self.curriculum_decay
        s=float(self._global_step.item())
        return k/(k+np.exp(s/k))

    def _run(self,x,adj,y=None,training=False):
        supports=self._supports(adj,x.device)
        hs=self._encode(x,supports)
        B,N=x.size(0),x.size(2)
        inp=torch.zeros(B,N,1,device=x.device,dtype=x.dtype)
        preds=[]; reps=[]
        ratio=self._teacher_ratio() if training and y is not None else 0.0
        for t in range(self.pred_len):
            z=inp
            for i,cell in enumerate(self.decoder):
                hs[i]=cell(z,hs[i],supports); z=hs[i]
            reps.append(z)
            p=self.proj(z); preds.append(p)
            if training and y is not None and t < y.size(1):
                use_teacher = torch.rand((),device=x.device).item() < ratio
                inp = y[:,t] if use_teacher else p
            else:
                inp=p
        if training: self._global_step += 1
        pred=torch.stack(preds,dim=1)
        rep=torch.stack(reps,dim=1).mean(dim=1)
        return ModelOutput(pred,rep)

    def training_forward(self,x,adj,y=None): return self._run(x,adj,y,True)
    def inference_forward(self,x,adj,pred_len):
        if pred_len != self.pred_len: raise ValueError((pred_len,self.pred_len))
        return self._run(x,adj,None,False)
