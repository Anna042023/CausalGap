"""
PyTorch port of the uploaded VeritasYin/STGCN_IJCAI-18 model structure.
The official repository is TensorFlow-1.x and is preserved untouched in class1/STGCN.
This file ports the same ST-Conv blocks, Chebyshev graph convolution, GLU temporal conv,
layer normalization, output layer, and autoregressive multi-step inference into PyTorch.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.linalg import eigs
from .base import UnifiedModel, ModelOutput


def scaled_laplacian(W):
    W = np.asarray(W, dtype=np.float64)
    d = W.sum(axis=1)
    L = -W.copy()
    np.fill_diagonal(L, d)
    denom = np.sqrt(np.outer(d, d))
    mask = denom > 0
    L[mask] = L[mask] / denom[mask]
    try:
        lm = float(np.real(eigs(L, k=1, which='LR', return_eigenvectors=False)[0]))
    except Exception:
        lm = float(np.max(np.linalg.eigvals(L).real))
    if abs(lm) < 1e-8:
        lm = 2.0
    return (2.0 * L / lm - np.eye(W.shape[0])).astype(np.float32)


def cheb_polynomials(W, Ks):
    L = scaled_laplacian(W)
    n = L.shape[0]
    out = [np.eye(n, dtype=np.float32)]
    if Ks > 1:
        out.append(L)
    for _ in range(2, Ks):
        out.append(2 * L @ out[-1] - out[-2])
    return np.stack(out[:Ks], axis=0).astype(np.float32)


class NodeChannelLayerNorm(nn.Module):
    def __init__(self, n, c):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1,1,n,c))
        self.beta = nn.Parameter(torch.zeros(1,1,n,c))
    def forward(self, x):
        mu = x.mean(dim=(2,3), keepdim=True)
        var = ((x-mu)**2).mean(dim=(2,3), keepdim=True)
        return (x-mu)/torch.sqrt(var+1e-6)*self.gamma + self.beta


class TemporalConv(nn.Module):
    def __init__(self, kt, c_in, c_out, act='relu'):
        super().__init__()
        self.kt, self.c_in, self.c_out, self.act = kt, c_in, c_out, act
        out_ch = 2*c_out if act == 'GLU' else c_out
        self.conv = nn.Conv2d(c_in, out_ch, kernel_size=(kt,1), padding=0)
        self.res_proj = nn.Conv2d(c_in, c_out, kernel_size=1) if c_in > c_out else None
    def _residual(self, x):
        # x [B,T,N,C] -> NCHW for conv
        B,T,N,C = x.shape
        if self.res_proj is not None:
            r = self.res_proj(x.permute(0,3,1,2)).permute(0,2,3,1)
        elif C < self.c_out:
            pad = torch.zeros(B,T,N,self.c_out-C, device=x.device, dtype=x.dtype)
            r = torch.cat([x,pad], dim=-1)
        else:
            r = x
        return r[:, self.kt-1:, :, :]
    def forward(self, x):
        r = self._residual(x)
        z = self.conv(x.permute(0,3,1,2)).permute(0,2,3,1)
        if self.act == 'GLU':
            a,b = z[...,:self.c_out], z[...,self.c_out:]
            return (a+r) * torch.sigmoid(b)
        if self.act == 'linear': return z
        if self.act == 'sigmoid': return torch.sigmoid(z)
        if self.act == 'relu': return F.relu(z+r)
        raise ValueError(self.act)


class GraphConv(nn.Module):
    def __init__(self, ks, c_in, c_out):
        super().__init__()
        self.ks, self.c_in, self.c_out = ks, c_in, c_out
        self.theta = nn.Parameter(torch.empty(ks*c_in, c_out))
        self.bias = nn.Parameter(torch.zeros(c_out))
        nn.init.xavier_uniform_(self.theta)
        self.res_proj = nn.Linear(c_in,c_out) if c_in>c_out else None
    def forward(self, x, cheb):
        # x [B,T,N,C], cheb [K,N,N]
        B,T,N,C = x.shape
        supports = torch.einsum('knm,btmc->btnkc', cheb, x)  # [B,T,N,K,C]
        z = supports.reshape(B,T,N,self.ks*C) @ self.theta + self.bias
        if self.res_proj is not None:
            r = self.res_proj(x)
        elif C < self.c_out:
            pad = torch.zeros(B,T,N,self.c_out-C, device=x.device, dtype=x.dtype)
            r = torch.cat([x,pad], dim=-1)
        else:
            r = x
        return F.relu(z+r)


class STBlock(nn.Module):
    def __init__(self, n, ks, kt, c_in, c_t, c_out, dropout):
        super().__init__()
        self.t1 = TemporalConv(kt,c_in,c_t,'GLU')
        self.g = GraphConv(ks,c_t,c_t)
        self.t2 = TemporalConv(kt,c_t,c_out,'relu')
        self.norm = NodeChannelLayerNorm(n,c_out)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,cheb):
        x = self.t1(x)
        x = self.g(x,cheb)
        x = self.t2(x)
        x = self.norm(x)
        return self.dropout(x)


class OutputLayer(nn.Module):
    def __init__(self,n,T,c,pred_len):
        super().__init__()
        self.t1 = TemporalConv(T,c,c,'GLU')
        self.norm = NodeChannelLayerNorm(n,c)
        self.t2 = TemporalConv(1,c,c,'sigmoid')
        self.pred_len = int(pred_len)
        self.fc = nn.Linear(c,self.pred_len)
    def forward(self,x):
        x = self.t1(x)
        x = self.norm(x)
        rep = self.t2(x)          # [B,1,N,C] : pre-head representation
        y = self.fc(rep)          # [B,1,N,H]
        y = y[:,0].permute(0,2,1).unsqueeze(-1)  # [B,H,N,1]
        return y, rep


class STGCN(UnifiedModel):
    model_name='STGCN'
    uses_graph=True
    model_protocol='v2.4-direct-multihorizon-readout'
    def __init__(self, num_nodes, history_len, pred_len, ks=3, kt=3, dropout=0.0,
                 blocks=((1,32,64),(64,32,128))):
        super().__init__()
        self.num_nodes=num_nodes; self.history_len=history_len; self.pred_len=int(pred_len); self.ks=ks
        remain = history_len
        modules=[]
        for c_in,c_t,c_out in blocks:
            modules.append(STBlock(num_nodes,ks,kt,c_in,c_t,c_out,dropout))
            remain -= 2*(kt-1)
        if remain <= 1:
            raise ValueError(f'STGCN history_len={history_len} is too short for two Kt={kt} blocks; remaining Ko={remain}.')
        self.blocks=nn.ModuleList(modules)
        self.output=OutputLayer(num_nodes,remain,blocks[-1][-1],self.pred_len)
        self._cached_key=None
        self.register_buffer('_cheb', torch.empty(0), persistent=False)

    def _get_cheb(self, adj, device):
        if isinstance(adj, torch.Tensor):
            a = adj.detach().cpu().numpy()
        else:
            a = np.asarray(adj)
        key = (a.shape, float(a.sum()), float((a*a).sum()))
        if self._cached_key != key or self._cheb.numel()==0:
            c = torch.from_numpy(cheb_polynomials(a,self.ks)).float().to(device)
            self._cheb = c
            self._cached_key = key
        return self._cheb

    def forward_one_step(self,x,adj):
        cheb=self._get_cheb(adj,x.device)
        h=x
        for b in self.blocks:
            h=b(h,cheb)
        y,rep=self.output(h)
        return y,rep

    def training_forward(self,x,adj,y=None):
        pred,rep=self.forward_one_step(x,adj)
        return ModelOutput(pred,rep)

    def inference_forward(self,x,adj,pred_len):
        if int(pred_len) != self.pred_len:
            raise ValueError(f'STGCN built with pred_len={self.pred_len}, requested {pred_len}')
        pred,rep=self.forward_one_step(x,adj)
        return ModelOutput(pred,rep)
