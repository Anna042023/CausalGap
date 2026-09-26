import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import UnifiedModel, ModelOutput

def norm(A): return A/A.sum(-1,keepdim=True).clamp_min(1e-6)
class DCGCNAdapter(UnifiedModel):
    """Traffic-only joint adaptation of the uploaded DCGCN/CASE predictor.

    The original repository trains a separate causal DAG generator before the cascade predictor.
    For the common 8-dataset benchmark this adapter learns a sample-conditioned causal graph jointly
    from the observed traffic latent while retaining the three-path (inter/intra/distance) propagation idea.
    """
    model_name='DCGCN'; uses_graph=True
    model_protocol='v3.0-dcgcn-joint-traffic-causal-graph-adaptation'
    representation_hook_name='final_causal_graph_temporal_latent'
    def __init__(self,num_nodes,history_len,pred_len,d_model=64,e_layers=2,dropout=.1):
        super().__init__(); self.pred_len=pred_len; self.num_nodes=num_nodes
        self.inp=nn.Linear(1,d_model); self.q=nn.Linear(d_model,d_model,bias=False); self.k=nn.Linear(d_model,d_model,bias=False)
        self.inter=nn.ModuleList([nn.Linear(d_model,d_model) for _ in range(e_layers)]); self.dist=nn.ModuleList([nn.Linear(d_model,d_model) for _ in range(e_layers)])
        self.gru=nn.GRU(d_model,d_model,batch_first=True); self.head=nn.Linear(2*d_model,pred_len); self.drop=nn.Dropout(dropout)
    def _forward(self,x,adj):
        h=self.inp(x); base=h.mean(dim=1)
        score=torch.einsum('bnd,bmd->bnm',self.q(base),self.k(base))/(base.size(-1)**.5); causal=F.softmax(score,dim=-1)
        A=adj if torch.is_tensor(adj) else torch.as_tensor(adj); A=A.to(x.device,x.dtype); A=norm(A+torch.eye(A.size(0),device=x.device,dtype=x.dtype))
        hi=base; hd=base
        for li,ld in zip(self.inter,self.dist):
            hi=F.relu(li(torch.einsum('bnm,bmd->bnd',causal,hi))+hi)
            hd=F.relu(ld(torch.einsum('nm,bmd->bnd',A,hd))+hd)
        # temporal latent preserves sequence information
        B,T,N,D=h.shape; z=h.permute(0,2,1,3).reshape(B*N,T,D); z,_=self.gru(z); z=z[:,-1].reshape(B,N,D)
        rep=torch.cat([z,0.5*(hi+hd)],dim=-1); pred=self.head(self.drop(rep)).permute(0,2,1).unsqueeze(-1)
        return ModelOutput(pred,rep)
    def training_forward(self,x,adj,y=None): return self._forward(x,adj)
    def inference_forward(self,x,adj,pred_len):
        if pred_len!=self.pred_len: raise ValueError((pred_len,self.pred_len))
        return self._forward(x,adj)
