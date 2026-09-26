import torch
import torch.nn as nn
from ..base import UnifiedModel, ModelOutput

def norm_adj(adj,x):
    A=adj if torch.is_tensor(adj) else torch.as_tensor(adj); A=A.to(x.device,x.dtype)
    A=A+torch.eye(A.size(0),device=x.device,dtype=x.dtype); return A/A.sum(-1,keepdim=True).clamp_min(1e-6)
class CCHMMAdapter(UnifiedModel):
    """Traffic-only CCHMM adaptation.

    The uploaded CCHMM requires bike/taxi/bus/speed + POI/weather/time contexts. Those modalities do
    not exist consistently across CausalGap's eight datasets. This adapter keeps graph-conditioned
    latent prior/posterior dynamics for the common traffic channel and does not fabricate unavailable modalities.
    """
    model_name='CCHMM'; uses_graph=True
    model_protocol='v3.0-cchmm-traffic-only-causal-latent'
    representation_hook_name='causal_latent_recurrent_state'
    def __init__(self,num_nodes,history_len,pred_len,d_model=64,dropout=.1):
        super().__init__(); self.pred_len=pred_len; self.num_nodes=num_nodes; self.d=d_model
        self.obs=nn.Linear(1,d_model); self.prior=nn.GRUCell(d_model,d_model); self.post=nn.GRUCell(2*d_model,d_model)
        self.mu=nn.Linear(d_model,d_model); self.logvar=nn.Linear(d_model,d_model); self.head=nn.Linear(d_model,pred_len); self.drop=nn.Dropout(dropout)
    def _forward(self,x,adj):
        A=norm_adj(adj,x); B=x.size(0); h=torch.zeros(B*self.num_nodes,self.d,device=x.device,dtype=x.dtype)
        for t in range(x.size(1)):
            o=self.obs(x[:,t]); g=torch.einsum('nm,bmd->bnd',A,o); gf=g.reshape(B*self.num_nodes,self.d); of=o.reshape(B*self.num_nodes,self.d)
            hp=self.prior(gf,h); h=self.post(torch.cat([of,gf],dim=-1),hp)
        mu=self.mu(h).reshape(B,self.num_nodes,self.d); lv=self.logvar(h).clamp(-8,8).reshape(B,self.num_nodes,self.d)
        z=mu if not self.training else mu+torch.randn_like(mu)*torch.exp(0.5*lv)
        pred=self.head(self.drop(z)).permute(0,2,1).unsqueeze(-1)
        return ModelOutput(pred,mu)
    def training_forward(self,x,adj,y=None): return self._forward(x,adj)
    def inference_forward(self,x,adj,pred_len):
        if pred_len!=self.pred_len: raise ValueError((pred_len,self.pred_len))
        was=self.training; self.eval(); out=self._forward(x,adj); self.train(was); return out
