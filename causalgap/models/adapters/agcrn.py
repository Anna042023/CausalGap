import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import UnifiedModel, ModelOutput

class AVWGCN(nn.Module):
    def __init__(self,dim_in,dim_out,cheb_k,embed_dim):
        super().__init__(); self.cheb_k=cheb_k
        self.weights_pool=nn.Parameter(torch.empty(embed_dim,cheb_k,dim_in,dim_out)); self.bias_pool=nn.Parameter(torch.empty(embed_dim,dim_out))
        nn.init.xavier_uniform_(self.weights_pool); nn.init.zeros_(self.bias_pool)
    def forward(self,x,emb):
        N=emb.size(0); S=F.softmax(F.relu(emb@emb.T),dim=1); ss=[torch.eye(N,device=x.device,dtype=x.dtype),S]
        for _ in range(2,self.cheb_k): ss.append(2*S@ss[-1]-ss[-2])
        ss=torch.stack(ss[:self.cheb_k]); W=torch.einsum('nd,dkio->nkio',emb,self.weights_pool); b=emb@self.bias_pool
        xg=torch.einsum('knm,bmc->bknc',ss,x).permute(0,2,1,3)
        return torch.einsum('bnki,nkio->bno',xg,W)+b
class Cell(nn.Module):
    def __init__(self,n,di,dh,k,e):
        super().__init__(); self.n=n; self.dh=dh; self.gate=AVWGCN(di+dh,2*dh,k,e); self.upd=AVWGCN(di+dh,dh,k,e)
    def forward(self,x,h,e):
        zr=torch.sigmoid(self.gate(torch.cat([x,h],-1),e)); z,r=torch.chunk(zr,2,-1)
        hc=torch.tanh(self.upd(torch.cat([x,z*h],-1),e)); return r*h+(1-r)*hc
class AGCRNAdapter(UnifiedModel):
    model_name='AGCRN'; uses_graph=False
    model_protocol='v3.0-agcrn-official-adaptive-graph'
    representation_hook_name='last_recurrent_hidden_state'
    def __init__(self,num_nodes,history_len,pred_len,hidden_dim=64,embed_dim=10,cheb_k=2,num_layers=2):
        super().__init__(); self.num_nodes=num_nodes; self.pred_len=pred_len; self.hidden_dim=hidden_dim; self.num_layers=num_layers
        self.node_embeddings=nn.Parameter(torch.randn(num_nodes,embed_dim)*0.1)
        self.cells=nn.ModuleList([Cell(num_nodes,1 if i==0 else hidden_dim,hidden_dim,cheb_k,embed_dim) for i in range(num_layers)])
        self.head=nn.Conv2d(1,pred_len,kernel_size=(1,hidden_dim))
    def _forward(self,x):
        B=x.size(0); hs=[torch.zeros(B,self.num_nodes,self.hidden_dim,device=x.device,dtype=x.dtype) for _ in range(self.num_layers)]
        seq=[]
        for t in range(x.size(1)):
            z=x[:,t]
            for i,c in enumerate(self.cells): hs[i]=c(z,hs[i],self.node_embeddings); z=hs[i]
            seq.append(z)
        rep=seq[-1]
        out=self.head(rep.unsqueeze(1)).squeeze(-1).permute(0,1,2).unsqueeze(-1)
        return ModelOutput(out,rep)
    def training_forward(self,x,adj,y=None): return self._forward(x)
    def inference_forward(self,x,adj,pred_len):
        if pred_len!=self.pred_len: raise ValueError((pred_len,self.pred_len))
        return self._forward(x)
