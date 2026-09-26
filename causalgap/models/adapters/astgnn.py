import torch
import torch.nn as nn
from ..base import UnifiedModel, ModelOutput


def norm_adj(adj, device, dtype):
    A = adj if torch.is_tensor(adj) else torch.as_tensor(adj)
    A=A.to(device=device,dtype=dtype)
    I=torch.eye(A.size(0),device=device,dtype=dtype)
    A=A+I
    d=A.sum(-1).clamp_min(1e-6)
    return A / d[:,None]


class GraphFFN(nn.Module):
    def __init__(self,d_model,dropout):
        super().__init__()
        self.lin1=nn.Linear(d_model,2*d_model); self.lin2=nn.Linear(2*d_model,d_model)
        self.drop=nn.Dropout(dropout); self.norm=nn.LayerNorm(d_model)
    def forward(self,x,A):
        # x B,T,N,D
        g=torch.einsum('nm,btmd->btnd',A,x)
        z=self.lin2(self.drop(torch.relu(self.lin1(g))))
        return self.norm(x+self.drop(z))


class ASTGNNAdapter(UnifiedModel):
    """Benchmark adapter preserving ASTGNN's graph propagation + temporal-attention mechanism.

    The uploaded ASTGNN assumes a dataset-specific dependency layout whose encoder length is
    tied to the prediction length.  CausalGap has 35->1, 19->1 and 12->12 protocols, so this
    adapter keeps the core spatial graph/temporal attention mechanism while using a
    horizon-matched readout.  This adaptation is explicitly recorded in the registry.
    """
    model_name='ASTGNN'; uses_graph=True
    model_protocol='v3.0-astgnn-horizon-context-adapter'
    representation_hook_name='final_graph_temporal_encoder_state'
    def __init__(self,num_nodes,history_len,pred_len,d_model=64,n_heads=4,e_layers=3,dropout=0.1):
        super().__init__(); self.num_nodes=num_nodes; self.history_len=history_len; self.pred_len=pred_len
        self.in_proj=nn.Linear(1,d_model)
        enc=nn.TransformerEncoderLayer(d_model,n_heads,4*d_model,dropout,batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(enc,num_layers=e_layers)
        self.graph_layers=nn.ModuleList([GraphFFN(d_model,dropout) for _ in range(max(1,e_layers//2))])
        self.time_pos=nn.Parameter(torch.randn(1,history_len,1,d_model)*0.02)
        self.head=nn.Linear(d_model,pred_len)
    def _forward(self,x,adj):
        A=norm_adj(adj,x.device,x.dtype)
        h=self.in_proj(x)+self.time_pos[:,:x.size(1)]
        for g in self.graph_layers: h=g(h,A)
        B,T,N,D=h.shape
        ht=h.permute(0,2,1,3).reshape(B*N,T,D)
        ht=self.temporal(ht)
        rep=ht[:,-1].reshape(B,N,D)
        y=self.head(rep).permute(0,2,1).unsqueeze(-1)
        return ModelOutput(y,rep)
    def training_forward(self,x,adj,y=None): return self._forward(x,adj)
    def inference_forward(self,x,adj,pred_len):
        if pred_len!=self.pred_len: raise ValueError((pred_len,self.pred_len))
        return self._forward(x,adj)
