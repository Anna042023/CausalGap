import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import UnifiedModel, ModelOutput

class NConv(nn.Module):
    def forward(self,x,A): return torch.einsum('bcnt,nm->bcmt',x,A).contiguous()
class GCN(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super().__init__(); self.nconv=NConv(); self.order=order; self.dropout=dropout
        self.mlp=nn.Conv2d((order*support_len+1)*c_in,c_out,kernel_size=1)
    def forward(self,x,supports):
        out=[x]
        for A in supports:
            x1=self.nconv(x,A); out.append(x1)
            for _ in range(2,self.order+1):
                x2=self.nconv(x1,A); out.append(x2); x1=x2
        return F.dropout(self.mlp(torch.cat(out,dim=1)),self.dropout,training=self.training)

def rw(A):
    d=A.sum(-1).clamp_min(1e-6); return A/d[:,None]

class GWNetAdapter(UnifiedModel):
    model_name='GWNet'; uses_graph=True
    model_protocol='v3.0-gwnet-official-architecture-compat-patch'
    representation_hook_name='end1_hidden_immediately_before_final_prediction_head'
    def __init__(self,num_nodes,history_len,pred_len,dropout=.3,residual_channels=32,dilation_channels=32,
                 skip_channels=128,end_channels=256,blocks=4,layers=2):
        super().__init__(); self.num_nodes=num_nodes; self.pred_len=pred_len; self.dropout=dropout
        self.blocks=blocks; self.layers=layers
        self.start=nn.Conv2d(1,residual_channels,1)
        self.filter=nn.ModuleList(); self.gate=nn.ModuleList(); self.res=nn.ModuleList(); self.skip=nn.ModuleList(); self.bn=nn.ModuleList(); self.gcn=nn.ModuleList()
        rf=1
        for _ in range(blocks):
            dilation=1; scope=1
            for _ in range(layers):
                self.filter.append(nn.Conv2d(residual_channels,dilation_channels,(1,2),dilation=(1,dilation)))
                self.gate.append(nn.Conv2d(residual_channels,dilation_channels,(1,2),dilation=(1,dilation)))
                self.res.append(nn.Conv2d(dilation_channels,residual_channels,1)); self.skip.append(nn.Conv2d(dilation_channels,skip_channels,1))
                self.bn.append(nn.BatchNorm2d(residual_channels)); self.gcn.append(GCN(dilation_channels,residual_channels,dropout,3,2))
                rf += scope; scope*=2; dilation*=2
        self.receptive_field=rf
        self.nodevec1=nn.Parameter(torch.randn(num_nodes,10)); self.nodevec2=nn.Parameter(torch.randn(10,num_nodes))
        self.end1=nn.Conv2d(skip_channels,end_channels,1); self.end2=nn.Conv2d(end_channels,pred_len,1)
    def _supports(self,adj,x):
        A=adj if torch.is_tensor(adj) else torch.as_tensor(adj)
        A=A.to(x.device,x.dtype)
        s1=rw(A).transpose(0,1); s2=rw(A.transpose(0,1)).transpose(0,1)
        adp=F.softmax(F.relu(self.nodevec1@self.nodevec2),dim=1)
        return [s1,s2,adp]
    def _forward(self,x,adj):
        # B,T,N,1 -> B,1,N,T
        z=x.permute(0,3,2,1)
        if z.size(3)<self.receptive_field: z=F.pad(z,(self.receptive_field-z.size(3),0,0,0))
        z=self.start(z); sk=None; supports=self._supports(adj,z)
        for i in range(self.blocks*self.layers):
            residual=z
            f=torch.tanh(self.filter[i](z)); g=torch.sigmoid(self.gate[i](z)); z=f*g
            s=self.skip[i](z); sk=s if sk is None else s+sk[:,:,:,-s.size(3):]
            z=self.gcn[i](z,supports)+residual[:,:,:,-z.size(3):]
            z=self.bn[i](z)
        rep=F.relu(sk)
        h=F.relu(self.end1(rep)); out=self.end2(h)
        # last temporal location; [B,H,N,1]
        pred=out[:,:,:,-1:].contiguous()
        # The paper uses the representation immediately preceding the final
        # prediction head.  end2 is the final head, so its input h (not the
        # earlier skip tensor rep) is the common extraction interface.
        rep2=h[:,:,:,-1].permute(0,2,1).contiguous()
        return ModelOutput(pred,rep2)
    def training_forward(self,x,adj,y=None): return self._forward(x,adj)
    def inference_forward(self,x,adj,pred_len):
        if pred_len!=self.pred_len: raise ValueError((pred_len,self.pred_len))
        return self._forward(x,adj)
