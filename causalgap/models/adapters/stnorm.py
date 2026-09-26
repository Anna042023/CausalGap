import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import UnifiedModel, ModelOutput

class SNorm(nn.Module):
    def __init__(self,c): super().__init__(); self.beta=nn.Parameter(torch.zeros(c)); self.gamma=nn.Parameter(torch.ones(c))
    def forward(self,x):
        v=x.var(2,keepdim=True,unbiased=False); m=x.mean(2,keepdim=True)
        return (x-m)/torch.sqrt(v+1e-5)*self.gamma.view(1,-1,1,1)+self.beta.view(1,-1,1,1)
class TNorm(nn.Module):
    def __init__(self,n,c,momentum=.1):
        super().__init__(); self.beta=nn.Parameter(torch.zeros(1,c,n,1)); self.gamma=nn.Parameter(torch.ones(1,c,n,1)); self.momentum=momentum
        self.register_buffer('running_mean',torch.zeros(1,c,n,1)); self.register_buffer('running_var',torch.ones(1,c,n,1))
    def forward(self,x):
        if self.training:
            m=x.mean((0,3),keepdim=True); v=x.var((0,3),keepdim=True,unbiased=False)
            with torch.no_grad(): self.running_mean.mul_(1-self.momentum).add_(self.momentum*m); self.running_var.mul_(1-self.momentum).add_(self.momentum*v)
        else: m,v=self.running_mean,self.running_var
        return (x-m)/torch.sqrt(v+1e-5)*self.gamma+self.beta

class STNormAdapter(UnifiedModel):
    model_name='ST-Norm'; uses_graph=False
    model_protocol='v3.0-stnorm-official-wavenet-path'
    representation_hook_name='pre_output_wavenet_hidden_feature'
    def __init__(self,num_nodes,history_len,pred_len,channels=32,kernel_size=2,blocks=4,layers=2,dropout=.2):
        super().__init__(); self.pred_len=pred_len; self.blocks=blocks; self.layers=layers
        self.start=nn.Conv2d(1,channels,1); self.tn=nn.ModuleList(); self.sn=nn.ModuleList(); self.filter=nn.ModuleList(); self.gate=nn.ModuleList(); self.res=nn.ModuleList(); self.skip=nn.ModuleList(); rf=1
        for _ in range(blocks):
            d=1; scope=1
            for _ in range(layers):
                self.tn.append(TNorm(num_nodes,channels)); self.sn.append(SNorm(channels))
                self.filter.append(nn.Conv2d(3*channels,channels,(1,kernel_size),dilation=(1,d)))
                self.gate.append(nn.Conv2d(3*channels,channels,(1,kernel_size),dilation=(1,d)))
                self.res.append(nn.Conv2d(channels,channels,1)); self.skip.append(nn.Conv2d(channels,channels,1)); rf+=scope; scope*=2; d*=2
        self.rf=rf; self.drop=nn.Dropout(dropout); self.end1=nn.Conv2d(channels,channels,1); self.end2=nn.Conv2d(channels,pred_len,1)
    def _forward(self,x):
        z=x.permute(0,3,2,1)
        if z.size(-1)<self.rf: z=F.pad(z,(self.rf-z.size(-1),0,0,0))
        z=self.start(z); sk=None
        for i in range(self.blocks*self.layers):
            res=z; cat=torch.cat([z,self.tn[i](z),self.sn[i](z)],dim=1)
            z=torch.tanh(self.filter[i](cat))*torch.sigmoid(self.gate[i](cat)); z=self.drop(z)
            s=self.skip[i](z); sk=s if sk is None else s+sk[:,:,:,-s.size(-1):]
            z=self.res[i](z)+res[:,:,:,-z.size(-1):]
        rep=F.relu(self.end1(F.relu(sk)))
        out=self.end2(rep)[:,:,:,-1:]
        rep2=rep[:,:,:,-1].permute(0,2,1)
        return ModelOutput(out,rep2)
    def training_forward(self,x,adj,y=None): return self._forward(x)
    def inference_forward(self,x,adj,pred_len):
        if pred_len!=self.pred_len: raise ValueError((pred_len,self.pred_len))
        return self._forward(x)
