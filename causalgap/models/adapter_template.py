"""Template for the remaining official baselines."""
from .base import UnifiedModel, ModelOutput

class YourModelAdapter(UnifiedModel):
    model_name='YourModel'
    uses_graph=True
    def __init__(self, official_model):
        super().__init__()
        self.model=official_model
    def training_forward(self,x,adj,y=None):
        # 1) convert canonical x/adj into official input format
        # 2) call the official model
        # 3) obtain pre-head representation z
        # 4) return pred as [B,Tout,N,1]
        raise NotImplementedError
    def inference_forward(self,x,adj,pred_len):
        raise NotImplementedError
