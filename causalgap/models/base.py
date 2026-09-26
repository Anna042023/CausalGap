from dataclasses import dataclass, field
from typing import Optional, Dict
import torch


@dataclass
class ModelOutput:
    pred: torch.Tensor
    representation: torch.Tensor
    loss: Optional[torch.Tensor] = None
    loss_terms: Dict[str, float] = field(default_factory=dict)


class UnifiedModel(torch.nn.Module):
    model_name = 'base'
    uses_graph = False
    # All current adapters are trained/evaluated in the unified normalized target
    # space. The evaluator applies the dataset scaler exactly once. This explicit
    # contract prevents accidental double inverse-transform bugs.
    output_space = 'normalized'

    def training_forward(self, x, adj, y=None):
        raise NotImplementedError

    def inference_forward(self, x, adj, pred_len):
        raise NotImplementedError
