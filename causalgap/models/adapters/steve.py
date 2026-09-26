from __future__ import annotations

from ..base import UnifiedModel, ModelOutput
from .steve_official_core import ReleasedSTEVECore


class STEVEAdapter(UnifiedModel):
    """Released-core STEVE adaptation under the common traffic-only input protocol.

    The supplied released repository requires auxiliary temporal/spatial/congestion
    labels and, for the original mobility datasets, two flow channels. CausalGap's
    common-input benchmark exposes one traffic channel and no such auxiliary labels.
    We therefore port the released prediction core but do not fabricate unavailable
    supervision. This distinction is recorded in model_protocol and run metadata.
    """

    model_name = 'STEVE'
    uses_graph = True
    native_uses_graph = True
    model_protocol = 'v3.3.3-steve-released-core-traffic-only'
    representation_hook_name = 'released_invariant_st_latent'
    output_space = 'normalized'

    def __init__(self, num_nodes, history_len, pred_len, d_model=64, dropout=.1):
        super().__init__()
        self.pred_len = int(pred_len)
        self.core = ReleasedSTEVECore(
            num_nodes=int(num_nodes), history_len=int(history_len), pred_len=int(pred_len),
            d_model=int(d_model), dropout=float(dropout), bank_multiplier=2.0, bank_gamma=0.5,
        )

    def _forward(self, x, adj):
        pred, rep, _ = self.core(x, adj)
        return ModelOutput(pred=pred, representation=rep)

    def training_forward(self, x, adj, y=None):
        return self._forward(x, adj)

    def inference_forward(self, x, adj, pred_len):
        if int(pred_len) != self.pred_len:
            raise ValueError((pred_len, self.pred_len))
        return self._forward(x, adj)
