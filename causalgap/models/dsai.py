import numpy as np
import torch
import torch.nn as nn
from .base import UnifiedModel
from ..spatial import dsai_raw_encoding, fit_fixed_bn, apply_fixed_bn, normalized_adjacency


class DSAIWrapper(UnifiedModel):
    """Track-B information-equivalent topology frontend (v2.4).

    The v2.3 static additive encoding could be cancelled by time-series
    normalization and was insensitive to edge-weight changes. v2.4 therefore
    provides two complementary signals:

      s_i : weight-aware static topology descriptors (fixed baseline BN)
      m_t = A_norm x_t : dynamic topology-conditioned neighbor message

    To preserve the native scalar input contract of all connected baselines, a
    shared near-identity frontend maps [x, m, x*s, m*s] -> scalar. A fixed 5%
    dynamic-message residual guarantees that topology information is actually
    present before the native predictor; a zero-initialized learnable residual
    then adapts the common interface during training.
    """
    model_protocol = 'v3.0-dsai-dynamic-weight-aware'

    def __init__(self, base_model, baseline_adj, coords=None,
                 coordinate_source='missing_zero_filled', dynamic_mix=0.05):
        super().__init__()
        self.base = base_model
        self.model_name = base_model.model_name
        self.output_space = getattr(base_model, 'output_space', 'normalized')
        self.is_zero_shot = bool(getattr(base_model, 'is_zero_shot', False))
        self.representation_hook_name = getattr(base_model, 'representation_hook_name', 'unknown')
        self.native_uses_graph = bool(getattr(base_model, 'uses_graph', False))
        self.uses_graph = True
        self.evaluation_track = 'B'
        self.coordinate_source = coordinate_source
        self.coords = None if coords is None else np.asarray(coords, dtype=np.float32)
        self.dynamic_mix = float(dynamic_mix)
        if not (0.0 < self.dynamic_mix < 1.0):
            raise ValueError('dynamic_mix must lie in (0,1).')

        raw0 = dsai_raw_encoding(baseline_adj, self.coords)
        mean, std = fit_fixed_bn(raw0)
        self.register_buffer('dsai_mean', torch.from_numpy(mean), persistent=True)
        self.register_buffer('dsai_std', torch.from_numpy(std), persistent=True)

        # [x, m, x*s(5), m*s(5)] = 12 features. The residual starts at zero,
        # while the fixed (1-beta)x + beta*m path guarantees observability.
        self.residual = nn.Linear(12, 1, bias=False)
        nn.init.zeros_(self.residual.weight)

        self._cache_key = None
        self.register_buffer('_cached_spatial', torch.empty(0), persistent=False)
        self.register_buffer('_cached_anorm', torch.empty(0), persistent=False)

    @staticmethod
    def _adj_key(a):
        a = np.asarray(a)
        return (a.shape, float(a.sum()), float((a*a).sum()), int(np.count_nonzero(a)))

    def _graph_tensors(self, adj, device, dtype):
        a = adj.detach().cpu().numpy() if torch.is_tensor(adj) else np.asarray(adj)
        key = self._adj_key(a)
        if self._cache_key != key or self._cached_spatial.numel() == 0 or self._cached_anorm.numel() == 0:
            raw = dsai_raw_encoding(a, self.coords)
            norm = apply_fixed_bn(
                raw,
                self.dsai_mean.detach().cpu().numpy(),
                self.dsai_std.detach().cpu().numpy(),
            )
            anorm = normalized_adjacency(a, self_loop=True)
            self._cached_spatial = torch.from_numpy(norm).to(device=device, dtype=dtype)
            self._cached_anorm = torch.from_numpy(anorm).to(device=device, dtype=dtype)
            self._cache_key = key
        return (
            self._cached_spatial.to(device=device, dtype=dtype),
            self._cached_anorm.to(device=device, dtype=dtype),
        )

    def _components(self, x, adj):
        # x [B,T,N,1], s [N,5], A_norm [N,N]
        s, anorm = self._graph_tensors(adj, x.device, x.dtype)
        m = torch.einsum('nm,btmc->btnc', anorm, x)
        sb = s.view(1,1,s.shape[0],5).expand(x.shape[0], x.shape[1], -1, -1)
        feat = torch.cat([x, m, x * sb, m * sb], dim=-1)
        return s, m, feat

    def _condition(self, x, adj):
        _, m, feat = self._components(x, adj)
        beta = self.dynamic_mix
        return (1.0-beta) * x + beta * m + self.residual(feat)

    def observability_diagnostics(self, x_base, adj_base, x_intervention, adj_intervention):
        """Measure whether Track-B input actually changes when the graph changes.

        This gate diagnoses the interface, not model rigidity. A model may still
        produce a frozen latent representation after receiving a changed input;
        that is a valid CausalGap outcome. But if the conditioned input itself
        does not change, DMIS=0 must not be interpreted as rigidity.
        """
        with torch.no_grad():
            s0, m0, _ = self._components(x_base, adj_base)
            sk, mk, _ = self._components(x_intervention, adj_intervention)
            c0 = self._condition(x_base, adj_base)
            ck = self._condition(x_intervention, adj_intervention)
            static_delta = torch.mean(torch.abs(sk - s0)).item()
            dynamic_delta = torch.mean(torch.abs(mk - m0)).item()
            conditioned_delta = torch.mean(torch.abs(ck - c0)).item()
            conditioned_rel = (
                torch.linalg.vector_norm((ck-c0).reshape(-1)) /
                (torch.linalg.vector_norm(c0.reshape(-1)) + 1e-12)
            ).item()
        observable = bool((static_delta > 1e-10 or dynamic_delta > 1e-10) and conditioned_delta > 1e-10)
        return {
            'dsai_static_delta': float(static_delta),
            'dsai_dynamic_message_delta': float(dynamic_delta),
            'dsai_conditioned_input_delta': float(conditioned_delta),
            'dsai_conditioned_input_relative_delta': float(conditioned_rel),
            'dsai_observable': observable,
        }


    def set_training_progress(self, epoch, max_epochs):
        if hasattr(self.base, 'set_training_progress'):
            self.base.set_training_progress(epoch, max_epochs)

    def training_forward(self, x, adj, y=None):
        return self.base.training_forward(self._condition(x, adj), adj, y)

    def training_forward_with_mask(self, x, adj, y, target_mask):
        conditioned = self._condition(x, adj)
        if hasattr(self.base, 'training_forward_with_mask'):
            return self.base.training_forward_with_mask(conditioned, adj, y, target_mask)
        return self.base.training_forward(conditioned, adj, y)

    def inference_forward(self, x, adj, pred_len):
        return self.base.inference_forward(self._condition(x, adj), adj, pred_len)

    def inference_forward_selective_rep(self, x, adj, pred_len, need_rep=True):
        conditioned = self._condition(x, adj)
        if hasattr(self.base, 'inference_forward_selective_rep'):
            return self.base.inference_forward_selective_rep(
                conditioned, adj, pred_len, need_rep=need_rep
            )
        return self.base.inference_forward(conditioned, adj, pred_len)

    def cache_stats(self):
        if hasattr(self.base, 'cache_stats'):
            return self.base.cache_stats()
        return {}
