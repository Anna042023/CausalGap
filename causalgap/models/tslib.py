from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace
import torch

# Time-Series-Library imports reformer_pytorch unconditionally inside a shared
# attention file. The six models used here do not instantiate ReformerLayer.
try:
    import reformer_pytorch  # noqa: F401
except Exception:
    stub = types.ModuleType('reformer_pytorch')

    class _UnusedLSH(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError('reformer-pytorch is required only when using the Reformer model.')

    stub.LSHSelfAttention = _UnusedLSH
    sys.modules['reformer_pytorch'] = stub

from .base import UnifiedModel, ModelOutput
from ..config import TSLIB_ROOT


def _add_tslib(path):
    p = str(Path(path))
    if p not in sys.path:
        sys.path.insert(0, p)


def _safe_odd_kernel(seq_len: int, requested: int) -> int:
    seq_len = max(1, int(seq_len))
    requested = max(1, int(requested))
    k = min(seq_len, requested)
    if k % 2 == 0:
        k -= 1
    return max(1, k)


class TSLibModel(UnifiedModel):
    uses_graph = False
    output_space = 'normalized'
    model_protocol = 'v3.3-tslib-node-as-variable-short-window-safe'

    def __init__(
        self,
        model_name,
        num_nodes,
        history_len,
        pred_len,
        tslib_root=TSLIB_ROOT,
        d_model=64,
        d_ff=128,
        e_layers=2,
        d_layers=1,
        n_heads=4,
        dropout=0.1,
        moving_avg=3,
        factor=1,
        patch_len=None,
        stride=None,
    ):
        super().__init__()
        self.model_name = model_name

        # Paper-aligned representation interface metadata.
        # For Autoformer/FEDformer, the actual hook below is attached to the
        # input of decoder.projection, i.e. the decoder latent immediately
        # before the final prediction projection.
        if model_name in ('FEDformer', 'Autoformer'):
            self.representation_hook_name = (
                'decoder_latent_immediately_before_final_projection_fixed_flattening'
            )
        elif model_name == 'TimesNet':
            self.representation_hook_name = 'projection_input_with_fixed_temporal_pooling'
        elif model_name == 'PatchTST':
            self.representation_hook_name = 'head_input_pre_prediction'
        elif model_name == 'iTransformer':
            self.representation_hook_name = 'projection_input_pre_prediction'
        elif model_name == 'DLinear':
            self.representation_hook_name = 'seasonal_trend_latents_pre_prediction'
        else:
            self.representation_hook_name = 'pre_prediction_representation'

        self.num_nodes = int(num_nodes)
        self.history_len = int(history_len)
        self.pred_len = int(pred_len)

        _add_tslib(tslib_root)
        from models import PatchTST, iTransformer, TimesNet, FEDformer, DLinear, Autoformer

        table = {
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'TimesNet': TimesNet,
            'FEDformer': FEDformer,
            'DLinear': DLinear,
            'Autoformer': Autoformer,
        }
        if model_name not in table:
            raise ValueError(f'Unsupported TSLib model {model_name}')

        # For traffic, sensors/regions are variables. Never flatten [B,L,N] into
        # [B*N,L,1]; doing so destroys iTransformer's cross-variable attention and
        # materially changes DLinear/FEDformer/Autoformer semantics.
        effective_ma = _safe_odd_kernel(history_len, moving_avg)
        effective_modes = max(1, min(32, int(history_len) // 2))
        label_len = max(1, min(int(history_len) // 2, int(history_len) - 1 if history_len > 1 else 1))

        cfg = SimpleNamespace(
            task_name='long_term_forecast',
            seq_len=int(history_len),
            pred_len=int(pred_len),
            label_len=label_len,
            enc_in=int(num_nodes),
            dec_in=int(num_nodes),
            c_out=int(num_nodes),
            d_model=int(d_model),
            d_ff=int(d_ff),
            e_layers=int(e_layers),
            d_layers=int(d_layers),
            n_heads=int(n_heads),
            factor=int(factor),
            dropout=float(dropout),
            activation='gelu',
            embed='fixed',
            freq='h',
            moving_avg=effective_ma,
            top_k=5,
            num_kernels=6,
            num_class=2,
            output_attention=False,
            individual=False,
            # FEDformer fields; harmless extras for the other TSLib models.
            version='Fourier',
            mode_select='random',
            modes=effective_modes,
            base='legendre',
            cross_activation='tanh',
        )
        self.effective_moving_avg = effective_ma
        self.effective_modes = effective_modes
        self.label_len = label_len

        if model_name == 'PatchTST':
            if patch_len is None:
                # Keep several patches for short traffic contexts instead of a
                # single full-length patch.
                pl = min(16, max(2, int(history_len) // 3))
            else:
                pl = max(2, min(int(patch_len), int(history_len)))
            st = max(1, int(stride) if stride is not None else pl // 2)
            st = min(st, pl)
            self.patch_len = pl
            self.stride = st
            self.model = table[model_name].Model(cfg, patch_len=pl, stride=st)
        else:
            self.patch_len = None
            self.stride = None
            self.model = table[model_name].Model(cfg)

        self._rep_cache = []
        self._register_rep_hook()

    def _register_rep_hook(self):
        def prehook(module, inputs):
            if inputs and torch.is_tensor(inputs[0]):
                self._rep_cache.append(inputs[0])

        m = self.model
        if self.model_name == 'PatchTST':
            m.head.register_forward_pre_hook(prehook)
        elif self.model_name in ('iTransformer', 'TimesNet'):
            m.projection.register_forward_pre_hook(prehook)
        elif self.model_name in ('FEDformer', 'Autoformer'):
            m.decoder.projection.register_forward_pre_hook(prehook)
        elif self.model_name == 'DLinear':
            m.Linear_Seasonal.register_forward_pre_hook(prehook)
            m.Linear_Trend.register_forward_pre_hook(prehook)

    def _reduce_rep(self):
        if not self._rep_cache:
            raise RuntimeError(f'No representation captured for {self.model_name}')

        if self.model_name == 'DLinear' and len(self._rep_cache) >= 2:
            rep = torch.cat([self._rep_cache[-2], self._rep_cache[-1]], dim=-1)
        else:
            rep = self._rep_cache[-1]

        if self.model_name == 'PatchTST' and rep.ndim == 4:
            # Existing fixed reduction for PatchTST.
            rep = rep.mean(dim=-1)  # [B,N,D]

        elif self.model_name == 'TimesNet' and rep.ndim == 3:
            # Keep the existing TimesNet temporal pooling protocol unchanged.
            rep = rep.mean(dim=1)

        elif self.model_name in ('FEDformer', 'Autoformer') and rep.ndim == 3:
            # The forward-pre-hook is attached to decoder.projection, so `rep`
            # is already the true decoder latent immediately before the final
            # prediction projection. Do NOT average over the decoder/time axis:
            # signed latent features can cancel and make ||Z|| nearly zero,
            # which artificially inflates relative representation response.
            # Use deterministic fixed flattening instead (no learned pooling).
            rep = rep.contiguous().reshape(rep.shape[0], -1)

        return rep

    def _safe_patchtst_forecast(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        normed = centered / stdev
        u = normed.permute(0, 2, 1)
        enc_out, n_vars = self.model.patch_embedding(u)
        enc_out, _ = self.model.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.model.head(enc_out).permute(0, 2, 1)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out[:, -self.pred_len:, :]

    def _safe_itransformer_forecast(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        centered = x_enc - means
        stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        normed = centered / stdev
        n_vars = normed.shape[-1]
        enc_out = self.model.enc_embedding(normed, None)
        enc_out, _ = self.model.encoder(enc_out, attn_mask=None)
        dec_out = self.model.projection(enc_out).permute(0, 2, 1)[:, :, :n_vars]
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out[:, -self.pred_len:, :]

    def _forward(self, x, y=None):
        if x.ndim != 4 or x.size(-1) != 1:
            raise ValueError(f'{self.model_name} expects [B,L,N,1], got {tuple(x.shape)}')
        x3 = x[..., 0]  # [B,L,N] -- sensors are variables
        B = x3.size(0)
        hist = x3[:, -self.label_len:, :]
        dec = torch.cat(
            [hist, torch.zeros(B, self.pred_len, self.num_nodes, device=x.device, dtype=x.dtype)],
            dim=1,
        )
        self._rep_cache.clear()

        if self.model_name == 'PatchTST':
            pred3 = self._safe_patchtst_forecast(x3)
        elif self.model_name == 'iTransformer':
            pred3 = self._safe_itransformer_forecast(x3)
        else:
            pred3 = self.model(x3, None, dec, None)
            if isinstance(pred3, (tuple, list)):
                pred3 = pred3[0]

        rep = self._reduce_rep()
        pred3 = pred3[:, -self.pred_len:, :self.num_nodes]
        pred = pred3.unsqueeze(-1)
        if pred.shape != (x.shape[0], self.pred_len, self.num_nodes, 1):
            raise RuntimeError(f'{self.model_name} produced unexpected shape {tuple(pred.shape)}')
        return ModelOutput(pred=pred, representation=rep)

    def training_forward(self, x, adj, y=None):
        return self._forward(x, y)

    def inference_forward(self, x, adj, pred_len):
        if int(pred_len) != self.pred_len:
            raise ValueError(f'{self.model_name} built with pred_len={self.pred_len}, requested {pred_len}')
        return self._forward(x, None)
