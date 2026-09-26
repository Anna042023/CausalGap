"""Frozen-readout steering bridge for CausalGap's actual PyTorch Track-B adapters.

Important: a ModelOutput.representation is not necessarily a callable prediction head.
We intercept the *actual pre-head tensor*, check it matches the declared representation,
then add a norm-controlled perturbation. Unsupported models fail closed.
"""
from contextlib import contextmanager
import numpy as np
import torch

# Format describes the tensor at the final readout's forward PRE-hook.
# bnc: [B,N,C], b1nc: [B,1,N,C], bcnt: [B,C,N,T], b1nd: [B,1,N,D].
READOUT = {
    'STGCN': ('output.fc', 'b1nc'),
    'ASTGNN': ('head', 'bnc'),
    'GWNet': ('end2', 'bcnt'),
    'AGCRN': ('head', 'b1nd'),
    'DCGCN': ('head', 'bnc'),
    'ST-Norm': ('end2', 'bcnt'),
    'CCHMM': ('head', 'bnc'),
    'CausalGRIT': ('head', 'bnc'),
    'TF-MVGNN': ('head', 'bnc'),
}


def true_base(model):
    return model.base if hasattr(model, 'base') else model


def to_rep(t, mode):
    if mode == 'bnc': return t
    if mode in ('b1nc', 'b1nd'): return t[:, 0]
    if mode == 'bcnt': return t[:, :, :, -1].permute(0, 2, 1).contiguous()
    raise ValueError(mode)


def replace_rep(t, mode, rep):
    if mode == 'bnc': return rep
    if mode in ('b1nc', 'b1nd'): return rep.unsqueeze(1)
    if mode == 'bcnt':
        y = t.clone()
        y[:, :, :, -1] = rep.permute(0, 2, 1)
        return y
    raise ValueError(mode)


class FrozenReadout:
    def __init__(self, model):
        self.model = model
        base = true_base(model)
        name = base.model_name
        if name not in READOUT:
            raise NotImplementedError(
                f'{name} lacks a verified differentiable frozen-readout bridge. '
                'Do not count it as an H2 completion or use an approximate direction MSE.')
        target, self.mode = READOUT[name]
        module = base
        for k in target.split('.'):
            module = getattr(module, k)
        self.module = module
        self.name = name
        self._replacement = None
        self._capture = None
        self._calls = 0

    @contextmanager
    def hook(self, replacement=None, grad_probe=False):
        self._replacement = replacement
        self._capture = None
        self._calls = 0
        def before(_module, inputs):
            self._calls += 1
            if self._calls != 1:
                raise RuntimeError('Ambiguous final readout: invoked more than once per forward')
            x = inputs[0]
            rep = to_rep(x, self.mode)
            self._capture = rep.detach()
            if grad_probe:
                if replacement is not None:
                    raise ValueError('grad_probe and replacement are mutually exclusive')
                probe = rep.detach().requires_grad_(True)
                self._probe = probe
                return (replace_rep(x, self.mode, probe),) + tuple(inputs[1:])
            if replacement is None: return None
            if tuple(replacement.shape) != tuple(rep.shape):
                raise RuntimeError(f'Head input shape {tuple(rep.shape)} != replacement {tuple(replacement.shape)}')
            return (replace_rep(x, self.mode, replacement),) + tuple(inputs[1:])
        handle = self.module.register_forward_pre_hook(before)
        try:
            yield self
            if self._calls != 1:
                raise RuntimeError(f'Final readout ran {self._calls} times (expected 1)')
        finally:
            handle.remove()
            self._replacement = None

    def forward(self, x, adj, pred_len, replacement=None, probe=False):
        with self.hook(replacement=replacement, grad_probe=probe):
            if hasattr(self.model, 'inference_forward_selective_rep'):
                out = self.model.inference_forward_selective_rep(x, adj, pred_len, need_rep=True)
            else:
                out = self.model.inference_forward(x, adj, pred_len)
            # On a steered forward ModelOutput.representation remains the unperturbed
            # encoder output for some adapters; don't use it as a steered head input.
            captured = self._capture
            if not probe and replacement is None:
                returned = out.representation
                if self.name == 'STGCN' and returned.ndim == 4:
                    returned = returned[:, 0]
                if returned.shape != captured.shape or not torch.allclose(
                    returned.detach(), captured, rtol=2e-4, atol=2e-5):
                    raise RuntimeError(
                        f'{self.name}: ModelOutput.representation != frozen head input; '
                        'cannot claim steering at the selected representation interface.')
            return out.pred, captured


def equal_norm(direction, z, ratio):
    if direction.shape != z.shape:
        raise ValueError((direction.shape, z.shape))
    d = direction.reshape(len(direction), -1)
    zp = z.reshape(len(z), -1)
    norms = d.norm(dim=1, keepdim=True).clamp_min(1e-12)
    # A *single*, prespecified norm policy shared by all directions.
    target = float(ratio) * zp.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return (d / norms * target).reshape_as(z)


def masked_mae(pred, y, mask):
    valid = mask & torch.isfinite(pred) & torch.isfinite(y)
    if not bool(valid.any()):
        raise RuntimeError('All targets masked')
    return (pred[valid] - y[valid]).abs().mean()


def to_eval(pred, scaler, model):
    if str(getattr(model, 'output_space', 'normalized')).lower() == 'original':
        return pred
    return pred * float(scaler.std) + float(scaler.mean)


def freeze(model):
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('Forecast model did not freeze')
    return model


def intervention_response(z0, zt, dist):
    from causalgap.metrics import dmis_generalized
    z0 = z0.detach().cpu().numpy().reshape(len(z0), -1)
    zt = zt.detach().cpu().numpy().reshape(len(zt), -1)
    dmis, dz, cosine, magnitude, dt = dmis_generalized(z0, zt, dist, alpha=.5)
    if not np.all(np.isfinite([dmis, dz, cosine, magnitude, dt])):
        raise RuntimeError('Non-finite H2 diagnostic coordinates')
    return dict(DMIS=float(dmis), DZ=float(dz), cosine=float(cosine),
                magnitude=float(magnitude), dT=float(dt))


def construct_features(zt, z0, rd, response, add_dmis):
    """Exactly information-matched: both see raw zt,z0,RD,DZ,dT,cos,mag.
    Baseline has an identically-sized *constant-zero* DMIS coordinate.
    """
    n = len(zt)
    zt = zt.reshape(n, -1)
    z0 = z0.reshape(n, -1)
    scalar = torch.tensor([rd, response['DZ'], response['dT'],
                           response['cosine'], response['magnitude'],
                           response['DMIS'] if add_dmis else 0.],
                          dtype=zt.dtype, device=zt.device).expand(n, -1)
    return torch.cat([zt, z0, scalar], dim=-1)


class SteeringHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden=32):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden),
                                       torch.nn.Tanh(), torch.nn.Linear(hidden, output_dim))
    def forward(self, x): return self.net(x)


class SteeringHeadFiLM(torch.nn.Module):
    """Matched-capacity conditional head: baseline receives DMIS=0.

    Last input coordinate is the explicitly standardized DMIS; all other
    coordinates, including the raw DMIS ingredients, are shared.
    """
    def __init__(self, input_dim, output_dim, hidden=32):
        super().__init__()
        self.content = torch.nn.Linear(input_dim-1, hidden)
        self.modulation = torch.nn.Linear(1, hidden)
        self.readout = torch.nn.Linear(hidden, output_dim)

    def forward(self, x):
        base = torch.tanh(self.content(x[:, :-1]))
        gate = 1.0 + 0.5 * torch.tanh(self.modulation(x[:, -1:]))
        return self.readout(base * gate)
