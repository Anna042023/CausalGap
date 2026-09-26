"""Frozen inference steering for additional models in the attached unified(4).

Exact original predictions are used, never a surrogate.  DCRNN is special:
its declared interface is the average decoder state, so a common offset is
applied at each decoder step. The full autoregressive decoder still executes.
"""
from contextlib import contextmanager
import torch
from core import FrozenReadout, true_base

TSLIB = {
    'PatchTST': ('model.head', 'mean_last'),
    'iTransformer': ('model.projection', 'identity'),
    'TimesNet': ('model.projection', 'mean_time'),
    'FEDformer': ('model.decoder.projection', 'flatten'),
    'DLinear': (('model.Linear_Seasonal', 'model.Linear_Trend'), 'dual'),
    'Autoformer': ('model.decoder.projection', 'flatten'),
}


def rep_from_tensor(t, mode):
    if mode == 'identity': return t
    if mode == 'mean_last': return t.mean(dim=-1)
    if mode == 'mean_time': return t.mean(dim=1)
    if mode == 'flatten': return t.contiguous().reshape(t.shape[0], -1)
    raise KeyError(mode)


def steer_tensor(t, mode, target, source):
    offset = target - source
    if mode == 'identity': return t + offset
    if mode == 'mean_last': return t + offset.unsqueeze(-1)
    if mode == 'mean_time': return t + offset.unsqueeze(1)
    if mode == 'flatten': return t + offset.reshape_as(t)
    raise KeyError(mode)


class ExtendedReadout(FrozenReadout):
    """TSLib/STEVE exact selected representation interface."""
    def __init__(self, model):
        self.model = model
        self.base = true_base(model)
        self.name = self.base.model_name
        if self.name == 'STEVE':
            target, self.mode = 'core.inv_head', 'identity'
        else:
            target, self.mode = TSLIB[self.name]
        def resolve(path):
            obj = self.base
            for seg in path.split('.'): obj = getattr(obj, seg)
            return obj
        self.modules = [resolve(path) for path in target] if isinstance(target, tuple) else [resolve(target)]
        self._capture = None
        self._probe = None
        self._calls = []

    @contextmanager
    def hook(self, replacement=None, grad_probe=False):
        if replacement is not None and grad_probe: raise ValueError('probe and replacement simultaneously')
        self._capture = None
        self._probe = None
        self._calls = []
        handles = []
        cached = {}
        def make_before(index):
            def before(module, inputs):
                t = inputs[0]
                self._calls.append(index)
                if self.mode == 'dual':
                    cached[index] = t
                    # Dual DLinear inputs are seasonal and trend latents, concatenated along last axis.
                    # Each half of the selected representation is steered independently.
                    if replacement is None and not grad_probe: return None
                    if self._probe is None and grad_probe:
                        # The probe is initialized on the first seasonal/trend invocation.
                        self._probe = torch.zeros((t.shape[0], t.shape[1], t.shape[2] * 2),
                                                  dtype=t.dtype, device=t.device, requires_grad=True)
                    value = replacement if replacement is not None else self._probe
                    half = t.shape[-1]
                    if value.shape != (t.shape[0], t.shape[1], 2 * half):
                        raise RuntimeError('DLinear selected representation shape mismatch')
                    delta = value[..., index*half:(index+1)*half]
                    new = delta if replacement is not None else (t + delta)
                    if replacement is None and not grad_probe: return None
                    return (new,)+tuple(inputs[1:])
                if index in cached: raise RuntimeError(f'{self.name}: duplicate projection invocation')
                cached[index] = t
                rep = rep_from_tensor(t, self.mode)
                if grad_probe:
                    self._probe = torch.zeros_like(rep).requires_grad_(True)
                    new = steer_tensor(t, self.mode, rep + self._probe, rep)
                    return (new,)+tuple(inputs[1:])
                if replacement is None: return None
                if replacement.shape != rep.shape:
                    raise RuntimeError(f'{self.name}: replacement mismatch {replacement.shape} != {rep.shape}')
                new = steer_tensor(t, self.mode, replacement, rep)
                return (new,)+tuple(inputs[1:])
            return before
        try:
            for index, module in enumerate(self.modules):
                handles.append(module.register_forward_pre_hook(make_before(index)))
            yield self
            expected = list(range(len(self.modules)))
            if sorted(self._calls) != expected:
                raise RuntimeError(f'{self.name}: expected {expected} readouts exactly once, got {self._calls}')
            if self.mode == 'dual':
                # DLinear registers seasonal & trend hooks; _rep_cache contains unmodified inputs.
                self._capture = torch.cat([cached[0], cached[1]],dim=-1).detach()
            else:
                self._capture = rep_from_tensor(cached[0],self.mode).detach()
        finally:
            for h in handles: h.remove()

    def forward(self, x, adj, pred_len, replacement=None, probe=False):
        with self.hook(replacement,probe):
            out = self.model.inference_forward_selective_rep(x,adj,pred_len,need_rep=True) if hasattr(self.model,'inference_forward_selective_rep') else self.model.inference_forward(x,adj,pred_len)
        if replacement is None and not probe:
            if out.representation.shape != self._capture.shape or not torch.allclose(out.representation.detach(),self._capture,rtol=2e-4,atol=2e-5):
                raise RuntimeError(f'{self.name}: selected representation mismatch with frozen readout')
        return out.pred,self._capture


class DCRNNReadout(FrozenReadout):
    """Inject at final decoder recurrent state in every autoregressive step.

    The original selected representation is the mean of the 12 decoder states.
    Adding the same delta to every decoder state changes this mean by delta,
    while preserving the original decoder recurrence, projections, and feedback.
    The state offset is defined relative to the UNMODIFIED per-step states.
    """
    def __init__(self,model):
        self.model=model
        self.base=true_base(model)
        self.name='DCRNN'
        self.module=self.base.decoder[-1]
        self._probe=None
        self._capture=None

    @contextmanager
    def hook(self,replacement=None,grad_probe=False):
        if grad_probe and replacement is not None: raise ValueError('probe vs replace')
        self._probe=None
        self._capture=None
        seq=[]
        # DCRNN pred_len calls to final decoder; original states are captured
        # before offset and averaged for consistency with model ModelOutput.
        delta=None
        def after(_mod,_args,t):
            nonlocal delta
            seq.append(t.detach())
            if replacement is None and not grad_probe: return None
            if grad_probe:
                if self._probe is None:
                    self._probe=torch.zeros_like(t,requires_grad=True)
                return t + self._probe
            if delta is None:
                # Replacement means an absolute desired mean Z, while the
                # baseline representation is intervention-specific. The caller
                # passes Z+delta. Cache the true baseline mean externally.
                if self._baseline is None: raise RuntimeError('DCRNN missing baseline mean')
                delta=replacement-self._baseline
            return t+delta
        handle=self.module.register_forward_hook(after)
        try:
            yield self
            if len(seq)!=self.base.pred_len:
                raise RuntimeError(f'DCRNN: expected {self.base.pred_len} decoder calls, got {len(seq)}')
            self._capture=torch.stack(seq,dim=1).mean(dim=1)
        finally: handle.remove()

    def forward(self,x,adj,pred_len,replacement=None,probe=False):
        if not hasattr(self,'_baseline'): self._baseline=None
        if replacement is not None:
            # Re-forward the SAME x/adj to determine the exact unperturbed mean
            # on the current mini-batch; never reuse the preceding intervention.
            with torch.no_grad():
                self.forward(x,adj,pred_len)
        with self.hook(replacement,probe):
            out=self.model.inference_forward_selective_rep(x,adj,pred_len,need_rep=True) if hasattr(self.model,'inference_forward_selective_rep') else self.model.inference_forward(x,adj,pred_len)
        if replacement is None and not probe:
            if not torch.allclose(out.representation.detach(),self._capture,rtol=2e-4,atol=2e-5):
                raise RuntimeError('DCRNN: original selected representation mismatch')
            self._baseline=self._capture.detach()
        return out.pred,self._capture


def create_readout(model):
    name=true_base(model).model_name
    if name=='DCRNN': return DCRNNReadout(model)
    if name=='STEVE' or name in TSLIB: return ExtendedReadout(model)
    return FrozenReadout(model)
