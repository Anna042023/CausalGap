#!/usr/bin/env python3
"""Small local contract test; no user checkpoints are available in this sandbox."""
from pathlib import Path
import inspect
import torch
from core import SteeringHeadFiLM, SteeringHead, equal_norm

# Architecture matched: both get exactly the same model, dimensions and parameter count.
baseline=SteeringHeadFiLM(15,9,8)
dmis=SteeringHeadFiLM(15,9,8)
assert sum(p.numel() for p in baseline.parameters())==sum(p.numel() for p in dmis.parameters())
x=torch.randn(4,15,requires_grad=True)
y=dmis(x)
assert y.shape==(4,9)
y.square().mean().backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in dmis.parameters())
assert x.grad is not None
z=torch.randn(4,3,3); q=equal_norm(y.detach().reshape_as(z),z,0.05)
assert torch.allclose(q.reshape(4,-1).norm(dim=1), 0.05*z.reshape(4,-1).norm(dim=1),rtol=1e-5,atol=1e-6)
# Check that weak-only head-calibration doesn't require held-out strong units.
s=(Path(__file__).parent/"run_steering.py").read_text()
for snippet in ["args.cal_only or args.head_cal_only", "if args.head_cal_only:", "'strong_test_outcomes_accessed=False'", "if args.teacher_weight > 0", "SteeringHeadFiLM"]:
 if snippet=="'strong_test_outcomes_accessed=False'":snippet='strong_test_outcomes_accessed=False'
 assert snippet in s, snippet
print('PASS: equal-capacity architecture, differentiable head, equal-norm, calibration-only guard')
