#!/usr/bin/env python3
"""Lightweight numerical tests for the Section 5.7 transformations."""
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_invariance_checks import givens_rotate_pair, permute_pair, scale_pair, centered_norm


def response(z0,zk,alpha=.5,eps=1e-12):
    a=np.asarray(z0,float).reshape(len(z0),-1); b=np.asarray(zk,float).reshape(len(zk),-1)
    a=a-a.mean(0,keepdims=True); b=b-b.mean(0,keepdims=True)
    av=a.ravel(); bv=b.ravel(); den=np.linalg.norm(av)*np.linalg.norm(bv)
    cos=1.0 if den<=eps and np.linalg.norm(av-bv)<=eps else (0.0 if den<=eps else float(np.clip(np.dot(av,bv)/den,-1,1)))
    cd=max(0.0,1.0-cos)
    na=np.linalg.norm(a); diff=np.linalg.norm(b-a)
    mg=0.0 if na<=eps and diff<=eps else (float('inf') if na<=eps else max(0.0,float(diff/na)))
    return alpha*cd+(1-alpha)*mg

rng=np.random.default_rng(57)
z0=rng.normal(size=(32,1024))
zk=z0+0.08*rng.normal(size=z0.shape)
base=response(z0,zk)

r0,rk,_=givens_rotate_pair(z0,zk,np.random.default_rng(1),64)
p0,pk,_=permute_pair(z0,zk,np.random.default_rng(2))
s0,sk,meta=scale_pair(z0,zk,np.random.default_rng(3),.25,4.0)

for name,a,b in [('rotation',r0,rk),('permutation',p0,pk),('scaling',s0,sk)]:
    v=response(a,b)
    err=abs(v-base)
    print(f'{name}: base={base:.12g} transformed={v:.12g} absdiff={err:.3g}')
    assert err < 1e-11, (name,err)

assert centered_norm(z0)>1e-12 and centered_norm(s0)>1e-12
print('PASS: rotation, permutation, and admissible positive scaling preserve DMIS response numerically.')
