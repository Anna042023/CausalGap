#!/usr/bin/env python3
"""Deterministic mathematical checks; NOT paper evidence."""
import numpy as np
from response_metrics import compute_measures, linear_cka_similarity, cca_dim_cap

rng=np.random.default_rng(2027)
x=rng.normal(size=(32,10000)).astype(np.float32)
ident=compute_measures(x,x)
cols=['resp_1mcka','resp_1msvcca','resp_rsa_distance','resp_1mpwcca']
assert all(abs(float(ident[k])) < 1e-6 for k in cols),ident
assert cca_dim_cap(32)==15
assert ident['svcca_kx']<=15 and ident['svcca_ky']<=15
assert ident['pwcca_kx']<=15 and ident['pwcca_ky']<=15

noise=compute_measures(x,x+rng.normal(size=x.shape).astype(np.float32)*0.2)
assert all(np.isfinite(noise[k]) for k in cols),noise
assert noise['resp_rsa_distance'] > ident['resp_rsa_distance']
assert noise['resp_1msvcca'] > 1e-8, noise
assert noise['resp_1mpwcca'] > 1e-8, noise

small=rng.normal(size=(20,13)); small=small-small.mean(axis=0)
from_gram=linear_cka_similarity(small,small*2.3)
assert abs(from_gram-1.0) < 1e-10
bad=False
try: compute_measures(x,x[:31])
except ValueError: bad=True
assert bad,'row-mismatched matrices must be rejected'
print('PASS: identity, perturbation, CCA centered-rank cap, Gram CKA and row-alignment safeguards')
print('Protocol: PCA variance target=0.99; CCA max dim=floor((n_rows-1)/2); n=32 -> 15')
