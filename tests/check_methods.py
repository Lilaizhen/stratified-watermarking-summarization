"""Small invariance and calibration checks; no model downloads."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
import numpy as np
from headmass_processor import partition,mass_preserving_tilt
from analysis import threshold,auc

torch.manual_seed(17)
s=torch.randn(4,127);green=torch.rand_like(s)>.5
s[0,:]=-torch.inf;s[0,4]=1  # Degenerate one-candidate distribution.
for rho in [.9,.95,.98]:
    for delta in [0.,4.,12.]:
        q=mass_preserving_tilt(s,green,[],rho,delta).softmax(-1);p=s.softmax(-1)
        torch.testing.assert_close(q.sum(-1),torch.ones(4))
        for b in partition(s,[],rho):torch.testing.assert_close((p*b).sum(-1),(q*b).sum(-1),atol=2e-6,rtol=2e-5)
        if delta==0:torch.testing.assert_close(q,p)
# One-color regions are invariant, including nontrivial distributions.
for color in [False,True]:torch.testing.assert_close(mass_preserving_tilt(s,torch.full_like(green,color),[],.98,12).softmax(-1),s.softmax(-1),atol=2e-6,rtol=2e-5)
assert threshold(np.arange(200))==198
assert threshold(np.arange(8))==float('inf')
assert np.mean(np.array([198,198,199])>198)==1/3
assert auc(np.array([-np.inf,1]),np.array([-np.inf,0]))==.625
print('PASS: regional mass, normalization, zero bias, one-color regions, calibration ties and unscorable AUC.')
