import torch as th
from torch import nn
import models.model_parts as mp
twopi = 2*th.pi

def FastFourierReal(x, dim=1):
    N = x.shape[dim]
    n = th.arange(N, device=x.device)
    k = th.arange(N, device=x.device)
    term = -twopi*n[None]*k[:,None] / N
    if dim == 1:
        term = term[None,...,None]
        x_ = x[:,None]
        sumdim = 2
    elif dim == 2:
        term = term[None,None]
        x_ = x[...,None,:]
        sumdim = -1
    
    return (x_*th.cos(term)).sum(sumdim)

class FFLayer(nn.Module):
    def __init__(self):
        super(FFLayer, self).__init__()
    
    def forward(self, x):
        return FastFourierReal(x, dim=1)

class FNetBlock(nn.Module):
    def __init__(self, ffn_dict):
        super(FNetBlock, self).__init__()
        self.norm1 = nn.LayerNorm(ffn_dict['indim'])
        self.norm2 = nn.LayerNorm(ffn_dict['indim'])
        self.fourier_layer = FFLayer()
        self.ffn = mp.FFN(**ffn_dict)

    def forward(self, x, **kwargs):
        out = x + self.fourier_layer(x)
        out = self.norm1(out)
        out = self.ffn(out)['out']
        out = self.norm2(out)

        return {'out': out, 'other': None}

