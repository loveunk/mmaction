import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registry import SEGMENTAL_CONSENSUSES

class _SimpleConsensus(torch.autograd.Function):
    """Simplest segmental consensus module"""

    '''def __init__(self,
                 consensus_type='avg',
                 dim=1):
        super(_SimpleConsensus, self).__init__()

        assert consensus_type in ['avg']
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None'''

    @staticmethod
    def forward(ctx, x, consensus_type='avg', dim=1):
        ctx.shape = x.size()
        ctx.consensus_type = consensus_type
        ctx.dim = dim
        if ctx.consensus_type == 'avg':
            output = x.mean(dim=ctx.dim, keepdim=True)
        else:
            output = None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.consensus_type == 'avg':
            grad_in = grad_output.expand(ctx.shape) / float(ctx.shape[ctx.dim])
        else:
            grad_in = None
        return grad_in, None, None


@SEGMENTAL_CONSENSUSES.register_module
class SimpleConsensus(nn.Module):
    def __init__(self, consensus_type, dim=1):
        super(SimpleConsensus, self).__init__()

        assert consensus_type in ['avg']
        self.consensus_type = consensus_type
        self.dim = dim

    def init_weights(self):
        pass

    def forward(self, input):
        return _SimpleConsensus.apply(input, self.consensus_type, self.dim)