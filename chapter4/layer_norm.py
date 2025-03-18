import torch
from torch import nn


class LayerNorm(nn.Module):
    """
    operates on the last dimension of the input tensor x
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # small value to prevent division by zero
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
