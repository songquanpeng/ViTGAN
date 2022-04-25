import torch
import torch.nn as nn

from models.vit import Attention, FeedForward, PreNorm


class SLNPreNorm(nn.Module):
    # Self Modulated LayerNorm
    def __init__(self, w_dim, f_dim, fn):
        super().__init__()
        self.affine_transform = nn.Linear(w_dim, f_dim * 2)
        self.norm = nn.LayerNorm(f_dim)
        self.fn = fn

    def forward(self, f, w, **kwargs):
        w = self.affine_transform(w)  # shape: [N, D]
        gamma, beta = torch.chunk(w, chunks=2, dim=1)
        # TODO: check if this is correct
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        f = (1 + gamma) * self.norm(f) + beta
        return self.fn(self.norm(f), **kwargs)


class SLNTransformer(nn.Module):
    # SLN means self-modulated layer norm
    def __init__(self, w_dim, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                SLNPreNorm(w_dim, dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                SLNPreNorm(w_dim, dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, w):
        for attn, ff in self.layers:
            x = attn(x, w) + x
            x = ff(x, w) + x
        return x


class LinearWithSinActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.sin(self.linear(x))

