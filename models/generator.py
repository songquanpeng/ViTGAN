import torch
from einops.layers.torch import Rearrange
from torch import nn

from models.layers import SLNTransformer, LinearWithSinActivation


class Generator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        image_height, image_width = args.image_size, args.image_size
        patch_height, patch_width = args.patch_size, args.patch_size
        assert image_height % patch_height == 0 and image_width % patch_width == 0
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        image_dim = args.image_dim
        patch_dim = image_dim * patch_height * patch_width
        self.pool = args.pool
        dim = args.g_dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        # self.dropout = nn.Dropout(emb_dropout)
        # TODO: improve this mapping network
        self.mapping_network = nn.Sequential(
            nn.Linear(args.z_dim, args.g_dim)
        )
        self.transformer = SLNTransformer(args.g_dim, dim, depth=args.g_blocks, heads=args.g_attention_head_num,
                                          dim_head=args.g_attention_head_dim, mlp_dim=dim,
                                          dropout=args.g_transformer_dropout)

        self.to_rgb = nn.Sequential(
            LinearWithSinActivation(dim, dim * 2),
            LinearWithSinActivation(dim * 2, args.patch_size * args.patch_size * image_dim)
        )

    def forward(self, z):
        w = self.mapping_network(z)
        x = self.transformer(self.pos_embedding.expand(z.shape[0], -1, -1), w)
        x = self.to_rgb(x)
        x = x.view(z.shape[0], self.args.image_dim, self.args.image_size, self.args.image_size)
        return x
