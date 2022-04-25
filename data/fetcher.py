import torch
from munch import Munch


class Fetcher:
    def __init__(self, loader, args):
        self.loader = loader
        self.device = torch.device(args.device)
        self.latent_dim = args.z_dim

    def __iter__(self):
        return self

    def __next__(self):
        try:
            x = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x = next(self.iter)

        z = torch.randn(x.size(0), self.latent_dim)
        inputs = Munch(x=x, z=z)

        return Munch({k: v.to(self.device) for k, v in inputs.items()})
