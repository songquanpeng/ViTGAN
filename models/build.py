import copy

from munch import Munch

from models.discriminator import Discriminator
from models.generator import Generator


# If you use pretrained models here, make sure they are using the `eval` mode.


def build_model(args):
    generator = Generator(args)
    discriminator = Discriminator(args)

    generator_ema = copy.deepcopy(generator)

    nets = Munch(generator=generator, discriminator=discriminator)
    nets_ema = Munch(generator=generator_ema)

    return nets, nets_ema
