from enum import auto
import logging
from torch import nn
from ps2s.utils.utils import EnumMapping
logger = logging.getLogger(__name__)

class Activation(EnumMapping):
    RELU = auto()
    LEAKYRELU = auto()
    PRELU = auto()
    TANH = auto()
    ELU = auto()

def get_activation_function(activation: str | nn.Module | Activation) -> nn.Module:
    if activation == 'selu':
        logger.warning('Accepting activation="selu" for backward compatibility.')
        activation = nn.modules.activation.SELU()
    if isinstance(activation, nn.Module):
        if isinstance(activation, nn.modules.activation.SELU):
            logger.warning('PS2S does not support self-normalization. Using SELU activation is not enough to achieve it.')
        return activation
    match Activation.get(activation):
        case Activation.RELU:
            return nn.ReLU()
        case Activation.LEAKYRELU:
            return nn.LeakyReLU(0.1)
        case Activation.PRELU:
            return nn.PReLU()
        case Activation.TANH:
            return nn.Tanh()
        case Activation.ELU:
            return nn.ELU()
        case _:
            raise RuntimeError('unreachable code reached!')
