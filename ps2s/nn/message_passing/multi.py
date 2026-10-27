import logging
from typing import Iterable, Sequence
from torch import Tensor, nn
from ps2s.data import BatchMolGraph
from ps2s.nn.hparams import HasHParams
from ps2s.nn.message_passing.proto import MessagePassing
logger = logging.getLogger(__name__)

class MulticomponentMessagePassing(nn.Module, HasHParams):

    def __init__(self, blocks: Sequence[MessagePassing], n_components: int, shared: bool=False):
        super().__init__()
        self.hparams = {'cls': self.__class__, 'blocks': [block.hparams for block in blocks], 'n_components': n_components, 'shared': shared}
        if len(blocks) == 0:
            raise ValueError("arg 'blocks' was empty!")
        if shared and len(blocks) > 1:
            logger.warning("More than 1 block was supplied but 'shared' was True! Using only the 0th block...")
        elif not shared and len(blocks) != n_components:
            raise ValueError(f"arg 'n_components' must be equal to `len(blocks)` if 'shared' is False! got: {n_components} and {len(blocks)}, respectively.")
        self.n_components = n_components
        self.shared = shared
        self.blocks = nn.ModuleList([blocks[0]] * self.n_components if shared else blocks)

    def __len__(self) -> int:
        return len(self.blocks)

    @property
    def output_dim(self) -> int:
        d_o = sum((block.output_dim for block in self.blocks))
        return d_o

    def forward(self, bmgs: Iterable[BatchMolGraph], V_ds: Iterable[Tensor | None]) -> list[Tensor]:
        if V_ds is None:
            return [block(bmg) for block, bmg in zip(self.blocks, bmgs)]
        else:
            return [block(bmg, V_d) for block, bmg, V_d in zip(self.blocks, bmgs, V_ds)]
