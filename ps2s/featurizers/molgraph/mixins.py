from dataclasses import dataclass, field
from rdkit.Chem.rdchem import Atom, Bond
from ps2s.featurizers.atom import MultiHotAtomFeaturizer
from ps2s.featurizers.base import VectorFeaturizer
from ps2s.featurizers.bond import MultiHotBondFeaturizer

@dataclass
class _MolGraphFeaturizerMixin:
    atom_featurizer: VectorFeaturizer[Atom] = field(default_factory=MultiHotAtomFeaturizer.v2)
    bond_featurizer: VectorFeaturizer[Bond] = field(default_factory=MultiHotBondFeaturizer)

    def __post_init__(self):
        self.atom_fdim = len(self.atom_featurizer)
        self.bond_fdim = len(self.bond_featurizer)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.atom_fdim, self.bond_fdim)
