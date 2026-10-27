import lightning
from packaging.version import Version
from ps2s.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer
DEFAULT_ATOM_FDIM, DEFAULT_BOND_FDIM = SimpleMoleculeMolGraphFeaturizer().shape
DEFAULT_HIDDEN_DIM = 600
DECODER_HIDDEN_DIM = 512
DECODER_HIDDEN_LAYERS = 3
DECODER_OUTPUT_DIM = 10
MAX_ATOM_NUM = 40
ATOM_FEATURES = 72
LIGHTNING_26_COMPAT_ARGS = {}
if Version(lightning.__version__) >= Version('2.6'):
    LIGHTNING_26_COMPAT_ARGS['weights_only'] = False
