from typing import NamedTuple
import numpy as np

class MolGraph(NamedTuple):
    V: np.ndarray
    E: np.ndarray
    edge_index: np.ndarray
    rev_edge_index: np.ndarray
    distance_matrix: np.ndarray
    smiles: str
