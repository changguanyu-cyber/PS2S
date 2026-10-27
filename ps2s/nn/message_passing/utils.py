from rdkit import Chem

def generate_smiles_variants(smiles, num_variants=3):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid SMILES string')
    canonical = Chem.MolToSmiles(mol, canonical=True)
    variants = set()
    while len(variants) < num_variants:
        rand_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        variants.add(rand_smiles)
    return (canonical, list(variants))
