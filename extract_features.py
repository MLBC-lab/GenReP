import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

def smiles_to_features(smiles, scaffolds=None):
    """
    Convert a SMILES string to a feature vector.
    If scaffolds are provided, scaffold-based features are included.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Calculate molecular fingerprint
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    mol_weight = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    num_h_donors = Descriptors.NumHDonors(mol)
    num_h_acceptors = Descriptors.NumHAcceptors(mol)
    
    # Scaffold-based features (optional)
    scaffold_features = []
    if scaffolds is not None and len(scaffolds) > 0:
        mol_scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
        scaffold_features = [1 if s == mol_scaffold else 0 for s in scaffolds]

    return np.concatenate([
        fingerprint, 
        [mol_weight, logp, num_rotatable_bonds, num_h_donors, num_h_acceptors],
        scaffold_features
    ])

def extract_scaffolds(smiles_list):
    """
    Extract common scaffolds from a list of SMILES strings.
    """
    scaffolds = set()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
            scaffolds.add(scaffold)
    return list(scaffolds)
