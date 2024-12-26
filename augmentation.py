import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolToSmiles, TautomerEnumerator
from imblearn.over_sampling import SMOTE

def smote_augmentation(X, y):
    """
    Applies SMOTE to balance the dataset. 
    X is expected to be a feature matrix, y the labels.
    Returns the resampled feature matrix X_res and label vector y_res.
    """
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def tautomer_enumeration_augmentation(smiles_list, max_tautomers=5):
    """
    Enumerate tautomers for each SMILES to expand the dataset with 
    potential alternative forms. Returns a list of new SMILES strings.
    """
    new_smiles = []
    enumerator = TautomerEnumerator()
    
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        # Generate tautomeric forms
        tautomers = enumerator.Enumerate(mol)
        # Convert them to SMILES and keep up to max_tautomers unique forms
        unique_smi = set()
        for tauto_mol in tautomers:
            t_smi = MolToSmiles(tauto_mol, isomericSmiles=True)
            unique_smi.add(t_smi)
            if len(unique_smi) >= max_tautomers:
                break
        
        # Add these new forms to new_smiles
        new_smiles.extend(list(unique_smi))
    
    return new_smiles

def random_molecular_substitution(smiles_list, substitution_fn=None):
    """
    Perform a 'random' substitution on each molecule, if substitution_fn is provided.
    Otherwise, attempts a trivial substitution (e.g., remove one ring substituent).
    This function can be made more complex to do real chemistry transformations.
    """
    augmented_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        # If a custom substitution function is provided, apply it
        if substitution_fn:
            new_mol = substitution_fn(mol)
            if new_mol is not None:
                augmented_smiles.append(MolToSmiles(new_mol))
            else:
                augmented_smiles.append(smi)
        else:
            # Basic example: remove first atom if possible
            if mol.GetNumAtoms() > 5: 
                emol = Chem.EditableMol(mol)
                emol.RemoveAtom(0)  # removing first atom
                new_mol = emol.GetMol()
                augmented_smiles.append(MolToSmiles(new_mol))
            else:
                augmented_smiles.append(smi)
                
    return augmented_smiles
