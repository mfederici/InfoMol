from rdkit.Chem.rdMolDescriptors import CalcNumRings, CalcNumAtoms, CalcNumAromaticRings, CalcNumHeavyAtoms
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Descriptors import MolWt
import numpy as np
from rdkit import Chem


def molecular_formula(smiles: str, max_atom=119) -> np.ndarray:
    mol = MolFromSmiles(smiles)
    onehot = np.eye(max_atom)
    chem_formula = np.zeros(max_atom)
    for atom in mol.GetAtoms():
        chem_formula += onehot[atom.GetAtomicNum()]

    hydrogens = CalcNumAtoms(mol) - CalcNumHeavyAtoms(mol)
    chem_formula[1] = hydrogens
    return chem_formula

def num_atoms(smiles: str) -> int:
    mol = MolFromSmiles(smiles)
    return CalcNumAtoms(mol)


def num_heavy_atoms(smiles: str) -> int:
    mol = MolFromSmiles(smiles)
    return CalcNumHeavyAtoms(mol)

def num_rings(smiles: str) -> int:
    mol = MolFromSmiles(smiles)
    return CalcNumRings(mol)

def num_aromatic_rings(smiles: str) -> int:
    mol = MolFromSmiles(smiles)
    return CalcNumAromaticRings(mol)

def molecular_weight(smiles: str) -> int:
    mol = MolFromSmiles(smiles)
    return MolWt(mol)
