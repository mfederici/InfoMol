from abc import abstractmethod

import numpy as np
import rdkit.Chem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdMolDescriptors import CalcNumAtoms, CalcNumHeavyAtoms, CalcNumRings, CalcNumAromaticRings

from encoder.base import Encoder, FINGERPRINT_ENCODER, PROPERTY_ENCODER


class RDKitFingerprinter(Encoder):
    type: str = FINGERPRINT_ENCODER
    @abstractmethod
    def _encode_mol(self, mol: rdkit.Chem.Mol):
        raise NotImplementedError()

    def _encode_one(self, smile: str) -> np.ndarray:
        mol = MolFromSmiles(smile)
        if mol is None:
            return self._nan_vector()
        else:
            return self._encode_mol(mol)
class EStateFingerprinter(RDKitFingerprinter):
    @staticmethod
    def get_name(**kwargs) -> str:
        return "EStateFingerprinter"

    @staticmethod
    def get_n_dim(**kwargs) -> int:
        return 79

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.columns = [
            f'EState {i}'
            for i in range(1, self.n_dim+1)
        ]

    def _encode_mol(self, mol: rdkit.Chem.Mol) -> np.ndarray:
        fingerprint = FingerprintMol(mol)[0]
        return fingerprint


class MolecularFormula(RDKitFingerprinter):
    MAX_ATOMS: int  = 119
    type: str = PROPERTY_ENCODER

    @staticmethod
    def get_name(**kwargs) -> str:
        return "MolecularFormula"

    @staticmethod
    def get_n_dim(**kwargs) -> int:
        return MolecularFormula.MAX_ATOMS

    def _encode_mol(self, mol: rdkit.Chem.Mol) -> np.ndarray:
        onehot = np.eye(self.MAX_ATOMS)
        chem_formula = np.zeros(self.MAX_ATOMS)
        for atom in mol.GetAtoms():
            chem_formula += onehot[atom.GetAtomicNum()]

        hydrogens = CalcNumAtoms(mol) - CalcNumHeavyAtoms(mol)
        chem_formula[1] = hydrogens
        return chem_formula

class NumAtoms(RDKitFingerprinter):
    type: str = 'property'
    @staticmethod
    def get_name(**kwargs) -> str:
        return "NumAtoms"

    @staticmethod
    def get_n_dim(**kwargs) -> int:
        return 1

    def _encode_mol(self, mol: rdkit.Chem.Mol) -> np.ndarray:
        if mol is None:
            return self._nan_vector()

        return np.array([CalcNumAtoms(mol)]).reshape(1)


class NumHeavyAtoms(RDKitFingerprinter):
    type: str = PROPERTY_ENCODER
    @staticmethod
    def get_name(**kwargs) -> str:
        return "NumHeavyAtoms"
    @staticmethod
    def get_n_dim(**kwargs) -> int:
        return 1

    def _encode_mol(self, mol: rdkit.Chem.Mol) -> np.ndarray:
        return np.array([CalcNumHeavyAtoms(mol)]).reshape(1)


class NumRings(RDKitFingerprinter):
    type: str = PROPERTY_ENCODER
    @staticmethod
    def get_n_dim(**kwargs) -> int:
        return 1
    @staticmethod
    def get_name(**kwargs) -> str:
        return "NumRings"
    def _encode_mol(self, mol: rdkit.Chem.Mol) -> np.ndarray:
        return np.array([CalcNumRings(mol)]).reshape(1)

class NumAromaticRings(RDKitFingerprinter):
    type: str = PROPERTY_ENCODER
    @staticmethod
    def get_n_dim(**kwargs) -> int:
        return 1

    @staticmethod
    def get_name(**kwargs) -> str:
        return "NumAromaticRings"

    def _encode_mol(self, mol: rdkit.Chem.Mol) -> np.ndarray:
        return np.array([CalcNumAromaticRings(mol)]).reshape(1)

class MolecularWeight(RDKitFingerprinter):
    type: str = PROPERTY_ENCODER
    @staticmethod
    def get_n_dim(**kwargs) -> int:
        return 1

    @staticmethod
    def get_name(**kwargs) -> str:
        return "MolecularWeight"

    def _encode_mol(self, mol: rdkit.Chem.Mol) -> np.ndarray:
        return np.array([MolWt(mol)]).reshape(1)
