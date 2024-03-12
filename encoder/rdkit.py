import numpy as np
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Chem import MolFromSmiles
from encoder.base import Encoder

class EStateFingerprinter(Encoder):
    @staticmethod
    def get_name(**kwargs) -> str:
        assert len(kwargs) == 0
        return "EStateFingerprinter"

    @staticmethod
    def get_n_dim(**kwargs) -> int:
        assert len(kwargs) == 0
        return 79

    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)
        self.columns = [
            f'EState {i}'
            for i in range(1, self.n_dim+1)
        ]

    def encode(self, smile: str) -> np.ndarray:
        mol = MolFromSmiles(smile)
        if mol is None:
            fingerprint = self._nan_vector()[0]
        else:
            fingerprint = FingerprintMol(mol)[0]

        assert len(fingerprint) == self.n_dim
        return fingerprint

