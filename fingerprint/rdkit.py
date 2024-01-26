import torch
from rdkit.Chem.EState import Fingerprinter
from fingerprint.base import Fingerprinter

class EState(Fingerprinter):
    def encode(self, smile: str) -> torch.FloatTensor:
        raise NotImplemented()

