import os
import tempfile
from datetime import datetime
from typing import List, Optional
from tqdm.auto import tqdm

import numpy as np
import padelpy
import xml.etree.cElementTree as ET
import multiprocessing

import pandas as pd

from encoder.base import Encoder
from utils.batching import make_batches

FINGERPRINTS = [
        "Fingerprinter",
        "ExtendedFingerprinter",
        "EStateFingerprinter",
        "GraphOnlyFingerprinter",
        "MACCSFingerprinter",
        "PubchemFingerprinter",
        "SubstructureFingerprinter",
        "SubstructureFingerprintCount",
        "KlekotaRothFingerprinter",
        "KlekotaRothFingerprintCount",
        "AtomPairs2DFingerprinter",
        "AtomPairs2DFingerprintCount",
]

# Not implemented
# "EStateFingerprinter": {"n_dim": 79, "prefix": "EStateFP"},


class PadelpyFingerprinter(Encoder):
    PREFIX: str = ""

    def __init__(
            self,
            timeout: int = 1000,
            maxruntime: int = -1,
            threads: int = -1,
            max_batch_size: Optional[int] = None,
            verbose: bool = False
    ):
        super().__init__(verbose)

        assert self.name in FINGERPRINTS

        # Create a tmpdir for the java stuff
        self.tmp_dir = os.path.join(tempfile.gettempdir(), 'padelpy')
        if not os.path.isdir(self.tmp_dir):
            os.mkdir(self.tmp_dir)

        self.timeout = timeout
        self.maxruntime = maxruntime

        if threads == -1:
            threads = multiprocessing.cpu_count()

        if verbose:
            print(f"Using {threads} threads.")

        self.threads = threads
        if max_batch_size is None:
            max_batch_size = threads * 5
        self.max_batch_size = max_batch_size

    def _write_descriptor_xml(self, filepath: str):
        root = ET.Element("Root")
        doc = ET.SubElement(root, "Group", name="Fingerprint")
        for descriptor in FINGERPRINTS:
            ET.SubElement(doc, "Descriptor", name=descriptor, value="true" if descriptor == self.name else "false")
        tree = ET.ElementTree(root)
        tree.write(
            filepath,
            encoding='utf-8', xml_declaration=True
        )

    @property
    def columns(self) -> List[str]:
        return [
            f'{self.PREFIX}{i}'
            for i in range(1, self.n_dim+1)
        ]

    def _run_padelpy(self, smiles: List[str]):
        assert len(smiles) > 0

        # Write a file for the smiles
        timestamp = datetime.now().strftime("%H%M%S%f")
        mol_filepath = os.path.join(self.tmp_dir, f'smile_{timestamp}.smi')
        with open(mol_filepath, "w") as smi_file:
            smi_file.write('\n'.join(smiles))

        # And a file to specify the descriptor to compute
        desc_filepath = os.path.join(self.tmp_dir, f'desc_{timestamp}.xml')
        self._write_descriptor_xml(desc_filepath)

        out_filepath = os.path.join(self.tmp_dir, f'out_{timestamp}.csv')

        try:
            padelpy.padeldescriptor(
                mol_dir=mol_filepath,
                d_file=out_filepath,
                sp_timeout=self.timeout,
                maxruntime=self.maxruntime,
                threads=self.threads,
                waitingjobs=-1,
                d_2d=False,
                d_3d=False,
                config=None,
                convert3d=True,
                descriptortypes=desc_filepath,
                detectaromaticity=False,
                fingerprints=True,
                log=False,
                maxcpdperfile=0,
                removesalt=False,
                retain3d=True,
                retainorder=True,
                standardizenitro=False,
                standardizetautomers=False,
                tautomerlist=None,
                usefilenameasmolname=False,
                headless=True
            )
            fingerprint_pd = pd.read_csv(out_filepath)
            assert len(fingerprint_pd.columns) == self.n_dim + 1
            fingerprints = fingerprint_pd[self.columns].values

            if fingerprints.shape[0] != len(smiles):
                raise Exception(f"Not all fingerprints have been computed: {fingerprints.shape[0]}/{len(smiles)}")

        except Exception as e:
            print(e)
            if len(smiles) == 1:
                if self.verbose:
                    print(f"Cannot compute the fingerprint for {smiles[0]}")
                return self._nan_vector()

            mid = len(smiles) // 2
            # Strategy for failures: Divide et impera
            if self.verbose:
                print(f"Retrying with smaller batches: {mid} instead of {len(smiles)}")

            l_fingerprints = self._run_padelpy(smiles[:mid])
            r_fingerprints = self._run_padelpy(smiles[mid:])
            fingerprints = np.concatenate([l_fingerprints, r_fingerprints], 0)

        finally:
            # Cleanup
            os.remove(mol_filepath)
            os.remove(desc_filepath)
            if os.path.isfile(out_filepath):
                os.remove(out_filepath)

        return fingerprints

    def encode_all(self, smiles: List[str]) -> np.ndarray:
        assert len(smiles) > 0
        # Split computation in batches of the specified size
        smiles_batches = make_batches(smiles, self.max_batch_size)
        fingerprints = []

        if self.verbose:
            smiles_batches = tqdm(smiles_batches)

        for smiles_batch in smiles_batches:
            fingerprint_batch = self._run_padelpy(
                smiles_batch
            )
            assert fingerprint_batch.shape[0] == len(smiles_batch), f"{fingerprint_batch.shape[0]}!={len(smiles_batch)}"
            fingerprints.append(
                fingerprint_batch
            )

        fingerprints = np.concatenate(fingerprints, 0)
        assert fingerprints.shape[0] == len(smiles), f"{fingerprints.shape[0]!=len(smiles)}"

        return fingerprints

    def encode(self, smile: str) -> np.ndarray:
        return self._run_padelpy([smile])[0]

    def __repr__(self):
        return self.name+"()"

class PadelPyFingerprinter(PadelpyFingerprinter):
    PREFIX: str = "FP"
    @staticmethod
    def get_name(**kwargs) -> str:
        return "Fingerprinter"

    @staticmethod
    def get_n_dim(**kwargs) -> int:
        return 1024


class ExtendedFingerprinter(PadelpyFingerprinter):
    PREFIX: str = "ExtFP"
    @staticmethod
    def get_name(**kwargs) -> str:
        return "ExtendedFingerprinter"

    @staticmethod
    def get_n_dim(**kwargs) -> int:
        return 1024

class GraphOnlyFingerprinter(PadelpyFingerprinter):
    PREFIX: str = "GraphFP"
    @staticmethod
    def get_name(**kwargs) -> str:
        return "GraphOnlyFingerprinter"

    @staticmethod
    def get_n_dim(**kwargs) -> int:
        return 1024
class MACCSFingerprinter(PadelpyFingerprinter):
    PREFIX: str = "MACCSFP"

    @staticmethod
    def get_name(**kwargs) -> str:
        return "MACCSFingerprinter"

    @staticmethod
    def get_n_dim(**kwargs) -> int:
        return 166

class PubchemFingerprinter(PadelpyFingerprinter):
    PREFIX: str = "PubchemFP"

    @staticmethod
    def get_name(**kwargs) -> str:
        return "PubchemFingerprinter"

    @staticmethod
    def get_n_dim(**kwargs) -> int:
        return 881

    @property
    def columns(self) -> List[str]:
        return [
            f'{self.FINGERPRINTS[self.name]["prefix"]}{i}'
            for i in range(0, self.n_dim)
        ]

class SubstructureFingerprinter(PadelpyFingerprinter):
    PREFIX: str = "SubFP"

    @staticmethod
    def get_name(**kwargs) -> str:
        return "SubstructureFingerprinter"

    @staticmethod
    def get_n_dim(**kwargs) -> int:
        return 307

class SubstructureFingerprintCount(PadelpyFingerprinter):
    PREFIX: str = "SubFPC"

    @staticmethod
    def get_name(**kwargs) -> str:
        return "SubstructureFingerprintCount"

    @staticmethod
    def get_n_dim(**kwargs) -> int:
        return 307


class KlekotaRothFingerprinter(PadelpyFingerprinter):
    PREFIX: str = "KRFP"

    @staticmethod
    def get_name(**kwargs) -> str:
        return "KlekotaRothFingerprinter"

    @staticmethod
    def get_n_dim(**kwargs) -> int:
        return 4860

class KlekotaRothFingerprintCount(PadelpyFingerprinter):
    PREFIX: str = "KRFPC"

    @staticmethod
    def get_name(**kwargs) -> str:
        return "KlekotaRothFingerprintCount"

    @staticmethod
    def get_n_dim(**kwargs) -> int:
        return 4860


class AtomPairs2DFingerprinter(PadelpyFingerprinter):
    PREFIX: str = "AD2D"

    @staticmethod
    def get_name(**kwargs) -> str:
        return "AtomPairs2DFingerprinter"

    @staticmethod
    def get_n_dim(**kwargs) -> int:
        return 780

class AtomPairs2DFingerprintCount(PadelpyFingerprinter):
    ATOMS : List[str] = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'X']
    PREFIX: str = "APC2D"

    @staticmethod
    def get_name(**kwargs) -> str:
        return "AtomPairs2DFingerprintCount"

    @staticmethod
    def get_n_dim(**kwargs) -> int:
        return 780

    @property
    def columns(self) -> List[str]:
        columns = []
        for n in range(1, 11):
            for i in range(len(self.ATOMS)):
                for j in range(i, len(self.ATOMS)):
                    columns.append(
                        f"{self.FINGERPRINTS[self.name]['prefix']}{n}_{self.ATOMS[i]}_{self.ATOMS[j]}"
                    )
        return columns


