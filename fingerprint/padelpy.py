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

from fingerprint.base import Fingerprinter


class PadelpyFingerprinter(Fingerprinter):
    DESCRIPTORS = {
        "Fingerprinter": {"n_dim": 1024, "prefix": "FP"},
        "ExtendedFingerprinter": {"n_dim": 1024, "prefix": "ExtFP"},
        "EStateFingerprinter": {"n_dim": 79, "prefix": "EStateFP"},
        "GraphOnlyFingerprinter": {"n_dim": 1024, "prefix": "GraphFP"},
        "MACCSFingerprinter": {"n_dim": 166, "prefix": "MACCSFP"},
        "PubchemFingerprinter": {"n_dim": 881, "prefix": "PubchemFP"},
        "SubstructureFingerprinter": {"n_dim": 307, "prefix": "SubFP"},
        "SubstructureFingerprintCount": {"n_dim": 307, "prefix": "SubFPC"},
        "KlekotaRothFingerprinter": {"n_dim": 4860, "prefix": "KRFP"},
        "KlekotaRothFingerprintCount": {"n_dim": 4860, "prefix": "KRFPC"},
        "AtomPairs2DFingerprinter": {"n_dim": 780, "prefix": "AD2D"},
    }

    def __init__(
            self,
            name: str,
            timeout: int = 500,
            maxruntime: int = -1,
            threads: int = -1,
            max_batch_size: Optional[int] = None,
            verbose: bool = True
    ):
        if not (name in self.DESCRIPTORS):
            raise ValueError(f"Please specify one of {self.DESCRIPTORS.keys()}")
        self.n_dim = self.DESCRIPTORS[name]["n_dim"]
        self.name = name
        super().__init__(verbose)

        # Create a tmpdir for the java stuff
        self.tmp_dir = os.path.join(tempfile.gettempdir(), 'padelpy')
        if not os.path.isdir(self.tmp_dir):
            os.mkdir(self.tmp_dir)

        self.columns = [f'{self.DESCRIPTORS[name]["prefix"]}{i}' for i in range(1, self.n_dim + 1)]
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
        for descriptor in self.DESCRIPTORS:
            ET.SubElement(doc, "Descriptor", name=descriptor, value="true" if descriptor == self.name else "false")
        tree = ET.ElementTree(root)
        tree.write(
            filepath,
            encoding='utf-8', xml_declaration=True
        )

    def _nan_vector(self) -> np.ndarray:
        empty_vector = np.empty(1, self.n_dim)
        empty_vector[:] = np.nan
        return empty_vector

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
        except Exception as e:
            print(e)
            if len(smiles) == 1:
                return self._nan_vector()

            # Strategy for failures: Divide et impera
            if self.verbose:
                print("Retrying with smaller batches")

            mid = len(smiles) // 2
            l_fingerprints = self.encode_all(smiles[:mid])
            r_fingerprints = self.encode_all(smiles[mid:])
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
        fingerprints = []
        n_batches = int(np.ceil(len(smiles) / self.max_batch_size))
        batches = range(n_batches)

        if self.verbose:
            batches = tqdm(batches)

        for batch in batches:
            fingerprints.append(
                self._run_padelpy(
                    smiles[batch * self.max_batch_size:(batch + 1) * self.max_batch_size]
                )
            )

        return np.concatenate(fingerprints, 0)

    def encode(self, smile: str) -> np.ndarray:
        return self._run_padelpy([smile])[0]

    def __repr__(self):
        return self.name+"()"


class ExtendedFingerprinter(PadelpyFingerprinter):
    def __init__(self, *args, **kwargs):
        super().__init__(name="ExtendedFingerprinter", *args, **kwargs)


class EStateFingerprinter(PadelpyFingerprinter):
    def __init__(self, *args, **kwargs):
        super().__init__(name="EStateFingerprinter", *args, **kwargs)


class GraphOnlyFingerprinter(PadelpyFingerprinter):
    def __init__(self, *args, **kwargs):
        super().__init__(name="GraphOnlyFingerprinter", *args, **kwargs)


class MACCSFingerprinter(PadelpyFingerprinter):
    def __init__(self, *args, **kwargs):
        super().__init__(name="MACCSFingerprinter", *args, **kwargs)


class PubchemFingerprinter(PadelpyFingerprinter):
    def __init__(self, *args, **kwargs):
        super().__init__(name="PubchemFingerprinter", *args, **kwargs)


class SubstructureFingerprinter(PadelpyFingerprinter):
    def __init__(self, *args, **kwargs):
        super().__init__(name="SubstructureFingerprinter", *args, **kwargs)


class SubstructureFingerprintCount(PadelpyFingerprinter):
    def __init__(self, *args, **kwargs):
        super().__init__(name="SubstructureFingerprintCount", *args, **kwargs)


class KlekotaRothFingerprinter(PadelpyFingerprinter):
    def __init__(self, *args, **kwargs):
        super().__init__(name="KlekotaRothFingerprinter", *args, **kwargs)


class KlekotaRothFingerprintCount(PadelpyFingerprinter):
    def __init__(self, *args, **kwargs):
        super().__init__(name="KlekotaRothFingerprintCount", *args, **kwargs)


class AtomPairs2DFingerprinter(PadelpyFingerprinter):
    def __init__(self, *args, **kwargs):
        super().__init__(name="AtomPairs2DFingerprinter", *args, **kwargs)
