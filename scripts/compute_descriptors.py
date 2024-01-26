import pickle

import numpy as np
from torch_geometric.datasets import MoleculeNet
from tqdm.auto import tqdm

from fingerprint.padelpy import *
from fingerprint.cached import CachedFingerprinter


TASKS = ['SIDER', 'Tox21', 'ClinTox', 'ToxCast']
ROOT = '/data'
CACHE_PATH = '/data/fingerprints'

if __name__ == '__main__':
    fp_extractors = [
        CachedFingerprinter(
            fingerprinter=fingerprinter,
            cache_path=CACHE_PATH
        ) for fingerprinter in [
            AtomPairs2DFingerprinter(),
            KlekotaRothFingerprintCount(),
            MACCSFingerprinter(),
            PubchemFingerprinter(),
            SubstructureFingerprintCount()
        ]
    ]
    for task in TASKS:
        print(task)
        dataset = MoleculeNet(ROOT, name=task)
        for fp_extractor in fp_extractors:
            print(fp_extractor)
            try:
                fp_extractor(dataset.smiles)
            except Exception as e:
                print(e)

    print("DONE")
