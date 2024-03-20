from torch_geometric.datasets import MoleculeNet

from datasets import FishTox, ZINCSmiles, AQSOLSmiles
from encoder import *

# Selection:
# - FishTox
# - SIDER
# - ESOL
# - Lipophilicity
# - FreeSolv
# - BACE

TASKS = ['sider', 'bbbp', 'esol', 'lipo', 'bace', 'freesolv']

ROOT = '/data'
CACHE_PATH = '/data/representations'

if __name__ == '__main__':
    encoders = [
        # ChemBERTaEncoder(verbose=True),
        # MoleBERTEncoder(verbose=True),
        # MolCLREncoder(verbose=True),
        # KPGTEncoder(verbose=True),
        # PubchemFingerprinter(verbose=True),
        AtomPairs2DFingerprintCount(verbose=True, cache_path=CACHE_PATH, read_only=False),
        # KlekotaRothFingerprintCount(verbose=True),
        # MACCSFingerprinter(verbose=True),
        # SubstructureFingerprintCount(verbose=True),
        # EStateFingerprinter(verbose=True),
        # OptimizedFishToxFingerprinter(verbose=True, cache_path=CACHE_PATH, read_only=False)
        NumAtoms(verbose=True, cache_path=CACHE_PATH, read_only=False),
        NumHeavyAtoms(verbose=True, cache_path=CACHE_PATH, read_only=False),
        NumRings(verbose=True, cache_path=CACHE_PATH, read_only=False),
        NumAromaticRings(verbose=True, cache_path=CACHE_PATH, read_only=False),
        MolecularWeight(verbose=True, cache_path=CACHE_PATH, read_only=False),
        MolecularFormula(verbose=True, cache_path=CACHE_PATH, read_only=False)
    ]

    datasets = [
        MoleculeNet(ROOT, name=task) for task in TASKS
    ]
    datasets += [AQSOLSmiles('/data')]
    datasets += [ZINCSmiles('/data')]
    datasets += [FishTox('/data')]


    for encoder in encoders:
        print('##########################\n\n\n')
        print(encoder)
        for dataset in datasets:
            print(dataset)
            encoder(dataset.smiles)
            encoder.write_cache_to_disk()
        del encoder

    print("DONE")
