from torch_geometric.datasets import MoleculeNet

from datasets import FishTox, ZINCSmiles, AQSOLSmiles
from encoder import *
from encoder.cached import CachedEncoder

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
    fingerprinters = [
        # ChemBERTaEncoder(verbose=True),
        # MoleBERTEncoder(verbose=True),
        # MolCLREncoder(verbose=True),
        # KPGTEncoder(verbose=True),
        # PubchemFingerprinter(verbose=True),
        # AtomPairs2DFingerprintCount(verbose=True),
        # KlekotaRothFingerprintCount(verbose=True),
        # MACCSFingerprinter(verbose=True),
        # SubstructureFingerprintCount(verbose=True),
        # EStateFingerprinter(verbose=True),
        OptimizedFishToxFingerprinter(verbose=True, cache_path=CACHE_PATH, cache_components=True)
    ]

    encoders = [
        CachedEncoder(
            encoder=fingerprinter,
            cache_path=CACHE_PATH,
            write_size=1000,
            read_only=False
        ) for fingerprinter in fingerprinters
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
