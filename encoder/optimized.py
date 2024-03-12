from typing import Optional

import numpy as np

from encoder.padelpy import Encoder, AtomPairs2DFingerprintCount, SubstructureFingerprintCount, \
    KlekotaRothFingerprintCount, MACCSFingerprinter, PubchemFingerprinter
from encoder.cached import CachedEncoder
from encoder.rdkit import EStateFingerprinter


class OptimizedFishToxFingerprinter(Encoder):

    @staticmethod
    def get_name(**kwargs) -> str:
        assert len(kwargs) == 0
        return 'OptimizedFishToxFingerprinter'

    @staticmethod
    def get_n_dim(**kwargs) -> int:
        return 340

    def __init__(
            self,
            cache_components: bool = False,
            cache_path: Optional[str] = None,
            read_only_cache: bool = False,
            verbose: bool = False,
    ):
        self.components = [
            {
                'encoder': AtomPairs2DFingerprintCount(verbose=verbose),
                'selection': [0, 1, 2, 6, 11, 13, 78, 79, 80, 84, 89, 101, 156, 157, 158, 162, 167, 234, 236, 240, 245,
                              312, 314, 318, 323, 390, 468, 546]
            },
            {
                'encoder': KlekotaRothFingerprintCount(verbose=verbose),
                'selection': [0, 17, 19, 137, 188, 296, 297, 340, 343, 361, 381, 397, 465, 492, 493, 503, 603, 646, 647,
                              668, 1145, 1147, 1148, 1149, 1153, 1192, 1262, 1405, 1563, 1565, 1591, 1641, 1644, 1768,
                              1931, 2258, 2379, 2546, 2594, 2666, 2672, 2693, 2854, 2948, 2949, 2974, 2978, 2985, 3009,
                              3024, 3138, 3223, 3267, 3368, 3394, 3407, 3439, 3454, 3466, 3467, 3549, 3586, 3590, 3605,
                              3639, 3658, 3670, 3681, 3691, 3736, 3739, 3743, 3748, 3749, 3772, 3787, 3820, 3868, 3880,
                              3881, 3925, 3942, 3955, 3999, 4018, 4079, 4116, 4236, 4330, 4478, 4708, 4828, 4842]
            },
            {
                'encoder': MACCSFingerprinter(verbose=verbose),
                'selection': [65, 66, 71, 73, 75, 80, 81, 83, 85, 86, 87, 89, 92, 95, 97, 98, 102, 103, 104, 105, 106,
                              107, 108, 111, 112, 113, 114, 115, 122, 124, 125, 127, 128, 130, 131, 133, 135, 136, 138,
                              139, 140, 141, 143, 144, 145, 148, 149, 150, 151, 152, 153, 154, 155, 156, 158, 159, 160,
                              161, 162, 163, 164]
            },
            {
                'encoder': PubchemFingerprinter(verbose=verbose),
                'selection': [0, 1, 2, 3, 10, 11, 12, 14, 15, 18, 19, 20, 30, 33, 37, 38, 39, 43, 44, 115, 116, 143,
                              178, 179, 181, 185, 186, 256, 257, 259, 283, 285, 286, 293, 294, 297, 299, 301, 308, 314,
                              327, 329, 332, 333, 335, 337, 339, 340, 341, 342, 344, 345, 346, 351, 352, 353, 360, 365,
                              366, 368, 373, 374, 380, 391, 393, 395, 405, 406, 407, 411, 413, 416, 417, 418, 419, 420,
                              430, 432, 434, 436, 439, 440, 441, 442, 443, 448, 451, 452, 479, 501, 514, 516, 529, 535,
                              539, 540, 550, 567, 568, 571, 579, 582, 591, 592, 598, 599, 607, 614, 617, 637, 643, 656,
                              663, 669, 672, 678, 679, 681, 696, 697, 699, 708, 709, 712, 716]
            },
            {
                'encoder': SubstructureFingerprintCount(verbose=verbose),
                'selection': [0, 1, 2, 4, 7, 11, 67, 87, 95, 170, 223, 273, 274, 286, 294, 299, 300, 301, 306]
            },
            {
                'encoder': EStateFingerprinter(verbose=verbose),
                'selection': [6, 8, 10, 11, 12, 15, 16, 18, 20, 33, 34, 35, 49, 53]
            },
        ]
        self.columns = []
        for component in self.components:
            self.columns += [component['encoder'].columns[idx] for idx in component['selection']]

        super().__init__(verbose=verbose)

        if cache_components:
            if cache_path is None:
                raise ValueError(
                    "Please spacify a cache_path when cache_components is True."
                )
            for i in range(len(self.components)):
                self.components[i]['encoder'] = CachedEncoder(
                    cache_path=cache_path,
                    encoder=self.components[i]['encoder'],
                    read_only=read_only_cache,
                )

    def encode(self, smile: str) -> np.ndarray:
        return np.concatenate(
            [component['encoder'](smile)[component['selection']] for component in self.components], -1
        )

    def encode_all(self, smiles: str) -> np.ndarray:
        return np.concatenate(
            [component['encoder'](smiles)[:,component['selection']] for component in self.components], -1
        )


