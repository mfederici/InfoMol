import os
import sys
from typing import Optional

import torch
from torch import nn
from rdkit import Chem

from encoder.deep import DeepEncoder

module_path = os.path.abspath(os.path.join('/'.join(__file__.split('/')[:-1]), '..', 'KPGT'))
sys.path.insert(0, module_path)
from src.model.light import LiGhTPredictor as LiGhT
from src.model_config import config_dict
from src.data.finetune_dataset import MoleculeDataset
from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
from src.data.featurizer import smiles_to_graph_tune


# Code from https://github.com/lihan97/KPGT
class KPGTEncoder(DeepEncoder):
    DEFAULT_VERSION = 'base'
    @staticmethod
    def get_name(version: str = DEFAULT_VERSION, pool: Optional[str] = None, **kwargs) -> str:
        assert pool is None
        return 'KPGT_'+version

    @staticmethod
    def get_n_dim(**kwargs) -> int:
        return 2304
    def __init__(self, version: str = DEFAULT_VERSION, **kwargs):
        model_path = os.path.join(module_path, 'model', f'{version}.pth')
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"The checkpoint {model_path} has not been found. Please download it as described in https://github.com/lihan97/KPGT. ")
        self.model_path = model_path
        self.config = config_dict[version]
        self.vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)

        super().__init__(pool=None, version=version, **kwargs)

    def _instantiate_model(self) -> nn.Module:
        # Model Initialization
        if self.verbose:
            print("Instantiating the LiGhT transformer.")
        model = LiGhT(
            d_node_feats=self.config['d_node_feats'],
            d_edge_feats=self.config['d_edge_feats'],
            d_g_feats=self.config['d_g_feats'],
            d_hpath_ratio=self.config['d_hpath_ratio'],
            n_mol_layers=self.config['n_mol_layers'],
            path_length=self.config['path_length'],
            n_heads=self.config['n_heads'],
            n_ffn_dense_layers=self.config['n_ffn_dense_layers'],
            input_drop=0,
            attn_drop=0,
            feat_drop=0,
            n_node_types=self.vocab.vocab_size
        )

        if self.verbose:
            print(f"Loading the weights from {self.model_path}.")
        # Loading weights
        model.load_state_dict({
            k.replace('module.', ''): v for k, v
            in torch.load(self.model_path, map_location=torch.device('cpu')).items()
        })


        model.eval()
        return model

    def _model_encode_smile(self, smile: str) -> torch.Tensor:
        generator = RDKit2DNormalized()
        md = torch.Tensor(generator.process(smile)[1:]).reshape(1, -1)

        mol = Chem.MolFromSmiles(smile)
        fp = torch.Tensor(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)).reshape(1, -1)

        graph = smiles_to_graph_tune(
            smile,
            max_length=self.config['path_length'],
            n_virtual_nodes=2
        )

        rep = self.model.generate_fps(graph, fp, md)
        return rep[0]