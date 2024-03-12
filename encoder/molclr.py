import os
from typing import Union

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
import networkx as nx
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data

import MolCLR
from encoder import Encoder
from encoder.deep import DeepGraphEncoder

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE,
    BT.DOUBLE,
    BT.TRIPLE,
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]



class MolCLREncoder(DeepGraphEncoder):
    ARCHITECTURES = {
        'GIN': 'ckpt/pretrained_gin/checkpoints/model.pth',
        'GCN': 'ckpt/pretrained_gcn/checkpoints/model.pth',
    }
    DEFAULT_ARCHITECTURE='GIN'
    POOL = [
        'mean',
        'sum',
        'max'
    ]

    @staticmethod
    def get_name(architecture: str = DEFAULT_ARCHITECTURE, pool: str = DeepGraphEncoder.DEFAULT_POOL, **kwargs) -> str:
        return  f'MolCLR_{architecture}_{pool}'

    @staticmethod
    def get_n_dim(**kwargs) -> int:
        return 256

    def __init__(
            self,
            architecture: str = DEFAULT_ARCHITECTURE,
            verbose: bool = False,
            pool: str= DeepGraphEncoder.DEFAULT_POOL,
            device: Union[torch.device, str] = 'cpu'
    ):

        if not (architecture in self.ARCHITECTURES):
            raise ValueError(f'The available architectures are {self.ARCHITECTURES.keys()} (default: "GIN")')

        self.architecture = architecture

        super().__init__(pool=pool, verbose=verbose, device=device, architecture=architecture)

        if self.verbose:
            print("Model loaded")


    def _instantiate_model(self) -> nn.Module:
        if self.architecture == 'GIN':
            from MolCLR.models.ginet_molclr import GINet

            model = GINet(
                num_layer=5,  # number of graph conv layers
                emb_dim=300,  # embedding dimension in graph conv layers
                feat_dim=512,  # output feature dimention
                drop_ratio=0,  # dropout ratio
                pool='add' if self.pool =='sum' else self.pool
            )

        else:
            from MolCLR.models.gcn_molclr import GCN

            model = GCN(
                num_layer=5,  # number of graph conv layers
                emb_dim=300,  # embedding dimension in graph conv layers
                feat_dim=512,  # output feature dimention
                drop_ratio=0,  # dropout ratio
                pool='add' if self.pool =='sum' else self.pool
            )

        if self.verbose:
            print('Loading the Model')
        model.load_state_dict(
            torch.load(
                os.path.join(MolCLR.__path__[0], self.ARCHITECTURES[self.architecture]), map_location='cpu')
        )

        return model

    def _mol2graph(self, mol: Chem.Mol) -> Data:
        # Code based on https://github.com/yuyangw/MolCLR/blob/master/dataset/dataset.py
        type_idx = []
        chirality_idx = []
        atomic_number = []
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()

        # Construct the original molecular graph from edges (bonds)
        edges = []
        for bond in bonds:
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        molGraph = nx.Graph(edges)

        for atom in atoms:
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
        x = torch.cat([x1, x2], dim=-1)
        # x shape (N, 2) [type, chirality]

        row_i, col_i, row_j, col_j = [], [], [], []
        edge_feat_i, edge_feat_j = [], []
        G_i_edges = list(molGraph.edges)
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            feature = [
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ]
            if (start, end) in G_i_edges:
                row_i += [start, end]
                col_i += [end, start]
                edge_feat_i.append(feature)
                edge_feat_i.append(feature)

        edge_index = torch.tensor([row_i, col_i], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat_i), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return data

    def _model_encode_graph(self, graph: Data) -> np.ndarray:
        _, z = self.model(graph)
        representation = z[0]
        return representation