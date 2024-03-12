
import torch
from rdkit import Chem
from torch import nn
from torch_geometric.data import Data

from encoder.deep import DeepGraphEncoder

# TODO: fix junky import for submodule
import os.path
import sys
module_path = os.path.abspath(os.path.join('/'.join(__file__.split('/')[:-1]), '..', 'Mole-BERT'))
sys.path.insert(0, module_path)
from model import GNN_graphpred
from loader import mol_to_graph_data_obj_simple


class MoleBERTEncoder(DeepGraphEncoder):
    POOL = [
        'mean',
        'sum',
        'max',
        'attention',
    ]

    @staticmethod
    def get_name(pool: str = DeepGraphEncoder.DEFAULT_POOL, **kwargs) -> str:
        return f'MoleBERT_{pool}'

    @staticmethod
    def get_n_dim(**kwargs) -> int:
        return 300

    def _instantiate_model(self) -> nn.Module:
        model = GNN_graphpred(
            num_layer=5,
            emb_dim=300,
            num_tasks=1,
            graph_pooling = self.pool,
            gnn_type='gin'
        )
        model.from_pretrained(os.path.join(module_path,'model_gin/Mole-BERT.pth'))

        return model

    def _mol2graph(self, mol: Chem.Mol) -> Data:
        return mol_to_graph_data_obj_simple(mol)

    def _model_encode_graph(self, graph: Data) -> torch.Tensor:
        _, representation_sequence = self.model(graph)
        representation = self.model.pool(representation_sequence, graph.batch)[0]
        return representation