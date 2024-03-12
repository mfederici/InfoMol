from typing import Optional, Union

import numpy as np
import torch
from rdkit import Chem
from torch import nn
from transformers import AutoModelWithLMHead, AutoTokenizer

from encoder.base import Encoder
from encoder.deep import DeepEncoder


class ChemBERTaEncoder(DeepEncoder):
    DEFAULT_MODEL = 'seyonec/ChemBERTa_zinc250k_v2_40k'
    POOL = ['mean', 'sum', 'max']

    @staticmethod
    def get_name(pretrained_model_name_or_path: str, pool:str, **kwargs) -> str:
        return pretrained_model_name_or_path.split('/')[-1]+'_'+pool

    @staticmethod
    def get_n_dim(**kwargs) -> int:
        return 768

    def __init__(
            self,
            pool: str = 'mean',
            verbose: bool = False,
            pretrained_model_name_or_path: Optional[str] = None,
            device: Union[str, torch.device] = 'cpu'
         ):
        if pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = self.DEFAULT_MODEL
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

        super().__init__(
            pool=pool,
            verbose=verbose,
            device=device,
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
    def _instantiate_model(self) -> nn.Module:
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        return AutoModelWithLMHead.from_pretrained(
                self.pretrained_model_name_or_path,
                output_hidden_states=True
        )

    def _pool(self, representation_sequence: torch.Tensor) -> torch.Tensor:
        # Pool the sequence dimension (1)
        if self.pool == 'mean':
            representation = representation_sequence.mean(1)
        elif self.pool == 'max':
            representation = torch.max(representation_sequence, 1)[0]
        else:
            representation = representation_sequence.sum(1)
        return representation

    def _model_encode_smile(self, smile: str) -> torch.Tensor:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            raise ValueError(f"Invalid Molecule {smile}")

        with torch.no_grad():
            tokens = self.tokenizer(smile, return_tensors='pt')
            representation_sequence = self.model(**tokens).hidden_states[-1]
            representation = self._pool(representation_sequence)[0]

        return representation
