from typing import List, Tuple, Any, Dict

import torch
from torch import nn


class LabelPretrainer(nn.Module):
    """
    Args:
        input_dim (int): dimension of encoded feature.
        hidden_dims (Union[List[int], int]): Hidden layer dimensions.
        output_dim (int): Number of classes.
        n_hiddens (Optional[int]): Number of hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        batch_norm = nn.BatchNorm1d

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            batch_norm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(
        self, encoded_x1: torch.Tensor, encoded_x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_x1 = self.model(encoded_x1)
        pred_x2 = self.model(encoded_x2)

        return pred_x1, pred_x2
