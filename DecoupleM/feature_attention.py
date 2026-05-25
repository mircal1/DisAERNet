from typing import List, Tuple, Any, Dict

import torch
from torch import nn


class FeatureAttention(nn.Module):

    def __init__(self,feature_dim, num_heads=2) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim,num_heads=num_heads,batch_first=True)

    def forward(self, mutual, salient):
        attoutput, _ = self.attn(query=mutual,key=salient,value=mutual)
        return attoutput+salient
