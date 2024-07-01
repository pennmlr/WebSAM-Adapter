import torch
import torch.nn as nn
from typing import Type

class PatchEmbeddingTune(nn.Module):
    def __init__(
            self, 
            embed_dim: int = 768, 
            scale_factor: float = 1.0
    ) -> None:
        """
        Module to refine patch embeddings from SAM's patch embedding layer.

        Args:
            embed_dim (int): Original embedding dimension (E).
            scale_factor (float): Scale factor (µ) to regulate the number of learnable parameters.
        """
        super().__init__()
        reduced_dim = int(embed_dim / scale_factor)
        self.linear_layer = nn.Linear(embed_dim, reduced_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_layer(x)
        return x