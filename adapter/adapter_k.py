# %%
import torch
import torch.nn as nn
from typing import Type

# %%
class SharedUpLayer(nn.Module):
    def __init__(
            self, 
            embed_dim: int = 768, 
            mlp_ratio: float = 4.0,
            output_size: int = 3072, # TODO: figure out right output size for SAM blocks
            num_layers: int = 3,
            act_layer: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Single MLP shared across all adapter modules

        Args:
            embed_dim (int): output dimension of the module-specific MLPs
            mlp_ratio (float): ratio of hidden layer size to input layer size
            num_layers (int): number of hidden layers
            act_layer (nn.Module): activation function
        """
        super().__init__()
        hidden_size = int(embed_dim * mlp_ratio)
        self.layers = nn.ModuleList([])
        self.embed_dim = embed_dim

        self.layers.append(nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            act_layer()
        ))

        for _ in range(num_layers - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                act_layer()
            ))
        
        self.layers.append(nn.Sequential(
            nn.Linear(hidden_size, output_size)
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class UnsharedLayer(nn.Module):
    def __init__(
            self, 
            input_size: int = 600, # TODO: figure out input size from EC + PE tunes 
            mlp_ratio: float = 4.0,
            output_size: int = 768,
            num_layers: int = 3,
            act_layer: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        MLP with weights unique to each adapter module

        Args:
            input_size (int): flattened size of EC + PE tune embeddings
            mlp_ratio (float): ratio of hidden layer size to input layer size
            output_size (int): output dimension (also input dimension for shared layer)
            num_layers (int): number of hidden layers
            act_layer (nn.Module): activation function
        """
        super().__init__()
        hidden_size = int(input_size * mlp_ratio)
        self.layers = nn.ModuleList([])

        self.layers.append(nn.Sequential(
            nn.Linear(input_size, hidden_size),
            act_layer()
        ))

        for _ in range(num_layers - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                act_layer()
            ))

        self.layers.append(nn.Sequential(
            nn.Linear(hidden_size, output_size)
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim = 1) # B H W C -> B (H W C)
        for layer in self.layers:
            x = layer(x)
        return x
    
# %%
class AdapterK(nn.Module):
    def __init__(
            self, 
            shared: Type[SharedUpLayer],
            input_size: int = 768, # TODO: figure out input size from EC + PE tunes 
            mlp_ratio: float = 4.0,
            num_layers: int = 3,
            act_layer: Type[nn.Module] = nn.GELU,
            
    ) -> None:
        """
        Individual Adapter Module

        Args:
            shared (SharedUpLayer): MLP shared across all adapter modules
            input_size (int): flattened size of EC + PE tune embeddings
            mlp_ratio (float): ratio of hidden layer size to input layer size
            num_layers (int): number of hidden layers
            act_layer (nn.Module): activation function
        """
        super().__init__()
        self.MLPUnshared = UnsharedLayer(input_size, mlp_ratio, shared.embed_dim, num_layers, act_layer)
        self.shared = shared
        self.layers = nn.ModuleList([
            self.MLPUnshared,
            nn.GELU(),
            self.shared
        ])

        # TODO: add unflatten layer to pass into SAM block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim = 1) # B H W C -> B (H W C)
        for layer in self.layers:
            x = layer(x)
        return x
