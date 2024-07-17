import torch
import torch.nn as nn
from typing import Type

class SharedUpLayer(nn.Module):
    def __init__(
            self, 
            input_size: int = 3072, 
            mlp_ratio: float = 4.0,
            output_size: int = 768,
            num_layers: int = 3,
            act_layer: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Single MLP shared across all adapter modules

        Args:
            input_size (int): output dimension of the module-specific MLPs
            mlp_ratio (float): ratio of hidden layer size to input layer size
            num_layers (int): number of hidden layers
            act_layer (nn.Module): activation function
        """
        super().__init__()
        hidden_size = int(input_size * mlp_ratio)
        self.layers = nn.ModuleList([])
        self.input_size = input_size

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
        for layer in self.layers:
            x = layer(x)
        return x

class UnsharedLayers(nn.Module):
    def __init__(
            self,
            num_adapters: int = 12,
            input_size: int = 768,
            mlp_ratio: float = 4.0,
            output_size: int = 3072,
            num_layers: int = 3,
            act_layer: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        MLP with weights unique to each adapter module

        Args:
            num_adapters: number of adapter modules (= number of SAM encoder blocks)
            input_size (int): flattened size of EC + PE tune embeddings
            mlp_ratio (float): ratio of hidden layer size to input layer size
            output_size (int): output dimension (= input dimension for shared layer)
            num_layers (int): number of hidden layers
            act_layer (nn.Module): activation function
        """
        super().__init__()
        hidden_size = int(input_size * mlp_ratio)
        self.layers = nn.ModuleList()
        for _ in range(num_adapters):
            layers = nn.ModuleList([])
            layers.append(nn.Sequential(
                nn.Linear(input_size, hidden_size),
                act_layer()
            ))

            for _ in range(num_layers - 1):
                layers.append(nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    act_layer()
                ))
            
            layers.append(nn.Sequential(
                nn.Linear(hidden_size, output_size)
            ))
            self.layers.append(layers)

    def forward(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        for l in self.layers[layer]:
            x = l(x)
        return x

class AdapterModule(nn.Module):
    def __init__(
            self, 
            shared: SharedUpLayer,
            num_adapters: int = 12,
            input_size: int = 768,
            mlp_ratio: float = 4.0,
            num_layers: int = 3,
            act_layer: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Single class containing all num_adapters adapter modules

        Args:
            shared (SharedUpLayer): MLP shared across all adapter modules
            num_adapters: number of adapter modules (= number of SAM encoder blocks)
            input_size (int): flattened size of EC + PE tune embeddings
            mlp_ratio (float): ratio of hidden layer size to input layer size
            num_layers (int): number of hidden layers
            act_layer (nn.Module): activation function
        """
        super().__init__()
        self.unshared = UnsharedLayers(num_adapters, input_size, mlp_ratio, shared.input_size, num_layers, act_layer)
        self.GELU = nn.GELU()
        self.shared = shared

    def forward(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        x = x.to(next(self.unshared.layers[layer][0].parameters()).device)  # Move to the same device as the unshared layer
        x = self.unshared(x, layer)
        x = self.GELU(x)
        x = x.to(next(self.shared.parameters()).device)  # Move to the same device as the shared layer
        x = self.shared(x)
        return x
