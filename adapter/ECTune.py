import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.SAMEncoder import PatchEmbed

class SobelExtraction(nn.Module):
    """
    Class to perform Sobel edge detection on an image.

    Args:
        mean (list): channel-wise dataset mean
        std (list): channel-wise dataset standard deviation
    """
    def __init__(
            self, 
            mean: list = [.485, .456, .406], 
            std: list = [.229, .224, .225] # TODO: figure out WEBIS statistics
        ) -> None:
        super().__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.std + self.mean
        gray = x.mean(dim=1, keepdim=True) # N x C x H x W -> N x 1 x H x W

        # Sobel kernels
        sk_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sk_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sk_x = sk_x.to(x.device)
        sk_y = sk_y.to(x.device)

        sobel_x = F.conv2d(gray, sk_x, padding=1)
        sobel_y = F.conv2d(gray, sk_y, padding=1)
        sobel = torch.sqrt(sobel_x ** 2 + sobel_y ** 2)

        return sobel

class ECTune(nn.Module):
    """
    Linear layer on top of Sobel-filtered image patches.
    """

    def __init__(self, embed_dim: int = 768, patch_size: int = 16) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_and_stride = (patch_size, patch_size)
        self.sobel = SobelExtraction()
        self.PatchEmbed = PatchEmbed(
            kernel_size = self.kernel_and_stride,
            stride = self.kernel_and_stride, 
            embed_dim = self.embed_dim,
            in_chans = 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x: N x C x H x W
        sobel = self.sobel(x)
        patches = self.PatchEmbed(sobel)
        return patches # N x (H / patch_size) x (W / patch_size) x embed_dim
