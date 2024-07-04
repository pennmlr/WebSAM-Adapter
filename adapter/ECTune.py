import torch
import torch.nn as nn
from utils.utils import SobelExtraction
from backbone.SAMEncoder import PatchEmbed

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
