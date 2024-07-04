import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelExtraction(nn.Module):
    """
    Class to perform Sobel edge detection on an image.

    Args:
        image (torch.Tensor): batched input images of shape N x C x H x W
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
