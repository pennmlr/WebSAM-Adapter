import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ../utils/utils.py import SobelExtraction
    

class ECTune(nn.Module):
    """
    Module to project Sobel image patches into a feature space using a linear layer.

    Args:
        feature_dim (int): The dimension of the feature space.
        patch_size (int): The size of the non-overlapping patches.
    """

    def __init__(self, feature_dim, patch_size):
        super(ECTune, self).__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.linear_layer = nn.Linear(patch_size * patch_size, feature_dim)

    def extract_features(self, sobel_image):
        patches = self.split_into_patches(sobel_image, self.patch_size)
        projected_patches = self.project_to_feature_space(patches)
        return projected_patches

    def split_into_patches(self, image, patch_size):
        image_tensor = torch.tensor(image, dtype=torch.float32)
        patches = image_tensor.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
        patches = patches.contiguous().view(-1, patch_size * patch_size)
        return patches

    def project_to_feature_space(self, patches):
        projected_patches = self.linear_layer(patches)
        return projected_patches

