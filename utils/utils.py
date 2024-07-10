from ultralytics import SAM
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
import cv2
import json
import os

from models.WebSAMAdapter import WebSAMAdapter

def draw_segmentations(image_path: str, json_path: str) -> None:
    """
    Plots and saves segmentations from JSON file on the image

    Args:
        image_path (str): image path
        json_path (str): JSON file path
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")

    # Read the JSON file
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Extract segmentation data
    segmentations = data.get('segmentations', {}).get('majority-vote', [])

    # Draw each segmentation on the image
    for segmentation in segmentations:
        for segment in segmentation:
            points = np.array(segment[0], dtype=np.int32)

            # Check if points are properly formatted
            if points.size == 0 or points.ndim != 2 or points.shape[1] != 2:
                raise ValueError("Segmentation points are not properly formatted or empty")

            cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    # Convert image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Display the image with segmentations
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
    save_path = os.path.join(os.getcwd(), "test.png")
    print(save_path)
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    print(f"Segmented image saved as {save_path}")

def load_pretrained(wsa: WebSAMAdapter, SAM_path: str) -> WebSAMAdapter:
    """
    Load SAM weights into corresponding WebSAM-Adapter layers

    Args:
        encoder: WebSAMEncoder
        SAM_path: path to pretrained SAM model

    Returns:
        WebSAMAdapter: model with SAM weights loaded
    """
    wsa_encoder = wsa.encoder
    wsa_decoder = wsa.decoder
    wsa_edict = wsa_encoder.state_dict()
    wsa_ddict = wsa_decoder.state_dict()

    sam_full = SAM(SAM_path).model
    sam_encoder = sam_full.image_encoder
    sam_decoder = sam_full.mask_decoder

    # load encoder then decoder weights
    for layer, weights in sam_encoder.state_dict().items():
        if layer in wsa_edict:
            wsa_edict[layer] = weights

    for layer, weights in sam_decoder.state_dict().items():
        if layer in wsa_ddict:
            wsa_ddict[layer] = weights

    wsa.encoder.load_state_dict(wsa_edict)
    wsa.decoder.load_state_dict(wsa_ddict)
    wsa = WebSAMAdapter(wsa.encoder, wsa.decoder)
    return wsa
