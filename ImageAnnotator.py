import os
import torch
import json
import pdb
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from backbone.transformer import TwoWayTransformer as transformer
from models.WebSAMAdapter import WebSAMEncoder, WebSAMDecoder, WebSAMAdapter
from utils.utils import load_pretrained

# Function to draw bounding boxes on the image
def draw_bounding_boxes(image, masks):
    draw = ImageDraw.Draw(image)
    for mask in masks:
        mask_np = np.array(mask)
        y_indices, x_indices = np.where(mask_np > 0.5)
        if len(y_indices) > 0 and len(x_indices) > 0:
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
    return image

# Load image
image_path = 'test_image.png'
image = Image.open(image_path).convert('RGB')

# Preprocess the image
resize_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])
image_tensor = resize_transform(image).unsqueeze(0)

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
encoder = WebSAMEncoder()
twt = transformer(depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048)
decoder = WebSAMDecoder(transformer_dim=256, transformer=twt)
model = WebSAMAdapter(encoder, decoder)

# Load model weights
checkpoint_path = '/shared_data/mlr_club/saved_models/model_epoch_12.pt'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Move image tensor to the GPU
image_tensor = image_tensor.to(device)
image_shape_tensor = torch.tensor(image.size).unsqueeze(0).to(device)
# Perform forward pass
# pdb.set_trace()

with torch.no_grad():
    output_masks = model(image_tensor, image_shape_tensor)
    output_masks = torch.sigmoid(output_masks[0])

# Process output masks
output_masks = output_masks.squeeze(0).cpu().numpy()
# Convert masks to binary format and draw bounding boxes
output_masks_binary = output_masks.round()


# to_pil = transforms.ToPILImage()
# resized_image = to_pil(image_tensor.squeeze(0))
annotated_image = draw_bounding_boxes(image, output_masks_binary)

# Save the annotated image
annotated_image_path = 'annotated_image.png'
annotated_image.save(annotated_image_path)
print(f"Annotated image saved at {annotated_image_path}")
