import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from dotenv import load_dotenv
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

load_dotenv()

class LocalDataLoader:
    def __init__(self, base_path):
        self.base_path = base_path

    def ls(self, sub_dir=''):
        full_path = os.path.join(self.base_path, sub_dir)
        directories = [d for d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d))]
        return directories

    def read_json(self, path):
        full_path = os.path.join(self.base_path, path)
        with open(full_path, 'r') as f:
            return pd.read_json(f)

    def read_csv(self, path):
        full_path = os.path.join(self.base_path, path)
        return pd.read_csv(full_path)

    def read_text(self, path):
        full_path = os.path.join(self.base_path, path)
        with open(full_path, 'r') as f:
            return f.read()

    def read_image(self, path):
        full_path = os.path.join(self.base_path, path)
        return Image.open(full_path)


class LocalBatchDataset(Dataset): 
    def __init__(self, data_loader, files, batch_size, transform=None): 
        self.data_loader = data_loader 
        self.files = files 
        self.batch_size = batch_size 
        self.transform = transform 
        self.batches = self.create_batches() 

    def create_batches(self): 
        batches = [self.files[i:i + self.batch_size] for i in range(0, len(self.files), self.batch_size)] 
        return batches 

    def segmentations_to_mask(self, segmentations, img_size=(1024, 1024)):
        mask = Image.new('L', img_size, 0)
        draw = ImageDraw.Draw(mask)
        for polygon in segmentations:
            for segment in polygon:
                segment = [(int(x), int(y)) for x, y in segment[0]]  # Ensure coordinates are integers
                draw.polygon(segment, outline=1, fill=1)
        return torch.from_numpy(np.array(mask)).float()

    def __len__(self): 
        return len(self.batches)

    def __getitem__(self, idx): 
        batch_files = self.batches[idx]
        data_batch = []

        for file_key in batch_files:
            index = file_key.split('/')[-1]
            image_key = f"webis-webseg-20-screenshots/{file_key}/screenshot.png"
            image = self.data_loader.read_image(image_key)
            # Ensure input shape to EC tune layer is correct
            if image.mode != 'RGB':
                image = image.convert('RGB')
            original_image_shape = image.size
            json_key = f"webis-webseg-20-ground-truth/{index}/ground-truth.json"
            ground_truth = self.data_loader.read_text(json_key)
            ground_truth = json.loads(ground_truth).get('segmentations', {}).get('majority-vote', [])
            if self.transform:
                image = self.transform(image)
            
            image = image.squeeze(0)
            mask_tensor = self.segmentations_to_mask(ground_truth, img_size=(1024, 1024)).squeeze(0)

            data_batch.append(((image, torch.tensor(original_image_shape)), mask_tensor))

        return data_batch

def combine_image_and_mask(image, mask):
    mask = Image.fromarray((mask * 255).astype(np.uint8))
    mask = mask.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 0, 0, 0))
    overlay.paste(mask, (0, 0), mask)
    combined = Image.alpha_composite(image.convert("RGBA"), overlay)
    return combined.convert("RGB")

if __name__ == "__main__":
    base_path = "/shared_data/mlr_club/3988124"
    
    data_loader = LocalDataLoader(base_path)

    contents = data_loader.ls('')
    print("Directories in base path:", contents)

    file_path = 'indices.txt'

    with open(file_path, 'r') as f:
        lines = f.readlines()

    indices = [line.strip() for line in lines]

    train_size = int(0.7 * len(indices))
    test_size = len(indices) - train_size
    train_indices, test_indices = torch.utils.data.random_split(indices, [train_size, test_size])

    train_indices = [indices[i] for i in train_indices.indices]
    test_indices = [indices[i] for i in test_indices.indices]

    batch_size = 16
    resize_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])

    train_dataset = LocalBatchDataset(data_loader, train_indices, batch_size, transform=resize_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    for i, batch in enumerate(train_dataloader):
        print(f"Batch {i + 1}")
        for image_tuple, ground_truth in batch:
            print("Resized Image Shape: ", image_tuple[0].shape)
            print("Original Image Shape: ", image_tuple[1])
            print("Ground Truth Mask Shape: ", ground_truth.shape)
            images, ground_truths = zip(*batch)
