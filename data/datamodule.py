import io
import os
import pdb
import json
import torch
import boto3
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from dotenv import load_dotenv
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

load_dotenv()

class S3DataLoader:
    def __init__(self,
                 bucket_name,
                 aws_access_key_id=None,
                 aws_secret_access_key=None,
                 aws_session_token=None,
                 region_name=None):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region_name
        )

    def ls(self, prefix=''):
        response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix, Delimiter='/')
        directories = [content['Prefix'] for content in response.get('CommonPrefixes', [])]
        return directories

    def read_json(self, key):
        obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        return pd.read_json(io.BytesIO(obj['Body'].read())) 
    
    def read_csv(self, key):
        obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        return pd.read_csv(io.BytesIO(obj['Body'].read()))

    def read_text(self, key):
        obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        return obj['Body'].read().decode('utf-8')

    def read_image(self, key):
        obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
        return Image.open(io.BytesIO(obj['Body'].read()))


class S3BatchDataset(Dataset): 
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
            #added so input shape to EC tune layer is correct
            if image.mode != 'RGB':
                image = image.convert('RGB')

            json_key = f"webis-webseg-20-ground-truth/{index}/ground-truth.json"
            ground_truth = self.data_loader.read_text(json_key)
            ground_truth = json.loads(ground_truth).get('segmentations', {}).get('majority-vote', [])
            if self.transform:
                image = self.transform(image)
            
            image = image.squeeze(0)
            mask_tensor = self.segmentations_to_mask(ground_truth, img_size=(1024, 1024)).squeeze(0)

            data_batch.append((image, mask_tensor))

        return data_batch

def combine_image_and_mask(image, mask):
    mask = Image.fromarray((mask * 255).astype(np.uint8))
    mask = mask.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 0, 0, 0))
    overlay.paste(mask, (0, 0), mask)
    combined = Image.alpha_composite(image.convert("RGBA"), overlay)
    return combined.convert("RGB")

if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    bucket_name = "webis-webseg20"

    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    data_loader = S3DataLoader(
        bucket_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    contents = data_loader.ls(prefix='')
    print("Directories in bucket:", contents)

    file_path = 'indices.txt'

    with open(file_path, 'r') as f:
        lines = f.readlines()

    indices = [line[:-1] for line in lines][:-1]
    
    train_size = int(0.7 * len(indices))
    test_size = len(indices) - train_size
    train_indices, test_indices = torch.utils.data.random_split(indices, [train_size, test_size])

    train_indices = [indices[i] for i in train_indices.indices]
    test_indices = [indices[i] for i in test_indices.indices]

    batch_size = 1
    resize_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])

    train_dataset = S3BatchDataset(data_loader, train_indices, batch_size, transform=resize_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    for i, batch in enumerate(train_dataloader):
        print(f"Batch {i + 1}")
        for image, ground_truth in batch:
            # for img, gt in zip(image, ground_truth):
            #     img = img.permute(1, 2, 0).mul(255).byte().numpy()
            #     img = Image.fromarray(img)
            #     mask = gt.numpy() 
            #     combined_image = combine_image_and_mask(img, mask)
            #     combined_image.save(f"combined_image_batch_{i+1}.png")
            #     print("Saved combined image for batch", i + 1)

            print("Image Shape: ", image.shape)
            print("Ground Truth Mask Shape: ", ground_truth.shape)
