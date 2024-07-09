import boto3
import pandas as pd
import io
import os
from dotenv import load_dotenv
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import concurrent.futures
from sklearn.model_selection import train_test_split
import pdb
from torchvision import transforms
import numpy as np

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

    def __len__(self): 
        return len(self.batches)

    def __getitem__(self, idx): 
        batch_files = self.batches[idx]
        data_batch = []

        for file_key in batch_files:
            index = file_key.split('/')[-1]
            image_key = f"webis-webseg-20-screenshots/{file_key}/{'screenshot.png'}"
            print(f"Image key: {image_key}")
            image = self.data_loader.read_image(image_key)
            json_key = f"webis-webseg-20-ground-truth/{index}/ground-truth.json"
            ground_truth = self.data_loader.read_json(json_key).get('segmentations', {}).get('majority-vote', [])

            if self.transform:
                image = self.transform(image)

            image_tensor = transforms.ToTensor()(image)

            data_batch.append((image_tensor, ground_truth))

        return data_batch


if __name__ == "__main__":
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
    
    train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=42)

    batch_size = 32 
    train_dataset = S3BatchDataset(data_loader, train_indices, batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    for batch in train_dataloader:
        for image, ground_truth in batch:
            print("Image size:", image.size())
            print("Ground truth shape:", len(ground_truth))
