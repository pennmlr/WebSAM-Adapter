import boto3
import pandas as pd
import io
import os
from dotenv import load_dotenv
from PIL import Image


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
        folders = [content['Prefix'] for content in response.get('CommonPrefixes', [])]
        return folders

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



if __name__ == "__main__":
    bucket_name = "webis-webseg20"

    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    data_loader = S3DataLoader(
        bucket_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    files = data_loader.ls()
    print("Files in bucket:", files)