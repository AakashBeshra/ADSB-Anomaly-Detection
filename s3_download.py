# s3_download.py
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os

# Setup S3 client (no credentials needed for public bucket)
s3 = boto3.client('s3', 
                  config=Config(signature_version=UNSIGNED),
                  region_name='us-east-1')

bucket_name = 'opensky-network'
prefix = 'data-samples/states/'

# List files in the states folder
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

# Get list of available files
files = []
for obj in response.get('Contents', []):
    if obj['Key'].endswith('.json.gz'):
        files.append(obj['Key'])

print(f"Found {len(files)} state vector files")

# Download first 30 files
for i, file_key in enumerate(files[:30], 1):
    try:
        filename = file_key.split('/')[-1]
        dataset_name = f"dataset_{i:02d}.json.gz"
        local_path = os.path.join("data/opensky", dataset_name)
        
        print(f"[{i:02d}/30] Downloading {filename}...")
        
        s3.download_file(bucket_name, file_key, local_path)
        
        print(f"  ✓ Saved as {dataset_name}")
        
    except Exception as e:
        print(f"  ✗ Error downloading {file_key}: {str(e)}")