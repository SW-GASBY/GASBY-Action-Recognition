import boto3
import os
from dotenv import dotenv_values

env = dotenv_values('.env')
AWS_ACCESS_KEY = env['AWS_Accesskey']
AWS_SECRET_KEY = env['AWS_Secretkey']
AWS_REGION = env['AWS_Region']
s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)

def download_file(bucket_name, bucket_folder, folder, file):
    if not os.path.exists(folder):
        os.mkdir(folder, exist_ok=True)
    res = s3.download_file(bucket_name, bucket_folder + '/' + file, folder + '/' + file)
    return res

def upload_file(bucket_name, bucket_folder, file):
    res = s3.upload_file(file, bucket_name, bucket_folder +'/'+ file)
    return res