import sys

### Add WML client install path ###
sys.path.append("/opt/ibm/wml_py_client/v4")

from watson_machine_learning_client import WatsonMachineLearningAPIClient
import os
import shutil
import json
import ibm_boto3
from botocore.client import Config
import requests

### Function to generate icp token. This won't be needed when using wml python client###
def gen_icp_token(wml_credentials):
    token_url = wml_credentials['url'] + '/v1/preauth/validateAuth'
    response = requests.get(token_url, auth=(wml_credentials['username'], wml_credentials['password']), verify=False)
    if response.status_code != 200:
        print('Getting auth token failed.')
        raise Exception(response.text)
    else:
        token = response.json()['accessToken']
    return token

### Function to get asset details using REST API. This won't be needed once python client adds attachment details in asset meta ###
def get_asset_details(wml_credentials, asset_id, space_id):
    request_url = os.path.join(wml_credentials['url'], 'v2/assets', asset_id + "?space_id=" + space_id)
    token = gen_icp_token(wml_credentials)
    headers = {
        'Authorization': 'Bearer ' + token,
        'Content-Type': 'application/json'
    }
    response = requests.get(request_url, headers=headers, verify=False)
    if response.status_code != 200:
        print('Getting asset details failed.')
        raise Exception(response.text)
    return response.json()

### Function to upload data to cos ###
def upload_to_cos(properties, src_path, bucket_name, dest_path):
    service_endpoint = properties['url']
    aws_access_key_id = properties['access_key']
    aws_secret_access_key = properties['secret_key']
    cos_client = ibm_boto3.resource('s3',
                                    endpoint_url=service_endpoint,
                                    aws_access_key_id=aws_access_key_id,
                                    aws_secret_access_key=aws_secret_access_key)
    cos_client.meta.client.upload_file(src_path, bucket_name, dest_path)


### Initialize WML python client ###
wml_credentials = {
    "url": "**YOUR URL**",
    "username": "**YOUR USERNAME**",
    "password": "**YOUR PASSWORD**",
    "instance_id": "wml_local",
    "version": "3.0.0"
}

client = WatsonMachineLearningAPIClient(wml_credentials)

### Initialize directories ###
input_dir = os.environ['BATCH_INPUT_DIR']
work_dir = os.environ['WORKDIR']
jobs_payload_file = os.environ['JOBS_PAYLOAD_FILE']
cos_file = os.path.join(input_dir, 'cos_input.zip')


### Read scoring payload and get asset and space id ###
with open(jobs_payload_file) as f:
    scoring_payload = json.load(f)
asset_url = scoring_payload['scoring']['input_data_references'][0]['location']['href']
asset_parts = asset_url.split('/v2/assets/')[-1].split('?space_id=')
asset_id = asset_parts[0]
if len(asset_parts)>1:
    space_id = asset_parts[1]
else:
    space_id = '1ddf79bf-9886-4dcc-851b-18e5acaaa2b6'

### Set space and download ###
client.set.default_space(space_id)
client.data_assets.download(asset_id, filename=cos_file)

### Copy to work directory. This is just a dummy code. Idea is to substitute this with ### 
### user program logic and use the work directory as the place to put intermediate or final ###
### data that user will use ###
shutil.copy(cos_file, work_dir)

### Get dest asset connection details ###
dest_asset_href = scoring_payload['scoring']['output_data_reference']['location']['href']
dest_asset_parts = dest_asset_href.split('/v2/assets/')[-1].split('?space_id=')
dest_asset_id = dest_asset_parts[0]
if len(dest_asset_parts)>1:
    dest_space_id = dest_asset_parts[1]
else:
    dest_space_id = '1ddf79bf-9886-4dcc-851b-18e5acaaa2b6' # Use a default space id if the asset href doesn't have it
src_path = os.path.join(work_dir, cos_file)
dest_asset_details = get_asset_details(wml_credentials, dest_asset_id, dest_space_id)
conn_id = dest_asset_details['attachments'][0]['connection_id']
conn_details = client.connections.get_details(conn_id)
conn_properties = conn_details['entity']['properties']
conn_path = dest_asset_details['attachments'][0]['connection_path'].lstrip('/')

### Upload to COS ###
path_parts = conn_path.split('/')
bucket_name = path_parts[0]
dest_path = ''.join(path_parts[i] + '/' for i in range(1, len(path_parts))).rstrip('/')
upload_to_cos(conn_properties, src_path, bucket_name, dest_path)
