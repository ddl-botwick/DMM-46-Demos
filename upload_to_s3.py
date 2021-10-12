# upload to S3
 
import boto3
from botocore.exceptions import NoCredentialsError
import os
import numpy as np
from datetime import date

#passing env var to python variables
domino_user = os.getenv('DOMINO_STARTING_USERNAME')
domino_project = os.getenv('DOMINO_PROJECT_NAME')

 
def upload(local_file, bucket):
    s3 = boto3.client('s3', aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                      aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
    
    s3_file_name = '{}'.format(os.path.basename(local_file))
    
    try:
        s3.upload_file(local_file, bucket, s3_file_name, ExtraArgs={'ACL':'public-read'})
        print(str(s3_file_name) + " Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

## NOT using or updating the below function (updated 10/12/2021)
def split_data_export(df,n,prefix):
    arrays = np.array_split(df,n)
    df_dict = {}
    file_names = list()
    
    for i in range(n):
        df_dict["partition{0}".format(i+1)] = arrays[i]
   
    for key, value in df_dict.items():
        if n ==1:
            #name = f'/mnt/Churn_Prediction&GroundTruthData/{prefix}_{date.today()}.csv'
            #name = os.environ['DOMINO_WORKING_DIR']+'/temp/{0}_{1}.csv'.format(prefix,date.today())
            name = f'/domino/datasets/{domino_user}/{domino_project}/scratch/{prefix}_{date.today()}.csv')
        else:
            #name = f'/mnt/Churn_Prediction&GroundTruthData/{prefix}_{key}_{date.today()}.csv'
            name =f'/domino/datasets/{domino_user}/{domino_project}/scratch/{prefix}_{key}_{date.today()}.csv')
            
        print('CSV name: ', name)
        file_names.append(name)   
        value.to_csv(name, index = False)
 
    return file_names
