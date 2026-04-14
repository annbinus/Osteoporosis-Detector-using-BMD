import pandas as pd
import numpy as np
import boto3


BUCKET = 'osteo-s3-demo-bucket'
DXA = 'dxa.csv'
DEMOGRAPHICS = 'demo.csv'

s3 = boto3.client('s3')
s3.download_file(BUCKET, DXA, '/tmp/dxa.csv')
s3.download_file(BUCKET, DEMOGRAPHICS, '/tmp/demo.csv')

dxa = pd.read_csv('/tmp/dxa.csv')
demo = pd.read_csv('/tmp/demo.csv')

print(f'DXA rows: {len(dxa)}, DEMO rows: {len(demo)}')

sentinel = 5.397605346934028e-79

def replace_sentinel(df):
    return df.replace(sentinel, np.nan).where(df.abs() > 1e-70, other=np.nan)

dxa = replace_sentinel(dxa)
demo = replace_sentinel(demo)
dxa = dxa[dxa['DXAEXSTS'] == 1].copy()
print(f'Valid DXA rows: {len(dxa)}')