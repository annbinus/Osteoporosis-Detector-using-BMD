# test_endpoint.py
import boto3, json
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

BUCKET   = 'osteo-s3-demo-bucket'
SENTINEL = 5.397605346934028e-79

FEATURE_COLS = [
    'DXXPEBMD', 'DXXTSBMD', 'DXXLRBMD', 'DXXRRBMD',
    'DXXLLBMD', 'DXXRLBMD', 'DXXLABMD', 'DXXRABMD',
    'DXXHEBMD', 'DXDTRBMD', 'DXXPEBMC', 'DXDTOFAT',
    'DXDTOLE',  'DXDTOPF',  'DXDSTFAT', 'DXDSTLE',
    'DXDSTPF',  'RIAGENDR', 'RIDAGEYR', 'RIDRETH1',
    'bmc_to_lean', 'fat_to_lean'
]

# ── Load and prep data ────────────────────────────────────────────────────
dxa  = pd.read_csv('1999-2000/DXX 1999-2000.csv').replace(SENTINEL, np.nan)
demo = pd.read_csv('1999-2000/DEMO 1999-2000.csv').replace(SENTINEL, np.nan)

df = pd.merge(dxa, demo[['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH1']], on='SEQN')
df = df[df['DXAEXSTS'] == 1]
df = df[df['RIDAGEYR'] >= 50]

df['bmc_to_lean'] = df['DXXPEBMC'] / df['DXDTOLE']
df['fat_to_lean'] = df['DXDTOFAT'] / df['DXDTOLE']

# Get first complete row
first_row = df[FEATURE_COLS].dropna().iloc[0]
sample    = first_row.tolist()

# ── Call endpoint ─────────────────────────────────────────────────────────
client = boto3.client('sagemaker-runtime', region_name='us-east-1')

response = client.invoke_endpoint(
    EndpointName='dxa-osteoporosis-v3',
    ContentType='application/json',
    Body=json.dumps(sample)
)

result = json.loads(response['Body'].read())
print(f"Patient info:")
print(f"  Age: {int(first_row['RIDAGEYR'])}")
print(f"  Sex: {'Male' if first_row['RIAGENDR']==1 else 'Female'}")
print()
print(f"Prediction:  {result['prediction']}")
print(f"Confidence:  {result['confidence']:.1%}")
print(f"Probabilities:")
for cls, prob in result['probabilities'].items():
    print(f"  {cls:15s}: {prob:.1%}")