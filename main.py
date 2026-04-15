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


# 'RIAGENDR' - Gender, 'RIDAGEYR' - Age, 'RIDRETH3' - Race
demo_cols = ['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH3']
df = pd.merge(dxa, demo[demo_cols], on='SEQN', how='inner')
df = df[df['RIDAGEYR'] >= 20].copy()
print(f'Merged rows: {len(df)}')

reference_pop = df[
    (df['RIDAGEYR'] >= 20) & 
    (df['RIDAGEYR'] < 30) &
    (df['DXDTOBMD'].notna())
].copy()

# Derive reference mean and SD for each gender and age group.

derived_ref = {}
for sex_code, sex_label in [(1, 'Male'), (2, 'Female')]:
    sex_data = reference_pop[reference_pop['RIAGENDR'] == sex_code]['DXDTOBMD']
    mean_val = sex_data.mean()
    sd_val   = sex_data.std()
    # Same reference mean/SD applies to all age groups (that's how T-score works)
    for age_group in ['20-29','30-39','40-49','50-59','60-69','70+']:
        derived_ref[(sex_code, age_group)] = (mean_val, sd_val)

print("\nDerived NHANES_REF:")
for key, val in derived_ref.items():
    print(f"  {key}: ({val[0]:.4f}, {val[1]:.4f})")