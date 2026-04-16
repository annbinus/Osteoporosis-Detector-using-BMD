import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
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
for gender_code, gender_label in [(1, 'Male'), (2, 'Female')]:
    gender_data = reference_pop[reference_pop['RIAGENDR'] == gender_code]['DXDTOBMD']
    mean_val = gender_data.mean()
    sd_val   = gender_data.std()
    # Same reference mean/SD applies to all age groups 
    for age_group in ['20-29','30-39','40-49','50-59','60-69','70+']:
        derived_ref[(gender_code, age_group)] = (mean_val, sd_val)

print("\nDerived NHANES_REF:")
for key, val in derived_ref.items():
    print(f"  {key}: ({val[0]:.4f}, {val[1]:.4f})")


def get_age_group(age):
    if age < 30:
        return '20-29'
    elif age < 40:
        return '30-39'
    elif age < 50:
        return '40-49'
    elif age < 60:
        return '50-59'
    elif age < 70:
        return '60-69'
    else:
        return '70+'
    
def compute_t_score(row):
    gender, age, bmd = row['RIAGENDR'], row['RIDAGEYR'], row['DXDTOBMD']
    if pd.isna(bmd) or pd.isna(age) or pd.isna(gender):
        return np.nan
    key = (int(gender), get_age_group(age))
    if key not in derived_ref:
        return np.nan
    mean_val, sd_val = derived_ref[key]
    if pd.isna(mean_val) or pd.isna(sd_val):
        return np.nan
    return (bmd - mean_val) / sd_val if sd_val else np.nan

def classify_tscore(t):
    if pd.isna(t):
        return np.nan
    elif t<= -2.5:
        return 2
    elif t <= -1.0:
        return 1
    else:
        return 0
    
df['T_SCORE'] = df.apply(compute_t_score, axis=1)
df['Bone_Status'] = df['T_SCORE'].apply(classify_tscore)
df = df.dropna(subset=['Bone_Status']).copy()
df['Bone_Status'] = df['Bone_Status'].astype(int)

label_map = {0: 'Normal', 1: 'Osteopenia', 2: 'Osteoporosis'}
print(df['Bone_Status'].value_counts().rename(index=label_map))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
counts = df['Bone_Status'].value_counts().rename(index=label_map)
axes[0].bar(counts.index, counts.values, color=['green', 'orange', 'red'])
axes[0].set_title('Bone Status Distribution', fontsize=14)

for cls, color in zip([0,1,2], ['green', 'orange', 'red']):
    subset = df[df['Bone_Status'] == cls]['DXDTOBMD'].dropna()
    axes[1].hist(subset, bins=30, alpha=0.7, label=label_map[cls], color=color)

axes[1].set_title('DXA BMD Distribution by Bone Status', fontsize=14)
axes[1].set_xlabel('DXA BMD', fontsize=12)
axes[1].legend()

plt.tight_layout()
plt.savefig('/tmp/bone_status_analysis.png', dpi=150)
s3.upload_file('/tmp/bone_status_analysis.png', BUCKET, 'outputs/bone_status_analysis.png')
plt.show()

# BMD features

bmd_cols = [
    'DXDTOBMD',   # Total body
    'DXDSTBMD',   # Subtotal
    'DXXLSBMD',   # Lumbar spine ← most important
    'DXXPEBMD',   # Pelvis
    'DXXTSBMD',   # Thoracic spine
    'DXXLRBMD',   # Left rib
    'DXXRRBMD',   # Right rib
    'DXXLLBMD',   # Left leg
    'DXXRLBMD',   # Right leg
    'DXXLABMD',   # Left arm
    'DXXRABMD',   # Right arm
    'DXXHEBMD',   # Head
    'DXDTRBMD',   # Trunk
]

# BMC features

bmc_cols = [
    'DXDTOBMC',   # Total body
    'DXDSTBMC',   # Subtotal
    'DXXLSBMC',   # Lumbar spine
    'DXXPEBMC',   # Pelvis
]

# Body composition features
comp_cols = [
    'DXDTOFAT',   # Total fat mass
    'DXDTOLE',    # Total lean mass
    'DXDTOPF',    # Total percent fat
    'DXDSTFAT',   # Subtotal fat
    'DXDSTLE',    # Subtotal lean
    'DXDSTPF',    # Subtotal percent fat
]

# Demographics
demo_cols = ['RIAGENDR', 'RIDAGEYR', 'RIDRETH3']

# Derived ratios
df['bmc_to_lean'] = df['DXDTOBMC'] / df['DXDTOLE']
df['fat_to_lean'] = df['DXDTOFAT'] / df['DXDTOLE']

all_features = bmd_cols + bmc_cols + comp_cols + demo_cols + ['bmc_to_lean', 'fat_to_lean']
feature_cols = [col for col in all_features if col in df.columns]

X = df[feature_cols]
y = df['Bone_Status']

print(f'Features: {len(feature_cols)}, Samples: {len(X)}')
print(f'\nFeatures selected:')
for col in feature_cols:
    print(f' - {col}')

# Train/Test Split + SMOTE

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50, stratify=y)

imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

smote = SMOTE(random_state=50)
X_train_bal, y_train_bal = smote.fit_resample(X_train_imp, y_train)

print(f'Train size after SMOTE: {len(X_train_bal)}')
print(pd.Series(y_train_bal).value_counts().rename(index=label_map))

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=50,
    eval_metric='mlogloss'
)

cv_scores = cross_val_score(model, X_train_bal, y_train_bal, cv=5, scoring='f1_macro')
print(f'CV F1 Macro Scores: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

model.fit(X_train_bal, y_train_bal)
