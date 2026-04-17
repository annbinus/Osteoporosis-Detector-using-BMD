import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import joblib, os, tarfile
import boto3


BUCKET = 'osteo-s3-demo-bucket'

s3 = boto3.client('s3')

CYCLES = {
    '1999-2000': {'dxa': '1999-2000/DXX 1999-2000.csv', 'demo': '1999-2000/DEMO 1999-2000.csv'},
    '2001-2002': {'dxa': '2001-2002/DXX 2001-02.csv', 'demo': '2001-2002/DEMO 2001-02.csv'},
    '2003-2004': {'dxa': '2003-2004/DXX 2003-04.csv', 'demo': '2003-2004/DEMO 2003-04.csv'},
    '2005-2006': {'dxa': '2005-2006/DXX 2005-06.csv', 'demo': '2005-2006/DEMO 2005-06.csv'},
    '2020':      {'dxa': '2020/DXA 2020.csv',      'demo': '2020/DEMO 2020.csv'},
}

sentinel = 5.397605346934028e-79

def replace_sentinel(df):
    return df.replace(sentinel, np.nan).where(df.abs() > 1e-70, other=np.nan)

dfs = []

for cycle, paths in CYCLES.items():

    # Download from S3 → local temp files
    dxa_path = f'/tmp/dxa_{cycle}.xpt'
    demo_path = f'/tmp/demo_{cycle}.xpt'

    s3.download_file(BUCKET, paths['dxa'], dxa_path)
    s3.download_file(BUCKET, paths['demo'], demo_path)

    dxa = pd.read_csv(dxa_path)
    demo = pd.read_csv(demo_path)

    dxa = replace_sentinel(dxa)
    demo = replace_sentinel(demo)  

    dxa = dxa[dxa['DXAEXSTS'] == 1].copy()
    eth_col = 'RIDRETH3' if 'RIDRETH3' in demo.columns else 'RIDRETH1'
    merged = pd.merge(dxa, demo[['SEQN', 'RIAGENDR', 'RIDAGEYR', eth_col]], on='SEQN', how='inner')

    merged['CYCLE'] = cycle
    merged = merged.rename(columns={eth_col: 'RIDRETH3'})
    dfs.append(merged)

    print(f'  {cycle}: {len(merged)} valid records')

df_all = pd.concat(dfs, ignore_index=True)
print(f"Total records: {len(df_all)}")


df = df_all[df_all['RIDAGEYR'] >= 50].copy()
print(f'Adults 50+: {len(df)}')
print(f"Sex: {df['RIAGENDR'].value_counts().rename({1:'Male', 2:'Female'}).to_dict()}")
print(f"Age range: {df['RIDAGEYR'].min():.0f} - {df['RIDAGEYR'].max():.0f}")

reference_pop = df_all[
    (df_all['RIDAGEYR'] >= 20) & 
    (df_all['RIDAGEYR'] < 30) &
    (df_all['DXDTOBMD'].notna())
].copy()

# Calculate mean and SD by sex
ref_values = (
    reference_pop
    .groupby('RIAGENDR')['DXDTOBMD']
    .agg(mean_bmd='mean', sd_bmd='std')
)
print('\nDerived reference values:')
print(ref_values.round(4))

# Derive reference mean and SD for each gender and age group.

derived_ref = {}
for gender_code in [1, 2]:
    mean_val = ref_values.loc[gender_code, 'mean_bmd']
    sd_val   = ref_values.loc[gender_code, 'sd_bmd']
    for age_group in ['50-59', '60-69', '70+']:
        derived_ref[(gender_code, age_group)] = (mean_val, sd_val)

print('\nDerived NHANES_REF:')
for key, val in derived_ref.items():
    print(f'  {key}: ({val[0]:.4f}, {val[1]:.4f})')


def get_age_group(age):
    if age < 60:
        return '50-59'
    elif age < 70:
        return '60-69'
    else:
        return '70+'
    
def compute_t_score(row):
    gender, age, bmd = row['RIAGENDR'], row['RIDAGEYR'], row['DXXLSBMD']
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
    eval_metric='logloss'
)

cv_scores = cross_val_score(model, X_train_bal, y_train_bal, cv=5, scoring='f1_macro')
print(f'CV F1 Macro Scores: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

model.fit(X_train_bal, y_train_bal)

y_pred = model.predict(X_test_imp)
y_proba = model.predict_proba(X_test_imp)
print(classification_report(y_test, y_pred, target_names=label_map.values()))
print(f'ROC AUC: {roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"):.4f}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))


cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=label_map.values()).plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix', fontsize=14)

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
colors = ['#2ecc71', '#f39c12', '#e74c3c']
labels = ['Normal', 'Osteopenia', 'Osteoporosis']

for i, (color, label) in enumerate(zip(colors, labels)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    axes[1].plot(fpr, tpr, color=color, label=f'{label} (AUC={auc(fpr,tpr):.2f})')

axes[1].plot([0, 1], [0, 1], 'k--')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve (One vs Rest)', fontsize=14)
axes[1].legend()

plt.tight_layout()
plt.savefig('/tmp/model_evaluation.png', dpi=150)
s3.upload_file('/tmp/model_evaluation.png', BUCKET, 'outputs/model_evaluation.png')
plt.show()

importance_df = pd.DataFrame({
    'feature':    feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance').tail(15)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(importance_df['feature'], importance_df['importance'], color='#3498db')
ax.set_title('Top 15 Feature Importances', fontsize=14)
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig('/tmp/feature_importance.png', dpi=150)
boto3.client('s3').upload_file('/tmp/feature_importance.png', BUCKET, 'outputs/feature_importance.png')
plt.show()


os.makedirs('/tmp/model', exist_ok=True)
joblib.dump(model,        '/tmp/model/model.joblib')
joblib.dump(imputer,      '/tmp/model/imputer.joblib')
joblib.dump(feature_cols, '/tmp/model/feature_cols.joblib')

with tarfile.open('/tmp/model.tar.gz', 'w:gz') as tar:
    for fname in ['model.joblib', 'imputer.joblib', 'feature_cols.joblib']:
        tar.add(f'/tmp/model/{fname}', arcname=fname)

boto3.client('s3').upload_file('/tmp/model.tar.gz', BUCKET, 'models/model.tar.gz')
print('Model saved to s3://osteo-s3-demo-bucket/models/model.tar.gz')


boto3.client('s3').upload_file(
    '/tmp/model.tar.gz',
    BUCKET,
    'models/model.tar.gz'
)
print(f'Uploaded to s3://{BUCKET}/models/model.tar.gz')