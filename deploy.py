import sagemaker
from sagemaker.xgboost import XGBoostModel
import os

BUCKET = 'osteo-s3-demo-bucket'
ROLE   = 'arn:aws:iam::654654339831:role/SageMakerExecutionRole'

sklearn_model = XGBoostModel(
    model_data        = f's3://{BUCKET}/models/model.tar.gz',
    role              = ROLE,
    entry_point       = 'inference.py',
    framework_version = '1.7-1'   # XGBoost version
)

predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium',
    endpoint_name='dxa-osteoporosis-v3'
)

print('Endpoint live: dxa-osteoporosis-v3')
