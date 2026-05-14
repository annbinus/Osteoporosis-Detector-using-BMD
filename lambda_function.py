import boto3
import json
import re
import os
import urllib.parse
from datetime import datetime

# ── AWS Clients ───────────────────────────────────────────────────────────
s3        = boto3.client('s3')
textract  = boto3.client('textract')
sagemaker = boto3.client('sagemaker-runtime')

# ── Config ────────────────────────────────────────────────────────────────
ENDPOINT_NAME  = os.environ.get('ENDPOINT_NAME', 'dxa-osteoporosis-v3')
RESULTS_BUCKET = os.environ.get('RESULTS_BUCKET', 'osteo-s3-demo-bucket')
RESULTS_PREFIX = 'results'

FEATURE_COLS = [
    'DXXPEBMD', 'DXXTSBMD', 'DXXLRBMD', 'DXXRRBMD',
    'DXXLLBMD', 'DXXRLBMD', 'DXXLABMD', 'DXXRABMD',
    'DXXHEBMD', 'DXDTRBMD', 'DXXPEBMC', 'DXDTOFAT',
    'DXDTOLE',  'DXDTOPF',  'DXDSTFAT', 'DXDSTLE',
    'DXDSTPF',  'RIAGENDR', 'RIDAGEYR', 'RIDRETH3',
    'bmc_to_lean', 'fat_to_lean'
]


# ── Main Handler ──────────────────────────────────────────────────────────
def lambda_handler(event, context):
    # Get uploaded file info from S3 event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key    = urllib.parse.unquote_plus(
                event['Records'][0]['s3']['object']['key'])

    print(f'Processing: s3://{bucket}/{key}')

    try:
        # Step 1 — Extract text via Textract
        text = extract_text(bucket, key)
        print(f'Extracted {len(text)} characters from document')

        # Step 2 — Parse BMD values from text
        values = parse_bmd_values(text)
        print(f'Parsed values: {values}')

        # Step 3 — Validate we have enough data
        missing = [k for k, v in values.items() if v is None]
        if len(missing) > len(FEATURE_COLS) * 0.5:
            raise ValueError(f'Too many missing values: {missing}')

        # Step 4 — Build feature vector
        sample = build_feature_vector(values)

        # Step 5 — Call SageMaker endpoint
        prediction = call_endpoint(sample)
        print(f'Prediction: {prediction}')

        # Step 6 — Build result object
        result = {
            'timestamp':    datetime.utcnow().isoformat(),
            'source_file':  f's3://{bucket}/{key}',
            'parsed_values': values,
            'missing_fields': missing,
            'prediction':   prediction['prediction'],
            'confidence':   prediction['confidence'],
            'probabilities': prediction['probabilities'],
            'status':       'success'
        }

    except Exception as e:
        print(f'Error: {str(e)}')
        result = {
            'timestamp':   datetime.utcnow().isoformat(),
            'source_file': f's3://{bucket}/{key}',
            'status':      'error',
            'error':       str(e)
        }

    # Step 7 — Save result to S3
    result_key = save_result(result, key)
    print(f'Result saved to s3://{RESULTS_BUCKET}/{result_key}')

    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }


# ── Textract OCR ──────────────────────────────────────────────────────────
def extract_text(bucket, key):
    file_ext = key.lower().split('.')[-1]

    if file_ext == 'pdf':
        # PDFs need async Textract job
        response = textract.start_document_text_detection(
            DocumentLocation={'S3Object': {'Bucket': bucket, 'Name': key}}
        )
        job_id = response['JobId']
        return wait_for_textract_job(job_id)

    else:
        # Images (png, jpg) can be processed synchronously
        response = textract.detect_document_text(
            Document={'S3Object': {'Bucket': bucket, 'Name': key}}
        )
        return ' '.join([
            block['Text']
            for block in response['Blocks']
            if block['BlockType'] == 'LINE'
        ])


def wait_for_textract_job(job_id):
    import time
    while True:
        response = textract.get_document_text_detection(JobId=job_id)
        status   = response['JobStatus']

        if status == 'SUCCEEDED':
            return ' '.join([
                block['Text']
                for block in response['Blocks']
                if block['BlockType'] == 'LINE'
            ])
        elif status == 'FAILED':
            raise RuntimeError(f'Textract job failed: {job_id}')

        print(f'Textract job {status} — waiting...')
        time.sleep(3)


# ── BMD Value Parser ──────────────────────────────────────────────────────
def parse_bmd_values(text):
    """
    Extract BMD values from DXA report text.
    Patterns cover common DXA report formats (Hologic, GE Lunar).
    """
    patterns = {
    'DXXPEBMD': [
        r'Pelvis\s+[\d.]+\s+[\d.]+\s+([\d.]+)',      # Pelvis  
    ],
    'DXXTSBMD': [
        r'T\s*Spine\s+[\d.]+\s+[\d.]+\s+([\d.]+)',   # T Spine 
    ],
    'DXXLLBMD': [
        r'L\s*Leg\s+[\d.]+\s+[\d.]+\s+([\d.]+)',     # L Leg  
    ],
    'DXXRLBMD': [
        r'R\s*Leg\s+[\d.]+\s+[\d.]+\s+([\d.]+)',     # R Leg  
    ],
    'DXXLABMD': [
        r'L\s*Arm\s+[\d.]+\s+[\d.]+\s+([\d.]+)',     # L Arm  
    ],
    'DXXRABMD': [
        r'R\s*Arm\s+[\d.]+\s+[\d.]+\s+([\d.]+)',     # R Arm  
    ],
    'DXXHEBMD': [
        r'Head\s+[\d.]+\s+[\d.]+\s+([\d.]+)',        # Head  
    ],
    'DXDTRBMD': [
        r'Subtotal\s+[\d.]+\s+[\d.]+\s+([\d.]+)',    # Subtotal
    ],
    'DXXLRBMD': [
        r'L\s*Ribs?\s+[\d.]+\s+[\d.]+\s+([\d.]+)',  # L Ribs 
    ],
    'DXXRRBMD': [
        r'R\s*Ribs?\s+[\d.]+\s+[\d.]+\s+([\d.]+)',  # R Ribs 
    ],
    'DXXPEBMC': [
        r'Pelvis\s+[\d.]+\s+([\d.]+)',               # Pelvis 
    ],
    'RIDAGEYR': [
        r'Age[\s:]+(\d{2,3})',
    ],
    'RIAGENDR': [
        r'Sex[\s:]+([MmFf]\w*)',
        r'Gender[\s:]+([MmFf]\w*)',
    ],
}

    values = {}
    for col, pattern_list in patterns.items():
        found = None
        for pattern in pattern_list:
            match = re.search(pattern, text)
            if match:
                raw = match.group(1)
                if col == 'RIAGENDR':
                    found = 1.0 if raw.upper().startswith('M') else 2.0
                else:
                    try:
                        found = float(raw)
                    except ValueError:
                        continue
                break
        values[col] = found

    return values


# ── Feature Vector Builder ────────────────────────────────────────────────
def build_feature_vector(values):
    """Build ordered feature list matching FEATURE_COLS, with None for missing."""

    # Compute derived ratios if possible
    bmc  = values.get('DXXPEBMC')
    lean = values.get('DXDTOLE')
    fat  = values.get('DXDTOFAT')

    values['bmc_to_lean'] = bmc / lean   if bmc  and lean else None
    values['fat_to_lean'] = fat / lean   if fat  and lean else None

    # Fill subtotal columns with total if missing
    for fat_col, sub in [('DXDTOFAT','DXDSTFAT'), ('DXDTOLE','DXDSTLE'), ('DXDTOPF','DXDSTPF')]:
        if values.get(sub) is None and values.get(fat_col) is not None:
            values[sub] = values[fat_col] * 0.95  # subtotal ≈ 95% of total

    # RIDRETH3 — default to 3 (Non-Hispanic White) if missing
    if values.get('RIDRETH3') is None:
        values['RIDRETH3'] = 3.0

    return [values.get(col) for col in FEATURE_COLS]


# ── SageMaker Call ────────────────────────────────────────────────────────
def call_endpoint(sample):
    # Replace None with 0.0 — imputer in the model handles missing values
    sample_clean = [v if v is not None else 0.0 for v in sample]

    response = sagemaker.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        Body=json.dumps(sample_clean)
    )
    return json.loads(response['Body'].read())


# ── Save Result to S3 ─────────────────────────────────────────────────────
def save_result(result, source_key):
    # Result filename mirrors source filename
    base_name  = source_key.split('/')[-1].rsplit('.', 1)[0]
    timestamp  = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    result_key = f'{RESULTS_PREFIX}/{base_name}_{timestamp}.json'

    s3.put_object(
        Bucket=RESULTS_BUCKET,
        Key=result_key,
        Body=json.dumps(result, indent=2),
        ContentType='application/json'
    )
    return result_key