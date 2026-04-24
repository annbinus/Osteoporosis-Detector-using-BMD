# Osteoporosis Detector

A machine learning application that predicts osteoporosis risk using DXA (Dual-Energy X-ray Absorptiometry) scan data.

## Overview

This project uses XGBoost to classify bone density levels into three categories:
- **Normal**
- **Osteopenia**
- **Osteoporosis**

## Setup

### Prerequisites
- Python 3.9+
- AWS Account with S3 and SageMaker access

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Files

- `main.py` - Data loading and preprocessing from S3
- `train.py` - Model training script
- `deploy.py` - SageMaker endpoint deployment
- `inference.py` - Model inference handler for SageMaker
- `requirements.txt` - Python dependencies

## Usage

### Train the Model
```sh
python main.py
```

### Deploy to SageMaker
```sh
python deploy.py
```

## Configuration

Update these variables in your scripts:
- `BUCKET` - S3 bucket name
- `ROLE` - AWS IAM role ARN
- `endpoint_name` - SageMaker endpoint name

## License

This project is licensed under the MIT License.