#!/bin/bash
# ── Setup OCR Pipeline ────────────────────────────────────────────────────
# Run this once to create the S3 upload bucket, Lambda function,
# and wire the S3 trigger.

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION="us-east-1"
RESULTS_BUCKET="osteo-s3-demo-bucket"
UPLOAD_BUCKET="osteo-dxa-uploads"
LAMBDA_NAME="dxa-report-parser"
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/SageMakerExecutionRole"

echo "Account: $ACCOUNT_ID"
echo "Upload bucket: $UPLOAD_BUCKET"

# ── Step 1: Create upload bucket ─────────────────────────────────────────
echo ""
echo "Creating upload bucket..."
aws s3 mb s3://$UPLOAD_BUCKET --region $REGION
echo "Done: s3://$UPLOAD_BUCKET"

# ── Step 2: Add Textract permissions to your IAM role ────────────────────
echo ""
echo "Adding Textract permissions..."
aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonTextractFullAccess
echo "Done"

# ── Step 3: Package Lambda function ──────────────────────────────────────
echo ""
echo "Packaging Lambda..."
zip lambda.zip lambda_function.py
echo "Done: lambda.zip"

# ── Step 4: Create Lambda function ───────────────────────────────────────
echo ""
echo "Creating Lambda function..."
aws lambda create-function \
  --function-name $LAMBDA_NAME \
  --runtime python3.10 \
  --handler lambda_function.lambda_handler \
  --role $ROLE_ARN \
  --zip-file fileb://lambda.zip \
  --timeout 300 \
  --memory-size 512 \
  --environment "Variables={
    ENDPOINT_NAME=dxa-osteoporosis-v3,
    RESULTS_BUCKET=$RESULTS_BUCKET
  }"
echo "Done: Lambda created"

# ── Step 5: Allow S3 to invoke Lambda ────────────────────────────────────
echo ""
echo "Adding S3 trigger permission..."
aws lambda add-permission \
  --function-name $LAMBDA_NAME \
  --statement-id s3-trigger \
  --action lambda:InvokeFunction \
  --principal s3.amazonaws.com \
  --source-arn arn:aws:s3:::$UPLOAD_BUCKET \
  --source-account $ACCOUNT_ID
echo "Done"

# ── Step 6: Add S3 event notification ────────────────────────────────────
echo ""
echo "Wiring S3 trigger to Lambda..."

LAMBDA_ARN=$(aws lambda get-function \
  --function-name $LAMBDA_NAME \
  --query 'Configuration.FunctionArn' \
  --output text)

aws s3api put-bucket-notification-configuration \
  --bucket $UPLOAD_BUCKET \
  --notification-configuration "{
    \"LambdaFunctionConfigurations\": [
      {
        \"LambdaFunctionArn\": \"$LAMBDA_ARN\",
        \"Events\": [\"s3:ObjectCreated:*\"],
        \"Filter\": {
          \"Key\": {
            \"FilterRules\": [
              {\"Name\": \"suffix\", \"Value\": \".png\"},
              {\"Name\": \"suffix\", \"Value\": \".jpg\"},
              {\"Name\": \"suffix\", \"Value\": \".jpeg\"},
              {\"Name\": \"suffix\", \"Value\": \".pdf\"}
            ]
          }
        }
      }
    ]
  }"
echo "Done"

echo ""
echo "=========================================="
echo "Pipeline ready!"
echo "Upload bucket:   s3://$UPLOAD_BUCKET"
echo "Results bucket:  s3://$RESULTS_BUCKET/results/"
echo "Lambda function: $LAMBDA_NAME"
echo ""
echo "Test it:"
echo "  aws s3 cp your_dxa_report.pdf s3://$UPLOAD_BUCKET/"
echo "  aws logs tail /aws/lambda/$LAMBDA_NAME --follow"
echo "=========================================="