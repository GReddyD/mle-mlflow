#!/bin/bash
set -a
source .env
set +a

export MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
export AWS_ACCESS_KEY_ID=$S3_ACCESS_KEY
export AWS_SECRET_ACCESS_KEY=$S3_SECRET_KEY
export AWS_BUCKET_NAME=$S3_BUCKET_NAME

mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root s3://$S3_BUCKET_NAME \
  --host 0.0.0.0 \
  --port 5001
