#!/bin/bash
set -a
source .env
set +a

export MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
export AWS_ACCESS_KEY_ID=$S3_ACCESS_KEY
export AWS_SECRET_ACCESS_KEY=$S3_SECRET_KEY

# Формируем URL базы данных
DB_URL="postgresql://$DB_DESTINATION_USER:$DB_DESTINATION_PASSWORD@$DB_DESTINATION_HOST:$DB_DESTINATION_PORT/$DB_DESTINATION_NAME"

# Используем mlflow db stamp для пометки базы как актуальной
mlflow db stamp --backend-store-uri "$DB_URL" heads

echo "✓ База данных помечена как актуальная"
