#!/usr/bin/env python3
"""
Скрипт для обучения модели и логирования в MLflow
"""
import os
import pandas as pd
import psycopg
from dotenv import load_dotenv
import mlflow
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    log_loss
)

# Загружаем переменные окружения
load_dotenv()

# Настройка S3
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("S3_ACCESS_KEY")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("S3_SECRET_KEY")

# Подключаемся к MLflow серверу
mlflow.set_tracking_uri("http://127.0.0.1:5001")

print("=" * 80)
print("ОБУЧЕНИЕ И ЛОГИРОВАНИЕ МОДЕЛИ В MLFLOW")
print("=" * 80)

# Загружаем данные из PostgreSQL
print("\n1. Загрузка данных из PostgreSQL...")
connection = {"sslmode": "require", "target_session_attrs": "read-write"}
postgres_credentials = {
    "host": os.getenv("DB_DESTINATION_HOST"),
    "port": os.getenv("DB_DESTINATION_PORT"),
    "dbname": os.getenv("DB_DESTINATION_NAME"),
    "user": os.getenv("DB_DESTINATION_USER"),
    "password": os.getenv("DB_DESTINATION_PASSWORD"),
}

connection.update(postgres_credentials)
TABLE_NAME = "users_churn"

with psycopg.connect(**connection) as conn:
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {TABLE_NAME}")
        data = cur.fetchall()
        columns = [col[0] for col in cur.description]

df = pd.DataFrame(data, columns=columns)
print(f"   Загружено {len(df)} строк")

# Подготовка данных
print("\n2. Подготовка данных...")
# Удаляем ненужные колонки
X = df.drop(['customer_id', 'begin_date', 'end_date', 'target'], axis=1)
y = df['target']

# Заполняем пропуски
X = X.fillna('Missing')

# Разделяем на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Train: {len(X_train)} строк")
print(f"   Test: {len(X_test)} строк")

# Определяем категориальные признаки (все текстовые колонки)
cat_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Конвертируем все категориальные признаки в строки
for col in cat_features:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

print(f"   Категориальных признаков: {len(cat_features)}")

# Обучение модели
print("\n3. Обучение модели CatBoost...")
model = CatBoostClassifier(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    random_seed=42,
    verbose=False,
    cat_features=cat_features
)

model.fit(X_train, y_train)
print("   Модель обучена!")

# Предсказания
print("\n4. Получение предсказаний...")
prediction = model.predict(X_test)
probas = model.predict_proba(X_test)[:, 1]

# Вычисление метрик
print("\n5. Вычисление метрик...")
tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
err1 = fp / (tn + fp + fn + tp)  # False Positive Rate
err2 = fn / (tn + fp + fn + tp)  # False Negative Rate

metrics = {
    "err1": float(err1),
    "err2": float(err2),
    "auc": float(roc_auc_score(y_test, probas)),
    "precision": float(precision_score(y_test, prediction)),
    "recall": float(recall_score(y_test, prediction)),
    "f1": float(f1_score(y_test, prediction)),
    "logloss": float(log_loss(y_test, prediction))
}

print(f"   AUC: {metrics['auc']:.4f}")
print(f"   F1: {metrics['f1']:.4f}")
print(f"   Precision: {metrics['precision']:.4f}")
print(f"   Recall: {metrics['recall']:.4f}")

# Логирование в MLflow
print("\n6. Логирование модели в MLflow...")
EXPERIMENT_NAME = "churn_fio"
RUN_NAME = "catboost_model_v1"

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name=RUN_NAME) as run:
    run_id = run.info.run_id

    # Логируем метрики
    mlflow.log_metrics(metrics)

    # Логируем параметры модели
    mlflow.log_params({
        "iterations": 100,
        "depth": 6,
        "learning_rate": 0.1
    })

    # Создаем сигнатуру модели
    signature = mlflow.models.infer_signature(X_test, prediction)
    input_example = X_test[:10]

    # Логируем модель
    model_info = mlflow.catboost.log_model(
        cb_model=model,
        artifact_path="models",
        signature=signature,
        input_example=input_example,
        pip_requirements=[
            "catboost",
            "scikit-learn",
            "pandas",
            "numpy"
        ]
    )

    print(f"   Run ID: {run_id}")
    print(f"   Model URI: {model_info.model_uri}")

# Получаем путь к артефактам
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
bucket_name = os.getenv("S3_BUCKET_NAME")
s3_path = f"s3://{bucket_name}/{experiment.experiment_id}/{run_id}/artifacts/models"

print("\n" + "=" * 80)
print("✓ ГОТОВО!")
print("=" * 80)
print(f"\nПуть к модели в S3:")
print(s3_path)
print("\nЭтот путь содержит:")
print("  - MLmodel")
print("  - model.cb")
print("  - conda.yaml")
print("  - python_env.yaml")
print("  - requirements.txt")
print("  - input_example.json")
print("=" * 80)
