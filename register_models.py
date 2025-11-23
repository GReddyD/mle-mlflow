#!/usr/bin/env python3
"""
Скрипт для регистрации моделей в MLflow Model Registry
"""
import os
import pandas as pd
import psycopg
from dotenv import load_dotenv
import mlflow
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Загружаем переменные окружения
load_dotenv()

# Настройка S3
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("S3_ACCESS_KEY")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("S3_SECRET_KEY")

# Подключаемся к MLflow серверу
mlflow.set_tracking_uri("http://127.0.0.1:5001")

print("=" * 80)
print("РЕГИСТРАЦИЯ МОДЕЛЕЙ В MODEL REGISTRY")
print("=" * 80)

# Загружаем данные
print("\n1. Загрузка данных...")
connection = {"sslmode": "require", "target_session_attrs": "read-write"}
postgres_credentials = {
    "host": os.getenv("DB_DESTINATION_HOST"),
    "port": os.getenv("DB_DESTINATION_PORT"),
    "dbname": os.getenv("DB_DESTINATION_NAME"),
    "user": os.getenv("DB_DESTINATION_USER"),
    "password": os.getenv("DB_DESTINATION_PASSWORD"),
}
connection.update(postgres_credentials)

with psycopg.connect(**connection) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users_churn")
        data = cur.fetchall()
        columns = [col[0] for col in cur.description]

df = pd.DataFrame(data, columns=columns)

# Подготовка данных
X = df.drop(['customer_id', 'begin_date', 'end_date', 'target'], axis=1)
y = df['target']
X = X.fillna('Missing')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cat_features = X_train.select_dtypes(include=['object']).columns.tolist()
for col in cat_features:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

EXPERIMENT_NAME = "churn_fio"
REGISTRY_MODEL_NAME = "churn_model_production"

mlflow.set_experiment(EXPERIMENT_NAME)

# Создаем две версии модели с разными параметрами
models_config = [
    {
        "name": "Model v1 (baseline)",
        "params": {"iterations": 100, "depth": 4, "learning_rate": 0.1},
        "description": "Baseline model with shallow trees"
    },
    {
        "name": "Model v2 (improved)",
        "params": {"iterations": 200, "depth": 6, "learning_rate": 0.05},
        "description": "Improved model with deeper trees and more iterations"
    }
]

registered_versions = []

for idx, config in enumerate(models_config, 1):
    print(f"\n{'=' * 80}")
    print(f"МОДЕЛЬ {idx}: {config['name']}")
    print('=' * 80)

    # Обучение модели
    print("\nОбучение модели...")
    model = CatBoostClassifier(
        iterations=config["params"]["iterations"],
        depth=config["params"]["depth"],
        learning_rate=config["params"]["learning_rate"],
        random_seed=42,
        verbose=False,
        cat_features=cat_features
    )

    model.fit(X_train, y_train)

    # Предсказания и метрики
    prediction = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    metrics = {
        "auc": float(roc_auc_score(y_test, probas)),
        "f1": float(f1_score(y_test, prediction)),
        "precision": float(precision_score(y_test, prediction)),
        "recall": float(recall_score(y_test, prediction))
    }

    print(f"AUC: {metrics['auc']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")

    # Логирование в MLflow
    print("\nЛогирование в MLflow...")
    with mlflow.start_run(run_name=config['name']) as run:
        run_id = run.info.run_id

        # Теги
        mlflow.set_tags({
            "model_version": f"v{idx}",
            "business_objective": "churn_prediction",
            "target_metric": "recall",
            "prediction_horizon": "30_days",
            "data_period": "2024-Q4",
            "feature_version": "v3.2",
            "deployment_ready": "true" if idx == 2 else "false"
        })

        # Метрики и параметры
        mlflow.log_metrics(metrics)
        mlflow.log_params(config["params"])

        # Логируем и регистрируем модель
        signature = mlflow.models.infer_signature(X_test, prediction)

        model_info = mlflow.catboost.log_model(
            cb_model=model,
            artifact_path="models",
            signature=signature,
            input_example=X_test[:5],
            registered_model_name=REGISTRY_MODEL_NAME,
            pip_requirements=["catboost", "scikit-learn", "pandas", "numpy"]
        )

        # Получаем версию зарегистрированной модели
        client = mlflow.tracking.MlflowClient()
        model_version = model_info.registered_model_version

        # Добавляем описание к версии модели
        client.update_model_version(
            name=REGISTRY_MODEL_NAME,
            version=model_version,
            description=config["description"]
        )

        registered_versions.append({
            "version": model_version,
            "name": config['name'],
            "run_id": run_id,
            "metrics": metrics
        })

        print(f"✅ Модель зарегистрирована как версия {model_version}")

print("\n" + "=" * 80)
print("ИТОГОВАЯ ИНФОРМАЦИЯ")
print("=" * 80)
print(f"\nНазвание зарегистрированной модели: {REGISTRY_MODEL_NAME}")
print(f"\nЗарегистрированные версии:")

for ver in registered_versions:
    print(f"\n  Версия {ver['version']}: {ver['name']}")
    print(f"    Run ID: {ver['run_id']}")
    print(f"    AUC: {ver['metrics']['auc']:.4f}")
    print(f"    F1: {ver['metrics']['f1']:.4f}")

print("\n" + "=" * 80)
print("СЛЕДУЮЩИЕ ШАГИ:")
print("=" * 80)
print("\n1. Откройте MLflow UI: http://127.0.0.1:5001")
print("\n2. Перейдите в раздел 'Models' (вверху страницы)")
print(f"\n3. Найдите модель '{REGISTRY_MODEL_NAME}'")
print("\n4. Для каждой версии:")
print("   - Нажмите на версию")
print("   - В правом верхнем углу нажмите 'Stage: None'")
print(f"   - Для версии {registered_versions[0]['version']} выберите 'Staging'")
print(f"   - Для версии {registered_versions[1]['version']} выберите 'Production'")

print("\n" + "=" * 80)
print("\nПосле установки stages используйте:")
print(f"\nmodel_name = \"{REGISTRY_MODEL_NAME}\"")
print(f"version_1 = \"{registered_versions[0]['version']}\"")
print(f"version_2 = \"{registered_versions[1]['version']}\"")
print(f"version_1_stage = \"Staging\"")
print(f"version_2_stage = \"Production\"")
print("=" * 80)
