import os
import time
import mlflow
import mlflow.catboost
import mlflow.sklearn
import pandas as pd
import numpy as np
import psycopg
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)
from catboost import CatBoostClassifier
from mlflow.models import infer_signature
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder

# Загрузка переменных окружения
load_dotenv('../.env')

print("="*70)
print("ЗАДАНИЕ 3: Grid Search и Random Search с MLflow")
print("="*70)

# ============================================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================================
print("\n1️⃣ ЗАГРУЗКА ДАННЫХ")
print("="*70)

TABLE_NAME = "users_churn"

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
        cur.execute(f"SELECT * FROM {TABLE_NAME}")
        data = cur.fetchall()
        columns = [col[0] for col in cur.description]

df = pd.DataFrame(data, columns=columns)

print(f"✅ Данные загружены: {df.shape}")
print(f"Колонки: {df.columns.tolist()[:5]}...")

# ============================================================================
# ПОДГОТОВКА ДАННЫХ
# ============================================================================
print("\n2️⃣ ПОДГОТОВКА ДАННЫХ")
print("="*70)

# Удаляем ненужные колонки
columns_to_drop = ['id', 'customer_id', 'begin_date', 'end_date']
df_clean = df.drop(columns=columns_to_drop)

# Преобразуем категориальные признаки в числовые
label_encoders = {}
categorical_columns = df_clean.select_dtypes(include=['object']).columns

for col in categorical_columns:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le

# Разделение на признаки и целевую переменную
X = df_clean.drop('target', axis=1)
y = df_clean['target']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"✅ X_train: {X_train.shape}")
print(f"✅ X_test: {X_test.shape}")
print(f"✅ Признаков: {X.shape[1]}")

# ============================================================================
# НАСТРОЙКА MLFLOW
# ============================================================================
print("\n3️⃣ НАСТРОЙКА MLFLOW")
print("="*70)

# Настройка S3 для MLflow
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("S3_ACCESS_KEY")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("S3_SECRET_KEY")

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000  # Исправлен порт на 5000
EXPERIMENT_NAME = "catboost_hyperparameter_tuning"
REGISTRY_MODEL_NAME = "catboost-churn-model"

# Используем локальное хранилище MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_registry_uri("file:./mlruns")

print(f"✅ MLflow Tracking URI: {mlflow.get_tracking_uri()}")

# Создаём или получаем эксперимент
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
experiment_id = experiment.experiment_id
print(f"✅ Используется эксперимент: {EXPERIMENT_NAME}")

print(f"✅ Experiment ID: {experiment_id}")

# ============================================================================
# ПАРАМЕТРЫ МОДЕЛИ
# ============================================================================
loss_function = "Logloss"
task_type = 'CPU'
random_seed = 42
iterations = 300
verbose = False

# Сетка гиперпараметров для Grid Search
grid_params = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7]
}

# Распределение параметров для Random Search
random_params = {
    'depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.15, 0.2],
    'l2_leaf_reg': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

print("\n4️⃣ ПАРАМЕТРЫ ПОИСКА")
print("="*70)
print(f"Grid Search - комбинаций: {len(grid_params['depth']) * len(grid_params['learning_rate']) * len(grid_params['l2_leaf_reg'])}")
print(f"Random Search - итераций: 20")
print(f"Кросс-валидация: 2 фолда")

# ============================================================================
# GRID SEARCH
# ============================================================================
print("\n" + "="*70)
print("5️⃣ GRID SEARCH")
print("="*70)

# Создание модели
model_grid = CatBoostClassifier(
    loss_function=loss_function,
    task_type=task_type,
    random_seed=random_seed,
    iterations=iterations,
    verbose=verbose
)

# GridSearchCV с cv=2
grid_search = GridSearchCV(
    estimator=model_grid,
    param_grid=grid_params,
    cv=2,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print(f"\n🔍 Запуск Grid Search...")
print(f"   Всего комбинаций: {len(grid_params['depth']) * len(grid_params['learning_rate']) * len(grid_params['l2_leaf_reg'])}")

start_time = time.time()
grid_search.fit(X_train, y_train)
grid_search_time = time.time() - start_time

print(f"✅ Grid Search завершён за {grid_search_time:.2f} секунд")

# Получение результатов
best_params_grid = grid_search.best_params_
best_score_grid = grid_search.best_score_
cv_results_grid = pd.DataFrame(grid_search.cv_results_)

print(f"\n📊 Лучшие параметры Grid Search:")
for param, value in best_params_grid.items():
    print(f"   {param}: {value}")
print(f"\n📊 Лучший CV ROC-AUC: {best_score_grid:.4f}")

# Создание модели с лучшими параметрами
model_best_grid = CatBoostClassifier(
    loss_function=loss_function,
    task_type=task_type,
    random_seed=random_seed,
    iterations=iterations,
    verbose=verbose,
    **best_params_grid
)

model_best_grid.fit(X_train, y_train)

# Предсказания на тестовой выборке
prediction_grid = model_best_grid.predict(X_test)
probas_grid = model_best_grid.predict_proba(X_test)[:, 1]

# Расчёт метрик на тестовой выборке
_, err1_grid, _, err2_grid = confusion_matrix(y_test, prediction_grid, normalize='all').ravel()

metrics_grid = {
    "test_err1": err1_grid,
    "test_err2": err2_grid,
    "test_auc": roc_auc_score(y_test, probas_grid),
    "test_precision": precision_score(y_test, prediction_grid),
    "test_recall": recall_score(y_test, prediction_grid),
    "test_f1": f1_score(y_test, prediction_grid),
    "test_logloss": log_loss(y_test, prediction_grid),
    "cv_best_score": best_score_grid,
    "mean_fit_time": cv_results_grid["mean_fit_time"].mean(),
    "std_fit_time": cv_results_grid["std_fit_time"].mean(),
    "mean_test_score": cv_results_grid["mean_test_score"].mean(),
    "std_test_score": cv_results_grid["std_test_score"].mean(),
    "total_search_time": grid_search_time
}

print(f"\n📊 Метрики на тестовой выборке:")
print(f"   ROC-AUC: {metrics_grid['test_auc']:.4f}")
print(f"   Precision: {metrics_grid['test_precision']:.4f}")
print(f"   Recall: {metrics_grid['test_recall']:.4f}")
print(f"   F1-Score: {metrics_grid['test_f1']:.4f}")

# ============================================================================
# ЛОГИРОВАНИЕ GRID SEARCH В MLFLOW
# ============================================================================
print("\n📦 Логирование Grid Search в MLflow...")

signature = infer_signature(X_test, prediction_grid)
input_example = X_test[:10]

with mlflow.start_run(run_name='model_grid_search', experiment_id=experiment_id) as run:
    # Логирование параметров
    mlflow.log_params(best_params_grid)
    mlflow.log_param("iterations", iterations)
    mlflow.log_param("loss_function", loss_function)
    mlflow.log_param("task_type", task_type)
    mlflow.log_param("random_seed", random_seed)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("cv_folds", 2)
    mlflow.log_param("search_method", "GridSearchCV")
    mlflow.log_param("n_combinations", len(grid_params['depth']) * len(grid_params['learning_rate']) * len(grid_params['l2_leaf_reg']))

    # Логирование метрик
    mlflow.log_metrics(metrics_grid)

    # Логирование модели CatBoost
    model_info_grid = mlflow.catboost.log_model(
        cb_model=model_best_grid,
        artifact_path='models',
        signature=signature,
        input_example=input_example,
        registered_model_name=REGISTRY_MODEL_NAME
    )

    # Логирование объекта GridSearchCV
    mlflow.sklearn.log_model(grid_search, artifact_path='grid_search_cv')

    # Логирование результатов GridSearch в виде артефакта
    cv_results_grid.to_csv("grid_search_results.csv", index=False)
    mlflow.log_artifact("grid_search_results.csv")

    # Сохранение Run ID
    run_id_grid_search = run.info.run_id

    print(f"\n{'='*70}")
    print(f"✅ GRID SEARCH - RUN ID: {run_id_grid_search}")
    print(f"{'='*70}")

print(f"✅ Grid Search залогирован в MLflow!")

# ============================================================================
# RANDOM SEARCH
# ============================================================================
print("\n" + "="*70)
print("6️⃣ RANDOM SEARCH")
print("="*70)

# Создание модели
model_random = CatBoostClassifier(
    loss_function=loss_function,
    task_type=task_type,
    random_seed=random_seed,
    iterations=iterations,
    verbose=verbose
)

# RandomizedSearchCV с cv=2
random_search = RandomizedSearchCV(
    estimator=model_random,
    param_distributions=random_params,
    n_iter=20,  # количество случайных комбинаций
    cv=2,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
    random_state=random_seed
)

print(f"\n🎲 Запуск Random Search...")
print(f"   Количество итераций: 20")

start_time = time.time()
random_search.fit(X_train, y_train)
random_search_time = time.time() - start_time

print(f"✅ Random Search завершён за {random_search_time:.2f} секунд")

# Получение результатов
best_params_random = random_search.best_params_
best_score_random = random_search.best_score_
cv_results_random = pd.DataFrame(random_search.cv_results_)

print(f"\n📊 Лучшие параметры Random Search:")
for param, value in best_params_random.items():
    print(f"   {param}: {value}")
print(f"\n📊 Лучший CV ROC-AUC: {best_score_random:.4f}")

# Создание модели с лучшими параметрами
model_best_random = CatBoostClassifier(
    loss_function=loss_function,
    task_type=task_type,
    random_seed=random_seed,
    iterations=iterations,
    verbose=verbose,
    **best_params_random
)

model_best_random.fit(X_train, y_train)

# Предсказания на тестовой выборке
prediction_random = model_best_random.predict(X_test)
probas_random = model_best_random.predict_proba(X_test)[:, 1]

# Расчёт метрик на тестовой выборке
_, err1_random, _, err2_random = confusion_matrix(y_test, prediction_random, normalize='all').ravel()

metrics_random = {
    "test_err1": err1_random,
    "test_err2": err2_random,
    "test_auc": roc_auc_score(y_test, probas_random),
    "test_precision": precision_score(y_test, prediction_random),
    "test_recall": recall_score(y_test, prediction_random),
    "test_f1": f1_score(y_test, prediction_random),
    "test_logloss": log_loss(y_test, prediction_random),
    "cv_best_score": best_score_random,
    "mean_fit_time": cv_results_random["mean_fit_time"].mean(),
    "std_fit_time": cv_results_random["std_fit_time"].mean(),
    "mean_test_score": cv_results_random["mean_test_score"].mean(),
    "std_test_score": cv_results_random["std_test_score"].mean(),
    "total_search_time": random_search_time
}

print(f"\n📊 Метрики на тестовой выборке:")
print(f"   ROC-AUC: {metrics_random['test_auc']:.4f}")
print(f"   Precision: {metrics_random['test_precision']:.4f}")
print(f"   Recall: {metrics_random['test_recall']:.4f}")
print(f"   F1-Score: {metrics_random['test_f1']:.4f}")

# ============================================================================
# ЛОГИРОВАНИЕ RANDOM SEARCH В MLFLOW
# ============================================================================
print("\n📦 Логирование Random Search в MLflow...")

signature_random = infer_signature(X_test, prediction_random)
input_example_random = X_test[:10]

with mlflow.start_run(run_name='model_random_search', experiment_id=experiment_id) as run:
    # Логирование параметров
    mlflow.log_params(best_params_random)
    mlflow.log_param("iterations", iterations)
    mlflow.log_param("loss_function", loss_function)
    mlflow.log_param("task_type", task_type)
    mlflow.log_param("random_seed", random_seed)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("cv_folds", 2)
    mlflow.log_param("search_method", "RandomizedSearchCV")
    mlflow.log_param("n_iter", 20)

    # Логирование метрик
    mlflow.log_metrics(metrics_random)

    # Логирование модели CatBoost
    model_info_random = mlflow.catboost.log_model(
        cb_model=model_best_random,
        artifact_path='models',
        signature=signature_random,
        input_example=input_example_random,
        registered_model_name=REGISTRY_MODEL_NAME
    )

    # Логирование объекта RandomizedSearchCV
    mlflow.sklearn.log_model(random_search, artifact_path='random_search_cv')

    # Логирование результатов RandomSearch в виде артефакта
    cv_results_random.to_csv("random_search_results.csv", index=False)
    mlflow.log_artifact("random_search_results.csv")

    # Сохранение Run ID
    run_id_random_search = run.info.run_id

    print(f"\n{'='*70}")
    print(f"✅ RANDOM SEARCH - RUN ID: {run_id_random_search}")
    print(f"{'='*70}")

print(f"✅ Random Search залогирован в MLflow!")

# ============================================================================
# СРАВНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================================
print("\n" + "="*70)
print("7️⃣ СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("="*70)

comparison_df = pd.DataFrame({
    'Метод': ['Grid Search', 'Random Search'],
    'ROC-AUC (test)': [metrics_grid['test_auc'], metrics_random['test_auc']],
    'Precision': [metrics_grid['test_precision'], metrics_random['test_precision']],
    'Recall': [metrics_grid['test_recall'], metrics_random['test_recall']],
    'F1-Score': [metrics_grid['test_f1'], metrics_random['test_f1']],
    'CV Best Score': [metrics_grid['cv_best_score'], metrics_random['cv_best_score']],
    'Время (сек)': [grid_search_time, random_search_time]
})

print("\n📊 ИТОГОВАЯ ТАБЛИЦА СРАВНЕНИЯ:")
print(comparison_df.to_string(index=False))

# Лучшие параметры
print("\n" + "="*70)
print("📋 ЛУЧШИЕ ПАРАМЕТРЫ")
print("="*70)

print("\n🔍 Grid Search:")
for param, value in best_params_grid.items():
    print(f"   {param}: {value}")

print("\n🎲 Random Search:")
for param, value in best_params_random.items():
    print(f"   {param}: {value}")

# Разница во времени
print("\n" + "="*70)
print("⏱️  АНАЛИЗ ВРЕМЕНИ ВЫПОЛНЕНИЯ")
print("="*70)
print(f"\nGrid Search:   {grid_search_time:.2f} секунд ({grid_search_time/60:.2f} минут)")
print(f"Random Search: {random_search_time:.2f} секунд ({random_search_time/60:.2f} минут)")
print(f"\nРазница:       {abs(grid_search_time - random_search_time):.2f} секунд")

if random_search_time < grid_search_time:
    speedup = grid_search_time / random_search_time
    print(f"Random Search быстрее в {speedup:.2f} раз")
else:
    speedup = random_search_time / grid_search_time
    print(f"Grid Search быстрее в {speedup:.2f} раз")

# Разница в качестве
print("\n" + "="*70)
print("📊 АНАЛИЗ КАЧЕСТВА МОДЕЛЕЙ")
print("="*70)
print(f"\nROC-AUC на тесте:")
print(f"   Grid Search:   {metrics_grid['test_auc']:.4f}")
print(f"   Random Search: {metrics_random['test_auc']:.4f}")
print(f"   Разница:       {abs(metrics_grid['test_auc'] - metrics_random['test_auc']):.4f}")

if metrics_grid['test_auc'] > metrics_random['test_auc']:
    print(f"\n✅ Grid Search показал лучший результат")
else:
    print(f"\n✅ Random Search показал лучший результат")

print("\n" + "="*70)
print("✅ ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
print("="*70)
print(f"\n🔗 MLflow UI: http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
print(f"🔗 Эксперимент: {EXPERIMENT_NAME}")
print(f"\n📋 Run IDs для проверки:")
print(f"   Grid Search:   {run_id_grid_search}")
print(f"   Random Search: {run_id_random_search}")

# ============================================================================
# ЗАДАНИЕ 3: ФИНАЛЬНЫЕ ПЕРЕМЕННЫЕ ДЛЯ ПРОВЕРКИ
# ============================================================================
print("\n" + "="*70)
print("✅ ЗАДАНИЕ 3: ПЕРЕМЕННЫЕ ДЛЯ ПРОВЕРКИ")
print("="*70)

print(f'\nrun_id_grid_search = "{run_id_grid_search}"')
print(f'run_id_random_search = "{run_id_random_search}"')

print("\n" + "="*70)
print("📝 ИНСТРУКЦИЯ ДЛЯ ПРОВЕРКИ")
print("="*70)
print("\nСкопируйте значения выше и вставьте их в соответствующие переменные:")
print("\nrun_id_grid_search = '...'    # ваш код здесь")
print("run_id_random_search = '...'  # ваш код здесь")

print("\n" + "="*70)
print("🎯 ОСНОВНЫЕ ВЫВОДЫ")
print("="*70)

print("\n1. Подбор гиперпараметров:")
print(f"   - Grid Search проверил ВСЕ {len(grid_params['depth']) * len(grid_params['learning_rate']) * len(grid_params['l2_leaf_reg'])} комбинаций параметров")
print(f"   - Random Search проверил случайные 20 комбинаций")

print("\n2. Время выполнения:")
print(f"   - Grid Search: {grid_search_time:.2f} сек ({grid_search_time/60:.2f} мин)")
print(f"   - Random Search: {random_search_time:.2f} сек ({random_search_time/60:.2f} мин)")

print("\n3. Качество на тестовой выборке:")
print(f"   - Grid Search ROC-AUC: {metrics_grid['test_auc']:.4f}")
print(f"   - Random Search ROC-AUC: {metrics_random['test_auc']:.4f}")

print("\n4. Рекомендация:")
if random_search_time < grid_search_time and abs(metrics_grid['test_auc'] - metrics_random['test_auc']) < 0.01:
    print("   Random Search показал сопоставимое качество за меньшее время")
elif metrics_grid['test_auc'] > metrics_random['test_auc']:
    print("   Grid Search нашёл более качественную модель")
else:
    print("   Random Search нашёл более качественную модель")

print("\n" + "="*70)
print("✅ ЗАДАНИЕ 3 ВЫПОЛНЕНО!")
print("="*70)
