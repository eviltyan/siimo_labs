import argparse
import os
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn

# --- Загрузка и предобработка ---
def load_and_clean_data():
    import pandas as pd
    import numpy as np
    
    # 1. Читаем файл как текст, находим начало данных и заголовки
    with open('chronic_kidney_disease.arff', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    attributes = []
    data_lines = []
    is_data = False
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        if line.lower() == '@data':
            is_data = True
            continue
            
        if not is_data:
            # Парсим атрибуты: @attribute age numeric
            if line.lower().startswith('@attribute'):
                parts = line.split()
                attr_name = parts[1].strip("'\"")
                attributes.append(attr_name)
        else:
            # Парсим данные
            data_lines.append(line)
            
    # 2. Разбираем данные
    parsed_data = []
    for line in data_lines:
        # Убираем пробелы вокруг значений и обрабатываем пропуски
        row = [val.strip() for val in line.split(',')]
        # Заменяем '?' на пустоту
        row = [np.nan if v == '?' else v for v in row]
        parsed_data.append(row)
        
    # 3. Создаем DataFrame
    # Берем только первые 25 значений из каждой строки (отбрасываем лишние, если есть)
    parsed_data = [row[:25] for row in parsed_data]
    df = pd.DataFrame(parsed_data, columns=attributes[:25])
    
    # 4. Приводим типы (то, что scipy не смог сделать)
    for col in df.columns:
        # Если колонка числовая, конвертируем
        if col in ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # Категориальные чистим от табов
            df[col] = df[col].astype(str).str.replace('\t', '').str.strip()
            df[col] = df[col].replace({'nan': np.nan, '': np.nan})
            
    # 5. Чистим целевую переменную (ОЧЕНЬ ВАЖНО!)
    df['class'] = df['class'].astype(str).str.strip().str.replace('\t', '')
    # Удаляем возможные пустые строки (если class = NaN)
    df = df[df['class'].isin(['ckd', 'notckd'])].reset_index(drop=True)

    # Удаляем id
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    return df

# --- Основной скрипт ---
p = argparse.ArgumentParser()
p.add_argument('--algorithm', choices=['lr', 'rf'], default='lr')
p.add_argument('--c', type=float, default=1.0)
p.add_argument('--trees', type=int, default=100)
p.add_argument('--depth', type=int, default=5)
p.add_argument('--register', action='store_true')
args = p.parse_args()

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))
mlflow.set_experiment('lab4-ckd')

df = load_and_clean_data()
y = df['class']
X = df.drop(columns=['class'])

# Разбиение 60/20/20
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev)

# Препроцессинг (ColumnTransformer)
num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ])

if args.algorithm == 'lr':
    clf = LogisticRegression(C=args.c, max_iter=2000, random_state=42)
else:
    clf = RandomForestClassifier(n_estimators=args.trees, max_depth=args.depth, random_state=42, n_jobs=1)

model = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])

with mlflow.start_run(run_name=f'ckd-{args.algorithm}'):
    # Параметры
    mlflow.log_params({'algorithm': args.algorithm, 'dataset': 'CKD', 'n_train': len(X_train), 'n_val': len(X_val)})
    if args.algorithm == 'lr':
        mlflow.log_params({'C': args.c})
    else:
        mlflow.log_params({'trees': args.trees, 'depth': args.depth})
        
    # Обучение и метрики
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    
    val_acc = accuracy_score(y_val, pred)
    val_f1 = f1_score(y_val, pred, average='macro')
    
    mlflow.log_metrics({'val_accuracy': val_acc, 'val_f1_macro': val_f1})
    
    # Логирование модели
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, artifact_path='model', signature=signature, input_example=X_train.head(2))
    
    print(f'val_accuracy={val_acc:.4f}, val_f1_macro={val_f1:.4f}')
    
    if args.register:
        # Регистрация модели
        result = mlflow.register_model(f'runs:/{mlflow.active_run().info.run_id}/model', 'ckd-classifier')
        print(f'REGISTERED_VERSION={result.version}')