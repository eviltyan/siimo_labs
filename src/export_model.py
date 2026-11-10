import json
import os
import shutil
from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))
c = MlflowClient()

# 1. Получаем версию по alias champion
v = c.get_model_version_by_alias('ckd-classifier', 'champion')
print(f"Экспортируем версию {v.version}...")

# 2. Загружаем модель
model = mlflow.sklearn.load_model(f'models:/ckd-classifier/{v.version}')

# 3. Сохраняем в папку
out = Path('exported_model')
if out.exists():
    shutil.rmtree(out)
mlflow.sklearn.save_model(model, str(out))

# 4. Загружаем данные для создания примеров
def load_and_clean_data():
    import io
    with open('chronic_kidney_disease.arff', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    attributes = []
    data_lines = []
    is_data = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith('%'): continue
        if line.lower() == '@data':
            is_data = True
            continue
        if not is_data:
            if line.lower().startswith('@attribute'):
                attributes.append(line.split()[1].strip("'\""))
        else:
            data_lines.append(line)
            
    parsed_data = []
    for line in data_lines:
        row = [val.strip() for val in line.split(',')]
        row = [np.nan if v == '?' else v for v in row]
        parsed_data.append(row[:25])
        
    df = pd.DataFrame(parsed_data, columns=attributes[:25])
    for col in df.columns:
        if col in ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = df[col].astype(str).str.replace('\t', '').str.strip()
            df[col] = df[col].replace({'nan': np.nan, '': np.nan})
    df['class'] = df['class'].astype(str).str.strip().str.replace('\t', '')
    df = df[df['class'].isin(['ckd', 'notckd'])].reset_index(drop=True)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    return df

df = load_and_clean_data()
X = df.drop(columns=['class'])
y = df['class']

# Берем те же test-индексы, что в train.py
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Создаем sample_request.json (3 объекта)
sample = X_test.head(3).to_dict(orient='records')
Path('sample_request.json').write_text(json.dumps({"instances": sample}, indent=2))

# 6. Создаем expected_predictions.json (эталонные ответы модели)
expected = model.predict(X_test.head(3)).tolist()
Path('expected_predictions.json').write_text(json.dumps(expected, indent=2))

# 7. Создаем metadata.json
metadata = {
    'name': 'ckd-classifier',
    'version': v.version,
    'run_id': v.run_id,
    'features': X.columns.tolist()
}
(out / 'metadata.json').write_text(json.dumps(metadata, indent=2))

print("Готово. Созданы: exported_model/, sample_request.json, expected_predictions.json")
print(f"Ожидаемые предсказания: {expected}")