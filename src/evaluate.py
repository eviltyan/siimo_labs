import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import mlflow.sklearn

# Загружаем экспортированную модель
model = mlflow.sklearn.load_model('exported_model')
meta = json.loads(Path('exported_model/metadata.json').read_text())

# Загружаем и чистим данные
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
                parts = line.split()
                attributes.append(parts[1].strip("'\""))
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
y = df['class']
X = df.drop(columns=['class'])

# Разбиваем (60/20/20)
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev)

# Предсказания
pred = model.predict(X_test)

# Сохраняем метрики
r = {
    'test_accuracy': accuracy_score(y_test, pred),
    'test_f1_macro': f1_score(y_test, pred, average='macro')
}

Path('test_metrics.json').write_text(json.dumps(r, indent=2))
print(r)