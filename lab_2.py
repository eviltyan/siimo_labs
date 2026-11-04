import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras import layers, models as keras_models, callbacks
import datetime

import time
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 200)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)

print('pandas:', pd.__version__)
print('numpy :', np.__version__)

df = pd.read_csv('chronic_kidney_disease.csv')
df.columns = df.columns.str.strip()

print('Размер датасета:', df.shape)
print('\nПервые 5 строк:')
print(df.head())
print('\nТипы данных и non-null counts:')
df.info()

df.describe(include='all').T

# 1.1 Очистка "грязных" значений и приведение типов

df_clean = df.copy()

# --- 1) Целевая переменная ---
df_clean['classification'] = df_clean['classification'].astype(str).str.strip()

# --- 2) Категориальные: strip + замена '?', '\t?', '[null]', '' на NaN ---
cat_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
for col in cat_cols:
    df_clean[col] = (df_clean[col]
                     .astype(str)
                     .str.strip()
                     .replace({r'\t\?': np.nan, r'\?': np.nan, '?': np.nan,
                               '[null]': np.nan, 'nan': np.nan, '': np.nan}))

# --- 3) Числовые: to_numeric с errors='coerce' (ловит '\t?', '?') ---
num_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc',
            'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
for col in num_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# --- 4) Удаляем служебный id ---
df_clean = df_clean.drop(columns=['id'])

print('Типы после очистки:')
print(df_clean.dtypes)
print('\nРазмер после очистки:', df_clean.shape)

# 1.2 Анализ пропусков

missing = df_clean.isna().sum().sort_values(ascending=False)
missing_pct = (missing / len(df_clean) * 100).round(2)

miss_df = pd.DataFrame({'Пропусков': missing, 'Доля, %': missing_pct})
miss_df = miss_df[miss_df['Пропусков'] > 0]
print('=== Признаки с пропусками ===')
print(miss_df)

# Визуализация пропусков
plt.figure(figsize=(14, 4))
sns.heatmap(df_clean.isna(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Карта пропусков в датасете CKD')
plt.tight_layout()
plt.show()

# 1.3 Распределение целевой переменной

class_counts = df_clean['classification'].value_counts()
print(class_counts)
print('\nДоли, %:')
print((class_counts / len(df_clean) * 100).round(2))

# Bar chart
ax = class_counts.plot(kind='bar', color=['#4C72B0', '#DD8452'], edgecolor='black')
ax.set_title('Распределение классов (ckd / notckd)')
ax.set_xlabel('Класс')
ax.set_ylabel('Количество объектов')
for i, v in enumerate(class_counts.values):
    ax.text(i, v + 2, f'{v} ({v/len(df_clean)*100:.1f}%)', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

# 2.1 Импутация пропусков (медиана для чисел, мода для категорий)

df_prep = df_clean.copy()

# Отделяем целевую переменную
y = df_prep['classification']
X = df_prep.drop(columns=['classification'])

# Определяем колонки
# (после очистки pcv, wc, rc стали float64, поэтому они попадут в числовые)
num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# 1) Импутация числовых медианой
imputer_num = SimpleImputer(strategy='median')
X[num_cols] = imputer_num.fit_transform(X[num_cols])

# 2) Импутация категориальных модой (самым частым значением)
imputer_cat = SimpleImputer(strategy='most_frequent')
X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])

print('Пропусков после импутации:', X.isna().sum().sum())

# 2.2 Кодирование категориальных признаков

# Используем One-Hot Encoding (pd.get_dummies)
# drop_first=False оставляем для наглядности, чтобы не терять информацию

X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=int)

print('Размерность после кодирования:', X_encoded.shape)
print('Пример названий колонок:', X_encoded.columns.tolist()[:10])

# 2.3 Разбиение на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split

# stratify=y гарантирует, что пропорция ckd/notckd сохранится и в train, и в test
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=0.2,           # 20% на тест
    random_state=RANDOM_STATE,
    stratify=y               # сохраняем пропорции классов!
)

print('Размер X_train:', X_train.shape)
print('Размер X_test :', X_test.shape)
print('\nПропорции классов в train:')
print(y_train.value_counts(normalize=True).round(3))
print('\nПропорции классов в test:')
print(y_test.value_counts(normalize=True).round(3))

# 2.4 Масштабирование числовых признаков (StandardScaler)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Обучаем scaler ТОЛЬКО на train, чтобы избежать утечки данных (data leakage)
# и применяем к train и test
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols]  = scaler.transform(X_test[num_cols])

print('Пример статистик после масштабирования (train):')
print(X_train_scaled[num_cols].describe().loc[['mean', 'std']].round(2))

# 3.1 Универсальная функция для оценки модели

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    # 1. Предсказания
    y_pred = model.predict(X_te)

    # 2. Вероятности для класса 'ckd' (он же положительный = 1)
    # model.classes_ возвращает ['ckd', 'notckd'], нам нужен индекс 'ckd'
    ckd_idx = list(model.classes_).index('ckd')
    y_proba = model.predict_proba(X_te)[:, ckd_idx]

    # 3. Бинарные метки для ROC-AUC: 1 = ckd, 0 = notckd
    y_te_bin = (y_te == 'ckd').astype(int)

    return {
        'Модель': name,
        'Accuracy':  accuracy_score(y_te, y_pred),
        'Precision': precision_score(y_te, y_pred, pos_label='ckd'),
        'Recall':    recall_score(y_te, y_pred, pos_label='ckd'),
        'F1-Score':  f1_score(y_te, y_pred, pos_label='ckd'),
        'ROC-AUC':   roc_auc_score(y_te_bin, y_proba),  # <-- ИСПРАВЛЕНО
        'Время (с)': round(train_time, 4),
        '_y_pred': y_pred,
        '_y_proba': y_proba,
        '_model': model
    }

# 3.2 Базовые модели (без настройки гиперпараметров)

# Для MultinomialNB/ComplementNB/BernoulliNB нужны неотрицательные данные.
# Поэтому для них используем MinMax-масштабированные версии (см. ниже),
# а для остальных — StandardScaler.
from sklearn.preprocessing import MinMaxScaler

# Готовим MinMax-версию (только для NB-вариантов, требующих >= 0)
scaler_mm = MinMaxScaler()
X_train_mm = X_train.copy()
X_test_mm  = X_test.copy()
X_train_mm[num_cols] = scaler_mm.fit_transform(X_train[num_cols])
X_test_mm[num_cols]  = scaler_mm.transform(X_test[num_cols])

models = [
    ('GaussianNB',       GaussianNB(),                    X_train_scaled, X_test_scaled),
    ('MultinomialNB',    MultinomialNB(),                 X_train_mm,     X_test_mm),
    ('ComplementNB',     ComplementNB(),                  X_train_mm,     X_test_mm),
    ('BernoulliNB',      BernoulliNB(binarize=0.5),       X_train_mm,     X_test_mm),
    ('Decision Tree',    DecisionTreeClassifier(random_state=RANDOM_STATE), X_train_scaled, X_test_scaled),
    ('LDA',              LinearDiscriminantAnalysis(),    X_train_scaled, X_test_scaled),
    ('SVM (RBF)',        SVC(probability=True, random_state=RANDOM_STATE), X_train_scaled, X_test_scaled),
    ('kNN (k=5)',        KNeighborsClassifier(n_neighbors=5), X_train_scaled, X_test_scaled),
]

results = []
for name, model, Xtr, Xte in models:
    res = evaluate_model(name, model, Xtr, y_train, Xte, y_test)
    results.append(res)
    print(f'{name:18s} | Acc={res["Accuracy"]:.4f} | F1={res["F1-Score"]:.4f} | AUC={res["ROC-AUC"]:.4f}')

# 3.3 Сводная таблица метрик

results_df = pd.DataFrame([
    {k: v for k, v in r.items() if not k.startswith('_')}
    for r in results
]).set_index('Модель')

print('=== Сводная таблица метрик (базовые модели) ===')
print(results_df.round(4).to_string())

# 3.4 Сравнение метрик — столбчатая диаграмма

metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

fig, ax = plt.subplots(figsize=(13, 6))
results_df[metrics_to_plot].plot(kind='bar', ax=ax, width=0.8)
ax.set_title('Сравнение моделей по метрикам (базовые гиперпараметры)')
ax.set_ylabel('Значение метрики')
ax.set_ylim(0, 1.05)
ax.legend(loc='lower right', ncol=5)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

# 3.5 ROC-кривые всех моделей

# Бинарные метки (1 = ckd)
y_test_bin = (y_test == 'ckd').astype(int)

plt.figure(figsize=(10, 8))

# Разные стили линий, чтобы кривые не сливались
styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

for i, r in enumerate(results):
    fpr, tpr, _ = roc_curve(y_test_bin, r['_y_proba'])
    plt.plot(fpr, tpr, linestyle=styles[i % len(styles)], color=colors[i % len(colors)],
             label=f"{r['Модель']} (AUC={r['ROC-AUC']:.4f})", linewidth=2, alpha=0.8)

plt.plot([0, 1], [0, 1], 'k--', label='Случайная модель', linewidth=1.5)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC-кривые базовых моделей')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10) # легенда сбоку
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 3.6 ROC-кривые каждой модели отдельно (сетка 2x4)

y_test_bin = (y_test == 'ckd').astype(int)

# Создаём сетку 2 строки x 4 столбца (8 моделей)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()  # разворачиваем в одномерный массив для удобства цикла

for i, r in enumerate(results):
    fpr, tpr, _ = roc_curve(y_test_bin, r['_y_proba'])

    # Рисуем кривую для текущей модели
    axes[i].plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {r['ROC-AUC']:.4f}")
    axes[i].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # диагональ

    axes[i].set_title(r['Модель'], fontsize=12, fontweight='bold')
    axes[i].set_xlabel('False Positive Rate')
    axes[i].set_ylabel('True Positive Rate')
    axes[i].legend(loc='lower right')
    axes[i].grid(alpha=0.3)

plt.suptitle('ROC-кривые для каждой модели (базовые гиперпараметры)', fontsize=16)
plt.tight_layout()
plt.show()

# 3.7 Матрицы ошибок для БАЗОВЫХ моделей (до настройки)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

for i, r in enumerate(results):
    # Берем предсказания базовой модели из словаря results
    cm = confusion_matrix(y_test, r['_y_pred'], labels=['ckd', 'notckd'])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=axes[i], cbar=False,
                xticklabels=['ckd', 'notckd'],
                yticklabels=['ckd', 'notckd'])

    axes[i].set_title(r['Модель'], fontweight='bold')
    axes[i].set_xlabel('Предсказано')
    axes[i].set_ylabel('Истинно')

plt.suptitle('Матрицы ошибок (базовые модели, до настройки)', fontsize=16)
plt.tight_layout()
plt.show()

# 4.1 Исследование влияния max_depth на Decision Tree

depths = range(1, 21)
dt_acc = []
dt_f1  = []

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_STATE)
    dt.fit(X_train_scaled, y_train)
    y_pred = dt.predict(X_test_scaled)
    dt_acc.append(accuracy_score(y_test, y_pred))
    dt_f1.append(f1_score(y_test, y_pred, pos_label='ckd'))

plt.figure(figsize=(10, 5))
plt.plot(depths, dt_acc, marker='o', label='Accuracy', color='blue')
plt.plot(depths, dt_f1, marker='s', label='F1-Score', color='orange')
plt.title('Влияние глубины дерева (max_depth) на качество')
plt.xlabel('max_depth')
plt.ylabel('Значение метрики')
plt.xticks(depths)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

best_dt_depth = depths[int(np.argmax(dt_f1))]
print(f'\nЛучшая глубина по F1: {best_dt_depth}')

# 4.2 Исследование влияния n_neighbors на kNN

k_values = range(1, 31, 2) # 1, 3, 5, ..., 29
knn_acc = []
knn_f1  = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    knn_acc.append(accuracy_score(y_test, y_pred))
    knn_f1.append(f1_score(y_test, y_pred, pos_label='ckd'))

plt.figure(figsize=(10, 5))
plt.plot(k_values, knn_acc, marker='o', label='Accuracy', color='green')
plt.plot(k_values, knn_f1, marker='s', label='F1-Score', color='red')
plt.title('Влияние числа соседей (k) на качество kNN')
plt.xlabel('n_neighbors (k)')
plt.ylabel('Значение метрики')
plt.xticks(k_values)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

best_k = list(k_values)[int(np.argmax(knn_f1))]
print(f'\nЛучшее число соседей по F1: {best_k}')

# 4.3 Исследование влияния C и gamma на SVM (RBF)

C_values = [0.01, 0.1, 1, 10, 100]
gamma_values = [0.001, 0.01, 0.1, 1, 'scale']

svm_results = []
for C in C_values:
    for gamma in gamma_values:
        svm = SVC(C=C, gamma=gamma, kernel='rbf', random_state=RANDOM_STATE)
        svm.fit(X_train_scaled, y_train)
        y_pred = svm.predict(X_test_scaled)
        f1 = f1_score(y_test, y_pred, pos_label='ckd')
        svm_results.append({'C': C, 'gamma': gamma, 'F1': f1})

svm_df = pd.DataFrame(svm_results).pivot(index='C', columns='gamma', values='F1')

# Тепловая карта
plt.figure(figsize=(8, 5))
sns.heatmap(svm_df, annot=True, fmt='.3f', cmap='YlGnBu')
plt.title('F1-Score SVM (RBF) в зависимости от C и gamma')
plt.ylabel('C (жёсткость границы)')
plt.xlabel('gamma (радиус влияния)')
plt.tight_layout()
plt.show()

# 4.4 Исследование влияния alpha на MultinomialNB и ComplementNB

alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
mnb_f1, cnb_f1, bnb_f1 = [], [], []

for a in alphas:
    # MultinomialNB
    mnb = MultinomialNB(alpha=a)
    mnb.fit(X_train_mm, y_train)
    mnb_f1.append(f1_score(y_test, mnb.predict(X_test_mm), pos_label='ckd'))

    # ComplementNB
    cnb = ComplementNB(alpha=a)
    cnb.fit(X_train_mm, y_train)
    cnb_f1.append(f1_score(y_test, cnb.predict(X_test_mm), pos_label='ckd'))

    # BernoulliNB
    bnb = BernoulliNB(alpha=a, binarize=0.5)
    bnb.fit(X_train_mm, y_train)
    bnb_f1.append(f1_score(y_test, bnb.predict(X_test_mm), pos_label='ckd'))

plt.figure(figsize=(10, 5))
plt.plot(alphas, mnb_f1, marker='o', label='MultinomialNB')
plt.plot(alphas, cnb_f1, marker='s', label='ComplementNB')
plt.plot(alphas, bnb_f1, marker='^', label='BernoulliNB')
plt.xscale('log') # логарифмическая шкала для наглядности
plt.title('Влияние параметра сглаживания (alpha) на F1-Score')
plt.xlabel('alpha (log scale)')
plt.ylabel('F1-Score')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# 4.5 Исследование влияния solver на LDA

solvers = ['svd', 'lsqr', 'eigen']
lda_results = []

for solver in solvers:
    try:
        lda = LinearDiscriminantAnalysis(solver=solver)
        lda.fit(X_train_scaled, y_train)
        y_pred = lda.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label='ckd')
        lda_results.append({'solver': solver, 'Accuracy': acc, 'F1': f1})
    except Exception as e:
        print(f'solver={solver} error: {e}')

lda_df = pd.DataFrame(lda_results)
print(lda_df.round(4).to_string(index=False))

# 4.6 Сравнение базовых и настроенных моделей

# Настроенные модели (впиши лучшие параметры, которые найдешь по графикам!)
best_dt = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
best_knn = KNeighborsClassifier(n_neighbors=5)
best_svm = SVC(C=1, gamma='scale', probability=True, random_state=RANDOM_STATE)
best_mnb = MultinomialNB(alpha=1.0)

# Обучаем и оцениваем
tuned_models = [
    ('Decision Tree (tuned)', best_dt, X_train_scaled, X_test_scaled),
    ('kNN (tuned)', best_knn, X_train_scaled, X_test_scaled),
    ('SVM (tuned)', best_svm, X_train_scaled, X_test_scaled),
    ('MultinomialNB (tuned)', best_mnb, X_train_mm, X_test_mm),
]

tuned_results = []
for name, model, Xtr, Xte in tuned_models:
    res = evaluate_model(name, model, Xtr, y_train, Xte, y_test)
    tuned_results.append(res)

tuned_df = pd.DataFrame([
    {k: v for k, v in r.items() if not k.startswith('_')} for r in tuned_results
]).set_index('Модель')

print('=== Сравнение базовых и настроенных моделей ===')
print(tuned_df.round(4).to_string())

# Лучшие параметры (по результатам Шага 4)
final_models = [
    ('GaussianNB',       GaussianNB(), X_train_scaled, X_test_scaled),
    ('MultinomialNB',    MultinomialNB(alpha=0.01), X_train_mm, X_test_mm),
    ('ComplementNB',     ComplementNB(alpha=0.01), X_train_mm, X_test_mm),
    ('BernoulliNB',      BernoulliNB(alpha=0.001, binarize=0.5), X_train_mm, X_test_mm),
    ('Decision Tree',    DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE), X_train_scaled, X_test_scaled),
    ('LDA',              LinearDiscriminantAnalysis(solver='svd'), X_train_scaled, X_test_scaled),
    ('SVM (RBF)',        SVC(C=1.0, gamma=0.1, probability=True, random_state=RANDOM_STATE), X_train_scaled, X_test_scaled),
    ('kNN (k=3)',        KNeighborsClassifier(n_neighbors=3), X_train_scaled, X_test_scaled),
]

final_results = []
for name, model, Xtr, Xte in final_models:
    res = evaluate_model(name, model, Xtr, y_train, Xte, y_test)
    final_results.append(res)
    print(f'{name:18s} | Acc={res["Accuracy"]:.4f} | F1={res["F1-Score"]:.4f} | AUC={res["ROC-AUC"]:.4f}')

    # 5.1 ROC-кривые финальных моделей

y_test_bin = (y_test == 'ckd').astype(int)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

for i, r in enumerate(final_results):
    fpr, tpr, _ = roc_curve(y_test_bin, r['_y_proba'])

    axes[i].plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {r['ROC-AUC']:.4f}")
    axes[i].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

    axes[i].set_title(r['Модель'], fontweight='bold')
    axes[i].set_xlabel('False Positive Rate')
    axes[i].set_ylabel('True Positive Rate')
    axes[i].legend(loc='lower right')
    axes[i].grid(alpha=0.3)

plt.suptitle('ROC-кривые (финальные настроенные модели)', fontsize=16)
plt.tight_layout()
plt.show()

# 5.2 Матрицы ошибок (Confusion Matrices) для финальных моделей

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

for i, r in enumerate(final_results):
    cm = confusion_matrix(y_test, r['_y_pred'], labels=['ckd', 'notckd'])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                xticklabels=['ckd', 'notckd'],
                yticklabels=['ckd', 'notckd'])

    axes[i].set_title(r['Модель'], fontweight='bold')
    axes[i].set_xlabel('Предсказано')
    axes[i].set_ylabel('Истинно')

plt.suptitle('Матрицы ошибок (финальные настроенные модели)', fontsize=16)
plt.tight_layout()
plt.show()

# 5.3 Итоговая сводная таблица финальных моделей

summary_df = pd.DataFrame([
    {k: v for k, v in r.items() if not k.startswith('_')} for r in final_results
]).set_index('Модель')

print('=== Итоговое качество финальных настроенных моделей ===')
print(summary_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']].round(4).to_string())

# 6.1 Подготовка данных

print('TensorFlow version:', tf.__version__)

# Переводим целевую переменную в бинарный формат (1 = ckd, 0 = notckd)
y_train_nn = (y_train == 'ckd').astype(int).values
y_test_nn  = (y_test == 'ckd').astype(int).values

# Наши данные уже масштабированы (StandardScaler), это отлично для нейросети
X_train_nn = X_train_scaled.values.astype('float32')
X_test_nn  = X_test_scaled.values.astype('float32')

print('Размер train:', X_train_nn.shape)
print('Размер test :', X_test_nn.shape)

# 6.2 Создание и компиляция нейронной сети

tf.random.set_seed(RANDOM_STATE)

def build_model(hidden_layers=[64, 32], activation='relu', dropout_rate=0.2, learning_rate=0.001):
    model = keras_models.Sequential()

    # Входной слой + первый скрытый
    model.add(layers.Dense(hidden_layers[0], activation=activation, input_shape=(X_train_nn.shape[1],)))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))

    # Остальные скрытые слои
    for units in hidden_layers[1:]:
        model.add(layers.Dense(units, activation=activation))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    # Выходной слой (вероятность ckd)
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Собираем базовую модель
model_base = build_model()
model_base.summary()

# 6.3 Обучение модели с логированием в TensorBoard

# Создаём папку для логов с уникальным именем (дата + время)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Обучаем модель
history_base = model_base.fit(
    X_train_nn, y_train_nn,
    validation_split=0.2,       # 20% от train уйдёт на валидацию
    epochs=50,
    batch_size=32,
    callbacks=[tensorboard_callback],
    verbose=1
)

# 6.4 Визуализация процесса обучения

def plot_history(history, title_suffix=''):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # График потерь
    axes[0].plot(history.history['loss'], label='Train Loss', color='blue')
    axes[0].plot(history.history['val_loss'], label='Val Loss', color='orange')
    axes[0].set_title(f'Потери (Loss) {title_suffix}')
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('Binary Crossentropy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # График точности
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', color='green')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', color='red')
    axes[1].set_title(f'Точность (Accuracy) {title_suffix}')
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_history(history_base, '(базовая модель)')

# 6.5 Эксперимент: эпохи и размер батча

experiments = [
    {'epochs': 10, 'batch_size': 32, 'name': '10 эпох, batch=32'},
    {'epochs': 50, 'batch_size': 32, 'name': '50 эпох, batch=32'},
    {'epochs': 50, 'batch_size': 128, 'name': '50 эпох, batch=128'},
]

histories_exp = {}
for exp in experiments:
    print(f"\n--- Обучение: {exp['name']} ---")
    tf.random.set_seed(RANDOM_STATE)
    model = build_model() # пересоздаём модель с нуля
    h = model.fit(
        X_train_nn, y_train_nn,
        validation_split=0.2,
        epochs=exp['epochs'],
        batch_size=exp['batch_size'],
        verbose=0
    )
    histories_exp[exp['name']] = h

# Визуализация всех экспериментов
plt.figure(figsize=(14, 6))
colors = ['blue', 'green', 'red', 'purple']

for i, (name, h) in enumerate(histories_exp.items()):
    plt.plot(h.history['val_loss'], label=f'Val Loss ({name})', color=colors[i], linestyle='-')
    plt.plot(h.history['val_accuracy'], label=f'Val Acc ({name})', color=colors[i], linestyle='--')

plt.title('Сравнение экспериментов: эпохи и размер батча')
plt.xlabel('Эпоха')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 6.6 Эксперимент: архитектура сети

arch_experiments = [
    {'hidden_layers': [16], 'dropout_rate': 0.0, 'name': '1 слой (16), без dropout'},
    {'hidden_layers': [64, 32], 'dropout_rate': 0.2, 'name': '2 слоя (64-32), dropout 0.2'},
    {'hidden_layers': [128, 64, 32], 'dropout_rate': 0.3, 'name': '3 слоя (128-64-32), dropout 0.3'},
]

arch_histories = {}
for exp in arch_experiments:
    print(f"\n--- Архитектура: {exp['name']} ---")
    tf.random.set_seed(RANDOM_STATE)
    model = build_model(hidden_layers=exp['hidden_layers'], dropout_rate=exp['dropout_rate'])
    h = model.fit(
        X_train_nn, y_train_nn,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=0
    )
    arch_histories[exp['name']] = h

# Визуализация архитектур
plt.figure(figsize=(14, 6))
colors = ['blue', 'green', 'red']

for i, (name, h) in enumerate(arch_histories.items()):
    plt.plot(h.history['val_loss'], label=f'Val Loss ({name})', color=colors[i])
    plt.plot(h.history['val_accuracy'], label=f'Val Acc ({name})', color=colors[i], linestyle='--')

plt.title('Сравнение архитектур нейронной сети')
plt.xlabel('Эпоха')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 6.7 Финальная оценка лучшей нейросети на тестовой выборке

# Обучим финальную модель (2 слоя, dropout 0.2, 50 эпох, batch=32)
tf.random.set_seed(RANDOM_STATE)
final_nn = build_model(hidden_layers=[64, 32], dropout_rate=0.2)
final_nn.fit(X_train_nn, y_train_nn, validation_split=0.2, epochs=50, batch_size=32, verbose=0)

# Предсказания
y_pred_nn_proba = final_nn.predict(X_test_nn).ravel() # вероятности
y_pred_nn = (y_pred_nn_proba > 0.5).astype(int)      # классы (порог 0.5)

# Метрики
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print('=== Метрики нейросети на тестовой выборке ===')
print(f'Accuracy : {accuracy_score(y_test_nn, y_pred_nn):.4f}')
print(f'Precision: {precision_score(y_test_nn, y_pred_nn):.4f}')
print(f'Recall   : {recall_score(y_test_nn, y_pred_nn):.4f}')
print(f'F1-Score : {f1_score(y_test_nn, y_pred_nn):.4f}')
print(f'ROC-AUC  : {roc_auc_score(y_test_nn, y_pred_nn_proba):.4f}')

# 7.1 Итоговая сводная таблица всех методов

# 1. Собираем результаты классических моделей
final_summary = []
for r in final_results:
    final_summary.append({
        'Метод': r['Модель'],
        'Accuracy': r['Accuracy'],
        'Precision': r['Precision'],
        'Recall': r['Recall'],
        'F1-Score': r['F1-Score'],
        'ROC-AUC': r['ROC-AUC'],
        'Тип': 'Классический ML'
    })

# 2. Добавляем результаты нейросети
final_summary.append({
    'Метод': 'Нейронная сеть (2 слоя)',
    'Accuracy': accuracy_score(y_test_nn, y_pred_nn),
    'Precision': precision_score(y_test_nn, y_pred_nn),
    'Recall': recall_score(y_test_nn, y_pred_nn),
    'F1-Score': f1_score(y_test_nn, y_pred_nn),
    'ROC-AUC': roc_auc_score(y_test_nn, y_pred_nn_proba),
    'Тип': 'Нейросеть'
})

# Создаём DataFrame и сортируем по F1-Score (от лучшего к худшему)
comparison_df = pd.DataFrame(final_summary)
comparison_df = comparison_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
comparison_df.index = comparison_df.index + 1 # Нумерация с 1

print('=== ИТОГОВОЕ СРАВНЕНИЕ ВСЕХ МЕТОДОВ (отсортировано по F1-Score) ===')
print(comparison_df.round(4).to_string())

# 7.2 Столбчатая диаграмма для всех методов

# Плавим таблицу для seaborn
plot_df = comparison_df.melt(id_vars=['Метод', 'Тип'],
                             value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                             var_name='Метрика', value_name='Значение')

plt.figure(figsize=(14, 6))
sns.barplot(data=plot_df, x='Метод', y='Значение', hue='Метрика', palette='viridis')
plt.title('Сравнение всех методов классификации (финальные результаты)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.ylim(0.85, 1.02) # Обрезаем ось Y, чтобы лучше видеть разницу
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()