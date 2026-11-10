import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score,
                             adjusted_mutual_info_score)
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import shapiro

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

df = pd.read_csv('glass.csv')
df.columns = df.columns.str.strip()

# В исходном CSV из UCI есть служебный столбец Id — удаляем, если присутствует
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

print('Размер датасета:', df.shape)
print('\nПервые 5 строк:')
print(df.head())
print('\nТипы данных и non-null counts:')
df.info()

y_true = df['Type'].copy()           # только для внешней оценки
X_raw  = df.drop(columns=['Type'])   # матрица признаков для кластеризации

print('Признаки:', X_raw.columns.tolist())
print('Распределение типов стекла (для справки):')
print(y_true.value_counts().sort_index())

print('Пропуски по столбцам:')
print(X_raw.isnull().sum())

print('\nОписательные статистики:')
print(X_raw.describe().T)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, col in enumerate(X_raw.columns):
    sns.histplot(X_raw[col], bins=25, kde=True, ax=axes[i], color='steelblue')
    axes[i].set_title(f'{col}  (skew={X_raw[col].skew():.2f})')
plt.tight_layout()
plt.show()

print(f'{"Признак":<6}{"W-статистика":>15}{"p-value":>12}{"Нормальность":>18}')
for col in X_raw.columns:
    stat, p = shapiro(X_raw[col])
    verdict = 'да' if p > 0.05 else 'нет'
    print(f'{col:<6}{stat:>15.4f}{p:>12.4f}{verdict:>18}')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col in zip(axes, ['RI', 'Na', 'Ba']):
    sns.boxplot(y=X_raw[col], ax=ax)
    ax.set_title(f'Boxplot: {col}')
plt.tight_layout()
plt.show()

# Количественная оценка выбросов по правилу 1.5·IQR
print()
print(f'{"Признак":<6}{"Q1":>8}{"Q3":>8}{"IQR":>8}{"Выбросов":>10}{"Доля, %":>10}')
for col in X_raw.columns:
    q1, q3 = X_raw[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = ((X_raw[col] < low) | (X_raw[col] > high)).sum()
    print(f'{col:<6}{q1:>8.3f}{q3:>8.3f}{iqr:>8.3f}{outliers:>10d}{outliers/len(X_raw)*100:>10.2f}')

plt.figure(figsize=(9, 7))
corr = X_raw.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, square=True)
plt.title('Корреляционная матрица числовых признаков')
plt.show()

# Пары с |r| > 0.5
print()
print('Сильные корреляции (|r| > 0.5):')
c = corr.abs()
for i in range(len(c.columns)):
    for j in range(i + 1, len(c.columns)):
        if c.iloc[i, j] > 0.5:
            print(f'  {c.columns[i]:>3} – {c.columns[j]:<3}: r = {corr.iloc[i, j]:+.3f}')

# Инициализируем скейлеры
scaler_standard = StandardScaler()
scaler_robust = RobustScaler()

# Применяем масштабирование
X_standard = scaler_standard.fit_transform(X_raw)
X_robust = scaler_robust.fit_transform(X_raw)

# Для удобства переведем обратно в DataFrame
X_standard_df = pd.DataFrame(X_standard, columns=X_raw.columns)
X_robust_df = pd.DataFrame(X_robust, columns=X_raw.columns)

print("Сравнение разброса значений после масштабирования (Стандартное отклонение):")
print("StandardScaler:")
print(X_standard_df.std().round(2).to_dict())
print("\nRobustScaler:")
print(X_robust_df.std().round(2).to_dict())

print("\nСравнение максимальных значений (выбросы):")
print(f"StandardScaler max(Ba): {X_standard_df['Ba'].max():.2f}")
print(f"RobustScaler max(Ba): {X_robust_df['Ba'].max():.2f}")

# Группа 1
group1 = ['RI', 'Na', 'Mg']
sns.pairplot(X_robust_df[group1], 
             diag_kind='hist', 
             corner=True, 
             plot_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'k', 'linewidth': 0.3},
             diag_kws={'bins': 15, 'edgecolor': 'black'})
plt.suptitle('Группа 1: RI, Na, Mg (Оптические свойства и легкие металлы)', y=1.02, fontsize=14)
plt.show()

# Группа 2
group2 = ['Al', 'Si', 'K']
sns.pairplot(X_robust_df[group2], 
             diag_kind='hist', 
             corner=True, 
             plot_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'k', 'linewidth': 0.3},
             diag_kws={'bins': 15, 'edgecolor': 'black'})
plt.suptitle('Группа 2: Al, Si, K (Основные компоненты стекла)', y=1.02, fontsize=14)
plt.show()

# Группа 3
group3 = ['Ca', 'Ba', 'Fe']
sns.pairplot(X_robust_df[group3], 
             diag_kind='hist', 
             corner=True, 
             plot_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'k', 'linewidth': 0.3},
             diag_kws={'bins': 15, 'edgecolor': 'black'})
plt.suptitle('Группа 3: Ca, Ba, Fe (Тяжелые металлы и примеси)', y=1.02, fontsize=14)
plt.show()

# Диапазон k от 2 до 10
k_range = range(2, 11)
inertias = []
silhouettes = []

for k in k_range:
    # n_init=10 для избежания локальных минимумов, как указано в методичке
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_robust)
    
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_robust, labels))

# Строим графики
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Метод локтя
axes[0].plot(k_range, inertias, 'bo-', markerfacecolor='red')
axes[0].set_title('Метод локтя (Elbow Method)')
axes[0].set_xlabel('Количество кластеров (k)')
axes[0].set_ylabel('Инерция (Inertia)')
axes[0].grid(True)

# Силуэтный анализ
axes[1].plot(k_range, silhouettes, 'go-', markerfacecolor='orange')
axes[1].set_title('Силуэтный анализ (Silhouette Score)')
axes[1].set_xlabel('Количество кластеров (k)')
axes[1].set_ylabel('Силуэт')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Выводим таблицу с метриками для наглядности
print()
print(f'{"k":<4}{"Inertia":>15}{"Silhouette":>15}')
for k, inertia, sil in zip(k_range, inertias, silhouettes):
    print(f'{k:<4}{inertia:>15.2f}{sil:>15.4f}')

# Строим дендрограмму
plt.figure(figsize=(15, 6))
# Используем метод Ward, который минимизирует внутрикластерную дисперсию
linkage_matrix = linkage(X_robust, method='ward')
dendrogram(linkage_matrix, truncate_mode='lastp', p=30, 
           leaf_rotation=90, leaf_font_size=10, show_contracted=True)
plt.title('Дендрограмма (метод Ward)')
plt.xlabel('Индекс объекта (или размер кластера)')
plt.ylabel('Евклидово расстояние')
plt.axhline(y=12, color='r', linestyle='--', label='Потенциальный порог (k=6)')
plt.legend()
plt.show()

# 1. K-means (k=6)
kmeans = KMeans(n_clusters=6, n_init=10, random_state=42)
labels_km = kmeans.fit_predict(X_robust)

# 2. Иерархическая кластеризация (k=6, Ward)
agg = AgglomerativeClustering(n_clusters=6, linkage='ward')
labels_agg = agg.fit_predict(X_robust)

# Собираем метрики в таблицу
def get_metrics(X, labels, y_true, name):
    return {
        'Метод': name,
        'Silhouette': silhouette_score(X, labels),
        'Calinski-Harabasz': calinski_harabasz_score(X, labels),
        'Davies-Bouldin': davies_bouldin_score(X, labels),
        'ARI (Rand)': adjusted_rand_score(y_true, labels),
        'AMI': adjusted_mutual_info_score(y_true, labels)
    }

results = pd.DataFrame([
    get_metrics(X_robust, labels_km, y_true, 'K-means (k=6)'),
    get_metrics(X_robust, labels_agg, y_true, 'Иерархическая (Ward)')
])

print("Сравнительная таблица метрик:")
display(results.round(4))

k_range = range(2, 11)
metrics_k = {'k': [], 'Silhouette': [], 'Calinski-Harabasz': [], 'Davies-Bouldin': []}

for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_robust)
    metrics_k['k'].append(k)
    metrics_k['Silhouette'].append(silhouette_score(X_robust, labels))
    metrics_k['Calinski-Harabasz'].append(calinski_harabasz_score(X_robust, labels))
    metrics_k['Davies-Bouldin'].append(davies_bouldin_score(X_robust, labels))

df_metrics = pd.DataFrame(metrics_k)

# Строим графики влияния k
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(df_metrics['k'], df_metrics['Silhouette'], 'bo-')
axes[0].set_title('Silhouette Score vs k')
axes[0].set_xlabel('k')
axes[0].set_ylabel('Silhouette')
axes[0].grid(True)

axes[1].plot(df_metrics['k'], df_metrics['Calinski-Harabasz'], 'ro-')
axes[1].set_title('Calinski-Harabasz vs k')
axes[1].set_xlabel('k')
axes[1].set_ylabel('CH Index')
axes[1].grid(True)

axes[2].plot(df_metrics['k'], df_metrics['Davies-Bouldin'], 'go-')
axes[2].set_title('Davies-Bouldin vs k')
axes[2].set_xlabel('k')
axes[2].set_ylabel('DB Index')
axes[2].grid(True)

plt.tight_layout()
plt.show()

# Проверим метрики при k=4 (оптимум по силуэту)
kmeans_4 = KMeans(n_clusters=4, n_init=10, random_state=42)
labels_km_4 = kmeans_4.fit_predict(X_robust)

agg_4 = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels_agg_4 = agg_4.fit_predict(X_robust)

results_k4 = pd.DataFrame([
    get_metrics(X_robust, labels_km_4, y_true, 'K-means (k=4)'),
    get_metrics(X_robust, labels_agg_4, y_true, 'Иерархическая (k=4)')
])

print("Метрики при k=4 (оптимум по силуэту):")
display(results_k4.round(4))

from sklearn.mixture import GaussianMixture

# 1. Для 2D-графика используем PCA на 2 компоненты
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_robust)

# 2. Обучаем EM-алгоритм (GMM)
gmm = GaussianMixture(n_components=6, random_state=42)
labels_gmm = gmm.fit_predict(X_robust)

# 3. Переносим центры K-means и GMM в 2D-пространство
centers_km_2d = pca_2d.transform(kmeans.cluster_centers_)
centers_gmm_2d = pca_2d.transform(gmm.means_)

# 4. Строим красивые 2D-графики с центрами
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- K-means ---
axes[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=labels_km, cmap='tab10', alpha=0.7, edgecolor='k', linewidth=0.3)
axes[0].scatter(centers_km_2d[:, 0], centers_km_2d[:, 1], c='black', marker='X', s=200, label='Центры')
axes[0].set_title('K-means (k=6)')
axes[0].set_xlabel('Главная компонента 1 (PC1)')
axes[0].set_ylabel('Главная компонента 2 (PC2)')
axes[0].legend()
axes[0].grid(True)

# --- GMM (EM) ---
axes[1].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=labels_gmm, cmap='tab10', alpha=0.7, edgecolor='k', linewidth=0.3)
axes[1].scatter(centers_gmm_2d[:, 0], centers_gmm_2d[:, 1], c='black', marker='X', s=200, label='Центры')
axes[1].set_title('EM-алгоритм (GMM, k=6)')
axes[1].set_xlabel('Главная компонента 1 (PC1)')
axes[1].set_ylabel('Главная компонента 2 (PC2)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Вывод дисперсии
print()
print(f"Дисперсия PC1: {pca_3d.explained_variance_ratio_[0]:.2%}")
print(f"Дисперсия PC2: {pca_3d.explained_variance_ratio_[1]:.2%}")
print(f"Суммарно три компоненты объясняют: {sum(pca_2d.explained_variance_ratio_[:2]):.2%} данных")

# 5. Строим 3D-график для K-means
pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X_robust)
centers_km_3d = pca_3d.transform(kmeans.cluster_centers_)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=labels_km, cmap='tab10', alpha=0.6, edgecolor='k', linewidth=0.2, s=40)
ax.scatter(centers_km_3d[:, 0], centers_km_3d[:, 1], centers_km_3d[:, 2], c='black', marker='X', s=200, label='Центры')
ax.set_title('3D визуализация K-means (PC1, PC2, PC3)')
ax.set_xlabel('PC1 (38.5%)')
ax.set_ylabel('PC2 (18.6%)')
ax.set_zlabel('PC3 (17.0%)')
ax.legend()
plt.show()

# Вывод дисперсии
print()
print(f"Дисперсия PC1: {pca_3d.explained_variance_ratio_[0]:.2%}")
print(f"Дисперсия PC2: {pca_3d.explained_variance_ratio_[1]:.2%}")
print(f"Дисперсия PC3: {pca_3d.explained_variance_ratio_[2]:.2%}")
print(f"Суммарно три компоненты объясняют: {sum(pca_3d.explained_variance_ratio_[:3]):.2%} данных")

# Сделаем точки более прозрачными, чтобы увидеть центр розового кластера

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Делаем точки более прозрачными (alpha=0.4), чтобы центры просвечивали
ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], 
           c=labels_km, cmap='tab10', alpha=0.4, edgecolor='none', s=30, zorder=1)

# Центры: огромный размер (s=400), белая обводка (edgecolor='white', linewidth=2) и черная заливка
ax.scatter(centers_km_3d[:, 0], centers_km_3d[:, 1], centers_km_3d[:, 2], 
           c='black', marker='X', s=400, edgecolor='white', linewidth=2, 
           label='Центры', zorder=10)

ax.set_title('3D визуализация K-means (PC1, PC2, PC3)')
ax.set_xlabel('PC1 (38.5%)')
ax.set_ylabel('PC2 (18.6%)')
ax.set_zlabel('PC3 (17.0%)')
ax.legend()
plt.show()

import plotly.express as px

# Создаем DataFrame для Plotly
df_3d = pd.DataFrame({
    'PC1': X_pca_3d[:, 0],
    'PC2': X_pca_3d[:, 1],
    'PC3': X_pca_3d[:, 2],
    'Кластер': labels_km.astype(str)
})

# Рисуем интерактивные точки
fig = px.scatter_3d(df_3d, x='PC1', y='PC2', z='PC3', color='Кластер',
                    opacity=0.7, title='3D визуализация K-means (интерактивная)')

# Добавляем центры поверх точек
fig.add_scatter3d(
    x=centers_km_3d[:, 0], y=centers_km_3d[:, 1], z=centers_km_3d[:, 2],
    mode='markers',
    marker=dict(size=10, color='black', symbol='x', line=dict(width=2, color='white')),
    name='Центры'
)

fig.show()

# Возвращаем центры кластеров к исходному масштабу (в проценты химического состава)
centers_km = scaler_robust.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(centers_km, columns=X_raw.columns)

# Добавляем информацию о размере кластера
centers_df['Размер кластера'] = pd.Series(labels_km).value_counts().sort_index()

print("Центры кластеров K-means (в исходных единицах, вес. %):")
display(centers_df.round(2))

# Для удобства интерпретации посмотрим, какие реальные типы стекла попали в каждый кластер
print("\nСоответствие кластеров и реальных типов стекла (K-means):")
crosstab = pd.crosstab(labels_km, y_true)
crosstab.columns = ['Тип 1', 'Тип 2', 'Тип 3', 'Тип 5', 'Тип 6', 'Тип 7']
display(crosstab)

# Считаем, насколько сильно различаются разбиения, полученные K-means и Иерархической кластеризацией
ari_between_methods = adjusted_rand_score(labels_km, labels_agg)
print(f"ARI между K-means и Иерархической кластеризацией: {ari_between_methods:.4f}")

# Выводим таблицу кросс-табуляции для иерархической кластеризации, чтобы сравнить
print("\nСоответствие кластеров и реальных типов стекла (Ward):")
crosstab_agg = pd.crosstab(labels_agg, y_true)
crosstab_agg.columns = ['Тип 1', 'Тип 2', 'Тип 3', 'Тип 5', 'Тип 6', 'Тип 7']
display(crosstab_agg)