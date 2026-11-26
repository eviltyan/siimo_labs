import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam

from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score,
                             normalized_mutual_info_score,
                             homogeneity_score, completeness_score)
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import TSNE
import umap

import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print('TensorFlow:', tf.__version__)

# Загрузка данных
df = pd.read_csv('glass.csv')
df.columns = df.columns.str.strip()
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

y_true = df['Type'].copy()
X_raw = df.drop(columns=['Type'])

# Масштабирование
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_raw)
print(f'Размерность данных: {X_scaled.shape}')

input_dim = X_scaled.shape[1]

def build_autoencoder(input_dim, latent_dim, l2_reg=0.0):
    """Строит автоэнкодер и энкодер с заданными параметрами."""
    reg = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(16, activation='relu', kernel_regularizer=reg)(inp)
    x = layers.Dense(8, activation='relu', kernel_regularizer=reg)(x)
    latent = layers.Dense(latent_dim, activation='linear', name='latent')(x)
    x = layers.Dense(8, activation='relu')(latent)
    x = layers.Dense(16, activation='relu')(x)
    out = layers.Dense(input_dim, activation='linear')(x)
    ae = models.Model(inp, out)
    enc = models.Model(inp, latent)
    return ae, enc

from itertools import product

param_grid = {
    'latent_dim': [2, 4, 8],
    'lr': [0.001, 0.01],
    'l2_reg': [0.0, 0.01]
}

search_results = []

for latent_dim, lr, l2_reg in product(param_grid['latent_dim'], param_grid['lr'], param_grid['l2_reg']):
    ae, enc = build_autoencoder(input_dim, latent_dim, l2_reg)
    ae.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    ae.fit(X_scaled, X_scaled, epochs=100, batch_size=16, verbose=0,
           callbacks=[callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)])
    
    X_lat = enc.predict(X_scaled, verbose=0)
    km = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE)
    lbl = km.fit_predict(X_lat)
    
    search_results.append({
        'latent_dim': latent_dim, 'lr': lr, 'l2_reg': l2_reg,
        'Silhouette': silhouette_score(X_lat, lbl),
        'Davies-Bouldin': davies_bouldin_score(X_lat, lbl),
        'Calinski-Harabasz': calinski_harabasz_score(X_lat, lbl),
        'ARI': adjusted_rand_score(y_true, lbl)
    })

search_df = pd.DataFrame(search_results).sort_values('Silhouette', ascending=False)
print("Результаты подбора гиперпараметров (топ-10):")
display(search_df.head(10).round(4))

# Выбираем лучшую конфигурацию по Silhouette
best_config = search_df.iloc[0]
BEST_LATENT = int(best_config['latent_dim'])
BEST_LR = float(best_config['lr'])
BEST_L2 = float(best_config['l2_reg'])

print(f"ЛУЧШАЯ КОНФИГУРАЦИЯ:")
print(f"latent_dim = {BEST_LATENT}")
print(f"lr         = {BEST_LR}")
print(f"l2_reg     = {BEST_L2}")
print(f"Silhouette = {best_config['Silhouette']:.4f}")

# Финальная модель с лучшими гиперпараметрами (зафиксировано вручную)
BEST_LATENT = 2
BEST_LR = 0.001
BEST_L2 = 0.0

autoencoder, encoder = build_autoencoder(input_dim, BEST_LATENT, BEST_L2)
autoencoder.compile(optimizer=Adam(learning_rate=BEST_LR), loss='mse')

early_stop = callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)

history = autoencoder.fit(
    X_scaled, X_scaled,
    epochs=200, batch_size=16, shuffle=True, verbose=0,
    callbacks=[early_stop]
)

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Reconstruction Loss')
plt.title('Обучение автокодировщика (лучшая конфигурация)')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()

X_latent = encoder.predict(X_scaled, verbose=0)

from sklearn.metrics import homogeneity_score, completeness_score

kmeans_latent = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE)
labels_deep = kmeans_latent.fit_predict(X_latent)

# Функция для расчёта всех метрик
def get_metrics_full(X, labels, y_true, name):
    return {
        'Метод': name,
        'Silhouette': silhouette_score(X, labels),
        'Davies-Bouldin': davies_bouldin_score(X, labels),
        'Calinski-Harabasz': calinski_harabasz_score(X, labels),
        'ARI': adjusted_rand_score(y_true, labels),
        'NMI': normalized_mutual_info_score(y_true, labels),
        'Homogeneity': homogeneity_score(y_true, labels),
        'Completeness': completeness_score(y_true, labels)
    }
# K-means на исходных данных
kmeans_raw = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE)
labels_raw = kmeans_raw.fit_predict(X_scaled)

# DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)

# GMM (EM-алгоритм)
gmm = GaussianMixture(n_components=4, random_state=RANDOM_STATE)
labels_gmm = gmm.fit_predict(X_scaled)

results_comparison = pd.DataFrame([
    get_metrics_full(X_scaled, labels_raw, y_true, 'K-means (исходные)'),
    get_metrics_full(X_latent, labels_deep, y_true, 'DEC (латентные)'),
    get_metrics_full(X_scaled, labels_dbscan, y_true, 'DBSCAN'),
    get_metrics_full(X_scaled, labels_gmm, y_true, 'GMM (EM)')
])

print("Расширенное сравнение методов:")
display(results_comparison.round(4))

# t-SNE
tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30)
X_tsne = tsne.fit_transform(X_latent)

# UMAP
reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
X_umap = reducer.fit_transform(X_latent)

# Графики
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# t-SNE
axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_deep, cmap='Set1', 
                alpha=0.8, edgecolor='k', linewidth=0.5, s=50)
axes[0].set_title('t-SNE латентного пространства (DEC)')
axes[0].grid(True)

# UMAP
axes[1].scatter(X_umap[:, 0], X_umap[:, 1], c=labels_deep, cmap='Set1', 
                alpha=0.8, edgecolor='k', linewidth=0.5, s=50)
axes[1].set_title('UMAP латентного пространства (DEC)')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Профили кластеров в исходных признаках
cluster_profiles = []
for i in range(4):
    cluster_data = X_raw[labels_deep == i]
    cluster_profiles.append(cluster_data.mean())

profiles_df = pd.DataFrame(cluster_profiles, columns=X_raw.columns)
profiles_df['Размер'] = pd.Series(labels_deep).value_counts().sort_index()
print("Профили кластеров (средние значения в исходных единицах):")
display(profiles_df.round(2))

# Кросс-табуляция с реальными типами стекла
print("\nСоответствие кластеров и реальных типов стекла:")
crosstab = pd.crosstab(labels_deep, y_true)
crosstab.columns = ['Тип 1', 'Тип 2', 'Тип 3', 'Тип 5', 'Тип 6', 'Тип 7']
display(crosstab)

# Оценка устойчивости разбиения
stability_aris = []
for seed in range(10):
    km_tmp = KMeans(n_clusters=4, n_init=10, random_state=seed)
    lbl_tmp = km_tmp.fit_predict(X_latent)
    stability_aris.append(adjusted_rand_score(labels_deep, lbl_tmp))

print(f"\nУстойчивость DEC (K-means в латентном пространстве):")
print(f"Средний ARI между запусками: {np.mean(stability_aris):.4f}")
print(f"Стандартное отклонение: {np.std(stability_aris):.4f}")
print(f"Минимум: {np.min(stability_aris):.4f}, Максимум: {np.max(stability_aris):.4f}")