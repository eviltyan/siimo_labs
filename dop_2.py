import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import tensorflow as tf
from tensorflow.keras import layers, models as keras_models, callbacks

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

pd.set_option('display.max_columns', 100)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)

print('TensorFlow version:', tf.__version__)

# --- Загрузка данных ---
df = pd.read_csv('chronic_kidney_disease.csv')
df.columns = df.columns.str.strip()

print('Размер датасета:', df.shape)

# Предобработка данных

df_clean = df.copy()

# 1. Очистка целевой переменной
df_clean['classification'] = df_clean['classification'].astype(str).str.strip()

# 2. Очистка категориальных признаков
cat_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
for col in cat_cols:
    df_clean[col] = (df_clean[col]
                     .astype(str)
                     .str.strip()
                     .replace({r'\t\?': np.nan, r'\?': np.nan, '?': np.nan,
                               '[null]': np.nan, 'nan': np.nan, '': np.nan}))

# 3. Приведение числовых к float
num_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc',
            'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
for col in num_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# 4. Удаление id
df_clean = df_clean.drop(columns=['id'])

# 5. Разделение на X и y
y = df_clean['classification']
X = df_clean.drop(columns=['classification'])

# 6. Импутация
num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

X[num_cols] = SimpleImputer(strategy='median').fit_transform(X[num_cols])
X[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(X[cat_cols])

# 7. One-Hot Encoding
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=int)

# 8. Train/Test Split (stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 9. Масштабирование
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols]  = scaler.transform(X_test[num_cols])

# 10. Для нейросети — numpy float32
X_train_nn = X_train_scaled.values.astype('float32')
X_test_nn  = X_test_scaled.values.astype('float32')
y_train_nn = (y_train == 'ckd').astype(int).values
y_test_nn  = (y_test == 'ckd').astype(int).values

print('Train:', X_train_nn.shape, 'Test:', X_test_nn.shape)
print('Пропусков:', pd.DataFrame(X_train_nn).isna().sum().sum() + pd.DataFrame(X_test_nn).isna().sum().sum())

# Baseline: Логистическая регрессия

baseline = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
baseline.fit(X_train_scaled, y_train)
baseline_pred = baseline.predict(X_test_scaled)
baseline_acc = accuracy_score(y_test, baseline_pred)

print(f"Baseline (Logistic Regression) Accuracy: {baseline_acc:.4f}")

  # Функция для создания модели

  def build_model(hidden_layers=[64, 32], activation='relu', dropout_rate=0.2, learning_rate=0.001):
      model = keras_models.Sequential()

      # Первый слой + input_shape
      model.add(layers.Dense(hidden_layers[0], activation=activation, input_shape=(X_train_nn.shape[1],)))
      if dropout_rate > 0:
          model.add(layers.Dropout(dropout_rate))

      # Остальные скрытые слои
      for units in hidden_layers[1:]:
          model.add(layers.Dense(units, activation=activation))
          if dropout_rate > 0:
              model.add(layers.Dropout(dropout_rate))

      # Выходной слой
      model.add(layers.Dense(1, activation='sigmoid'))

      model.compile(
          optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
          loss='binary_crossentropy',
          metrics=['accuracy']
      )
      return model

  print('Модель определена.')

# Функция для кросс-валидации

def evaluate_nn_cv(hidden_layers, learning_rate, dropout_rate, epochs=50, batch_size=32):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    accuracies = []
    losses = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_nn, y_train_nn)):
        X_fold_train, X_fold_val = X_train_nn[train_idx], X_train_nn[val_idx]
        y_fold_train, y_fold_val = y_train_nn[train_idx], y_train_nn[val_idx]

        tf.random.set_seed(RANDOM_STATE)
        model = build_model(hidden_layers=hidden_layers, learning_rate=learning_rate, dropout_rate=dropout_rate)

        # Логирование для TensorBoard
        log_dir = f"logs/cv/lr{learning_rate}_layers{len(hidden_layers)}_fold{fold}_" + datetime.datetime.now().strftime("%H%M%S")
        tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

        model.fit(
            X_fold_train, y_fold_train,
            validation_data=(X_fold_val, y_fold_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[tb_cb],
            verbose=0
        )

        val_loss, val_acc = model.evaluate(X_fold_val, y_fold_val, verbose=0)
        accuracies.append(val_acc)
        losses.append(val_loss)
        print(f"    Фолд {fold+1}: acc={val_acc:.4f}, loss={val_loss:.4f}")

        del model
        tf.keras.backend.clear_session()

    return np.mean(accuracies), np.std(accuracies), np.mean(losses)

# Grid Search

learning_rates = [0.001, 0.01, 0.1]
architectures = [
    ([64, 32], 0.2),
    ([128, 64, 32], 0.3),
    ([32], 0.0)
]

results_grid = []
start = time.time()

for lr in learning_rates:
    for hidden_layers, dropout in architectures:
        print(f"\n=== lr={lr}, layers={hidden_layers}, dropout={dropout} ===")
        mean_acc, std_acc, mean_loss = evaluate_nn_cv(hidden_layers, lr, dropout)
        results_grid.append({
            'learning_rate': lr,
            'architecture': str(hidden_layers),
            'dropout': dropout,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'mean_loss': mean_loss
        })
        print(f"  Средняя Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

print(f"\nОбщее время Grid Search: {time.time() - start:.1f} сек.")

# Сводная таблица Grid Search

grid_df = pd.DataFrame(results_grid)
grid_df = grid_df.sort_values('mean_accuracy', ascending=False).reset_index(drop=True)

print("=== Сводная таблица Grid Search (сортировка по Accuracy) ===")
print(grid_df.round(4).to_string(index=False))

best = grid_df.iloc[0]
print(f"\nЛучшая конфигурация:")
print(f"  LR: {best['learning_rate']}, Arch: {best['architecture']}, Dropout: {best['dropout']}")
print(f"  Mean Acc: {best['mean_accuracy']:.4f} ± {best['std_accuracy']:.4f}")

# Финальная оценка лучшей модели на тесте

best_layers = eval(best['architecture'])
best_dropout = best['dropout']
best_lr = best['learning_rate']

tf.random.set_seed(RANDOM_STATE)
final_model = build_model(hidden_layers=best_layers, learning_rate=best_lr, dropout_rate=best_dropout)
final_model.fit(X_train_nn, y_train_nn, validation_split=0.2, epochs=50, batch_size=32, verbose=0)

y_pred_final = (final_model.predict(X_test_nn).ravel() > 0.5).astype(int)
final_acc = accuracy_score(y_test_nn, y_pred_final)

print("=== Финальная оценка ===")
print(f"Baseline (Logistic Regression): {baseline_acc:.4f}")
print(f"Лучшая нейросеть (CV + Grid):  {final_acc:.4f}")
print(f"Прирост:                        {final_acc - baseline_acc:+.4f}")

print("Для просмотра TensorBoard выполните в терминале: tensorboard --logdir logs/cv")