import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.seasonal import STL
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 5)

# Загрузка данных
df_ts = pd.read_csv('AirPassengers.csv', index_col=False)

# Берем только первые две колонки и переименовываем для удобства
df_ts = df_ts.iloc[:, :2]
df_ts.columns = ['Date', 'Passengers']

# Приведение к datetime и сортировка
df_ts['Date'] = pd.to_datetime(df_ts['Date'])
df_ts = df_ts.sort_values('Date').reset_index(drop=True)

# Визуализация
plt.figure(figsize=(14, 6))
plt.plot(df_ts['Date'], df_ts['Passengers'], marker='o', linestyle='-', color='b')
plt.title('Динамика количества авиапассажиров (Air Passengers)')
plt.xlabel('Дата')
plt.ylabel('Количество пассажиров (тыс.)')
plt.grid(True)
plt.show()

print('Первые 5 строк:')
display(df_ts.head())

# Создаем копию для добавления признаков
df_feat = df_ts.copy()

# 1. Календарные признаки
df_feat['Year'] = df_feat['Date'].dt.year
df_feat['Month'] = df_feat['Date'].dt.month
df_feat['Quarter'] = df_feat['Date'].dt.quarter

# 2. Лаги (сдвиги в прошлое)
for lag in [1, 2, 3, 12]:
    df_feat[f'Lag_{lag}'] = df_feat['Passengers'].shift(lag)

# 3. Скользящие статистики (окно 3 месяца)
df_feat['Rolling_Mean_3'] = df_feat['Passengers'].shift(1).rolling(window=3).mean()
df_feat['Rolling_Median_3'] = df_feat['Passengers'].shift(1).rolling(window=3).median()
df_feat['Rolling_Std_3'] = df_feat['Passengers'].shift(1).rolling(window=3).std()
df_feat['Rolling_Min_3'] = df_feat['Passengers'].shift(1).rolling(window=3).min()
df_feat['Rolling_Max_3'] = df_feat['Passengers'].shift(1).rolling(window=3).max()

# Удаляем строки с NaN (первые 12 месяцев не будут иметь лагов)
df_feat = df_feat.dropna().reset_index(drop=True)

print(f'Размер датасета после генерации признаков: {df_feat.shape}')
display(df_feat.head())

# Разделение на train/test (80% на обучение, 20% на тест)
# Без перемешивания (shuffle=False)
train_size = int(len(df_feat) * 0.8)
train, test = df_feat.iloc[:train_size], df_feat.iloc[train_size:]

features = ['Year', 'Month', 'Quarter', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_12',
            'Rolling_Mean_3', 'Rolling_Median_3', 'Rolling_Std_3',
            'Rolling_Min_3', 'Rolling_Max_3']
X_train, y_train = train[features], train['Passengers']
X_test, y_test = test[features], test['Passengers']

model_ts = Ridge(alpha=1.0)

print("Кросс-валидация (TimeSeriesSplit):")
tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_index, val_index) in enumerate(tscv.split(X_train)):
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

    model_ts.fit(X_tr, y_tr)
    preds = model_ts.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f'  Фолд {fold+1}: RMSE = {rmse:.2f}')

# Обучение на всей тренировочной выборке
model_ts.fit(X_train, y_train)
y_pred = model_ts.predict(X_test)

# Рекурсивный прогноз на H шагов
H = 12
recursive_preds = []
current_input = X_test.iloc[0:1].copy()

for step in range(H):
    pred = model_ts.predict(current_input)[0]
    recursive_preds.append(pred)

    if step < H - 1:
        next_input = X_test.iloc[step+1:step+2].copy()
        next_input['Lag_1'] = pred  # Используем предсказание как лаг
        current_input = next_input

y_true_h = y_test.iloc[:H].values

plt.figure(figsize=(12, 5))
plt.plot(range(H), y_true_h, marker='o', label='Реальные значения', color='black')
plt.plot(range(H), recursive_preds, marker='x', label='Рекурсивный прогноз', color='red', linestyle='--')
plt.title(f'Рекурсивный прогноз на {H} шагов вперед')
plt.xlabel('Шаг прогноза (месяцы)')
plt.ylabel('Количество пассажиров')
plt.legend()
plt.grid(True)
plt.show()

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
wape = (np.sum(np.abs(y_test - y_pred)) / np.sum(np.abs(y_test))) * 100

print('Метрики качества на тестовой выборке (прогноз на 1 шаг):')
print(f'  MSE:  {mse:.2f}')
print(f'  RMSE: {rmse:.2f}')
print(f'  MAE:  {mae:.2f}')
print(f'  MAPE: {mape:.2f}%')
print(f'  WAPE: {wape:.2f}%')

# 1. Метрики на тренировочной выборке
y_train_pred = model_ts.predict(X_train)
mape_train = mean_absolute_percentage_error(y_train, y_train_pred) * 100
wape_train = (np.sum(np.abs(y_train - y_train_pred)) / np.sum(np.abs(y_train))) * 100
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

# Сводная таблица метрик
metrics_df = pd.DataFrame({
    'Выборка': ['Train', 'Test'],
    'RMSE': [rmse_train, rmse],
    'MAPE (%)': [mape_train, mape],
    'WAPE (%)': [wape_train, wape]
})
print('Сравнение метрик на Train и Test:')
display(metrics_df.round(2))

# 2. Анализ ошибок по месяцам (на тестовой выборке)
errors = np.abs(y_test.values - y_pred)
test_analysis = test[['Date', 'Month', 'Passengers']].copy()
test_analysis['Predicted'] = y_pred
test_analysis['Absolute_Error'] = errors

# Находим ТОП-5 месяцев, где модель ошиблась сильнее всего
top_errors = test_analysis.sort_values(by='Absolute_Error', ascending=False).head(5)
print('\nТОП-5 месяцев с наибольшей абсолютной ошибкой на тесте:')
display(top_errors)

# 3. Визуализация ошибок по месяцам
plt.figure(figsize=(14, 5))
plt.plot(test_analysis['Date'], test_analysis['Absolute_Error'], marker='o', color='purple')
plt.title('Абсолютная ошибка прогноза по месяцам (тестовая выборка)')
plt.xlabel('Дата')
plt.ylabel('Абсолютная ошибка (тыс. пассажиров)')
plt.grid(True)
plt.show()

stl = STL(df_ts.set_index('Date')['Passengers'], period=12, robust=True)
res = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
res.observed.plot(ax=axes[0], title='Исходный ряд')
res.trend.plot(ax=axes[1], title='Тренд')
res.seasonal.plot(ax=axes[2], title='Сезонность')
res.resid.plot(ax=axes[3], title='Остаток (шум)')
plt.tight_layout()
plt.show()

# 1. Создаем копию исходного датасета и сразу добавляем STL-компоненты
df_stl = df_ts.copy()

# Берем значения напрямую из res (они выровнены по индексу с df_ts)
df_stl['STL_Trend'] = res.trend.values
df_stl['STL_Seasonal'] = res.seasonal.values

# 2. Генерируем базовые признаки (как делали раньше)
df_stl['Year'] = df_stl['Date'].dt.year
df_stl['Month'] = df_stl['Date'].dt.month
df_stl['Quarter'] = df_stl['Date'].dt.quarter

for lag in [1, 2, 3, 12]:
    df_stl[f'Lag_{lag}'] = df_stl['Passengers'].shift(lag)

df_stl['Rolling_Mean_3'] = df_stl['Passengers'].shift(1).rolling(window=3).mean()
df_stl['Rolling_Std_3'] = df_stl['Passengers'].shift(1).rolling(window=3).std()

# 3. Удаляем все строки, где есть хоть один NaN (это уберет первые 12 месяцев и края STL)
df_stl = df_stl.dropna().reset_index(drop=True)

# 4. Разделяем на Train/Test
train_size_stl = int(len(df_stl) * 0.8)
train_stl = df_stl.iloc[:train_size_stl]
test_stl = df_stl.iloc[train_size_stl:]

# 5. Определяем признаки (базовые + STL)
features_stl = ['Year', 'Month', 'Quarter', 'Lag_1', 'Lag_2', 'Lag_3', 'Lag_12',
                'Rolling_Mean_3', 'Rolling_Std_3', 'STL_Trend', 'STL_Seasonal']

X_train_stl = train_stl[features_stl]
y_train_stl = train_stl['Passengers']
X_test_stl = test_stl[features_stl]
y_test_stl = test_stl['Passengers']

# 6. Обучаем модель
model_stl = Ridge(alpha=1.0)
model_stl.fit(X_train_stl, y_train_stl)
y_pred_stl = model_stl.predict(X_test_stl)

# 7. Считаем метрики
mape_with_stl = mean_absolute_percentage_error(y_test_stl, y_pred_stl) * 100
rmse_with_stl = np.sqrt(mean_squared_error(y_test_stl, y_pred_stl))

# 8. Сравнительная таблица
comparison_stl = pd.DataFrame({
    'Модель': ['Без STL (базовые признаки)', 'С STL (Тренд + Сезонность)'],
    'RMSE': [rmse, rmse_with_stl],
    'MAPE (%)': [mape, mape_with_stl]
})
print('Сравнение качества модели с добавлением компонент STL:')
display(comparison_stl.round(2))