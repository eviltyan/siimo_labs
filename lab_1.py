import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
sns.set_style('whitegrid')

df = pd.read_csv('boston.csv')
df.columns = df.columns.str.strip()

print('Размер датасета:', df.shape)
print('\nПервые 5 строк:')
print(df.head())
print('\nТипы данных и non-null counts:')
df.info()

# Описательная статистика
print('Описательная статистика числовых признаков:')
display(df.describe().T)

# Проверка пропусков
print('\nПропуски по столбцам:')
print(df.isna().sum())

# Уникальные значения категориальных признаков
for col in df.select_dtypes(include='object').columns:
    print(f'\n{col}: {df[col].nunique()} уникальных значений')
    print(df[col].value_counts().head())

# Визуализация распределения только целевой переменной
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

sns.histplot(df['MEDV'], bins=30, kde=True, ax=axes[0], color='steelblue')
axes[0].set_title('Распределение целевой переменной MEDV')
axes[0].set_xlabel('MEDV (тыс. $)')
axes[0].set_ylabel('Частота')

sns.boxplot(x=df['MEDV'], ax=axes[1], color='lightgreen')
axes[1].set_title('Boxplot целевой переменной MEDV')
axes[1].set_xlabel('MEDV (тыс. $)')

plt.tight_layout()
plt.show()

# Визуализация распределений всех независимых признаков
df.drop(columns='MEDV').hist(figsize=(16, 12), bins=25, edgecolor='black', color='cadetblue')
plt.suptitle('Распределение независимых признаков датасета Boston Housing', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

# Разделение на признаки (X) и целевую переменную (y)
X = df.drop(columns='MEDV')
y = df['MEDV']

# Разделение на train/test в пропорции 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Инициализация и обучение StandardScaler на тренировочных данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Преобразуем обратно в DataFrame для удобства дальнейшей работы и визуализации
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print(f'Размер тренировочной выборки: {X_train.shape[0]} строк')
print(f'Размер тестовой выборки: {X_test.shape[0]} строк')
print('\nСредние значения признаков после стандартизации (train, должны быть ~0):')
print(X_train_scaled.mean().round(2).to_string())

# Построение полной матрицы корреляций Пирсона
plt.figure(figsize=(16, 14))  # Увеличили размер для лучшей читаемости
corr_matrix = df.corr()

# Строим тепловую карту
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.75}, annot_kws={"size": 9})  # Уменьшили шрифт цифр

plt.title('Полная матрица корреляций признаков датасета Boston Housing', fontsize=16)
plt.tight_layout()
plt.show()

# Вывод корреляций с целевой переменной в порядке убывания модуля (оставляем, это удобно)
target_corr = corr_matrix['MEDV'].drop('MEDV').sort_values(key=abs, ascending=False)
print('Корреляция признаков с целевой переменной MEDV (по убыванию |r|):')
print(target_corr.round(3).to_string())

# Расчет VIF для оценки мультиколлинеарности
X = df.drop(columns='MEDV')

vif_data = pd.DataFrame()
vif_data['Признак'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Сортируем по убыванию VIF
vif_data = vif_data.sort_values(by='VIF', ascending=False).reset_index(drop=True)

print('Коэффициенты инфляции дисперсии (VIF):')
print(vif_data.round(2).to_string(index=False))


# Функция для оценки модели: кросс-валидация, обучение, тест, метрики
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    # 1. Кросс-валидация на тренировочной выборке (5 фолдов)
    # Используем negative RMSE, так как sklearn ожидает максимизацию метрики
    cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                scoring='neg_root_mean_squared_error')
    cv_rmse_mean = -cv_scores.mean()  # Возвращаем положительное значение

    # 2. Обучение модели на всей тренировочной выборке
    model.fit(X_train, y_train)

    # 3. Предсказания на тестовой выборке
    y_pred = model.predict(X_test)

    # 4. Расчет метрик на тесте
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100   # В процентах

    # Вывод результатов
    print(f'--- {model_name} ---')
    print(f'CV RMSE (5-fold, train): {cv_rmse_mean:.4f}')
    print(f'Test RMSE:               {rmse:.4f}')
    print(f'Test R²:                 {r2:.4f}')
    print(f'Test MAPE:               {mape:.2f}%')
    print()

    return {
        'Модель': model_name,
        'CV RMSE (train)': cv_rmse_mean,
        'Test RMSE': rmse,
        'Test R²': r2,
        'Test MAPE (%)': mape
    }


# Словарь для хранения результатов всех моделей
results_baseline = {}

# Инициализация моделей с базовыми параметрами
models = {
    'Linear Regression': LinearRegression(),
    'Lasso (alpha=1.0)': Lasso(alpha=1.0, random_state=42),
    'Ridge (alpha=1.0)': Ridge(alpha=1.0, random_state=42)
}

# Обучаем и оцениваем каждую модель на СТАНДАРТИЗИРОВАННЫХ данных
for name, model in models.items():
    results_baseline[name] = evaluate_model(
        model, X_train_scaled, y_train, X_test_scaled, y_test, name
    )

# Преобразуем результаты в DataFrame для красивого вывода
results_df = pd.DataFrame(results_baseline).T

# Переиндексируем, чтобы строки шли в нужном порядке
results_df = results_df.reset_index(drop=True)

# Округляем значения для красоты
results_df = results_df.round(4)

# Выводим таблицу
print('Сводная таблица метрик качества моделей (до применения PCA):')
display(results_df)

# Создаем фигуру для трех графиков остатков
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes = axes.flatten()

# Инициализируем и обучаем модели заново (они уже были обучены, но нам нужны предсказания)
models = {
    'Linear Regression': LinearRegression(),
    'Lasso (alpha=1.0)': Lasso(alpha=1.0, random_state=42),
    'Ridge (alpha=1.0)': Ridge(alpha=1.0, random_state=42)
}

for i, (name, model) in enumerate(models.items()):
    # Обучаем модель
    model.fit(X_train_scaled, y_train)
    # Получаем предсказания на тестовой выборке
    y_pred = model.predict(X_test_scaled)
    # Считаем остатки
    residuals = y_test - y_pred

    # Строим scatter-plot
    axes[i].scatter(y_pred, residuals, alpha=0.6, color='steelblue', edgecolor='k')
    # Добавляем горизонтальную линию на уровне 0
    axes[i].axhline(y=0, color='red', linestyle='--', linewidth=2)

    axes[i].set_title(f'График остатков: {name}')
    axes[i].set_xlabel('Предсказанные значения MEDV (тыс. $)')
    axes[i].set_ylabel('Остатки (Реальные - Предсказанные)')
    axes[i].grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()

# Инициализируем PCA. Для начала не ограничиваем число компонент, чтобы посмотреть на все
pca = PCA(random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Строим график каменистой осыпи (Scree plot)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         np.cumsum(pca.explained_variance_ratio_),
         marker='o', linestyle='--', color='b')

plt.title('График каменистой осыпи (Scree Plot)')
plt.xlabel('Номер главной компоненты')
plt.ylabel('Накопленная доля объясненной дисперсии')
plt.grid(True, linestyle=':', alpha=0.7)
plt.axhline(y=0.95, color='r', linestyle='-', label='95% объясненной дисперсии')
plt.legend()
plt.show()

# Выведем, сколько компонент нужно для 95% дисперсии
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f'Количество компонент для объяснения 95% дисперсии: {n_components_95}')

# Отбираем нужное количество компонент (то, что выдал предыдущий код, например, 10)
pca_opt = PCA(n_components=n_components_95, random_state=42)
X_train_pca = pca_opt.fit_transform(X_train_scaled)
X_test_pca = pca_opt.transform(X_test_scaled)

# Словарь для результатов на PCA
results_pca = {}

# Те же самые модели
models_pca = {
    'Linear Regression': LinearRegression(),
    'Lasso (alpha=1.0)': Lasso(alpha=1.0, random_state=42),
    'Ridge (alpha=1.0)': Ridge(alpha=1.0, random_state=42)
}

print("=== Оценка моделей на главных компонентах (PCA) ===")
for name, model in models_pca.items():
    results_pca[name] = evaluate_model(
        model, X_train_pca, y_train, X_test_pca, y_test, name
    )

# Объединяем результаты в один датафрейм для сравнения
df_baseline = pd.DataFrame(results_baseline).T.reset_index(drop=True)
df_pca = pd.DataFrame(results_pca).T.reset_index(drop=True)

# Добавляем суффиксы, чтобы отличить модели
df_baseline['Модель'] = df_baseline['Модель'] + ' (исходные)'
df_pca['Модель'] = df_pca['Модель'] + ' (PCA)'

# Объединяем
comparison_df = pd.concat([df_baseline, df_pca], ignore_index=True)

# Округляем и выводим
comparison_df = comparison_df.round(4)

# Для удобства отсортируем по RMSE на тесте
comparison_df = comparison_df.sort_values(by='Test RMSE')

print('Сравнение качества моделей: исходные признаки vs главные компоненты')
display(comparison_df)

# 1. График Test RMSE
plt.figure(figsize=(10, 6))
x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, df_baseline['Test RMSE'], width, label='Исходные признаки', color='steelblue')
plt.bar(x + width/2, df_pca['Test RMSE'], width, label='Главные компоненты (PCA)', color='coral')

plt.title('Сравнение Test RMSE: Исходные данные vs PCA', fontsize=14)
plt.xticks(x, models, rotation=0)
plt.ylabel('RMSE (тыс. $)\n(меньше = лучше)')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Добавляем подписи значений над столбцами
for i in range(len(models)):
    plt.text(x[i] - width/2, df_baseline['Test RMSE'].iloc[i] + 0.05,
             f"{df_baseline['Test RMSE'].iloc[i]:.2f}", ha='center', va='bottom', fontsize=9)
    plt.text(x[i] + width/2, df_pca['Test RMSE'].iloc[i] + 0.05,
             f"{df_pca['Test RMSE'].iloc[i]:.2f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# 2. График Test R²
plt.figure(figsize=(10, 6))

plt.bar(x - width/2, df_baseline['Test R²'], width, label='Исходные признаки', color='steelblue')
plt.bar(x + width/2, df_pca['Test R²'], width, label='Главные компоненты (PCA)', color='coral')

plt.title('Сравнение Test R²: Исходные данные vs PCA', fontsize=14)
plt.xticks(x, models, rotation=0)
plt.ylabel('Коэффициент детерминации R²\n(больше = лучше)')
plt.ylim(0, 0.8)  # Ограничим ось Y, чтобы разница была заметнее
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Добавляем подписи значений
for i in range(len(models)):
    plt.text(x[i] - width/2, df_baseline['Test R²'].iloc[i] + 0.01,
             f"{df_baseline['Test R²'].iloc[i]:.3f}", ha='center', va='bottom', fontsize=9)
    plt.text(x[i] + width/2, df_pca['Test R²'].iloc[i] + 0.01,
             f"{df_pca['Test R²'].iloc[i]:.3f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# 3. График Test MAPE
plt.figure(figsize=(10, 6))

plt.bar(x - width/2, df_baseline['Test MAPE (%)'], width, 
        label='Исходные признаки', color='steelblue')
plt.bar(x + width/2, df_pca['Test MAPE (%)'], width, 
        label='Главные компоненты (PCA)', color='coral')

plt.title('Сравнение Test MAPE: Исходные данные vs PCA', fontsize=14)
plt.xticks(x, models, rotation=0)
plt.ylabel('Средняя абсолютная ошибка в %\n(меньше = лучше)')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Добавляем подписи значений
for i in range(len(models)):
    plt.text(x[i] - width/2, df_baseline['Test MAPE (%)'].iloc[i] + 0.2,
             f"{df_baseline['Test MAPE (%)'].iloc[i]:.2f}%", 
             ha='center', va='bottom', fontsize=9)
    plt.text(x[i] + width/2, df_pca['Test MAPE (%)'].iloc[i] + 0.2,
             f"{df_pca['Test MAPE (%)'].iloc[i]:.2f}%", 
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Построение матрицы корреляций главных компонент
# Берем X_train_pca (это матрица компонент, которую мы получили ранее)
# Преобразуем в DataFrame для удобства
pca_df = pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(X_train_pca.shape[1])])

plt.figure(figsize=(10, 8))
sns.heatmap(pca_df.corr(), annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.75})
plt.title('Матрица корреляций главных компонент (PCA)', fontsize=14)
plt.tight_layout()
plt.show()
