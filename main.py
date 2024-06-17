import numpy as np
import scipy.stats as stats

def linear_regression_ols(X, y):

    # Добавляем столбец единиц к матрице признаков для учета свободного члена
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    # Применяем формулу OLS: (X^T * X)^(-1) * X^T * y
    X_transpose = np.transpose(X)
    beta_hat = np.linalg.inv(X_transpose @ X) @ X_transpose @ y

    return beta_hat, X

def f_statistic(y, y_pred, p):
    n = len(y)
    SSR = np.sum((y_pred - np.mean(y)) ** 2)
    SSE = np.sum((y - y_pred) ** 2)
    MST = SSR / p
    MSE = SSE / (n - p - 1)
    F = MST / MSE
    return F

def t_statistics(X, y, beta_hat, s2):

    # Вычисляем стандартные ошибки для каждого коэффициента
    se = np.sqrt(np.diagonal(s2 * np.linalg.inv(X.T @ X)))

    # Вычисляем t-статистики для каждого коэффициента
    t_stats = beta_hat / se

    return t_stats, se

def r_squared(y, y_pred):

    SS_res = np.sum((y - y_pred) ** 2)
    SS_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (SS_res / SS_tot)

# Вектор признаков x (независимая переменная)
#x = np.array([270925.21, 294609.72, 309998.17, 334013.79, 360779.24, 362345.24, 395690.91, 419712.24, 460202.21, 552728.30]) #с учетом зп сфо
x = np.array([10.26,	10.39,	10.47,	10.58,	10.70,	9.58,	9.58,	9.49,	9.41,	9.66])

# Вектор целевых значений (зависимая переменная)
#y = np.array([3058, 3104, 3162, 3264, 3250, 2753, 2783, 2341, 2624, 2304]) #количество образовательных мигрантов
y = np.array([11473,	12203,	12582,	12960,	14847,	11852,	11189,	8697,	9341,	8532])


# Формируем матрицу признаков X для парной регрессии
X = x.reshape(-1, 1)

# Вычисление оценок параметров
beta_hat, X_with_intercept = linear_regression_ols(X, y)

# Предсказанные значения y
y_pred = X_with_intercept @ beta_hat

# Число независимых переменных
p = X_with_intercept.shape[1] - 1

# Вычисление F-статистики
F = f_statistic(y, y_pred, p)
print("F-статистика:", F)

# Критическое значение F-распределения
alpha = 0.05
dfn = p  # степени свободы числителя
dfd = len(y) - p - 1  # степени свободы знаменателя
F_critical = stats.f.ppf(1 - alpha, dfn, dfd)
print("Критическое значение F:", F_critical)

# Проверка значимости модели
if F > F_critical:
    print("Модель значима.")
else:
    print("Модель не значима.")

# Вычисление остатков и дисперсии остатков
residuals = y - y_pred
s2 = np.sum(residuals ** 2) / (len(y) - X_with_intercept.shape[1])

# Вычисление t-статистик и стандартных ошибок
t_stats, se = t_statistics(X_with_intercept, y, beta_hat, s2)
print("t-статистики для каждого параметра:", t_stats)
print("Стандартные ошибки коэффициентов:", se)

# Критическое значение t-распределения
t_critical = stats.t.ppf(1 - alpha / 2, len(y) - X_with_intercept.shape[1])
print("Критическое значение t:", t_critical)

# Проверка значимости каждого параметра
for i, t in enumerate(t_stats):
    if abs(t) > t_critical:
        print(f"Параметр {i} значим.")
    else:
        print(f"Параметр {i} не значим.")

# Вычисление коэффициента детерминации (R²)
R2 = r_squared(y, y_pred)
print(f"Коэффициент детерминации (r²): {R2}")