
# Практична робота: Створення KNN-регресора у Python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# 1. Генеруємо випадкові дані
np.random.seed(42)
X = np.sort(np.random.rand(1000, 1) * 10, axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, 1000)

# 2. Нормалізація
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3. Розподіл на тренувальну і тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 4. Підбір найкращого K
k_range = range(1, 21)
mse_list = []

for k in k_range:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_list.append(mse)

best_k = k_range[np.argmin(mse_list)]
print(f"Найкраще значення K: {best_k}, MSE: {min(mse_list):.4f}")

# 5. Візуалізація MSE залежно від K
plt.plot(k_range, mse_list, marker='o')
plt.xlabel('K')
plt.ylabel('Середньоквадратична помилка (MSE)')
plt.title('MSE залежно від кількості сусідів (K)')
plt.grid(True)
plt.savefig("plot1_mse_vs_k.png")

# 6. Побудова регресійної кривої
model = KNeighborsRegressor(n_neighbors=best_k)
model.fit(X_train, y_train)

X_line = np.linspace(0, 1, 500).reshape(-1, 1)
y_line = model.predict(X_line)

plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='lightgray', label='Навчальні дані', alpha=0.5)
plt.plot(X_line, y_line, color='red', label=f'KNN-регресія (K={best_k})', linewidth=2)
plt.xlabel('X (нормалізований)')
plt.ylabel('y')
plt.title('KNN-регресія з найкращим K')
plt.legend()
plt.grid(True)
plt.savefig("plot2_knn_curve.png")
