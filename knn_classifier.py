import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Завантаження даних
df = pd.read_csv('fruit_data_with_colors.txt', sep='\t')
print(df.head())

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

features = ['mass', 'width', 'height', 'color_score']
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

X = df[features]
y = df['fruit_label']  # або 'fruit_name'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k_values = range(1, 20)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Візуалізація
plt.plot(k_values, accuracies, marker='o')
plt.xlabel('K значення')
plt.ylabel('Точність')
plt.title('Точність класифікації в залежності від K')
plt.grid(True)
plt.savefig("knn_accuracy_plot.png")


# Вибір кращого значення
best_k = k_values[np.argmax(accuracies)]
print(f'Найкраща точність: {max(accuracies):.2f} при K = {best_k}')
