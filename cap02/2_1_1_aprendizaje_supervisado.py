# Estructura básica de un problema supervisado: carga el dataset Iris, mezcla y divide los datos a mano, 
# entrena una regresión logística y compara predicciones con valores reales.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(42)  # Para reproducibilidad, fijamos la semilla aleatoria

# 1. Cargar el dataset
# load_iris() devuelve un objeto con atributos:
#   - data: las características (features) de cada muestra
#   - target: la etiqueta de cada muestra
#   - feature_names: nombres de las características
iris = load_iris()
X = iris.data    # Matriz de características, forma (150, 4)
y = iris.target  # Vector de etiquetas, forma (150,)
print("Dimensiones de X:", X.shape)
print("Dimensiones de y:", y.shape)

# 2. Mezclar y dividir los datos en entrenamiento (train) y prueba (test)
test_ratio = 0.2  # 20 % de los datos para test, 80 % para entrenamiento
num_samples = X.shape[0]
num_test = int(num_samples * test_ratio)
num_train = num_samples - num_test

# Mezclamos los índices aleatoriamente
indices = np.arange(num_samples)
np.random.shuffle(indices)

# Dividimos los datos
train_indices = indices[:num_train]
test_indices = indices[num_train:]
X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]
print("Número de muestras de entrenamiento:", X_train.shape[0])
print("Número de muestras de prueba:", X_test.shape[0])

# 3. Crear y entrenar el modelo (regresión logística, un modelo supervisado clásico)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4. Hacer predicciones y evaluar
y_pred = model.predict(X_test)
print("Exactitud (accuracy):", accuracy_score(y_test, y_pred))

# 5. Analizar las predicciones
for i in range(len(y_test)):
    print(f"Muestra {i}: Real={y_test[i]}, Predicha={y_pred[i]}")