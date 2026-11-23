#Carga los tres datasets que se usan en los ejemplos del tema (diabetes, iris y vinos) 
# y los convierte en DataFrames.
import pandas as pd
from sklearn.datasets import load_diabetes, load_iris, load_wine

# 1. Dataset de REGRESIÓN: predicción de progresión de diabetes
diabetes = load_diabetes()
X_reg, y_reg = diabetes.data, diabetes.target
print("Dataset de regresión:")
print("Características:", X_reg.shape)
print("Variable objetivo:", y_reg.shape)
print()

# 2. Dataset de CLASIFICACIÓN: clasificación de flores iris
iris = load_iris()
X_clf, y_clf = iris.data, iris.target
print("Dataset de clasificación:")
print("Características:", X_clf.shape)
print("Clases:", set(y_clf))
print()

# 3. Dataset de CLUSTERING: análisis de vinos (dataset real)
wine = load_wine()
X_cluster = wine.data
y_cluster = wine.target  # Aunque existe la etiqueta, no la usaremos para entrenar
print("Dataset de clustering:")
print("Características:", X_cluster.shape)
print("Número de variables:", len(wine.feature_names))
print()

# Convertimos a DataFrames para facilitar la exploración
df_reg = pd.DataFrame(X_reg, columns=diabetes.feature_names)
df_clf = pd.DataFrame(X_clf, columns=iris.feature_names)
df_cluster = pd.DataFrame(X_cluster, columns=wine.feature_names)