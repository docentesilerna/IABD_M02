from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X_reg, y_reg = load_diabetes(return_X_y=True)

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

# Creación y entrenamiento del modelo
model_lin = LinearRegression()
model_lin.fit(X_train, y_train)

# Predicciones y evaluación
y_pred = model_lin.predict(X_test)
print("Error cuadrático medio (MSE):", mean_squared_error(y_test, y_pred))
print("Coeficiente de determinación (R²):", r2_score(y_test, y_pred))