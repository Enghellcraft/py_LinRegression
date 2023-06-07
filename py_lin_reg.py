import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.svm import SVC
import seaborn as sns


# Se separa aleatoriamente los sets de Train y Test para X e Y
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 0)
print('La matriz de Variables independientes Train es: \n')
print(x_train)
print('La matriz de Variables independientes Test es: \n')
print(x_test)
print('El array de Variable dependiente Train es: \n')
print(y_train)
print('El array de Variable dependiente Test es: \n')
print(y_test)

# Se procesan los datos para la Regresion Lineal
df_aux = pd.DataFrame({'x_train': x_train.flatten(), 'y_train': y_train.flatten()})
print('El set Train es: \n')
print(df_aux)
df_aux = pd.DataFrame({'x_test': x_test.flatten(), 'y_test': y_test.flatten()})
print('El set Test es: \n')
print(df_aux)

# Se plantea la Regresion con los datos de entrenamiento
regression = LinearRegression()
regression.fit(x_train, y_train)

# Se obtiene el interceptor:
print("El interceptor es: ")
print(regression.intercept_)
# Se obtiene la pendiente
print("La pendiente es: ")
print(regression.coef_)

# Se arman las Predicciones
y_pred = regression.predict(x_test)

# Se plantea el grafico correspondiente al Train
fig = plt.figure(figsize=(6,5), facecolor='ivory')
plt.scatter(x_train, y_train, color = "blue")
plt.plot(x_train, regression.predict(x_train), color = "red")
plt.title("Demencia vs Volumen Cerebral (Train Set)",size = 18, color = 'black')
plt.xlabel("Volumen Cerebral")
plt.ylabel("Demencia")
plt.show()

# Se plantea el grafico correspondiente al Test
fig = plt.figure(figsize=(6,5), facecolor='ivory')
plt.scatter(x_test, y_test, color = "orange")
plt.plot(x_train, regression.predict(x_train), color = "red")
plt.title("Demencia vs Volumen Cerebral (Test Set)",size = 18, color = 'black')
plt.xlabel("Volumen Cerebral")
plt.ylabel("Demencia")
plt.show()

# Se plantea un Grafico de Barras para Regresion Lineal Simple con Train y Test
df1 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df1 = df1.head(50)
df1.plot(kind='bar',figsize=(12,7))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
plt.grid(which='minor', linestyle='dotted', linewidth='1', color='green')
plt.title("Demencia vs Volumen Cerebral (Actual y Prediccion)",size = 18, color = 'black')
plt.xlabel("Cantidad de registros")
plt.ylabel("Demencia")
plt.show()


# Se plantean las Metricas de Evaluacion
print('Error Medio Absoluto (MAE) del Caso 1:', metrics.mean_absolute_error(y_test, y_pred))
print('Error cuadrático medio (MSE) del Caso 1:', metrics.mean_squared_error(y_test, y_pred))
print('Raíz cuadrada del error cuadrático medio (RMSE) del Caso 1:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# Error Medio Absoluto (MAE) del Caso 1: 0.14066423665478933
# Error cuadrático medio (MSE) del Caso 1: 0.04189760470831364
# Raíz cuadrada del error cuadrático medio (RMSE) del Caso 1: 0.20468904393814938



