# TP: 
# Con el generador de pares
# Hacer regresion lineal -> funcion lineal
# hacer la cuadrática
# hacer la exponencial
# Compararlos


#
# TP Métodos Numéricos - 2023
# Alumnos: 
#          • Bianchi, Guillermo
#          • Martin, Denise
#          • Nava, Alejandro

# Profesores: para poder correr adecuadamente el programa es necesario tenes instaladas las
#             bibliotecas de sympy, numpy y matplotlib.
#             Se puede ver el código comentado pero con "play" toda la teoría y práctica aplicada


# ------------------------------------------------------------------------------------------------------------
# Imports
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

# ------------------------------------------------------------------------------------------------------------
# Funs

# Generators
def generador_pares(cota_minima, cota_maxima):
    # Genera 20 pares de numeros enteros aleatorios según una cota mínima y máxima
    rango = np.arange(cota_minima, cota_maxima)

    # Para evitar errores de un mismo valor xi con varios yi, el replace=False hace que no peudan repetirse esos 
    # numeros aleatorios. En el caso de yi puede repetirse. Cumpliendo con la Inyectividad
    x_set = np.random.choice(rango, size=20, replace=True)
    y_set = np.random.choice(rango, size=20, replace=True)

    # Ordena los pares de forma ascendente
    lista_pares = list(zip(x_set, y_set))
    return sorted(lista_pares, key=lambda x: x[0])


def separador_pares_x_y(pares):
    print()
    # Establece dos listas vacias para llenar con los valores de y y x
    pares_x = []
    pares_y = []
    # Agrega en cada una los valores
    for i in range(len(pares)):
        pares_x.append(pares[i][0])
        pares_y.append(pares[i][1])
    pares_x = np.array(pares_x)
    pares_y = np.array(pares_y)
    return pares_x, pares_y

# Lineal Regression

""" # Se separa aleatoriamente los sets de Train y Test para X e Y
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

# ------------------------------------------------------------------------------------------------------------
# Plots

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
# Raíz cuadrada del error cuadrático medio (RMSE) del Caso 1: 0.20468904393814938 """


# nuevo plot
def reg_lineal_graph(X, Y, a, b):
    plt.plot(X, Y, "o", label="Puntos")
    plt.plot(X, a*X + b, label="Recta")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Recta que mejor se ajusta a los puntos con criterio de cuadrados minimos")
    plt.grid()
    plt.legend()
    plt.show()


# ------------------------------------------------------------------------------------------------------------
# Prints
#  null) Task + Pres
print("                                                                                  ")
print("**********************************************************************************")
print("*              METODOS NUMERICOS - 2023 - TP METODOS DE REGRESION                *")
print("**********************************************************************************")
print("    ALUMNOS:                                                                      ")
print("            • Bianchi, Guillermo                                                  ")
print("            • Martin, Denise                                                      ")
print("            • Nava, Alejandro                                                     ")
print("                                                                                  ")
print("**********************************************************************************")
print("*                                   OBJETIVO                                     *")
print("**********************************************************************************")
print("  Lograr regresión lineal, cuadrática y exponencial de un set de datos dados      ")
print("                                                                                  ")
print("**********************************************************************************")
print("*                                   CONSIGNAS                                    *")
print("**********************************************************************************")
print("                       ")
print("                                                                                  ")


#  I) Theory
print("                                                                                  ")
print("**********************************************************************************")
print("*                                      TEORIA                                    *")
print("**********************************************************************************")
print(" Los Métodos de Interpolación son técnicas utilizadas en cálculo para aproximar una")
print(" función desconocida a partir de un conjunto finito de puntos conocidos, hallando ")
print(" un polinomio de grado mínimo que pase por esos puntos                            ")
print("                                                                                  ")
print("                             ********* NEWTON *********                           ")
print(" El método de Newton utiliza un polinomio interpolante que se construye como una  ")
print(" suma de términos de los polinomios conocidos a los que se agrega el producto entre ")
print(" una constante y las raices de los puntos hasta allí utilizados en el polinomio.  ")
print(" Logicamente el primer polinomio es de grado cero, es decir una constante y       ")
print(" Teniendo un polinomio Pm(x) con varios puntos tomados en cuenta, podriamos agregar")
print(" el siguiente punto como: Pm+1(x) = P(x) + c * (x - x0) (x - x1) ... (x - xm)     ")
print(" Donde cada valor de x0, x1 ... xm, representan los valores que anulan al segundo ")
print(" término, para usar el polinomio anterior que se ajustaba a dichos puntos del set")
print(" C es una conmstante que despejada en cada nuevo polinomio, tiene formato:        ")
print("                 ym+1 - Pm(xm+1)                                                  ")
print(" C = ________________________________________                                     ")
print("     (xm+1 - x0) * (xm+1 - x0) *  (xm+1 - x0)                                     ")
print(" Donde ym+1 es el valor correspondiente al par de xm+1, y Pm(xm+1) es el polinomio")
print(" anterior evaluado en el x del set actual: xm+1                                   ")
print(" La cantidad de datos (n) condiciona el grado del polinomio (P):  gr(P) <= n      ")
print("                                                                                  ")
print("                           ********* LAGRANGE *********                           ")
print(" El método de Lagrange utiliza un polinomio interpolante que se construye como una")
print(" suma de polinomios para cada uno de los puntos del set:                          ")
print(" P(x) = L0(x) y0 + L1(x) y1 + … + Ln(x) yn                                        ")
print(" Donde cada polinomio de lagrange ( L ) evaluado en cada punto del set, puede ser ")
print(" expresado genéricamente como:                                                    ")
print("            (x - x0) * (x - x1) ..... ( x - xn )                                  ")
print(" Li(x) = _________________________________________                                ")
print("            (xi - x0) * (xi - x1) .... (xi - xn)                                  ")
print(" En el demonimador ueda logicamente una constante que depende del valor evaluado  ")
print(" y en el numerador las variables que permiten hacer el polinomio.                 ")
print(" La cantidad de datos (n) condiciona el grado del polinomio (P):  gr(P) <= n      ")
print("                                                                                  ")
print("                     ********* DIFERENCIAS DIVIDIDAS *********                    ")
print(" El método de Diferencias Divididas utiliza un polinomio interpolante que se      ")
print(" construye como una sumatoria de diferencias dvididas, según sea:                 ")
print(" • Progresivo:                                                                    ")
print(" Pn−1(x) = f [x0] + f [x0, x1] * (x − x0) + f [x0, x1, x2] * (x − x0) * (x − x1) ···")
print(" .... + f [x0, x1, ... , xn] · (x − x0) · (x − x1) ... (x − xn−1)                 ")
print(" • Regresivo:                                                                    ")
print(" Pn−1(x) = f [xn] + f [xn, xn−1] * (x − xn) + f [xn, xn−1, xn−2] * (x − xn) * (x − xn−1)")
print(" .... + f [xn, xn−1, ... , x1] * (x − xn) · (x − xn−1) ... (x − x1)                ")
print(" Donde se toma todo el dataset y se realiza las diferencias divididas de todos.   ")
print(" En ambos casos el resultado debería ser igual o similar.                         ")
print("                                                                                  ")

#  II) Examples
print("                                                                                  ")
print("**********************************************************************************")
print("*                                    EJEMPLOS                                    *")
print("**********************************************************************************")
pares = generador_pares(0, 50)
X, Y = separador_pares_x_y(pares)
len_pares = len(pares)
print(f"Los pares ordenados son:\n {pares}")
print(X)
print(Y)
sumaX = sum(X)
sumaY = sum(Y)
sumaXY = sum(X*Y)
suma_X2 = sum(X**2)
sumaX2 = (sum(X))**2
print(f"\nLa suma total de todos los X: {sumaX}"
      f"\nLa suma total de todos los Y: {sumaY}"
      f"\nLa suma total de todos los X.Y: {sumaXY}"
      f"\nLa suma total de todos los X al cuadrado: {suma_X2}"
      f"\nEl cuadrado de la suma de todos los X: {sumaX2}")
print("                                                                                  ")
# Calculo de 'a'(pendiente) y 'b'(ordenada de origen) de la ecuacion 'y = ax + b' para encontrar
# la mejor recta que se aproxime a todos los puntos, con el minimo valor de error posible
a = (len_pares*sumaXY - sumaX*sumaY) / (len_pares*suma_X2 - sumaX2)
b = (suma_X2*sumaY - sumaX*sumaXY) / (len_pares*suma_X2 - sumaX2)
errorCuadratico = sum((a*X + b - Y)**2)
print(f"\nValor de 'a': {a}"
      f"\nValor de 'b': {b}"
      f"\nError cuadratico: {errorCuadratico}")

reg_lineal_graph(X, Y, a, b)

## IV) Conclusions
print("                                                                                  ")
print("**********************************************************************************")
print("*                                  CONCLUSIONES                                  *")
print("**********************************************************************************")
print(" • .                                        ")
print("                                                                                  ")
print("                                                                                  ")
print(" • NOTA1: en las líneas 178 y 179 se encuentra el generador de pares, donde dice: ")
print("   size, se puede modificar el valor para visualizar mejor y con menos cambios abruptos")
print("   la interpolación de los polinomios, por ejemplo en 3 podrán verse las funciones")
print("   cuadráticas con curvas mas suaves y más probabilidad de encontrar una raíz.    ")
print("                                                                                  ")


