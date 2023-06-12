# TP: 
# Con el generador de pares
# Hacer regresion lineal -> funcion lineal
# hacer la cuadrática
# hacer la exponencial
# Compararlos


# PENDIENTES:
# [*] Hacer la funcion lineal una sola funcion que imprima todo
# [*] Hacer la funcion cuadratica en base a la teoria que hice
# [*] Hacer que la cuadratica imprima todo igual que la lineal
# [ ] hacer la funcion exponencial en base a la teoria que hice
# [ ] Hacer que la exponencial imprima todo igual que la lineal
# [*] Hacer grafico para las 3 juntas donde diga el error de cada una
# [ ] Hacer una funcion que tome los 3 errores y diga cual es el mejor fit
# [ ] Buscar 3 dataset: uno ideal para lineal, otro para cuadrática y otro para exponencial
# [ ] Dejar los numeros aleatorios pq ese va  a ser el tercer test
# COMENTAR EL CODIGO APB


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
from fractions import Fraction


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
def my_regressions(pares):
    X, Y = separador_pares_x_y(pares)
    len_pares = len(pares)
    print(f"Los pares ordenados son:\n {pares}")
    print(X)
    print(Y)
    sumaX = sum(X)
    sumaY = sum(Y)
    sumaXY = sum(X * Y)
    suma_X2 = sum(X ** 2)
    sumaX2 = (sum(X)) ** 2
    print("===[REGRESIÓN LINEAL]===")
    print(f"\n\t- La suma total de todos los X: {sumaX}"
          f"\n\t- La suma total de todos los Y: {sumaY}"
          f"\n\t- La suma total de todos los X.Y: {sumaXY}"
          f"\n\t- La suma total de todos los X al cuadrado: {suma_X2}"
          f"\n\t- El cuadrado de la suma de todos los X: {sumaX2}")
    print("                                                                                  ")
    # Cálculo de 'a'(pendiente) y 'b'(ordenada de origen) de la ecuacion 'y = ax + b' para encontrar
    # la mejor recta que se aproxime a todos los puntos, con el minimo valor de error posible
    a = (len_pares * sumaXY - sumaX * sumaY) / (len_pares * suma_X2 - sumaX2)
    b = (suma_X2 * sumaY - sumaX * sumaXY) / (len_pares * suma_X2 - sumaX2)
    error_cuad_lineal = sum((a * X + b - Y) ** 2)
    print(f"\nValor de 'a': {a:.2f}"
          f"\nValor de 'b': {b:.2f}"
          f"\nError cuadratico: {error_cuad_lineal:.2f}")

    suma_X4 = sum(X ** 4)
    suma_X3 = sum(X ** 3)
    sumaX2_Y = sum((X ** 2) * Y)

    # # [VERSION DE ENUNCIADO ORIGINAL - DA RARO]
    # sumaX2_2 = sum((X ** 2) * 2)
    # a_diapo =  (len_pares * sumaX2_Y - sumaX2 * sumaY) / (len_pares * suma_X4 - sumaX2_2)
    # b_diapo = (suma_X4 * sumaY - suma_X2 * sumaX2_Y) / (len_pares * suma_X4 - sumaX2_2)
    # error_cuad_diapo = sum((a_diapo * X**2 + b_diapo - Y) ** 2)
    # print("===[REGRESIÓN CUADRATICA (DIAPO)]===")
    # print(f"\nPara este sistema:"
    #       f"\nValor de 'a': {a_diapo:.2f}"
    #       f"\nValor de 'b': {b_diapo:.2f}"
    #       f"\nError cuadratico: {error_cuad_diapo:.2f}")

    # Calculo de los tres coeficientes del sistema cuadratico. Al ser más complejo se resuelve via numpy.
    cuad_mat = np.array([[suma_X4, suma_X3, suma_X2], [suma_X3, suma_X2, sumaX], [suma_X2, sumaX, len_pares]])
    cuad_resmat = np.array([sumaX2_Y, sumaXY, sumaY])

    cuad_abc_mat = np.linalg.solve(cuad_mat, cuad_resmat)
    error_cuad_cuad = sum((cuad_abc_mat[0] * X ** 2 + cuad_abc_mat[1] * X + cuad_abc_mat[2] - Y) ** 2)
    print("===[REGRESIÓN CUADRATICA]===")
    print(f"\nSe agregan a lo anterior:"
          f"\n\t- La suma total de todos los X al cubo: {suma_X3}"
          f"\n\t- La suma total de todos los X a la cuarta: {suma_X4}"
          f"\n\t- La suma total de todos los X^2 por Y: {sumaX2_Y}")
    print(f"\nPara este sistema:"
          f"\nValor de 'a': {cuad_abc_mat[0]:.2f}"
          f"\nValor de 'b': {cuad_abc_mat[1]:.2f}"
          f"\nValor de 'c': {cuad_abc_mat[2]:.2f}"
          f"\nError cuadratico: {error_cuad_cuad:.2f}")

    suma_lnX = sum(np.log(X))
    suma_lnX_lnX = sum(np.log(X) * np.log(X))
    suma_lnX_2 = sum(np.log(X)) ** 2
    suma_lnX_3 = sum(np.log(X)) ** 3
    suma_lnY = sum(np.log(Y))
    suma_x_lnY = sum(X * np.log(Y))
    suma_y_lnX = sum(Y * np.log(X))
    sumaX_2 = sum(X) ** 2

    # Intento 1 de resolver exponencial (y = a * b^x) como dice el desarrollo original
    a_exp = (sumaY * suma_lnX) / (suma_lnX)
    b_exp = ((suma_y_lnX * suma_lnX * suma_lnX_lnX) / (suma_lnX_3 - suma_y_lnX * suma_lnX / suma_lnX_2)) / (
            suma_lnX * suma_lnX_lnX / suma_lnX_2)

    # Intento 2 de resolver exponencial ( y = a * e^(bx) ), versión de guille.
    exp_mat = np.array([[len_pares, sumaX], [sumaX, sumaX2]])
    exp_resmat = np.array([suma_lnY, suma_x_lnY])
    exp_ab_mat = np.linalg.solve(exp_mat, exp_resmat)
    exp_ab_mat = (np.exp(exp_ab_mat[0]), exp_ab_mat[1])  # Lo de arriba resuelve "ln(y) = ln(a) * bx"

    # exp_ab_mat = (a_exp, b_exp) # Comentar para usar versión guille, dejar activo para version OG.
    error_cuad_exp = sum(exp_ab_mat[0] * (np.e ** (exp_ab_mat[1] * X)))
    print("===[REGRESIÓN EXPONENCIAL]===")
    print(f"\nSe agregan a lo anterior:"
          f"\n\t- La suma total de todos los ln(Y): {suma_lnY}"
          f"\n\t- La suma total de todos los X * ln(Y): {suma_x_lnY}")
    print(f"\nPara este sistema:"
          f"\nValor de 'a': {exp_ab_mat[0]:.2f}"
          f"\nValor de 'b': {exp_ab_mat[1]:.2f}"
          f"\nError cuadratico: {error_cuad_exp:.2f}")

    # regressions_graph(X, Y, a, b, a_cuadratica, b_cuadratica)

    # regressions_graph(X, Y, (a, b), (a_diapo, b_diapo), abc_mat)
    regressions_graph(X, Y, (a, b), error_cuad_lineal, cuad_abc_mat, error_cuad_cuad, exp_ab_mat, error_cuad_exp)


# ------------------------------------------------------------------------------------------------------------
# Plots
# Linear Regression
def regressions_graph(X, Y, ab_lineal, err_lineal, cuad_mat, err_cuad, exp_mat, err_exp):
    plt.plot(X, Y, "o", label="Dataset")
    # plt.ylim(0, Y.max() * 1.2)
    a, b = ab_lineal
    label_lin = f"Regresión Lineal\n[E = {err_lineal:.2f}]"
    plt.plot(X, a * X + b, label=label_lin)
    # a_diapo, b_diapo = ab_diapo
    # plt.plot(X, a_diapo * (X ** 2) + b_diapo, label="Regresión Cuadrática (Diapo)")
    a_cuad, b_cuad, c_cuad = cuad_mat
    label_cuad = f"Regresión Cuadrática\n[E = {err_cuad:.2f}]"
    plt.plot(X, a_cuad * (X ** 2) + b_cuad * X + c_cuad, label=label_cuad)
    a_exp, b_exp = exp_mat
    label_exp = f"Regresión Exponencial\n[E = {err_exp:.2f}]"
    plt.plot(X, a_exp * np.exp(b_exp * X), label=label_exp)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Recta que mejor se ajusta a los puntos con criterio de cuadrados minimos")
    plt.grid()
    # plt.legend(loc='upper right', bbox_to_anchor=(1.2, 0.6))
    plt.legend()
    plt.tight_layout()
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
print("                                                                                  ")
print("                                                                                  ")

#  I) Theory
print("                                                                                  ")
print("**********************************************************************************")
print("*                                      TEORIA                                    *")
print("**********************************************************************************")
print("                                                                                  ")
print("                       ********* REGRESION LINEAL *********                       ")
print("                                                                                  ")
print(" El método de Regresion Lineal relaciona puntos de un dataset y puede proveer alguna")
print(" predicción sobre nuevos puntos.                                                  ")
print(" Modela la relación entre una variable dependiente y una o más variables independientes")
print(" utilizando la ecuación lineal: y = a*x + b, donde 'a' es la pendiente y 'b' la   ")
print(" ordenada al origen y son las variables que caracterizaran a la recta encontrada. ")
print(" Para que la recta sea lo mas fidedigna al dataset, es importante minimizar el    ")
print(" error respecto a cada una de las coordenadas y trabjar con datos aproximadamente lineales.")
print(" • ERROR: para hallar extremos locales (mínimos, máximos, puntos de inflexión) de ")
print("          una función de varias variables, es necesario derivarla en función a cada")
print("          variable y luego igualarlas a cero.                                   ")
print("          En este caso para minimizar el error de los puntos a la recta utilizaremos")
print("          el método de cuadrados mínimos, ya que es más sencillo de calcular que el ")
print("          caso de módulo.                                                         ")
print("          Siendo el Error Cuadrático: Σ(a*xi+b-yi)^2, se deriva y despeja en base a:")
print("          - la pendiente: E'a(a,b) = 0                                            ")
print("                          a * Σxi^2 + b * Σxi = Σxiyi                             ")
print("          - la ordenada al origen: E'b(a,b) = 0                                   ")
print("                          a * Σxi + b * n = Σyi                                   ")
print("          Para hallar a y b, se realiza un sistema matricial de 2*2:              ")
print("          a Σ xi2 + b Σ xi = Σ xiyi [1]                                           ")
print("          a Σ xi  + b n    = Σ yi   [2]                                           ")
print("          Despejando ambos valores se pueden obtener mediante:                    ")
print("          - la pendiente:                  n Σ xiyi – Σ xi Σ yi                   ")
print("                                     a = ________________________                 ")
print("                                           n Σ xi^2 – (Σ xi)^2                    ")
print("          - la ordenada al origen:        Σ xi^2 Σ yi – Σ xi Σ xiyi               ")
print("                                     b = ________________________                 ")
print("                                           n Σ xi^2 – (Σ xi)^2                    ")
print("          + Si existen sólo dos pares en el dataset, el error será cero, pues la recta")
print("            pasará obligadamente por ambos (siempre que tengan ≠ x). También dará ")
print("            cero en caso de tener todas las coordenadas alineadas.                ")
print(" • EXISTENCIA Y UNICIDAD: pueden verificarse mediante la determinante del sistema,")
print("                          si el determinante es ≠ 0 entonces existe una sola recta")
print("                          si es = 0, hay infinitas rectas (esto puede suceder cuando")
print("                          los valores Xi son iguales) = sistema compatible indeterminado.")
print("          - Para que 'a' y 'b' queden indefinidos debería haber sólo un punto como")
print("            dataset dejando infinitas rectas como solución del sistema.           ")
print("          - En el caso de que todos los Yi sean iguales, la pendiente = 0.        ")
print("                                                                                  ")
print("                      ********* REGRESION CUADRATICA *********                    ")
print("                                                                                  ")
print(" Similar método de Regresion Lineal, la Regresión Cuadrática relaciona puntos de  ")
print(" un dataset y puede proveer algun tipo de predicción sobre nuevos puntos.         ")
print(" Aquí se utiliza una parábola sin el término lineal: y = a * x^2 + b , donde 'a'  ")
print(" es el coeficiente principal y 'b' la ordenada al origen.                         ")
print(" • ERROR: para hallar extremos locales (mínimos, máximos, puntos de inflexión) de ")
print("          una función de varias variables, es necesario derivarla en función a cada")
print("          variable y luego igualarlas a cero.                                     ")
print("          Siendo el Error Cuadrático: Σ(a*xi^2+b-yi)^2, se deriva y despeja en base a:")
print("          - el Coeficiente Principal: E'a(a,b) = 0                                ")
print("                          a Σ xi^4 + b Σ xi^2 = Σ xi^2yi                             ")
print("          - la ordenada al origen: E'b(a,b) = 0                                   ")
print("                          a Σ xi^2 + b n = Σ yi                                    ")
print("          Para hallar a y b, se realiza un sistema matricial de 2*2:              ")
print("          a Σ xi^4 + b Σ xi^2 = Σ xi^2yi [1]                                      ")
print("          a Σ xi^2 + b n      = Σ yi     [2]                                      ")
print("          Despejando ambos valores se pueden obtener mediante:                    ")
print("          - el coeficiente pricipal:       n Σ xi^2yi – Σ xi2 Σ yi                ")
print("                                     a = ___________________________              ")
print("                                           n Σ xi^4 – (Σ xi^2)^2                  ")
print("          - la ordenada al origen:       Σ xi^4 Σ yi – Σ xi^2 Σ xi^2yi            ")
print("                                     b = ___________________________              ")
print("                                            n Σ xi^4 – (Σ xi^2)^2                 ")
print("          + Las parábolas, incluyen en su cálculo a la familia de las rectas, por lo ")
print("            tanto para un data set, el Error de una parábola es ≤ el de una recta. ")
print("                                                                                  ")
print("                     ********* REGRESION EXPONENCIAL *********                    ")
print("                                                                                  ")
print(" La Regresión Exponencial es una técnica utilizada para encontrar una función     ")
print(" exponencial que mejor se acomode a lospuntos de un dataset y puede proveer algun ")
print(" tipo de predicción sobre nuevos puntos.                                          ")
print(" Aquí se utiliza: y = a*b^x , donde 'a' ≠ 0.                                      ")
print(" Se utiliza en casos donde los datos crecen lentamente al principio y luego muy   ")
print(" aceleradamente.                                                                  ")
print(" • ERROR: para hallar extremos locales (mínimos, máximos, puntos de inflexión) de ")
print("          una función de varias variables, es necesario derivarla en función a cada")
print("          variable y luego igualarlas a cero.                                     ")
print("          Siendo el Error Cuadrático:  Σ(y_i - (a * b^x_i))^2, se deriva y despeja.")
print("          Los valores obtenidos son:                                              ")
print("          - la constante:                     Σy_i * ln(x_i)                      ")
print("                                     a = ___________________________              ")
print("                                                   Σln(x_i)                       ")
print(
    "          - la base:       Σ(yi*ln(xi))*Σ(ln(xi))*Σ(ln(xi) *ln(xi)) / Σ(ln(xi))^3 - Σ(yi*ln(xi))*Σ(ln(xi)) / Σ(ln(xi))^2 ")
print(
    "                        b = _____________________________________________________________________________________________ ")
print(
    "                                                 Σ(ln(xi))*Σ(ln(xi)*ln(xi)) / Σ(ln(xi))^2                                 ")
print("                                                                                  ")
print("                     ********* REGRESION CUADRÁTICA (GUILLE) *********                     ")
print(" El método de Regresion Cuadrática es similar al lineal, solo que modela para una parábola en vez de una recta.")
print(" Modela la relación entre una variable dependientre y una o más variables independientes")
print(" utilizando la ecuación lineal: y = a*x^2 + bx + c, donde 'a' es la curvatura (a != 0), 'b' es el")
print("desplazamiento horizontal y 'c' el desplazamiento vertical de la parábola encontrada. ")
print(" • ERROR: En este caso para minimizar el error de los puntos a la recta utilizaremos")
print("          el método de cuadrados mínimos como en el caso lineal, ")
print("          Siendo el Error Cuadrático: Σ[ a*(xi^2) + b*xi + c - yi ]^2, se deriva y despeja en base a:")
print("          - la curvatura:")
print("                            E'a(a,b,c) = 0                                            ")
print("                            a * Σ[xi^4] + b * Σ[xi^3] + c * Σ[xi] = Σ[ yi * (xi^2) ]                 ")
print("          - el desplazamiento horizontal:")
print("                            E'b(a,b,c) = 0                                   ")
print("                            a * Σ[xi^3] + b * Σ[xi^2] + c * Σ[xi] = Σ[yi*xi]                 ")
print("          - el desplazamiento vertical:")
print("                            E'c(a,b,c) = 0                                   ")
print("                            a * Σ[xi^2] + b * Σ[xi] + c * n = Σ[yi]                 ")
print("          Para hallar a y b, se realiza un sistema matricial de 3*3:              ")
print("          a Σ xi4 + b Σ xi3 + c Σ xi2 = Σ [xi2 * yi]    [1]                                     ")
print("          a Σ xi3 + b Σ xi2 + c Σ xi  = Σ xiyi          [2]                                     ")
print("          a Σ xi2 + b Σ xi  + c * n   = Σ yi            [3]                                     ")
print("                                                                                                ")
print("          Podemos hacer uso de este sistema, reemplazar con los valores calculados              ")
print("          para este dataset y despejar los valores de a, b y c.                                 ")
print("                                                                                  ")
print("                     ********* REGRESION EXPONENCIAL (GUILLE) *********                     ")
print(" El método de Regresion Exponencial utiliza la ecuación: y = a * e(b*x), donde 'a' es la constante")
print("y 'b' el modificador de la curvatura en la exponencial. ")
print(" • ERROR: En este caso para minimizar el error de los puntos a la recta utilizaremos")
print("          el método de cuadrados mínimos como en el caso lineal, ")
print("          Siendo el Error Cuadrático: Σ[ a*e^(b*x) - yi ]^2, se deriva y despeja en base a:")
print("          - la constante:")
print("                            E'a(a,b) = 0                                            ")
print("                            a * Σ[b * x * e^(b*x)] + b * n = Σ[ b * yi * xi ]                 ")
print("          - la curvatura:")
print("                            E'b(a,b,c) = 0                                   ")
print("                            a * Σ[xi^3] + b * Σ[xi^2] + c * Σ[xi] = Σ[yi*xi]                 ")
print("          - el desplazamiento vertical:")
print("                            E'c(a,b,c) = 0                                   ")
print("                            a * Σ[xi^2] + b * Σ[xi] + c * n = Σ[yi]                 ")
print("          Para hallar a y b, se realiza un sistema matricial de 3*3:              ")
print("          a Σ xi4 + b Σ xi3 + c Σ xi2 = Σ [xi2 * yi]    [1]                                     ")
print("          a Σ xi3 + b Σ xi2 + c Σ xi  = Σ xiyi          [2]                                     ")
print("          a Σ xi2 + b Σ xi  + c * n   = Σ yi            [3]                                     ")
print("                                                                                                ")
print("          Podemos hacer uso de este sistema, reemplazar con los valores calculados              ")
print("          para este dataset y despejar los valores de a, b y c.                                 ")

#  II) Examples
print("                                                                                  ")
print("**********************************************************************************")
print("*                                    EJEMPLOS                                    *")
print("**********************************************************************************")
pares = generador_pares(1, 50)
my_regressions(pares)
## IV) Conclusions
print("                                                                                  ")
print("**********************************************************************************")
print("*                                  CONCLUSIONES                                  *")
print("**********************************************************************************")
print(" • La Regresión Lineal es una herramienta poderosa para analizar las relaciones entre")
print(" variables y pronosticar, pero tiene sus limitaciones debido a sus supuestos de   ")
print(" linealidad y distribución normal. Es crucial comprender estos supuestos y sus    ")
print(" implicaciones cuando se utiliza la regresión lineal para aplicaciones prácticas. ")
print("                                                                                  ")
print(" • VENTAJAS DE REGRESION LINEAL:                                                  ")
print("   + Simplicidad para su implementación                                           ")
print("   + Transparente e interpretable (visualy analíticamente)                        ")
print("   + Versátil y flexible para distintos tipos de datos                            ")
print("                                                                                  ")
print(" • DESVENTAJAS DE REGRESION LINEAL:                                               ")
print("   + Sensibilidad a datos: cuando las variables independientes tengan alta correlatividad")
print("     puede afectar la estabilidad y presición de los coeficientes                 ")
print("   + Pobre performance por ser proclibe a overfitting y underfitting (mucha exactitud")
print("     o muy poca).                                                                 ")
print("   + Asume relación lineal entre las variables cuando podría no ser el caso.      ")
print("   + Requiere valores Gaussianos de distribución para mejor calce (normalización) ")
print("                                                                                  ")
print("                                 ****************                                 ")
print("                                                                                  ")
print(" • La Regresión Cuadrática es una extensión de la regersión lineal, como regresión")
print(" lineal múltiple, donde la relación entre la variable dependiente y la independiente")
print(" se modela al grado del polinomio (en este caso 2).                               ")
print("                                                                                  ")
print(" • VENTAJAS DE REGRESION CUADRATICA:                                              ")
print("   + Mejora el calce respecto a la regesión lineal.                               ")
print("   + Las curvas de las parábolas son mas flexibles para acomodarse mejor al dataset.")
print("                                                                                  ")
print(" • DESVENTAJAS DE REGRESION CUADRATICA:                                           ")
print("   + Al ser mas flexible también puede traer mas errores de overfitting.          ")
print("   + Más complejo de entender e interpretar que la lineal.                        ")
print("   + Computacionalmente más costoso.                                              ")
print("                                                                                  ")
print("                                 ****************                                 ")
print("                                                                                  ")
print(" • La Regresión Exponencial es ideal para casos de crecimiento o decrecimiento    ")
print(" exponencial, es de técnica robusta pero no sirve para todos los sistemas.        ")
print(" Su poder predictivo se toma como R^2 entre 0 y 1 (más cercano al 1, mejor es).        ")
print("                                                                                  ")
print(" • VENTAJAS DE REGRESION EXPONENCIAL:                                              ")
print("   + Es excelente en casos de crecimiento o decrecimiento exponencial.             ")
print("   + Suele ser mas robusta incluso que algunos polinomios.                        ")
print("                                                                                  ")
print(" • DESVENTAJAS DE REGRESION EXPONENCIAL:                                           ")
print("   + No es bueno con casos de variacion de datos.                                 ")
print("   + No es confiable en casos que no tengas relaciones tipo exponencial.          ")
print("   + Asume continuidad de datos y peude afectar las predicciones.                 ")
print("                                                                                  ")
print(" • NOTA1:  ")
print("                                                                                  ")
