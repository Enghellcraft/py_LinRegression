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
import sympy as sym
from numpy.compat import long
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


# Regressions
# a) Linear
def find_ab_lin_reg(X_set, Y_set, X_name="X", Y_name="Y", a_name="a", b_name="b"):
    len_pares = len(X_set)
    sumaX = np.sum(X_set)
    sumaY = np.sum(Y_set)
    sumaXY = np.sum(X_set * Y_set)
    suma_X2 = np.sum(X_set ** 2)
    sumaX_2 = (np.sum(X_set)) ** 2
    print(f"\n\t- La suma total de todos los '{X_name}': {sumaX}"
          f"\n\t- La suma total de todos los '{Y_name}': {sumaY}"
          f"\n\t- La suma total de todos los '{X_name}.{Y_name}': {sumaXY}"
          f"\n\t- La suma total de todos los '{X_name}' al cuadrado: {suma_X2}"
          f"\n\t- El cuadrado de la suma de todos los '{X_name}': {sumaX_2}"
          "                                                              ")
    a = (len_pares * sumaXY - sumaX * sumaY) / (len_pares * suma_X2 - sumaX_2)
    b = (suma_X2 * sumaY - sumaX * sumaXY) / (len_pares * suma_X2 - sumaX_2)
    print(f"\nValor de '{a_name}': {a:.2f}"
          f"\nValor de '{b_name}': {b:.2f}")
    return a, b

# b) Cuadratic
def find_abc_quad_reg(X_set, Y_set):
    len_pares = len(X_set)
    sumaX = np.sum(X_set)
    sumaY = np.sum(Y_set)
    sumaXY = np.sum(X_set * Y_set)
    suma_X2 = np.sum(X_set ** 2)
    sumaX_2 = (np.sum(X_set)) ** 2
    suma_X4 = np.sum(X_set ** 4)
    suma_X3 = np.sum(X_set ** 3)
    sumaX2_Y = np.sum((X_set ** 2) * Y_set)
    print(f"\n\t- La suma total de todos los 'X': {sumaX}"
          f"\n\t- La suma total de todos los 'Y': {sumaY}"
          f"\n\t- La suma total de todos los 'X.Y': {sumaXY}"
          f"\n\t- La suma total de todos los 'X' al cuadrado: {suma_X2}"
          f"\n\t- El cuadrado de la suma de todos los 'X': {sumaX_2}"
          f"\n\t- La suma total de todos los 'X' al cubo: {suma_X3}"
          f"\n\t- La suma total de todos los 'X' a la cuarta: {suma_X4}"
          f"\n\t- La suma total de todos los 'X^2 . Y': {sumaX2_Y}"
          "                                                              ")

    # Calculo de los tres coeficientes del sistema cuadratico. Al ser más complejo se resuelve via numpy.
    cuad_mat = np.array([[suma_X4, suma_X3, suma_X2], [suma_X3, suma_X2, sumaX], [suma_X2, sumaX, len_pares]])
    cuad_resmat = np.array([sumaX2_Y, sumaXY, sumaY])
    a, b, c = np.linalg.solve(cuad_mat, cuad_resmat)
    print(f"\nValor de 'a': {a:.2f}"
          f"\nValor de 'b': {b:.2f}"
          f"\nValor de 'c': {c:.2f}")
    return a, b, c

# c) Exponential
def create_f_sym_exponential(a_exp, b_exp):
    x = sym.symbols('x')
    f_sym = b_exp * (x ** a_exp)
    f = sym.lambdify(x, f_sym)
    return f

def create_f_sym_exponential_euler(a_exp, b_exp):
    x = sym.symbols('x')
    f_sym_euler = b_exp * sym.exp(a_exp * x)
    f = sym.lambdify(x, f_sym_euler)
    return f

def pretty_print_sym_exp(f_sym):
    x = sym.symbols('x')
    f_lambda = sym.sympify(sym.lambdify(x, f_sym))
    f_lambda_str = str(f_lambda)
    return f_lambda_str

# Prints
# All Regressions
def my_regressions(pares):
    X, Y = separador_pares_x_y(pares)
    len_pares = len(X)
    sumaX = np.sum(X)
    sumaY = np.sum(Y)
    suma_X2 = np.sum(X ** 2)
    print(f"Los pares ordenados son:\n {pares}")
    print(X)
    print(Y)
    
    print("===[REGRESIÓN LINEAL]===")
    # Cálculo de 'a'(pendiente) y 'b'(ordenada de origen) de la ecuacion 'y = ax + b' para encontrar
    # la mejor recta que se aproxime a todos los puntos, con el minimo valor de error posible
    a, b = find_ab_lin_reg(X, Y)
    error_cuad_lineal = np.sum((a * X + b - Y) ** 2)
    print(f"Error cuadratico: {error_cuad_lineal:.2f}")

    # Plot Funcion Lineal
    f_lin = np.poly1d((a, b))
    print("La expresion de la función lineal es:")
    print(f_lin)
    regressions_graph_unit(X, Y, f_lin, error_cuad_lineal, "Regresion lineal", 'orange')

    suma_X4 = np.sum(X ** 4)
    suma_X3 = np.sum(X ** 3)
    sumaX2_Y = np.sum((X ** 2) * Y)

    # # [VERSION DE ENUNCIADO ARIEL - DA RARO]
    # sumaX2_2 = sum((X ** 2) * 2)
    # a_diapo =  (len_pares * sumaX2_Y - sumaX2 * sumaY) / (len_pares * suma_X4 - sumaX2_2)
    # b_diapo = (suma_X4 * sumaY - suma_X2 * sumaX2_Y) / (len_pares * suma_X4 - sumaX2_2)
    # error_cuad_diapo = sum((a_diapo * X**2 + b_diapo - Y) ** 2)
    # print("===[REGRESIÓN CUADRATICA (DIAPO)]===")
    # print(f"\nPara este sistema:"
    #       f"\nValor de 'a': {a_diapo:.2f}"
    #       f"\nValor de 'b': {b_diapo:.2f}"
    #       f"\nError cuadratico: {error_cuad_diapo:.2f}")

    print("===[REGRESIÓN CUADRATICA]===")

    # Calculo de los tres coeficientes del sistema cuadratico. Al ser más complejo se resuelve via numpy.
    cuad_abc_mat = find_abc_quad_reg(X, Y)

    error_cuad_cuad = np.sum((cuad_abc_mat[0] * X ** 2 + cuad_abc_mat[1] * X + cuad_abc_mat[2] - Y) ** 2)
    print(f"\nError cuadratico: {error_cuad_cuad:.2f}")

    # Plot Funcion Cuadratica
    f_cuad = np.poly1d(cuad_abc_mat)
    print("La expresion de la función cuadrática es:")
    print(f_cuad)
    regressions_graph_unit(X, Y, f_cuad, error_cuad_cuad, "Regresion cuadrática", 'purple')

    suma_lnX = np.sum(np.log(X))
    suma_lnX_lnX = np.sum(np.log(X) * np.log(X))
    suma_lnX_2 = np.sum(np.log(X)) ** 2
    suma_lnX_3 = np.sum(np.log(X)) ** 3
    suma_lnY = np.sum(np.log(Y))
    suma_x_lnY = np.sum(X * np.log(Y))
    suma_y_lnX = np.sum(Y * np.log(X))
    sumaX_2 = np.sum(X) ** 2

    print("===[REGRESIÓN EXPONENCIAL: y = b * x^a]===")

    # # Intento 1 de resolver exponencial (y = a * b^x) como dice el desarrollo original
    # a_exp = (sumaY * suma_lnX) / suma_lnX
    # b_exp = ((suma_y_lnX * suma_lnX * suma_lnX_lnX) / (suma_lnX_3 - suma_y_lnX * suma_lnX / suma_lnX_2)) / (
    #         suma_lnX * suma_lnX_lnX / suma_lnX_2)

    # Intento 2 de resolver exponencial (y = b * x^a) con la funcion modularizada
    # Resuelvo ln(y) = ln(b) + a * ln(x)
    a_exp, ln_b_exp = find_ab_lin_reg(np.log(X), np.log(Y), X_name="ln(X)", Y_name="ln(Y)", b_name="ln(b)")
    # Luego potencio con euler para obtener los valores que quiero
    b_exp = np.exp(ln_b_exp)
    error_cuad_exp = np.sum(b_exp * (X ** a_exp))
    print(f"\nError cuadratico: {error_cuad_exp:.2f}")

    # Plot Funcion Exponencial
    f_exp = create_f_sym_exponential(a_exp, b_exp)
    print("La expresion de la función exponencial es:")
    # print( pretty_print_sym_exp(f_exp) )
    print(str(f_exp))
    regressions_graph_unit(X, Y, f_exp, error_cuad_exp, "Regresion exponencial", 'sienna')

    print("===[REGRESIÓN EXPONENCIAL EULER: y = b * e^(a*x)]===")

    # Intento de resolver exponencial ( y = b * e^(ax) ).
    a_exp_euler, ln_b_exp_euler = find_ab_lin_reg(X, np.log(Y), Y_name="ln(Y)", b_name="ln(b)")
    b_exp_euler = np.exp(ln_b_exp_euler)
    error_cuad_exp_euler = np.sum(b_exp * np.exp(a_exp_euler * X))

    # Plot Funcion Exponencial de Euler
    f_exp_euler = create_f_sym_exponential_euler(a_exp_euler, b_exp_euler)
    print("La expresion de la función exponencial Euler es:")
    print(str(f_exp_euler))
    regressions_graph_unit(X, Y, f_exp_euler, error_cuad_exp_euler, "Regresion exponencial Euler", 'tomato')

    regressions_graph(
        X, Y,
        (a, b), error_cuad_lineal,
        cuad_abc_mat, error_cuad_cuad,
        (a_exp, b_exp), error_cuad_exp,
        f_exp_euler, error_cuad_exp_euler
    )


# ------------------------------------------------------------------------------------------------------------
# Plots
# Linear Regression
def regressions_graph(X, Y, ab_lineal, err_lineal, cuad_mat, err_cuad, exp_mat, err_exp, euler_lambda_exp, err_euler):
    plt.plot(X, Y, "o", label="Dataset", color='turquoise')

    a, b = ab_lineal
    label_lin = f"Regresión Lineal\n[E = {err_lineal:.2f}]"
    plt.plot(X, a * X + b, label=label_lin, color='orange')

    # a_diapo, b_diapo = ab_diapo
    # plt.plot(X, a_diapo * (X ** 2) + b_diapo, label="Regresión Cuadrática (Diapo)")
    a_cuad, b_cuad, c_cuad = cuad_mat
    label_cuad = f"Regresión Cuadrática\n[E = {err_cuad:.2f}]"
    plt.plot(X, a_cuad * (X ** 2) + b_cuad * X + c_cuad, label=label_cuad, color='purple')
    
    a_exp, b_exp = exp_mat
    label_exp = f"Regresión Exponencial\n[E = {err_exp:.2f}]"
    plt.plot(X, b_exp * (X ** a_exp), label=label_exp, color='sienna')
    label_euler = f"Regresión Exp. (Euler)\n[E = {err_euler:.2f}]"
    # a_euler, b_euler = euler_mat
    # VER CON GUILLE. CAMBIO LA FUNCION PARA QUE RECIBA LA EXPRESION LAMBDA
    # plt.plot(X, b_exp * np.exp(a_exp * X), label=label_euler)
    
    plt.ylim(-(Y.max() / 4), Y.max() * 1.1)
    yax = plt.gca().yaxis
    for item in yax.get_ticklabels(): 
        item.set_rotation(45)
    
    plt.plot(X, euler_lambda_exp(X), label=label_euler, color='tomato')
    plt.xlabel("Días", fontweight='bold')
    plt.ylabel("Acumulación de individuos infectados", fontweight='bold')
    plt.legend() 
    plt.title("Grafico de cuadrados mínimos")
    plt.grid()
    # plt.legend(loc='upper right', bbox_to_anchor=(1.2, 0.6))
    plt.tight_layout()
    plt.show()

def regressions_graph_unit(X, Y, func, err, msg, c):
    plt.title(msg)

    plt.plot(X, Y, 'o', color='turquoise', markersize=5, label="Dataset")
    plt.plot(X, func(X), color=c, linestyle='-', linewidth=2, label=msg+f" [E = {err:.2f}]")

    plt.xlabel("Días", fontweight='bold')
    plt.ylabel("Acumulación de individuos infectados", fontweight='bold')
    plt.legend()    
    
    yax = plt.gca().yaxis
    for item in yax.get_ticklabels(): 
        item.set_rotation(45)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.ylim(-(Y.max() / 4), Y.max() * 1.1)
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
print("         ********* REGRESION CUADRATICA SIN COEFICIENTE LINEAL *********          ")
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
print("         ********* REGRESION CUADRATICA CON COEFICIENTE LINEAL *********          ")
print(" De mismo modo que sin coeficiente, modela una relación tipo parábola del dataset.")
print(" Utiliza la ecuación lineal: y = a*x^2 + bx + c, donde 'a' es la curvatura (a != 0),")
print(" 'b' desplazamiento y 'c' la ordenada al origen.                                  ")
print(" • ERROR: En este caso para minimizar el error de los puntos a la parábola        ")
print("          utilizaremos el método de cuadrados mínimos como en el caso anterior,   ")
print("          Siendo el Error Cuadrático: Σ[ a*(xi^2) + b*xi + c - yi ]^2,            ")
print("           se deriva y despeja en base a:                                         ")
print("          - la curvatura: E'a(a,b,c) = 0                                          ")
print("                            a * Σ[xi^4] + b * Σ[xi^3] + c * Σ[xi] = Σ[yi * (xi^2)]")
print("          - el desplazamiento: E'b(a,b,c) = 0                                     ")
print("                            a * Σ[xi^3] + b * Σ[xi^2] + c * Σ[xi] = Σ[yi*xi]      ")
print("          - la ordenada al origen: E'c(a,b,c) = 0                                 ")
print("                            a * Σ[xi^2] + b * Σ[xi] + c * n = Σ[yi]               ")
print("          Para hallar a y b, se realiza un sistema matricial de 3*3:              ")
print("          a Σ xi4 + b Σ xi3 + c Σ xi2 = Σ [xi2 * yi]    [1]                       ")
print("          a Σ xi3 + b Σ xi2 + c Σ xi  = Σ xiyi          [2]                       ")
print("          a Σ xi2 + b Σ xi  + c * n   = Σ yi            [3]                       ")
print("          Podemos hacer uso de este sistema, reemplazar con los valores calculados")
print("          para este dataset y despejar los valores de a, b y c.                   ")
print("                                                                                  ")
print("                     ********* REGRESION EXPONENCIAL *********                    ")
print("                                                                                  ")
print(" La Regresión Exponencial es una técnica utilizada para encontrar una función     ")
print(" exponencial que mejor se acomode a lospuntos de un dataset y puede proveer algun ")
print(" tipo de predicción sobre nuevos puntos.                                          ")
print(" Aquí se utiliza: y = a*b^x , donde 'a' ≠ 0 y constante, y 'b' la base.           ")
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
print("          - la base:     Σ(yi*ln(xi))*Σ(ln(xi))*Σ(ln(xi) *ln(xi)) / Σ(ln(xi))^3 - Σ(yi*ln(xi))*Σ(ln(xi)) / Σ(ln(xi))^2 ")
print("                     b = _____________________________________________________________________________________________ ")
print("                                               Σ(ln(xi))*Σ(ln(xi)*ln(xi)) / Σ(ln(xi))^2                                ")
print("                                                                                  ")
print("                   ********* REGRESION EXPONENCIAL EULER *********                ")
print(" El método de Regresion Exponencial utiliza la ecuación: y = a * e^(b*x), donde    ")
print(" donde 'a' ≠ 0 y constante, y 'b' el modificador de la curvatura en la exponencial.")
print(" • ERROR: En este caso para minimizar el error de los puntos a la recta utilizaremos")
print("          el método de cuadrados mínimos como en el caso lineal,                  ")
print("          Siendo el Error Cuadrático: Σ[ a*e^(b*x) - yi ]^2, se deriva y despeja: ")
print("          - la constante: E'a(a,b) = 0                                            ")
print("                            a * Σ[b * x * e^(b*x)] + b * n = Σ[ b * yi * xi ]     ")
print("          - la curvatura: E'b(a,b,c) = 0                                          ")
print("                            a * Σ[xi^3] + b * Σ[xi^2] + c * Σ[xi] = Σ[yi*xi]      ")
print("          - el desplazamiento vertical: E'c(a,b,c) = 0                            ")
print("                            a * Σ[xi^2] + b * Σ[xi] + c * n = Σ[yi]               ")
print("          Para hallar a y b, se realiza un sistema matricial de 3*3:              ")
print("          a Σ xi4 + b Σ xi3 + c Σ xi2 = Σ [xi2 * yi]    [1]                       ")
print("          a Σ xi3 + b Σ xi2 + c Σ xi  = Σ xiyi          [2]                       ")
print("          a Σ xi2 + b Σ xi  + c * n   = Σ yi            [3]                       ")
print("                                                                                  ")
print("          Podemos hacer uso de este sistema, reemplazar con los valores calculados")
print("          para este dataset y despejar los valores de a, b y c.                   ")

#  II) Examples
print("                                                                                  ")
print("**********************************************************************************")
print("*                                    EJEMPLOS                                    *")
print("**********************************************************************************")
# pares = generador_pares(1, 50)
# my_regressions(pares)

pares_ejercicio = ((1, 1), (2, 1), (3, 2), (4, 8), (5, 9), (6, 12), (7, 17), (8, 19), (9, 21), (10, 31), (11, 34), (12, 45), (13, 56), (14, 76), (15, 78), (16, 97), (17, 128), (18, 158), (19, 225), (20, 265), (21, 301), (22, 385), (23, 502), (24, 588), (25, 689), (26, 744), (27, 819), (28, 965), (29, 1053), (30, 1132), (31, 1264), (32, 1352), (33, 1450), (34, 1553), (35, 1627), (36, 1715), (37, 1795), (38, 1894), (39, 1975), (40, 2142), (41, 2208), (42, 2277), (43, 2443), (44, 2571), (45, 2669), (46, 2758), (47, 2839), (48, 2941), (49, 3031), (50, 3144), (51, 3288), (52, 3435), (53, 3607), (54, 3780), (55, 3892), (56, 4003), (57, 4127), (58, 4285), (59, 4428), (60, 4532), (61, 4681), (62, 4783), (63, 4887), (64, 5020), (65, 5208), (66, 5371), (67, 5611), (68, 5776), (69, 6034), (70, 6265), (71, 6563), (72, 6879), (73, 7134), (74, 7479), (75, 7805), (76, 8068), (77, 8371), (78, 8809), (79, 9283), (80, 9931), (81, 10649), (82, 11353), (83, 12076), (84, 12628), (85, 13228), (86, 13933), (87, 14702), (88, 15419), (89, 16214), (90, 16851), (91, 17415), (92, 18319), (93, 19268), (94, 20197), (95, 21037), (96, 22020), (97, 22794), (98, 23620), (99, 24761), (100, 25987), (101, 27373), (102, 28764), (103, 30295), (104, 31577), (105, 32785), (106, 34159), (107, 35552), (108, 37510), (109, 39570), (110, 41204), (111, 42785), (112, 44931), (113, 47216), (114, 49851), (115, 52457), (116, 55343), (117, 57744), (118, 59933), (119, 62268), (120, 64530), (121, 67197), (122, 69941), (123, 72786), (124, 75376), (125, 77815), (126, 80447), (127, 83426), (128, 87030), (129, 90694), (130, 94060), (131, 97059), (132, 100166), (133, 103265), (134, 106910), (135, 111160), (136, 114783), (137, 119301), (138, 122524), (139, 126755), (140, 130774), (141, 136118), (142, 141900), (143, 148027), (144, 153520), (145, 158334), (146, 162526), (147, 167416), (148, 173355), (149, 178996), (150, 185373), (151, 191302), (152, 196543), (153, 201919), (154, 206743), (155, 213535), (156, 220682), (157, 228195), (158, 235677), (159, 241811), (160, 246499), (161, 253868), (162, 260911), (163, 268574), (164, 276072), (165, 282437), (166, 289100), (167, 294569), (168, 299126), (169, 305966))
pares_ejercicio = [(long(x), long(y)) for (x, y) in pares_ejercicio]

my_regressions(pares_ejercicio)

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
