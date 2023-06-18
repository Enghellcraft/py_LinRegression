# PENDIENTES
# [ ] Mejorar print de funciones y derivadas en consola
# [ ] Arreglar error por long_scalars

#
# TP Métodos Numéricos - 2023
# Alumnos: 
#          • Bianchi, Guillermo
#          • Martin, Denise
#          • Nava, Alejandro

# Profesores: para poder correr adecuadamente el programa es necesario tenes instaladas las
#             bibliotecas de sympy, numpy, pandas, seaborn y matplotlib.
#             Se puede ver el código comentado pero con "play" toda la teoría y práctica aplicada


# ------------------------------------------------------------------------------------------------------------
# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sym
from scipy.optimize import curve_fit
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
import math


# ------------------------------------------------------------------------------------------------------------
# Funs
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
def create_f_sym_lin(coeff):
    a_lin, b_lin = coeff
    f_lin = np.poly1d((a_lin, b_lin))
    f_lin_str = f'({a_lin})*x+({b_lin})'
    return f_lin, f_lin_str


def create_f_sym_cuad(coeff):
    a, b, c = coeff
    f_cuad_str = f"({round(a, 4)})*x^2+({round(b, 4)})*x+({round(c, 4)})"
    x = sym.symbols('x')
    f_cuad_sym = sym.sympify(f_cuad_str)
    f_cuad_sym = sym.lambdify(x, f_cuad_sym)
    return f_cuad_sym, f_cuad_str


def create_f_sym_poly(a_exp, b_exp):
    x = sym.symbols('x')
    f_sym = round(b_exp, 2) * (x ** round(a_exp, 2))
    f = sym.lambdify(x, f_sym)
    f_str = str(sym.sympify(f_sym)).replace("**", "^")
    return f, f_str


def create_f_sym_exponential_euler(a_exp, b_exp):
    x = sym.symbols('x')
    f_sym_euler = round(b_exp, 2) * sym.exp(round(a_exp, 2) * x)
    # print('f_sym_euler', f_sym_euler)
    f = sym.lambdify(x, f_sym_euler)
    f_str = f"{round(b_exp, 2)}*e^({round(a_exp, 2)}*x)"
    return f, f_str

def create_f_sym_exponential_eulerless(a_exp, b_exp):
    x = sym.symbols('x')
    f_sym_eulerless = round(b_exp, 2) * (round(a_exp, 2) ** x)
    f = sym.lambdify(x, f_sym_eulerless)
    f_str = f"{round(b_exp, 2)}*{round(a_exp, 2)}^x"
    return f, f_str

# Best Result Search
def find_best_fit(results_list):
    # Encuentra el valor máximo en la lista
    max_value = max(results_list, key=lambda x: x[1])[1]
    
    # Filtra los elementos que tengan el valor máximo
    best_fits = [item for item in results_list if item[1] == max_value]
    
    return best_fits

# Prints
# All Regressions
def my_regressions(pares):
    
    # Dataset
    X, Y = separador_pares_x_y(pares)
    len_pares = len(X)
    print(f"Los pares ordenados son:\n {pares}")
    print(X)
    print(Y)

    print("===[REGRESIÓN LINEAL]===")
    # Cálculo de 'a'(pendiente) y 'b'(ordenada de origen) de la ecuacion 'y = ax + b' para encontrar
    # la mejor recta que se aproxime a todos los puntos, con el minimo valor de error posible
    a_lin, b_lin = find_ab_lin_reg(X, Y)
    error_cuad_lineal = np.sum((a_lin * X + b_lin - Y) ** 2)
    print(f"Error cuadratico: {error_cuad_lineal:.2f}")

    # Cálculo R^2 de Lineal
    # Formato de función
    def lin_func(x, a, b):
        return a * x + b

    # Calce de función lineal en el dataset
    a, b = find_ab_lin_reg(X, Y)
    # Cálculo de R^2
    residuals = Y - lin_func(X, a, b)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r_lineal = 1 - (ss_res / ss_tot)
    print(f"\nR para lineal es: {r_lineal:.2f} \n")

    # Plot Funcion Lineal
    f_lin, f_lin_str = create_f_sym_lin((a_lin, b_lin))
    print("La expresion de la función lineal es:")
    print(np.poly1d((a_lin, b_lin)))
    f_name = "Regresion lineal"
    regressions_graph_unit(X, Y, f_lin, r_lineal, f_name, 'orange')

    # Guardar resultado en una lista que contiene otra lista de:
    # La funcion, el r y el nombre
    results_list = []
    results_list.append([f_lin_str, round(r_lineal, 2), f_name])

    print()

    # ***************************************************************************************************
    print("===[REGRESIÓN CUADRATICA]===")

    # Calculo de los tres coeficientes del sistema cuadratico. Al ser más complejo se resuelve via numpy.
    cuad_abc_mat = find_abc_quad_reg(X, Y)

    error_cuad_cuad = np.sum((cuad_abc_mat[0] * X ** 2 + cuad_abc_mat[1] * X + cuad_abc_mat[2] - Y) ** 2)
    print(f"\nError cuadratico: {error_cuad_cuad:.2f}")

    # Cálculo R^2 de Cuadrática
    # Formato de función
    def quad_func(x, a, b, c):
        return a * x ** 2 + b * x + c

    # Calce de función cuadrática en el dataset
    a, b, c = find_abc_quad_reg(X, Y)
    popt = [a, b, c]
    # Cálculo de R^2
    residuals = Y - quad_func(X, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r_cuad = 1 - (ss_res / ss_tot)
    print(f"\nR para cuadrática es: {r_cuad:.2f} \n")

    # Plot Funcion Cuadratica
    f_cuad, f_cuad_str = create_f_sym_cuad(cuad_abc_mat)
    # TODO: VER MANERA DE HACER FUNCION QUE DEVUELVA LA EVALUAUBLE Y UN STRING QUE best_fit_graph PUEDA DERIVAR Y GRAFICAR OK
    print("La expresion de la función cuadrática es:")
    print(np.poly1d(cuad_abc_mat))
    f_name = "Regresion cuadrática"
    regressions_graph_unit(X, Y, f_cuad, r_cuad, f_name, 'purple')

    # Guardar resultado
    results_list.append([f_cuad_str, round(r_cuad, 2), f_name])
    print()

    # ***************************************************************************************************
    print("                                                                                  ")
    print("===[REGRESIÓN POLINOMICA: y = b * x^a]===")

    # Intento de resolver exponencial (y = b * x^a) con la funcion modularizada
    # Resuelvo ln(y) = ln(b) + a * ln(x)
    a_poly, ln_b_poly = find_ab_lin_reg(np.log(X), np.log(Y), X_name="ln(X)", Y_name="ln(Y)", b_name="ln(b)")
    # Luego potencio con euler para obtener los valores que quiero
    b_poly = np.exp(ln_b_poly)
    print(f"Valor de 'b': {round(b_poly, 2)}")
    error_cuad_poly = np.sum(b_poly * (X ** a_poly))
    print(f"\nError cuadratico: {error_cuad_poly:.2f}")

    # Cálculo R^2 de polinómica
    # Formato de función
    def poly_func(x, a, b):
        return b * x ** a
    # Calce de función polinómica en el dataset
    a_exp, ln_b_exp = find_ab_lin_reg(np.log(X), np.log(Y), X_name="ln(X)", Y_name="ln(Y)", b_name="ln(b)")
    b_exp = np.exp(ln_b_exp)
    popt = [a_exp, b_exp]
    # Cálculo de R^2
    residuals = Y - poly_func(X, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r_poly = 1 - (ss_res / ss_tot)
    print(f"\nR para polinómica es: {r_poly:.2f} \n")

    # Plot Funcion Exponencial
    f_poly, f_poly_str = create_f_sym_poly(a_poly, b_poly)
    print("La expresion de la función polinomica es:")
    print(f_poly_str)
    f_name = "Regresion polinómica"
    regressions_graph_unit(X, Y, f_poly, r_poly, f_name, 'sienna')

    # Guardar resultado
    results_list.append([f_poly_str, round(r_poly, 2), f_name])

    print()

    # ***************************************************************************************************

    print("===[REGRESIÓN EXPONENCIAL EULER: y = b * e^(a*x)]===")

    # Intento de resolver exponencial ( y = b * e^(ax) ).
    a_exp_euler, ln_b_exp_euler = find_ab_lin_reg(X, np.log(Y), Y_name="ln(Y)", b_name="ln(b)")
    b_exp_euler = np.exp(ln_b_exp_euler)
    print(f"Valor de 'b': {b_exp_euler:.2f}")
    error_cuad_exp_euler = np.sum(b_exp_euler * np.exp(a_exp_euler * X))

    # Cálculo R^2 de Exponencial Euler
    # Formato de función
    def exp_euler_func(x, a, b):
        return b * np.exp(a * x)

    # Calce de función exponencial Euler en el dataset
    ln_Y = np.log(Y)
    a_exp_euler, ln_b_exp_euler = np.polyfit(X, ln_Y, 1)
    b_exp_euler = np.exp(ln_b_exp_euler)
    popt, _ = curve_fit(exp_euler_func, X, Y, p0=[a_exp_euler, b_exp_euler])
    # Cálculo R^2 
    residuals = Y - exp_euler_func(X, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r_exp_euler = 1 - (ss_res / ss_tot)
    print(f"\rR para exponencial Euler es: {r_exp_euler:.2f}\n")

    # Plot Funcion Exponencial de Euler
    f_exp_euler, f_exp_euler_str = create_f_sym_exponential_euler(a_exp_euler, b_exp_euler)
    print("La expresion de la función exponencial Euler es:")
    f_name = "Regresion exponencial Euler"
    print(f_exp_euler_str)
    regressions_graph_unit(X, Y, f_exp_euler, r_exp_euler, f_name, 'tomato')

    # Guardar resultado
    results_list.append([f_exp_euler_str, round(r_exp_euler, 2), f_name])
    print()

    # ***************************************************************************************************
    print("===[REGRESIÓN EXPONENCIAL SIN EULER: y = b * a^x]===")

    ln_a_exp_eulerless, ln_b_exp_eulerless = find_ab_lin_reg(
        X,
        np.log(Y),
        Y_name="ln(Y)",
        a_name="ln(a)",
        b_name="ln(b)"
    )
    a_exp_eulerless = np.exp(ln_a_exp_eulerless)
    b_exp_eulerless = np.exp(ln_b_exp_eulerless)
    print(f"Valor de 'a': {a_exp_eulerless:.2f}")
    print(f"Valor de 'b': {b_exp_eulerless:.2f}")
    error_cuad_exp_eulerless = np.sum(b_exp_eulerless * (a_exp_eulerless ** X))

    # Cálculo R^2 de Exponencial sin Euler
    # Formato de función
    def exp_func(x, a, b):
        return b * (a ** x)
    popt, _ = curve_fit(exp_func, X, Y, p0=[a_exp_eulerless, b_exp_eulerless])
    # Cálculo R^2
    residuals = Y - exp_func(X, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r_exp_eulerless = 1 - (ss_res / ss_tot)
    print(f"\rR para exponencial sin Euler es: {r_exp_eulerless:.2f}\n")

    # Plot Funcion Exponencial de Euler
    f_exp_eulerless, f_exp_eulerless_str = create_f_sym_exponential_eulerless(a_exp_eulerless, b_exp_eulerless)
    f_name = "Regresion exponencial sin Euler"
    print("La expresion de la función exponencial sin Euler es:")
    print(f_exp_eulerless_str)
    regressions_graph_unit(X, Y, f_exp_eulerless, r_exp_eulerless, f_name, 'indigo')
    
    # Guardar resultado
    results_list.append([f_exp_eulerless_str, round(r_exp_eulerless, 2), f_name])
    print()

    # ***************************************************************************************************

    regressions_graph(
        X, Y,
        (a_lin, b_lin), r_lineal,
        cuad_abc_mat, r_cuad,
        (a_poly, b_poly), r_poly,
        f_exp_euler, r_exp_euler, a_exp_euler, b_exp_euler,
        f_exp_eulerless, r_exp_eulerless, a_exp_eulerless, b_exp_eulerless
    )

    # Evaluacion de la funcion con mejor calce
    # print(results_list)
    bests_fits = find_best_fit(results_list)
    # print(bests_fits)

    for e in bests_fits:
        f_best_fit, r_best_fit, best_fit_name = e

        print(f"• Mejor calce con:\n* {best_fit_name}\nFunción = {f_best_fit}\nR = {r_best_fit:.2f}\n")
        best_fit_graph(X, Y, f_best_fit, r_best_fit, best_fit_name)
        
    # Doubbling Time
    xf = Y[-1]
    x0 = Y[0]
    t = X[-1]
    growth_rate = math.log(xf / x0) / t
    doubling_time = math.log(2) / math.log(1 + growth_rate)
    print(f"\nEl tiempo de Duplicación es: {doubling_time:.2f} " )

# ------------------------------------------------------------------------------------------------------------
# Plots
# Linear Regression
def regressions_graph(X, Y,
                      ab_lineal, r_lineal,
                      cuad_mat, r_cuad,
                      exp_mat, r_exp,
                      euler_lambda_exp, r_euler, a_euler, b_euler,
                      eulerless_lambda_exp, r_eulerless, a_eulerless, b_eulerless):
    
    plt.figure(figsize=(10, 6))
    plt.plot(X, Y, "o", label="Dataset", color='turquoise')

    a, b = ab_lineal
    label_lin = f"Regresión Lineal\nCoeficientes: a={round(a, 2)}, b={round(b, 2)}\n[R = {r_lineal:.2f}]"
    plt.plot(X, a * X + b, label=label_lin, color='orange')

    a_cuad, b_cuad, c_cuad = cuad_mat
    label_cuad = f"Regresión Cuadrática\nCoeficientes: a={round(a_cuad, 2)}, b={round(b_cuad, 2)}, c={round(c_cuad, 2)}\n[R = {r_cuad:.2f}]"
    plt.plot(X, a_cuad * (X ** 2) + b_cuad * X + c_cuad, label=label_cuad, color='purple')

    a_exp, b_exp = exp_mat
    label_exp = f"Regresión Polinómica\nCoeficientes: a={round(a_exp, 2)}, b={round(b_exp, 2)}\n[R = {r_exp:.2f}]"
    plt.plot(X, b_exp * (X ** a_exp), label=label_exp, color='sienna')

    label_euler = f"Regresión Exp. (Euler)\nCoeficientes: a={round(a_euler, 2)}, b={round(b_euler, 2)}\n[R = {r_euler:.2f}]"
    plt.plot(X, euler_lambda_exp(X), label=label_euler, color='tomato')

    label_eulerless = f"Regresión Exp. (Sin Euler)\nCoeficientes: a={round(a_eulerless, 2)}, b={round(b_eulerless, 2)}\n[R = {r_eulerless:.2f}]"
    plt.plot(X, eulerless_lambda_exp(X), label=label_eulerless, color='indigo')

    plt.ylim(0, Y.max() * 1.1)
    yax = plt.gca().yaxis
    for item in yax.get_ticklabels():
        item.set_rotation(45)

    plt.xlabel("Días", fontweight='bold')
    plt.ylabel("Acumulación de individuos infectados", fontweight='bold')
    plt.legend()
    plt.title("Grafico de cuadrados mínimos")
    plt.grid()
    # plt.legend(loc='upper right', bbox_to_anchor=(1.2, 0.6))
    # plt.tight_layout()
    plt.show()


def regressions_graph_unit(X, Y, func, r, msg, _color):
    plt.title(msg)

    plt.plot(X, Y, 'o', color='turquoise', markersize=5, label="Dataset")
    plt.plot(X, func(X), color=_color, linestyle='-', linewidth=2, label=msg + f" [R = {r:.2f}]")

    plt.xlabel("Días", fontweight='bold')
    plt.ylabel("Acumulación de individuos infectados", fontweight='bold')
    plt.legend()

    yax = plt.gca().yaxis
    for item in yax.get_ticklabels():
        item.set_rotation(45)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.ylim(0, Y.max() * 1.1)
    plt.show()


def best_fit_graph(X, Y, func, r, f_name_str):  
    plt.title('Funcion con mejor FIT para el Dataset\n' + f_name_str)

    x = sym.symbols('x')
    func_str = str(func).replace("^", "**")
    # print(func_str)
    # func_sym = sym.sympify(func_str, evaluate=False)
    func_lamb = sym.lambdify(x, func_str)
    func_first_diff = sym.diff(func_lamb(x), x)
    func_first_diff_lamb = sym.lambdify(x, func_first_diff.doit())
    func_second_diff = sym.diff(func_first_diff_lamb(x), x)
    func_second_diff_lamb = sym.lambdify(x, func_second_diff.doit())

    print(f"\nLa primera derivada de la {f_name_str} es:")
    print(f"{(func_first_diff).evalf(n=2)}")
    print(f"\nLa segunda derivada de la {f_name_str} es:")
    print(f"{(func_second_diff).evalf(n=2)}")

    # se evalua si la funcion es graficable
    def have_x(func):
        return 'x' in str(func) 
    
    plt.plot(X, Y, 'o', color='turquoise', markersize=5, label="Dataset")
    plt.plot(X, func_lamb(X), color='forestgreen', linestyle='-', linewidth=2, label=f_name_str + f" [r = {r:.2f}]")
    if have_x(func_first_diff):
        plt.plot(X, func_first_diff_lamb(X), color='darkorange', linewidth=2, label='Primera Derivada')
    if have_x(func_second_diff):
        plt.plot(X, func_second_diff_lamb(X), color='lightcoral', linewidth=2, label='Segunda Derivada')


    plt.xlabel("Días", fontweight='bold')
    plt.ylabel("Acumulación de individuos infectados", fontweight='bold')
    plt.legend()

    yax = plt.gca().yaxis
    for item in yax.get_ticklabels():
        item.set_rotation(45)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.ylim(0, Y.max() * 1.1)
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
print("  Lograr regresión lineal, polinómica y exponencial de un set de datos dados      ")
print("                                                                                  ")
print("**********************************************************************************")
print("*                                   CONSIGNAS                                    *")
print("**********************************************************************************")
print("                                                                                  ")
print("  Dado un dataset provisto en excell sobre individuos invfectados a lo largo de n días,")
print("  definir el tipo de relación, realizando reagresión lineal, polinómica y exponencial.")
print("  Graficar con los datos las curvas y mostrar en ellas 'a', 'b' y R.              ")
print("  De la regresión con mejor calce calcular las derivadas primera y segunda.       ")
print("  Calcular el tiempo de duplicación.                                              ")
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
print("                                                                                  ")
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
print("                     ********* REGRESION POLINOMICA *********                     ")
print("                                                                                  ")
print(" Modela una relación de forma similar a la cuadrática, pero al tener un grado     ")
print(" mayor permite una aproximación ligeramente distinta. Esto podría traducirse      ")
print(" (aunque no es garantizado) en un error menor al de la mencionada previamente.    ")
print("                                                                                  ")
print(" Para resolverla de una manera más comoda, removiendo de manera más sencilla los  ")
print(" valores a y b luego de derivar, hacemos uso de linealizar la formula. Esto lo    ")
print(" haremos mediante el logaritmo natural, aprovechando que con esto podemos reducir ")
print(" el exponente de x^a a un mero producto, mucho mas facil de trabajar.             ")
print("                                                                                  ")
print("                    y  = b * x^a                                                  ")
print("                 ln(y) = ln(b * x^a)                                              ")
print("                 ln(y) = ln(b) + ln(x^a)                                          ")
print("                 ln(y) = ln(b) + a * ln(x)                                        ")
print("                                                                                  ")
print("          Podemos llamar a estos terminos (X', Y') = ( ln(x), ln(y) )             ")
print("                                          (A', B') = (a, ln(b))                   ")
print("                                                                                  ")
print("          Siendo el Error Cuadrático: Σ[ a * X'i + B' - Y'i ]^2,                  ")
print("          se deriva y despeja como lineal (ver más arriba) resultando en:         ")
print("                                                                                  ")
print("          - la pendiente:                  n Σ[X'i Y'i] - Σ[X'i] Σ [Y'i]          ")
print("                                     a = _________________________________        ")
print("                                              n Σ[X'i^2] - (Σ[X'i])^2             ")
print("                                                                                  ")
print("          - la ordenada al origen:          Σ[X'i^2] Σ[Y'i] - Σ[X'i] Σ[X'iY'i]    ")
print("                                     B' = ______________________________________  ")
print("                                                 n Σ[X'i^2] - (Σ[X'i])^2          ")
print("                                                                                  ")
print("          Con esto despejamos a, pero para obtener b necesitamos volver a la      ")
print("          formula original:           B' = ln(b)                                  ")
print("                                  e^(B') = e^( ln(b) )                            ")
print("                                       e^(B') = b                                 ")
print("                                                                                  ")
print("               ********* REGRESION EXPONENCIAL (CON EULER) *********              ")
print("                                                                                  ")
print(" El método de Regresion Exponencial utiliza la ecuación: y = b * e^(a*x), donde    ")
print(" donde 'b' ≠ 0 y constante, y 'a' el modificador de la curvatura en la exponencial.")
print(" • ERROR: En este caso para minimizar el error de los puntos a la recta utilizaremos")
print("          el método de cuadrados mínimos como en el caso lineal, luego de linealizar")
print("          la función para poder despejar a y b. Esto lo hacemos aplicando logaritmo ")
print("          natural a ambos terminos                                                ")
print("                                                                                  ")
print("                       ln(y) = ln( b * e^(a*x) )                                  ")
print("                       ln(y) = ln(b) + ln( e^(a*x) )                              ")
print("                       ln(y) = ln(b) + a*x                                        ")
print("                                                                                  ")
print("          Podemos llamar a estos terminos (X', Y') = (x, ln(y)) junto a B'= ln(b) ")
print("                                                                                  ")
print("          Siendo el Error Cuadrático: Σ[ a * xi + B' - Y'i ]^2,                   ")
print("          se deriva y despeja como lineal (ver más arriba) resultando en:         ")
print("                                                                                  ")
print("          - la pendiente:                    n Σ[xi Y'i] - Σ[xi] Σ [Y'i]          ")
print("                                     a = _________________________________        ")
print("                                               n Σ[xi^2] - (Σ[xi])^2              ")
print("                                                                                  ")
print("          - la ordenada al origen:          Σ[xi^2] Σ[Y'i] - Σ[xi] Σ[xi * Y'i]    ")
print("                                     B' = ______________________________________  ")
print("                                                  n Σ[xi^2] - (Σ[xi])^2           ")
print("                                                                                  ")
print("          Con esto despejamos a, pero para obtener b necesitamos volver a la      ")
print("          formula original:           B' = ln(b)                                  ")
print("                                  e^(B') = e^( ln(b) )                            ")
print("                                       e^(B') = b                                 ")
print("                                                                                  ")
print("               ********* REGRESION EXPONENCIAL (SIN EULER) *********              ")
print("                                                                                  ")
print(" La Regresión Exponencial tambien puede expresarse de la forma:                   ")
print("                  y = a*b^x , donde 'a' ≠ 0 y constante, y 'b' la base.           ")
print("                                                                                  ")
print(" Se utiliza en casos donde los datos crecen lentamente al principio y luego muy   ")
print(" aceleradamente. Resolvemos similar al caso anterior, pero en esta ocasión los    ")
print(" reemplazos son:                                                                  ")
print("                                                                                  ")
print("                       ln(y) = ln( a * b^x )                                      ")
print("                       ln(y) = ln(a) + ln( b^x )                                  ")
print("                       ln(y) = ln(a) + x * ln(b)                                  ")
print("                       ln(y) = ln(a) + ln(b) * x                                  ")
print("                                                                                  ")
print("          Podemos llamar a estos terminos (X', Y') = (x, ln(y))                   ")
print("                                          (A', B') = (ln(a), ln(b))               ")
print("                                                                                  ")
print(" Podemos resolver la regresión lineal Y' = A' * X' + B' de la misma forma que     ")
print(" resolvimos 'y = ax + b' previamente. Una vez calculados A' y B', los convertimos ")
print(" en nuestros a y b originales de la siguiente forma:                              ")
print("                                                                                  ")
print("                          A' = ln(a)                                              ")
print("                      e^(A') = e^( ln(a) )                                        ")
print("                      e^(A') = a                                                  ")
print("                                                                                  ")
print("                          B' = ln(b)                                              ")
print("                      e^(B') = e^( ln(b) )                                        ")
print("                      e^(B') = b                                                  ")
print("                                                                                  ")
print(" Ya con estos valores calculados, podemos graficar la curva.                      ")
print("                                                                                  ")
print("                   ********* Coeficiente de Correlación *********                 ")
print("                                                                                  ")
print(" r es una medida que cuantifica cuán estrecha es la relación lineal entre dos     ")
print(" variables, es decir, determina el grado de movimiento de asociación de ambas variables.")
print(" Puede calcularse como:                                                           ")
print(" r = [n * (Σ(x * y) - (Σx) * (Σy)) / √([n * (Σ(x^2) - (Σx)^2) * (n * (Σ(y^2) - (Σy)^2)])]")
print(" • FUERZA: varía entre -1 y 1, siendo -1 indicativo de una correlación fuertemente ")
print("           negativa, 0 indica que no existe correlación, y 1 una relación fuertemente")
print("           positiva.                                                              ")
print("                                                                                  ")
print("                          ********* R Cuadrado *********                          ")
print("                                                                                  ")
print(" R es una medida estadística que representa la proporción de varianza entre la    ")
print(" variable dependiente y la independiente. Se calcula como el cuasdrado del        ")
print(" * Obtener el modelo de regresión lineal,                                         ")
print(" * Calcular los valores de predicción obtenidos por el modelo,                    ")
print(" * Obtener el error cuadrático realizando la diferencia entre los valores del     ")
print("   dataset y los de la predicción, y elevándolos al cuadrado                      ")
print(" * Calcular la varianza total restando el promedio del valor actual de cada valor ")
print("   y elevando al cuadrado la sumatoria de los mismos.                             ")
print(" * Realizar la sumatoria de los errores cuadrados obtenidos                       ")
print(" * Calcular R como el cociente entre la sumatoria de errores cuadrados y el total ")
print("   de varianza, y al resultado se lo resta de 1.                                  ")
print(" • PODER PREDICTIVO: varía entre 0 y 1, siendo 0 indicativo de que el modelo no explica")
print("                     la varianza en absoluto, y 1 explica el 100 %, del mismo.     ")
print("                                                                                  ")
print("                          ********* Derivadas *********                           ")
print("                                                                                  ")
print(" * DERIVADA PRIMERA de una función representa la tasa de cambio de la función en  ")
print("   cualquier punto dado.                                                          ")
print(" * DERIVADA SEGUNDA de una función proporciona información sobre la concavidad    ")
print("   o curvatura de la función.                                                     ")
print(" • LINEAL:                                                                        ")
print("    a) Derivada Primera: mide la tasa de cambio instantánea, que es constante para")
print("                         una función lineal, ya que la derivada da por resultado  ")
print("                         la pendiente de la misma.                                ")
print("    a) Derivada Segunda: siempre es cero. Esto significa que la gráfica de una    ")
print("                         función lineal es una línea recta sin curvatura, ya que  ")
print("                         mide la velocidad a la que cambia la pendiente de la función.")
print(" • CUADRATICA:                                                                    ")
print("    a) Derivada Primera: mide la tasa de cambio instantáneaen cualquier punto dado.")
print("                         En la forma f(x) = ax^2 + bx + c, la primera derivada está")
print("                         dada por f'(x) = 2x + b.                                 ")
print("                         Proporciona información sobre la pendiente en cuanto a si")
print("                         la función es creciente (cuando la derivada es positiva) o")
print("                         decreciente (cuando la derivada es negativa) en un punto ")
print("                         específico, y también ayuda a identificar puntos críticos.")
print("    a) Derivada Segunda: brinda información sobre la concavidad o curvatura de la función.")
print("                         La segunda derivada es f''(x) = 2a, es decir, que la     ")
print("                         es positiva (a > 0), la función cuadrática es cóncava    ")
print("                         hacia arriba, si es negativa (a < 0), la función cuadrática")
print("                         es cóncava hacia abajo. También ayuda a identificar los  ")
print("                         puntos de inflexión.                                     ")
print(" • POLINOMICA:                                                                    ")
print("    a) Derivada Primera: brinda información sobre la pendiente y el comportamiento ")
print("                         creciente o decreciente de la función en cualquier punto dado.")
print("                         Para una función polinómica de la forma: ")
print("                         f(x) = a_nx^n + a_{n-1}x^{n-1} + ... + a_1x + a_0, la primera")
print("                         derivada se obtiene diferenciando cada término con respecto a X.")
print("                         Representa la tasa de cambio de la función y puede ayudar a")
print("                         identificar puntos críticos, extremos locales e intervalos")
print("                         de comportamiento creciente o decreciente.                ")
print("    a) Derivada Segunda: brinda información sobre la concavidad o curvatura de la función.")
print("                         Se obtiene derivando la primera derivada con respecto a x,")
print("                         e indica cómo está cambiando la pendiente de la función. ")
print("                         Si la segunda derivada es positiva, la función es cóncava")
print("                         hacia arriba, y si es negativa, la función es cóncava hacia abajo.")
print(" • EXPONENCIAL:                                                                    ")
print("    a) Derivada Primera: representa la tasa de cambio de la función en cualquier punto dado. ")
print("                         Para una función exponencial de la forma f(x) = a^x,    ")
print("                         se obtiene derivando la función con respecto a x, es decir,")
print("                         la primera derivada de una función exponencial es igual a")
print("                         la función multiplicada por una constante.               ")
print("                         Esta propiedad es exclusiva de las funciones exponenciales y")
print("                         es el resultado de que la función exponencial es su propia derivada.")
print("                         Por ejemplo, si f(x) = e^x, entonces f'(x) = e^x.        ")
print("    a) Derivada Segunda: proporciona información sobre la curvatura de la función,")
print("                         en cualquier punto x, es decir, cómo evoluciona su forma ")
print("                         a medida que cambia x.                                   ")
print("                         La segunda derivada de una función exponencial es igual  ")
print("                         a la función multiplicada por una constante.             ")
print("                         Por ejemplo, si f(x) = a^x, entonces la segunda derivada ")
print("                         es f''(x) = (ln(a))^2 * a^x.                             ")
print("   A considerar que la tasa de cambio es constante (cuando la base es e) o es una ")
print("   tasa de cambio proporcional al logaritmo natural de la base (cuando la base no es e).")
print("                                                                                  ")
print("                    ********* Tiempo de Duplicación *********                     ")
print("                                                                                  ")
print(" El tiempo de duplicación se refiere a la cantidad de tiempo que tarda una cantidad")
print(" en duplicar su tamaño o valor a una tasa de crecimiento constante.               ")
print(" Se puede calcular usando varias fórmulas, pero comúnmente se utiliza la regla de 70.")
print(" La regla de 70 establece que para encontrar el tiempo de duplicación, divida el  ")
print(" número 70 por la tasa de crecimiento (expresada como porcentaje).                ")
print(" Esto estima rápidamente el tiempo de duplicación, considerando una tasa de       ")
print(" Se calcula primero la variación de crecimiento: r = ln(xf / x0) / t, y luego     ")
print(" el tiempo de duplicacion doubling time = log(2) / log(1 + r), donde xf es el valor")
print(" final, x0 el valor inicial y t el tiempo trasncurrido total.                      ")
print(" Cabe destacar que el tiempo de duplicación es típicamente calculado para los casos")
print(" de exponenciales, y que en ambos casos con o sin euler, el cálculo será igual.    ")

#  II) Solution
print("                                                                                  ")
print("**********************************************************************************")
print("*                                   SOLUCION                                     *")
print("**********************************************************************************")
print("                                                                                  ")
print(" Para este proyecto contamos con un dataset provisto donde se toman la cantidad de")
print(" días trasncurridos vs la cantidad de contagiados.")

pares_ejercicio = (
    (1, 1), (2, 1), (3, 2), (4, 8), (5, 9), (6, 12), (7, 17), (8, 19), (9, 21), (10, 31), (11, 34), (12, 45), (13, 56),
    (14, 76), (15, 78), (16, 97), (17, 128), (18, 158), (19, 225), (20, 265), (21, 301), (22, 385), (23, 502),
    (24, 588),
    (25, 689), (26, 744), (27, 819), (28, 965), (29, 1053), (30, 1132), (31, 1264), (32, 1352), (33, 1450), (34, 1553),
    (35, 1627), (36, 1715), (37, 1795), (38, 1894), (39, 1975), (40, 2142), (41, 2208), (42, 2277), (43, 2443),
    (44, 2571),
    (45, 2669), (46, 2758), (47, 2839), (48, 2941), (49, 3031), (50, 3144), (51, 3288), (52, 3435), (53, 3607),
    (54, 3780),
    (55, 3892), (56, 4003), (57, 4127), (58, 4285), (59, 4428), (60, 4532), (61, 4681), (62, 4783), (63, 4887),
    (64, 5020),
    (65, 5208), (66, 5371), (67, 5611), (68, 5776), (69, 6034), (70, 6265), (71, 6563), (72, 6879), (73, 7134),
    (74, 7479),
    (75, 7805), (76, 8068), (77, 8371), (78, 8809), (79, 9283), (80, 9931), (81, 10649), (82, 11353), (83, 12076),
    (84, 12628), (85, 13228), (86, 13933), (87, 14702), (88, 15419), (89, 16214), (90, 16851), (91, 17415), (92, 18319),
    (93, 19268), (94, 20197), (95, 21037), (96, 22020), (97, 22794), (98, 23620), (99, 24761), (100, 25987),
    (101, 27373),
    (102, 28764), (103, 30295), (104, 31577), (105, 32785), (106, 34159), (107, 35552), (108, 37510), (109, 39570),
    (110, 41204), (111, 42785), (112, 44931), (113, 47216), (114, 49851), (115, 52457), (116, 55343), (117, 57744),
    (118, 59933), (119, 62268), (120, 64530), (121, 67197), (122, 69941), (123, 72786), (124, 75376), (125, 77815),
    (126, 80447), (127, 83426), (128, 87030), (129, 90694), (130, 94060), (131, 97059), (132, 100166), (133, 103265),
    (134, 106910), (135, 111160), (136, 114783), (137, 119301), (138, 122524), (139, 126755), (140, 130774),
    (141, 136118),
    (142, 141900), (143, 148027), (144, 153520), (145, 158334), (146, 162526), (147, 167416), (148, 173355),
    (149, 178996),
    (150, 185373), (151, 191302), (152, 196543), (153, 201919), (154, 206743), (155, 213535), (156, 220682),
    (157, 228195),
    (158, 235677), (159, 241811), (160, 246499), (161, 253868), (162, 260911), (163, 268574), (164, 276072),
    (165, 282437),
    (166, 289100), (167, 294569), (168, 299126), (169, 305966))
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
print(" • La Regresión Polinómica es una extensión de la regersión lineal, como regresión")
print(" lineal múltiple, donde la relación entre las variables dependiente y la independiente")
print(" se modela al grado del polinomio no linear (este incluye la cuadrática).         ")
print("                                                                                  ")
print(" • VENTAJAS DE REGRESION POLINOMICA:                                              ")
print("   + Mejoras respecto a la relación de las variables.                             ")
print("   + Se adapta a grandes cambios y es considerablemente mas flexible.             ")
print("   + Ideal para exploración por sus curvaturas e inflexiones.                     ")
print("                                                                                  ")
print(" • DESVENTAJAS DE REGRESION POLINOMICA:                                           ")
print("   + Al ser mas flexible también puede traer mas errores de overfitting.          ")
print("   + Puede no generalizar bien fuera del dataset dado, por su gran capacidad de adaptarse al set dado.")
print("   + Uno o dos valores solos que salgan del esquema de relación del dataset,      ")
print("     podrían cambiar considerablemente el curso del polinomio resultante.         ")
print("                                                                                  ")
print("                                 ****************                                 ")
print("                                                                                  ")
print(" • La Regresión Exponencial es ideal para casos de crecimiento o decrecimiento    ")
print(" exponencial, es de técnica robusta pero no sirve para todos los sistemas.        ")
print("                                                                                  ")
print(" • VENTAJAS DE REGRESION EXPONENCIAL:                                             ")
print("   + Es excelente en casos de crecimiento o decrecimiento exponencial.            ")
print("   + Suele ser mas robusta incluso que algunos polinomios.                        ")
print("                                                                                  ")
print(" • DESVENTAJAS DE REGRESION EXPONENCIAL y EXPONENCIAL EULER:                      ")
print("   + No es bueno con casos de variacion de datos.                                 ")
print("   + No es confiable en casos que no tengas relaciones tipo exponencial.          ")
print("   + Asume continuidad de datos y peude afectar las predicciones.                 ")
print("                                                                                  ")
print("                                 ****************                                 ")
print("                                                                                  ")
print(" Realizando las regresiones lineal, cuadrática, polinómica, exponencial y         ")
print(" exponencial euler, se puede observar que en crecimientos biológicos hay una      ")
print(" notable relación exponencial. En este caso el número de infectados por día,      ")
print(" tiene un R = 1 en casos de exponencial y exponencial euler. Si bien las gráficas ")
print(" no son iguales, sus corrimientos respecto del data set son equivalentes y por ello")
print(" ambas son las más adecuadas para explicar la curva de contagios.                 ")
print("                                                                                  ")
print("                                 ****************                                 ")
print("                                                                                  ")
print(" Las derivadas se realizaron sobre los dos mejores calces que fueron la exponencial")
print(" y la exponencial euler. Es visible en las gráficas, como la tasa de cambio de la ")
print(" derivada primera acompaña a la curva exponencial, ya que en la medida que avanza  ")
print(" mayor es su cambio en la curvatura (típico de exponencial). En el caso de la     ")
print(" derivada segunda su curvatura se ve menos pronunciada ya que muestra la evolución")
print(" de la curvatura de la función obtenida. En ambas exponenciales las derivadas tienen")
print(" relación con los coeficientes 'a' para la expoenencial y 'e' para la exponencial euler.")
print("                                                                                  ")
print("                                 ****************                                 ")
print("                                                                                  ")
print(" El tiempo de duplicación obtenido es de 9.62, lo que implica que el número de    ")
print(" contagiados se duplica cada 9.62 días.                                           ")
print("                                                                                  ")
print("                                 ****************                                 ")
print("                                                                                  ")
print(" • NOTA: Es importante tener en cuenta que la Regla del 70 es una aproximación y  ")
print("         funciona mejor para tasas de crecimiento por debajo del 15 %. Para tasas ")
print("         de crecimiento más altas, se pueden requerir métodos y fórmulas          ") 
print("         más precisos.                                                            ") 
print("                                                                                  ")


