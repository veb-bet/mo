import math
import numpy as np
import random
from typing import Callable
import scipy
from scipy import optimize

def function_input ():

    def function_is_it_number(number):
        if number.isdigit():
            number = int(number)
        else:
            number = 1234567890

        return number

    start_condition = ("\n Выберите уравнение: \n "
                            "\n 1: Функция Шаффера N2"
                            "\n 2: Функция подставка для яиц"
                            "\n 3: Функция Экли"
                            "\n Номер уравнения для решения: ")

    choice_yr = input(start_condition)

    choice_yr = function_is_it_number(choice_yr)

    iter = 0

    if choice_yr not in [1,2,3]:
        while True:
             choice_yr = input(" Вы ввели неверный номер уравнения, пожалуйста, введите номер от 1 до 3: ")

             choice_yr = function_is_it_number(choice_yr)

             iter += 1
             if (iter % 4 == 0 and choice_yr not in [1,2,3]):
                 print(start_condition)
             elif (choice_yr in [1,2,3]):
                 break

    if choice_yr in [1, 2, 3]:  # создает пустые размерные матрицы
        x = np.zeros(2, float)

    for i in range(len(x)):                      # ввод данных
        x[i] = float(input(f"X[{i}]: "))

    method_choice_condition = ("\n Выберите метод для решения: \n "
                           " 1. \"Сверхбыстрый отжиг\" \n "
                           " 2. \"Больцмановский отжиг\" \n "
                           "\n Номер метода: ")

    choice_method = input(method_choice_condition)       # выбор метода решения

    choice_method = function_is_it_number(choice_method)

    iter = 0

    if choice_method not in [1,2]:
        while True:
             choice_method = input(" Вы ввели неверный номер метода, пожалуйста, введите номер от 1 до 2: ")

             choice_method = function_is_it_number(choice_method)

             iter += 1
             if (iter % 4 == 0 and choice_method not in [1,2]):
                 print(method_choice_condition)
             elif (choice_method in [1,2]):
                 break

    return choice_yr, choice_method, x

global choice_yr, choice_method

def function (x0):
    try:
        if (choice_yr == 1):

            x, y = x0
            #x1, x2 = x
            return 0.5 + ((np.sin(x ** 2 - y ** 2) ** 2 - 0.5) / ( 1 + 0.001 * (x ** 2 + y ** 2)) ** 2)  # Функция Шаффера N2  -100 100 f(0, 0)=0

            #return 0.5 + (math.cos(math.sin(math.fabs(x**2 - y**2)))**2 - 0.5) / (1 + 0.001*(x**2 + y**2))**2      # Функция Шаффера N4

        if (choice_yr == 2):
            x, y = x0
            return -(y + 47) * np.sin(np.sqrt(np.abs(x * 0.5 + (y + 47)))) - x * np.sin( np.sqrt(np.abs(x - (y + 47))))
            #return -math.fabs( math.sin(x) * math.cos(y) * math.exp( math.fabs(1 - (x**2 + y**2)**0.5/math.pi) ) )      # Табличная функция Хольдера

        if (choice_yr == 3):
            x, y = x0
            return -20 * np.exp(-0.2 * np.sqrt((x * x + y * y) / 2)) - np.exp((np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)) / 2) + 20 + np.exp(1)  # Экли -5 5 f(0, 0)=0
            #return -math.cos(x) * math.cos(y) * math.exp( -( (x - math.pi)**2 + (y - math.pi)**2 ) )     # Функция Изома


    except NameError:
        print("\n\n Произошли непредвиденные ошибки при вычислении занчения функции.")
        exit()

# ------------------------------------------------- ОТЖИГИ -------------------------------------------------------------

def simulated_annealing(func: Callable[[np.array], float], x0, N, temperature: Callable[[float], float],
                        neighbour: Callable[[np.array, float], np.array],
                        passage: Callable[[float, float, float], float]):
    """
        Алгоритм имитации отжига
        Метод сверхбыстрого отжига
        """
    k = 1
    x = np.array(x0)
    x_optimal = x
    e_optimal = func(x_optimal)
    while k < N:
        t = temperature(k)
        x_new = neighbour(x, t)
        e_old = func(x)
        e_new = func(x_new)
        if e_new < e_old or passage(e_old, e_new, t) >= random.random():
            x = x_new

        if e_new < e_optimal:
            x_optimal = x_new
            e_optimal = e_new

        k += 1

    if func(x) < e_optimal:
        x_optimal = x

    return x_optimal

def ultrafast_annealing(func: Callable[[np.array], float], x0, M: int, N: int, t, R, b):
    x = x0
    k = 0
    while k < N:
        y_xlam = []
        f = []

        for _ in range(M):
            e = np.random.uniform(-1, 1, len(x))
            y = x + t * e / np.linalg.norm(e)
            y_xlam.append(y)
            f.append(func(y))

        min_index = np.argmin(f)
        f_min = f[min_index]

        if f_min < func(x):
            x = y_xlam[min_index]
            k += 1
        else:
            if t <= R:
                return x
            elif t > R:
                t *= b

    return x

def boltzmann_method(x0, t0, function, N=5000):

    annealing = lambda k: t0 / math.log(1. + k)
    # passage = lambda e_old, e_new, t: 1. / (1. + math.exp((e_new - e_old) / t))
    passage = lambda e_old, e_new, t: math.exp(-1. * (e_new - e_old) / t)
    neighbour = lambda x_old, t: x_old + t * np.random.standard_normal(len(x_old))
    return simulated_annealing(function, x0, N, annealing, neighbour, passage)


#def QA(x0, t0, f, N=100000):
#    annealing = lambda k: t0 / math.pow(k, 1. / len(x0))
#    passage = lambda e_old, e_new, t: math.exp(-1. * (e_new - e_old) / t)
#    neighbour = lambda x_old, t: x_old + t * np.random.standard_cauchy(1)
#    return simulated_annealing(f, x0, N, annealing, neighbour, passage)


if __name__ == '__main__':
    while True:
        choice_yr, choice_method, x0 = function_input()
        if choice_method == 1:
            answer = ultrafast_annealing(function, x0, 100, 400, 1, 0.00001, 0.1)
            #answer = QA(x0, 1, function, 100000)
        if choice_method == 2:
            answer = boltzmann_method(x0, 1., function, 100000)
            #answer = QA(x0, 1, function, 100000)
        print (f"\n Результат: \n  x: {np.round(answer, 6)} \n f(x): {np.round(function(answer), 6)} \n")

