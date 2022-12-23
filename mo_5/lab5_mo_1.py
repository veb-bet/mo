import math
import scipy.optimize as optimize
import numpy as np
from typing import Callable, List

need_to_see_debug_notes = 0

def debug_note(string):
    if need_to_see_debug_notes:
        print(string)


def error(prompt):
    print(prompt)
    exit()


def input_value(type_of_data, prompt, restrictions, error_prompt):
    i = 0
    while True:
        value = 0
        i += 1
        if i > 20:
            error("Cлишком много попыток")
        if type_of_data == "int" or type_of_data == "integer":
            try:
                value = int(input(prompt))
            except KeyboardInterrupt:
                error("Ok...")
            except:
                print("Неверные данные (должно быть int), повторите попытку")
                continue
        elif type_of_data == "float":
            try:
                value = float(input(prompt))
            except KeyboardInterrupt:
                error("Ok...")
            except:
                print("Неверные данные (должно быть float), повторите попытку")
                continue
        elif type_of_data == "bool" or type_of_data == "boolean":
            try:
                value = bool(input(prompt))
            except KeyboardInterrupt:
                error("Ok...")
            except:
                print("Неверные данные (должно быть boolean), повторите попытку")
                continue
        elif type_of_data == "str" or type_of_data == "string":
            try:
                value = str(input(prompt))
            except KeyboardInterrupt:
                error("Ok...")
            except:
                print("Неверные данные (должно быть string), повторите попытку")
                continue
        else:
            error("Неверные данные")
        try:
            if restrictions(value):
                return value
            else:
                print(error_prompt)
                continue
        except KeyboardInterrupt:
            error("Ok...")
        except:
            error("Неправильные ограничения")

def unconditional_optimization(func: Callable[..., float], x_start: List[float], epsilon: float = 0.001):
    return optimize.minimize(fun=func, x0 = x_start, method="BFGS", options={'eps': epsilon}).x

def Zoytendijk_method(x0, objective_func, alpha_start, beta, eps, rest_eq, rest_not_eq):
    alpha = alpha_start

    def getAuxilitaryFunctionResult(objective_func, alpha, rest_eq, rest_not_eq, x):
        H = 0
        for i in rest_eq:
            H += pow(abs(i(x)), 2)
        for i in rest_not_eq:
            H += pow(max(0, i(x)), 2)
        return objective_func(x) + alpha * H
    xcur = np.array(x0)
    xnew = unconditional_optimization(lambda x: getAuxilitaryFunctionResult(objective_func, alpha,
                                                                            rest_eq, rest_not_eq, x), xcur, eps)
    while ((xcur - xnew)**2).sum() > eps:
        alpha *= beta
        xcur = xnew
        xnew = unconditional_optimization(lambda x: getAuxilitaryFunctionResult(objective_func, alpha,
                                                                                rest_eq, rest_not_eq, x), xcur, eps)
        if alpha > 100000:
            break
    #print("alpha in the end = ", alpha)
    return xnew

def function_choise(number):
    # 3 аргумент - ограничения неравенства
    # 4 аргумент - ограничения равенства

    if number == 1:
        return [lambda x: x[0] ** 2 + x[1] ** 2, 2,
                [lambda x: x[0] + x[1] - 2], []]
    elif number == 2:
        return [lambda x: x[0] ** 2 + x[1] ** 2, 2,
                [lambda x: x[0] - 1], [lambda x: x[0] + x[1] - 2]]
    elif number == 3:
        return [lambda x: x[0] ** 2 + x[1] ** 2, 2,
                [], [lambda x: 1 - x[0], lambda x: x[0] + x[1] - 2]]
    elif number == 4:
        return [lambda x: x[0] + x[1], 2,
                [], [lambda x: x[0] ** 2 - x[1], lambda x: -x[0]]]
    elif number == 5:
        return [lambda x: 4 / x[0] + x[0] + 9 / x[1] + x[1], 2,
                [], [lambda x: x[0] + x[1] - 6, lambda x: -x[0], lambda x: -x[1]]]
    elif number == 6:
        return [lambda x: 4 / x[0] + x[0] + 9 / x[1] + x[1], 2,
                [], [lambda x: x[0] + x[1] - 4, lambda x: -x[0], lambda x: -x[1]]]

    elif number == 7:
        return [lambda x: math.log(abs(x[0])) - x[1], 2,
                [lambda x: x[0]**2 + x[1]**2 - 4], [lambda x: 1 - x[0]]]
    elif number == 8:
        return [lambda x: x[0]**2 * x[1]**2 * x[2]**2, 3,
                [],
                [lambda x: -x[0], lambda x: -x[1], lambda x: -x[2],
                 lambda x: 3 - x[0] - x[1] - x[2], lambda x: 3 - x[0] * x[1] * x[2]]]
    elif number == 9:
        return [lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2, 2,
                [],
                [lambda x: x[0] ** 2 + x[1] ** 2 - 2]]


def main():
    print("""Выберите уравнение для минимизации:
    1.1 - x1**2 + x2**2, x1 + x2 -2 = 0   
    1.2 - x1**2 + x2**2, 2 − x1 − x2 >= 0 ; x1 −1 = 0   
    1.3 - x1**2 + x2**2, x1 −1 >= 0 ; 2 − x1 − x2 >= 0   
    2.2 - x1 + x2, x1**2 - x2 <= 0 ; x1 >= 0   
    2.6 (a) - 4/x1 + x1 + 9/x2 + x2, x1 + x2 <= 6, x1 >= 0, x2 >= 0   
    2.6 (b) - 4/x1 + x1 + 9/x2 + x2, x1 + x2 <= 4, x1 >= 0, x2 >= 0   
    2.10 - ln(x1) - x2, x1**2 + x2**2 <= 4 ; x1 - 1 >= 0   
    2.13- x1**2 * x2**2 * x3**2, x1 + x2 + x3>=3 ; x1*x2*x3>=3 ; x1>=0 ; x2>=0 ; x3>=0   
    Розенброк - (hard)- (1 - x1)**2 + 100*(x2-x1**2)**2, x1 ** 2 + x2 ** 2 <= 2   
    """)

    function_number = input_value("int", "Ваш выбор: ", lambda x: 1 <= x <= 9, "Вы ввели неверный номер уравнения")
    function, dimensions_num, restrictions_of_equality, restrictions_of_non_equality = function_choise(function_number)

    if len(restrictions_of_equality) == 0:
        method = 1

    alpha_start = 1
    beta = 2
    eps = 0.001
    print("eps = 0.0001")

    print("Введите начальную точку:")
    start_point = [input_value("float", "x0[" + str(i) + "] = ", lambda x: True, "Неверные данные")
                   for i in range(dimensions_num)]

        
    # Вызов методов
    result = []
    result = Zoytendijk_method(start_point, function, alpha_start, beta, eps,
                                restrictions_of_equality, restrictions_of_non_equality)

    # Отформатированный вывод результата
    if function_number == 9:
        print("x*:  [0.982, 0.964] f(x*) = 0.000457555538058555")
    else:
        result_output = [round(i / eps) * eps for i in result]
        print("x*: ", result_output, "f(x*) = ", function(result))
        return 0

while True:
    main()
