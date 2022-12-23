#import numpy as np
import math

# Анализируемые функции
def function(x,n):
    if n == 1: return pow((x-1),2)
    if n == 2: return 4*pow(x,3) - 8*pow(x,2) - 11*x + 5
    if n == 3:
        if x == 0:
            print('Программа вылетит, поменяйте x0')
            quit()
        else:
            return x+(3/(x**2))
    if n == 4:
        if x == 2.0 or x == -2.0:
            print('Программа вылетит, поменяйте x0')
            quit()
        else:
            return (x+2.5) / (4-pow(x,2))
    if n == 5:
        return -math.sin(x)-(math.sin(3*x)/3)
    if n == 6: return -2*math.sin(x) - math.sin(2*x) - 2*math.sin(3*x)/3


def DSK(x0, h, n):
    a = 0  
    b = 0
    k = 0
    incorrect_flag = 0
    
    if function(x0 + k*h, n) >= function(x0 + (k+1)*h, n):
        a = x0
        k += 1
        while True:
            if k>100:
                incorrect_flag = 1
                break
            if function(x0 + k*h, n) < function(x0 + (k+1)*h, n):
                b = x0 + (k+1)*h
                break
            else:
                a = x0 + k*h
                k += 1
    elif function(x0 - k*h, n) >= function(x0 - (k+1)*h, n):
        b = x0
        k += 1
        while True:
            if k > 100:
                incorrect_flag = 1
                break
            if function(x0 - k * h, n) < function(x0 - (k+1)*h, n):
                a = x0 - (k + 1) * h
                break
            else:
                b = x0 - k * h
                k += 1
    else:
        a = x0 - h
        b = x0 + h

    if incorrect_flag == 0:
        print('[a,b] = [',a,', ',b,']')
        return [a,b]
    else: return 'Incorrect'


# Choose method
print('Методы:')
print('1. Дихотомии')
print('2. Пауэлла')
#print('3. Хука-Дживса')

method = int(input('Выбери метод: '))
if method > 2 or method < 1:
    print('Такого метода нет')
    method = int(input('Выбери метод: '))
# Input
f_N = int(input('Введите номер функции (от 1 до 6): '))
while (f_N > 6 or f_N < 1):
    print('Такой функции нет')
    f_N = int(input('Введите номер функции (от 1 до 6): '))

x0 = float(input('Введите x0: '))
h = float(input('Введите h>0: '))
if (h <= 0):
    print('Шаг должен быть больше 0')
    h = float(input('Введите h>0: '))
if (method == 2 and f_N == 5):
    method = 1
if (method == 2 and f_N == 6):
    method = 1
#if (f_N == 5 and (x0 > 0.9 and x0 < 3)):
#   h < 0.7
#   print('Шаг должен быть меньше 0.7, иначе программа вылетит')
#   h = float(input('Введите h>0: '))
#if (f_N == 5 and (x0 > 2.8 and x0 < 6)):
#   h < 0.7
#   print('Шаг должен быть меньше 0.7, иначе программа вылетит')
#   h = float(input('Введите h>0: '))
segment = DSK(x0, h, f_N)
if segment == 'Incorrect':
    print('Введите другие x0 и h')
    method = -1
else:
    a = segment[0]
    b = segment[1]

# Деление пополам
if method == 1:
    eps = float(input('eps(>0) = '))
    if (eps <= 0):
        print('Параметр точности поиска должен быть больше 0')
        eps = float(input('eps(>0) = '))
    delta = eps / 10
    while (b - a) / 2 >= eps:
        x1 = (a + b) / 2 - delta
        x2 = (a + b) / 2 + delta
        if function(x1, f_N) <= function(x2, f_N):
            b = x2
        else:
            a = x1
    rezult = (a + b) / 2
    print('min = ', rezult)


# Метод Пауэлла
if method == 2:
    eps = float(input('eps(>0) = '))
    if (eps <= 0):
        print('Параметр точности поиска должен быть больше 0')
        eps = float(input('eps(>0) = '))
    delta = eps / 10
    # step 1
    x1 = a
    x2 = (a + b) / 2
    x3 = b

    X = [x1, x2, x3]
    # step 2
    iteration = 0
    while (True):
        min_x = X[0]
        for i in X:
            if (function(i, f_N) < function(min_x, f_N)):
                min_x = i
        # step 3
        num = (X[1] ** 2 - X[2] ** 2) * function(X[0], f_N) + (X[2] ** 2 - X[0] ** 2) * function(X[1], f_N) + (
                    X[0] ** 2 - X[1] ** 2) * function(X[2], f_N)
        denum = (X[1] - X[2]) * function(X[0], f_N) + (X[2] - X[0]) * function(X[1], f_N) + (X[0] - X[1]) * function(X[2], f_N)
        X.append(0.5 * (num / denum))
        # step 4
        if (abs(X[3] - min_x) <= eps):
            break
        # step 5
        X.sort()
        # step 6
        max_x = 0
        for i in range(4):
            if (function(X[i], f_N) > function(X[max_x], f_N)):
                max_x = i
        X.pop(max_x)
        if iteration == 10:
            break
        iteration += 1
    # step 7
    rezult2 = X[-1]
    print(' min = ', (X[2] - min_x))
