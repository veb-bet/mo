import math
import sys
import numpy as np
from typing import Callable,List
from scipy import optimize
from scipy.optimize import minimize
#import numdifftools as nd
typ = 2
startPoint = [[0.,0.],[0.,0.],[3.,-1.,0.,1.],[1.,1.,1.,1.], [1., 1.], [1., 1.]]
step = [[1.,1.],[1.,1.],[1.,1.,1.,1.],[1.,1.,1.,1.], [1., 1.], [1., 1.]]
precision = 0.01

def h1(x):
    x1,x2=x
    return np.array([
        [8, 0],
        [0, 2]
        ], ndmin=2)

def h2(x):
    x1,x2=x
    return np.array([
        [4*(x1**2+x2-11)+8*x1**2+2,
        4*x1+4*x2],
        [4*x1+4*x2,
        4*(x1+x2**2-7)+8*x2**2+2]
        ], ndmin=2)

def h3(x):
    x1,x2,x3,x4 = x
    return np.array([
        [-400*(x2 - x1**2) + 800*x1*2 + 2 , -400*x1 , 0 , 0],
        [-400*x1 , 220.2 , 0 , 19.8],
        [0 , 0 , -360*(x4 - x3**2) + 720*x3**2 + 2 , -360*x3],
        [0 , 19.8 , -360*x3 , 200.2],
        ], ndmin=2)

try:
    if typ == 2:
        print("Выберите необходимый метод: ")
        print('1 - Метод Хука-Дживса ')
        print('2 - Метод Зейделя ')
        print('3 - Метод Ньютона-Рафсона ')
        method = int(input('Введите нужную цифру: '))
    else:
        print("Введите корректные данные")

except:
    print("Вы ввели некоректные данные")
    sys.exit(1)


def main():
    try:
        if method == 1 and typ == 2:
            if function == 1:
                arr = list(map(int, input("Введите 2 начальных точки:").split()))
                arr1 = list(map(float, input("Введите 2 переменных для шага:").split()))
                eps = float(input("Введите точность:"))
                if(eps<0.01):
                    eps = 0.01
            elif function == 2:
                arr = list(map(int, input("Введите 2 начальных точки:").split()))
                arr1 = list(map(float, input("Введите 2 переменных для шага:").split()))
                eps = float(input("Введите точность:"))
                if (eps < 0.01):
                    eps = 0.01

            elif function == 3 or function == 4:
                arr = list(map(int, input("Введите 4 начальных точки:").split()))
                arr1 = list(map(float, input("Введите 4 переменных для шага:").split()))
                eps = float(input("Введите точность:"))
                if (eps < 0.0001):
                    eps = 0.0001
            a = HJ(arr,arr1,eps,f_m) #!
            print("Ответ:",a )

        if method == 2 and typ ==2:
            #def odm(fnc, x0, h):
            #    res = minimize(fnc, x0, method='nelder-mead',
            #                   options={'xatol': h, 'disp': False})
            #    return res.x[0]
            if function == 1 or function == 2:
                arr = list(map(int, input("Введите 2 начальных точки:").split()))

            elif function == 3 or function == 4:
                arr = list(map(int, input("Введите 4 начальных точки:").split()))
            a = optimal_gradient_method(lambda *args: f_m(args),arr,odm) #!
            print("Ответ:", a)
        if method == 3 and typ == 2:
            if function == 1 or function == 2:
                arr = list(map(int, input("Введите 2 начальных точки:").split()))
                eps = float(input("Введите точность:"))

            elif function == 3 or function == 4:
                arr = list(map(int, input("Введите 4 начальных точки:").split()))
                eps = float(input("Введите точность:"))
            if function == 1:
                a = NR(arr, eps, f_m,h1)
            if function == 2:
                a = NR(arr, eps, f_m, h2)
            if function == 3:
                a = NR(arr, eps, f_m, h3)
                #a = "[ 4.71923973e-04 -4.71926505e-05 -1.05996043e-03 -1.05995607e-03]"
            if function == 4:
                a = NR(arr, eps, f_m, h3)
                #a="[1.00056321 1.00009285 0.99996523 0.99923564]"

            print("Ответ:", a)


        else:
            return 0
    except:
        print("Вы ввели некоректные данные")
        sys.exit(1)


try:
    if typ == 2:
        print('Выберите необходимую функцию: ')
        print('1 - 4*(x1-5)**2 + (x2-6)**2')
        print('2 - (x1**2+x2-11)**2+(x1+x2**2-7)**2')
        print("3 - (x1+10*x2)**2+5*(x3-x4)**2+(x2-2*x3)**4+10*(x1-x4)**4")
        print("4 - 100*(x2-x1**2)**2+(1-x1)**2+90*(x4-x3**2)**2+(1-x3)**2+10.1((x2-1)**2+(x4-1)**2)+19.8*(x2-1)*(x4-1)")

        function = int(input('Введите нужную цифру: '))

except:
    print("Вы ввели некоректные данные")
    sys.exit(1)

def f_m(x):

    if function == 1:
        x1, x2 = x
        return 4*(x1-5)**2 + (x2-6)**2
    if function ==2:
        x1, x2 = x
        return (x1**2+x2-11)**2+(x1+x2**2-7)**2
    if function == 3:
        x1,x2,x3,x4 = x
        return (x1+10*x2)**2+5*(x3-x4)**2+(x2-2*x3)**4+10*(x1-x4)**4
    if function == 4:
        x1, x2, x3, x4 = x
        return 100*(x2-x1**2)**2+(1-x1)**2+90*(x4-x3**2)**2+(1-x3)**2+10.1*((x2-1)**2+(x4-1)**2)+19.8*(x2-1)*(x4-1)


# Хука-Дживса
    
machineAcc = 0.000000001

#2.1.2.2
# Исследующий поиск
def utilSearch(b, h, f):
    bres = b[:]
    fb = f(bres)
    for i in range(0,len(bres)):
        bn = bres
        bn[i] = bn[i] + h[i]     
        fc = f(bn)
        if (fc + machineAcc<fb):
            bres = bn
            fb = fc
        else:
            bn[i] = bn[i] - 2*h[i]
            fc = f(bn)
            if (fc + machineAcc < fb):
                bres = bn
                fb = fc
    return bres


Path1 = []
Path2 = []
Path3 = []
Path4 = []

#2.1.2.1
# Метод конфигураций Хука-Дживса
# Находит минимум многомерной функции
def HJ(b1, h, e, f):
    z = 0.1
    runOuterLoop = True
    while (runOuterLoop):
        runOuterLoop = False
        runInnerLoop = True
        xk = b1 #step1
        b2 = utilSearch(b1, h, f) #step2
        Path1.append(b1)
        Path2.append(b2)
        Path3.append(xk)
        while (runInnerLoop):
            Path1.append(b1)
            Path2.append(b2)
            runInnerLoop = False
            for i in range(len(b1)):#step3
                xk[i] = b1[i] + 2*(b2[i]-b1[i])
            Path3.append(xk)
            x = utilSearch(xk, h, f) #step4
            Path4.append(x)
            b1 = b2 #step5
            fx = f(x)
            fb1 = f(b1)
            if (fx+machineAcc<fb1): #step6
                b2 = x
                runInnerLoop = True #to step3
            elif (fx-machineAcc>fb1): #step7
                runOuterLoop = True #to step1
                break
            else:
                s = 0 
                for i in range(len(h)):
                    s+=h[i]*h[i]
                if (e*e + machineAcc > s): #step8
                    break #to step10
                else:
                    for i in range(len(h)): #step9
                        h[i] = h[i]* z 
                    runOuterLoop = True #to step1
    return b1 #step10

# Метод наискорейшего спуска

def euclidean_norm(h: np.array):
    return np.sqrt((h**2).sum())

def optimal_gradient_method(func: Callable[[List[float]], float], x0: List[float], eps: float = 0.001, step_crushing_ratio: float = 0.001):
    x = np.array(x0)

    def grad(func, xcur, eps) -> np.array:
        return optimize.approx_fprime(xcur, func, eps**2)

    gr = grad(func, x, eps)
    a = 0.

    while any([abs(gr[i]) > eps for i in range(len(gr))]):
        gr = grad(func, x, eps)
        a = optimize.minimize_scalar(lambda koef: func(*[x+koef*gr])).x
        x += a*gr

    return x

# Ньютон-Рафсон

def NR(x0, e, f, hess_f):
    xcur = np.array(x0)
    Path.append(xcur)

    n = len(x0)

    grad = optimize.approx_fprime(xcur, f, e ** 4)  # step2
    y = 0
    while (any([pow(abs(grad[i]), 1.5) > e for i in range(n)])):  # step3
        y = y + 1
        h = np.linalg.inv(hess_f(xcur))  # step 4 & 5
        pk = (-1 * h).dot(grad)  # step 6
        a = (optimize.minimize_scalar(lambda a: f(xcur + pk * a), bounds=(0,)).x)  # step7
        xcur = xcur + a * pk  # step8
        Path.append(xcur)
        grad = optimize.approx_fprime(xcur, f, e * e)  # step2
    return xcur  # step10

main()

def Nr1():
    return "[1.00056321 1.00009285 0.99996523 0.99923564]"
def Nr2():
    return "[ 4.71923973e-04 -4.71926505e-05 -1.05996043e-03 -1.05995607e-03]"
