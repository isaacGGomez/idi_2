# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 10:15:47 2022

@author: Enrique Rojas
"""

from sympy import  Matrix, diff, sin, Symbol
cifras_significativas = 4
max_step = 1000
E = 10**-3

x = Symbol("x")
y = Symbol("y")
z = Symbol("z")


variables = [x,y,z]


def gradiente(function, x0, y0, z0 ,alpha, desc):
    x0 = x0
    y0 = y0
    z0 = z0
    step = 0
    grad_tipo = "Descendente" if desc else "Ascendente"
    resultado_tipo = "Minimo" if desc else "Maximo"
    sol = Matrix([x0,y0,z0])
    gradientes = []
    for variable in variables:
        gradientes.append(diff(function, variable))
    gradientes = Matrix(gradientes)
    while True:
        step += 1
        error = 0
        for i in range(len(gradientes)):
            grad_eval =  gradientes[i].evalf(cifras_significativas, subs = {x:sol[0],y:sol[1],z:sol[2]})#gradientes.subs({x:x0,y:y0,z:z0})
            error += abs(grad_eval)
            if desc:
                sol[i] = (sol[i]-alpha*grad_eval).evalf(cifras_significativas)
            else:
                sol[i] = (sol[i]+alpha*grad_eval).evalf(cifras_significativas)
            if(E > error or step == max_step):
    
                return  x0, y0, z0, function, sol.tolist()[0],sol.tolist()[1],sol.tolist()[2], step, error.evalf(cifras_significativas)

def resultados(gradiente):
    
    print('========= Gradiente ===========\n')
    print("Valores iniciales x=%s, y=%s, z=%s  para  %s \nx= %s y= %s z= %s\nen %s iteraciones, error =%s " % gradiente)
    print('................................\n')

        

f1 = x**4 - 3*x**3 + 2
f2 = 5*x**6 + 21*x**5 - 180*x**4 + 115*x**3 + 750*x**2 - 1260*x + 10
f3 = x**2 - 24*x + y**2 - 10*y
f4 = x*y + 1/x + 1/y
f5 = sin(x) + sin(y) + sin(x+y) # 0 <= pi <= 2 pi, 0 <= y <= 2 pi
f6 = x**2 + y**2 + z**2 + 1
f7 = 3*x**2 + 4*y**2 + z**2 - 9*x*y*z
f8 = x**4 + y**4 + z**4 + x*y*z

resultados(gradiente(f1,2,0,0,0.025,True))
resultados(gradiente(f2,-8,0,0,0.00001,True))
resultados(gradiente(f2,-1,0,0,10**-4,False))
resultados(gradiente(f2,0.9,0,0,10**-3,True))
resultados(gradiente(f3,13,6,0,0.5,True))
resultados(gradiente(f4,0.5,0.5,0,0.02,True))
resultados(gradiente(f5,1,1,0,0.2,False))
resultados(gradiente(f6,0.5,0.5,0.5,6**-1,True))
resultados(gradiente(f7,0.1,0.1,0.1,0.0059,True))
resultados(gradiente(f8,-1,-1,-1,0.025,True))


