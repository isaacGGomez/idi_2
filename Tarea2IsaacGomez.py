from sympy.abc import x, y, z
from sympy import sin, ln, Matrix, Symbol, pprint

def newton_raphson(ecuaciones,valores_iniciales):
    ejercicios = ecuaciones
    ecuaciones = Matrix(ecuaciones)
    x_i = Matrix(list(valores_iniciales.values())).evalf(5)
    variables =  list(valores_iniciales.keys())
    E  = 10**-4 
    Jac_inv = ecuaciones.jacobian(variables).inv()
    Jac_det = ecuaciones.jacobian(variables).det()
    print("-------------------------------------------------")
    print(f'\nEcuaciones: {ejercicios}')
    step = 0
    while True:
        subs_dic = dict(zip(variables, x_i))
        Jac_det_ev = Jac_det.subs(subs_dic)
        if (Jac_det_ev == 0):
            print("El determinante del Jacobiano es 0, no es posible realizar operaciones")
            break
        functions_ev = ecuaciones.subs(subs_dic)
        norma = sum(abs(functions_ev))
        cumple_exactitud = norma < E
        if(cumple_exactitud):  
                print(f"Solucion: {subs_dic}. \nIteraciones: {step}")
                print(f"Valores iniciales: {valores_iniciales}")
                print(f"El valor encontrado {norma} es menor que el error {E}")
                print("\n-------------------------------------------------")
                break

        Jac_inv_eval = Jac_inv.subs(subs_dic)
        # Algoritmo Newton-Raphson. Limitado a cifras significativas dadas.
        x_i = (x_i - (Jac_inv_eval * functions_ev)).evalf(5)
        # print(f"x_i: {x_i}") # En caso de necesitar ver los valores de las iteraciones, descomentar.
        step+=1

Ejercicio_1 = [x**2+y-1,x-2*y**2]
Ejercicio_2 = [x**2-10*x+y**2+5, x*y**2+x-10*y+8]
Ejercicio_3 = [x*sin(y)-1, x**2+y**2-4]
Ejercicio_4 = [y**2*ln(x)-3, y-x**2]
Ejercicio_5 = [x+y-z+2, x**2+y, z-y**2-1]

newton_raphson(Ejercicio_1, {x:0.7,y:0.6})
newton_raphson(Ejercicio_1, {x:1.36,y:-0.83})
newton_raphson(Ejercicio_2, {x:0.7,y:1.0})
newton_raphson(Ejercicio_2, {x:2.1,y:3.4})
newton_raphson(Ejercicio_3, {x:1.0,y:2.0})
newton_raphson(Ejercicio_3, {x:2.0,y:0.5})
newton_raphson(Ejercicio_3, {x:-1.1,y:-1.8})
newton_raphson(Ejercicio_3, {x:-2.0,y:-0.6})
newton_raphson(Ejercicio_4, {x:1.6,y:2.6})
newton_raphson(Ejercicio_5, {x:0.9,y:-0.9,z:1.9})
newton_raphson(Ejercicio_5, {x:-0.5,y:-0.3,z:1.1})
#%%

#%%
