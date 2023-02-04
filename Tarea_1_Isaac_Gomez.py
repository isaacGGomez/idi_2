# %%
import sympy as sp
def newton_rapson(func, x0,exactitud):
    valin = x0
    funcder = sp.diff(func)
    x1 = x0 - func.evalf(subs={x: x0}) / funcder.evalf(subs={x: x0})
    t = 1
    dif = abs(x1 - x0)
    while dif > exactitud:
        x0 = x1
        x1 = x0 - func.evalf(subs={x: x0}) / funcder.evalf(subs={x: x0})
        t += 1
        dif = abs(x1-x0)
    if abs(x1) < 1:
        x1 = round(x1,10)
    elif abs(x1) > 1:
        x1 = round(x1,9)
    print("La aproximación para la ecuación {} es {} con {} iteraciones y valor inicial {}".format(func,x1,t,valin))

#%%
x = sp.symbols("x")

f = x ** 3 - 2 * x ** 2 - 5
newton_rapson(f,3,10**-4)

#%%
f2 = sp.cos(x) - x
newton_rapson(f2,2,10**-4)

#%%
f3 = 0.3*sp.sin(x) -x + 0.8
newton_rapson(f3,2,10**-4)

#%%
f4 = sp.ln(x-1)+sp.cos(x-1)
newton_rapson(f4,1.7,10**-4)

#%%
f5 = 3*x**2 - sp.exp(x)
newton_rapson(f5,1,10**-4)
newton_rapson(f5,-3,10**-4)
newton_rapson(f5,3,10**-4)


#%%
f6 = sp.sqrt(5) - x
newton_rapson(f6,2,10**-4)

#%% Problema final
f7 = sp.ln(x**2+1)-sp.exp(0.4*x)*sp.cos(sp.pi*x)
newton_rapson(f7,-1,10**-6)
