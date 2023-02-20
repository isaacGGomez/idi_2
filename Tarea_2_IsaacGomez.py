import sympy as sp
x,y = sp.symbols('x,y')

M = sp.Matrix([x*y**2-1,x+y-5])

J = M.jacobian([x,y])
Jinv = J.inv()


#Preguntar generico para n