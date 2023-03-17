import pandas as pd
import numpy as np

df = pd.read_excel('tabla_para_probar.xlsx')
dftrain = df[np.logical_not(df["d1"].isin(["?"]))]
#Inicializacion de variables
a= 1
alpha = 0.01
N = 4#inputs
M = 2 #Salidas conocidas
Q = 6 #Patrones de aprendizaje
L = 4 #Neuronas
wh = np.random.uniform(-1, 1, (L, N))#Vector de pesos
wo = np.random.uniform(-1, 1, (M, L))
x = np.float64((dftrain.iloc[:, :N]).to_numpy())
d = np.float64((dftrain.iloc[:, N:]).to_numpy())
y = np.zeros([Q,M])


#%%
while(True):
    #Forward
    for i in range(Q):
        net_h = wh @ x[i].transpose()
        y_h = np.reshape(1/(1+np.exp(-a*net_h)),(L,1))
        net_o = wo @ y_h
        y = 1 / (1 + np.exp(-a*net_o) )


        #Backward

        d_o = ( np.reshape(d[i],(M,1)) - y)  *  y * (1 - y)
        d_h = y_h * (1 - y_h) * (np.transpose(wo) @ d_o)
        wh += alpha * d_h @ np.reshape(x[i], (1, N))
        wo += alpha * d_o @ y_h.transpose()



    if  np.linalg.norm(d_o) < 10**-4:
        break

#Originales
res_o = []
for i in range(Q):
    net_h = wh @ x[i].transpose()
    y_h = np.reshape(1/(1+np.exp(-a*net_h)),(L,1))
    net_o = wo @ y_h
    y = 1 / (1 + np.exp(-a*net_o) )
    res_o = np.append(res_o,np.round(y))

res_o = np.reshape(res_o,(Q,M))

res_o
d
#Comparacion
res_o==d