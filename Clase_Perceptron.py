#%%
import pandas as pd
import numpy as np

df = pd.read_excel('tabla_para_probar.xlsx')
df_training = df[np.logical_not(df["d1"].isin(["?"]))]
#Inicializacion de variables
a= 1
alfa = 0.01
N = 4#inputs
M = 2 #Salidas conocidas
Q = 6 #Patrones de aprendizaje
L = 4 #Perceptrones
wh = np.random.uniform(-1,1,(L,N))#Vector de pesos
wo = np.random.uniform(-1,1,(M,L))
x = np.float64((df_training.iloc[:,:N]).to_numpy())
d = np.float64((df_training.iloc[:,N:]).to_numpy())
y = np.zeros([Q,M])
while(E>10**-3):
    for i in range(Q):
        xi=np.feshape (x[i], (N,1))
        di=np.reshape(d[i],(M,1))
        #FORWARD
        neth=wh@xi
        yh=sigmoid(neth)
        neto=wo@yh
        y=sigmoid(neto)
        #BACKWARD
        deltao=(di-y)y(1-y)
        deltah=yh*(1-yh)*(wo. T@deltao)
        wo+=alfa*deltao@yh.T
        wh+=alfa*deltah@xi.T
    E-np. linalg.norm(deltao)