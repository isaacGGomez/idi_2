import pandas as pd
import numpy as np

df = pd.read_excel('tabla_para_probar.xlsx')
df_training = df[np.logical_not(df["d1"].isin(["?"]))]
#Inicializacion de variables
a= 1
alpha = 0.01
N = 4#inputs
M = 2 #Salidas conocidas
Q = 6 #Patrones de aprendizaje
L = 4 #Perceptrones
W_h = np.random.uniform(-1,1,(L,N))#Vector de pesos
W_o = np.random.uniform(-1,1,(M,L))
x = np.float64((df_training.iloc[:,:N]).to_numpy())
d = np.float64((df_training.iloc[:,N:]).to_numpy())
y = np.zeros([Q,M])


#%%
while(True):
    #Forward
    for i in range(Q):
        net_h = W_h @ x[i].transpose()
        y_h = np.reshape(1/(1+np.exp(-a*net_h)),(L,1))
        net_o = W_o @ y_h
        y = 1 / (1 + np.exp(-a*net_o) )


        #Backward

        d_o = ( np.reshape(d[i],(M,1)) - y)  *  y * (1 - y)
        d_h = y_h * (1 - y_h) * ( np.transpose(W_o) @ d_o )
        W_h +=  alpha * d_h  @ np.reshape(x[i],(1,N))
        W_o +=  alpha * d_o  @ y_h.transpose()



    if  np.linalg.norm(d_o) < 10**-4:
        break



#%%
#Originales
res_o = []
for i in range(Q):
    net_h = W_h @ x[i].transpose()
    y_h = np.reshape(1/(1+np.exp(-a*net_h)),(L,1))
    net_o = W_o @ y_h
    y = 1 / (1 + np.exp(-a*net_o) )
    res_o = np.append(res_o,np.round(y))

res_o = np.reshape(res_o,(Q,M))

res_o
df