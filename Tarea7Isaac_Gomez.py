import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
df = pd.read_excel("PercMultAplicado.xlsx")
#%%
df.head()
#%%
df.hist(figsize=(10,10))

#%%
df["Cargo"] = df["Mensualidad"]/df["Ingreso mensual"]
df["Monto"] = df["Monto"]**(1/2)
df["Monto"] = (df["Monto"]-df["Monto"].mean())/df["Monto"].std()
df["Antigüedad laboral (meses)"] = df["Antigüedad laboral (meses)"]/max(df["Antigüedad laboral (meses)"])
df["Mora"] = df["Mora"]=="SI"


#%%
dffin = df[["Monto","Antigüedad laboral (meses)","Cargo","Mora"]]
dffin.head()
dffin.hist(figsize=(10,10))


#%%
Xt,xt,Yt,yt=train_test_split(dffin[["Monto","Antigüedad laboral (meses)","Cargo"]],dffin['Mora'])

#%%
#Inicializacion de variables
a= 1
alpha = 0.01
N = Xt.shape[1]#inputs
M = 1 #Salidas conocidas
Q = len(Xt)#Patrones de aprendizaje
L = 5 #Neuronas
wh = np.random.uniform(-1, 1, (L, N))#Vector de pesos
wo = np.random.uniform(-1, 1, (M, L))
x = np.float64(Xt.to_numpy())
d = np.float64(Yt.to_numpy())
y = np.zeros([Q,M])

#%%
while(True):
    #Forward
    for i in range(Q):
        net_h = wh @ x[i].transpose()
        y_h = np.reshape(1/(1+np.exp(-a*net_h)),(L,1))
        net_o = wo @ y_h
        y = 1 / (1 + np.exp(-a*net_o))


        #Backward

        d_o = ( np.reshape(d[i],(M,1)) - y)*y* (1 - y)
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
    y = 1 / (1 + np.exp(-a*net_o))
    res_o = np.append(res_o,np.round(y))

res_o = np.reshape(res_o,(Q,M))


#%%
accuracy = accuracy_score(Yt,res_o)
accuracy

#%%
testlen = len(xt)
xtest = np.float64(xt.to_numpy())
ytest = np.float64(yt.to_numpy())
restest = []
for i in range(testlen):
    net_h = wh @ xtest[i].transpose()
    y_h = np.reshape(1/(1+np.exp(-a*net_h)),(L,1))
    net_o = wo @ y_h
    y = 1 / (1 + np.exp(-a*net_o))
    restest = np.append(restest,np.round(y))

restest = np.reshape(restest,(testlen,M))
accuracy_score(ytest,restest)