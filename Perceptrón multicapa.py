import numpy as np
import pandas as pd

df = pd.read_excel('tabla_para_probar.xlsx')
df_training = df[np.logical_not(df["d1"].isin(["?"]))]



alpha = 0.05
a =  1

"Entradas"
N = 4
"Patrones de aprendizaje"
Q = 6
"Perceptrones" 
L = 6
"Salidas conocidas"
M = 2



W_h = np.random.uniform(-1,1,(L,N))
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

comp = d==res_o
print(f'\n\nOriginales\n{d}')
print(f'\nPredichos\n{res_o}')
print (f'{comp}')


print('\n\nPrediccion') 
df_test  = df[ df['d1'] == '?' ]

Q_t = len(df_test)
y_test = np.float64(np.zeros([Q_t,M]))
x_test = np.float64((df_test.iloc[:,:N]).to_numpy())
res = []
for i in range(Q_t):
    net_h = W_h @ x_test[i].transpose()
    y_h = np.reshape(1/(1+np.exp(-a*net_h)),(L,1))
    net_o = W_o @ y_h
    y_test = 1 / (1 + np.exp(-a*net_o) )
    res = np.append(res,np.round(y_test))

res = np.reshape(res,(Q_t,M))    
print(res)