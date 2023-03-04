#%%

import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)
learning_rate = 1 * 10**-5
n = 100
x1 = np.arange(1,n+1)
y1 = x1 + 3*np.random.uniform(0,1)
data = np.hstack([np.reshape(x1, [n,1]), np.reshape(y1, [n,1])])
df = pd.read_excel('tareaRGD.xlsx')
X = df[['x1','x2','x3']].to_numpy()
y =  df['y'].to_numpy()
n = X.shape[0]

def rlgd(a, b, data):
    i = 1
    Y = data[:, -1]
    X = data[:,:-1][:,0]
    while True :
        if not (np.isfinite(a) and np.isfinite(b)):
            return None, None, i
        error = a * X + b - Y
        Ea = np.mean( error * X )
        Eb = np.mean( error )
        a = a - learning_rate *  Ea
        b = b - learning_rate *  Eb
        if abs(error).mean() < 1:
            return sp.Float(a).evalf(4), sp.Float(b).evalf(4), i
        i += 1

a, b, i   = rlgd(1, -3, data)
print('a:', a)
print('b:', b)
print(i, 'iteraciones')

#%%
plt.scatter(x1,y1,s=5, label = "Y")
f= b + a*x1
plt.plot(x1,f,'r',label = "ax + b")
plt.legend(loc = "upper left")
#%%
class RLM:
    def __init__(self, n_features, eta = 0.05, step= 150):
        self.eta = eta
        self.step = step
        self.a = np.zeros(X.shape[1])
        self.b = 0
    def forward(self, X):
        return X.dot(self.a) + self.b
    def ecm(self,y_hat, y):
        s = y_hat - y
        avg_loss = np.mean(s**2)
        return np.expand_dims(s, axis=1), avg_loss
    def fit(self, X, y):
        for step in range(self.step):
            # Forward pass
            y_hat = self.forward(X)
            # Calcular las loss
            s, avg_loss = self.ecm(y_hat, y)
            # Calcular gradiente
            db = np.mean(2*s)
            da = np.mean(2*X*s, axis=0)
            # Actualizamos param.
            self.b = self.b - self.eta * db
            self.a = self.a - self.eta * da
            # Mostrar stats.
            if step % 40 == 0:
                print(f'step {step} train avg_loss {avg_loss:.4f}')
reg_lineal_mult = RLM(n, eta=0.1, step=1000)
reg_lineal_mult.fit(X, y)

print(reg_lineal_mult.a)
print(reg_lineal_mult.b)
print(reg_lineal_mult.step)
print(reg_lineal_mult.eta)
#%%


