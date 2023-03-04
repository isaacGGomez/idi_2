import numpy as np
import pandas as pd

df = pd.read_excel('tareaRGD.xlsx')

X = df[['x1','x2','x3']].to_numpy()

y =  df['y'].to_numpy()

n = X.shape[0]

class RegresionLinealMultiple:
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
reg_lineal_mult = RegresionLinealMultiple(n, eta=0.1, step=750)
reg_lineal_mult.fit(X, y)

#%%
