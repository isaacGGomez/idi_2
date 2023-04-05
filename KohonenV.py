import numpy  as np
import pandas as pd

def centroides(df, clases, caracteristicas):
    tolerancia = .1
    error = 99999
    distancias = []
    n_clases = int(clases)
    j = 2
    if n_clases < 2:
        return None
    obs = df.__len__()
    centroids = np.random.random([n_clases, caracteristicas])*(abs(np.min(df[:,:-1], axis = 0)-np.max(df[:,:-1], axis = 0))).T + np.min( df[:,:-1], axis = 0)
    while error > tolerancia :
        for i in range(obs):
            df[i, caracteristicas] = asignar(df[i,:-1],centroids)
            vector = (df[i, :-1] - centroids[int(df[i, -1])-1,:]) / j
            distancias.append(sum(abs(vector)))
            centroids[ int(df[i, -1])-1,:] += vector
        error = max(distancias)
        distancias = []
        j +=1
    return centroids

def asignar( entrada, centroids ):
    return np.argmin( np.sum( (entrada - centroids) ** 2, axis=1) ) + 1
#%%
#------------------------------------------------------------------
np.random.seed( 123 )
df = pd.read_excel("Kohonen.xlsx", sheet_name = "Datos")
df["Cent"] = 0
df = df.to_numpy()
caracteristicas = 2
clases = 2
para_asignar = [38, 66]

cent = centroides(df, clases, caracteristicas)


#------------------------------------------------------------------


print( 'Centroides: ' )
print( centroides, '\n' )


#%%
