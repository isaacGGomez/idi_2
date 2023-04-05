import numpy  as np
import pandas as pd
def generar_centroides (data, n_clases, n_caracteristicas):
    n_clases = int(n_clases)
    if n_clases < 2:
        return None
    n_obs, _ = data.shape
    data = np.append(data, np.zeros([n_obs,1]), axis=1)
    centroides = np.random.random([n_clases, n_caracteristicas]) * (abs(np.min(data[:,:-1], axis = 0) - np.max(data[:,:-1], axis = 0))).T + np.min( data[:,:-1], axis = 0)
    prev_centroides = None
    while np.not_equal(prev_centroides, centroides).any() :
        for i in range(n_obs):
            data[i, n_caracteristicas] = asignar_a_clase( data[i,:-1] , centroides )
        prev_centroides = centroides
        for i in range(n_clases):
            tmp = data[np.where(data[:,-1] == i+1)] 
            print(tmp)
            centroides[i] = (np.mean(tmp[:,0]), np.mean(tmp[:,1]))
    return centroides
def asignar_a_clase ( entrada, centroides ):
    return np.argmin(np.sum( (entrada - centroides) ** 2, axis=1)) + 1