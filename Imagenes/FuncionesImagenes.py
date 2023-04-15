import numpy  as np
from PIL import Image as IM
from pytictoc import TicToc

def asignar( entrada, centroids ):
    return np.argmin( np.sum( (entrada - centroids) ** 2, axis=1) ) + 1

def centroides(df, clases, caracteristicas,tolerancia,error):
    tolerancia = tolerancia
    error = error
    distancias = []
    n_clases = int(clases)
    j = 2
    if n_clases < 2 or n_clases >32:
        return None
    obs = df.__len__()
    centroids = np.random.random([n_clases, caracteristicas])*(abs(np.min(df[:,:-1], axis = 0)-np.max(df[:,:-1], axis = 0))).T + np.min( df[:,:-1], axis = 0)
    while error > tolerancia:
        for i in range(obs):
            df[i, caracteristicas] = asignar(df[i,:-1],centroids)
            vector = (df[i, :-1] - centroids[int(df[i, -1])-1,:]) / j
            distancias.append(sum(abs(vector)))
            centroids[ int(df[i, -1])-1,:] += vector
        error = max(distancias)
        distancias = []
        j +=1

    return centroids

def asignar_a_clase ( entrada, centroides ):
    return np.argmin(np.sum( (entrada - centroides) ** 2, axis=1)) + 1

def kmeans(data, n_clases, n_caracteristicas,tolerance):
    prev_centroides = None
    n_clases = int(n_clases)
    if n_clases < 2 or n_clases >32:
        return None
    n_obs, _ = data.shape
    centroides = np.random.random([n_clases, n_caracteristicas]) * (abs(np.min(data[:,:-1], axis = 0) - np.max(data[:,:-1], axis = 0))).T + np.min( data[:,:-1], axis = 0)
    #while True:
    for j in range(300):
        for i in range(n_obs):
            data[i, n_caracteristicas] = asignar_a_clase(data[i,:-1] ,centroides)
        prev_centroides = np.copy(centroides)
        for i in range(n_clases):
            tmp = data[np.where(data[:,-1] == i+1)]
            #print(tmp)
            if tmp.any():
                centroides[i] = (np.mean(tmp[:,0]), np.mean(tmp[:,1]),np.mean(tmp[:,2]))
        error = max(sum((centroides - prev_centroides)**2,1))
        if error -1 < tolerance:
            break
    return centroides


#%%
def crearimg(img,vector,imname,tolerance,error):
    vec = vector
    for h in vec:
        img1Arr = np.array(img)[:,:,:3]
        img1ArrS = img1Arr.shape
        img1Arr = np.reshape(img1Arr,(img1ArrS[0]*img1ArrS[1],-1))
        img1Arr = np.append(img1Arr,np.zeros((img1Arr.__len__(),1)),axis = 1)
        centroid = centroides(img1Arr,h,3,tolerance,error)
        img1Arr = img1Arr.astype(int)
        for i in range(len(img1Arr)):
            img1Arr[i,0] = centroid[img1Arr[i,3]-1,0]
            img1Arr[i,1] = centroid[img1Arr[i,3]-1,1]
            img1Arr[i,2] = centroid[img1Arr[i,3]-1,2]
        img1Arr=np.reshape(img1Arr[:,:3],(img1ArrS[0],img1ArrS[1],3))
        img1mean = IM.fromarray(np.uint8(img1Arr))
        var = imname+str(h)+'.png'
        img1mean.save(var)

def crearimgKm(img,vector,imname,tolerance):
    vec = vector
    for h in vec:
        img1Arr = np.array(img)[:,:,:3]
        img1ArrS = img1Arr.shape
        img1Arr = np.reshape(img1Arr,(img1ArrS[0]*img1ArrS[1],-1))
        img1Arr = np.append(img1Arr,np.zeros((img1Arr.__len__(),1)),axis = 1)
        centroid = kmeans(img1Arr,h,3,tolerance)
        img1Arr = img1Arr.astype(int)
        for i in range(len(img1Arr)):
            img1Arr[i,0] = centroid[img1Arr[i,3]-1,0]
            img1Arr[i,1] = centroid[img1Arr[i,3]-1,1]
            img1Arr[i,2] = centroid[img1Arr[i,3]-1,2]
        img1Arr=np.reshape(img1Arr[:,:3],(img1ArrS[0],img1ArrS[1],3))
        img1mean = IM.fromarray(np.uint8(img1Arr))
        var = imname+str(h)+'.png'
        img1mean.save(var)