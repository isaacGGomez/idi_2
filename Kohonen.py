import numpy as np
import matplotlib.pyplot as plt



class Kohonen:
    
    def __init__(self,n=100,k=3,tolerancia=1):
        
        self.n=n
        self.k=k
        self.tolerancia = tolerancia
        self.data = np.random.uniform(1,1000,size=(n,2))
        max_lim,min_lim = round(np.max(self.data)),round(np.min(self.data))
        self.c = np.random.uniform(min_lim,max_lim,size=(k,2))
        step = 1
        c_old = self.c.copy()
        error = 100
    
        while error > self.tolerancia:
            
            for i in range(len(self.data)):
                distancias = []
                for j in range(len(self.c)):       
                    step+=1
                    dist = np.linalg.norm(self.data[i]-self.c[j],2)
                    distancias = np.append(distancias,dist)       
                    pos = np.argmin(distancias)
                    c_old = self.c
                    self.c[pos]= (1/step+1 )*(self.data[i]-self.c[pos])+self.c[pos]
            plt.scatter(self.data[:,0],self.data[:,1])
            plt.scatter(self.c[:,0],self.c[:,1])
            plt.show()
 
            error = max(sum((self.c-c_old)**2,1))
        
        
    def class_summary(self):
        
        indx_centroide = []
        for i in range(len(self.data)):
            distancias = []
            for j in range(len(self.c)):
                dist = np.linalg.norm(self.data[i]-self.c[j],2)
                distancias.append(dist)
                pos = np.argmin(distancias)
            indx_centroide.append(pos)
            
        class_sum = np.column_stack((self.data,(indx_centroide)))
        print(class_sum)


        
        
        

test = Kohonen(50)
test.class_summary()