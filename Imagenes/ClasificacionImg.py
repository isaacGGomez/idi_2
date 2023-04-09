from pytictoc import TicToc
t = TicToc() #create instance of class
from Imagenes.KohonenFunc import *
#%%
img1 = IM.open('imagen1Deutschland-1.jpg')
img1Arr = np.array(img1)[:,:,:3]
img1ArrS = img1Arr.shape
img1Arr = np.reshape(img1Arr,(img1ArrS[0]*img1ArrS[1],-1))
img1Arr = np.append(img1Arr,np.zeros((img1Arr.__len__(),1)),axis = 1)
#%%
t.tic() #Start timer
centroid = kmeans(img1Arr,3,3)
t.toc() #Time elapsed since t.tic()
#%%
img1Arr = img1Arr.astype(int)
#%%
for i in range(len(img1Arr)):
    img1Arr[i,0] = centroid[img1Arr[i,3]-1,0]
    img1Arr[i,1] = centroid[img1Arr[i,3]-1,1]
    img1Arr[i,2] = centroid[img1Arr[i,3]-1,2]
img1Arr=np.reshape(img1Arr[:,:3],(img1ArrS[0],img1ArrS[1],3))
#%%
img1mean = IM.fromarray(np.uint8(img1Arr))
#%%
img1mean.save('KmeansTest3.png')

#%%
img2 = IM.open('imagen2México.jpg')
t.tic()
crearimg(img2)
t.toc()
#%%
img3 = IM.open('imagen3Gandhi.jpg')
t.tic()
crearimg(img3)
t.toc()
#%%
img4 = IM.open('imagen4Lauterbrunnen.jpg')
iteration = [26]
t.tic()
crearimg(img=img4,vector=iteration)
t.toc()
#%%
#for h in [2,3,10]:
#    img1Arr = np.array(img1)[:,:,:3]
#    img1ArrS = img1Arr.shape
#    img1Arr = np.reshape(img1Arr,(img1ArrS[0]*img1ArrS[1],-1))
#    img1Arr = np.append(img1Arr,np.zeros((img1Arr.__len__(),1)),axis = 1)
#    centroid = centroides(img1Arr,h,3)
#    img1Arr = img1Arr.astype(int)
#    for i in range(len(img1Arr)):
#        img1Arr[i,0] = centroid[img1Arr[i,3]-1,0]
#        img1Arr[i,1] = centroid[img1Arr[i,3]-1,1]
#        img1Arr[i,2] = centroid[img1Arr[i,3]-1,2]
#    img1Arr=np.reshape(img1Arr[:,:3],(img1ArrS[0],img1ArrS[1],3))
#    img1mean = IM.fromarray(np.uint8(img1Arr))
#    var = 'Test2'+str(h)+'.png'
#    img1mean.save(var)

#%%

