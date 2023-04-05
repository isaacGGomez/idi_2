from PIL import Image as IM
import numpy as np
from Imagenes.KohonenFunc import *
#%%
img1 = IM.open('imagen1Deutschland-1.jpg')

#%%
img1Arr = np.array(img1)[:,:,:3]
img1ArrS = img1Arr.shape
img1Arr = np.reshape(img1Arr,(img1ArrS[0]*img1ArrS[1],-1))
img1Arr = np.append(img1Arr,np.zeros((img1Arr.__len__(),1)),axis = 1)
#%%
img1mean = IM.fromarray(np.uint8(img1ArrB))
#%%
img1mean.save('Test.png')
#%%

img1Arr = np.append(img1Arr,np.zeros((img1Arr.__len__(),1)),axis = 1)
df = np.append(df,np.zeros((df.__len__(),1)),axis=1)