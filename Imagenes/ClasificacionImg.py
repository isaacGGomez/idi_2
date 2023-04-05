from PIL import Image as IM
import Kohonen
import numpy as np
#%%
img1 = IM.open('imagen1Deutschland-1.jpg')


#%%
img1Arr = np.array(img1)[:,:,:3]
np.reshape(img1Arr,(60*100,-1))
#%%
img1ArrB = np.mean(img1Arr,2)
#%%
img1mean = IM.fromarray(np.uint8(img1ArrB))
#%%
img1mean.save('Test.png')
#%%
