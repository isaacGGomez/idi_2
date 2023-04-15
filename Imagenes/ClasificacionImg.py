##### Importacion de funciones #############
from Imagenes.FuncionesImagenes import *
t = TicToc() #create instance of class
vectorImg = [2,3,10]

############ Imagenes K Means ###############
5#%%
#Kmeans Img 1
img1Km = IM.open('imagen1Deutschland-1.jpg')
img1kmvar = 'imagen1KM'
crearimgKm(img1Km,vectorImg,img1kmvar,0.0001)

#%%Imagen
img2Km = IM.open('imagen2México.jpg')
img2kmvar = 'imagen2KM'
crearimgKm(img2Km,vectorImg,img2kmvar,0.0001)

#%% Imagen 3
img3Km = IM.open('imagen3Gandhi.jpg')
img3kmvar = 'imagen3KM'
crearimgKm(img3Km,vectorImg,img3kmvar,0.0001)

#%% Imagen 4
img4Km = IM.open('imagen4Lauterbrunnen.jpg')
img4kmvar = 'imagen4KM'
crearimgKm(img4Km,vectorImg,img4kmvar,0.0001)

############ Imagenes Kohonen ###############
##Kohonen Img 1
t.tic()
img4KH = IM.open('imagen1Deutschland-1.jpg')
img4kHvar = 'imagen1KH'
crearimg(img4KH,vectorImg,img4kHvar,1,10)
t.toc()

##Kohonen Img 2
t.tic()
img4KH = IM.open('imagen2México.jpg')
img4kHvar = 'imagen2KH'
crearimg(img4KH,vectorImg,img4kHvar,1,7)
t.toc()


##Kohonen Img 3
t.tic()
img4KH = IM.open('imagen3Gandhi.jpg')
img4kHvar = 'imagen3KH'
crearimg(img4KH,vectorImg,img4kHvar,1,3)
t.toc()


##Kohonen Img 4
t.tic()
img4KH = IM.open('imagen4Lauterbrunnen.jpg')
img4kHvar = 'imagen4KH'
crearimg(img4KH,vectorImg,img4kHvar,1,2)
t.toc()


