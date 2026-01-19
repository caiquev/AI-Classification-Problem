# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 22:58:54 2020

@author: Caique
"""
# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""
import matplotlib.pyplot as plt

from skimage.feature import hog
import numpy as np
import random
from skimage import data, exposure


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

## lecture_cifar prenant en argument le chemin du répertoire contenant les données, 
## et renvoyant une matrice X

def lecture_cifar(path,batch):
    X = np.empty((0,3072))
    Y = np.empty((1,0),dtype=int)
    if batch == 1:
        
        file = path + "\\data_batch_" + str(batch)
        
        dictionary = unpickle(file)
        X = dictionary[b'data']
        Y = np.asarray(dictionary[b'labels'])
           
    else:
            for i in range(1,batch+1):
                 file = path + "\\data_batch_" + str(i)        
                 dictionary = unpickle(file)
                 X = np.vstack([X,dictionary[b'data']])
                 temp = np.asarray(dictionary[b'labels'])
                 Y = np.append(Y,temp)
                
                 
    X = np.float32(X)   
    return X, Y
    
## decoupage_donnees prenant en argument les matrices X et Y, et renvoyant les
## données d'apprentissage et de test : Xapp, Yapp, Xtest, Ytest.


def decoupage_donnees(X,Y):

    training_indice = random.sample(range(len(X)),k=round(0.8*len(X)))
    training_indice = sorted(np.asarray(training_indice))
    
    test_indice = np.setdiff1d(range(len(X)), training_indice)
    
    
    Xapp = X[training_indice,:]
    Yapp = Y[training_indice]
    Yapp = np.reshape(Yapp, (len(Yapp), 1) )
    Xtest = X[test_indice,:]
    Ytest = Y[test_indice]
    
    return Xapp,Yapp,Xtest,Ytest

## kppv_distances prenant en argument Xtest et Xapp et
## renvoyant la matrice des distances Dist

def kppv_distances(Xtest,Xapp):

    Xapp_squared = np.square(Xapp)
    Xtest_squared = np.square(Xtest)
    
    threeSums = np.sum(Xtest_squared[:,np.newaxis,:], axis=2) - 2 * Xtest.dot(Xapp.T) + np.sum(Xapp_squared, axis=1)
    Dist = np.sqrt(threeSums)
    
    return Dist

## kppv_predict prenant en argument Dist, Yapp et K le nombre de voisins

def kppv_predict(Dist,Yapp,K):
    
    Yapp = np.transpose(Yapp)
    index = np.argpartition(Dist,K,axis=-1)[:,0:K]
    Ytemp = Yapp[index[:,:],0]
    
    num_rows = np.shape(Ytemp)[0]
    Ypred = np.zeros((num_rows,1))
    
    for i in range(num_rows):
            Ypred[i] = np.argmax(np.bincount(Ytemp[i,:]))

    Ypred = Ypred.astype(int)
    Ypred = np.transpose(Ypred)
    return Ypred

## evaluation_classifieur prenant en argument Ytest et Ypred et
## renvoyant le taux de classification (Accuracy).

def evaluation_classifieur(Ytest,Ypred):
    
    Accuracy =  np.sum(Ypred == Ytest)/Ytest.size
    
    return Accuracy

## Nfold_data has inputs X,Y,N where N is the number for folds
## returns the arrays NX and NY which contains data divided into N parts 

def Nfold_data(X,Y,N):

    
    NX = np.split(X,N,0)
    NY = np.split(Y,N,0)
    
    NXnum_rows,NXnum_col = np.shape(NX[0])
    NYnum_rows,NYnum_col = np.shape(NY)
    
    Xapp = np.empty((N,NXnum_rows*(N-1),NXnum_col))
    Yapp = np.empty((N,1,NYnum_col*(N-1)),dtype=int)
    
    Xtest = np.empty((N,NXnum_rows,NXnum_col))
    Ytest = np.empty((N,1,NYnum_col),dtype=int)
    
    for i in range(N):
        
        tempX = np.roll(NX,-i,axis=0)
        tempY = np.roll(NY,-i,axis=0)
        Xtest[i] = tempX[0]
        Ytest[i] = tempY[0]      
            
        Xapp[i] = np.vstack(tempX[1:])      
        Yapp[i] = np.concatenate(tempY[1:,:]) 
             
             

    return NX,NY,Xapp,Yapp,Xtest,Ytest

def unflatten_image(img_flat):
    img_R = img_flat[0:1024].reshape((32, 32))
    img_G = img_flat[1024:2048].reshape((32, 32))
    img_B = img_flat[2048:3072].reshape((32, 32))
    img = np.dstack((img_R, img_G, img_B))
    return img    


###################################################################

path = r"D:\Windows.old.000\Users\Caique_\Desktop\Master\M2\Deep Learning\TD1\cifar-10-batches-py"
batch = 1
# K = 3 # Number of neighboors 
N = 5 # Nfold

X,Y = lecture_cifar(path,batch)

Xhog = np.zeros((len(Y),2560),dtype='float32')


for i in range(len(Y)):
    image = unflatten_image(X[i,:])
    image = np.uint(image)
    Xhog[i,:] = hog(image, orientations=10, pixels_per_cell=(2, 2),
	cells_per_block=(1, 1), transform_sqrt=True, block_norm="L1")
    print(i)
    

# Acc = []
# for K in range(1,21,2):
# K= 3
#     Xapp,Yapp,Xtest,Ytest = decoupage_donnees(Xhog,Y)
#     Dist = kppv_distances(Xtest,Xapp)
#     Ypred = kppv_predict(Dist,Yapp.T,K)
#     Acc.append(evaluation_classifieur(Ytest, Ypred))

    
# plt.plot(range(1,21,2),Acc,'ko')
# plt.ylabel('Accuracy')
# plt.xlabel('K-neighbors')
# plt.xticks(np.arange(1,21,2))
# plt.title('80% Learn - 20% Test division')
# plt.show()


NX,NY,NXapp,NYapp,NXtest,NYtest = Nfold_data(X,Y,N)
NXhog,NYhog,NXapphog,NYapphog,NXtesthog,NYtesthog = Nfold_data(Xhog,Y,N)
Acc_Nfold= np.empty(N)
Acc_Nfoldhog= np.empty(N)
Acc = []
e = []
Acchog = []
ehog = []

for K in range(1,21,2):
    
    for i in range(N):
        
        Dist = kppv_distances(NXtest[i],NXapp[i])
        
        Disthog = kppv_distances(NXtesthog[i],NXapphog[i])
        
        Ypred = kppv_predict(Dist,NYapp[i],K)
        
        Ypredhog = kppv_predict(Disthog,NYapphog[i],K)
        
        Acc_Nfold[i] = evaluation_classifieur(NYtest[i],Ypred)
        
        Acc_Nfoldhog[i] = evaluation_classifieur(NYtesthog[i],Ypredhog)
    
    Acc.append(np.mean(Acc_Nfold))
    e.append(np.std(Acc_Nfold))
    
    Acchog.append(np.mean(Acc_Nfoldhog))
    ehog.append(np.std(Acc_Nfoldhog))
    
plt.errorbar(range(1,21,2),Acc,yerr=e,fmt='ko')
plt.errorbar(range(1,21,2),Acchog,yerr=ehog,fmt='bo')
plt.legend(('Normal','Hog descrip'),loc='lower right')
plt.title('5-fold Cross Validation')
plt.ylabel('Accuracy')
plt.xlabel('K-neighbors')
plt.xticks(np.arange(1,21,2))
plt.show()