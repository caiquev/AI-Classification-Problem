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


def lecture_cifar(path,batch):
# ##################################    
# lecture_cifar import the images from database CIFAR 10 
# Arguments:
# path -- path that leads to databate directory
# batch -- The database contains 50k images divided into 5 minibatches, this argument
# is a scalar and tells how many minibatches will be imported. 
## Returns: 
#  X -- A matrix size (N,M) where N is the number of images and M the size of the flattened image
# Y -- A 1D vector size (N,1) that contains the correct classification for each image
# ##################################    

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
# ##################################
# decoupage_donnees takes the outputs from lecture_cifar and divides the data 
# into 80% apprentisage set and 20% test set.
## Arguments:
# X -- A matrix size (N,M) where N is the number of images and M the size of the flattened image
# Y -- A 1D vector size (N,1) that contains the correct classification for each image 
## Returns: 
#  Xapp,Xtest -- HAve the same shape as the X input
# Yapp, Ytest -- Have the same shape as the Y input
# ##################################
    
    training_indice = random.sample(range(len(X)),k=round(0.8*len(X)))
    training_indice = sorted(np.asarray(training_indice))
    
    test_indice = np.setdiff1d(range(len(X)), training_indice)
    
    
    Xapp = X[training_indice,:]
    Yapp = Y[training_indice]
    Yapp = np.reshape(Yapp, (len(Yapp), 1) )
    Xtest = X[test_indice,:]
    Ytest = Y[test_indice]
    
    return Xapp,Yapp,Xtest,Ytest


def Nfold_data(X,Y,N):
# ##################################    
# Nfold_data divides the data into N folds
# Arguments : 
# X -- A matrix size (N,M) where N is the number of images and M the size of the flattened image
# Y -- A 1D vector size (N,1) that contains the correct classification for each image 
## Returns: 
# Xapp,Xtest -- A matrix size (x,x,N) where first two dimensions are used to store all data for the 
# respective fold indicated by the third dimension.
# Yapp,Ytest -- A matrix size (1,x,N) where first two dimensions are used to store the classification of data
# for respective fold indicated by the third dimension.
# ##################################
 
    
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
             
             

    return Xapp,Yapp,Xtest,Ytest

def unflatten_image(img_flat):
# ##################################
# unflatten_image reconsctruct the image obtained in the CIFAR database.
# Arguments:
# img_flat -- Vector size (1,3072), a flattened RGB image obtained in the CIFAR database
# Return:
# img -- An image size (32,32,3) pixels in RGB.
# ##################################
    img_R = img_flat[0:1024].reshape((32, 32))
    img_G = img_flat[1024:2048].reshape((32, 32))
    img_B = img_flat[2048:3072].reshape((32, 32))
    img = np.dstack((img_R, img_G, img_B))
    return img   

def hog_image(X):
# ##################################
# hog_image function performs a histogram of oriented gradients of the imported images by lecture_cifar.
# Arguments:
# X -- A matrix (m,3072) containing m images that will be transformed by HOG process.
# Return:
# Xhog -- A matrix (m,1024) containing m images transformed by the HOG process.
# ##################################    
    m = X.shape[0]    
    Xhog = np.empty((32,32,m),dtype='float32')
    
    for i in range(m):
        image = unflatten_image(X[i,:])
        image = np.uint(image)
        _,Xhog[:,:,i] = hog(image,visualize=True, multichannel=True)
    
    
    Xhog = Xhog.flatten().reshape(len(Y), 1024) 
    
    return Xhog


class KPPV:
    
    def __init__(self):
        self.Acc = [] # Accuracy list
        
    def distances(self,Xtest,Xapp):
    
        ## kppv_distances prenant en argument Xtest et Xapp et
        ## renvoyant la matrice des distances Dist
            
        Xapp_squared = np.square(Xapp)
        Xtest_squared = np.square(Xtest)
        
        threeSums = np.sum(Xtest_squared[:,np.newaxis,:], axis=2) - 2 * Xtest.dot(Xapp.T) + np.sum(Xapp_squared, axis=1)
        Dist = np.sqrt(threeSums)
        
        return Dist
    
   
    
    def predict(self,Dist,Yapp,K):
        
     ## kppv_predict prenant en argument Dist, Yapp et K le nombre de voisins
        
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
    
    def evaluation_classifieur(self,Ytest,Ypred):
        
        Accuracy =  np.sum(Ypred == Ytest)/Ytest.size
        
        return Accuracy
    

 
if __name__ == '__main__':

    ###################################################################
    
    path = r"D:\Windows.old.000\Users\Caique_\Desktop\Master\M2\Deep Learning\TD1\cifar-10-batches-py"
    batch = 1
    K = 3 # Number of neighboors 
    N = 5 # Nfold
    
    X,Y = lecture_cifar(path,batch)
       
    kppv = KPPV()
    
    Xapp_fold,Yapp_fold,Xtest_fold,Ytest_fold = Nfold_data(X,Y,N)
    
    Xhog = hog_image(X)
        
    
    NXapp,NYapp,NXtest,NYtest = Nfold_data(X,Y,N)
    Acc_Nfold= np.empty(N)
    Acc = []
    e = []
    

    for K in range(1,21,2):    
        for i in range(N):
            
            Dist = kppv.distances(Xtest_fold[i],Xapp_fold[i])
            
            Ypred = kppv.predict(Dist,Yapp_fold[i],K)
            
            Acc_Nfold[i] = kppv.evaluation_classifieur(Ytest_fold[i],Ypred)
            
        
        Acc.append(np.mean(Acc_Nfold))
        e.append(np.std(Acc_Nfold))
        
    plt.errorbar(range(1,21,2),Acc,e,fmt='ko')
    plt.title(str(N) + "-fold Cross Validation")
    plt.ylabel('Accuracy')
    plt.xlabel('K-neighbors')
    plt.xticks(np.arange(1,21,2))
    plt.show()
    
    

        
    # plt.plot(range(1,21,2),Acc,'ko')
    # plt.ylabel('Accuracy')
    # plt.xlabel('K-neighbors')
    # plt.xticks(np.arange(1,21,2))
    # plt.title('80% Learn - 20% Test division')
    # plt.show()
    
    