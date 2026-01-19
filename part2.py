# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 17:10:35 2020

@author: Caique
"""
import numpy as np
np.random.seed(1) # pour que l'exécution soit déterministe
import matplotlib.pyplot as plt

def sigmoid(x):
    # Prevent overflow.
    x = np.clip( x, -500, 500 )
    
    return 1/(1+np.exp(-x))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dX, Z):
    sig = sigmoid(Z)
    return dX * sig * (1 - sig)

def relu_backward(dX, Z):
    dZ = np.array(dX, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;


def basic_backward_prop(dX_curr, W_curr, b_curr, Z_curr, X_prev, activation="sigmoid"):
    m = X_prev.shape[1]
    
    if activation is "relu":
        backward_activation_func = relu_backward
    elif activation is "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')
    
    dZ_curr = backward_activation_func(dX_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, X_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dX_prev = np.dot(W_curr.T, dZ_curr)

    return dX_prev, dW_curr, db_curr

##########################
# Import Cifar database  #
##########################

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


def evaluation_classifieur(Ytest,Ypred):
    
    Accuracy =  np.sum(Ypred == Ytest)/Ytest.size
    
    return Accuracy


###############################
# Divide data X,Y into Nfolds #
###############################

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

# def softmax(X):
#     exps = np.exp(X)
#     return exps / np.sum(exps)

def softmax(y_hat):
        tmp = y_hat - y_hat.max(axis=1).reshape(-1, 1)
        exp_tmp = np.exp(tmp)
        return exp_tmp / exp_tmp.sum(axis=1).reshape(-1, 1)


def delta_cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector. 
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    y.astype(int)
    grad = softmax(X)
    grad[range(m),y] -= 1
    grad = grad/m
    return grad

path = r"D:\Windows.old.000\Users\Caique_\Desktop\Master\M2\Deep Learning\TD1\cifar-10-batches-py"
batch = 1
# K = 3 # Number of neighboors

X,Y = lecture_cifar(path,batch)
Nf = 2 
Acc_Nfold= np.empty(Nf)

NX,NY,NXapp,NYapp,NXtest,NYtest = Nfold_data(X,Y,Nf)

NX = NX/np.max(NX)

NXapp = NXapp/np.max(NXapp)
NXapp -= np.mean(NXapp,axis =0)

NXtest = NXtest/np.max(NXtest)
NXtest -= np.mean(NXtest,axis =0)

#Normalize data to range between 0-1

j=0

##########################
# Génération des données #
##########################
# N est le nombre de données d'entrée
# D_in est la dimension des données d'entrée
# D_h le nombre de neurones de la couche cachée
# D_out est la dimension de sortie (nombre de neurones de la couche de sortie)

N, D_in, D_h, D_out = len(NYtest[0,0]), 3072, 600, 10
Dextra = 300 #Number of neurons on extra layer between W1 and W2 

# Création d'une matrice d'entrée X et de sortie Y avec des valeurs aléatoires
# X = np.random.random((N, D_in))
# Y = np.random.random((N, D_out))
# Initialisation aléatoire des poids du réseau

def weight_kaiming(D_in,D_out):
    W = np.random.randn(D_in, D_out) * np.sqrt(2/D_in)
    B = np.zeros((1,D_out))
    return W,B

####################################################
#                      3 LAYERS                    #
####################################################

W1,B1 =  weight_kaiming(D_in,D_h)

Wextra,Bextra = weight_kaiming(D_h,Dextra)

W2,B2 = weight_kaiming(Dextra,D_out)

####################################################
#                      2 LAYERS                    #
####################################################

# W1,B1 =  weight_kaiming(D_in,D_h)
# W2,B2 = weight_kaiming(D_h,D_out)
#

####################################################
# Passe avant : calcul de la sortie prédite Y_pred #
####################################################

Z1 = NXapp[j].dot(W1) + B1 # Potentiel d'entrée de la couche cachée
N1 = sigmoid(Z1) # Sortie de la couche cachée (fonction d'activation de type sigmoïde)

Zextra = N1.dot(Wextra) + Bextra # Potentiel d'entrée de la couche cachée
Nextra = sigmoid(Zextra)

Z2 = Nextra.dot(W2) + B2 # Potentiel d'entrée de la couche de sortie
N2 = softmax(Z2) # Sortie de la couche de sortie (fonction d'activation de type sigmoïde)


####################################################
# Passe avant :             2 layers               #
####################################################

# Z1 = NXapp[j].dot(W1) + B1 # Potentiel d'entrée de la couche cachée
# N1 = sigmoid(Z1) # Sortie de la couche cachée (fonction d'activation de type sigmoïde)

# Z2 = N1.dot(W2) + b2 # Potentiel d'entrée de la couche de sortie
# N2 = sigmoid(Z2) # Sortie de la couche de sortie (fonction d'activation de type sigmoïde)

Y_pred = N2 # Les valeurs prédites sont les sorties de la couche de sortie

Yapp = np.zeros((N,10))

for i in range(N):
           
        Yapp[i,NYapp[0,0,i]] = 1
    
    
########################################################
# Calcul et affichage de la fonction perte de type MSE #
########################################################
# loss = [np.square(Y_pred - Yapp).sum() / 2]
loss =  [-np.sum(Yapp * np.log(Y_pred))/Yapp.shape[0]]

# loss = [np.sum(-Yapp * np.log(Y_pred))]

print(loss)


########################################################
#             Gradient Descent Calculation             #
########################################################
m = N
backward_activation_func = sigmoid_backward 
learning_rate = 0.001 
epochs = 150

first_moment_W1 = 0
second_moment_W1 = 0 

first_moment_W2 = 0
second_moment_W2 = 0 

first_moment_Wextra = 0
second_moment_Wextra = 0 

first_moment_B1 = 0
second_moment_B1 = 0 

first_moment_B2 = 0
second_moment_B2 = 0 

first_moment_Bextra = 0
second_moment_Bextra = 0 

beta1 = 0.9
beta2 = 0.999

for i in range(1,epochs):

    # dYpred = (Y_pred - Yapp) # FOR MSE LOSS FUNCTION
    dYpred = delta_cross_entropy(Y_pred,NYapp[j].T) # Cross entropy loss derivative
        
    dZ2 = backward_activation_func(dYpred, Z2)
    
    dW2 = np.dot(Nextra.T,dZ2) / m
    
    dB2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    
    # dX1 = np.dot(dZ2,W2.T) 
    
    # dZ1 = backward_activation_func(dX1, Z1)
    
    # dW1 = np.dot(NXapp[j].T,dZ1) / m
    
    # db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    
    dXextra = np.dot(dZ2,W2.T) 
    
    dZextra = backward_activation_func(dXextra, Zextra)
    
    dWextra = np.dot(N1.T,dZextra) / m
    
    dBextra = np.sum(dZextra, axis=0, keepdims=True) / m
    
    
    dX1 = np.dot(dZextra,Wextra.T) 
    
    dZ1 = backward_activation_func(dX1, Z1)
    
    dW1 = np.dot(NXapp[j].T,dZ1) /m
    
    dB1 = np.sum(dZ1, axis=0, keepdims=True) /m

####### ADAM 
    
    first_moment_W1 = beta1 * first_moment_W1 + (1-beta1) * dW1
    second_moment_W1 = beta2 * second_moment_W1 + (1-beta2) *dW1 * dW1
    first_unbias_W1 = first_moment_W1/(1 - beta1 ** i)
    second_unbias_W1 = second_moment_W1/(1 - beta2 ** i)
    
    W1 -= learning_rate * first_unbias_W1 / (np.sqrt(second_unbias_W1)+1e-5)     

    first_moment_W2 = beta1 * first_moment_W2 + (1-beta1) * dW2
    second_moment_W2 = beta2 * second_moment_W2 + (1-beta2) *dW2 * dW2
    first_unbias_W2 = first_moment_W2/(1 - beta1 ** i)
    second_unbias_W2 = second_moment_W2/(1 - beta2 ** i)
    
    W2 -= learning_rate * first_unbias_W2 / (np.sqrt(second_unbias_W2)+1e-5) 
    

    first_moment_Wextra = beta1 * first_moment_Wextra + (1-beta1) * dWextra
    second_moment_Wextra = beta2 * second_moment_Wextra + (1-beta2) *dWextra * dWextra
    first_unbias_Wextra = first_moment_Wextra/(1 - beta1 ** i)
    second_unbias_Wextra = second_moment_Wextra/(1 - beta2 ** i)
    
    Wextra -= learning_rate * first_unbias_Wextra / (np.sqrt(second_unbias_Wextra)+1e-5) 
    
    first_moment_B1 = beta1 * first_moment_B1 + (1-beta1) * dB1
    second_moment_B1 = beta2 * second_moment_B1 + (1-beta2) *dB1 * dB1
    first_unbias_B1 = first_moment_B1/(1 - beta1 ** i)
    second_unbias_B1 = second_moment_B1/(1 - beta2 ** i)
    
    B1 -= learning_rate * first_unbias_B1 / (np.sqrt(second_unbias_B1)+1e-5) 
    
    
    first_moment_B2 = beta1 * first_moment_B2 + (1-beta1) * dB2
    second_moment_B2 = beta2 * second_moment_B2 + (1-beta2) *dB2 * dB2
    first_unbias_B2 = first_moment_B2/(1 - beta1 ** i)
    second_unbias_B2 = second_moment_B2/(1 - beta2 ** i)
    
    B2 -= learning_rate * first_unbias_B2 / (np.sqrt(second_unbias_B2)+1e-5) 
    
    
    first_moment_Bextra = beta1 * first_moment_Bextra + (1-beta1) * dBextra
    second_moment_Bextra = beta2 * second_moment_Bextra + (1-beta2) *dBextra * dBextra
    first_unbias_Bextra = first_moment_Bextra/(1 - beta1 ** i)
    second_unbias_Bextra = second_moment_Bextra/(1 - beta2 ** i)
    
    Bextra -= learning_rate * first_unbias_Bextra / (np.sqrt(second_unbias_Bextra)+1e-5) 

####################### 2 LAYERS ##############################################
    
    # Z1 = NXapp[j].dot(W1) + b1 # Potentiel d'entrée de la couche cachée
    # N1 = sigmoid(Z1) # Sortie de la couche cachée (fonction d'activation de type sigmoïde)
    
    # Z2 = N1.dot(W2) + b2 # Potentiel d'entrée de la couche de sortie
    # N2 = sigmoid(Z2) # Sortie de la couche de sortie (fonction d'activation de type sigmoïde)

####################### 3 LAYERS ##############################################
    
    Z1 = NXapp[j].dot(W1) + B1 # Potentiel d'entrée de la couche cachée
    N1 = sigmoid(Z1) # Sortie de la couche cachée (fonction d'activation de type sigmoïde)

    Zextra = N1.dot(Wextra) + Bextra # Potentiel d'entrée de la couche cachée
    Nextra = sigmoid(Zextra)

    Z2 = Nextra.dot(W2) + B2 # Potentiel d'entrée de la couche de sortie
    N2 = softmax(Z2) # Sortie de la couche de sortie (fonction d'activation de type sigmoïde)
    
    Y_pred = N2 # Les valeurs prédites sont les sorties de la couche de sortie

    ########################################################
    # Calcul et affichage de la fonction perte de type MSE #
    ########################################################
    # loss.append(np.square(Y_pred - Yapp).sum() / Yapp.size)
    loss.append(-np.sum(Yapp * np.log(Y_pred))/Yapp.shape[0])
    
    # if i == 20:
    #     learning_rate = learning_rate*0.1
    
    # if i == 40:
    #     learning_rate = learning_rate*0.1
        
    # if i == 60:
    #     learning_rate = learning_rate*0.1
    if i == 80:
        learning_rate = learning_rate*0.1
    # if i == 100:
    #     learning_rate = learning_rate*0.1

    # if i == 120:
    #     learning_rate = learning_rate*0.1
        
plt.plot(range(epochs),loss)
plt.title('Neural Network - 3 layers - Sigmoid activation - ' + str(D_h) + ' Neurons')
plt.ylabel('Loss [Cross entropy]')
plt.xlabel('Epochs')
plt.show()

####################### 2 LAYERS ##############################################
# Z1 = NXtest[j].dot(W1) + B1 # Potentiel d'entrée de la couche cachée
# N1 = sigmoid(Z1) # Sortie de la couche cachée (fonction d'activation de type sigmoïde)
    
# Z2 = N1.dot(W2) + B2 # Potentiel d'entrée de la couche de sortie
# N2 = sigmoid(Z2) # Sortie de la couche de sortie (fonction d'activation de type sigmoïde)

####################### 3 LAYERS ##############################################
Z1 = NXtest[j].dot(W1) + B1 # Potentiel d'entrée de la couche cachée
N1 = sigmoid(Z1) # Sortie de la couche cachée (fonction d'activation de type sigmoïde)

Zextra = N1.dot(Wextra) + Bextra # Potentiel d'entrée de la couche cachée
Nextra = sigmoid(Zextra)

Z2 = Nextra.dot(W2) + B2 # Potentiel d'entrée de la couche de sortie
N2 = softmax(Z2) # Sortie de la couche de sortie (fonction d'activation de type sigmoïde)
###############################################################################
    
Y_pred = N2 # Les valeurs prédites sont les sorties de la couche de sortie
Y_pred = Y_pred.argmax(axis=1)


Acc_Nfold[j] = evaluation_classifieur(NYtest[j],Y_pred)

########################### TEST ON TRAIN DATA ###############################

Z1 = NXapp[j].dot(W1) + B1 # Potentiel d'entrée de la couche cachée
N1 = sigmoid(Z1) # Sortie de la couche cachée (fonction d'activation de type sigmoïde)

Zextra = N1.dot(Wextra) + Bextra # Potentiel d'entrée de la couche cachée
Nextra = sigmoid(Zextra)

Z2 = Nextra.dot(W2) + B2 # Potentiel d'entrée de la couche de sortie
N2 = softmax(Z2) # Sortie de la couche de sortie (fonction d'activation de type sigmoïde)
###############################################################################
    
Y_pred = N2 # Les valeurs prédites sont les sorties de la couche de sortie
Y_pred = Y_pred.argmax(axis=1)

Acc_Nfold[1] = evaluation_classifieur(NYapp[j],Y_pred)