# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 00:44:38 2020

@author: Caique
"""
import numpy as np
import random
import matplotlib.pylab as plt
 
 
class ANN:
    def __init__(self, layers_size):
        self.layers_size = layers_size
        self.parameters = {}
        self.L = len(self.layers_size)
        self.n = 0
        self.costs = []
        self.accuracy = []
        self.accuracy_test = []
        
    def sigmoid(self, Z):
        Z = np.clip( Z, -500, 500 ) # Prevent overflow
        return 1 / (1 + np.exp(-Z))
 
    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z)) # Shift the max value of Z to avoid the exponential to blow up
        return expZ / expZ.sum(axis=0, keepdims=True)
    
    def sigmoid_derivative(self, Z):
        Z = np.clip( Z, -500, 500 ) # Prevent overflow
        s = 1 / (1 + np.exp(-Z)) 
        return s * (1 - s)

    def relu(self,X):        
        return np.clip(X,0,6) # Relu6 implemented to avoid problems with +inf (ref Alex Krizhevsky article)
    
    def relu_derivative(self,X):
        X[X<=0] = 0
        X[X>0] = 1
        return X
 
    def initialize_parameters(self):
        np.random.seed(1)
        
        
        # Kaiming Weight Initialization
        
        for l in range(1, len(self.layers_size)):
            self.parameters["W" + str(l)] = np.random.randn(self.layers_size[l], self.layers_size[l - 1]) / \
                np.sqrt(2 / self.layers_size[l])
                
            self.parameters["b" + str(l)] = np.zeros((self.layers_size[l], 1))
 
    def forward(self, X, activ_function):
        store = {}
        
        A = X.T
        for l in range(self.L - 1):
            Z = self.parameters["W" + str(l + 1)].dot(A) + self.parameters["b" + str(l + 1)]
            A = activ_function(Z)
            store["A" + str(l + 1)] = A
            store["W" + str(l + 1)] = self.parameters["W" + str(l + 1)]
            store["Z" + str(l + 1)] = Z
 
        Z = self.parameters["W" + str(self.L)].dot(A) + self.parameters["b" + str(self.L)]
        A = self.softmax(Z) # Output layer
        store["A" + str(self.L)] = A
        store["W" + str(self.L)] = self.parameters["W" + str(self.L)]
        store["Z" + str(self.L)] = Z
 
        return A, store
 
 
    def backward(self, X, Y, store,activ_deriv):
 
        derivatives = {}
        
        store["A0"] = X.T
 
        A = store["A" + str(self.L)]
        
        
        dZ = A - Y.T # Cross entropy 
        
        
        dW = dZ.dot(store["A" + str(self.L - 1)].T) / self.n
        db = np.sum(dZ, axis=1, keepdims=True) / self.n
        dAPrev = store["W" + str(self.L)].T.dot(dZ)
 
        derivatives["dW" + str(self.L)] = dW
        derivatives["db" + str(self.L)] = db
 
        for l in range(self.L - 1, 0, -1):
            dZ = dAPrev * activ_deriv(store["Z" + str(l)])
            dW = 1. / self.n * dZ.dot(store["A" + str(l - 1)].T)
            db = 1. / self.n * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = store["W" + str(l)].T.dot(dZ)
 
            derivatives["dW" + str(l)] = dW
            derivatives["db" + str(l)] = db
 
        return derivatives
 
    def fit(self, X, Y,activ_function,learning_rate = 1  , n_iterations=1000):
        
        if activ_function == 'relu':
            function = self.relu
            deriv = self.relu_derivative
        elif activ_function == 'sigmoid':
            function = self.sigmoid
            deriv = self.sigmoid_derivative
        else:
            print('Function not supported')
            exit()
        
        if len(X) == 2: #Assumes that both train and test batches were given at once
            train_x,test_x = X
            train_y,test_y = Y
            X = train_x
            Y = train_y
            
        
        learning_init = learning_rate
        np.random.seed(1)
        
        self.n = X.shape[0]
        self.layers_size.insert(0, X.shape[1])
         
        #
        D = 10 #Number of learning rate restarts during training
        self.cycle = np.array(range(round(n_iterations/D)))
       
 
        self.initialize_parameters()
        for loop in range(n_iterations):
            
            learning_rate = 0.5 * learning_init * (1 + np.cos(np.roll(self.cycle,-loop)[0] * np.pi / D)) 
            A, store = self.forward(X,function)
            cost = -np.mean(Y * np.log(A.T+ 1e-8))
            derivatives = self.backward(X, Y, store,deriv)
 
            for l in range(1, self.L + 1):
                
                
                self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - learning_rate * derivatives[
                    "dW" + str(l)]
                self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - learning_rate * derivatives[
                    "db" + str(l)]
 
            if loop % 100 == 0:
                print("Cost: ", cost, "Train Accuracy:", self.predict(X, Y,function))
                # Restart for Learning rate
                
                
            if loop % 10 == 0:
                self.costs.append(cost)
                self.accuracy.append(self.predict(X, Y,function))
                self.accuracy_test.append(self.predict(test_x, test_y,function))
                
    def predict(self, X, Y,function):
        A, cache = self.forward(X,function)
        
        # These two lines bellow do the oposite of the hot_encoder function
        # transforms the output data with size (N,C) to (N,1) to evaluate the accuracy
        
        y_hat = np.argmax(A, axis=0)
        Y = np.argmax(Y, axis=1)
        accuracy = (y_hat == Y).mean()
        return accuracy * 100
 
    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.title("Neural Network - " + str(sum(layers_dims[1:-1])) + " Neurons - " + str(len(layers_dims)-1) + " Layers" )
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.show()        


    def  plot_acc(self):        
        plt.figure()
        plt.plot(np.arange(len(self.accuracy)), self.accuracy)
        plt.title("Neural Network - " + str(sum(layers_dims[1:-1])) + " Neurons - " + str(len(layers_dims)-1) + " Layers" )
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.show()     
        
    def plot(self):
        
        plt.figure()
        
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.title("Neural Network - " + str(sum(layers_dims[1:-1])) + " Neurons - " + str(len(layers_dims)-1) + " Layers" )
        plt.ylabel("cost")
        
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(len(self.accuracy)), self.accuracy)
        plt.plot(np.arange(len(self.accuracy_test)), self.accuracy_test)
        plt.legend(['Train','Test'])
        plt.ylabel("accuracy")
        plt.xlabel("epochs")
        plt.show() 

def hot_encoder(Y):
 # Changes the input Y with size (N,1) to a 2D matrix with size (N,C)
 # where N is the number of data and C the number of different classes
 # C = 10.  
    C = 10 ;  
    out = np.zeros((len(Y),C))
    
    for i in range(len(Y)):
           
        out[i,Y[i]] = 1
        
    return out
 
def pre_process_data(train_x, train_y, test_x, test_y):
    # Normalize
    train_x = train_x / 255.
    test_x = test_x / 255.
    
    # Subtract the mean image from data.
    train_x -= np.mean(train_x,axis=0)
    test_x -= np.mean(test_x,axis=0)
    
    # Transform Data from size (N,1) to (N,C) where N is the number of data 
    # and C the number of different classes
    train_y = hot_encoder(train_y)
    test_y = hot_encoder(test_y)
 
    return train_x, train_y, test_x, test_y

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

def minibatch (X,Y,N):

    X = X[:N]    
    Y = Y[:N]
    
    return X,Y

if __name__ == '__main__':
    
    path = r"D:\Windows.old.000\Users\Caique_\Desktop\Master\M2\Deep Learning\TD1\cifar-10-batches-py"
    batch = 1

    X,Y = lecture_cifar(path,batch)
    M = 5000
    X,Y = minibatch(X, Y, M) # keep only M images
    
    train_x, train_y, test_x, test_y = decoupage_donnees(X,Y)
 
    train_x, train_y, test_x, test_y = pre_process_data(train_x, train_y, test_x, test_y)
    
    train_x =  train_x.astype(np.float64)
    test_x =  test_x.astype(np.float64)
    
    X = [train_x,test_x]
    Y = [train_y,test_y]
    
    
    
    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))
 
    layers_dims = [100,10]
 
    ann = ANN(layers_dims)
    ann.fit(X, Y, 'relu',learning_rate=1, n_iterations = 5000 )
    
    print("Train Accuracy:", ann.predict(train_x, train_y,ann.relu))
    print("Test Accuracy:", ann.predict(test_x, test_y,ann.relu))
    ann.plot_cost()
    ann.plot_acc()
    ann.plot()
    