# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 00:44:18 2019

@author: dhruvil
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 20:03:47 2019

@author: dhruvil
"""


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
X_axis=[]
Y_axis=[]
class StochasticGradientDescent:
    
    def __init__(self):
        
        dataset = pd.read_csv("3D_spatial_network.csv")
        #print(dataset)
        dataset=(dataset-dataset.min())/(dataset.max()-dataset.min())
        self.X=dataset.iloc[:,1:3].values
        self.Y=dataset.iloc[:,3].values
        
        self.X_train,self.X_test,self.Y_train,self.Y_test=train_test_split(self.X,self.Y,test_size=0.2,random_state=8)
        self.X_train = list(self.X_train)
        self.Y_train = list(self.Y_train)
        self.X_test = list(self.X_test)
        self.Y_test = list(self.Y_test)
        #print (self.X_train)
        self.W0 = 0
        self.W1 = 0
        self.W2 = 0
        self.rate = 0.0000015
    
    
    def Model_training(self):
        
        for i in range(400zz):
            for j in range(len(self.X_train)):
                val = self.W0 + self.W1*self.X_train[j][0] + self.W2*self.X_train[j][1] - self.Y_train[j]
                self.W0 -= self.rate*(val)
                self.W1 -= self.rate*(val*self.X_train[i][0])
                self.W2 -= self.rate*(val*self.X_train[i][1])
                #print(self.W0,self.W1,self.W2,diff_w0,diff_w1,diff_w2)
            if i%20==0:
                X_axis.append(i)
                Y_axis.append(self.Loss())
            #print (i)
    def Loss(self):
        loss=0
        for i in range(len(self.X_train)):
            loss += (self.W0 + self.W1*self.X_train[i][0] + self.W2*self.X_train[i][1] - self.Y_train[i])**2
        loss/=2
        return loss 
    
    def Predict(self):
        
        Y_pred = []
        for i in range(len(self.X_test)):
            
            val = self.W0 + self.W1*self.X_test[i][0] + self.W2*self.X_test[i][1]
            Y_pred.append(val)
        return Y_pred
    

def RMSE(Y1,Y2):
    rmse = 0
    for i in range(len(Y1)):
        rmse += (Y1[i]-Y2[i])*(Y1[i]-Y2[i])
    rmse/=len(Y1)
    return rmse**0.5
   
if __name__ == '__main__':
    
    sgd = StochasticGradientDescent()
    Y_actual = sgd.Y_test
    sgd.Model_training()
    Y_pred = sgd.Predict()
    #print (Y_pred)
    print(sgd.W0,sgd.W1,sgd.W2)
    print(RMSE(Y_pred,Y_actual))
    print(r2_score(sgd.Y_test,Y_pred ))
    plt.plot(X_axis,Y_axis)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('StochasticGradientDescent')
    plt.show()
    plt.savefig("StochasticGradientDescent.png")
    sequence_containing_x_vals = list(np.transpose(sgd.X_test)[0])
    sequence_containing_y_vals = list(np.transpose(sgd.X_test)[1])
    sequence_containing_z_vals = list(sgd.Y_test)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals,sequence_containing_z_vals,c="green")
    x = np.linspace(-0.2,1,110)
    y = np.linspace(-0.2,1,110)
    X,Y = np.meshgrid(x,y)
    Z = sgd.W0 + (sgd.W1 * X) + (sgd.W2 * Y)
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_zlabel('Altitude', fontsize=10)
    fig.savefig("ActualvsPredicted.png")
    
    
    
    
        
        
        
        
        
    
