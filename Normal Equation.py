# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 12:44:36 2019

@author: dhruvil
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class NormalEquation:
    
    def __init__(self):
        
        dataset = pd.read_csv("3D_spatial_network.csv")
        dataset=(dataset-dataset.min())/(dataset.max()-dataset.min())
        self.X = dataset.iloc[:,1:3].values
        self.Y = dataset.iloc[:,3].values
        self.X_train,self.X_test,self.Y_train,self.Y_test=train_test_split(self.X,self.Y,test_size=0.3,random_state=1)
        self.X_train = list(self.X_train)
        self.Y_train = list(self.Y_train)
        self.X_test = list(self.X_test)
        self.Y_test = list(self.Y_test)
        
    
    def ModelTraining(self):
        
        # Solve Equation (X^T*X)^-1*Y
        X = []
        for i in range(len(self.X_train)):
            n=[]
            n.append(1)
            for j in range(len(self.X_train[i])):
                n.append(self.X_train[i][j])
            X.append(n)
        X_trans = np.transpose(X)
        XTX = np.dot(X_trans,X)
        XTX_inv = np.linalg.inv(XTX)
        XTX_inv = np.dot(XTX_inv,X_trans)
        Param = np.dot(XTX_inv,self.Y_train)
        
        return list(Param)
    
    def Predicted(self):
        
        Values = self.ModelTraining()
        Y_pred=[]
        for i in range(len(self.X_test)):
            ans = Values[0] + Values[1]*self.X_test[i][0] + Values[2]*self.X_test[i][1]
            Y_pred.append(ans)
        
        return Y_pred,Values
    
def RMSE(Y1,Y2):
    rmse = 0
    for i in range(len(Y1)):
        rmse += (Y1[i]-Y2[i])*(Y1[i]-Y2[i])
    rmse/=len(Y1)
    return rmse**0.5
   
if __name__ == '__main__':
    
    ne = NormalEquation()
    Y_actual = ne.Y_test
    Y_pred , Param = ne.Predicted()
    print(RMSE(Y_actual,Y_pred))
    print (Param)
   # print(Y_pred)
            
        
        
        
        

        