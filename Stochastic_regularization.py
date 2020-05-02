# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 20:03:47 2019

@author: dhruvil
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def sign(a):
    if a==0:
        return 0
    if a>0:
        return 1
    return -1

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

class GradientDescent:
    
    def __init__(self):
        
        dataset = pd.read_csv("3D_spatial_network.csv")
        #print(dataset)
        #dataset = normalize(dataset)
        dataset=(dataset-dataset.min())/(dataset.max()-dataset.min())
       # print(dataset)
        self.X=dataset.iloc[:,1:3].values
        self.Y=dataset.iloc[:,3].values
        
        self.X_train,self.X_test,self.Y_train,self.Y_test=train_test_split(self.X,self.Y,test_size=0.3,random_state=8)
        self.X_train = list(self.X_train)
        self.Y_train = list(self.Y_train)
        self.X_test = list(self.X_test)
        self.Y_test = list(self.Y_test)
        #print (self.X_train)
        self.W0 = 0
        self.W1 = 0
        self.W2 = 0
        self.rate = 0.0000015
        self.reg_coeff= 0.001
    
    
    def sumofError(self):
        diff_w0 = 0
        diff_w1 = 0
        diff_w2 = 0
        
        for i in range(len(self.X_train)):
            diff_w0 += self.W0 + self.W1*self.X_train[i][0] + self.W2*self.X_train[i][1] - self.Y_train[i]
            diff_w1 += (self.W0 + self.W1*self.X_train[i][0] + self.W2*self.X_train[i][1] - self.Y_train[i])*self.X_train[i][0]
            diff_w2 += (self.W0 + self.W1*self.X_train[i][0] + self.W2*self.X_train[i][1] - self.Y_train[i])*self.X_train[i][1]
        
        return diff_w0,diff_w1,diff_w2
    
    
    def Model_training_L2(self):
        
        for i in range(20):
            
            diff_w0,diff_w1,diff_w2 = self.sumofError()
            self.W0 -= self.rate*(diff_w0 + 2*self.reg_coeff*self.W0)
            self.W1 -= self.rate*(diff_w1 + 2*self.reg_coeff*self.W1)
            self.W2 -= self.rate*(diff_w2 + 2*self.reg_coeff*self.W2)
            #print(self.W0,self.W1,self.W2,diff_w0,diff_w1,diff_w2)
    
    def Model_training_L1(self):
        
        for i in range(20):
            
            diff_w0,diff_w1,diff_w2 = self.sumofError()
            self.W0 -= self.rate*(diff_w0 + self.reg_coeff*sign(self.W0))
            self.W1 -= self.rate*(diff_w1 + self.reg_coeff*sign(self.W1))
            self.W2 -= self.rate*(diff_w2 + self.reg_coeff*sign(self.W2))
    
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
    
    L2 = GradientDescent()
    Y2_actual = L2.Y_test
    L2.Model_training_L2()
    Y2_pred = L2.Predict()
    #print (Y_pred)
    print(L2.W0,L2.W1,L2.W2)
    print(RMSE(Y2_pred,Y2_actual))
    L1 = GradientDescent()
    Y1_actual = L1.Y_test
    L1.Model_training_L1()
    Y1_pred = L1.Predict()
    #print (Y_pred)
    print(L1.W0,L1.W1,L1.W2)
    print(RMSE(Y1_pred,Y1_actual))
    
    
    
    
    
    
        
        
        
        
        
    
