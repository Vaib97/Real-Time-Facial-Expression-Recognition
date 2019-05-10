# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 12:03:55 2019

@author: saurav
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def preprocess(df_train):
    X=df_train['pixels'].values
    X=X.reshape(35887,1)
    X_train=np.zeros((30000,2304))
    X_test=np.zeros((5887,2304))
    
    
    for i in range(0,30000):
        t=X[i][0].split(' ')
        for j in range(0,2304):
            X_train[i][j]=t[j]
    for k in range(30000,35887):
        t=X[k][0].split(' ')
        for j in range(0,2304):
            X_test[k-30000][j]=t[j]
            
    from sklearn.preprocessing import StandardScaler
    st=StandardScaler()
    X_train=st.fit_transform(X_train)
    X_test=st.transform(X_test)
    
    pics=X_train.reshape(30000,48,48)
    pic_train=pics.reshape(30000,48,48,1)
    index=2
    plt.imshow(pics[index])
    pic_test=X_test.reshape(5887,48,48,1)
    
    y=df_train['emotion'].values
    y=y.reshape(35887,1)
    y_train=y[0:30000,:]
    y_test=y[30000:35887,:]
    
    
    from sklearn.preprocessing import OneHotEncoder
    en=OneHotEncoder(categorical_features=[0])
    y_train=en.fit_transform(y_train).toarray()
    
    return pic_train,pic_test,y_train,y_test
    
    
    
    
    
    
    
    
    
    
    
    
    
    