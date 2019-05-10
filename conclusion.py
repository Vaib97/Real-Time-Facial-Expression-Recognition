# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 12:12:51 2019

@author: saurav
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def result(history,y_pred,y_test):
    plt.plot(history.history['acc'])
    
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['loss'])
    
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    y_prediction=np.zeros((5887,1))
    for i in range(0,5887):
        max=y_pred[i][0]
        m_index=0
        for j in range(1,7):
            if max<y_pred[i][j]:
                max=y_pred[i][j]
                m_index=j
            y_prediction[i][0]=m_index
    
    from sklearn.metrics import accuracy_score
    accuracy=accuracy_score(y_prediction,y_test)
    print(accuracy)
    
    confusion=np.zeros((7,7))
    from sklearn.metrics import confusion_matrix
    confusion=confusion_matrix(y_test, y_prediction)
    print(confusion)
    
    ar=np.zeros((7,1))
    for i in range(0,7):
        sum=0
        for j in range(0,7):
            sum=sum+confusion[j][i]
        ar[i]=(confusion[i][i]*100)/sum
    print("angry = ",ar[0])
    print("disgust = ",ar[1])
    print("fear = ",ar[2])
    print("happy = ",ar[3])
    print("sad = ",ar[4])
    print("surprise = ",ar[5])
    print("neutral = ",ar[6])