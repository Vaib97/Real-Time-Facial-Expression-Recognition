# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 11:57:38 2019

@author: saurav
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization

def mymodel(weights_path=None):
    classifier=Sequential()
    classifier.add(Convolution2D(filters=64,kernel_size=(3,3),padding='same',activation='relu',input_shape=(48,48,1)))
    classifier.add(Dropout(0.3))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    classifier.add(Convolution2D(filters=128,kernel_size=(5,5),padding='same',activation='relu'))
    classifier.add(Dropout(0.3))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    classifier.add(Convolution2D(filters=256,kernel_size=(3,3),padding='same',activation='relu'))
    classifier.add(Dropout(0.3))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    classifier.add(Convolution2D(filters=256,kernel_size=(3,3),padding='same',activation='relu'))
    classifier.add(Dropout(0.3))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim=256,activation='relu'))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(output_dim=512,activation='relu'))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(output_dim=7,activation='sigmoid'))
    print ("Create model successfully")
    if weights_path:
        classifier.load_weights(weights_path)
    
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    
    return classifier
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    