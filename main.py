# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 12:24:19 2019

@author: saurav
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocess
from Mymodel import mymodel
from conclusion import result

df_train=pd.read_csv('fer2013.csv')

X_train,X_test,y_train,y_test=preprocess(df_train)

model=mymodel()



model.save_weights('weights1.h5')
y_pred=model.predict(X_test)

result(history,y_pred,y_test)





