# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 22:10:28 2019

@author: saurav
"""

import cv2
import sys
import matplotlib as plt
import Mymodel as mpm
from keras.preprocessing import image
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import matplotlib as plt


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

model = mpm.mymodel('weights.h5')

emo     = ['Angry', 'disgust', 'fear',
           'happy', 'sad','surprise', 'Neutral']

while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(48, 48)
    )
    for (x, y, w, h) in faces:
        img=cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        im=img
    cv2.imshow('Video', frame)
    detected_face = frame[int(y):int(y+h), int(x):int(x+w)]
    detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
    detected_face = cv2.resize(detected_face, (48, 48))
    detected_face = detected_face.astype(np.float64)
    st=StandardScaler()
    img_pixels=st.fit_transform(detected_face)
    img_pixels=img_pixels.reshape(1,48,48,1)
    
    predictions = model.predict(img_pixels)
    max_index = np.argmax(predictions[0])
    emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    emotion = emotions[max_index]
    print(emotion)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()