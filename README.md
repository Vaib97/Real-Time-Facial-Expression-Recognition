# Real-Time-Facial-Expression-Recognition
In this project, I have created convolutional neural networks (CNN) for real time facial expression recognition task. In this we have used dataset of fer2013 which contain 35887 grey 48X48 images. we have used webcam for live prediction of our facial expression.
first i have done data preprocessing in which i have imported CSV file and standardize the image pixels(preprocessing.py contain its code)
then i have created my CNN architecture which has 4 CNN layers and 2 ANN and an output layer in which i have used binary cross entropy as loss function and sigmoid as an activation function in output layer(Mymodel.py contain its code).
then i calculated the accuracy ,plotted the graph between loss vs epoh and accuracy vs epoh and also determine the confusion matrix(conclusion.py contain its code).
then i have created the main.py in which i run all the previosuly created function.
then i have used the openCv to capture the live video from webcam and we have cropped the faces and resized it into 48x48 resolution and converted it into greyscale images and then using the trained model i have predicted the emotion of the image(cam.py contain its code).
link for dataset:https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
