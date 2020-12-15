#!/usr/bin/env python
# coding: utf-8

# In[22]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import os,sys 
from IPython.display import Image


import os, re, glob
import cv2
import numpy as np
import shutil
from numpy import argmax
from keras.models import load_model

from sklearn.model_selection import train_test_split


# # 전체이미지

# In[50]:



path1='/Users/jaewon/Desktop/spectrogram/'
patharr=[]
for file in os.listdir(path1):
    patharr.append(file)

X=[]
Y=[]

image_w = 30
image_h = 30

for i in patharr:
    temp=[]
    for file in os.listdir(path1+i+'/'):
        img=cv2.imread(path1+i+'/'+file)
        img=cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
        if "chan01" in file:
            img1=img
        elif "chan02" in file:
            img2=img
        elif "chan03" in file:
            img3=img
            s=cv2.vconcat([img1,img2,img3])
            X.append(s)
            if "right" in file:
                Y.append(0)
            elif "left in file":
                Y.append(1)
            else:
                print("error")
        else:
            print("error")

X = np.array(X)
Y = np.array(Y)
 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)


# # 일부이미지

# In[54]:


import os, re, glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


filepath='/Users/jaewon/Desktop/spectrogram/B0301T/'
X=[]
Y=[]

image_w = 120
image_h = 120

def getimg(filepath):
    for file in os.listdir(filepath):
        temp=[]
        img=cv2.imread(filepath+file)
        img=cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
        if "chan01" in file:
            img1=img
        elif "chan02" in file:
            img2=img
        elif "chan03" in file:
            img3=img
            s=cv2.vconcat([img1,img2,img3])
            X.append(s)
            if "right" in file:
                Y.append(0)
            elif "left in file":
                Y.append(1)
            else:
                print("error")
        else:
            print("error")
    
    
getimg(filepath)
filepath='/Users/jaewon/Desktop/spectrogram/B0302T/'   
getimg(filepath)

X = np.array(X)
Y = np.array(Y)
 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)


# # 학습코드

# In[60]:





#학습용
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(2, (3, 3), padding='same', activation='relu', input_shape=(image_w*3,image_h,3)))
#model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))


model.add(tf.keras.layers.Conv2D(2, (2, 2), padding='same', activation='relu'))
#model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))


model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(227, activation='relu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(2, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=20, nb_epoch=50)


# In[61]:


test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2) #실제 test_data 테스트 결과


# In[ ]:




