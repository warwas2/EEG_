# -*- coding: utf-8 -*-
"""0329_FBCSP_spectrogram_SVM&RF&KNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rK6xTAQewZCuGIJ7F99PsZJibLsAwqz2

# 이미지 불러오기
- 구글 드라이브에서 가져온 이미지
- subject4에 대해서만 테스트
- resizing (224, 224, 3)
"""

''' ***** 이미지 가져오기 ***** '''
import glob
# 채널 구분이 없기 때문에 왼쪽, 오른쪽으로만 나누면 됨

# 왼쪽 데이터
left_images= []

# 오른쪽 데이터
right_images = []

images_path = "C:/Users/user/Desktop/sub1/"

for f in glob.glob("C:/Users/user/Desktop/sub1/*.png"):
  if "left" in f:
    left_images.append(f)
  elif "right" in f:
    right_images.append(f)
  else:
    print("left/right 파일 분류 오류")

print(len(left_images))
print(len(right_images))

left_images.sort()
right_images.sort()

''' ***** 이미지 리사이징 ***** '''
import cv2

# resize 후 "전체 왼쪽이미지" + "전체 오른쪽이미지" 
x = [] # 데이터 : left + right
y = [] # 라벨

# left = 0
for i in range(len(left_images)):
  x.append(cv2.resize(cv2.imread(left_images[i]),(224,224)))
  y.append(0)

# right = 1
for i in range(len(right_images)):
  x.append(cv2.resize(cv2.imread(right_images[i]),(224,224)))
  y.append(1)

print("resizing 완료")

''' ***** Augmentation ***** '''
#! pip install Augmentor

z = list(zip(x))

print(len((z)))

def unzip(z):
  x_p = [a[0] for a in z]
  return x_p

import random
random.seed(1997)

import Augmentor

p = Augmentor.DataPipeline(z,y)

p.rotate(probability = 0.2,
         max_left_rotation = 5, 
         max_right_rotation = 5)
p.flip_left_right(probability = 0.2)
p.zoom_random(probability = 0.1, percentage_area = 0.8)
p.flip_top_bottom(probability = 0.3)
p.gaussian_distortion(probability = 0.05, grid_width = 4, grid_height = 4, magnitude = 3, 
                      corner = 'bell', method = 'in', mex = 0.5, mey = 0.5, sdx = 0.05, sdy = 0.05)
p.random_brightness(probability = 0.05, min_factor = 0.7, max_factor = 1.3)
p.random_color(probability = 0.05, min_factor = 0.6, max_factor = 0.9)
p.random_contrast(probability = 0.05, min_factor = 0.6, max_factor = 0.9)
p.random_distortion(probability = 0.2, grid_width = 4, grid_height = 4, magnitude = 2)

augmented_images, labels = p.sample(1000)

x_p = unzip(augmented_images)
y_p = labels

#print(augmented_images[0][0].shape)

# 기존 배열에 Augmented 이미지 추가 
x.extend(x_p)
y.extend(y_p)

# (sub1기준) (1182+1000)장의 이미지
print(len(x))
print(len(y))

import random
random.seed(100)

zz = list(zip(x,y))
random.shuffle(zz)

z1 = [a[0] for a in zz]
zy = [a[1] for a in zz]

# 0 ~ (split-1)까지 train 데이터
# split ~ 끝까지 test 데이터

split = int(0.8*(len(z1)))
print(split)

# train set
x1_tr = z1[:split]
y_tr = zy[:split]

#test set
x1_ts = z1[split:]
y_ts = zy[split:]

import numpy as np

x1_tr = np.array(x1_tr, dtype = 'float32')
#print(x1_tr.shape) #(1745, 224, 224, 3)

y_tr = np.array(y_tr, dtype = 'float32')
y_ts = np.array(y_ts, dtype = 'float32')

x1_tr = x1_tr/255.

print(np.mean(x1_tr))

x1_ts = np.array(x1_ts, dtype = 'float32')

print(np.mean(x1_ts))

x1_ts = x1_ts/255.

print(np.mean(x1_ts))


''' ***** 모델 실행 ***** '''
from keras.applications.resnet50 import ResNet50
#from keras.preprocessing import image
#from keras.applications.resnet50 import preprocess_input

model = ResNet50(weights='imagenet', include_top=False, pooling="avg")

print("x1_tr.shape{}".format(x1_tr.shape))

''' ***** TRAIN SET PREDICT ***** '''
fx1_tr = model.predict(x1_tr)
print("fx1_tr.shape{}".format(fx1_tr.shape))

#fx_tr = np.concatenate((fx1_tr),axis=1)
fx_tr = fx1_tr
print("fx_tr.shape{}".format(fx_tr.shape))

''' ***** TEST SET PREDICT ***** '''
fx1_ts = model.predict(x1_ts)
print("fx1_ts.shape{}".format(fx1_ts.shape))

#fx_ts = np.concatenate((fx1_ts), axis=1)
fx_ts = fx1_ts
print("fx_ts.shape{}".format(fx_ts.shape))


''' ***** TRAIN SET SKLEARN ***** '''
#train
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(fx_tr)

print(scaler.mean_)

Xtr = scaler.transform(fx_tr)

print(np.mean(Xtr))
print(np.var(Xtr))

class_weight = {0: 0.5, 1: 0.5}

''' SVM '''
from sklearn.model_selection import cross_val_score
from sklearn import svm

clf1 = svm.SVC(kernel = 'linear',
               C = 10, 
               class_weight = class_weight)

scores = cross_val_score(clf1, Xtr, y_tr, cv = 10)
print('SVM 10 fold CV Accuracy : ', scores.mean(), scores.std())

''' RF '''
from sklearn.ensemble import RandomForestClassifier

clf2 = RandomForestClassifier(n_estimators = 10,
                              max_depth = None, 
                              min_samples_split = 2, 
                              random_state = 0, 
                              class_weight = class_weight)

scores = cross_val_score(clf2, Xtr, y_tr, cv = 10)
print('Random Forest 10 fold Accuracy : ',scores.mean(), scores.std())

''' KNN '''
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors = 3)

scores = cross_val_score(neigh, Xtr, y_tr, cv=10)
print('KNN 10 fold Accuracy : ', scores.mean(), scores.std())

''' ***** TEST SET SKLEARN ***** '''

# test
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, average_precision_score, f1_score, accuracy_score, confusion_matrix

scaler = StandardScaler()
scaler.fit(fx_ts)

print(scaler.mean_)

Xts = scaler.transform(fx_ts)

print(np.mean(Xts)) # MEAN IS ZERO 
print(np.var(Xts)) # VARIANCE IS 1

''' SVM TEST '''
clf1.fit(Xtr,y_tr)
y_p1 = clf1.predict(Xts)
print('SVM test accuracy : {}'.format(accuracy_score(y_ts, y_p1)))

''' RF TEST '''
clf2.fit(Xtr,y_tr)
y_p1 = clf2.predict(Xts)    
print('RF test accuracy : {}'.format(accuracy_score(y_ts, y_p1)))

''' KNN TEST '''
neigh.fit(Xtr,y_tr)
y_p1 = neigh.predict(Xts)
print('KNN test accuracy : {}'.format(accuracy_score(y_ts, y_p1)))