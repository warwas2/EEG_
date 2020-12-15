# -*- coding: utf-8 -*-
"""0324_aug_ssd_csp.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TMuv9H9PSzV0Q3twNHEjJWvZx8S-rRxq

# 이미지 불러오기
- 구글 드라이브에서 가져온 이미지
- subject4에 대해서만 테스트
- resizing (224, 224, 3)
"""

import glob
from google.colab import drive
drive.mount('/content/SSD8~30_CSP8~14')

pwd

!ls "/content/SSD8~30_CSP8~14/My Drive/SSD8~30_CSP8~14"

"""**데이터 shape 확인**"""

import glob 
import numpy as np

data = np.load('/content/SSD8~30_CSP8~14/My Drive/SSD8~30_CSP8~14/sub2_data.npy') #npy 데이터 로드
label = np.load('/content/SSD8~30_CSP8~14/My Drive/SSD8~30_CSP8~14/sub2_label.npy') #npy 데이터 로드

print(data.shape)
print(label.shape)
# print(data)

"""데이터 연결
- 기존 channel 개수 3개에서 CSP에 의해 shape이 (120,4)로 변경됨
- 의미는 다르지만 ch4를 만들어서 합침
- 기존 데이터는 left, right 데이터가 나눠져 있어서 합쳐줘야 했지만, 위의 데이터는 이미 합쳐져 있기 때문에 합칠 필요 X
"""

#label, data 파일 분류

label_files=[]
data_files=[]
for f in glob.glob("/content/SSD8~30_CSP8~14/My Drive/SSD8~30_CSP8~14/*.npy"):
  if "data" in f:
    data_files.append(f)
    # data = np.load(f)

  if "label" in f:
    label_files.append(f)
    # label = np.load(f)

label_files.sort()
data_files.sort()

print(label_files)
print()
print(data_files)

import pandas as pd
for subject in range(0,9):
  # print(subject)

  data = np.load(data_files[subject])
  label = np.load(label_files[subject])

  print(np.array(data).shape)
  print(np.array(label).shape)
  
  #전처리 함수(pre_func) 안에서 model_func 실행
  print("{}번째 subject".format(subject))
  pre_func(data, label)
  print()
  # break
  # test_print()

import random
import numpy as np
random.seed(100)

def pre_func(data, label):
  # 채널 나누기 (x, 4) : 4개로

  data=pd.DataFrame(data)

  ch1=data.iloc[:][0]
  ch2=data.iloc[:][1]
  ch3=data.iloc[:][2]
  ch4=data.iloc[:][3]

  print("채널 1~4 분리 완료")

  zz = list(zip(ch1,ch2,ch3,ch4,label))
  random.shuffle(zz)

  z1 = [a[0] for a in zz]
  z2 = [a[1] for a in zz]
  z3 = [a[2] for a in zz]
  z4 = [a[3] for a in zz] 

  zy = [a[4] for a in zz]

  print("shuffle 완료")

  # 0 ~ (split-1)까지 train 데이터
  # split ~ 끝까지 test 데이터
  split = int(0.8*(len(z1)))

  # train set
  x1_tr = z1[:split]
  x2_tr = z2[:split]
  x3_tr = z3[:split]
  x4_tr = z4[:split]
  y_tr = zy[:split]

  #test set
  x1_ts = z1[split:]
  x2_ts = z2[split:]
  x3_ts = z3[split:]
  x4_ts = z4[split:]
  y_ts = zy[split:]
  print("train, test 분리 완료:{}".format(split))

  x1_tr = np.array(x1_tr, dtype = 'float32') #(96,)
  x2_tr = np.array(x2_tr, dtype = 'float32')
  x3_tr = np.array(x3_tr, dtype = 'float32')
  x4_tr = np.array(x4_tr, dtype = 'float32')

  x1_tr = x1_tr.reshape(-1,1)
  x2_tr = x2_tr.reshape(-1,1)
  x3_tr = x3_tr.reshape(-1,1)
  x4_tr = x4_tr.reshape(-1,1)

  x1_ts = np.array(x1_ts, dtype = 'float32')
  x2_ts = np.array(x2_ts, dtype = 'float32')
  x3_ts = np.array(x3_ts, dtype = 'float32')
  x4_ts = np.array(x4_ts, dtype = 'float32')

  x1_ts = x1_ts.reshape(-1,1)
  x2_ts = x2_ts.reshape(-1,1)
  x3_ts = x3_ts.reshape(-1,1)
  x4_ts = x4_ts.reshape(-1,1)

  y_tr = np.array(y_tr, dtype = 'float32')
  y_ts = np.array(y_ts, dtype = 'float32')

  fx_tr = np.concatenate((x1_tr, x2_tr, x3_tr, x4_tr),axis=1)
  # print(fx_tr)
  fx_ts = np.concatenate((x1_ts, x2_ts, x3_ts, x4_ts), axis=1)
  # print(fx_ts)

  print("fx_tr.shape : {}".format(np.array(fx_tr).shape))
  # print("fx_tr(-1,1).shape : {}".format(np.array(test1).shape))
  # print("fx_tr(1,-1).shape : {}".format(np.array(test2).shape))
  print("fx_ts.shape : {}".format(np.array(fx_ts).shape))

  model_func(fx_tr, fx_ts, y_tr, y_ts)

#파일 하나 별로 돌아가는 거 확인하려고 만든 거
def test_print():
  print("확인")

"""###모델 실행
- subject별로 data, lable 불러온 뒤 모델 실행

## Train Set sklearn
"""

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Models : SVM, RF, KNN

def model_func(fx_tr, fx_ts, y_tr, y_ts):
  scaler = StandardScaler()
  scaler.fit(fx_tr)

  print(scaler.mean_)

  Xtr = scaler.transform(fx_tr)

  print(np.mean(Xtr))
  print(np.var(Xtr))

  class_weight = {0: 0.5, 1: 0.5} 

  # SVM
  clf1 = svm.SVC(kernel = 'linear', C = 10, class_weight = class_weight)
  scores = cross_val_score(clf1, Xtr, y_tr, cv = 10)
  print('SVM 10 fold CV Accuracy : ', scores.mean(), scores.std())

  # Random Forest
  clf2 = RandomForestClassifier(n_estimators = 10,
                              max_depth = None, 
                              min_samples_split = 2, 
                              random_state = 0, 
                              class_weight = class_weight)

  scores = cross_val_score(clf2, Xtr, y_tr, cv = 10)
  print('Random Forest 10 fold Accuracy : ',scores.mean(), scores.std())

  # KNN
  neigh = KNeighborsClassifier(n_neighbors = 3)

  scores = cross_val_score(neigh, Xtr, y_tr, cv=10)
  print('KNN 10 fold Accuracy : ', scores.mean(), scores.std())

  model_test_func(fx_tr, fx_ts, y_tr, y_ts, clf1, clf2, neigh, Xtr)

"""## Test Set sklearn"""

# test
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, average_precision_score, f1_score, accuracy_score, confusion_matrix

def model_test_func(fx_tr, fx_ts, y_tr, y_ts, clf1, clf2, neigh, Xtr):
  scaler = StandardScaler()
  scaler.fit(fx_ts)

  print(scaler.mean_)

  Xts = scaler.transform(fx_ts)

  print(np.mean(Xts)) # MEAN IS ZERO 
  print(np.var(Xts)) # VARIANCE IS 1

  #SVM Test
  clf1.fit(Xtr,y_tr)
  y_p1 = clf1.predict(Xts)
  print('SVM test accuracy : {}'.format(accuracy_score(y_ts, y_p1)))

  #RF Test
  clf2.fit(Xtr,y_tr)
  y_p1 = clf2.predict(Xts)    
  print('RF test accuracy : {}'.format(accuracy_score(y_ts, y_p1)))

  #KNN Test
  neigh.fit(Xtr,y_tr)
  y_p1 = neigh.predict(Xts)
  print('KNN test accuracy : {}'.format(accuracy_score(y_ts, y_p1)))