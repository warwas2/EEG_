import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

TYPE=1 #이미지 옆으로 쌓을지, 뒤로 쌓을지

#filepath='/Users/etri-sw-soc/Desktop/si/'
filepath="./seven/"
testfile = filepath+"sub7_left_1_ch1.png"
testimg = cv2.imread(testfile,cv2.IMREAD_COLOR)
img_height,img_width,img_channels = testimg.shape

#print(img_height)
#print(img_width)
#print(img_channels)

#cv2.imshow('testimg_title',testimg)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

X=[]
Y=[]

#image_w = int(img_width*0.2)#66
#image_h = int(img_height*0.2)#43
image_w = 32   
image_h = 32
#print(image_w, image_h)

sorted_file=[]
def sort(filepath):
    for file in os.listdir(filepath):
        sorted_file.append(file)
        
    sorted_file.sort()
#    for file in sorted_file:
#        print("file:{}".format(file))

def getimg(filepath):
    for file in sorted_file:
        img=cv2.imread(filepath+file)
        img=cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
        
#        print("file:{}".format(file))
        
        if "ch1" in file:
            img1=img
            file_now=file[:-7]
#            print("file_now:{}".format(file_now))
        elif "ch2" in file:
            img2=img
            if file_now!=file[:-7]:
                print("Error_img2")
        elif "ch3" in file:
            img3=img
            if file_now!=file[:-7]:
                print("Error_img3")
                
            if TYPE==1:#이미지를 뒤로 쌓을때
                s_fir=[] 
                for fir in range(image_h):
                    s_sec=[]
                    for sec in range(image_w):
                        s_sec.append(np.concatenate((img1[fir][sec], img2[fir][sec],img3[fir][sec]), axis=None))
                    s_fir.append(s_sec)
                X.append(s_fir)
#                print("X 추가:{}".format(np.array(X).shape))
                
#                print(np.array(s_fir).shape)
            
#            else:#이미지를 아래로 연결할때
#                s=cv2.vconcat([img1,img2,img3]) 
#                X.append(s) 
                
            if "right" in file:
                Y.append(0)
#                print("Y 추가:{}".format(np.array(Y).shape))
            elif "left" in file:
                Y.append(1)
#                print("Y 추가:{}".format(np.array(Y).shape))
            else:
                print("error_in_third")
                
            print("{} 채널 합침".format(file_now)) 
        else:
            print("error_in_first")
        
sort(filepath)
getimg(filepath)
print("*** 채널 합치기 완료 ***")

X = np.array(X)
Y = np.array(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20,shuffle=True)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, LSTM, GaussianNoise, Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras.optimizers import Adam

batch_size = 16
num_classes = 2
epochs = 200
epoch_count=1
input_shape = (image_h, image_w, 9)
input_shape2 = (image_h*image_w, 9)

y_train=np_utils.to_categorical(y_train,num_classes)
y_test=np_utils.to_categorical(y_test,num_classes)

#model = Sequential()
#model.add(Conv2D(64, (3, 3), padding='same',input_shape=input_shape))
#model.add(Activation('relu'))
#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#
#model.add(Flatten())
#model.add(Dense(10,activation='softmax'))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(num_classes,activation='softmax'))
#model.add(Activation('softmax'))

# initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)

#model.summary()
## Let's train the model using RMSprop
#model.compile(loss='categorical_crossentropy',
#              optimizer=opt,
#              metrics=['accuracy'])
#
#x_train = x_train.astype('float32')
#x_test = x_test.astype('float32')
##x_train /= 255
##x_test /= 255
#
#model.fit(x_train, y_train,
#          batch_size=batch_size,
#          epochs=epochs,
#          validation_data=(x_test, y_test),
#          shuffle=True)

# ***** LSTM try *****
dim1, dim2 = x_train.shape[1],x_train.shape[2]
#print(dim1, dim2)
model = Sequential()
model.add(Conv1D(64,3,activation='relu',input_shape=(dim1,dim2)))
model.add(MaxPooling1D(3))
#
##model.add(Conv2D(64, (3, 3), padding='same',input_shape=input_shape))
##model.add(Activation('relu'))
##model.add(Conv2D(64, (3, 3)))
##model.add(Activation('relu'))
##model.add(MaxPooling2D(pool_size=(2, 2)))
##model.add(Dropout(0.25))
#
model.add(LSTM(64,activation='tanh',return_sequences=True))
model.add(LSTM(64,activation='tanh',return_sequences=True))
model.add(GaussianNoise(stddev=0.5))
model.add(Flatten())
#model.add(Dropout(0.5))
#
model.add(Dense(num_classes,activation='softmax'))
#
opt = Adam(lr=0.0011,decay=0.001)
#
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
#model.build(input_shape)
model.summary()
model.fit(x_train,y_train,
          batch_size=batch_size,
          epochs=epoch_count,
          shuffle=False,
          validation_split=0.1, 
          verbose=1)
