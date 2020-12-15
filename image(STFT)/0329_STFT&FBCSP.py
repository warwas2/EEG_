from src.data_preparation.data_preparation_v2_연속해서overlap import read_eeg_file
from src.algorithms.csp.CSP_with_explain import CSP
from scipy import signal
from src.algorithms.fbcsp.MIBIFFeatureSelection import MIBIFFeatureSelection
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

TIME_LENGTH = 2000
TIME_WINDOW = 2000
EPOCH_SIZE = 2000
DATA_FOLDER = "testdata"
K_FOLD = 10
FS = 250
subjects = range(4, 10)
accuracies = {
#    "GNB": np.zeros((len(subjects), K_FOLD)),
#    "SVM": np.zeros((len(subjects), K_FOLD)),
#    "LDA": np.zeros((len(subjects), K_FOLD))
}

band_length =2
min_freq = 8
max_freq = 30
bands = [(x, x+band_length) for x in range(min_freq, max_freq, band_length)]
quantity_bands = len(bands)

del band_length
del min_freq
del max_freq

def filter_bank(x):
    data = np.zeros((quantity_bands, *x.shape))
    for n_trial in range(x.shape[0]):
        trial = x[n_trial, :, :]
        filter_bank = np.zeros((quantity_bands, *trial.shape))

        for (i, (low_freq, high_freq)) in enumerate(bands):
            # Create a 5 order Chebyshev Type 2 filter to the specific band (low_freq - high_freq)
            b, a = signal.cheby2(2, 14, [low_freq, high_freq], btype="bandpass", fs=FS)

            filter_bank[i, :, :] = signal.filtfilt(b, a, trial, axis=0)
        data[:, n_trial, :, :] = filter_bank

    return data

print("Loading data ...")
data_by_subject = []

for subject in subjects:
    left_data_file = f"{DATA_FOLDER}/ssd-{subject}-left.csv"
    right_data_file = f"{DATA_FOLDER}/ssd-{subject}-right.csv"
    
#    데이터에 EEG 형태로 저장
#    오버래핑 시킨 상태로 데이터를 가져옴 
    data = read_eeg_file(left_data_file, right_data_file, TIME_LENGTH, TIME_WINDOW, EPOCH_SIZE)
    data.X = np.concatenate((data.left_data, data.right_data))
    data_by_subject.append(data)

del subject
del left_data_file
del right_data_file
del data

print("Data loaded")

for (i, data) in enumerate(data_by_subject):
    print("Subject: ", i+1)

    cv = KFold(n_splits=K_FOLD, shuffle=True)
    for (k, (train_index, test_index)) in enumerate(cv.split(data.X)):
        trials = len(data.left_data)
#        왼쪽이 0, 오른쪽이 1
#        print(trials) # sub 1의 left : 581, right : 601
#        print("왼쪽 데이터 shape : {}".format(data.left_data.shape))

        train_left_index = [index for index in train_index if index < trials]
        train_right_index = [index - trials for index in train_index if index >= trials]
        X_left_train, X_right_train = data.left_data[train_left_index], data.right_data[train_right_index]
        
#        print("훈련 데이터") #(526, 2000, 3), (537, 2000, 3)
#        print("X_left_train.shape : {}, X_right_train.shape{}".format(X_left_train.shape,X_right_train.shape))

        test_left_index = [index for index in test_index if index < trials]
        test_right_index = [index - trials for index in test_index if index >= trials]
        X_left_test, X_right_test = data.left_data[test_left_index], data.right_data[test_right_index]

#        print("시험 데이터")
#        print("X_left_test.shape : {}, X_right_test.shape{}".format(X_left_test.shape,X_right_test.shape))
#        KFold에 의해 훈련, 시험 데이터 랜덤 비율로 나뉨

        y_train, y_test = data.labels[train_index], data.labels[test_index]

        # Feature extraction
#        필터 개수가 11
        N_CSP_COMPONENTS = 2 #기존 #csp_by_band = (11,)
        csp_by_band = [CSP(average_trial_covariance=False, n_components=N_CSP_COMPONENTS)
                       for _ in bands]
        
#        print("csp_by_band{}".format(np.array(csp_by_band).shape))

        left_bands_training = filter_bank(X_left_train)
        right_bands_training = filter_bank(X_right_train)
        left_bands_test = filter_bank(X_left_test)
        right_bands_test = filter_bank(X_right_test)
        
#        각각 왼쪽, 오른쪽 train 데이터에 대한 11개 필터와 동일
#        left_bands_training : (11, 534, 2000, 3), right_bands_training : (11, 529, 2000, 3)
#        print("left_bands_training : {}, right_bands_training : {}"
#              .format(left_bands_training.shape,right_bands_training.shape))

        features_train = None
        features_test = None
        for n_band in range(quantity_bands):
            left_band_training = left_bands_training[n_band]
            right_band_training = right_bands_training[n_band]
            left_band_test = left_bands_test[n_band]
            right_band_test = right_bands_test[n_band]

            csp = csp_by_band[n_band]
            csp.fit(left_band_training, right_band_training)
            
#            print("csp 왼쪽 데이터 형태:{}".format(csp.left_data.shape)) # ex:(516, 2000, 3)
            x_train = np.concatenate((left_band_training, right_band_training))
            x_test = np.concatenate((left_band_test, right_band_test))
            
#            print(x_train.shape) # ex:(1063, 2000, 3) -> 왼쪽 오른쪽 데이터를 concatenate

            if n_band == 0:
                features_train = csp.compute_features(x_train)
                features_test = csp.compute_features(x_test)
                
#                첫번째 밴드일 때 compute_features 결과 
#                print(np.array(features_train).shape) #(1063, 2)
            else:
#                2씩 증가 ex:(359,4) -> (359,6)
                features_train = np.concatenate((features_train, csp.compute_features(x_train)), axis=1)
                features_test = np.concatenate((features_test, csp.compute_features(x_test)), axis=1)

#        n_band(11번) 다 돌고나서 결과
#        print(np.array(features_train).shape) #(149,22)

        # Feature Selection
        selected_features = MIBIFFeatureSelection(features_train, features_test, 
                                                  y_train, N_CSP_COMPONENTS, 4, scale=True)

        selected_training_features = selected_features.training_features
        selected_test_features = selected_features.test_features
        
        selected_features = np.concatenate((selected_training_features,selected_test_features))
        
        label_all = np.concatenate((y_train,y_test))
        
#        print(np.array(selected_training_features).shape) #(1063, 22)
#        print(selected_training_features[0])
#        print(np.array(selected_test_features).shape) #(119, 22)
#        print(np.array(selected_features).shape) #(1182, 22)
#        
#        print()
#        print(np.array(y_train).shape) #(1063,)
#        print(np.array(y_test).shape) #(119,)
#        print(np.array(label_all).shape) #(1182,)
#        
#        break
        
    cnt=1
    flag = ''
    for idx_features,data_features in enumerate(selected_features):
        if(label_all[idx_features]==0):
            flag='left'
        else:
            flag='right'
        
#        for idx in range(0,22):
        subnumber = i+4
        imgname="all_images/sub"+str(subnumber)+"/sub"+str(subnumber)+"_"+flag+"_"+str(cnt)+".png"
        print(imgname)
        
#        print(np.array(data_features).shape) #(22,)
        f, t, Z = signal.stft(data_features, fs=FS, nperseg=1)
        Zr = abs(Z)
        plot.pcolormesh(Zr)
        plot.ylim(0,1)
        plot.axis('off')
#        plot.show()
        plot.savefig(imgname, bbox_inches = 'tight', pad_inches = 0, transparent = True)
        cnt=cnt+1