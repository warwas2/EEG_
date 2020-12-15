from src.data_preparation.data_preparation import read_eeg_file
from src.algorithms.csp.CSP import CSP
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
TIME_WINDOW = 1500
EPOCH_SIZE = None
DATA_FOLDER = "testdata"
K_FOLD = 10
FS = 250
#4, 5이 우선 acc이 높아서 둘로 실행
subjects = range(4, 5)
accuracies = {
#    "GNB": np.zeros((len(subjects), K_FOLD)),
#    "SVM": np.zeros((len(subjects), K_FOLD)),
    "LDA": np.zeros((len(subjects), K_FOLD))
}

band_length = 4
min_freq = 4
max_freq = 40
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
            b, a = signal.cheby2(5, 48, [low_freq, high_freq], btype="bandpass", fs=250)

            filter_bank[i, :, :] = signal.filtfilt(b, a, trial, axis=0)
        data[:, n_trial, :, :] = filter_bank

    return data

print("Loading data ...")
data_by_subject = []

for subject in subjects:
    left_data_file = f"{DATA_FOLDER}/{subject}-left.csv"
    right_data_file = f"{DATA_FOLDER}/{subject}-right.csv"
    
#    left_data_file = f"{DATA_FOLDER}/1-left.csv"
#    right_data_file = f"{DATA_FOLDER}/1-right.csv"
    
#    데이터에 EEG 형태로 저장
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
#        print(trials) 왼쪽이 0, 오른쪽이 1
#        sub 4는 201개의 trial, sub 5는 193개의 trial(왼쪽)
#        print(trials)
#        print("왼쪽 데이터 shape : {}".format(data.left_data.shape))
#        print(data.labels[74])

        train_left_index = [index for index in train_index if index < trials]
        train_right_index = [index - trials for index in train_index if index >= trials]
        X_left_train, X_right_train = data.left_data[train_left_index], data.right_data[train_right_index]
        
#        print("훈련 데이터")
#        print("X_left_train.shape : {}, X_right_train.shape{}".format(X_left_train.shape,X_right_train.shape))

        test_left_index = [index for index in test_index if index < trials]
        test_right_index = [index - trials for index in test_index if index >= trials]
        X_left_test, X_right_test = data.left_data[test_left_index], data.right_data[test_right_index]

#        print("시험 데이터")
#        print("X_left_test.shape : {}, X_right_test.shape{}".format(X_left_test.shape,X_right_test.shape))
#        KFold에 의해 훈련, 시험 데이터 랜덤 비율로 나뉨

        y_train, y_test = data.labels[train_index], data.labels[test_index]

        # Feature extraction
#        필터 개수가 9
        N_CSP_COMPONENTS = 2 #기존 #csp_by_band = (9,)
#        N_CSP_COMPONENTS = 3 #csp_by_band = (9,)
        csp_by_band = [CSP(average_trial_covariance=False, n_components=N_CSP_COMPONENTS)
                       for _ in bands]
        
#        print("csp_by_band{}".format(np.array(csp_by_band).shape))

        left_bands_training = filter_bank(X_left_train)
        right_bands_training = filter_bank(X_right_train)
        left_bands_test = filter_bank(X_left_test)
        right_bands_test = filter_bank(X_right_test)
        
#        각각 왼쪽, 오른쪽 train 데이터에 대한 9개 필터와 동일 -> ex : (9,180,700,3)
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
            
#            print("csp 왼쪽 데이터 형태:{}".format(csp.left_data.shape)) # ex:(181,700,3)
            x_train = np.concatenate((left_band_training, right_band_training))
            x_test = np.concatenate((left_band_test, right_band_test))
            
#            print(x_train.shape) # ex:(149,700,3) -> 왼쪽 오른쪽 데이터를 concatenate

            if n_band == 0:
                features_train = csp.compute_features(x_train)
                features_test = csp.compute_features(x_test)
                
#                첫번째 밴드일 때 compute_features 결과 
#                print(np.array(features_train).shape) #(149,2)
            else:
#                2씩 증가 ex:(359,4) -> (359,6)
                features_train = np.concatenate((features_train, csp.compute_features(x_train)), axis=1)
                features_test = np.concatenate((features_test, csp.compute_features(x_test)), axis=1)

#        n_band(9번) 다 돌고나서 결과
#        print(np.array(features_train).shape) #(149,18)

        # Feature Selection
        selected_features = MIBIFFeatureSelection(features_train, features_test, 
                                                  y_train, N_CSP_COMPONENTS, 4, scale=True)

        selected_training_features = selected_features.training_features
        selected_test_features = selected_features.test_features
        
        selected_features = np.concatenate((selected_training_features,selected_test_features))
        
        label_all = np.concatenate((y_train,y_test))
        
#        print(np.array(selected_training_features).shape) 
#        print(np.array(selected_test_features).shape)
#        print(np.array(selected_features).shape) 
#        
#        print()
#        print(np.array(y_train).shape) 
#        print(np.array(y_test).shape)
#        print(np.array(label_all).shape) 
#        
#        break
        
#    cnt=1
#    flag = ''
#    for idx_features,data_features in enumerate(selected_features):
#        if(label_all[idx_features]==0):
#            flag='left'
#        else:
#            flag='right'
#            
#        i_selected = pd.DataFrame(data_features)
#        
#        for idx in range(0,18):
#            imgname="all_images/ch"+str(idx+1)+"/sub"+str(i+4)+"_"+flag+"_"+str(cnt)+"_ch"+str(idx+1)+".png"
#            
#            f, t, Z = signal.stft(i_selected.iloc[idx], fs=FS, nperseg=TIME_WINDOW)
#            Zr = abs(Z)
#            plot.pcolormesh(Zr)
##            plot.ylim(0,40)
#            plot.axis('off')
#            plot.show()
#        plt.savefig(imgname, bbox_inches = 'tight', pad_inches = 0, transparent = True)
        
    cnt=1
    flag = ''
    for idx_features,data_features in enumerate(selected_features):
        if(label_all[idx_features]==0):
            flag='left'
        else:
            flag='right'
            
#        i_selected = pd.DataFrame(data_features)
        
#        for idx in range(0,18):
#        imgname="all_images/ch"+str(idx+1)+"/sub"+str(i+4)+"_"+flag+"_"+str(cnt)+"_ch"+str(idx+1)+".png"
        
        f, t, Z = signal.stft(data_features, fs=FS, nperseg=TIME_WINDOW)
        Zr = abs(Z)
        plot.pcolormesh(Zr)
        plot.ylim(0,40)
#        plot.axis('off')
        plot.show()
            
#    print(np.array(selected_training_features).shape)
#        
#    plot.specgram(pd.DataFrame(selected_training_features[0]).iloc[:,0], NFFT=250, Fs=250, noverlap=125)
#    plot.show()
        
#        trainig_features는 마지막에 한 개 증가, test_features는 한 개 감소
#        print(selected_training_features.shape) #(359,18) -> (360,18)
#        print(selected_test_features.shape) #(40,18) -> (39,18)

#        # LDA classifier
#        lda = LinearDiscriminantAnalysis()
##        데이터, 라벨
#        lda.fit(selected_training_features, y_train)
#        lda_predictions = lda.predict(selected_test_features)
#        lda_accuracy = accuracy_score(y_test, lda_predictions)
#        accuracies["LDA"][i][k] = lda_accuracy

#for classifier in accuracies:
#    print(classifier)
#    for subject, cv_accuracies in enumerate(accuracies[classifier]):
#        acc_mean = np.mean(cv_accuracies)
#        acc_std = np.std(cv_accuracies)
#        print(f"\tSubject {subject+1} average accuracy: {acc_mean:.4f} +/- {acc_std:.4f}")
#    average_acc_mean = np.mean(accuracies[classifier])
#    average_acc_std = np.std(accuracies[classifier])
#    print(f"\tAverage accuracy: {average_acc_mean:.4f} +/- {average_acc_std:.4f}")