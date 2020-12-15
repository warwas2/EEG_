from src.data_preparation.data_preparation_overlapping60 import read_eeg_file
from scipy import signal
from src.algorithms.csp.CSP import CSP
import pywt
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plot
import scipy.misc
import pandas as pd

TIME_LENGTH = 750
TIME_WINDOW = 750
EPOCH_SIZE = 500
DATA_FOLDER = "testdata"
CSP_COMPONENTS = 2
FS = 250
WAVELET = "coif1"
K_FOLD = 10
subjects = range(1, 2)
subjects_set = set(subjects)
accuracies = {
    "GNB": np.zeros((len(subjects), K_FOLD)),
    "SVM": np.zeros((len(subjects), K_FOLD)),
    "LDA": np.zeros((len(subjects), K_FOLD))
}

finalarr=[]
for subject in subjects:
    print("========= Subject: ", subject)
    # Load data
    print("Loading data ...")
    left_data_file = f"{DATA_FOLDER}/{subject}-left.csv"
    right_data_file = f"{DATA_FOLDER}/{subject}-right.csv"
    data = read_eeg_file(left_data_file, right_data_file, TIME_LENGTH, TIME_WINDOW, EPOCH_SIZE)

    # Pre-processing
    print("Pre-processing ...")
    print("Applying 5º order Butterworth bandpass filter (7-30 Hz)")
    b, a = signal.butter(5, [7, 30], btype="bandpass", fs=FS)

    # left_data2, right_data2 전달된 상태
    data.left_data = signal.filtfilt(b, a, data.left_data, axis=1)
    data.right_data = signal.filtfilt(b, a, data.right_data, axis=1)

    print("Spatial-filtering ...")
    data.X = np.concatenate((data.left_data, data.right_data))

    csp = CSP(average_trial_covariance=True, n_components=CSP_COMPONENTS)
    csp.fit(data.left_data, data.right_data)
    
#    print(csp.W.shape) #(3,2)
    
#    print(data.left_data.shape)  #(159, 500, 3) 
#    print(data.right_data.shape)# (165, 500, 3)
   
    print(data.X.shape)
    
    counts=0
    flag=1
    print("왼쪽 데이터 이미지 생성 시작")
    imgtype='left'
    for i in data.X:
        counts+=1
        
        if counts-1==len(data.left_data) and flag==1: 
            print("오른쪽 데이터 이미지 생성 시작")
            imgtype='right'
            counts=1
            flag=2
            
        imgname="images_overlapping60_transparent/sub"+str(subject)+"_"+str(imgtype)+"_"+str(counts)+"_ch1.png"
        plot.specgram(pd.DataFrame(i).iloc[:,0], NFFT=64, Fs=250, noverlap=32)
        plot.axis('off')
#        plot.show()
        plot.savefig(imgname, bbox_inches = 'tight', pad_inches = 0, transparent = True)
        
        imgname="images_overlapping60_transparent/sub"+str(subject)+"_"+imgtype+"_"+str(counts)+"_ch2.png"
        plot.specgram(pd.DataFrame(i).iloc[:,1], NFFT=64, Fs=250, noverlap=32)
        plot.axis('off')
#        plot.show()
        plot.savefig(imgname, bbox_inches = 'tight', pad_inches = 0, transparent = True)
        
        imgname="images_overlapping60_transparent/sub"+str(subject)+"_"+imgtype+"_"+str(counts)+"_ch3.png"
        plot.specgram(pd.DataFrame(i).iloc[:,2], NFFT=64, Fs=250, noverlap=32)
        plot.axis('off')
#        plot.show()
        plot.savefig(imgname, bbox_inches = 'tight', pad_inches = 0, transparent = True)
        
#        trial 1개만 돌릴 때 사용
#        break
#    subject1만 돌릴 때 사용
#    break

    
