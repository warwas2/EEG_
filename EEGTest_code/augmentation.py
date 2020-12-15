from src.data_preparation.data_preparation import read_eeg_file
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
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from pyyawt import wavedec, detcoef, wkeep

TIME_LENGTH = 2000
TIME_WINDOW = 1500
EPOCH_SIZE = None
DATA_FOLDER = "testdata"
CSP_COMPONENTS = 2
FS = 250
lev=3

K_FOLD = 10
subjects = range(5,10)
subjects_set = set(subjects)
accuracies = {
    "GNB": np.zeros((len(subjects), K_FOLD)),
    "SVM": np.zeros((len(subjects), K_FOLD)),
    "LDA": np.zeros((len(subjects), K_FOLD))
}

for subject in subjects:
    print("========= Subject: ", subject)
    # Load data
    print("Loading data ...")
    left_data_file = f"{DATA_FOLDER}/{subject}-left.csv"
    right_data_file = f"{DATA_FOLDER}/{subject}-right.csv"
    data = read_eeg_file(left_data_file, right_data_file, TIME_LENGTH, TIME_WINDOW, EPOCH_SIZE)
#    print(data.left_data)
#    print(data.right_data.shape)
    
#    [c,l] = wavedec(data.left_data,lev,'coif2')
##    (c,l) = dwt(data.left_data,wavelet='coif2')
#    
#    sig_len = len(data.left_data)
#    cfd = np.zeros((lev,sig_len))
##    
#    for idx in range(0,3):
#        d = detcoef(c,l,idx+1)
#        
#        tmp=[]
##        print(np.array(d).shape)
#        for p_row in range(0,pow(2,idx+1)):
#            tmp.append(d)
##        print(np.array(tmp).shape)
#        tmp=pd.DataFrame(tmp)
#        
#        wd=[]
#        for t_col in range(len(tmp.columns)):
#            wd.append(tmp[t_col])
#        wd=np.array(wd).flatten()
##        print(wd.shape)
#        
#        cfd[idx,:]=wkeep(wd,len(data.left_data))
##        print(cfd.shape)
#        
#        tmp=[]
#        cfd=pd.DataFrame(cfd)
#        for c_col in range(len(cfd.columns)):
#            print(c_col)
#            
#        image(cfd)
#        break
    
#    이미지 생성 부분 ******
#    왼손 
#    left_data_len = len(data.left_data)
#    cnt=1
#    for i in data.left_data:
#     
#    오른손 
    right_data_len = len(data.right_data)
    cnt=1
    for i in data.right_data:
        
#        print(i.shape) 
#        print(i)
#        print(len(data.right_data))
        
        i=pd.DataFrame(i)
        #left_c_one = i.iloc[:,0] #첫번째 채널
        #left_c_two = i.iloc[:,1] #두번째 채널
        #left_c_three = i.iloc[:,2] #세번째 채널
        
        for idx in range(0,3):
            imgname="all_images/ch"+str(idx+1)+"/sub"+str(subject)+"_right_"+str(cnt)+"_ch"+str(idx+1)+".png"
#            imgname="all_images/ch"+str(idx+1)+"/sub"+str(subject)+"_left_"+str(cnt)+"_ch"+str(idx+1)+".png"
                
            f, t, Z = signal.stft(i.iloc[:,idx], fs=FS, nperseg=TIME_WINDOW)
            Zr = abs(Z)
            plt.pcolormesh(Zr)
            plt.ylim(0,40)
            plt.axis('off')
#            plt.show()
            plt.savefig(imgname, bbox_inches = 'tight', pad_inches = 0, transparent = True)
#        break
        cnt = cnt+1
#    이미지 생성 부분 ******
    
#    plot.specgram(pd.DataFrame(i).iloc[:,0], NFFT=64, Fs=250, noverlap=32)
