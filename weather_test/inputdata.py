import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.models import optimizers
from keras.layers import Dense, Dropout, Activation, Flatten,Reshape
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import np_utils
from keras.layers import LSTM, LeakyReLU
from keras.callbacks import CSVLogger, ModelCheckpoint
import h5py
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
 
class PastSampler:
    '''
    Forms training samples for predicting future values from past value
    '''
     
    def __init__(self, N, K, sliding_window = True):
        '''
        Predict K future sample using N previous samples
        '''
        self.K = K
        self.N = N
        self.sliding_window = sliding_window
 
    def transform(self, A):
        M = self.N + self.K     #Number of samples per row (sample + target)
        #indexes
        if self.sliding_window:
            I = np.arange(M) + np.arange(A.shape[0] - M + 1).reshape(-1, 1)
        else:
            if A.shape[0]%M == 0:
                I = np.arange(M)+np.arange(0,A.shape[0],M).reshape(-1,1)
                
            else:
                I = np.arange(M)+np.arange(0,A.shape[0] -M,M).reshape(-1,1)
            
        B = A[I].reshape(-1, M * A.shape[1], A.shape[2])
        ci = self.N * A.shape[1]    #Number of features per sample
        return B[:, :ci], B[:, ci:] #Sample matrix, Target matrix

#data file path
for node in range(1,21):
#    os.makedirs('RNNweight'+str(node))
    dfp = '2016years_timestamps_level2/addingweather'+str(node)+'.csv'
#Columns of price data to use

    columns = ['temperature','humidity','visibility','apparentTemperature',	'pressure',
            'windSpeed','cloudCover','Timestamp','windBearing','precipIntensity','dewPoint',	
           'precipProbability','Close']

#columns = ['Close','Timestamp']
    df = pd.read_csv(dfp)
    time_stamps = df['Timestamp']
    df = df.loc[:,columns]
    original_df = pd.read_csv(dfp).loc[:,columns]

  


    scaler = MinMaxScaler()
# normalization
    for c in columns:
        df[c] = scaler.fit_transform(df[c].values.reshape(-1,1))
    
#Features are input sample dimensions(channels)
    A = np.array(df)[:,None,:]
    original_A = np.array(original_df)[:,None,:]
    time_stamps = np.array(time_stamps)[:,None,None]

    #Make samples of temporal sequences of pricing data (channel)
    NPS, NFS =2,1        #Number of past and future samples
    ps = PastSampler(NPS, NFS, sliding_window=True)
    B, Y = ps.transform(A)
    input_times, output_times = ps.transform(time_stamps)
    original_B, original_Y = ps.transform(original_A)