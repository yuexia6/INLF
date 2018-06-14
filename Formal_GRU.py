import numpy as np
import pandas as pd
import os 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Reshape
from keras.layers import Conv1D, MaxPooling1D, LeakyReLU
from keras.utils import np_utils
from keras.layers import GRU,CuDNNGRU
from keras.callbacks import CSVLogger, ModelCheckpoint
import h5py
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
class PastSampler:
    def __init__(self, N, K, sliding_window = True):
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
        
for node in range(20,21): 
        dfp = 'level2/2016years_timestamps_level2/nodee'+str(node)+'.csv'
        os.mkdir('level2/2GRU of weight of'+str(node))
        
        #Columns of price data to use
        columns = ['Close']
        df = pd.read_csv(dfp)
        time_stamps = df['Timestamp']
        df = df.loc[:,columns]
        original_df = pd.read_csv(dfp).loc[:,columns]
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        # normalization
        #for c in columns:
         #  df[c] = scaler.fit_transform(df[c].values.reshape(-1,1))   
        #Features are input sample dimensions(channels)
        A = np.array(df)[:,None,:]
        original_A = np.array(original_df)[:,None,:]
        time_stamps = np.array(time_stamps)[:,None,None]
        #Make samples of temporal sequences of pricing data (channel)
        NPS, NFS =16,2        #Number of past and future samples
        ps = PastSampler(NPS, NFS, sliding_window=False)
        B, Y = ps.transform(A)
        input_times, output_times = ps.transform(time_stamps)
        original_B, original_Y = ps.transform(original_A)                
        with h5py.File('level2/h5file/test'+str(node)+'.h5', 'w') as f:
            
              
            f.create_dataset("inputs", data = B)
            f.create_dataset('outputs', data = Y)
            f.create_dataset("input_times", data = input_times)
            f.create_dataset('output_times', data = output_times)
            f.create_dataset("original_datas", data=np.array(original_df))
            f.create_dataset('original_inputs',data=original_B)
            f.create_dataset('original_outputs',data=original_Y)
        
        
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))
        
        with h5py.File(''.join(['level2/h5file/test'+str(node)+'.h5']), 'r') as hf:
            datas = hf['inputs'].value
            labels = hf['outputs'].value
        
        
        output_file_name='level2/2GRU of weight of'+str(node)+'/weight_GRU'
        
        step_size = datas.shape[1]
        units= 50
        batch_size = 50
        nb_features = datas.shape[2]
        epochs = 50
        output_size=NFS
        #split training validation
        training_size = int(0.8* datas.shape[0])
        training_datas = datas[:training_size,:]
        training_labels = labels[:training_size,:,0]
        validation_datas = datas[training_size:,:]
        validation_labels = labels[training_size:,:,0]
        
        #build model
        model = Sequential()
        model.add(GRU(units=units, input_shape=(step_size,nb_features),return_sequences=False))
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(output_size))
        model.add(Activation('relu'))
        model.compile(loss='mean_absolute_percentage_error', optimizer='adam', metrics=['mse','accuracy'])
        model.summary()
        model.fit(training_datas, training_labels, batch_size=batch_size,validation_data=(validation_datas,validation_labels), epochs = epochs, callbacks=[CSVLogger(output_file_name+'.csv', append=True),ModelCheckpoint(output_file_name+'-{epoch:02d}-{val_loss:.5f}.hdf5', monitor='val_loss', verbose=1,mode='min')])
