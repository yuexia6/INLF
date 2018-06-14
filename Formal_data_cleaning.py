import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
#os.makedirs('level1/2016years_level1')
#delete the duplication and the lossing gap before the point of 103800
for apt in range(1,115):
                #give a header
                reader=pd.read_csv('raw_data/Apt'+str(apt)+'_2016.csv')
                reader.to_csv('level1/2016years_level1/test'+str(apt)+'.csv',index=False,header=['time','load'])
                #delete the duplication
                reader=pd.read_csv('level1/2016years_level1/test'+str(apt)+'.csv')
                f=reader.load[reader.load>30]
                x=f.index
                x=np.array(x)
                if len(x)!=0:
                    reader.load[reader.load>30]=reader.load.loc[x[0]-1]/2+reader.load.loc[x[0]+1]/2
                time=reader.time
                a=time.loc[0]
                c=pd.date_range(start=a,periods=len(time),freq='60s')
                reader['time']=c
                #reader.drop_duplicates(subset=['time'], keep='first', inplace=True)
#                reader=reader[103800:]
                reader.to_csv('level1/2016years_level1/test'+str(apt)+'.csv',index=False)
                x=[]
                f=[]
                print('finish:'+str(apt))

#check the time interval
record=[]
for apt in range(1,115):
                print('now:'+str(apt))
                df1=pd.read_csv('level1/2016years_level1/test'+str(apt)+'.csv')
                df1=np.array(df1)
                min1=pd.Series(df1[:,0])
                min2=pd.Series(df1[1::,0])
                interval=pd.to_datetime(min2)-pd.to_datetime(min1)
                interval=interval.dropna(axis=0,how='any')
                interval=interval.drop_duplicates()
                interval=interval.drop([0])
                print(interval)
                record.append(apt)
                record.append(np.array(interval))
print(len(record)-113)
         

small_record=[]
#check the long time missing recor
for apt in range(1,115):
                df2=pd.read_csv('level1/2016years_level1/test'+str(apt)+'.csv')
                df2=np.array(df2)
                span=1
                average=0
                summ=0
                aver=[]
                sumdata=[]
                small=[]
                for num in range(1,df2.shape[0]):
                    summ=summ+df2[num,1]
                    if num%span == 0:
                            sumdata.append(summ)
                            summ=0      
                sumdata=pd.DataFrame(sumdata)
                small=sumdata[sumdata>20]
                small=small.dropna(axis=0,how='any')
                if len(small)>=1:
                    small_record.append(apt)
                    small_record.append(len(small))
                    print(apt,len(small))

#get the version of on hearder
for apt in range(1,115):                  
                    data=pd.read_csv('level1/2016years_level1/test'+str(apt)+'.csv',header=None)
                    updata = data.drop(0)
                    updata.to_csv('level1/2016years_timestamps_level1/test'+str(apt)+'.csv',index=False,header=None)
               
'''
plt.plot(sumdata)     
plt.xlabel('Days')
plt.ylabel('Load')
plt.title('Histogram of the Apartment Load Usage')
plt.grid(True)
plt.show()

allloss=pd.read_csv('weight_level2_test_close_LSTM_1_tanh_leaky_.csv')
loss=np.array(allloss)
plt.plot(loss[:,2])
plt.xlabel('epochs')
plt.ylabel('error:mse')
plt.title('Curve of Aggregation of the First 30 Apartment RNN Forecasting')
plt.grid(True)
plt.show()
'''
