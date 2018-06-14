ßimport matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
'''
plt.figure()
plt.figure(figsize=(12,8))
#plt.subplot(1,2,1)
reader=pd.read_csv('level3/2016years_level3/test1.csv')
#reader.load[reader.load>300]=12
df = pd.DataFrame(reader)
df['time'] = pd.to_datetime(df['time'])
df_show = df.loc[(df["time"].dt.month >=1 ),:]

#prediction_df = prediction_df.loc[(prediction_df["times"].dt.year == 2016 )&(prediction_df["times"].dt.month >=1 ),: ]
#ground_true_df = ground_true_df.loc[(ground_true_df["times"].dt.year == 2016 )&(ground_true_df["times"].dt.month >=1 ),:]
plt.plot(df_show.time,df_show.load)
#plt.legend(loc='upper right')

#plt.gca().xaxis.set_major_locator(MaxNLocator(prune='upper'))

plt.xlabel('Time')
plt.ylabel('Load')


plt.subplot(1,2,2)
reader=pd.read_csv('level1/2016years_level1/test23.csv')
#reader.load[reader.load>80]=12
reader.load[reader.load>20]=12
df = pd.DataFrame(reader)
df['time'] = pd.to_datetime(df['time'])
df_show1 = df.loc[(df["time"].dt.month == 3 )&(df["time"].dt.day ==15 ),:]
df_show2 = df.loc[(df["time"].dt.month == 7 )&(df["time"].dt.day ==15 ),:]
df_show3 = df.loc[(df["time"].dt.month == 11 )&(df["time"].dt.day ==15 ),:]
#prediction_df = prediction_df.loc[(prediction_df["times"].dt.year == 2016 )&(prediction_df["times"].dt.month >=1 ),: ]
#ground_true_df = ground_true_df.loc[(ground_true_df["times"].dt.year == 2016 )&(ground_true_df["times"].dt.month >=1 ),:]
x = np.arange(1,24 ,23/1440)
df_show1.index=x
df_show2.index=x
df_show3.index=x
plt.plot(df_show1.load,label = 'Apartment23 03/15')
plt.plot(df_show2.load,label = 'Apartment23 07/15')
plt.plot(df_show3.load,label = 'Apartment23 11/15')
plt.legend(loc='upper left')
plt.xlim( 1, 24 )
plt.xlabel('Hour')
plt.ylabel('Load')
plt.subplot(1,2,1)
reader=pd.read_csv('level1/2016years_level1/test1.csv')
#reader.load[reader.load>80]=12
reader.load[reader.load>20]=12
df = pd.DataFrame(reader)
df['time'] = pd.to_datetime(df['time'])
df_show1 = df.loc[(df["time"].dt.month == 3 )&(df["time"].dt.day ==15 ),:]
df_show2 = df.loc[(df["time"].dt.month == 7 )&(df["time"].dt.day ==15 ),:]
df_show3 = df.loc[(df["time"].dt.month == 11 )&(df["time"].dt.day ==15 ),:]
#prediction_df = prediction_df.loc[(prediction_df["times"].dt.year == 2016 )&(prediction_df["times"].dt.month >=1 ),: ]
#ground_true_df = ground_true_df.loc[(ground_true_df["times"].dt.year == 2016 )&(ground_true_df["times"].dt.month >=1 ),:]
x = np.arange(1,24 ,23/1440)
df_show1.index=x
df_show2.index=x
df_show3.index=x
plt.plot(df_show1.load,label = 'Apartment1 03/15')
plt.plot(df_show2.load,label = 'Apartment1 07/15')
plt.plot(df_show3.load,label = 'Apartment1 11/15')
plt.legend(loc='upper left')
plt.xlim( 1, 24 )
plt.xlabel('Hour')
plt.ylabel('Load')
'''
'''
mean=pd.DataFrame()
for n  in range(1,29):
        reader=pd.read_csv('level3/2016years_level3/test1.csv')
        reader.load[reader.load>300]=12
        df = pd.DataFrame(reader)       
        df['time'] = pd.to_datetime(df['time'])        
        df1 = df.loc[(df["time"].dt.month == 2 )&(df["time"].dt.day==n ),:]
        df1.reset_index(drop=True, inplace=True)
        mean['10 load in day'+str(n)]=df1['load']
        print(n)

for n  in range(1,32):
        reader=pd.read_csv('level3/2016years_level3/test1.csv')
        reader.load[reader.load>300]=12
        df = pd.DataFrame(reader)       
        df['time'] = pd.to_datetime(df['time'])        
        df1 = df.loc[(df["time"].dt.month == 3 )&(df["time"].dt.day==n ),:]
        df1.reset_index(drop=True, inplace=True)
        mean['11 load in day'+str(n)]=df1['load']
        print(n)
        
for n  in range(1,31):
        reader=pd.read_csv('level3/2016years_level3/test1.csv')
        reader.load[reader.load>300]=12
        df = pd.DataFrame(reader)       
        df['time'] = pd.to_datetime(df['time'])        
        df1 = df.loc[(df["time"].dt.month >= 4 )&(df["time"].dt.day==n ),:]
        df1.reset_index(drop=True, inplace=True)
        mean['11 load in day'+str(n)]=df1['load']
        print(n)        

a=mean.mean(1)
b=mean.median(1)
c=mean.std(1)



x = np.arange(1,24 ,23/len(a))
a.index=x
b.index=x
c.index=x
plt.subplot(1,2,1)
plt.plot(b,label='meidian')
plt.plot(a,'red',label='mean')
plt.xlabel('Hour')
plt.xlim( 1, 24 )
plt.ylabel('mean and median')
plt.legend(loc='upper left')
plt.subplot(1,2,2)
plt.errorbar(x, a, yerr=c, linestyle='None', marker='o',label='std')
plt.xlabel('hour')
plt.xlim( 1, 24 )
plt.ylabel('std')
plt.legend(loc='upper left')
'''
'''
result=pd.read_csv('level1/CNN of weight34/weight_level1_test_close_CNN_.csv')
acc=result.acc
loss=result.loss
val_acc=result.val_acc
val_loss=result.val_loss
plt.subplot(1,2,1)
plt.plot(acc,label='acc')
plt.plot(val_acc,label='val_acc')
plt.subplot(1,2,2)
plt.plot(loss,label='loss')
plt.plot(val_loss,label='val_loss')
'''
import numpy as np
import matplotlib.ticker as ticker
RNN_sum=pd.DataFrame()
CNN_sum=pd.DataFrame()
GRU_sum=pd.DataFrame()
for node in range(1,21):
    if node!=54: 
        print(node)
        RNN=pd.read_csv('level2/RNN of weight2 of'+str(node)+'/weight_LSTM.csv')
        CNN=pd.read_csv('level2/CNN of weight2 of'+str(node)+'/weight_CNN_.csv')
        GRU=pd.read_csv('level2/2GRU of weight of'+str(node)+'/weight_GRU.csv')
        RNN_sum[str(node)]=RNN.loss
        CNN_sum[str(node)]=CNN.loss
        GRU_sum[str(node)]=GRU.loss
    else:
        continue
Naive=pd.read_csv('level2/result of the naive forecasting',header=None)
#Naive[Naive>1500]=1500
Naive.index= range(1, len(Naive)+1)
N=np.array(Naive)
Nai=[]
for i in range(0,len(N)):
    x=float(N[i])
    Nai.append(x)
RNN_min=RNN_sum.min(0)
CNN_min=CNN_sum.min(0)
GRU_min=GRU_sum.min(0)
bar_width=0.2
index =np.linspace(1,len(RNN_min),len(RNN_min))
plt.figure()
plt.figure(figsize=(20,15))
plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
plt.bar(index-bar_width,RNN_min, bar_width,label='RNN')   
plt.bar(index,CNN_min, bar_width,label='CNN')   
plt.bar(index+bar_width,GRU_min, bar_width,label='GRU')   
plt.bar(index+2*bar_width,Nai, bar_width,label='Naive')   
plt.xlabel('Node')
plt.ylabel('MAPE')
plt.legend(loc='upper left')
'''
naive=pd.read_csv('level3/result of the naive forecasting')
RNN=pd.read_csv('level3/RNN of weight2 of'+str(1)+'/weight_LSTM.csv')
RGU=pd.read_csv('level3/2GRU of weight of'+str(1)+'/weight_GRU.csv')
CNN=pd.read_csv('level3/CNN of weight2 of'+str(1)+'/weight_CNN_.csv')
rnn=RNN.loss
rgu=RGU.loss
cnn=CNN.loss
plt.plot(naive,label='Naive')
plt.plot(rnn,label='RNN')
plt.plot(rgu,label='RGU')
plt.plot(cnn,label='CNN')
plt.xlabel('epochs')
plt.xlim( 1, 50 )
plt.ylabel('MAPE')
plt.legend(loc='upper left')

RNN=pd.read_csv('level2/RNN of weight'+str(1)+'/weight_level3_LSTM.csv')
val_acc=RNN.val_acc
acc=RNN.acc
plt.plot(val_acc,label='val_acc')
plt.plot(acc,label='acc')
plt.xlabel('epochs')
plt.xlim( 1, 50 )
plt.ylabel('acc')
plt.legend(loc='upper left')
'''