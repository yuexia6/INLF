from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
import datetime
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
'''
load=[]
data=[]
with open('book1.csv','r') as csvfile:
                        reader = csv.reader(csvfile)
                        loadd = [row[0] for row in reader]
                    
                        for items in loadd:
                            load.append(items)
with open('book1.csv','r') as csvfile:
    reader = csv.reader(csvfile)  
    timee = [row[1] for row in reader]
    for items in timee:
        timeStamp = float(items)
        dateArray = datetime.datetime.utcfromtimestamp(timeStamp)
        otherStyleTime = dateArray.strftime("%Y-%m-%d %H:%M:%S")
        data.append(otherStyleTime)
with open('book2.csv','w') as file:
                        file.write('Close,Timestamp\n')
                        for n,p in zip(load,data):
                            file.write("{},{}\n".format(n,p))
'''

import pandas as pd
def parser(x):
	return datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

err=pd.DataFrame()
for node in range (1,21):
    if (node!=54):
        series = read_csv('2016years_timestamps_level2/addingweather'+str(node)+'.csv', header=0,usecols=['Close ','Timestamp'], index_col='Timestamp', squeeze=True)
        # Create lagged dataset
        
        error=[]
        h=0
        for n in range(1,100):
          
            values = DataFrame(series.values)
            t=n+(n-1)*10
            h=h+1
            dataframe = concat([values.shift(n*24*60), values], axis=1)
            dataframe.columns = ['t-'+str(t), 't+'+str(t)]
            #print(dataframe.head(5))
             
            # split into train and test sets
            X = dataframe.values
            train_size = int(len(X) * 0.8)
            train, test = X[1:train_size], X[train_size:]
            train_X, train_y = train[:,0], train[:,1]
            test_X, test_y = test[:,0], test[:,1]
             
            # persistence model
            def model_persistence(x):
            	return x
             
            # walk-forward validation
            predictions = list()
            for x in test_X:
            	yhat = model_persistence(x)
            	predictions.append(yhat)
            test_score =mean_absolute_percentage_error(test_y, predictions)
            error.append(test_score)
            print(h)
            print('Test MSE: %.3f' % test_score)
            
            values=[]
            dataframe=[]
            X=[] 
            err['error'+str(node)]=error
            print('add node'+str(node))
            error=[]
    else:
        continue
        

'''
plt.plot(err)
plt.xlabel('epochs')
plt.ylabel('MAPE')
err.to_csv('level2/result of the naive forecasting',index=False,header=None)
'''
final=err.min(0)
final.to_csv('level1/result of the naive forecasting',header=None,index=False)
bar_width=0.2
index =np.linspace(1,len(final),len(final))
#plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
plt.bar(index-bar_width,final, bar_width)   
plt.xlabel('node')
plt.ylabel('MAPE')
plt.show()   

'''
# plot predictions and expected results
pyplot.plot(train_y)
pyplot.plot([None for i in train_y] + [x for x in test_y])
pyplot.plot([None for i in train_y] + [x for x in predictions])
pyplot.show()
'''