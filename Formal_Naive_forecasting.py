from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
import datetime
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
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
def parser(x):
	return datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
 
series = read_csv('level1/2016years_level1/test55.csv', header=0,  index_col=0, squeeze=True)
# Create lagged dataset
error=[]
h=0
for n in range(1,365*6):
    values = DataFrame(series.values)
    t=n+(n-1)*10
    h=h+1
    dataframe = concat([values.shift(n), values], axis=1)
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
    test_score = mean_squared_error(test_y, predictions)
    error.append(test_score)
    print(h)
    print('Test MSE: %.3f' % test_score)
    values=[]
    dataframe=[]
    X=[]
plt.plot(error)     
plt.xlabel('Time span')
plt.ylabel('error:mse')
plt.title('Curve of Naive Forecasting with Weather data')
plt.grid(True)
plt.show()   

''' 
# plot predictions and expected results
pyplot.plot(train_y)
pyplot.plot([None for i in train_y] + [x for x in test_y])
pyplot.plot([None for i in train_y] + [x for x in predictions])
pyplot.show()
'''